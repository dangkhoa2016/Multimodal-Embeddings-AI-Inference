"""Offline, subprocess-based operator lifecycle tests (never touches Modal)."""
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "scripts" / "manage_client_keys.py"
BUNDLE = ROOT / "scripts" / "client_registry_bundle.py"


def tool(script, *args, success=True):
    result = subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def test_create_rotate_revoke_and_private_output(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    registry = private / "client-keys.json"
    first = private / "alice.key"
    second = private / "alice-rotated.key"

    created = tool(MANAGER, "create", "--registry", registry,
                   "--client-id", "alice", "--token-out", first)
    raw = first.read_text().strip()
    assert len(raw) >= 48
    assert raw not in created.stdout + created.stderr
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert stat.S_IMODE(registry.stat().st_mode) == 0o600
    first_digest = hashlib.sha256(raw.encode("ascii")).hexdigest()
    assert json.loads(registry.read_text()) == {"alice": [first_digest]}

    rotated = tool(MANAGER, "rotate", "--registry", registry,
                   "--client-id", "alice", "--token-out", second)
    new_token = second.read_text().strip()
    assert raw != new_token
    assert new_token not in rotated.stdout + rotated.stderr
    second_digest = hashlib.sha256(new_token.encode("ascii")).hexdigest()
    assert json.loads(registry.read_text())["alice"] == [first_digest, second_digest]

    old_revoke = tool(MANAGER, "revoke-key", "--registry", registry,
                      "--client-id", "alice", "--digest", first_digest)
    assert first_digest not in old_revoke.stdout
    assert json.loads(registry.read_text()) == {"alice": [second_digest]}
    tool(MANAGER, "revoke-client", "--registry", registry,
         "--client-id", "alice", "--confirm-revoke-all")
    assert json.loads(registry.read_text()) == {}
    assert "alice:" not in tool(MANAGER, "inspect", "--registry", registry).stdout


def test_tool_rejects_unsafe_paths_duplicate_client_and_overwrite(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    registry = private / "clients.json"
    out = private / "alice.key"
    tool(MANAGER, "create", "--registry", registry, "--client-id", "alice",
         "--token-out", out)
    tool(MANAGER, "create", "--registry", registry, "--client-id", "alice",
         "--token-out", private / "never.key", success=False)
    tool(MANAGER, "rotate", "--registry", registry, "--client-id", "alice",
         "--token-out", out, success=False)
    tool(MANAGER, "revoke-client", "--registry", registry,
         "--client-id", "alice", success=False)
    open_dir = tmp_path / "insecure"
    open_dir.mkdir(mode=0o755)
    open_dir.chmod(0o755)
    tool(MANAGER, "create", "--registry", open_dir / "registry.json",
         "--client-id", "bob", "--token-out", open_dir / "bob.key", success=False)
    assert not (open_dir / "bob.key").exists()
    assert len(json.loads(registry.read_text())["alice"]) == 1


def test_staging_retains_owner_token_revisions_and_registry(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    registry = private / "clients.json"
    token_file = private / "alice.key"
    tool(MANAGER, "create", "--registry", registry,
         "--client-id", "alice", "--token-out", token_file)
    owner = private / "modal-owner.env"
    original_token = "operator-owner-private-token-123456789"
    owner.write_text(
        f"VISION_EMBEDDINGS_API_TOKEN={original_token}\n"
        f"VISION_EMBEDDINGS_QWEN8_REVISION={'a' * 40}\n"
        f"VISION_EMBEDDINGS_WEMM9_REVISION={'b' * 40}\n"
        "VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL=7\n",
        encoding="utf-8",
    )
    owner.chmod(0o600)
    staged = private / "staged.env"
    result = tool(BUNDLE, "--owner-file", owner, "--registry", registry,
                  "--output", staged)
    contents = staged.read_text()
    assert original_token in contents
    assert "VISION_EMBEDDINGS_QWEN8_REVISION=" + "a" * 40 in contents
    assert "VISION_EMBEDDINGS_WEMM9_REVISION=" + "b" * 40 in contents
    assert "VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL=7" in contents
    assert "VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON=" in contents
    assert json.loads(registry.read_text())["alice"][0] in contents
    assert original_token not in result.stdout + result.stderr
    assert stat.S_IMODE(staged.stat().st_mode) == 0o600
    tool(BUNDLE, "--owner-file", owner, "--registry", registry,
         "--output", staged, success=False)
    assert staged.read_text() == contents


def test_staging_never_drops_unknown_registry_silently(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    owner = private / "owner.env"
    owner.write_text(
        f"VISION_EMBEDDINGS_API_TOKEN={'x' * 32}\n"
        f"VISION_EMBEDDINGS_QWEN8_REVISION={'a' * 40}\n"
        f"VISION_EMBEDDINGS_WEMM9_REVISION={'b' * 40}\n",
    )
    owner.chmod(0o600)
    target = private / "target.env"
    missing = private / "missing.json"
    tool(BUNDLE, "--owner-file", owner, "--registry", missing,
         "--output", target, success=False)
    assert not target.exists()
    tool(BUNDLE, "--owner-file", owner, "--registry", missing,
         "--output", target, "--confirm-empty-registry")
    assert "VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON={}\n" in target.read_text()
    # An invalid existing owner configuration is not silently recreated.
    owner.chmod(0o644)
    tool(BUNDLE, "--owner-file", owner, "--registry", missing,
         "--output", private / "new.env", "--confirm-empty-registry", success=False)


ROTATE_OWNER = ROOT / "scripts" / "rotate_owner_token.py"


def rotation_input(tmp_path, clients=None):
    private = tmp_path / "rotation-private"
    private.mkdir(mode=0o700)
    clients = clients if clients is not None else {
        "client_main": [hashlib.sha256(b"sample-client-token").hexdigest()]
    }
    registry = private / "client-keys.json"
    registry.write_text(json.dumps(clients))
    registry.chmod(0o600)
    source = private / "gateway-next.env"
    original = "owner-original-private-" + "x" * 48
    source.write_text(
        f"VISION_EMBEDDINGS_API_TOKEN={original}\n"
        f"VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON={json.dumps(clients, separators=(',', ':'))}\n"
        f"VISION_EMBEDDINGS_QWEN8_REVISION={'a' * 40}\n"
        f"VISION_EMBEDDINGS_WEMM9_REVISION={'b' * 40}\n"
        "VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL=7\n"
    )
    source.chmod(0o600)
    return source, registry, original


def parse_env(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def test_owner_rotation_preserves_registry_and_every_other_setting(tmp_path):
    source, registry, old_token = rotation_input(tmp_path)
    source_before, registry_before = source.read_bytes(), registry.read_bytes()
    destination = source.parent / "gateway-rotated.env"
    result = tool(ROTATE_OWNER, "--source", source, "--registry", registry,
                  "--output", destination)
    original, rotated = parse_env(source), parse_env(destination)
    assert rotated["VISION_EMBEDDINGS_API_TOKEN"] != old_token
    assert len(rotated["VISION_EMBEDDINGS_API_TOKEN"]) >= 48
    for key in original.keys() - {"VISION_EMBEDDINGS_API_TOKEN"}:
        assert rotated[key] == original[key]
    assert source.read_bytes() == source_before
    assert registry.read_bytes() == registry_before
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert old_token not in result.stdout + result.stderr
    assert rotated["VISION_EMBEDDINGS_API_TOKEN"] not in result.stdout + result.stderr
    assert "No Modal calls" in result.stdout


def test_owner_rotation_refuses_missing_or_mismatched_registry(tmp_path):
    source, registry, _ = rotation_input(tmp_path)
    output = source.parent / "never.env"
    registry.unlink()
    failure = tool(ROTATE_OWNER, "--source", source, "--registry", registry,
                   "--output", output, success=False)
    assert "Canonical client registry is missing" in failure.stderr
    registry.write_text(json.dumps({"different": [
        hashlib.sha256(b"different").hexdigest()
    ]}))
    registry.chmod(0o600)
    failure = tool(ROTATE_OWNER, "--source", source, "--registry", registry,
                   "--output", output, success=False)
    assert "registries differ" in failure.stderr
    assert not output.exists()


def test_owner_rotation_refuses_incomplete_source_and_overwrite(tmp_path):
    source, registry, _ = rotation_input(tmp_path)
    output = source.parent / "never.env"
    source.write_text(
        "\n".join(line for line in source.read_text().splitlines()
                  if not line.startswith("VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON="))
        + "\n"
    )
    failure = tool(ROTATE_OWNER, "--source", source, "--registry", registry,
                   "--output", output, success=False)
    assert "Missing client registry" in failure.stderr
    assert not output.exists()
    output.write_text("do not replace\n")
    tool(ROTATE_OWNER, "--source", source, "--registry", registry,
         "--output", output, success=False)
    assert output.read_text() == "do not replace\n"


def test_owner_rotation_requires_opt_in_for_no_clients(tmp_path):
    source, registry, _ = rotation_input(tmp_path, clients={})
    output = source.parent / "rotated.env"
    tool(ROTATE_OWNER, "--source", source, "--registry", registry,
         "--output", output, success=False)
    assert not output.exists()
    tool(ROTATE_OWNER, "--source", source, "--registry", registry,
         "--output", output, "--confirm-empty-registry")
    assert parse_env(output)["VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON"] == "{}"


def test_owner_rotation_rejects_insecure_source_and_output(tmp_path):
    source, registry, _ = rotation_input(tmp_path)
    output = source.parent / "new.env"
    source.chmod(0o644)
    tool(ROTATE_OWNER, "--source", source, "--registry", registry,
         "--output", output, success=False)
    assert not output.exists()
    source.chmod(0o600)
    output.symlink_to(source)
    tool(ROTATE_OWNER, "--source", source, "--registry", registry,
         "--output", output, success=False)
    assert output.is_symlink()
