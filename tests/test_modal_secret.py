"""Offline Modal Secret helper tests; never call the real Modal CLI."""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
Q = "2c4565515e0f265c6511776e7193b22c0968ddc7"
W = "00c52839de57a6d4fd5b78cf5522ccf0ac8ea482"


def helper():
    spec = importlib.util.spec_from_file_location(
        "multimodal_secret", ROOT / "scripts" / "create_modal_secret.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_token(contents):
    return next(
        line.partition("=")[2]
        for line in contents.splitlines()
        if line.startswith("VISION_EMBEDDINGS_API_TOKEN=")
    )


def arguments(*flags):
    return ["helper", "--qwen8-revision", Q, "--wemm9-revision", W, *flags]


def test_rotation_distinct_from_force(monkeypatch, tmp_path, capsys):
    script = helper()
    monkeypatch.setattr(script.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(script, "_require_modal_tooling", lambda parser: None)
    calls = []

    def fake_modal(command, *, check):
        assert check
        assert command[:6] == [
            sys.executable, "-m", "modal", "secret", "create", "--from-dotenv"
        ]
        staged = Path(command[6])
        assert staged.is_file() and staged.stat().st_mode & 0o777 == 0o600
        calls.append((command, staged.read_text()))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script.subprocess, "run", fake_modal)
    path = tmp_path / ".config" / "multimodal-embeddings-ai-inference" / "modal-secret.env"
    monkeypatch.setattr(sys, "argv", arguments())
    assert script.main() == 0
    initial = saved_token(path.read_text())
    assert len(initial) >= 32 and path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(sys, "argv", arguments("--force"))
    assert script.main() == 0
    assert saved_token(path.read_text()) == initial
    monkeypatch.setattr(sys, "argv", arguments("--force", "--rotate-token"))
    assert script.main() == 0
    rotated = saved_token(path.read_text())
    assert rotated != initial
    assert saved_token(calls[-1][1]) == rotated
    assert "--force" in calls[-1][0] and len(calls) == 6
    for i in (0, 2, 4):
        model_command, model_env = calls[i]
        gateway_command, gateway_env = calls[i + 1]
        assert model_command[-1] == "multimodal-embeddings-model-config"
        assert gateway_command[-1] == "multimodal-embeddings-api"
        assert "VISION_EMBEDDINGS_API_TOKEN" not in model_env
        assert "VISION_EMBEDDINGS_QWEN8_REVISION=" in model_env
        assert "VISION_EMBEDDINGS_WEMM9_REVISION=" in model_env
        assert "VISION_EMBEDDINGS_API_TOKEN=" in gateway_env
    printed = capsys.readouterr().out
    assert initial not in printed and rotated not in printed


def test_rotation_requires_force(monkeypatch, tmp_path):
    script = helper()
    monkeypatch.setattr(script.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", arguments("--rotate-token"))
    monkeypatch.setattr(
        script.subprocess, "run", lambda *a, **k: pytest.fail("Must not call Modal")
    )
    with pytest.raises(SystemExit) as err:
        script.main()
    assert err.value.code == 2
    assert not list(tmp_path.rglob("*.env"))


def test_failed_rotation_preserves_local_file(monkeypatch, tmp_path):
    script = helper()
    monkeypatch.setattr(script.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(script, "_require_modal_tooling", lambda parser: None)
    path = tmp_path / ".config" / "multimodal-embeddings-ai-inference" / "modal-secret.env"
    path.parent.mkdir(parents=True)
    existing = "VISION_EMBEDDINGS_API_TOKEN=original-test-token-" + "x" * 32 + "\n"
    path.write_text(existing)
    path.chmod(0o600)
    monkeypatch.setattr(sys, "argv", arguments("--force", "--rotate-token"))

    def fail(command, *, check):
        assert Path(command[6]).is_file()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(script.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        script.main()
    assert path.read_text() == existing
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob(".modal-secret-*.env"))



@pytest.mark.parametrize(
    ("missing_module", "distribution"),
    [("dotenv", "python-dotenv"), ("modal", "modal")],
)
def test_missing_setup_dependency_fails_before_side_effects(
    monkeypatch, tmp_path, capsys, missing_module, distribution
):
    script = helper()
    monkeypatch.setattr(script.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", arguments())
    monkeypatch.setattr(
        script,
        "find_spec",
        lambda name: None if name == missing_module else object(),
    )
    monkeypatch.setattr(
        script.subprocess, "run", lambda *a, **k: pytest.fail("Must not call Modal")
    )
    with pytest.raises(SystemExit) as err:
        script.main()
    assert err.value.code == 2
    error = capsys.readouterr().err
    assert distribution in error
    assert "python -m pip install -e '.[modal]'" in error
    assert "Traceback" not in error
    assert not list(tmp_path.rglob("*.env"))
    assert not (tmp_path / ".config").exists()


def test_modal_install_extra_contains_dotenv():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]["modal"]
    assert any(value.startswith("modal>=") for value in extras)
    assert any(value.startswith("python-dotenv>=") for value in extras)


@pytest.mark.parametrize("registry_filename", ["client-keys.json", "gateway-next.env"])
def test_setup_force_refuses_existing_managed_registry(
    monkeypatch, tmp_path, registry_filename
):
    script = helper()
    monkeypatch.setattr(script.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(script, "_require_modal_tooling", lambda parser: None)
    directory = tmp_path / ".config" / "multimodal-embeddings-ai-inference"
    directory.mkdir(parents=True)
    original = directory / "modal-secret.env"
    original.write_text("VISION_EMBEDDINGS_API_TOKEN=keep-existing-owner-token\n")
    existing = directory / registry_filename
    existing.write_text("private-registry-marker")
    monkeypatch.setattr(sys, "argv", arguments("--force", "--rotate-token"))
    monkeypatch.setattr(
        script.subprocess, "run", lambda *a, **kw: pytest.fail("Must not call Modal")
    )
    with pytest.raises(SystemExit) as outcome:
        script.main()
    assert outcome.value.code == 2
    assert original.read_text() == "VISION_EMBEDDINGS_API_TOKEN=keep-existing-owner-token\n"
    assert existing.read_text() == "private-registry-marker"
