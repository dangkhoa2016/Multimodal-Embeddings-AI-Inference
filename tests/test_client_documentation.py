"""Verify bilingual client examples together with their actual documentation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_local_mock_setup_runs_the_documented_python_client(tmp_path):
    """The documented persistent server and client share a real token and endpoint."""
    import os
    import site
    import socket
    import subprocess
    import venv

    for name, heading in (
        ("docs/CLIENT_INTEGRATION.md", "## Local mock server"),
        ("docs/CLIENT_INTEGRATION.vi.md", "## Mock server cục bộ"),
    ):
        content = (ROOT / name).read_text()
        assert heading in content, "The local client path needs a runnable server setup"
        section = content.split(heading, 1)[1].split("\n## ", 1)[0]
        blocks = [block.split("```", 1)[0] for block in section.split("```bash\n")[1:]]
        setup, stop = blocks
        code = content.split("```python\n", 1)[1].split("```", 1)[0]
        checkout = tmp_path / Path(name).name
        checkout.mkdir()
        venv.EnvBuilder(with_pip=False).create(checkout / ".venv")
        (checkout / "client.py").write_text(code)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = os.environ.copy()
        # Reuse installed test dependencies with a real activation script, even
        # when pytest itself runs outside a virtualenv (as in CI).
        env.update(
            PYTHONPATH=os.pathsep.join([str(ROOT / "src"), *site.getsitepackages()]),
            MOCK_API_PORT=str(port),
        )
        wait_for_server = '''
ready=false
for attempt in {1..50}; do
  if curl --silent --fail --output /dev/null \\
      -H "Authorization: Bearer $VISION_EMBEDDINGS_API_TOKEN" \\
      "$VISION_EMBEDDINGS_BASE_URL/v1/health"; then ready=true; break; fi
  sleep 0.1
done
test "$ready" = true
'''
        program = (
            "set -e\n"
            "trap 'if [ -n \"${MOCK_API_PID:-}\" ]; then kill \"$MOCK_API_PID\" "
            "2>/dev/null || true; wait \"$MOCK_API_PID\" 2>/dev/null || true; fi' EXIT\n"
            + setup + "\n" + wait_for_server + "\npython client.py\n" + stop
            + '\nwait "$MOCK_API_PID" 2>/dev/null || true\nMOCK_API_PID=""\n'
        )
        result = subprocess.run(
            ["bash", "-c", program], cwd=checkout, env=env, capture_output=True,
            text=True, timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert "qwen3-vl-embedding-8b 1024" in result.stdout


def test_csharp_http_client_timeout_is_configurable_for_cold_load():
    """Keep the bilingual .NET example aligned with Python's 900s client timeout."""
    for name in ("docs/CLIENT_INTEGRATION.md", "docs/CLIENT_INTEGRATION.vi.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "VISION_EMBEDDINGS_REQUEST_TIMEOUT" in text
        assert 'Timeout = TimeSpan.FromSeconds(timeoutSeconds)' in text
        assert '? 900 : int.Parse(timeoutValue)' in text
        assert "timeoutSeconds is < 1 or > 3600" in text
        assert "PostAsJsonAsync(" in text
        assert "retry" in text.lower()


def test_readme_abbreviated_json_retains_numeric_response_types():
    """An illustrative response retains the real types, even when shortened."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # This test first appears before the bilingual READMEs in A-only history.
    if not (root / "README.md").is_file():
        import pytest
        pytest.skip("Bilingual README has not been introduced at this commit.")
    for name, heading in (
        ("README.md", "### Example response (abbreviated)"),
        ("README.vi.md", "### Ví dụ response rút gọn"),
    ):
        readme = (root / name).read_text(encoding="utf-8")
        block = readme.split(heading, 1)[1].split("```json", 1)[1].split("```", 1)[0]
        payload = json.loads(block)
        assert payload["model"] == "wemm-embedding-9b"
        assert payload["dimension"] == 1024
        assert isinstance(payload["vector"], list)
        assert len(payload["vector"]) == 3
        assert all(type(x) in (float, int) for x in payload["vector"])
        assert type(payload["elapsed_ms"]) is float
        assert "1,024" in readme or "1.024" in readme
        assert "CLIENT_INTEGRATION" in readme


def test_readme_python_client_anchor_matches_localized_heading():
    import re
    from urllib.parse import unquote

    import pytest

    root = Path(__file__).resolve().parents[1]
    if not (root / "README.md").exists():
        pytest.skip("README is introduced later in the A-only history.")
    for readme, guide in (
        ("README.md", "docs/CLIENT_INTEGRATION.md"),
        ("README.vi.md", "docs/CLIENT_INTEGRATION.vi.md"),
    ):
        text = (root / readme).read_text(encoding="utf-8")
        match = re.search(r"\(" + re.escape(guide) + r"#([^)]+)\)", text)
        assert match is not None, readme
        fragment = unquote(match.group(1))
        headings = re.findall(r"^#{1,6}\s+(.+?)\s*$",
                              (root / guide).read_text(encoding="utf-8"), re.M)
        slugs = {re.sub(r"[^\w -]", "", h.lower()).replace(" ", "-")
                 for h in headings}
        assert fragment in slugs, (readme, fragment)


def test_minimal_multimodal_client_examples_are_valid_python():
    """Both localized guides must offer a runnable text + image cosine path."""
    for name in ("docs/CLIENT_INTEGRATION.md", "docs/CLIENT_INTEGRATION.vi.md"):
        content = (ROOT / name).read_text(encoding="utf-8")
        snippets = [section.split("```", 1)[0]
                    for section in content.split("```python")[1:]]
        assert len(snippets) >= 2
        complete = snippets[0] + "\n" + snippets[1]
        compile(complete, name, "exec")
        assert "image_base64" in complete
        assert "zip(result[\"vector\"], image_result[\"vector\"], strict=True)" in complete
        assert "RejectRedirects" in complete


def test_minimal_cosine_client_checks_revision_and_config_before_math(monkeypatch, tmp_path):
    """Execute both documented code snippets with synthetic HTTP responses."""
    import io
    import json
    import urllib.request
    from contextlib import redirect_stdout
    from unittest.mock import patch

    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.test")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "not-a-real-token")
    (tmp_path / "YOUR_AUTHORIZED_IMAGE.webp").write_bytes(b"synthetic fixture")

    for guide in ("docs/CLIENT_INTEGRATION.md", "docs/CLIENT_INTEGRATION.vi.md"):
        content = (ROOT / guide).read_text(encoding="utf-8")
        blocks = [part.split("```", 1)[0]
                  for part in content.split("```python")[1:]]
        assert len(blocks) >= 2
        program = compile(blocks[0] + chr(10) + blocks[1], guide, "exec")

        base = {
            "model": "qwen3-vl-embedding-8b",
            "revision": "pinned-checkpoint",
            "dimension": 1024,
            "vector": [1.0] + [0.0] * 1023,
            "embedding_metadata": {"config_sha256": "a" * 64},
        }
        for scenario in ("matched", "revision_changed", "config_changed",
                         "missing_metadata", "missing_revision", "invalid_hash"):
            image = json.loads(json.dumps(base))
            if scenario == "revision_changed":
                image["revision"] = "another-checkpoint"
            elif scenario == "config_changed":
                image["embedding_metadata"]["config_sha256"] = "b" * 64
            elif scenario == "missing_metadata":
                image.pop("embedding_metadata")
            elif scenario == "missing_revision":
                image.pop("revision")
            elif scenario == "invalid_hash":
                image["embedding_metadata"]["config_sha256"] = "not-a-sha"

            calls = []
            class FakeOpener:
                def __init__(self, text_response, image_response, recorded_calls):
                    self.text_response = text_response
                    self.image_response = image_response
                    self.recorded_calls = recorded_calls

                def open(self, request, timeout=None):
                    self.recorded_calls.append(request)
                    payload = (self.text_response if len(self.recorded_calls) == 1
                               else self.image_response)
                    return io.BytesIO(json.dumps(payload).encode("utf-8"))

            output = io.StringIO()
            with patch.object(urllib.request, "build_opener",
                              lambda *args, b=base, i=image, c=calls: FakeOpener(b, i, c)):
                if scenario == "matched":
                    with redirect_stdout(output):
                        exec(program, {})
                    assert "cosine" in output.getvalue().lower()
                else:
                    with redirect_stdout(output), pytest.raises(ValueError):
                        exec(program, {})
                    assert "cosine" not in output.getvalue().lower()
            assert len(calls) == 2, (guide, scenario)
