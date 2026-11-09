"""CPU-only real-ASGI benchmark Bearer preflight; no external HTTP or GPU."""
import hashlib
import io
import json
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.config import Settings

BOOK = Path(__file__).resolve().parents[1] / "notebooks/multimodal_embeddings_benchmark.ipynb"
OWNER = "benchmark-owner-test-token-123456789"
CLIENT = "benchmark-client-test-token-123456789"
INVALID = "benchmark-invalid-test-token-123456789"


@pytest.mark.parametrize("token,authorized", [(OWNER, True), (CLIENT, True), (INVALID, False)])
def test_benchmark_preflight_validates_token_before_any_post_or_idle(
    tmp_path, monkeypatch, token, authorized
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://benchmark.invalid")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", token)
    monkeypatch.setattr("time.sleep", lambda *_: pytest.fail("Must not idle on preflight"))
    settings = Settings(
        api_token=OWNER, backend="modal",
        client_key_hashes={"client_main": (hashlib.sha256(CLIENT.encode()).hexdigest(),)},
    )
    notebook = json.loads(BOOK.read_text(encoding="utf-8"))
    code = {
        item["id"]: "".join(item["source"])
        for item in notebook["cells"] if item["cell_type"] == "code"
    }
    with TestClient(create_app(settings)) as client:
        called = []

        class LocalGateway:
            def open(self, request, timeout=None):
                assert request.get_method() == "GET", "GPU POST is forbidden"
                assert request.full_url.startswith("https://benchmark.invalid/")
                path = request.full_url.removeprefix("https://benchmark.invalid")
                called.append(path)
                response = client.get(path, headers={
                    "Authorization": request.headers.get("Authorization", ""),
                })
                if response.status_code >= 400:
                    raise urllib.error.HTTPError(
                        request.full_url, response.status_code, "Rejected", {}, None
                    )
                return io.BytesIO(response.content)

        monkeypatch.setattr(urllib.request, "build_opener", lambda *_: LocalGateway())
        scope = {}
        with redirect_stdout(io.StringIO()):
            exec(compile(code["config"], "config", "exec"), scope)
            scope["RUN_BENCHMARK"] = True
            if authorized:
                exec(compile(code["client"], "client", "exec"), scope)
            else:
                with pytest.raises(RuntimeError, match="HTTP 401"):
                    exec(compile(code["client"], "client", "exec"), scope)

        assert called[:2] == ["/health", "/v1/models"]
        assert "/v1/models/qwen3-vl-embedding-8b" in called
        if authorized:
            assert "/v1/models/wemm-embedding-9b" in called
            assert "preflight" in scope["run_completed_phases"]
        else:
            assert "/v1/models/wemm-embedding-9b" not in called
            assert "preflight" not in scope["run_completed_phases"]
            report = json.loads(
                (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
            )
            assert report["status"] == "INCOMPLETE_FAILED"
            assert report["failure"]["phase"] == "preflight"


@pytest.mark.parametrize(
    ("environment", "secrets", "expected_prompt_count"),
    [
        ({}, {}, 2),
        ({"VISION_EMBEDDINGS_BASE_URL": "https://env.invalid"},
         {"VISION_EMBEDDINGS_API_TOKEN": CLIENT}, 0),
        ({}, {"VISION_EMBEDDINGS_BASE_URL": "https://secret.invalid",
              "VISION_EMBEDDINGS_API_TOKEN": CLIENT}, 0),
        ({"VISION_EMBEDDINGS_API_TOKEN": CLIENT},
         {"VISION_EMBEDDINGS_BASE_URL": "https://secret.invalid"}, 0),
        ({}, {"VISION_EMBEDDINGS_BASE_URL": "https://secret.invalid"}, 1),
    ],
)
def test_benchmark_credentials_env_secrets_and_hidden_prompt(
    monkeypatch, capsys, environment, secrets, expected_prompt_count,
):
    """Resolve only missing values, without exposing an entered Bearer token."""
    import ast
    import os
    import sys
    import types

    for name in ("VISION_EMBEDDINGS_BASE_URL", "VISION_EMBEDDINGS_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    source = next("".join(c["source"]) for c in json.loads(
        BOOK.read_text())["cells"] if c["id"] == "client")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "resolve_benchmark_credential")
    requests = []

    class FakeSecrets:
        def get_secret(self, name):
            if name not in secrets:
                raise RuntimeError("Private secret failure must not leak")
            return secrets[name]

    module = types.ModuleType("kaggle_secrets")
    module.UserSecretsClient = FakeSecrets
    monkeypatch.setitem(sys.modules, "kaggle_secrets", module)

    def safe_prompt(label):
        requests.append(label)
        return ("https://prompt.invalid" if "HTTPS" in label else CLIENT)

    scope = {"os": os, "Path": Path, "getpass": safe_prompt}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "credential-helper", "exec"), scope)
    resolve = scope["resolve_benchmark_credential"]
    url = resolve("VISION_EMBEDDINGS_BASE_URL", "HTTPS URL", "URL HTTPS")
    token = resolve("VISION_EMBEDDINGS_API_TOKEN", "Bearer token", "Mã truy cập")
    assert url == environment.get("VISION_EMBEDDINGS_BASE_URL", secrets.get(
        "VISION_EMBEDDINGS_BASE_URL", "https://prompt.invalid"))
    assert token == environment.get("VISION_EMBEDDINGS_API_TOKEN", secrets.get(
        "VISION_EMBEDDINGS_API_TOKEN", CLIENT))
    assert len(requests) == expected_prompt_count
    captured = capsys.readouterr().out
    assert CLIENT not in captured
    assert "Private secret failure must not leak" not in captured


def test_missing_interactive_credentials_preserves_previous_report(
    tmp_path, monkeypatch,
):
    """An unattended missing-secret run must not overwrite prior benchmark evidence."""
    import sys
    import types

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    for name in ("VISION_EMBEDDINGS_BASE_URL", "VISION_EMBEDDINGS_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    module = types.ModuleType("kaggle_secrets")

    class AbsentSecrets:
        def get_secret(self, _name):
            raise RuntimeError("Secret permission error: confidential details")

    module.UserSecretsClient = AbsentSecrets
    monkeypatch.setitem(sys.modules, "kaggle_secrets", module)

    def reject_prompt(_label):
        raise EOFError

    monkeypatch.setattr("getpass.getpass", reject_prompt)
    original = '{"status":"LIVE_BENCHMARK_MEASURED","retain":true}\n'
    previous = tmp_path / "multimodal_embeddings_performance_summary.json"
    previous.write_text(original, encoding="utf-8")

    class BlockNetwork:
        def open(self, *_args, **_kwargs):
            pytest.fail("Missing credentials must not reach the API")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: BlockNetwork())
    book = json.loads(BOOK.read_text())
    code = {cell["id"]: "".join(cell["source"])
            for cell in book["cells"] if cell["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), scope)
        # Safe default never asks for credentials.
        exec(compile(code["client"], "client-preview", "exec"), scope)
        assert scope["RUN_BENCHMARK"] is False
        scope["RUN_BENCHMARK"] = True
        with pytest.raises(RuntimeError, match="Interactive input unavailable"):
            exec(compile(code["client"], "client-missing", "exec"), scope)
    assert previous.read_text(encoding="utf-8") == original
    assert scope["run_completed_phases"] == set()
