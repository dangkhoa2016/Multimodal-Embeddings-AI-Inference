"""Offline checks for the live embedding compatibility test harness."""

import hashlib
import importlib.util
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_instruction_compat.py"
SPEC = importlib.util.spec_from_file_location("smoke_instruction_compat", PATH)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def load_smoke_script(name):
    path = PATH.with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke_endpoint(monkeypatch, tmp_path):
    """Real local HTTP servers expose credential forwarding and revision mistakes."""
    state = {"redirect": None, "sink_auth": [], "posts": 0, "wrong_model": None}
    revisions = {"qwen3-vl-embedding-8b": "a" * 40, "wemm-embedding-9b": "b" * 40}

    class Sink(BaseHTTPRequestHandler):
        def do_GET(self):
            state["sink_auth"].append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        do_POST = do_GET

        def log_message(self, *_):
            pass

    sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)

    class Endpoint(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                payload = {"status": "ok", "backend": "modal"}
            elif self.path == "/v1/models":
                payload = {"models": [
                    {"id": model, "revision": revision}
                    for model, revision in revisions.items()
                ]}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            state["posts"] += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if state["redirect"]:
                self.send_response(state["redirect"])
                self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/sink")
                self.end_headers()
                return
            model = body["model"]
            result = {
                "model": model, "backend": "modal", "dimension": 1024,
                "input_sha256": hashlib.sha256(b"authorized image fixture").hexdigest(),
                "vector": [1.0] + [0.0] * 1023, "elapsed_ms": 1.0,
                "revision": "c" * 40 if model == state["wrong_model"] else revisions[model],
            }
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *_):
            pass

    endpoint = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    servers = (endpoint, sink)
    threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.01})
               for s in servers]
    for thread in threads:
        thread.start()
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-test-token-at-least-16")
    monkeypatch.setenv("VISION_EMBEDDINGS_QWEN8_REVISION", "a" * 40)
    monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", "b" * 40)
    image = tmp_path / "authorized.png"
    image.write_bytes(b"authorized image fixture")
    try:
        yield f"http://127.0.0.1:{endpoint.server_port}", image, state
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


@pytest.mark.parametrize("name", ["smoke_both", "smoke_api", "smoke_instruction_compat"])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_smoke_clients_never_forward_credentials_on_redirect(
    monkeypatch, smoke_endpoint, name, status,
):
    url, image, state = smoke_endpoint
    state["redirect"] = status
    module = load_smoke_script(name)
    argv = [name, "--base-url", url, "--image", str(image)]
    if name == "smoke_both":
        argv += ["--mode", "sync", "--allow-mock"]
    monkeypatch.setattr(sys, "argv", argv)
    outcome = 0
    try:
        if name == "smoke_instruction_compat":
            module.post(url, "offline-test-token-at-least-16", "/v1/embeddings",
                        {"model": module.QWEN, "text": "hello"}, 5)
        else:
            outcome = module.main()
    except (HTTPError, KeyError, AssertionError):
        outcome = 1
    assert outcome != 0
    assert state["sink_auth"] == []
    assert state["posts"] == 1


@pytest.mark.parametrize("wrong_model", ["qwen3-vl-embedding-8b", "wemm-embedding-9b"])
def test_smoke_both_rejects_unexpected_checkpoint_revision(
    monkeypatch, smoke_endpoint, capsys, wrong_model,
):
    url, image, state = smoke_endpoint
    state["wrong_model"] = wrong_model
    module = load_smoke_script("smoke_both")
    monkeypatch.setattr(sys, "argv", [
        "smoke_both", "--base-url", url, "--image", str(image),
        "--mode", "sync", "--allow-mock",
    ])
    with pytest.raises(ValueError, match="revision"):
        module.main()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(record.get("model") != wrong_model for record in records)


def test_smoke_both_accepts_expected_checkpoint_revisions(monkeypatch, smoke_endpoint, capsys):
    url, image, _state = smoke_endpoint
    module = load_smoke_script("smoke_both")
    monkeypatch.setattr(sys, "argv", [
        "smoke_both", "--base-url", url, "--image", str(image),
        "--mode", "sync", "--allow-mock",
    ])
    assert module.main() == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["revision"] for record in records if record.get("model")] == [
        "a" * 40, "b" * 40,
    ]
    assert all(record["status"] == "PASS" for record in records)


@pytest.mark.parametrize("revision", [None, "main", "not-a-sha"])
def test_smoke_both_requires_pinned_revisions_before_network(
    monkeypatch, smoke_endpoint, revision,
):
    url, image, state = smoke_endpoint
    if revision is None:
        monkeypatch.delenv("VISION_EMBEDDINGS_QWEN8_REVISION")
    else:
        monkeypatch.setenv("VISION_EMBEDDINGS_QWEN8_REVISION", revision)
    module = load_smoke_script("smoke_both")
    monkeypatch.setattr(sys, "argv", [
        "smoke_both", "--base-url", url, "--image", str(image),
        "--mode", "sync",
    ])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert state["posts"] == 0


def test_synthetic_inputs_are_complete_and_deterministic():
    a, b = smoke.inputs(), smoke.inputs()
    assert a == b
    assert list(a) == ["qwen_text", "qwen_image", "wemm_text", "wemm_image"]


def test_compare_allows_identical_embeddings_and_rejects_drift():
    a = [1.0] + [0.0] * 1023
    smoke.compare(a, a, "identical")
    with pytest.raises(ValueError, match="compatibility"):
        smoke.compare(a, [-1.0] + [0.0] * 1023, "drift")


def test_provenance_does_not_expose_instruction():
    import hashlib

    digest = hashlib.sha256(smoke.LEGACY.encode()).hexdigest()
    response = {
        "embedding_metadata": {
            "preprocessing_version": "qwen3-vl-adapter-v1",
            "instruction_sha256": digest,
            "config_sha256": "a" * 64,
        }
    }
    assert smoke.provenance(response, smoke.QWEN, smoke.LEGACY) == (
        response["embedding_metadata"]
    )
    with pytest.raises(ValueError, match="hash"):
        smoke.provenance(response, smoke.QWEN, smoke.CUSTOM)


def test_valid_vector_rejects_non_normalized_and_non_finite():
    response = {"model": smoke.QWEN, "vector": [1.0] + [0.0] * 1023}
    assert math.isclose(sum(x * x for x in smoke.validate(response, smoke.QWEN)), 1.0)
    with pytest.raises(ValueError):
        smoke.validate({"model": smoke.QWEN, "vector": [float("nan")] * 1024}, smoke.QWEN)
