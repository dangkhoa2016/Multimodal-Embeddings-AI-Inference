"""CPU-only regression coverage for the cold-start-safe HTTPS smoke client."""

import hashlib
import importlib.util
import io
import json
import sys
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_both.py"
spec = importlib.util.spec_from_file_location("smoke_both_async", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

IMAGE = b"owned synthetic fixture; no real GPU input"
DIGEST = hashlib.sha256(IMAGE).hexdigest()
REVISIONS = {
    "qwen3-vl-embedding-8b": "a" * 40,
    "wemm-embedding-9b": "b" * 40,
}
JOB_ID = "c" * 64


class Response(io.BytesIO):
    def __init__(self, payload, status=200):
        super().__init__(json.dumps(payload).encode())
        self.status = status
        self.headers = {"Retry-After": "3"}


class FakeGateway:
    def __init__(self):
        self.posts = []
        self.gets = []
        self.jobs = {}
        self.post_error = None
        self.terminal = "SUCCEEDED"
        self.model_override = None
        self.catalog_revision_override = None
        self.bad_vector = False

    def embedding(self, request):
        model = request["model"]
        return {
            "model": self.model_override or model,
            "repository": smoke.REPOSITORIES[model],
            "revision": REVISIONS[model],
            "backend": "modal",
            "dimension": 1024,
            "input_sha256": DIGEST,
            "vector": ([0.0] * 1024 if self.bad_vector else
                       [1.0] + [0.0] * 1023),
            "elapsed_ms": 8.0,
            "timing": {"model_state": "initial_load", "model_load_ms": 5.0,
                       "gpu_encode_ms": 3.0},
        }

    def record(self, job_id, state):
        request = self.jobs[job_id]["request"]
        item = {"job_id": job_id, "state": state}
        if state == "SUCCEEDED":
            item["result"] = self.embedding(request)
        return item

    def open(self, request, timeout):
        assert timeout > 0
        assert request.get_header("Authorization") == "Bearer offline-test-token-12345"
        parsed = urllib.parse.urlsplit(request.full_url)
        path, method = parsed.path, request.get_method()
        if method == "GET":
            self.gets.append(path)
            if path == "/health":
                return Response({"status": "ok", "backend": "modal"})
            if path == "/v1/models":
                return Response({
                    "models": [
                        {"id": model, "revision": (
                            "f" * 40 if model == self.catalog_revision_override
                            else revision
                        )}
                        for model, revision in REVISIONS.items()
                    ]
                })
            if "/jobs/by-request/" in path:
                request_id = urllib.parse.unquote(path.rsplit("/", 1)[-1])
                for job_id, data in self.jobs.items():
                    if data["request"]["request_id"] == request_id:
                        return Response(self.record(job_id, data["state"]), (
                            202 if data["state"] in ("QUEUED", "RUNNING") else 200
                        ))
                raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)
            if "/jobs/" in path:
                job_id = path.rsplit("/", 1)[-1]
                if job_id not in self.jobs:
                    raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)
                data = self.jobs[job_id]
                if data["state"] == "QUEUED":
                    data["state"] = "RUNNING"
                elif data["state"] == "RUNNING":
                    data["state"] = self.terminal
                return Response(self.record(job_id, data["state"]), (
                    202 if data["state"] in ("QUEUED", "RUNNING") else 200
                ))
        if method == "POST" and path == "/v1/embeddings/async":
            payload = json.loads(request.data)
            self.posts.append(payload)
            self.jobs[JOB_ID] = {"request": payload, "state": "QUEUED"}
            if self.post_error == "lost_ack":
                raise urllib.error.URLError("simulated ACK loss")
            if isinstance(self.post_error, int):
                raise urllib.error.HTTPError(
                    request.full_url, self.post_error, "unknown", {}, None,
                )
            return Response(self.record(JOB_ID, "QUEUED"), 202)
        raise AssertionError(f"Unexpected request: {method} {path}")


@pytest.fixture
def harness(monkeypatch, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(IMAGE)
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-test-token-12345")
    monkeypatch.setenv("VISION_EMBEDDINGS_QWEN8_REVISION", REVISIONS["qwen3-vl-embedding-8b"])
    monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", REVISIONS["wemm-embedding-9b"])
    monkeypatch.setattr(smoke.time, "sleep", lambda _seconds: None)
    server = FakeGateway()
    monkeypatch.setattr(smoke.urllib.request, "build_opener", lambda *_: server)

    def run(*extra):
        args = [
            "smoke_both", "--base-url", "https://example.invalid",
            "--image", str(image), "--poll-interval", "0.001",
            "--poll-timeout", "2", *extra,
        ]
        monkeypatch.setattr(sys, "argv", args)
        return smoke.main()
    return server, run


def test_default_async_preflight_and_two_models_pass(harness, capsys):
    server, run = harness
    assert run() == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(server.posts) == 2
    assert all("request_id" in item for item in server.posts)
    assert server.posts[0]["request_id"] != server.posts[1]["request_id"]
    assert [record["model"] for record in records if record.get("status") == "PASS"
            and record.get("mode") == "async"] == list(smoke.MODELS)
    assert all("vector" not in record and "Authorization" not in record for record in records)
    assert records[0]["gpu_readiness"] == "not_checked"
    assert any("/jobs/" + JOB_ID in path for path in server.gets)


def test_selected_model_needs_only_its_revision(monkeypatch, harness):
    server, run = harness
    monkeypatch.delenv("VISION_EMBEDDINGS_QWEN8_REVISION")
    assert run("--model", "wemm-embedding-9b") == 0
    assert len(server.posts) == 1
    assert server.posts[0]["model"] == "wemm-embedding-9b"


def test_lost_ack_uses_only_get_by_request_and_never_reposts(harness, capsys):
    server, run = harness
    server.post_error = "lost_ack"
    assert run("--model", "wemm-embedding-9b") == 0
    assert len(server.posts) == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(item.get("status") == "ACK_UNCERTAIN" for item in records)
    assert any("/jobs/by-request/" in path for path in server.gets)


@pytest.mark.parametrize("status", [401, 429, 503, 504])
def test_uncertain_http_post_is_not_resubmitted(harness, status):
    server, run = harness
    server.post_error = status
    assert run("--model", "wemm-embedding-9b") == 0
    assert len(server.posts) == 1


def test_resume_only_reads_by_request_and_recovers(harness):
    server, run = harness
    assert run("--model", "wemm-embedding-9b", "--resume") == 2
    assert server.posts == []
    request_id = smoke._stable_request_id(
        "https://example.invalid", "wemm-embedding-9b", DIGEST, 1024,
        REVISIONS["wemm-embedding-9b"],
    )
    server.jobs[JOB_ID] = {
        "request": {"model": "wemm-embedding-9b", "request_id": request_id},
        "state": "SUCCEEDED",
    }
    assert run("--model", "wemm-embedding-9b", "--resume") == 0
    assert server.posts == []


@pytest.mark.parametrize("status", [401, 429, 503, 504])
def test_lost_ack_and_no_record_stops_without_retry(harness, status):
    server, run = harness
    server.post_error = status
    original = server.open

    def server_with_missing_claim(request, timeout):
        try:
            return original(request, timeout)
        finally:
            if request.get_method() == "POST":
                server.jobs.clear()

    server.open = server_with_missing_claim
    assert run("--model", "qwen3-vl-embedding-8b") == 2
    assert len(server.posts) == 1


@pytest.mark.parametrize("terminal", ["FAILED", "INDETERMINATE"])
def test_terminal_uncertainty_stops_without_new_post(harness, terminal):
    server, run = harness
    server.terminal = terminal
    assert run("--model", "wemm-embedding-9b") == 1
    assert len(server.posts) == 1


def test_poll_timeout_is_not_interpreted_as_gpu_cancellation(harness, capsys):
    server, run = harness
    assert run("--poll-timeout", "0", "--model", "wemm-embedding-9b") == 2
    assert len(server.posts) == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1]["state"] == "UNKNOWN"
    assert records[-1]["action"] == "resume_with_--resume_GET_only"


def test_mismatched_catalog_revision_fails_before_any_post(harness):
    server, run = harness
    server.catalog_revision_override = "wemm-embedding-9b"
    with pytest.raises(ValueError, match="catalog revision"):
        run("--model", "wemm-embedding-9b")
    assert server.posts == []


@pytest.mark.parametrize("condition", ["model_override", "bad_vector"])
def test_response_integrity_enforced(harness, condition):
    server, run = harness
    if condition == "model_override":
        server.model_override = "qwen3-vl-embedding-8b"
    else:
        server.bad_vector = True
    with pytest.raises(ValueError):
        run("--model", "wemm-embedding-9b")
    assert len(server.posts) == 1


def test_invalid_custom_request_id_rejected_before_gpu_submission(harness):
    server, run = harness
    with pytest.raises(ValueError, match="Invalid request_id"):
        run("--model", "wemm-embedding-9b", "--request-id", "short")
    assert server.posts == []
