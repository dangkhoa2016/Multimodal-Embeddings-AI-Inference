"""CPU-only contract tests for the read-only job inspection CLI."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest


def inspector():
    path = Path(__file__).resolve().parents[1] / "scripts" / "inspect_async_job.py"
    spec = importlib.util.spec_from_file_location("inspect_async_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_id_to_job_key_matches_coordinator():
    from multimodal_embeddings_ai_inference.async_jobs import job_key

    script = inspector()
    rid = "demo-" + "a" * 40
    assert script.job_id_for_request(rid) == job_key(rid)
    assert len(script.job_id_for_request(rid)) == 64
    with pytest.raises(ValueError):
        script.job_id_for_request("bad")


def test_inspection_get_only_redacts_vectors_and_credentials():
    script = inspector()
    job_id = hashlib.sha256(b"known-job").hexdigest()
    secret = "test-bearer-please-never-print-me"
    vector = [0.5, 0.5, 0.5, 0.5]
    payload = {
        "job_id": job_id,
        "state": "SUCCEEDED",
        "created_at": 100,
        "updated_at": 200,
        "result": {
            "model": "wemm-embedding-9b",
            "revision": "known-revision",
            "dimension": 4,
            "vector": vector,
            "input_sha256": "a" * 64,
        },
    }
    calls = []

    class Opener:
        def open(self, request, timeout=0):
            calls.append(request)
            assert timeout == 30
            assert request.get_method() == "GET"
            assert request.get_header("Authorization") == "Bearer " + secret
            return io.BytesIO(json.dumps(payload).encode())

    result = script.inspect(
        "https://example.modal.run", secret, job_id, opener=Opener()
    )
    assert len(calls) == 1
    assert result["state"] == "SUCCEEDED"
    assert result["result_metadata"]["vector_length"] == 4
    # A completed GPU job and a GET-only inspection are different facts.
    assert result["inspector_sent_inference_post"] is False
    assert "gpu_request_sent" not in result
    assert "vector" not in result
    serialized = json.dumps(result)
    assert secret not in serialized and '"input_sha256"' not in serialized


@pytest.mark.parametrize("state", ["QUEUED", "RUNNING", "INDETERMINATE", "FAILED"])
def test_inspector_never_claims_original_gpu_job_was_not_submitted(state):
    """A GET-only inspector cannot determine whether the original GPU was started."""
    script = inspector()
    job_id = "a" * 64
    payload = {"job_id": job_id, "state": state}

    class Opener:
        def open(self, request, timeout=0):
            assert request.get_method() == "GET"
            return io.BytesIO(json.dumps(payload).encode())

    result = script.inspect(
        "https://example.modal.run", "test-bearer-token-1234", job_id,
        opener=Opener(),
    )
    assert result["state"] == state
    assert result["inspector_sent_inference_post"] is False
    assert "gpu_request_sent" not in result
    assert "vector" not in result


def test_inspection_rejects_cross_origin_and_invalid_identifiers():
    script = inspector()
    job_id = "a" * 64

    for base in (
        "http://example.com", "https://user:pass@example.com",
        "https://example.com/v1", "https://example.com/?x=1",
    ):
        with pytest.raises(ValueError):
            script.inspect(base, "test-bearer-token-1234", job_id)

    with pytest.raises(ValueError):
        script.inspect("https://example.com", "test-bearer-token-1234", "bad")


def test_missing_backend_record_exits_without_retry(monkeypatch, capsys):
    script = inspector()
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "test-bearer-token-1234")

    def missing(*args, **kwargs):
        raise HTTPError("https://example.modal.run", 404, "not found", None, None)

    monkeypatch.setattr(script, "inspect", missing)
    code = script.main([
        "--base-url", "https://example.modal.run",
        "--request-id", "demo-" + "b" * 40,
    ])
    assert code == 3
    assert "no inference post" in capsys.readouterr().err.lower()
