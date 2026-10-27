"""CPU-only behavioural tests for the standalone demo's async resume journal."""
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from scripts.sync_notebooks import parse_source

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/source/multimodal_embeddings_api_demo.py"
RAW = b"a mountain landscape"
BODY = {
    "model": "wemm-embedding-9b",
    "text": RAW.decode("utf-8"),
    "dimension": 64,
}


def load_helpers(monkeypatch, tmp_path, api_request):
    monkeypatch.setenv(
        "VISION_EMBEDDINGS_NOTEBOOK_CACHE",
        str(tmp_path / "multimodal_embeddings_api_demo_cache.json"),
    )
    cell = parse_source(SOURCE)["embedding-code"]
    prefix = cell.split("def checked_embedding(", 1)[0]
    context = {
        "configuration_ready": True,
        "require_current_configuration": lambda: None,
        "api_request": api_request,
        "json": json,
        "hashlib": hashlib,
        "current_connection_identity": lambda: ("https://example.test", "f" * 64),
        "selected_model_info": {"revision": "pinned-revision"},
        "message_pair": lambda en, vi: en,
    }
    exec(compile(prefix, str(SOURCE), "exec"), context)
    return context


@pytest.mark.parametrize("use_async", [False, True])
def test_declared_modal_backend_requires_approval_when_real_check_disabled(
    notebook_connection, use_async,
):
    """Disabling a backend assertion is not approval for paid requests."""
    import io
    from contextlib import redirect_stdout

    scope, cells, requests, *_ = notebook_connection
    scope["health"]["backend"] = "modal"
    source = cells["embedding-code"].replace(
        "USE_ASYNC_JOBS = True", f"USE_ASYNC_JOBS = {use_async}"
    )
    with redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="approved"):
        exec(compile(source, "embedding-code", "exec"), scope)
    assert not any(method == "POST" for _, method in requests)


@pytest.mark.parametrize("cell_id", ["http-verification-code", "embedding-code", "results-code"])
@pytest.mark.parametrize("name,value", [
    ("VISION_EMBEDDINGS_MODEL", "wemm-embedding-9b"),
    ("VISION_EMBEDDINGS_DIMENSION", "128"),
])
def test_changed_environment_blocks_stale_notebook_actions(
    monkeypatch, verified_notebook_session, name, value, cell_id,
):
    import io
    from contextlib import redirect_stdout

    scope, cells, requests, *_ = verified_notebook_session
    before = len(requests)
    monkeypatch.setenv(name, value)
    with redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="Configuration helpers"):
        exec(compile(cells[cell_id], cell_id, "exec"), scope)
    assert len(requests) == before


def test_gpu_posts_require_explicit_approval(monkeypatch, tmp_path):
    requests = []
    ctx = load_helpers(monkeypatch, tmp_path,
                       lambda path, data=None: requests.append((path, data)))
    assert ctx["USE_ASYNC_JOBS"] is True
    assert ctx["APPROVE_NEW_GPU_JOBS"] is False
    with pytest.raises(RuntimeError, match="No new GPU POSTs approved"):
        ctx["_async_result"](BODY, RAW)
    assert requests == []
    assert not ctx["_ASYNC_CACHE_PATH"].exists()


def test_completed_vector_is_reused_after_notebook_restart(monkeypatch, tmp_path):
    calls = []
    job_id = "a" * 64
    def api_request(path, payload=None):
        calls.append((path, payload))
        if path.endswith("/async"):
            return {"state": "QUEUED", "job_id": job_id}
        if "/jobs/" in path:
            return {
                "state": "SUCCEEDED", "job_id": job_id,
                "result": {"request_id": calls[0][1]["request_id"],
                           "vector": [1.0, 0.0], "backend": "modal"},
            }
        raise AssertionError(path)

    ctx = load_helpers(monkeypatch, tmp_path, api_request)
    ctx["APPROVE_NEW_GPU_JOBS"] = True
    result, rid, _ = ctx["_async_result"](BODY, RAW)
    assert len(calls) == 2 and calls[0][0].endswith("/async")
    assert calls[1][0].endswith("/jobs/" + job_id)
    ctx["_store_verified_async_result"](rid, result)

    # Fresh helper scope reads the verified vector from the on-disk checkpoint.
    after_restart = load_helpers(
        monkeypatch, tmp_path,
        lambda path, payload=None: pytest.fail("Cached vector made an HTTP call"),
    )
    restored, restored_id, elapsed = after_restart["_async_result"](BODY, RAW)
    assert restored["request_id"] == result["request_id"] and restored_id == rid
    assert elapsed is None
    assert restored["_client_timing"]["cache_hit"] is True
    cache_file = ctx["_ASYNC_CACHE_PATH"]
    assert stat.S_IMODE(os.stat(cache_file).st_mode) == 0o600
    content = cache_file.read_text()
    assert "a mountain landscape" not in content
    assert "https://example.test" not in content
    assert "Bearer" not in content


def test_resumes_pending_job_with_get_only(monkeypatch, tmp_path):
    calls = []
    job_id = "b" * 64

    def pending(path, payload=None):
        calls.append((path, payload))
        return {"state": "QUEUED", "job_id": job_id}

    ctx = load_helpers(monkeypatch, tmp_path, pending)
    ctx["APPROVE_NEW_GPU_JOBS"] = True
    ctx["MAX_POLL_SECONDS"] = 0  # Immediate test-only pause, never waits for GPU.
    with pytest.raises(RuntimeError, match="Job still pending"):
        ctx["_async_result"](BODY, RAW)
    assert sum(path.endswith("/async") for path, _ in calls) == 1

    def done(path, payload=None):
        calls.append((path, payload))
        assert path.endswith("/jobs/" + job_id)
        return {"state": "SUCCEEDED", "job_id": job_id,
                "result": {"request_id": rid, "vector": [1.0]}}

    rid = calls[0][1]["request_id"]
    recovered = load_helpers(monkeypatch, tmp_path, done)
    assert recovered["APPROVE_NEW_GPU_JOBS"] is False
    result, _, _ = recovered["_async_result"](BODY, RAW)
    assert result["request_id"] == rid
    assert sum(path.endswith("/async") for path, _ in calls) == 1


def test_uncertain_response_recovers_by_get_without_second_post(monkeypatch, tmp_path):
    """Lost POST ACK is reconciled by GET with the same deterministic server key."""
    import hashlib

    posts = []

    def lost(path, body=None):
        posts.append((path, body))
        raise RuntimeError("network timeout")

    scope = load_helpers(monkeypatch, tmp_path, lost)
    scope["APPROVE_NEW_GPU_JOBS"] = True
    with pytest.raises(RuntimeError, match="network timeout"):
        scope["_async_result"](BODY, RAW)
    rid = posts[0][1]["request_id"]
    assert scope["_ASYNC_CACHE"]["records"][rid]["state"] == "UNCERTAIN"
    server_id = hashlib.sha256(
        ("multimodal-embeddings-job-v1:" + rid).encode("utf-8")
    ).hexdigest()

    gets = []

    def existing_job(path, body=None):
        gets.append((path, body))
        assert body is None
        assert path.endswith("/jobs/by-request/" + rid)
        return {"state": "SUCCEEDED", "job_id": server_id,
                "result": {"request_id": rid, "vector": [1.0]}}

    resumed = load_helpers(monkeypatch, tmp_path, existing_job)
    resumed["APPROVE_NEW_GPU_JOBS"] = True
    result, _, _ = resumed["_async_result"](BODY, RAW)
    assert result["request_id"] == rid
    assert len(posts) == 1 and len(gets) == 1
    assert resumed["_ASYNC_CACHE"]["records"][rid]["job_id"] == server_id


def test_uncertain_expired_backend_record_never_resubmits(monkeypatch, tmp_path):
    """A missing/expired server job is not authorization for another GPU POST."""
    attempts = []

    def lost(path, body=None):
        attempts.append((path, body))
        raise RuntimeError("network timeout")

    first = load_helpers(monkeypatch, tmp_path, lost)
    first["APPROVE_NEW_GPU_JOBS"] = True
    with pytest.raises(RuntimeError, match="network timeout"):
        first["_async_result"](BODY, RAW)

    def no_job(path, body=None):
        assert path.startswith("/v1/embeddings/jobs/") and body is None
        raise RuntimeError("HTTP 404: job record unavailable")

    resumed = load_helpers(monkeypatch, tmp_path, no_job)
    resumed["APPROVE_NEW_GPU_JOBS"] = True
    with pytest.raises(RuntimeError, match="404"):
        resumed["_async_result"](BODY, RAW)
    assert len(attempts) == 1
    record = resumed["_ASYNC_CACHE"]["records"][attempts[0][1]["request_id"]]
    assert record["state"] == "UNCERTAIN" and len(record["job_id"]) == 64



def test_failed_job_never_auto_retries(monkeypatch, tmp_path):
    calls = []
    def broken(path, body=None):
        calls.append(path)
        if path.endswith("/async"):
            return {"state": "FAILED", "job_id": "d" * 64}
        raise AssertionError("A failed job must not be queried or resubmitted")

    scope = load_helpers(monkeypatch, tmp_path, broken)
    scope["APPROVE_NEW_GPU_JOBS"] = True
    with pytest.raises(RuntimeError, match="GPU job failed"):
        scope["_async_result"](BODY, RAW)
    scope = load_helpers(monkeypatch, tmp_path, broken)
    with pytest.raises(RuntimeError, match="This job failed"):
        scope["_async_result"](BODY, RAW)
    assert len(calls) == 1


def test_bearer_rotation_preserves_request_identity(monkeypatch, tmp_path):
    """Rotating a credential must not create another billable GPU job."""
    ctx = load_helpers(monkeypatch, tmp_path, lambda *_: None)

    def make_id(token):
        ctx["current_connection_identity"] = lambda: (
            "https://example.test",
            hashlib.sha256(token.encode("utf-8")).hexdigest(),
        )
        return ctx["_async_identity"](BODY, RAW)

    first = make_id("old-placeholder-bearer")
    rotated = make_id("new-placeholder-bearer")
    assert rotated == first
    ctx["current_connection_identity"] = lambda: (
        "https://another-deployment.test", "another-token-hash",
    )
    assert ctx["_async_identity"](BODY, RAW) != first


def test_client_timing_separates_cache_hit_and_polling(monkeypatch, tmp_path):
    """A cached vector is not a zero-millisecond GPU execution."""
    calls = []
    def fake(path, payload=None):
        calls.append(path)
        if path.endswith("/async"):
            return {"state": "QUEUED", "job_id": "e" * 64}
        return {
            "state": "SUCCEEDED", "job_id": "e" * 64,
            "result": {"request_id": rid, "vector": [1.0]},
        }

    ctx = load_helpers(monkeypatch, tmp_path, fake)
    rid = ctx["_async_identity"](BODY, RAW)
    ctx["APPROVE_NEW_GPU_JOBS"] = True
    result, _, duration = ctx["_async_result"](BODY, RAW)
    assert duration is not None and duration >= 0
    assert result["_client_timing"]["cache_hit"] is False
    assert result["_client_timing"]["submission_ms"] is not None
    assert result["_client_timing"]["total_client_ms"] >= 0
    ctx["_store_verified_async_result"](rid, result)

    resumed = load_helpers(monkeypatch, tmp_path,
                           lambda *_: pytest.fail("Cache hit made an HTTP call"))
    cached, _, reused_duration = resumed["_async_result"](BODY, RAW)
    assert reused_duration is None
    assert cached["_client_timing"]["cache_hit"] is True
    assert cached["_client_timing"]["total_client_ms"] is None


def test_rotated_token_can_resume_saved_legacy_job_by_explicit_id(monkeypatch, tmp_path):
    """A deliberate mapping only polls the saved job, never POSTs GPU work."""
    original = load_helpers(
        monkeypatch, tmp_path, lambda *_: pytest.fail("Unexpected HTTP in setup"),
    )
    old_id = original["_async_identity"](BODY, RAW, legacy=True)
    job_id = "c" * 64
    original["_ASYNC_CACHE"]["records"][old_id] = {
        "state": "RUNNING", "job_id": job_id,
    }
    original["_save_async_cache"]()
    calls = []

    def get_only(path, payload=None):
        calls.append((path, payload))
        assert path == "/v1/embeddings/jobs/" + job_id and payload is None
        return {
            "state": "SUCCEEDED", "job_id": job_id,
            "result": {"request_id": old_id, "vector": [1.0]},
        }

    rotated = load_helpers(monkeypatch, tmp_path, get_only)
    rotated["current_connection_identity"] = (
        lambda: ("https://example.test", "0" * 64)
    )
    stable = rotated["_async_identity"](BODY, RAW)
    assert rotated["_async_identity"](BODY, RAW, legacy=True) != old_id
    monkeypatch.setenv("VISION_EMBEDDINGS_LEGACY_REQUEST_ID_MAP",
                       json.dumps({stable: old_id}))
    result, request_id, elapsed = rotated["_async_result"](BODY, RAW)
    assert request_id == old_id and result["request_id"] == old_id
    assert elapsed is not None
    assert len(calls) == 1


def test_legacy_id_mapping_missing_record_fails_closed_even_when_approved(
        monkeypatch, tmp_path):
    calls = []
    ctx = load_helpers(
        monkeypatch, tmp_path,
        lambda path, payload=None: calls.append((path, payload)),
    )
    ctx["APPROVE_NEW_GPU_JOBS"] = True
    stable = ctx["_async_identity"](BODY, RAW)
    monkeypatch.setenv("VISION_EMBEDDINGS_LEGACY_REQUEST_ID_MAP",
                       json.dumps({stable: "demo-" + "a" * 40}))
    with pytest.raises(RuntimeError, match="not found in saved cache"):
        ctx["_async_result"](BODY, RAW)
    assert calls == []


def test_legacy_checkpoint_recovered_without_gpu_post(monkeypatch, tmp_path):
    """A notebook upgrade must not strand previous credential-bound cache keys."""
    ctx = load_helpers(monkeypatch, tmp_path,
                       lambda *_: pytest.fail("Legacy cache unexpectedly sent HTTP"))
    legacy = ctx["_async_identity"](BODY, RAW, legacy=True)
    stable = ctx["_async_identity"](BODY, RAW)
    assert legacy != stable
    ctx["_ASYNC_CACHE"]["records"][legacy] = {
        "state": "SUCCEEDED",
        "result": {"request_id": legacy, "vector": [1.0]},
    }
    ctx["_save_async_cache"]()
    resumed = load_helpers(monkeypatch, tmp_path,
                           lambda *_: pytest.fail("Legacy cache unexpectedly sent HTTP"))
    result, rid, elapsed = resumed["_async_result"](BODY, RAW)
    assert rid == legacy and result["request_id"] == legacy
    assert elapsed is None and result["_client_timing"]["cache_hit"] is True


def test_uncertain_per_client_key_rotation_resolves_server_v2_without_post(monkeypatch, tmp_path):
    """A client key rotates but the stable tenant namespace still owns the job."""
    sent = []
    first = load_helpers(monkeypatch, tmp_path,
                         lambda path, payload=None: sent.append((path, payload))
                         or (_ for _ in ()).throw(RuntimeError("network timeout")))
    first["APPROVE_NEW_GPU_JOBS"] = True
    with pytest.raises(RuntimeError, match="network timeout"):
        first["_async_result"](BODY, RAW)
    rid = sent[0][1]["request_id"]
    legacy_id = hashlib.sha256(
        ("multimodal-embeddings-job-v1:" + rid).encode()
    ).hexdigest()
    client_job = hashlib.sha256(
        ("multimodal-embeddings-job-v2:client_a:" + rid).encode()
    ).hexdigest()
    assert client_job != legacy_id
    reads = []

    def readonly(path, payload=None):
        reads.append((path, payload))
        assert payload is None
        assert path == "/v1/embeddings/jobs/by-request/" + rid
        return {
            "state": "SUCCEEDED", "job_id": client_job,
            "result": {"request_id": rid, "vector": [1.0]},
        }

    resumed = load_helpers(monkeypatch, tmp_path, readonly)
    resumed["APPROVE_NEW_GPU_JOBS"] = False
    result, _, _ = resumed["_async_result"](BODY, RAW)
    assert result["request_id"] == rid
    assert len(sent) == len(reads) == 1
    assert resumed["_ASYNC_CACHE"]["records"][rid]["job_id"] == client_job


def test_uncertain_legacy_server_404_lookup_falls_back_to_get_only(monkeypatch, tmp_path):
    calls = []
    initial = load_helpers(monkeypatch, tmp_path, lambda path, data=None: (
        calls.append((path, data)),
        (_ for _ in ()).throw(RuntimeError("network timeout")),
    )[1])
    initial["APPROVE_NEW_GPU_JOBS"] = True
    with pytest.raises(RuntimeError, match="network timeout"):
        initial["_async_result"](BODY, RAW)
    rid = calls[0][1]["request_id"]
    expected = hashlib.sha256(
        ("multimodal-embeddings-job-v1:" + rid).encode()
    ).hexdigest()
    lookups = []

    def old_server(path, payload=None):
        lookups.append((path, payload))
        assert payload is None
        if path.endswith("/jobs/by-request/" + rid):
            raise RuntimeError("API request failed: HTTP 404 on " + path)
        assert path.endswith("/jobs/" + expected)
        return {
            "state": "SUCCEEDED", "job_id": expected,
            "result": {"request_id": rid, "vector": [1.0]},
        }

    resumed = load_helpers(monkeypatch, tmp_path, old_server)
    resumed["APPROVE_NEW_GPU_JOBS"] = False
    result, _, _ = resumed["_async_result"](BODY, RAW)
    assert result["request_id"] == rid
    assert len(lookups) == 2 and len(calls) == 1
