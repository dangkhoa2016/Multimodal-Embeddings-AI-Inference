"""CPU-only regression: async GPU preparation before sync warm measurements."""
import hashlib
import io
import json
from collections import Counter
from contextlib import redirect_stdout

import pytest
from test_cold_diagnostics import fake_response, scope_for


def armed_scope(monkeypatch, tmp_path):
    scope, cells = scope_for(monkeypatch, tmp_path)
    scope.update(RUN_BENCHMARK=True, RUN_WARM=True, RUN_COLD=False,
                 RUN_ALTERNATE=False, RUN_CONCURRENT=False, RUN_MIXED_BATCH=False)
    scope["run_requested_phases"] = {"warm"}
    scope["run_completed_phases"].add("preflight")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    return scope, cells


def fake_async_gateway(scope, *, sync_state="warm", terminal="SUCCEEDED",
                       worker_changed=False):
    """No network, checkpoints, server or Modal GPU is involved."""
    calls, jobs, polls = [], {}, Counter()
    prepared = Counter()

    def worker(body):
        model = body["model"]
        slot = "a" if model == scope["MODELS"][0] else "b"
        if worker_changed and model == scope["MODELS"][0] and body.get("image_base64"):
            slot = "c"
        return slot * 32

    def http(path, body=None, **kwargs):
        method = "POST" if body is not None else "GET"
        calls.append((method, path, body.get("model") if body else None))
        if path == "/v1/embeddings/async":
            if terminal in {"HTTP_504", "ACK_LOST"}:
                raise RuntimeError(
                    "English: HTTP 504 on /v1/embeddings/async; no retry"
                    if terminal == "HTTP_504"
                    else "English: HTTP network/timeout failure; no retry"
                )
            request_id = body["request_id"]
            job_id = hashlib.sha256(request_id.encode()).hexdigest()
            jobs[job_id] = body
            return {"job_id": job_id, "state": "QUEUED"}
        if path.startswith("/v1/embeddings/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            polls[job_id] += 1
            if terminal == "WRONG_JOB":
                return {"job_id": "f" * 64, "state": "SUCCEEDED"}
            if terminal in {"FAILED", "INDETERMINATE"}:
                return {"job_id": job_id, "state": terminal}
            if terminal == "TIMEOUT":
                return {"job_id": job_id, "state": "RUNNING"}
            body = jobs[job_id]
            model = body["model"]
            prepared[model] += 1
            result = fake_response(
                scope, body, "initial_load" if prepared[model] == 1 else "warm",
                worker(body),
            )
            return {"job_id": job_id, "state": "SUCCEEDED", "result": result}
        assert path == "/v1/embeddings", path
        return fake_response(scope, body, sync_state, worker(body))

    return http, calls, polls


def test_warm_uses_eight_journaled_async_preparations_then_sixteen_sync_measurements(
    monkeypatch, tmp_path,
):
    scope, cells = armed_scope(monkeypatch, tmp_path)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    scope["http_json"], calls, polls = fake_async_gateway(scope)
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["warm-run"], "warm-run", "exec"), scope)
        exec(compile(cells["report"], "report", "exec"), scope)
    report = json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())
    assert report["status"] == "LIVE_BENCHMARK_MEASURED"
    assert Counter(path for method, path, _ in calls if method == "POST") == {
        "/v1/embeddings/async": 8, "/v1/embeddings": 16,
    }
    assert sum(polls.values()) == 8
    assert Counter(row["phase"] for row in report["trials"]) == {"warmup": 8, "warm": 16}
    assert all(row["status"] == "completed" for row in report["trials"])
    assert all(row["async_post_attempts"] == 1 and row["async_state"] == "SUCCEEDED"
               for row in report["trials"] if row["phase"] == "warmup")
    assert all(row["confirmed"] and row["readiness_classification"] == "warm_ready"
               for row in report["trials"] if row["phase"] == "warm")
    assert len({row["async_request_id"] for row in report["trials"]
                if row["phase"] == "warmup"}) == 8
    assert all(value["client"]["n"] == 4 for value in report["summary"].values())


@pytest.mark.parametrize("terminal", [
    "FAILED", "INDETERMINATE", "HTTP_504", "ACK_LOST", "WRONG_JOB", "TIMEOUT",
])
def test_warm_async_fail_closed_preserves_identity_and_no_second_post(
    monkeypatch, tmp_path, terminal,
):
    scope, cells = armed_scope(monkeypatch, tmp_path)
    scope["COLD_JOB_TIMEOUT_SECONDS"] = 60
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    clock = [100.0]
    if terminal == "TIMEOUT":
        monkeypatch.setattr("time.perf_counter", lambda: clock[0])
        monkeypatch.setattr("time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    else:
        monkeypatch.setattr("time.sleep", lambda seconds: None)
    scope["http_json"], calls, _ = fake_async_gateway(scope, terminal=terminal)
    with pytest.raises(RuntimeError), redirect_stdout(io.StringIO()):
        exec(compile(cells["warm-run"], "warm-failed", "exec"), scope)
    report_path = tmp_path / "multimodal_embeddings_performance_summary.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "INCOMPLETE_FAILED"
    assert len(report["trials"]) == 1 and report["trials"][0]["phase"] == "warmup"
    trial = report["trials"][0]
    assert trial["status"] == "failed" and trial["confirmed"] is False
    assert trial["async_request_id"].startswith("warmup-")
    assert trial["async_post_attempts"] == 1
    assert sum(path == "/v1/embeddings/async" for _, path, _ in calls) == 1
    assert not any(path == "/v1/embeddings" for _, path, _ in calls)
    with pytest.raises(RuntimeError):
        exec(compile(cells["warm-run"], "warm-blocked-replay", "exec"), scope)
    assert sum(path == "/v1/embeddings/async" for _, path, _ in calls) == 1
    # Re-running is blocked and the failed run remains incomplete.
    assert json.loads(report_path.read_text())["status"] == "INCOMPLETE_FAILED"
    if terminal in {"HTTP_504", "ACK_LOST"}:
        assert trial["async_job_id"] is None
    else:
        assert trial["async_job_id"] is not None


@pytest.mark.parametrize("sync_state", ["initial_load", "model_reload"])
def test_warm_measured_cold_state_fails_before_more_sync_requests(
    monkeypatch, tmp_path, sync_state,
):
    scope, cells = armed_scope(monkeypatch, tmp_path)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    scope["http_json"], calls, _ = fake_async_gateway(scope, sync_state=sync_state)
    with (
        pytest.raises(RuntimeError, match="WarmGPUStateUnconfirmed"),
        redirect_stdout(io.StringIO()),
    ):
        exec(compile(cells["warm-run"], "warm-sync-not-warm", "exec"), scope)
    report = json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())
    assert report["status"] == "INCOMPLETE_FAILED"
    assert sum(path == "/v1/embeddings/async" for _, path, _ in calls) == 2
    assert sum(path == "/v1/embeddings" for _, path, _ in calls) == 1
    assert report["trials"][-1]["confirmed"] is False
    assert (
        report["trials"][-1]["readiness_classification"]
        == "model_cold_or_worker_changed_during_measurement"
    )
    assert report["summary"] == {}


def test_warm_setup_worker_changed_is_not_silently_relabelled(monkeypatch, tmp_path):
    scope, cells = armed_scope(monkeypatch, tmp_path)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    scope["http_json"], calls, _ = fake_async_gateway(scope, worker_changed=True)
    with (
        pytest.raises(RuntimeError, match="GPUWorkerChangedDuringPreparation"),
        redirect_stdout(io.StringIO()),
    ):
        exec(compile(cells["warm-run"], "warm-worker-changed", "exec"), scope)
    assert sum(path == "/v1/embeddings/async" for _, path, _ in calls) == 2
    assert not any(path == "/v1/embeddings" for _, path, _ in calls)


def test_readiness_expiry_rejects_sync_without_post(monkeypatch, tmp_path):
    scope, _ = armed_scope(monkeypatch, tmp_path)
    model = scope["MODELS"][0]
    scope["gpu_readiness"][model] = {
        "worker_id": "a" * 32,
        "observed_at": scope["time"].perf_counter() - 500,
        "phase": "warmup", "round": 1,
    }
    scope["http_json"] = lambda *args, **kwargs: pytest.fail("No HTTP allowed")
    with pytest.raises(RuntimeError, match="StaleGPUReadinessEvidence"):
        scope["require_recent_gpu"](model, "warm")
    report = json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())
    assert report["status"] == "INCOMPLETE_FAILED" and report["summary"] == {}


def test_mixed_batch_cold_worker_retains_specific_failure_and_no_retry(
    monkeypatch, tmp_path,
):
    """A batch HTTP 200 that cold-loads must not be labelled warm or retried."""
    scope, cells = scope_for(monkeypatch, tmp_path)
    scope.update(
        RUN_BENCHMARK=True, RUN_COLD=False, RUN_WARM=False, RUN_ALTERNATE=False,
        RUN_CONCURRENT=False, RUN_MIXED_BATCH=True, MIXED_BATCH_REPETITIONS=1,
    )
    scope["run_requested_phases"] = {"mixed_batch"}
    scope["run_completed_phases"].add("preflight")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    called = Counter()

    def mock_http(path, body=None, **kwargs):
        called[path] += 1
        if path == "/v1/embeddings/async":
            model = body["model"]
            worker = ("a" if model == scope["MODELS"][0] else "b") * 32
            return {
                "job_id": hashlib.sha256(body["request_id"].encode()).hexdigest(),
                "state": "SUCCEEDED",
                "result": fake_response(scope, body, "warm", worker),
            }
        assert path == "/v1/embeddings/batches"
        return {
            "count": 2,
            "items": [
                fake_response(
                    scope, item, "initial_load" if i == 0 else "warm",
                    ("a" if i == 0 else "b") * 32,
                )
                for i, item in enumerate(body["items"])
            ],
        }

    scope["http_json"] = mock_http
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["concurrent-run"], "concurrent-definitions", "exec"), scope)
    with pytest.raises(RuntimeError, match="WarmGPUStateUnconfirmed"):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["mixed-batch-run"], "mixed-cold-start", "exec"), scope)
    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["error_type"] == "WarmGPUStateUnconfirmed"
    assert report["mixed_model_batches"][0]["confirmed"] is False
    assert report["mixed_model_batches"][0]["status"] == "failed"
    assert report["mixed_model_batches"][0]["readiness_classification"] == (
        "model_cold_or_worker_changed"
    )
    assert called["/v1/embeddings/async"] == 2
    assert called["/v1/embeddings/batches"] == 1
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        exec(compile(cells["mixed-batch-run"], "mixed-replay", "exec"), scope)
    assert called["/v1/embeddings/batches"] == 1


def test_interrupted_async_warmup_marks_partial_report_without_retry(
    monkeypatch, tmp_path,
):
    scope, cells = armed_scope(monkeypatch, tmp_path)
    submitted = []

    def http(path, body=None, **kwargs):
        assert path == "/v1/embeddings/async"
        submitted.append(body["request_id"])
        return {
            "job_id": hashlib.sha256(body["request_id"].encode()).hexdigest(),
            "state": "QUEUED",
        }

    scope["http_json"] = http

    def interrupt_sleep(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("time.sleep", interrupt_sleep)
    with pytest.raises(KeyboardInterrupt):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["warm-run"], "warm-interrupted", "exec"), scope)
    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "INCOMPLETE_INTERRUPTED"
    assert report["trials"][0]["async_state"] == "QUEUED"
    assert report["trials"][0]["status"] == "interrupted"
    assert len(submitted) == 1
    with pytest.raises(RuntimeError):
        exec(compile(cells["warm-run"], "warm-replay", "exec"), scope)
    assert len(submitted) == 1
