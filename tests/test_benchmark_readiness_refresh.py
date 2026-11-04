"""Offline regressions for bounded dual-GPU warm-readiness refresh."""
import hashlib
import io
import json
from contextlib import redirect_stdout

import pytest
from test_cold_diagnostics import fake_response, scope_for


def prepared_scope(monkeypatch, tmp_path):
    scope, cells = scope_for(monkeypatch, tmp_path)
    scope.update(RUN_BENCHMARK=True, RUN_COLD=False, RUN_WARM=False,
                 RUN_ALTERNATE=True, RUN_CONCURRENT=False, RUN_MIXED_BATCH=False)
    scope["run_requested_phases"] = {"alternate"}
    scope["run_completed_phases"].add("preflight")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    clock = [239.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    qwen, wemm = scope["MODELS"]
    scope["gpu_readiness"].update({
        qwen: {"worker_id": "a" * 32, "observed_at": 0, "phase": "alternate_setup", "round": 0},
        wemm: {"worker_id": "b" * 32, "observed_at": 239, "phase": "alternate_setup", "round": 0},
    })
    return scope, clock, cells


def test_one_stale_gpu_gets_one_journaled_refresh_with_same_input_profile(
    monkeypatch, tmp_path,
):
    scope, clock, _ = prepared_scope(monkeypatch, tmp_path)
    qwen, wemm = scope["MODELS"]
    calls = []

    def refresh(model, modality, phase, round_number, *, context=None):
        calls.append((model, modality, phase, round_number, context))
        clock[0] += 2
        scope["gpu_readiness"][model] = {
            "worker_id": "c" * 32, "observed_at": clock[0],
            "phase": phase, "round": round_number,
        }

    scope["async_prepare_gpu"] = refresh
    workers = scope["require_pair_gpu_ready"]("alternate", "alternate_setup", 0)
    assert workers == {qwen: "c" * 32, wemm: "b" * 32}
    assert len(calls) == 1
    assert calls[0][:4] == (qwen, "text", "alternate_setup", 0)
    assert calls[0][4]["readiness_refresh"] is True
    assert calls[0][4]["stale_evidence_age_ms"] == 239000.0


def test_no_refresh_when_both_gpu_evidence_are_fresh(monkeypatch, tmp_path):
    scope, clock, _ = prepared_scope(monkeypatch, tmp_path)
    scope["gpu_readiness"][scope["MODELS"][0]]["observed_at"] = clock[0] - 2
    scope["async_prepare_gpu"] = lambda *a, **kw: pytest.fail("No POST allowed")
    workers = scope["require_pair_gpu_ready"]("alternate", "alternate_setup", 0)
    assert len(workers) == 2


@pytest.mark.parametrize("stale_after_refresh", ["both_before", "second_after"])
def test_bounded_refresh_fails_closed_without_second_post(
    monkeypatch, tmp_path, stale_after_refresh,
):
    scope, clock, _ = prepared_scope(monkeypatch, tmp_path)
    if stale_after_refresh == "both_before":
        scope["gpu_readiness"][scope["MODELS"][1]]["observed_at"] = 0
    posts = []

    def slow_refresh(model, modality, phase, round_number, *, context=None):
        posts.append(model)
        clock[0] += 200
        scope["gpu_readiness"][model] = {
            "worker_id": "c" * 32, "observed_at": clock[0],
            "phase": phase, "round": round_number,
        }

    scope["async_prepare_gpu"] = slow_refresh
    expected = (
        "MultipleStaleGPUReadinessEvidence" if stale_after_refresh == "both_before"
        else "StaleGPUReadinessEvidence"
    )
    with pytest.raises(RuntimeError, match=expected), redirect_stdout(io.StringIO()):
        scope["require_pair_gpu_ready"]("alternate", "alternate_setup", 0)
    assert len(posts) == (0 if stale_after_refresh == "both_before" else 1)
    report = json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["summary"] == {}


def test_alternating_slow_wemm_repairs_qwen_then_finalizes_report(
    monkeypatch, tmp_path,
):
    """239s WeMM setup must not turn a cold Qwen into a warm measurement."""
    scope, clock, cells = prepared_scope(monkeypatch, tmp_path)
    scope["gpu_readiness"].clear()
    clock[0] = 0.0
    calls = []
    count = {model: 0 for model in scope["MODELS"]}

    def fake_http(path, body=None, **kwargs):
        if path == "/v1/embeddings/async":
            model = body["model"]
            count[model] += 1
            calls.append(("async", model))
            clock[0] += 239 if model == scope["MODELS"][1] else 1
            worker = (
                "b" if model == scope["MODELS"][1]
                else ("a" if count[model] == 1 else "c")
            ) * 32
            return {
                "job_id": hashlib.sha256(body["request_id"].encode()).hexdigest(),
                "state": "SUCCEEDED",
                "result": fake_response(scope, body, "initial_load", worker),
            }
        assert path == "/v1/embeddings"
        calls.append(("sync", body["model"]))
        clock[0] += 0.1
        worker = ("b" if body["model"] == scope["MODELS"][1] else "c") * 32
        return fake_response(scope, body, "warm", worker)

    scope["http_json"] = fake_http
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["alternate-run"], "mock-alternate", "exec"), scope)
        exec(compile(cells["report"], "mock-report", "exec"), scope)
    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "LIVE_BENCHMARK_MEASURED"
    assert sum(method == "async" for method, _ in calls) == 3
    assert sum(method == "sync" for method, _ in calls) == 10
    assert len(report["trials"]) == 13
    refresh = [row for row in report["trials"] if row.get("readiness_refresh")]
    assert len(refresh) == 1 and refresh[0]["model"] == scope["MODELS"][0]
    assert refresh[0]["async_state"] == "SUCCEEDED"
    assert refresh[0]["async_post_attempts"] == 1
    assert all(row["confirmed"] for row in report["trials"])
    assert report["config"]["maximum_readiness_refresh_per_pair"] == 1


def test_mixed_batch_slow_second_setup_refreshes_first_with_one_extra_post(
    monkeypatch, tmp_path,
):
    scope, cells = scope_for(monkeypatch, tmp_path)
    scope.update(RUN_BENCHMARK=True, RUN_COLD=False, RUN_WARM=False,
                 RUN_ALTERNATE=False, RUN_CONCURRENT=False,
                 RUN_MIXED_BATCH=True, MIXED_BATCH_REPETITIONS=1)
    scope["run_requested_phases"] = {"mixed_batch"}
    scope["run_completed_phases"].add("preflight")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    clock = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    counts = {"async": 0, "batch": 0}
    qwen_calls = [0]

    def fake_http(path, body=None, **kwargs):
        if path == "/v1/embeddings/async":
            counts["async"] += 1
            model = body["model"]
            if model == scope["MODELS"][0]:
                qwen_calls[0] += 1
            clock[0] += 239 if model == scope["MODELS"][1] else 1
            worker = ("b" if model == scope["MODELS"][1]
                      else ("a" if qwen_calls[0] == 1 else "c")) * 32
            return {
                "job_id": hashlib.sha256(body["request_id"].encode()).hexdigest(),
                "state": "SUCCEEDED",
                "result": fake_response(scope, body, "initial_load", worker),
            }
        assert path == "/v1/embeddings/batches"
        counts["batch"] += 1
        clock[0] += 0.5
        return {
            "count": 2,
            "items": [
                fake_response(
                    scope, item, "warm",
                    ("c" if item["model"] == scope["MODELS"][0] else "b") * 32,
                )
                for item in body["items"]
            ],
        }

    scope["http_json"] = fake_http
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["concurrent-run"], "mock-concurrent-defs", "exec"), scope)
        exec(compile(cells["mixed-batch-run"], "mock-mixed", "exec"), scope)
        exec(compile(cells["report"], "mock-mixed-report", "exec"), scope)
    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "LIVE_BENCHMARK_MEASURED"
    assert counts == {"async": 3, "batch": 1}
    assert len(report["trials"]) == 3
    assert sum(bool(row.get("readiness_refresh")) for row in report["trials"]) == 1
    assert len(report["mixed_model_batches"]) == 1
    assert report["mixed_model_batches"][0]["confirmed"] is True
