"""Offline regression coverage for GPU cold diagnostics and legacy API compatibility."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from multimodal_embeddings_ai_inference.embedding_config import embedding_metadata

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/multimodal_embeddings_benchmark.ipynb"
)


def scope_for(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VISION_EMBEDDINGS_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_EMBEDDINGS_API_TOKEN", raising=False)
    monkeypatch.delenv("BENCHMARK_IMAGE_PATH", raising=False)
    cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]}
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["config"], "config", "exec"), scope)
        exec(compile(cells["client"], "client", "exec"), scope)
        exec(compile(cells["client-request-helpers"], "client-request-helpers", "exec"), scope)
    return scope, cells


def fake_response(scope, body, state, worker_id=None):
    return {
        "model": body["model"], "backend": "modal", "revision": "test-model",
        "request_id": body["request_id"], "dimension": 1024,
        "embedding_metadata": embedding_metadata(
            body["model"], "test-model", 1024, body.get("instruction")
        ).model_dump(),
        "input_sha256": scope["hashes"]["image" if "image_base64" in body else "text"],
        "vector": [1.0] + [0.0] * 1023,
        "elapsed_ms": 2.0,
        "timing": {
            "model_state": state, "gpu_worker_first_request": state == "initial_load",
            "gpu_worker_id": worker_id, "model_load_ms": 1.0 if state != "warm" else 0.0,
            "gpu_worker_ms": 1.5, "gpu_encode_ms": 1.0, "batch_size": 1,
            "gpu_dispatch_overhead_ms": 0.1,
        },
    }


def test_cold_only_records_confirmed_and_unconfirmed(monkeypatch, tmp_path):
    scope, cells = scope_for(monkeypatch, tmp_path)
    scope.update(RUN_BENCHMARK=True, RUN_COLD=True, RUN_WARM=False, RUN_ALTERNATE=False,
                 COLD_REPETITIONS=1)
    scope["run_requested_phases"] = {"cold"}
    scope["run_completed_phases"].add("preflight")  # Already validated in a separate test.
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    calls = []
    def reply(path, body):
        assert path == "/v1/embeddings"
        calls.append(body)
        return fake_response(scope, body, "initial_load" if len(calls) == 1 else "warm", "a" * 32)
    scope["http_json"] = reply
    with patch("time.sleep", return_value=None), redirect_stdout(io.StringIO()):
        exec(compile(cells["cold-run"], "cold-run", "exec"), scope)
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["warm-run"], "warm-run", "exec"), scope)
        exec(compile(cells["alternate-run"], "alternate-run", "exec"), scope)
        exec(compile(cells["report"], "report", "exec"), scope)
    report_path = tmp_path / "multimodal_embeddings_performance_summary.json"
    result = json.loads(report_path.read_text())
    assert len(calls) == len(result["trials"]) == 2
    assert [r["confirmed"] for r in result["trials"]] == [True, False]
    classes = [r["cold_classification"] for r in result["trials"]]
    assert classes == ["confirmed_initial_load", "warm"]
    for trial in result["trials"]:
        for name in ("started_at_utc", "finished_at_utc", "idle_started_at_utc",
                     "idle_finished_at_utc", "idle_elapsed_seconds", "gpu_worker_id"):
            assert name in trial
        assert trial["gpu_worker_id"] == "a" * 32


def test_failed_cold_persists_idle_context(monkeypatch, tmp_path):
    scope, cells = scope_for(monkeypatch, tmp_path)
    scope.update(RUN_BENCHMARK=True, RUN_COLD=True, RUN_WARM=False, RUN_ALTERNATE=False,
                 COLD_REPETITIONS=1)
    scope["run_completed_phases"].add("preflight")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    called = []
    def failure(path, body):
        called.append(body)
        raise RuntimeError("sensitive URL must not appear in report")
    scope["http_json"] = failure
    with patch("time.sleep", return_value=None), redirect_stdout(io.StringIO()):
        with pytest.raises(RuntimeError):
            exec(compile(cells["cold-run"], "cold-run", "exec"), scope)
    raw = (tmp_path/"multimodal_embeddings_performance_summary.json").read_text()
    report = json.loads(raw)
    assert len(called) == 1 and report["status"] == "INCOMPLETE_FAILED"
    trial = report["trials"][0]
    assert trial["cold_classification"] == "failed"
    assert "idle_started_at_utc" in trial and "idle_elapsed_seconds" in trial
    assert "sensitive URL" not in raw and scope["TEXT_SAMPLE"] not in raw


def test_legacy_server_without_worker_id(monkeypatch, tmp_path):
    scope, _ = scope_for(monkeypatch, tmp_path)
    scope["RUN_BENCHMARK"] = True
    def legacy(path, body):
        result = fake_response(scope, body, "warm")
        result["timing"].pop("gpu_worker_id")
        return result
    scope["http_json"] = legacy
    trial = scope["sample"](scope["MODELS"][0], "text", "warm", 1)
    assert trial["gpu_worker_id"] is None


def test_warm_progress_prints_per_block_not_per_request(monkeypatch, tmp_path):
    scope, cells = scope_for(monkeypatch, tmp_path)
    scope.update(RUN_BENCHMARK=True, RUN_COLD=False, RUN_WARM=True, RUN_ALTERNATE=False,
                 WARM_REPETITIONS=4, WARMUP_REQUESTS=2)
    scope["run_completed_phases"].add("preflight")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    calls = []
    def sample(model, modality, phase, round_no):
        calls.append((model, modality, phase, round_no))
        return {"model_state": "warm", "gpu_worker_first_request": False, "confirmed": False}
    scope["sample"] = sample
    stream = io.StringIO()
    with redirect_stdout(stream):
        exec(compile(cells["warm-run"], "warm-run", "exec"), scope)
    assert len(calls) == 2 * 2 * (2 * 2 + 4)
    assert stream.getvalue().count("Warm block done") == 4


def test_single_model_cold_selection(monkeypatch, tmp_path):
    scope, cells = scope_for(monkeypatch, tmp_path)
    scope.update(RUN_BENCHMARK=True, RUN_COLD=True, RUN_WARM=False, RUN_ALTERNATE=False,
                 COLD_REPETITIONS=2, COLD_ACTIVE_MODELS=(scope["MODELS"][1],))
    scope["run_completed_phases"].add("preflight")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    used=[]
    def reply(path,body):
        used.append(body["model"])
        return fake_response(scope,body,"initial_load","b"*32)
    scope["http_json"]=reply
    with patch("time.sleep",return_value=None),redirect_stdout(io.StringIO()):
        exec(compile(cells["cold-run"],"cold-run","exec"),scope)
    assert used == [scope["MODELS"][1]] * 2
