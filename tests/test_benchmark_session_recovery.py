"""Verify optional-only benchmark provenance with fake API responses."""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from multimodal_embeddings_ai_inference.embedding_config import embedding_metadata

BOOK = Path(__file__).resolve().parents[1] / "notebooks/multimodal_embeddings_benchmark.ipynb"


def notebook():
    book = json.loads(BOOK.read_text())
    return book


@pytest.mark.parametrize("phase", ["concurrent-run", "mixed-batch-run"])
def failure_scope(monkeypatch, tmp_path):
    """Load the real notebook code in an offline, disposable execution scope."""
    monkeypatch.chdir(tmp_path)
    book = notebook()
    code = {c["id"]: "".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), scope)
        exec(compile(code["client"], "client", "exec"), scope)
        exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
    scope.update(
        RUN_BENCHMARK=True, RUN_COLD=False, RUN_WARM=False, RUN_ALTERNATE=False,
        RUN_CONCURRENT=True, RUN_MIXED_BATCH=True, CONCURRENT_WORKERS=2,
        CONCURRENT_REQUESTS=8, MIXED_BATCH_REPETITIONS=3,
    )
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    scope["run_completed_phases"].add("preflight")
    scope["run_requested_phases"] = {"concurrent", "mixed_batch"}
    return code, scope, tmp_path / "multimodal_embeddings_performance_summary.json"

@pytest.mark.parametrize(
    "case",
    [
        "valid", "optional_worker_absent", "empty_timing", "missing_state",
        "unknown_state", "missing_first_flag", "non_boolean_first_flag",
        "missing_batch_size", "boolean_batch_size", "wrong_batch_size",
        "missing_worker_ms", "negative_load_ms", "missing_api_ms",
        "invalid_worker_id", "non_finite_encode_ms",
    ],
)
def test_concurrent_gpu_timing_instrumentation_contract(monkeypatch, tmp_path, case):
    """Completed concurrent responses require the same GPU evidence as sequential."""
    code, scope, _ = failure_scope(monkeypatch, tmp_path)
    scope["RUN_CONCURRENT"] = False
    exec(compile(code["concurrent-run"], "concurrent-definitions", "exec"), scope)
    model = scope["MODELS"][0]
    timing = {
        "model_state": "warm",
        "gpu_worker_first_request": False,
        "gpu_worker_id": "a" * 32,
        "batch_size": 1,
        "gpu_worker_ms": 3.0,
        "model_load_ms": 0.0,
        "gpu_encode_ms": 2.0,
        "gpu_dispatch_overhead_ms": 0.1,
    }
    if case == "optional_worker_absent":
        timing.pop("gpu_worker_id")
    elif case == "empty_timing":
        timing.clear()
    elif case == "missing_state":
        timing.pop("model_state")
    elif case == "unknown_state":
        timing["model_state"] = "unverified_state"
    elif case == "missing_first_flag":
        timing.pop("gpu_worker_first_request")
    elif case == "non_boolean_first_flag":
        timing["gpu_worker_first_request"] = 0
    elif case == "missing_batch_size":
        timing.pop("batch_size")
    elif case == "boolean_batch_size":
        timing["batch_size"] = True
    elif case == "wrong_batch_size":
        timing["batch_size"] = 2
    elif case == "missing_worker_ms":
        timing.pop("gpu_worker_ms")
    elif case == "negative_load_ms":
        timing["model_load_ms"] = -0.1
    elif case == "invalid_worker_id":
        timing["gpu_worker_id"] = "private-device-name"
    elif case == "non_finite_encode_ms":
        timing["gpu_encode_ms"] = float("nan")

    def fake_http(path, body):
        assert path == "/v1/embeddings"
        return {
            "model": body["model"], "backend": "modal",
            "request_id": body["request_id"],
            "dimension": scope["DIMENSION"],
            "revision": "fixture-revision", "input_sha256": scope["hashes"]["text"],
            "vector": [1.0] + [0.0] * 1023,
            "embedding_metadata": embedding_metadata(
                model, "fixture-revision", 1024, body.get("instruction")
            ).model_dump(),
            "elapsed_ms": None if case == "missing_api_ms" else 4.0,
            "timing": timing,
        }

    scope["http_json"] = fake_http
    trial, revision, metadata = scope["observed_concurrent"](model, 0)
    assert revision == ("fixture-revision" if case in ("valid", "optional_worker_absent")
                        else None)
    if case in ("valid", "optional_worker_absent"):
        assert trial["status"] == "completed"
        assert trial["gpu_instrumentation_verified"] is True
        assert trial["gpu_worker_first_request"] is False
        assert trial["model_state"] == "warm"
        assert trial["gpu_worker_id"] == (None if case == "optional_worker_absent"
                                           else timing["gpu_worker_id"])
        assert metadata["preprocessing_version"]
    else:
        assert trial["status"] == "failed"
        assert trial["error_type"] == "RuntimeError"
        assert "gpu_instrumentation_verified" not in trial
        assert metadata is None


def test_invalid_concurrent_instrumentation_stops_and_redacts_report(monkeypatch, tmp_path):
    """An HTTP 200 response with timing={} must not count toward throughput."""
    code, scope, path = failure_scope(monkeypatch, tmp_path)
    scope.update(RUN_CONCURRENT=True, RUN_MIXED_BATCH=False, CONCURRENT_REQUESTS=2,
                 CONCURRENT_WORKERS=2)

    def fake_http(route, body):
        assert route == "/v1/embeddings"
        return {
            "model": body["model"], "backend": "modal",
            "request_id": body["request_id"],
            "dimension": 1024, "revision": "fixture",
            "input_sha256": scope["hashes"]["text"],
            "vector": [1.0] + [0.0] * 1023,
            "embedding_metadata": embedding_metadata(
                body["model"], "fixture", 1024, body.get("instruction")
            ).model_dump(),
            "elapsed_ms": 3.0, "timing": {},
        }

    scope["http_json"] = fake_http
    with redirect_stdout(io.StringIO()):
        with pytest.raises(RuntimeError, match="Benchmark stopped"):
            exec(compile(code["concurrent-run"], "concurrent-run", "exec"), scope)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["phase"] == "concurrent"
    assert report["concurrent"]["summary"] == []
    assert report["mixed_model_batches"] == []
    assert all(t["status"] == "failed" for t in report["concurrent"]["trials"])
    assert not any(t.get("gpu_instrumentation_verified") for t in report["concurrent"]["trials"])


def run_warm_fixture(monkeypatch, tmp_path):
    """Start the real notebook client with an offline GET catalog and fake POSTs."""
    import urllib.request

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.modal.run")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-not-a-secret-12345678")
    book = notebook()
    code = {
        c["id"]: "".join(c["source"]) for c in book["cells"]
        if c["cell_type"] == "code"
    }
    scope = {}

    class OfflinePreflight:
        def open(self, request, timeout=None):
            assert request.get_method() == "GET"
            if request.full_url.endswith("/health"):
                payload = {"backend": "modal"}
            elif request.full_url.endswith("/v1/models"):
                payload = {
                    "models": [
                        {"id": "qwen3-vl-embedding-8b"},
                        {"id": "wemm-embedding-9b"},
                    ],
                }
            else:
                model = request.full_url.rsplit("/v1/models/", 1)[-1]
                assert model in ("qwen3-vl-embedding-8b", "wemm-embedding-9b")
                assert request.headers.get("Authorization", "").startswith("Bearer ")
                payload = {"id": model}
            return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: OfflinePreflight())

    def start(configuration=None):
        with redirect_stdout(io.StringIO()):
            exec(compile(configuration or code["config"], "config", "exec"), scope)
            scope["RUN_BENCHMARK"] = True
            exec(compile(code["client"], "client", "exec"), scope)
            exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)

    sent = []

    def fake_http(path, body, **kwargs):
        assert path in {"/v1/embeddings", "/v1/embeddings/async"}
        sent.append((body["model"], body["request_id"]))
        model = body["model"]
        response = {
            "model": model, "backend": "modal",
            "request_id": body["request_id"],
            "dimension": 1024, "revision": "offline-revision",
            "input_sha256": scope["hashes"]["text" if "text" in body else "image"],
            "vector": [1.0] + [0.0] * 1023,
            "embedding_metadata": embedding_metadata(
                model, "offline-revision", 1024, body.get("instruction")
            ).model_dump(),
            "elapsed_ms": 2.0,
            "timing": {
                "model_state": "warm", "gpu_worker_first_request": False,
                "gpu_worker_id": ("a" if model == scope["MODELS"][0] else "b") * 32,
                "batch_size": 1, "gpu_worker_ms": 1.5,
                "model_load_ms": 0.0, "gpu_encode_ms": 1.1,
                "gpu_dispatch_overhead_ms": 0.2,
            },
        }
        if path == "/v1/embeddings/async":
            return {"job_id": "a" * 64, "state": "SUCCEEDED",
                    "result": response}
        return response

    def run_warm():
        scope["http_json"] = fake_http
        with redirect_stdout(io.StringIO()):
            exec(compile(code["warm-run"], "warm-run", "exec"), scope)

    def run_report():
        with redirect_stdout(io.StringIO()):
            exec(compile(code["report"], "report", "exec"), scope)
        return json.loads(
            (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
        )

    start()
    assert scope["run_completed_phases"] == {"preflight"}
    return code, scope, sent, start, run_warm, run_report


def test_rerunning_warm_after_config_cell_edit_never_combines_measurements(
    monkeypatch, tmp_path
):
    """Reproduce reported 24→56 corruption and require clean 24→32 instead."""
    code, scope, sent, start, run_warm, run_report = run_warm_fixture(
        monkeypatch, tmp_path
    )
    run_warm()
    first = run_report()
    assert len(sent) == len(first["trials"]) == 24
    assert first["config"]["warm_attempts_per_modality_per_model"] == 4
    original_run_id = first["config"]["run_id"]
    changed = code["config"].replace(
        "WARM_REPETITIONS = 4 #@param",
        "WARM_REPETITIONS = 6 #@param",
    )
    assert changed != code["config"]
    # Simulate notebook users re-executing configuration only.
    with redirect_stdout(io.StringIO()):
        exec(compile(changed, "modified-config", "exec"), scope)
    scope["RUN_BENCHMARK"] = True
    with pytest.raises(RuntimeError, match="RunConfigurationChanged"):
        run_warm()
    assert len(sent) == 24  # No new POST was sent.
    incomplete = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert incomplete["status"] == "INCOMPLETE_FAILED"
    assert incomplete["failure"] == {
        "phase": "warm", "error_type": "RunConfigurationChanged"
    }
    assert incomplete["summary"] == {}
    assert len(incomplete["trials"]) == 24
    assert incomplete["config"]["run_id"] == original_run_id

    # Rerun client/preflight to begin a fresh, independently identified run.
    start(changed)
    run_warm()
    second = run_report()
    assert len(sent) == 24 + 32
    assert len(second["trials"]) == 32  # No old samples retained.
    assert second["config"]["warm_attempts_per_modality_per_model"] == 6
    assert second["config"]["run_id"] != original_run_id
    assert second["status"] == "LIVE_BENCHMARK_MEASURED"
    assert second["config"]["run_configuration_sha256"] != (
        first["config"]["run_configuration_sha256"]
    )
    for model in scope["MODELS"]:
        for modality in ("text", "image"):
            assert second["summary"]["warm / " + model + " / " + modality]["attempts"] == 6


def test_repeated_completed_warm_phase_rejected_before_post(monkeypatch, tmp_path):
    _, scope, sent, _, run_warm, run_report = run_warm_fixture(monkeypatch, tmp_path)
    run_warm()
    assert run_report()["status"] == "LIVE_BENCHMARK_MEASURED"
    with pytest.raises(RuntimeError, match="PhaseAlreadyCompleted"):
        run_warm()
    assert len(sent) == 24
    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["error_type"] == "PhaseAlreadyCompleted"


@pytest.mark.parametrize(
    "change",
    ("repetitions", "text_input", "instruction_mode", "phase_selection", "disabled"),
)
def test_report_rejects_stale_active_run_without_new_requests(
    monkeypatch, tmp_path, change
):
    _, scope, sent, _, run_warm, run_report = run_warm_fixture(monkeypatch, tmp_path)
    run_warm()
    assert len(sent) == 24
    if change == "repetitions":
        scope["WARM_REPETITIONS"] = 6
    elif change == "text_input":
        scope["TEXT_SAMPLE"] = "changed text that was never measured"
    elif change == "instruction_mode":
        scope["QWEN_INSTRUCTION_MODE"] = "custom"
    elif change == "phase_selection":
        scope["RUN_MIXED_BATCH"] = True
    else:
        scope["RUN_BENCHMARK"] = False
    with pytest.raises(RuntimeError, match="RunConfigurationChanged"):
        run_report()
    assert len(sent) == 24
    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["phase"] == "report"
    assert report["summary"] == {}
    assert len(report["trials"]) == 24


def test_report_rejects_duplicate_trial_even_if_marked_complete(monkeypatch, tmp_path):
    _, scope, sent, _, run_warm, run_report = run_warm_fixture(monkeypatch, tmp_path)
    run_warm()
    scope["rows"].append(dict(scope["rows"][0]))  # Simulate an accidental stale append.
    with pytest.raises(RuntimeError, match="TrialCountMismatch"):
        run_report()
    assert len(sent) == 24
    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["summary"] == {}


def test_repeating_report_does_not_duplicate_trials(monkeypatch, tmp_path):
    _, _, sent, _, run_warm, run_report = run_warm_fixture(monkeypatch, tmp_path)
    run_warm()
    first = run_report()
    second = run_report()
    assert len(first["trials"]) == len(second["trials"]) == 24
    assert len(sent) == 24
    assert first["config"]["run_id"] == second["config"]["run_id"]


def test_interrupt_warm_during_fourth_post_preserves_rows_and_prevents_retry(
    monkeypatch, tmp_path
):
    """An interrupted in-flight POST cannot authorize a rerun of its warm cell."""
    _, scope, sent, start, run_warm, run_report = run_warm_fixture(monkeypatch, tmp_path)
    original_checked = scope["_sample_checked"]
    attempts = 0

    def interrupted_request(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 4:
            # The server may already have accepted a request when a user presses Stop.
            sent.append(("interrupted-in-flight", "unknown-response"))
            raise KeyboardInterrupt
        return original_checked(*args)

    scope["_sample_checked"] = interrupted_request
    initial_run = scope["run_id"]
    with pytest.raises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
        run_warm()
    assert attempts == 4 and len(sent) == 6  # Two async warm-ups precede sync measurements.
    assert scope["run_started_phases"] == {"warm"}
    assert scope["run_completed_phases"] == {"preflight"}
    assert scope["run_failure"] == {"phase": "warm", "error_type": "KeyboardInterrupt"}
    output = tmp_path / "multimodal_embeddings_performance_summary.json"
    report = json.loads(output.read_text())
    assert report["status"] == "INCOMPLETE_INTERRUPTED"
    assert report["summary"] == {}
    assert report["config"]["run_id"] == initial_run
    assert len(report["trials"]) == 6
    assert [row["status"] for row in report["trials"]] == [
        "completed", "completed", "completed", "completed", "completed", "interrupted"
    ]
    assert report["trials"][-1]["confirmed"] is False
    assert "unknown-response" not in output.read_text()

    # Re-executing warm must never dispatch the 24 scheduled requests.
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        run_warm()
    assert len(sent) == 6
    assert output.read_text() == json.dumps(report, indent=2, ensure_ascii=False) + "\n"

    # Only a fresh preflight resets the failure and starts a new, isolated run.
    scope["_sample_checked"] = original_checked
    start()
    run_warm()
    fresh = run_report()
    assert fresh["status"] == "LIVE_BENCHMARK_MEASURED"
    assert fresh["config"]["run_id"] != initial_run
    assert len(fresh["trials"]) == 24
    assert len(sent) == 6 + 24


def test_phase_start_marker_blocks_rerun_even_without_interrupt_handler(
    monkeypatch, tmp_path
):
    """Simulate a UI Stop that bypasses Python's KeyboardInterrupt handler."""
    _, scope, sent, _, run_warm, _ = run_warm_fixture(monkeypatch, tmp_path)

    # SystemExit models a forced cell termination outside the explicit handler.
    def force_stop(*_args):
        raise SystemExit(130)

    scope["_sample_checked"] = force_stop
    with pytest.raises(SystemExit), redirect_stdout(io.StringIO()):
        run_warm()
    assert scope["run_failure"] is None
    assert scope["run_started_phases"] == {"warm"}
    assert scope["run_completed_phases"] == {"preflight"}
    incomplete = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert incomplete["status"] == "IN_PROGRESS" and incomplete["phase"] == "warm"
    # Even without a caught exception, no subsequent request can be scheduled.
    with pytest.raises(RuntimeError, match="PhaseAlreadyStarted"):
        run_warm()
    assert len(sent) == 2  # Async warm-up completed before the forced sync Stop.
    rejected = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert rejected["status"] == "INCOMPLETE_FAILED"
    assert rejected["failure"]["error_type"] == "PhaseAlreadyStarted"


def test_interrupt_during_cold_idle_never_sends_a_post(monkeypatch, tmp_path):
    """Idle sleep is also guarded: an interrupted cold phase cannot be resumed."""
    code, scope, sent, start, _, _ = run_warm_fixture(monkeypatch, tmp_path)
    cold_config = (
        code["config"]
        .replace('BENCHMARK_MODE = "warm"', 'BENCHMARK_MODE = "cold"')
        .replace("COLD_REPETITIONS = 5", "COLD_REPETITIONS = 1")
    )
    start(cold_config)

    def stop_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("time.sleep", stop_sleep)
    with pytest.raises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
        exec(compile(code["cold-run"], "cold-run", "exec"), scope)
    assert scope["run_started_phases"] == {"cold"}
    assert sent == []
    partial = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert partial["status"] == "INCOMPLETE_INTERRUPTED"
    assert partial["failure"]["phase"] == "cold"
    assert partial["trials"] == []
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        exec(compile(code["cold-run"], "cold-rerun", "exec"), scope)
    assert sent == []


def test_optional_mixed_batch_interrupted_before_response_rejects_retry(
    monkeypatch, tmp_path
):
    """A batch POST can be in flight when the notebook stops."""
    code, scope, sent, start, _, _ = run_warm_fixture(monkeypatch, tmp_path)
    batch_config = (
        code["config"]
        .replace('BENCHMARK_MODE = "warm"', 'BENCHMARK_MODE = "mixed_batch"')
        .replace("MIXED_BATCH_REPETITIONS = 3", "MIXED_BATCH_REPETITIONS = 1")
    )
    start(batch_config)
    seen = []

    def fake_batch(path, body):
        seen.append((path, len(body["items"])))
        raise KeyboardInterrupt

    scope["async_prepare_gpu"] = lambda model, modality, phase, index: scope[
        "gpu_readiness"].update({model: {
            "worker_id": ("a" if model.startswith("qwen") else "b") * 32,
            "observed_at": scope["time"].perf_counter(),
            "phase": phase, "round": index,
        }})
    scope["http_json"] = fake_batch
    with pytest.raises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
        exec(compile(code["mixed-batch-run"], "batch", "exec"), scope)
    assert seen == [("/v1/embeddings/batches", 2)]
    partial = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert partial["status"] == "INCOMPLETE_INTERRUPTED"
    assert partial["failure"]["phase"] == "mixed_batch"
    assert partial["summary"] == {}
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        exec(compile(code["mixed-batch-run"], "batch-rerun", "exec"), scope)
    assert len(seen) == 1


def test_benchmark_quickstart_arms_run_before_client_preflight(monkeypatch, tmp_path):
    """A first-run guide must never turn the flag on after API preflight."""
    nb = notebook()
    intro = next("".join(cell["source"]) for cell in nb["cells"]
                 if cell["id"] == "intro")
    assert "before** API client/preflight" in intro
    assert "trước** cell API client/preflight" in intro
    assert "RUN_BENCHMARK=True" in intro
    code, scope, sent, start, run_warm, run_report = run_warm_fixture(
        monkeypatch, tmp_path
    )
    assert "RUN_BENCHMARK = False #@param" in code["config"]
    start()
    assert scope["RUN_BENCHMARK"] is True
    assert scope["run_completed_phases"] == {"preflight"}
    assert scope["run_started_phases"] == set()
    run_warm()
    report = run_report()
    assert report["status"] == "LIVE_BENCHMARK_MEASURED"
    assert len(sent) == len(report["trials"]) == 24




def test_full_mode_offline_runs_all_five_phases_and_one_report(monkeypatch, tmp_path):
    """Simulate full mode without network, GPU or real cold idle."""
    code, scope, _sent, start, _run_warm, run_report = run_warm_fixture(
        monkeypatch, tmp_path
    )
    selected = code["config"]
    for old, new in (
        ('BENCHMARK_MODE = "warm"', 'BENCHMARK_MODE = "full"'),
        ("COLD_REPETITIONS = 5", "COLD_REPETITIONS = 1"),
        ("WARM_REPETITIONS = 4", "WARM_REPETITIONS = 2"),
        ("ALTERNATE_REPETITIONS = 5", "ALTERNATE_REPETITIONS = 1"),
        ("CONCURRENT_REQUESTS = 4", "CONCURRENT_REQUESTS = 2"),
        ("MIXED_BATCH_REPETITIONS = 3", "MIXED_BATCH_REPETITIONS = 1"),
    ):
        selected = selected.replace(old, new)
    start(selected)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    submitted = []
    polled = []
    registry = {}

    def fake_result(body):
        model, request_id = body["model"], body["request_id"]
        cold = request_id.startswith("cold-")
        return {
            "model": model, "backend": "modal", "request_id": request_id,
            "dimension": 1024, "revision": "offline-revision",
            "input_sha256": scope["hashes"]["text" if "text" in body else "image"],
            "vector": [1.0] + [0.0] * 1023,
            "embedding_metadata": embedding_metadata(
                model, "offline-revision", 1024, body.get("instruction")
            ).model_dump(),
            "elapsed_ms": 4.0, "timing": {
                "model_state": "initial_load" if cold else "warm",
                "gpu_worker_first_request": cold,
                "gpu_worker_id": ("a" if model == scope["MODELS"][0] else "b") * 32,
                "batch_size": 1, "gpu_worker_ms": 3.0,
                "model_load_ms": 1.5 if cold else 0.0,
                "gpu_encode_ms": 0.5, "gpu_dispatch_overhead_ms": 0.1,
            },
        }

    def fake_http(path, body=None, **kwargs):
        if path.startswith("/v1/embeddings/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            polled.append(job_id)
            return {"job_id": job_id, "state": "SUCCEEDED",
                    "result": fake_result(registry[job_id])}
        submitted.append(path)
        if path == "/v1/embeddings/async":
            job_id = ("a" if body["model"] == scope["MODELS"][0] else "b") * 64
            registry[job_id] = body
            return {"job_id": job_id, "state": "QUEUED"}
        if path == "/v1/embeddings":
            return fake_result(body)
        assert path == "/v1/embeddings/batches"
        return {"count": len(body["items"]),
                "items": [fake_result(item) for item in body["items"]]}

    scope["http_json"] = fake_http
    assert scope["benchmark_config"]["benchmark_mode"] == "full"
    assert scope["run_requested_phases"] == {
        "cold", "warm", "alternate", "concurrent", "mixed_batch"
    }
    with redirect_stdout(io.StringIO()):
        for phase in ("cold-run", "warm-run", "alternate-run",
                      "concurrent-run", "mixed-batch-run"):
            exec(compile(code[phase], phase, "exec"), scope)

    report = run_report()
    assert report["status"] == "LIVE_BENCHMARK_MEASURED"
    assert report["config"]["benchmark_mode"] == "full"
    assert scope["run_completed_phases"] == {
        "preflight", "cold", "warm", "alternate", "concurrent", "mixed_batch"
    }
    assert sleeps.count(210) == 2 and sleeps.count(15) == 16
    assert len(submitted) == 31
    assert submitted.count("/v1/embeddings") == 14
    assert submitted.count("/v1/embeddings/async") == 16
    assert len(polled) == 16
    assert submitted.count("/v1/embeddings/batches") == 1
    assert len(scope["rows"]) == 26
    assert len(report["concurrent"]["trials"]) == 4
    assert len(report["mixed_model_batches"]) == 1
    assert len({row["sequence"] for row in scope["rows"]}) == len(scope["rows"])
    with pytest.raises(RuntimeError, match="PhaseAlreadyCompleted"):
        with redirect_stdout(io.StringIO()):
            exec(compile(code["cold-run"], "cold-rerun", "exec"), scope)
    assert len(submitted) == 31
    saved = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert saved["status"] == "INCOMPLETE_FAILED"
    assert saved["failure"]["phase"] == "cold"


@pytest.mark.parametrize(
    ("terminal", "failure_classification"),
    [
        ("FAILED", "async_job_failed_unclassified"),
        ("INDETERMINATE", "async_indeterminate"),
        ("HTTP_504", "http_504_unconfirmed_cause"),
        ("WRONG_JOB", "async_request_or_validation_failed"),
        ("WAIT_TIMEOUT", "async_job_wait_timeout_unconfirmed"),
    ],
)
def test_cold_async_fail_closed_and_never_resends_post(
    monkeypatch, tmp_path, terminal, failure_classification,
):
    code, scope, _sent, start, _run_warm, _run_report = run_warm_fixture(
        monkeypatch, tmp_path
    )
    configuration = code["config"].replace(
        'BENCHMARK_MODE = "warm"', 'BENCHMARK_MODE = "cold"'
    ).replace("COLD_REPETITIONS = 5", "COLD_REPETITIONS = 1").replace(
        "COLD_JOB_TIMEOUT_SECONDS = 1800", "COLD_JOB_TIMEOUT_SECONDS = 60"
    )
    start(configuration)
    clock = [100.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    submitted = []
    polled = []
    job_id = "a" * 64

    def fake_http(path, body=None, **kwargs):
        if path == "/v1/embeddings/async":
            submitted.append(body["request_id"])
            saved = json.loads(
                (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
            )
            assert saved["trials"][-1]["async_request_id"] == body["request_id"]
            assert saved["trials"][-1]["async_state"] == "SUBMITTING"
            if terminal == "HTTP_504":
                raise RuntimeError("English: HTTP 504 on /v1/embeddings/async; no retry")
            return {"state": "QUEUED", "job_id": job_id}
        assert path == "/v1/embeddings/jobs/" + job_id
        polled.append(path)
        if terminal == "WRONG_JOB":
            return {"state": "SUCCEEDED", "job_id": "b" * 64}
        if terminal == "WAIT_TIMEOUT":
            return {"state": "QUEUED", "job_id": job_id}
        return {"state": terminal, "job_id": job_id}

    scope["http_json"] = fake_http
    with pytest.raises(RuntimeError), redirect_stdout(io.StringIO()):
        exec(compile(code["cold-run"], "cold-async-failure", "exec"), scope)
    assert len(submitted) == 1
    assert len(polled) == (0 if terminal == "HTTP_504" else
                           3 if terminal == "WAIT_TIMEOUT" else 1)
    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["phase"] == "cold"
    assert len(report["trials"]) == 1
    trial = report["trials"][0]
    assert trial["status"] == "failed"
    assert trial["async_request_id"] == submitted[0]
    assert trial["async_job_id"] == (None if terminal == "HTTP_504" else job_id)
    assert trial["failure_classification"] == failure_classification
    with pytest.raises(RuntimeError):
        exec(compile(code["cold-run"], "no-cold-retry", "exec"), scope)
    assert len(submitted) == 1
