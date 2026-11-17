"""Verify optional-only benchmark provenance with fake API responses."""

import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from multimodal_embeddings_ai_inference.embedding_config import embedding_metadata

BOOK = Path(__file__).resolve().parents[1] / "notebooks/multimodal_embeddings_benchmark.ipynb"


def notebook():
    book = json.loads(BOOK.read_text())
    return book


def stub_gpu_preparation(scope):
    """Provide isolated warm worker readiness for non-network phase tests.

    Real async POST, polling and fail-closed behavior have their own regression tests.
    """
    def prepare(model, modality, phase, round_number):
        worker = ("0" if model == scope["MODELS"][0] else "1") * 32
        scope["sequence"] += 1
        row = {
            "sequence": scope["sequence"], "phase": phase, "model": model,
            "modality": modality, "round": round_number, "status": "completed",
            "confirmed": True, "model_state": "warm",
            "gpu_worker_first_request": False, "gpu_worker_id": worker,
        }
        scope["rows"].append(row)
        scope["gpu_readiness"][model] = {
            "worker_id": worker, "observed_at": scope["time"].perf_counter(),
            "phase": phase, "round": round_number,
        }
        return row
    scope["async_prepare_gpu"] = prepare


@pytest.mark.parametrize("phase", ["concurrent-run", "mixed-batch-run"])
@pytest.mark.parametrize("drift", [False, True])
def test_optional_provenance(phase, drift, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    nb = notebook()
    code = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    g = {}
    with redirect_stdout(io.StringIO()):
        exec(code["config"], g)
        exec(code["client"], g)
        exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), g)
    g.update(
        RUN_BENCHMARK=True,
        RUN_WARM=False,
        RUN_COLD=False,
        RUN_ALTERNATE=False,
        RUN_CONCURRENT=phase == "concurrent-run",
        RUN_MIXED_BATCH=phase == "mixed-batch-run",
        CONCURRENT_REQUESTS=2,
        CONCURRENT_WORKERS=2,
        MIXED_BATCH_REPETITIONS=2,
    )
    g["run_requested_phases"] = {
        "concurrent" if phase == "concurrent-run" else "mixed_batch"
    }
    g["run_completed_phases"].add("preflight")  # Offline fixture simulates successful GETs.
    # The offline fixture intentionally simulates a new run after changing defaults.
    g["run_configuration_snapshot"] = g["current_run_configuration"]()
    count = defaultdict(int)
    first = g["MODELS"][0]

    def result(body, rev):
        model = body["model"]
        return {
            "model": model,
            "backend": "modal",
            "request_id": body["request_id"],
            "dimension": 1024,
            "revision": rev,
            "input_sha256": g["hashes"]["text"],
            "vector": [1.0] + [0.0] * 1023,
            "embedding_metadata": embedding_metadata(model, rev, 1024).model_dump(),
            "elapsed_ms": 3.0,
            "timing": {
                "model_state": "warm", "gpu_worker_first_request": False,
                "gpu_worker_id": ("0" * 32 if model == first else "1" * 32),
                "batch_size": 1, "gpu_worker_ms": 2.0, "model_load_ms": 0.0,
                "gpu_encode_ms": 1.0, "gpu_dispatch_overhead_ms": 0.1,
            },
        }

    def fake(path, body=None):
        if path == "/v1/embeddings":
            model = body["model"]
            count[model] += 1
            rev = "r2" if drift and model == first and count[model] == 2 else "r1"
            return result(body, rev)
        assert path == "/v1/embeddings/batches"
        count["batch"] += 1
        data = [
            result(x, "r2" if drift and count["batch"] == 2 and x["model"] == first else "r1")
            for x in body["items"]
        ]
        return {"count": 2, "items": data}

    g["http_json"] = fake
    stub_gpu_preparation(g)
    with redirect_stdout(io.StringIO()):
        if phase == "mixed-batch-run":
            exec(code["concurrent-run"], g)  # Supplies uuid4 as in Run All.
        if drift:
            with pytest.raises(RuntimeError, match="Benchmark stopped"):
                exec(code[phase], g)
        else:
            exec(code[phase], g)
            exec(code["report"], g)
    report = json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())
    for model in g["MODELS"]:
        assert report["models"][model]["revision"] == g["revisions"].get(model)
        assert report["embedding_configs"].get(model) == g["embedding_configs"].get(model)
    assert report["status"] == ("INCOMPLETE_FAILED" if drift else "LIVE_BENCHMARK_MEASURED")
    if drift:
        assert report["summary"] == {}
        assert report["failure"]["phase"] in {"concurrent", "mixed_batch"}
    if drift:
        trials = (
            report["concurrent"]["trials"]
            if phase == "concurrent-run"
            else report["mixed_model_batches"]
        )
        assert any(t["status"] == "failed" for t in trials)


@pytest.mark.parametrize(
    "defect",
    (
        "valid", "optional_gpu_encode_unavailable", "missing_timing",
        "missing_state", "missing_first_flag", "boolean_first_flag",
        "missing_worker_elapsed", "negative_worker_elapsed",
        "nan_worker_elapsed", "infinite_model_load", "missing_model_load",
        "boolean_batch_size", "mismatched_batch_size", "invalid_worker_id",
        "missing_api_elapsed", "invalid_dimension",
    ),
)
def test_mixed_batch_validates_each_gpu_response_and_reports_failure(
    defect, monkeypatch, tmp_path
):
    """No real HTTP: one invalid per-worker response must fail the whole batch."""
    monkeypatch.chdir(tmp_path)
    nb = notebook()
    code = {
        c["id"]: "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code"
    }
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), scope)
        exec(compile(code["client"], "client", "exec"), scope)
        exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
        exec(compile(code["concurrent-run"], "concurrent-run", "exec"), scope)

    scope.update(
        RUN_BENCHMARK=True, RUN_COLD=False, RUN_WARM=False,
        RUN_ALTERNATE=False, RUN_CONCURRENT=False, RUN_MIXED_BATCH=True,
        MIXED_BATCH_REPETITIONS=2,
    )
    scope["run_requested_phases"] = {"mixed_batch"}
    scope["run_completed_phases"].add("preflight")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    requests = []

    def fake_http(path, body):
        assert path == "/v1/embeddings/batches"
        requests.append(path)
        result = []
        for index, item in enumerate(body["items"]):
            model = item["model"]
            data = {
                "model": model, "backend": "modal",
                "request_id": item["request_id"],
                "dimension": 1024, "revision": "fixture-r1",
                "input_sha256": scope["hashes"]["text"],
                "vector": [1.0] + [0.0] * 1023,
                "embedding_metadata": embedding_metadata(
                    model, "fixture-r1", 1024
                ).model_dump(),
                "elapsed_ms": 2.0,
                "timing": {
                    "model_state": "warm",
                    "gpu_worker_first_request": False,
                    "gpu_worker_id": ("a" if index == 0 else "b") * 32,
                    "batch_size": 1,
                    "gpu_worker_ms": 1.6,
                    "model_load_ms": 0.0,
                    "gpu_encode_ms": 1.1,
                    "gpu_dispatch_overhead_ms": 0.3,
                },
            }
            # Corrupt only the second model to ensure the whole POST is rejected.
            if index == 1:
                timing = data["timing"]
                if defect == "optional_gpu_encode_unavailable":
                    timing["gpu_encode_ms"] = None
                    timing["gpu_dispatch_overhead_ms"] = None
                elif defect == "missing_timing":
                    del data["timing"]
                elif defect == "missing_state":
                    del timing["model_state"]
                elif defect == "missing_first_flag":
                    del timing["gpu_worker_first_request"]
                elif defect == "boolean_first_flag":
                    timing["gpu_worker_first_request"] = "false"
                elif defect == "missing_worker_elapsed":
                    del timing["gpu_worker_ms"]
                elif defect == "negative_worker_elapsed":
                    timing["gpu_worker_ms"] = -1.0
                elif defect == "nan_worker_elapsed":
                    timing["gpu_worker_ms"] = float("nan")
                elif defect == "infinite_model_load":
                    timing["model_load_ms"] = float("inf")
                elif defect == "missing_model_load":
                    del timing["model_load_ms"]
                elif defect == "boolean_batch_size":
                    timing["batch_size"] = True
                elif defect == "mismatched_batch_size":
                    timing["batch_size"] = 2
                elif defect == "invalid_worker_id":
                    timing["gpu_worker_id"] = "not-a-valid-id"
                elif defect == "missing_api_elapsed":
                    del data["elapsed_ms"]
                elif defect == "invalid_dimension":
                    data["dimension"] = True
            result.append(data)
        return {"count": len(result), "items": result}

    scope["http_json"] = fake_http
    stub_gpu_preparation(scope)
    original_prepare = scope["async_prepare_gpu"]
    def prepared_for_batch(model, modality, phase, round_number):
        trial = original_prepare(model, modality, phase, round_number)
        worker = ("a" if model == scope["MODELS"][0] else "b") * 32
        trial["gpu_worker_id"] = worker
        scope["gpu_readiness"][model]["worker_id"] = worker
        return trial
    scope["async_prepare_gpu"] = prepared_for_batch
    report_path = tmp_path / "multimodal_embeddings_performance_summary.json"
    if defect in {"valid", "optional_gpu_encode_unavailable"}:
        with redirect_stdout(io.StringIO()):
            exec(compile(code["mixed-batch-run"], "mixed-batch-run", "exec"), scope)
            exec(compile(code["report"], "report", "exec"), scope)
        report = json.loads(report_path.read_text())
        assert report["status"] == "LIVE_BENCHMARK_MEASURED"
        assert len(requests) == 2
        assert len(report["mixed_model_batches"]) == 2
        for trial in report["mixed_model_batches"]:
            assert trial["gpu_instrumentation_verified"] is True
            assert set(trial["gpu_states"]) == set(scope["MODELS"])
            assert all(meta["worker_batch_size"] == 1
                       for meta in trial["gpu_states"].values())
    else:
        with pytest.raises(RuntimeError, match="Benchmark stopped"):
            with redirect_stdout(io.StringIO()):
                exec(compile(code["mixed-batch-run"], "mixed-batch-run", "exec"), scope)
        report = json.loads(report_path.read_text())
        assert report["status"] == "INCOMPLETE_FAILED"
        assert report["failure"]["phase"] == "mixed_batch"
        assert report["summary"] == {}
        assert len(requests) == 1  # No retry and no next batch.
        assert len(report["mixed_model_batches"]) == 1
        assert report["mixed_model_batches"][0]["status"] == "failed"
        assert report["mixed_model_batches"][0]["error_type"] == "RuntimeError"
        assert "api_token" not in report_path.read_text().lower()


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
    stub_gpu_preparation(scope)
    return code, scope, tmp_path / "multimodal_embeddings_performance_summary.json"


def test_first_concurrent_failure_aborts_before_next_model_or_mixed_batch(monkeypatch, tmp_path):
    code, scope, output = failure_scope(monkeypatch, tmp_path)
    old_marker = "previous successful report must not survive"
    output.write_text(json.dumps({"status": "LIVE_BENCHMARK_MEASURED", "secret": old_marker}))
    attempted = []

    def failing_api(path, body):
        assert path == "/v1/embeddings"
        attempted.append(body["model"])
        raise RuntimeError("private-URL-or-secret-should-never-appear")

    scope["http_json"] = failing_api
    with pytest.raises(RuntimeError, match="Benchmark stopped"), redirect_stdout(io.StringIO()):
        for phase in ("concurrent-run", "mixed-batch-run", "report"):
            exec(compile(code[phase], phase, "exec"), scope)
    assert 1 <= len(attempted) <= scope["CONCURRENT_WORKERS"]
    assert set(attempted) == {scope["MODELS"][0]}
    report = json.loads(output.read_text())
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"] == {"phase": "concurrent", "error_type": "RuntimeError"}
    assert report["summary"] == {} and report["retries"] == 0
    assert len(report["concurrent"]["trials"]) == len(attempted)
    assert all(t["status"] == "failed" for t in report["concurrent"]["trials"])
    assert report["mixed_model_batches"] == []
    assert old_marker not in output.read_text() and "private-URL" not in output.read_text()


@pytest.mark.parametrize("stop_at_wait", [1, 2, "submit"])
def test_concurrent_interrupt_preserves_dispatched_attempts_and_blocks_retry(
    monkeypatch, tmp_path, stop_at_wait,
):
    """Stop before collection must preserve completed work and its pre-dispatch journal."""
    import concurrent.futures

    code, scope, output = failure_scope(monkeypatch, tmp_path)
    attempted, journal_observations = [], []
    first_model = scope["MODELS"][0]

    def offline_api(path, body):
        assert path == "/v1/embeddings"
        attempted.append(body["request_id"])
        journal = json.loads(output.read_text())["concurrent"]["trials"]
        journal_observations.append(any(t["status"] == "in_progress" for t in journal))
        return {
            "model": body["model"], "backend": "modal",
            "request_id": body["request_id"], "dimension": 1024,
            "revision": "fixture-revision", "input_sha256": scope["hashes"]["text"],
            "vector": [1.0] + [0.0] * 1023,
            "embedding_metadata": embedding_metadata(
                body["model"], "fixture-revision", 1024
            ).model_dump(),
            "elapsed_ms": 3.0,
            "timing": {
                "model_state": "warm", "gpu_worker_first_request": False,
                "gpu_worker_id": "a" * 32, "batch_size": 1,
                "gpu_worker_ms": 2.0, "model_load_ms": 0.0,
                "gpu_encode_ms": 1.0, "gpu_dispatch_overhead_ms": 0.1,
            },
        }

    original_wait, waits = concurrent.futures.wait, 0

    def stop_before_collection(futures, **_kwargs):
        nonlocal waits
        waits += 1
        # Deterministically finish the real executor's current requests first.
        completed = original_wait(futures)
        if waits == stop_at_wait:
            raise KeyboardInterrupt("private interruption detail")
        return completed

    scope["http_json"] = offline_api
    scope["gpu_readiness"][first_model] = {
        "worker_id": "a" * 32, "observed_at": scope["time"].perf_counter(),
        "phase": "concurrent_setup", "round": 0,
    }
    # Preserve the same worker ID after the mocked setup.
    original_prepare = scope["async_prepare_gpu"]
    def prepared(model, modality, phase, round_number):
        row = original_prepare(model, modality, phase, round_number)
        row["gpu_worker_id"] = "a" * 32
        scope["gpu_readiness"][model]["worker_id"] = "a" * 32
        return row
    scope["async_prepare_gpu"] = prepared
    monkeypatch.setattr(concurrent.futures, "wait", stop_before_collection)
    if stop_at_wait == "submit":
        original_submit = concurrent.futures.ThreadPoolExecutor.submit

        def stop_after_submit(pool, *args, **kwargs):
            future = original_submit(pool, *args, **kwargs)
            future.result()
            # The POST finished, but the caller has not received its Future yet.
            raise KeyboardInterrupt("private interruption detail")

        monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", stop_after_submit)
    with pytest.raises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
        exec(compile(code["concurrent-run"], "stop-concurrent", "exec"), scope)
    report = json.loads(output.read_text())
    trials = report["concurrent"]["trials"]
    assert report["status"] == "INCOMPLETE_INTERRUPTED"
    assert report["summary"] == {} and report["concurrent"]["summary"] == []
    expected_attempts = 1 if stop_at_wait == "submit" else 2 * stop_at_wait
    assert len(trials) == len(attempted) == expected_attempts
    assert all(t["status"] == "completed" and t["model"] == first_model for t in trials)
    assert len({t["index"] for t in trials}) == len(trials)
    assert all(journal_observations)
    assert report["mixed_model_batches"] == []
    assert "private interruption detail" not in output.read_text()
    assert "vector" not in trials[0] and "request_id" not in trials[0]
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        exec(compile(code["concurrent-run"], "reject-concurrent-retry", "exec"), scope)
    assert len(attempted) == expected_attempts
    assert json.loads(output.read_text()) == report


def test_first_mixed_batch_failure_stops_remaining_batches(monkeypatch, tmp_path):
    code, scope, output = failure_scope(monkeypatch, tmp_path)
    scope["RUN_CONCURRENT"] = False
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    scope["run_requested_phases"] = {"mixed_batch"}
    counts = []

    def failing_api(path, body):
        assert path == "/v1/embeddings/batches"
        counts.append(len(body["items"]))
        raise ConnectionError("redacted-api-host")

    scope["http_json"] = failing_api
    with redirect_stdout(io.StringIO()):
        exec(code["concurrent-run"], scope)  # Defines uuid4, without sending requests.
        with pytest.raises(RuntimeError, match="Benchmark stopped"):
            exec(code["mixed-batch-run"], scope)
    assert counts == [2]  # No later batches and no retry.
    report = json.loads(output.read_text())
    assert report["failure"]["phase"] == "mixed_batch"
    assert report["status"] == "INCOMPLETE_FAILED" and report["summary"] == {}
    assert len(report["mixed_model_batches"]) == 1
    assert report["mixed_model_batches"][0]["error_type"] == "ConnectionError"
    assert "redacted-api-host" not in output.read_text()


@pytest.mark.parametrize("worker_case", ["missing", "shared"])
def test_alternate_routing_failure_preserves_prior_attempts(monkeypatch, tmp_path,
                                                             worker_case):
    code, scope, output = failure_scope(monkeypatch, tmp_path)
    scope["RUN_ALTERNATE"] = True
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    scope["run_requested_phases"].add("alternate")
    sequence = []

    def fake_sample(model, modality, phase, round_number):
        worker = None if worker_case == "missing" else "b" * 32
        trial = {
            "phase": phase, "model": model, "modality": modality, "round": round_number,
            "gpu_worker_id": worker, "status": "completed", "confirmed": False,
        }
        scope["rows"].append(trial)
        sequence.append(model)
        return trial

    scope["sample"] = fake_sample
    scope["async_prepare_gpu"] = fake_sample
    with redirect_stdout(io.StringIO()):
        with pytest.raises(RuntimeError, match="Benchmark stopped"):
            exec(code["alternate-run"], scope)
    expected = 1 if worker_case == "missing" else 2
    assert len(sequence) == expected
    report = json.loads(output.read_text())
    assert len(report["trials"]) == expected
    assert report["trials"][-1]["status"] == "failed"
    assert report["trials"][-1]["confirmed"] is False
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["error_type"] == (
        "MissingGPUWorkerID" if worker_case == "missing" else "SharedGPUWorkerID"
    )
    assert report["summary"] == {} and report["mixed_model_batches"] == []


def test_invalid_credentials_preserve_existing_benchmark_evidence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "http://invalid.example")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "example-" + "q" * 40)
    output = tmp_path / "multimodal_embeddings_performance_summary.json"
    output.write_text('{"status":"LIVE_BENCHMARK_MEASURED","marker":"stale"}')
    code = {c["id"]: "".join(c["source"])
            for c in notebook()["cells"] if c["cell_type"] == "code"}
    config = code["config"].replace(
        'RUN_BENCHMARK = False #@param', 'RUN_BENCHMARK = True #@param'
    )
    scope = {}
    exec(config, scope)
    with pytest.raises(ValueError, match="HTTPS base URL"):
        exec(code["client"], scope)
    # Input validation happens before IN_PROGRESS, so published evidence survives.
    assert json.loads(output.read_text()) == {
        "status": "LIVE_BENCHMARK_MEASURED", "marker": "stale"
    }
    assert "invalid.example" not in output.read_text()


def test_verification_markdown_and_curl_prerequisites_remain_readable():
    root = BOOK.parents[1]
    if not (root / "README.md").is_file():
        pytest.skip("Published README curl instructions are added in the final commit")
    for suffix in (".md", ".vi.md"):
        verification = (root / "docs" / ("VERIFICATION" + suffix)).read_text()
        readme = (root / ("README" + suffix)).read_text()
        assert chr(92) + "n" + chr(92) + "n" not in verification
        assert "curl" in readme
        assert ("For each deployment session, record GPU or hosted-UI observations" in verification
                or "Với mỗi phiên sử dụng deployment, ghi kết quả quan sát GPU" in verification)

def test_offline_benchmark_evidence_audit_reports_source_mismatches():
    """Archived source hashes must not silently become deployment attestation."""
    import subprocess
    import sys
    root = BOOK.parents[1]
    script = root / "scripts" / "audit_benchmark_evidence.py"
    result = subprocess.run([sys.executable, str(script)], cwd=root,
                            capture_output=True, text=True, check=True)
    audit = json.loads(result.stdout)
    assert audit["network_or_gpu_requests"] == 0
    assert len(audit["results"]) == 3
    assert all(not item["source_snapshot_verified"] for item in audit["results"])
    overload = next(x for x in audit["results"] if "overload.json" in x["report"])
    assert any(x["status"] == "differs_from_checkout"
               for x in overload["source_digest_comparisons"])
    assert all(x["recorded_at_utc"].endswith("+00:00") for x in audit["results"])


def test_offline_benchmark_evidence_strict_mode_rejects_mismatched_checkout():
    import subprocess
    import sys
    root = BOOK.parents[1]
    script = root / "scripts" / "audit_benchmark_evidence.py"
    result = subprocess.run([sys.executable, str(script), "--strict-checkout"],
                            cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 1
