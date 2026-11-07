"""Verify optional-only benchmark provenance with fake API responses."""

import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from multimodal_embeddings_ai_inference.embedding_config import embedding_metadata

BOOK = Path(__file__).resolve().parents[1] / "notebooks/multimodal_embeddings_benchmark.ipynb"


@pytest.mark.parametrize("phase", ["concurrent-run", "mixed-batch-run"])
@pytest.mark.parametrize("drift", [False, True])
def test_optional_provenance(phase, drift, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    nb = json.loads(BOOK.read_text())
    code = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    g = {}
    with redirect_stdout(io.StringIO()):
        exec(code["config"], g)
        exec(code["client"], g)
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
    nb = json.loads(BOOK.read_text(encoding="utf-8"))
    code = {
        c["id"]: "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code"
    }
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), scope)
        exec(compile(code["client"], "client", "exec"), scope)
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
    book = json.loads(BOOK.read_text(encoding="utf-8"))
    code = {c["id"]: "".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), scope)
        exec(compile(code["client"], "client", "exec"), scope)
    scope.update(
        RUN_BENCHMARK=True, RUN_COLD=False, RUN_WARM=False, RUN_ALTERNATE=False,
        RUN_CONCURRENT=True, RUN_MIXED_BATCH=True, CONCURRENT_WORKERS=2,
        CONCURRENT_REQUESTS=8, MIXED_BATCH_REPETITIONS=3,
    )
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    scope["run_completed_phases"].add("preflight")
    scope["run_requested_phases"] = {"concurrent", "mixed_batch"}
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


def test_stale_report_replaced_during_preflight_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "http://invalid.example")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "example-" + "q" * 40)
    output = tmp_path / "multimodal_embeddings_performance_summary.json"
    output.write_text('{"status":"LIVE_BENCHMARK_MEASURED","marker":"stale"}')
    code = {c["id"]: "".join(c["source"])
            for c in json.loads(BOOK.read_text())["cells"] if c["cell_type"] == "code"}
    config = code["config"].replace(
        'RUN_BENCHMARK = False #@param', 'RUN_BENCHMARK = True #@param'
    )
    scope = {}
    exec(config, scope)
    with pytest.raises(ValueError, match="HTTPS base URL"):
        exec(code["client"], scope)
    report = json.loads(output.read_text())
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["phase"] == "preflight"
    assert report["summary"] == {}
    assert "stale" not in output.read_text()
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
        assert ("not independently verified" in verification
                or "chưa xác minh độc lập" in verification)

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
    book = json.loads(BOOK.read_text(encoding="utf-8"))
    code = {
        c["id"]: "".join(c["source"]) for c in book["cells"]
        if c["cell_type"] == "code"
    }
    scope = {}

    class OfflinePreflight:
        def open(self, request, timeout=None):
            assert request.get_method() == "GET"
            if request.full_url.endswith("/v1/health"):
                payload = {"backend": "modal"}
            else:
                assert request.full_url.endswith("/v1/models")
                payload = {
                    "models": [
                        {"id": "qwen3-vl-embedding-8b"},
                        {"id": "wemm-embedding-9b"},
                    ],
                }
            return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: OfflinePreflight())

    def start(configuration=None):
        with redirect_stdout(io.StringIO()):
            exec(compile(configuration or code["config"], "config", "exec"), scope)
            scope["RUN_BENCHMARK"] = True
            exec(compile(code["client"], "client", "exec"), scope)

    sent = []

    def fake_http(path, body):
        assert path == "/v1/embeddings"
        sent.append((body["model"], body["request_id"]))
        model = body["model"]
        return {
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
    assert attempts == 4 and len(sent) == 4
    assert scope["run_started_phases"] == {"warm"}
    assert scope["run_completed_phases"] == {"preflight"}
    assert scope["run_failure"] == {"phase": "warm", "error_type": "KeyboardInterrupt"}
    output = tmp_path / "multimodal_embeddings_performance_summary.json"
    report = json.loads(output.read_text())
    assert report["status"] == "INCOMPLETE_INTERRUPTED"
    assert report["summary"] == {}
    assert report["config"]["run_id"] == initial_run
    assert len(report["trials"]) == 4
    assert [row["status"] for row in report["trials"]] == [
        "completed", "completed", "completed", "interrupted"
    ]
    assert report["trials"][-1]["confirmed"] is False
    assert "unknown-response" not in output.read_text()

    # Re-executing warm must never dispatch the 24 scheduled requests.
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        run_warm()
    assert len(sent) == attempts == 4
    assert output.read_text() == json.dumps(report, indent=2, ensure_ascii=False) + "\n"

    # Only a fresh preflight resets the failure and starts a new, isolated run.
    scope["_sample_checked"] = original_checked
    start()
    run_warm()
    fresh = run_report()
    assert fresh["status"] == "LIVE_BENCHMARK_MEASURED"
    assert fresh["config"]["run_id"] != initial_run
    assert len(fresh["trials"]) == 24
    assert len(sent) == 4 + 24


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
    assert sent == []
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
        .replace("RUN_COLD = False", "RUN_COLD = True")
        .replace("RUN_WARM = True", "RUN_WARM = False")
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
        .replace("RUN_WARM = True", "RUN_WARM = False")
        .replace("RUN_MIXED_BATCH = False", "RUN_MIXED_BATCH = True")
        .replace("MIXED_BATCH_REPETITIONS = 3", "MIXED_BATCH_REPETITIONS = 1")
    )
    start(batch_config)
    seen = []

    def fake_batch(path, body):
        seen.append((path, len(body["items"])))
        raise KeyboardInterrupt

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
    nb = json.loads(BOOK.read_text())
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
