"""Offline safety and structural contract for general-purpose performance notebook."""
import io
import json
import re
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from test_performance_notebook import notebook

from multimodal_embeddings_ai_inference.embedding_config import embedding_metadata

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "multimodal_embeddings_benchmark.ipynb"
)




@pytest.fixture(autouse=True)
def isolate_benchmark_test_artifacts(monkeypatch, tmp_path):
    """Every notebook test writes only inside its own temporary test directory.

    The notebook prefers /kaggle/working when present; suppress that branch in
    offline tests, including when tests themselves run from a Kaggle runtime.
    """
    original_is_dir = Path.is_dir

    def safe_is_dir(path):
        if str(path) == "/kaggle/working":
            return False
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", safe_is_dir)
    monkeypatch.chdir(tmp_path)


def test_rerun_failure_report_drops_previous_optional_trials(monkeypatch, tmp_path):
    """F01 regression: a second run fails at preflight in the same kernel."""
    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.modal.run")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-test-token-123456789")
    monkeypatch.delenv("BENCHMARK_IMAGE_PATH", raising=False)
    code = {
        c["id"]: "".join(c["source"])
        for c in notebook()["cells"] if c["cell_type"] == "code"
    }
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), scope)
    scope.update(RUN_BENCHMARK=True, RUN_COLD=False)

    class FakeOpener:
        def __init__(self, models):
            self.models = models

        def open(self, request, timeout=None):
            name = request.full_url.rsplit("/", 1)[-1]
            if name == "health":
                body = {"backend": "modal"}
            elif name == "models":
                body = {"models": [{"id": model} for model in self.models]}
            else:
                raise AssertionError("Unexpected request " + name)
            return io.BytesIO(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(
        urllib.request, "build_opener",
        lambda *args: FakeOpener(scope["MODELS"]),
    )
    with redirect_stdout(io.StringIO()):
        exec(compile(code["client"], "client-first", "exec"), scope)
        exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
    scope["rows"].append({"status": "PREVIOUS_RUN"})
    scope["concurrent_trials"].append({"status": "PREVIOUS_RUN"})
    scope["concurrent_summary"].append({"status": "PREVIOUS_RUN"})
    scope["batch_trials"].append({"status": "PREVIOUS_RUN"})

    monkeypatch.setattr(
        urllib.request, "build_opener",
        lambda *args: FakeOpener(scope["MODELS"][:1]),
    )
    with pytest.raises(RuntimeError, match="Both embedding models"):
        exec(compile(code["client"], "client-second", "exec"), scope)

    report = json.loads(
        (tmp_path / "multimodal_embeddings_performance_summary.json").read_text()
    )
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["phase"] == "preflight"
    assert report["trials"] == []
    assert report["concurrent"] == {"summary": [], "trials": []}
    assert report["mixed_model_batches"] == []
    assert "PREVIOUS_RUN" not in json.dumps(report)


def test_benchmark_template_fingerprint_and_failed_run_provenance(monkeypatch, tmp_path):
    """No server SHA inferred from client or checkpoint; failures keep safe config."""
    import hashlib

    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BENCHMARK_CLIENT_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("BENCHMARK_SERVER_SOURCE_SHA", "b" * 40)
    monkeypatch.delenv("BENCHMARK_IMAGE_PATH", raising=False)
    nb = notebook()
    code = {
        cell["id"]: "".join(cell["source"])
        for cell in nb["cells"] if cell["cell_type"] == "code"
    }

    canonical = []
    # Fingerprint the published code cells in their original order.
    published = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in published["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            src = re.sub(
                r'NOTEBOOK_TEMPLATE_SHA256 = "[^"]+"',
                'NOTEBOOK_TEMPLATE_SHA256 = "EXCLUDED"', src
            )
            canonical.append(src)
    expected_hash = hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()
    scope = {}
    exec(compile(code["config"], "config", "exec"), scope)
    assert scope["NOTEBOOK_TEMPLATE_SHA256"] == expected_hash
    exec(compile(code["client"], "client", "exec"), scope)
    exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
    provenance = scope["provenance"]
    assert provenance["notebook_template"]["code_sha256"] == expected_hash
    assert provenance["client_source_commit"] == {
        "value": "a" * 40, "status": "operator_provided"
    }
    assert provenance["server_source_commit"] == {
        "value": "b" * 40, "status": "operator_provided"
    }
    scope["save_failed_report"]("preflight", "MockFailure")
    report_file = tmp_path / "multimodal_embeddings_performance_summary.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["provenance"] == provenance
    assert report["config"]["warmup_requests_per_modality_per_model_per_block"] == 1
    assert report["config"]["warmup_requests_per_modality_per_model"] == 2
    assert report["config"]["run_cold"] is False
    assert report["config"]["run_warm"] is True
    assert report["config"]["cold_attempts_per_model"] == 0
    assert "input_text" not in report["config"]
    assert "API_TOKEN" not in report_file.read_text()
    assert '"provenance": provenance' in code["report"]
    assert '"config": benchmark_config' in code["report"]

    monkeypatch.delenv("BENCHMARK_SERVER_SOURCE_SHA")
    absent = {}
    exec(compile(code["config"], "config", "exec"), absent)
    exec(compile(code["client"], "client", "exec"), absent)
    exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), absent)
    assert absent["provenance"]["server_source_commit"] == {
        "value": None, "status": "unavailable"
    }
    monkeypatch.setenv("BENCHMARK_SERVER_SOURCE_SHA", "not-a-commit")
    invalid = {}
    exec(compile(code["config"], "config", "exec"), invalid)
    with pytest.raises(ValueError, match="Git SHA"):
        exec(compile(code["client"], "client", "exec"), invalid)


def test_invalid_benchmark_token_never_enters_header_or_report(monkeypatch, tmp_path):
    """Invalid token and transport exceptions must never appear in JSON or traceback."""
    import traceback

    import pytest

    monkeypatch.chdir(tmp_path)
    secret = "FAKE_BENCHMARK_TOKEN_123" + chr(10) + "PRIVATE_SECOND_LINE"
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.modal.run")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", secret)
    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    scope = {}
    exec(compile(code["config"], "config", "exec"), scope)
    scope["RUN_BENCHMARK"] = True
    with pytest.raises(ValueError) as caught:
        exec(compile(code["client"], "client", "exec"), scope)
    trace = "".join(traceback.format_exception(caught.value))
    assert secret not in trace and "FAKE_BENCHMARK_TOKEN" not in trace
    report_path = tmp_path / "multimodal_embeddings_performance_summary.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"]["phase"] == "preflight"
    assert secret not in report_path.read_text()

    # Reusing the helper with a mutated token must fail before a header is built.
    with pytest.raises(ValueError) as caught:
        scope["http_json"]("/v1/health")
    assert "FAKE_BENCHMARK_TOKEN" not in str(caught.value)
    safe_token = "safe-token-123456789"
    scope["token"] = safe_token

    class UnsafeTransport:
        def open(self, request, timeout):
            raise ValueError("Bad header Bearer " + safe_token)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: UnsafeTransport())
    with pytest.raises(RuntimeError) as caught:
        scope["http_json"]("/v1/health")
    assert safe_token not in "".join(traceback.format_exception(caught.value))


def test_warmup_covers_both_modalities_and_report_counts(monkeypatch, tmp_path):
    """Two blocks, two models, two input types; measured rows stay balanced."""
    monkeypatch.chdir(tmp_path)
    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    scope = {}
    exec(compile(code["config"], "config", "exec"), scope)
    scope.update(RUN_BENCHMARK=True, RUN_WARM=True,
                 WARMUP_REQUESTS=1, WARM_REPETITIONS=4)
    calls = []

    def fake_sample(model, modality, phase, block):
        calls.append((model, modality, phase, block))
        return {"model_state": "warm", "gpu_worker_first_request": False}

    scope["sample"] = fake_sample
    scope["assert_run_active"] = lambda phase: None  # Isolate warm scheduling.
    scope["run_completed_phases"] = set()
    with redirect_stdout(io.StringIO()):
        exec(compile(code["warm-run"], "warm-run", "exec"), scope)
    from collections import Counter

    counts = Counter((model, modality, phase) for model, modality, phase, _ in calls)
    for model in scope["MODELS"]:
        assert counts[(model, "text", "warmup")] == 2
        assert counts[(model, "image", "warmup")] == 2
        assert counts[(model, "text", "warm")] == 4
        assert counts[(model, "image", "warm")] == 4
    assert len(calls) == 24
    # No measured request from a block precedes its two modality warm-ups.
    for model in scope["MODELS"]:
        for block in (1, 2):
            block_calls = [phase for m, _, phase, b in calls if m == model and b == block]
            assert block_calls == ["warmup", "warmup", "warm", "warm", "warm", "warm"]


def test_benchmark_markdown_warmup_matches_default_request_schedule():
    """Guard user-facing 24-request cost and warm-up descriptions in both languages."""
    book = notebook()
    docs = {cell["id"]: "".join(cell["source"]) for cell in book["cells"]}
    rules = docs["rules"]
    warm = docs["warm-intro"]
    assert rules.count("24 POST") == 2
    assert "20 POST" not in rules
    assert "one text and one image unmeasured warm-up" in warm
    assert "một request text và một request ảnh" in warm
    assert "Unmeasured requests per modality, model and block" in rules
    assert "Một request không tính cho mỗi modality, model và khối" in rules

    code = {cell["id"]: "".join(cell["source"]) for cell in book["cells"]
            if cell["cell_type"] == "code"}
    scope = {}
    exec(compile(code["config"], "config", "exec"), scope)
    assert scope["WARMUP_REQUESTS"] == 1
    assert scope["WARM_REPETITIONS"] == 4

    sent = []
    def fake_sample(model, modality, phase, block):
        sent.append((model, modality, phase, block))
        return {"model_state": "warm", "gpu_worker_first_request": False}

    scope.update(RUN_BENCHMARK=True, sample=fake_sample, run_completed_phases=set())
    scope["assert_run_active"] = lambda phase: None  # Isolate warm scheduling.
    with redirect_stdout(io.StringIO()):
        exec(compile(code["warm-run"], "warm-run", "exec"), scope)
    assert len(sent) == 24
    assert sum(x[2] == "warmup" for x in sent) == 8
    assert sum(x[2] == "warm" for x in sent) == 16

    root = NOTEBOOK.parents[1]
    # Check the published README examples when the README is available.
    if not (root / "README.md").is_file():
        import pytest
        pytest.skip("README usage examples unavailable in this checkout")
    for path in ("README.md", "README.vi.md",
                 "benchmarks/README.md", "benchmarks/README.vi.md"):
        content = (root / path).read_text(encoding="utf-8")
        assert "24" in content
        assert "20 POST request" not in content


def test_rerun_report_preserves_preflight_failure(monkeypatch, tmp_path):
    """Running Report alone must not erase a failed preflight JSON artifact."""
    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "http://invalid.example.test")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-token-123456789")
    cells = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["config"], "config", "exec"), scope)
    scope["RUN_BENCHMARK"] = True
    with pytest.raises(ValueError, match="HTTPS base URL"):
        exec(compile(cells["client"], "client", "exec"), scope)
        exec(compile(cells["client-request-helpers"], "client-request-helpers", "exec"), scope)
    path = tmp_path / "multimodal_embeddings_performance_summary.json"
    before = path.read_bytes()
    failed = json.loads(before)
    assert failed["status"] == "INCOMPLETE_FAILED"
    assert failed["failure"] == {"phase": "preflight", "error_type": "ValueError"}
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        exec(compile(cells["report"], "report-rerun", "exec"), scope)
    assert path.read_bytes() == before
    assert scope["run_failure"] == failed["failure"]


def test_report_rejects_skipped_enabled_phases(monkeypatch, tmp_path):
    """Even a successful preflight cannot justify a report if phases were skipped."""
    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.modal.run")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-token-123456789")
    cells = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
             if c["cell_type"] == "code"}

    class PreflightOnly:
        def __init__(self):
            self.paths = []

        def open(self, request, timeout=None):
            path = request.full_url.rsplit("/", 1)[-1]
            self.paths.append(path)
            assert request.get_method() == "GET"
            if path == "health":
                result = {"backend": "modal"}
            else:
                assert path == "models"
                result = {"models": [{"id": m} for m in ("qwen3-vl-embedding-8b",
                                                          "wemm-embedding-9b")]}
            return io.BytesIO(json.dumps(result).encode("utf-8"))

    for missing_phase in ("warm", "concurrent"):
        fake = PreflightOnly()
        monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers, current=fake: current)
        scope = {}
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["config"], "config", "exec"), scope)
        scope.update(RUN_BENCHMARK=True, RUN_WARM=missing_phase == "warm",
                     RUN_CONCURRENT=missing_phase == "concurrent")
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["client"], "client", "exec"), scope)
            exec(compile(cells["client-request-helpers"], "client-request-helpers", "exec"), scope)
        assert scope["run_completed_phases"] == {"preflight"}
        assert fake.paths == ["health", "models"]
        with pytest.raises(RuntimeError, match="MissingPhases_" + missing_phase):
            exec(compile(cells["report"], "report-skipped", "exec"), scope)
        path = tmp_path / "multimodal_embeddings_performance_summary.json"
        before = path.read_bytes()
        report = json.loads(before)
        assert report["status"] == "INCOMPLETE_FAILED"
        assert report["failure"]["phase"] == "report"
        assert missing_phase in report["failure"]["error_type"]
        with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
            exec(compile(cells["report"], "report-again", "exec"), scope)
        assert path.read_bytes() == before


def test_default_benchmark_counts_real_http_dispatches_offline(monkeypatch, tmp_path):
    """Count GET preflight and all actual POST calls through a fake HTTP opener."""
    import base64
    import hashlib
    from collections import Counter
    from urllib.parse import urlsplit

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.modal.run")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-token-123456789")
    cells = {c["id"]: "".join(c["source"])
             for c in json.loads(NOTEBOOK.read_text())["cells"]
             if c["cell_type"] == "code"}
    calls = []

    class FakeOpener:
        def open(self, request, timeout=None):
            path = urlsplit(request.full_url).path
            body = json.loads(request.data) if request.data is not None else None
            calls.append((request.get_method(), path, body))
            if path == "/v1/health":
                response = {"backend": "modal"}
            elif path == "/v1/models":
                response = {"models": [{"id": m} for m in
                            ("qwen3-vl-embedding-8b", "wemm-embedding-9b")]}
            else:
                assert path == "/v1/embeddings", path
                model = body["model"]
                raw = (body["text"].encode("utf-8") if "text" in body
                       else base64.b64decode(body["image_base64"]))
                response = {
                    "backend": "modal", "model": model, "dimension": 1024,
                    "revision": "offline-revision", "request_id": body["request_id"],
                    "input_sha256": hashlib.sha256(raw).hexdigest(),
                    "vector": [1.0] + [0.0] * 1023,
                    "embedding_metadata": embedding_metadata(
                        model, "offline-revision", 1024, body.get("instruction")
                    ).model_dump(),
                    "elapsed_ms": 2.0,
                    "timing": {
                        "model_state": "warm", "gpu_worker_first_request": False,
                        "gpu_worker_id": ("a" if model.startswith("qwen") else "b") * 32,
                        "batch_size": 1, "gpu_worker_ms": 2.0,
                        "model_load_ms": 0.0, "gpu_encode_ms": 1.0,
                        "gpu_dispatch_overhead_ms": 0.1,
                    },
                }
            return io.BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: FakeOpener())
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["config"], "config", "exec"), scope)
    scope["RUN_BENCHMARK"] = True
    with redirect_stdout(io.StringIO()):
        for cell_id in ("client", "client-request-helpers", "cold-run", "warm-run", "alternate-run",
                        "concurrent-run", "mixed-batch-run", "report"):
            exec(compile(cells[cell_id], cell_id, "exec"), scope)
    get = [(method, path) for method, path, _ in calls if method == "GET"]
    post = [(path, body) for method, path, body in calls if method == "POST"]
    assert get == [("GET", "/v1/health"), ("GET", "/v1/models")]
    assert len(post) == 24 and all(path == "/v1/embeddings" for path, _ in post)
    assert len(scope["rows"]) == 24
    assert Counter(row["phase"] for row in scope["rows"]) == {"warmup": 8, "warm": 16}
    assert scope["run_completed_phases"] == {"preflight", "warm"}
    assert json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())[
        "status"
    ] == "LIVE_BENCHMARK_MEASURED"
    report = json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())
    context = report["sample_size_context"]
    assert context["default_warm_post_requests"] == 24
    assert context["default_warmup_requests"] == 8
    assert context["default_measured_requests"] == 16
    assert context["default_warm_measurements_per_model_modality"] == 4
    assert "production" in context["note_en"]
    assert "p95" in context["note_vi"]
    assert "Latency n" in scope["lines"][0]
    assert "at most 4" in scope["sample_note"]
    assert len(report["summary"]) == 4
    assert all(v["client"]["n"] == 4 for v in report["summary"].values())


def test_benchmark_intro_shows_bilingual_recovery_table():
    intro = "".join(notebook()["cells"][0]["source"])
    assert "| Situation | Start a fresh run in this order |" in intro
    assert "| Tình huống | Bắt đầu lượt mới theo thứ tự |" in intro
    assert intro.count("**Benchmark configuration → API client/preflight") == 2
    assert intro.count("**Cấu hình benchmark → API client/preflight") == 2
    assert "Do not resume a partially completed phase" in intro
    assert "Không tiếp tục phase đang dở" in intro


@pytest.mark.parametrize("interrupt_index", [0, 1])
def test_interrupted_mixed_batch_persists_attempt_and_never_retries(interrupt_index):
    """A dispatched POST is still represented when its reply is unknown."""
    scope = {}
    code = {cell["id"]: "".join(cell["source"]) for cell in notebook()["cells"]
            if cell["cell_type"] == "code"}
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), scope)
        exec(compile(code["client"], "client", "exec"), scope)
        exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
    scope.update(RUN_BENCHMARK=True, RUN_WARM=False, RUN_MIXED_BATCH=True,
                 MIXED_BATCH_REPETITIONS=3)
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    scope["run_completed_phases"].add("preflight")
    output = Path("multimodal_embeddings_performance_summary.json")
    calls = []

    def fake_http(path, body=None):
        assert path == "/v1/embeddings/batches"
        saved = json.loads(output.read_text())
        assert saved["status"] == "IN_PROGRESS"
        assert len(saved["mixed_model_batches"]) == len(calls) + 1
        pending = saved["mixed_model_batches"][-1]
        assert pending["index"] == len(calls) + 1
        assert pending["status"] == "in_progress" and pending["confirmed"] is False
        assert pending["started_at_utc"]
        assert "items" not in pending
        calls.append(path)
        if len(calls) == interrupt_index + 1:
            raise KeyboardInterrupt("offline interruption")

        results = []
        for item in body["items"]:
            model = item["model"]
            meta = embedding_metadata(
                model, "offline-revision", 1024, item.get("instruction")
            ).model_dump()
            results.append({
                "model": model, "backend": "modal", "request_id": item["request_id"],
                "revision": "offline-revision", "dimension": 1024,
                "input_sha256": scope["hashes"]["text"],
                "vector": [1.0] + [0.0] * 1023, "embedding_metadata": meta,
                "elapsed_ms": 3.0,
                "timing": {"model_state": "warm", "gpu_worker_first_request": False,
                           "gpu_worker_id": ("a" if model.startswith("qwen") else "b") * 32,
                           "batch_size": 1, "gpu_worker_ms": 2.0,
                           "model_load_ms": 0.0, "gpu_encode_ms": 1.0,
                           "gpu_dispatch_overhead_ms": 0.1},
            })
        return {"count": 2, "items": results}

    scope["http_json"] = fake_http
    with pytest.raises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
        exec(compile(code["mixed-batch-run"], "interrupt-mixed-batch", "exec"), scope)
    report = json.loads(output.read_text())
    assert report["status"] == "INCOMPLETE_INTERRUPTED"
    assert report["failure"] == {
        "phase": "mixed_batch", "error_type": "KeyboardInterrupt"
    }
    assert report["summary"] == {}
    attempts = report["mixed_model_batches"]
    assert len(attempts) == len(calls) == interrupt_index + 1
    assert [item["index"] for item in attempts] == list(range(1, len(calls) + 1))
    assert all(item["started_at_utc"] for item in attempts)
    assert attempts[-1]["status"] == "interrupted"
    assert attempts[-1]["confirmed"] is False
    assert attempts[-1]["error_type"] == "KeyboardInterrupt"
    assert attempts[-1]["finished_at_utc"]
    if interrupt_index:
        assert attempts[0]["status"] == "completed"
        assert attempts[0]["confirmed"] is True
        assert attempts[0]["gpu_instrumentation_verified"] is True
    assert "offline interruption" not in output.read_text()
    before = output.read_bytes()
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        exec(compile(code["mixed-batch-run"], "replayed-mixed-batch", "exec"), scope)
    assert len(calls) == interrupt_index + 1
    assert output.read_bytes() == before


def test_failed_mixed_batch_retains_exactly_one_redacted_attempt():
    """A failed POST does not create duplicate attempts or schedule a retry."""
    cells = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["config"], "config", "exec"), scope)
        exec(compile(cells["client"], "client", "exec"), scope)
        exec(compile(cells["client-request-helpers"], "client-request-helpers", "exec"), scope)
    scope.update(RUN_BENCHMARK=True, RUN_WARM=False, RUN_MIXED_BATCH=True,
                 MIXED_BATCH_REPETITIONS=3)
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    scope["run_completed_phases"].add("preflight")
    calls = []

    def failing_post(path, body):
        assert path == "/v1/embeddings/batches"
        calls.append(path)
        raise RuntimeError("untrusted remote error details")

    scope["http_json"] = failing_post
    with pytest.raises(RuntimeError, match="partial report saved"):
        exec(compile(cells["mixed-batch-run"], "failed-mixed-batch", "exec"), scope)
    report_path = Path("multimodal_embeddings_performance_summary.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["failure"] == {"phase": "mixed_batch", "error_type": "RuntimeError"}
    trials = report["mixed_model_batches"]
    assert len(trials) == len(calls) == 1
    assert trials[0]["index"] == 1 and trials[0]["status"] == "failed"
    assert trials[0]["confirmed"] is False
    assert trials[0]["started_at_utc"] and trials[0]["finished_at_utc"]
    assert "untrusted remote error details" not in report_path.read_text()
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        exec(compile(cells["mixed-batch-run"], "retry-mixed-batch", "exec"), scope)
    assert len(calls) == 1



@pytest.mark.parametrize("url", [
    "https://user:PRIVATE_URL_PASSWORD@private.example：443",
    "https://private customer.example",
    "https://private.example:not-a-port",
    "https://private.example:99999",
    "https://private.\nexample",
])
def test_benchmark_malformed_hidden_url_is_redacted_before_network(monkeypatch, tmp_path, url):
    import traceback

    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", url)
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-url-token-123456")
    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    scope = {}
    exec(compile(code["config"], "config", "exec"), scope)
    scope["RUN_BENCHMARK"] = True

    class ForbiddenTransport:
        def open(self, *args, **kwargs):
            pytest.fail("An invalid endpoint must not reach the network")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: ForbiddenTransport())
    with pytest.raises(ValueError) as caught:
        exec(compile(code["client"], "url-input", "exec"), scope)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "PRIVATE_URL_PASSWORD" not in rendered and "private.example" not in rendered
    report = tmp_path / "multimodal_embeddings_performance_summary.json"
    assert json.loads(report.read_text())["status"] == "INCOMPLETE_FAILED"
    assert "private.example" not in report.read_text()


def test_benchmark_sanitizes_invalid_url_transport_without_retry(monkeypatch):
    import http.client
    import traceback

    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    scope = {}
    for key in ("config", "client"):
        exec(compile(code[key], key, "exec"), scope)
    token = "offline-url-token-123456"
    scope.update(token=token, base_url="https://example.invalid")
    calls = []

    class InvalidTransport:
        def open(self, request, timeout):
            calls.append(request.get_method())
            raise http.client.InvalidURL("private-host " + token)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: InvalidTransport())
    with pytest.raises(RuntimeError) as caught:
        scope["http_json"]("/v1/embeddings", {"text": "public fixture"})
    rendered = "".join(traceback.format_exception(caught.value))
    assert token not in rendered and "private-host" not in rendered
    assert calls == ["POST"]


def test_sequential_attempt_is_persisted_before_dispatch(monkeypatch, tmp_path):
    from test_benchmark_provenance import run_warm_fixture

    _, scope, sent, _start, run_warm, _report = run_warm_fixture(monkeypatch, tmp_path)
    checked = scope["_sample_checked"]
    output = tmp_path / "multimodal_embeddings_performance_summary.json"

    def inspect_before_http(*args):
        journal = json.loads(output.read_text())
        assert len(journal["trials"]) == len(sent) + 1
        assert journal["trials"][-1]["status"] == "in_progress"
        return checked(*args)

    scope["_sample_checked"] = inspect_before_http
    run_warm()
    assert len(sent) == 24
    journal = json.loads(output.read_text())
    assert len(journal["trials"]) == 24
    assert all(trial["status"] == "completed" for trial in journal["trials"])


def test_stop_after_response_preserves_the_dispatched_attempt(monkeypatch, tmp_path):
    from test_benchmark_provenance import run_warm_fixture

    _, scope, sent, _start, run_warm, _report = run_warm_fixture(monkeypatch, tmp_path)
    clock = scope["utc_now"]
    interrupted = False

    def stop_after_http():
        nonlocal interrupted
        if sent and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return clock()

    scope["utc_now"] = stop_after_http
    with pytest.raises(KeyboardInterrupt):
        run_warm()
    assert len(sent) == 1
    report = json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())
    assert report["status"] == "INCOMPLETE_INTERRUPTED"
    assert len(report["trials"]) == 1
    assert report["trials"][0]["status"] == "interrupted"
    assert report["trials"][0]["confirmed"] is False
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        run_warm()
    assert len(sent) == 1
