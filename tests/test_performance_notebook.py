"""Offline safety and structural contract for general-purpose performance notebook."""
import io
import json
import re
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import pytest

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


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_benchmark_structure_and_python_syntax():
    nb = notebook()
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert len(nb["cells"]) >= 10
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), cell["id"], "exec")
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for marker in ("RUN_BENCHMARK = False", "COLD_IDLE_WAIT_SECONDS = (",
                   "COLD_REPETITIONS = 5", "WARM_REPETITIONS = 4",
                   "ALTERNATE_REPETITIONS = 5", "gpu_worker_first_request", "RejectRedirect"):
        assert marker in src
    assert "face_verification" not in src and "face_benchmark" not in src


def test_disabled_benchmark_does_not_call_api_or_write_fake_results(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VISION_EMBEDDINGS_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_EMBEDDINGS_API_TOKEN", raising=False)
    monkeypatch.delenv("BENCHMARK_IMAGE_PATH", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Network call in disabled benchmark")
    monkeypatch.setattr(urllib.request, "build_opener",
                        lambda *args: type("NoNetwork", (), {"open": forbidden})())
    scope, stdout = {}, io.StringIO()
    with redirect_stdout(stdout):
        for cell in notebook()["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), cell["id"], "exec"), scope)
    assert scope["RUN_BENCHMARK"] is False
    assert scope["rows"] == [] and scope["sequence"] == 0
    assert not list(tmp_path.glob("*performance_summary.json"))
    assert "No live benchmark" in stdout.getvalue()


def test_custom_image_read_failures_do_not_reveal_paths(monkeypatch, tmp_path):
    """Both initial and provenance reads must sanitize missing and denied files."""
    import pytest

    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    image = tmp_path / "personal-images" / "restricted.png"

    def prepare():
        scope = {}
        exec(compile(code["config"], "config", "exec"), scope)
        exec(compile(code["client"], "client", "exec"), scope)
        return scope

    monkeypatch.setenv("BENCHMARK_IMAGE_PATH", str(image))
    with pytest.raises(ValueError, match="Cannot read the selected benchmark image") as error:
        prepare()
    assert str(image) not in str(error.value)
    assert "Không thể đọc ảnh benchmark" in str(error.value)

    image.parent.mkdir()
    image.write_bytes(b"irrelevant: read deliberately denied")
    original_read_bytes = Path.read_bytes

    def deny_read(path):
        if path == image:
            raise PermissionError("private path: " + str(image))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", deny_read)
    with pytest.raises(ValueError, match="Cannot read the selected benchmark image") as error:
        prepare()
    assert str(image) not in str(error.value)
    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)

    monkeypatch.delenv("BENCHMARK_IMAGE_PATH")
    default_scope = prepare()
    image.write_bytes(default_scope["sample_png"]())
    monkeypatch.setenv("BENCHMARK_IMAGE_PATH", str(image))
    scope = prepare()
    image.unlink()
    with pytest.raises(ValueError, match="Cannot read the selected benchmark image") as error:
        scope["current_run_configuration"]()
    assert str(image) not in str(error.value)


def test_vietnamese_instruction_mode_uses_api_default_not_cached_vectors():
    """Check localized guide details when the guide is available."""
    import pytest

    guide = NOTEBOOK.parents[1] / "docs" / "NOTEBOOK_GUIDE.vi.md"
    if not guide.is_file():
        pytest.skip("Localized notebook guide unavailable in this checkout")
    content = guide.read_text(encoding="utf-8")
    assert "Không gửi trường `instruction`; sử dụng mặc định của API" in content
    assert "giữ vector cũ" not in content


def test_identical_png_and_symmetric_protocol():
    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    cold = code["cold-run"]
    assert cold.index("time.sleep(COLD_IDLE_WAIT_SECONDS)") < cold.index("sample(")
    assert '"gpu_worker_first_request"' in cold and '"initial_load"' in cold
    warm = code["warm-run"]
    assert "reversed(MODELS)" in warm and "Random(20261107 + block)" in warm
    assert "WARM_REPETITIONS // 2" in warm
    assert '"warm"' in code["alternate-run"]
    scope = {}
    exec(compile(code["config"], "config", "exec"), scope)
    scope.update(IMAGE_PATH="", TEXT_SAMPLE="sample", RUN_BENCHMARK=False)
    exec(compile(code["client"], "client", "exec"), scope)
    png = scope["sample_png"]()
    assert png == scope["image_bytes"]
    assert png[:8] == bytes([137,80,78,71,13,10,26,10])
    assert int.from_bytes(png[16:20], "big") == 128

def test_cold_idle_is_calculated_from_editable_deployment_values():
    code = next("".join(cell["source"]) for cell in notebook()["cells"] if cell["id"] == "config")
    default = {}
    exec(compile(code, "benchmark-config", "exec"), default)
    assert default["COLD_IDLE_WAIT_SECONDS"] == 210
    assert default["COLD_TOTAL_IDLE_SECONDS"] == 0
    assert default["RUN_COLD"] is False
    assert default["RUN_WARM"] is True
    assert default["WARMUP_REQUESTS"] == 1
    assert default["WARM_REPETITIONS"] == 4
    expected_requests = 2 * len(default["MODELS"]) * (
        2 * default["WARMUP_REQUESTS"] + default["WARM_REPETITIONS"]
    )
    assert expected_requests == 24
    edited = code.replace(
        'GATEWAY_SCALEDOWN_SECONDS = 60 #@param',
        'GATEWAY_SCALEDOWN_SECONDS = 50 #@param',
    ).replace(
        'GPU_SCALEDOWN_SECONDS = 180 #@param',
        'GPU_SCALEDOWN_SECONDS = 120 #@param',
    )
    assert edited != code
    alternate = {}
    exec(compile(edited, "benchmark-config-alternative", "exec"), alternate)
    assert alternate["COLD_IDLE_WAIT_SECONDS"] == 150
    assert alternate["COLD_TOTAL_IDLE_SECONDS"] == 0
    enabled_cold = edited.replace('RUN_COLD = False #@param', 'RUN_COLD = True #@param')
    cold = {}
    exec(compile(enabled_cold, "benchmark-cold-opt-in", "exec"), cold)
    assert cold["COLD_TOTAL_IDLE_SECONDS"] == (
        len(cold["MODELS"]) * cold["COLD_REPETITIONS"] * 150
    )



def test_public_docs_match_inference_and_safe_notebook_defaults():
    """Guard against contradictory topology, stale test totals and dead entry points."""
    root = NOTEBOOK.parents[1]
    # Check the published README examples when the README is available.
    if not (root / "README.md").is_file():
        import pytest
        pytest.skip("README usage examples unavailable in this checkout")
    for suffix in (".md", ".vi.md"):
        changelog = (root / ("CHANGELOG" + suffix)).read_text(encoding="utf-8")
        verification = (root / "docs" / ("VERIFICATION" + suffix)).read_text(encoding="utf-8")
        license_notes = (root / ("LICENSE-NOTES" + suffix)).read_text(encoding="utf-8")
        readme = (root / ("README" + suffix)).read_text(encoding="utf-8")
        guide = (root / "docs" / ("NOTEBOOK_GUIDE" + suffix)).read_text(encoding="utf-8")
        assert "routinging" not in changelog
        assert "two-request" not in changelog and "hai request" not in changelog
        pytest_row = next(
            line for line in verification.splitlines()
            if line.startswith("| ") and "pytest" in line
        )
        assert "PASS" in pytest_row and "actions/workflows/ci.yml" in pytest_row
        assert "pytest, Ruff" in pytest_row
        assert "WebP" in license_notes
        assert "bash scripts/setup.sh" in readme
        assert "source .venv/bin/activate" in readme
        assert "multimodal_embeddings_benchmark.ipynb" in readme
        assert "colab.research.google.com" in readme
        assert "colab.research.google.com" in guide
        assert "RUN_BENCHMARK" in readme
        operations = (root / "docs" / ("OPERATIONS" + suffix)).read_text()
        assert "OPERATIONS" + suffix in readme
        assert "TOKEN_FILE=" in operations
        assert "set -a" in operations
        assert ("without an API URL or token" in readme
                or "không cần URL API hoặc token" in readme)
        assert "## Unreleased" not in changelog
        assert "## Chưa phát hành" not in changelog
        assert "## 1.0.0" in changelog
        assert "RUN_COLD" in (root / "benchmarks" / ("README" + suffix)).read_text(encoding="utf-8")


def test_hosted_notebook_document_links_work_without_repository_checkout():
    """Kaggle/Colab may import a standalone .ipynb with no sibling docs."""
    root = NOTEBOOK.parents[1]
    if not (root / "docs" / "NOTEBOOK_GUIDE.md").is_file():
        import pytest
        pytest.skip("Hosted notebook guide unavailable in this checkout")
    prefix = "https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/"
    for path in (NOTEBOOK, NOTEBOOK.parent / "multimodal_embeddings_api_demo.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        markdown = "\n".join("".join(c["source"]) for c in nb["cells"]
                             if c["cell_type"] == "markdown")
        links = re.findall(r"\]\(([^)]+)\)", markdown)
        assert links, path
        for url in links:
            assert url.startswith("https://"), (path, url)
            if url.startswith(prefix):
                assert (root / url[len(prefix):].split("#", 1)[0]).is_file(), url


def test_user_facing_full_model_names():
    import json
    import re
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "notebooks"
    path /= "multimodal_embeddings_benchmark.ipynb"
    data = json.loads(path.read_text())
    md = "\n".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "markdown")
    assert "Qwen3-VL-Embedding-8B" in md
    assert "Tencent WeMM-Embedding-9B" in md
    assert not re.search(r"(?<![\w/])(?:Qwen|WeMM)(?![\w/-])", md)


def test_both_notebooks_share_bilingual_markdown_format():
    import re
    for path in (
        NOTEBOOK,
        NOTEBOOK.parent / "multimodal_embeddings_api_demo.ipynb",
    ):
        data = json.loads(path.read_text())
        cells = [c for c in data["cells"] if c["cell_type"] == "markdown"]
        for index, cell in enumerate(cells):
            content = "".join(cell["source"])
            assert content.count("\n---\n") == 1, (path, cell["id"])
            en, vi = content.split("\n---\n")
            assert en.count("### English") == 1
            assert vi.count("### Tiếng Việt") == 1
            assert "<details>" not in content
            if index > 0:
                assert " / " in content.splitlines()[0]
            # The API Demo image gallery uses one shared bilingual table.
            if cell["id"] != "builtin-512px-three-image-preview":
                assert len(re.findall(r"^\s*\| .* \|$", en, re.M)) == len(
                    re.findall(r"^\s*\| .* \|$", vi, re.M)
                )
            assert len(re.findall(r"^\d+\. ", en, re.M)) == len(
                re.findall(r"^\d+\. ", vi, re.M)
            )
            assert len(re.findall(r"^- ", en, re.M)) == len(
                re.findall(r"^- ", vi, re.M)
            )
        if path == NOTEBOOK:
            assert len(cells) == 8
            for cell in cells[1:]:
                content = "".join(cell["source"])
                en, vi = content.split("\n---\n")
                assert len(re.findall(r"^\*\*[^*\n]+:\*\*", en, re.M)) == len(
                    re.findall(r"^\*\*[^*\n]+:\*\*", vi, re.M)
                )


def test_failed_sample_writes_partial_report(monkeypatch, tmp_path):
    """Record successful and failed requests without retrying or fabricating measurements."""
    import pytest
    monkeypatch.chdir(tmp_path)
    scope = {}
    cells = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]}
    exec(compile(cells["config"], "config", "exec"), scope)
    exec(compile(cells["client"], "client", "exec"), scope)
    scope["RUN_BENCHMARK"] = True
    calls = []

    def fake_http(path, body=None):
        assert path == "/v1/embeddings"
        calls.append(body)
        if len(calls) == 2:
            raise RuntimeError("simulated HTTP 503")
        return {
            "model": body["model"], "backend": "modal",
            "request_id": body["request_id"],
            "input_sha256": scope["hashes"]["text"],
            "revision": "test-snapshot",
            "embedding_metadata": embedding_metadata(
                body["model"], "test-snapshot", 1024, body.get("instruction")
            ).model_dump(),
            "dimension": 1024, "vector": [1.0] + [0.0] * 1023,
            "elapsed_ms": 3.0,
            "timing": {
                "model_state": "warm", "gpu_worker_first_request": False,
                "batch_size": 1, "gpu_worker_ms": 2.0,
                "model_load_ms": 0.0, "gpu_encode_ms": 1.0,
                "gpu_dispatch_overhead_ms": 0.5,
            },
        }

    scope["http_json"] = fake_http
    model = scope["MODELS"][0]
    first = scope["sample"](model, "text", "warm", 1)
    first["confirmed"] = True
    with pytest.raises(RuntimeError, match="HTTP 503"):
        scope["sample"](model, "text", "warm", 2)
    assert len(calls) == 2
    output = tmp_path / "multimodal_embeddings_performance_summary.json"
    content = output.read_text(encoding="utf-8")
    report = json.loads(content)
    assert report["status"] == "INCOMPLETE_FAILED"
    assert report["retries"] == 0
    assert len(report["trials"]) == 2
    assert [t["status"] for t in report["trials"]] == ["completed", "failed"]
    assert report["trials"][0]["confirmed"] is True
    assert report["trials"][1]["confirmed"] is False
    assert report["trials"][1]["error_type"] == "RuntimeError"
    assert scope["TEXT_SAMPLE"] not in content
    assert "simulated HTTP 503" not in content


def test_sequential_benchmark_rejects_invalid_declared_dimension(monkeypatch, tmp_path):
    """Fake response has a valid 1024-vector but a defective dimension field."""
    import pytest

    monkeypatch.chdir(tmp_path)
    code = {cell["id"]: "".join(cell["source"])
            for cell in notebook()["cells"] if cell["cell_type"] == "code"}
    for variant in ("valid", "missing", "wrong", "boolean", "string"):
        scope = {}
        with redirect_stdout(io.StringIO()):
            exec(compile(code["config"], "config", "exec"), scope)
            exec(compile(code["client"], "client", "exec"), scope)
        scope["RUN_BENCHMARK"] = True

        def response(path, body=None, *, scope=scope, variant=variant):
            assert path == "/v1/embeddings"
            data = {
                "backend": "modal", "model": body["model"],
                "request_id": body["request_id"],
                "input_sha256": scope["hashes"]["text"],
                "revision": "fixture-revision", "dimension": 1024,
                "elapsed_ms": 2.0, "vector": [1.0] + [0.0] * 1023,
                "embedding_metadata": embedding_metadata(
                    body["model"], "fixture-revision", 1024, body.get("instruction")
                ).model_dump(),
                "timing": {
                    "model_state": "warm", "gpu_worker_first_request": False,
                    "gpu_worker_id": "a" * 32, "batch_size": 1,
                    "gpu_worker_ms": 2.0, "model_load_ms": 0.0,
                    "gpu_encode_ms": 1.0, "gpu_dispatch_overhead_ms": 0.1,
                },
            }
            if variant == "missing":
                del data["dimension"]
            elif variant == "wrong":
                data["dimension"] = 64
            elif variant == "boolean":
                data["dimension"] = True
            elif variant == "string":
                data["dimension"] = "1024"
            return data

        scope["http_json"] = response
        model = scope["MODELS"][0]
        if variant == "valid":
            result = scope["sample"](model, "text", "warm", 1)
            assert result["status"] == "completed"
            assert len(scope["rows"]) == 1
        else:
            with pytest.raises(RuntimeError, match="Incorrect declared embedding dimension"):
                scope["sample"](model, "text", "warm", 1)
            assert len(scope["rows"]) == 1
            assert scope["rows"][0]["status"] == "failed"
            assert scope["rows"][0]["confirmed"] is False
            path = tmp_path / "multimodal_embeddings_performance_summary.json"
            assert json.loads(path.read_text())["status"] == "INCOMPLETE_FAILED"


def test_live_benchmark_catalog_preflight_uses_dictionary_keys(monkeypatch, tmp_path):
    """Exercise the enabled real-API preflight with a fake transport (zero GPU calls)."""
    import pytest

    monkeypatch.chdir(tmp_path)

    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.modal.run")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-test-token-123456789")
    monkeypatch.delenv("BENCHMARK_IMAGE_PATH", raising=False)
    original = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), original)
    original.update(RUN_BENCHMARK=True, RUN_COLD=False)
    models = original["MODELS"]

    for available in (models, models[:1]):
        seen = []

        class FakeOpener:
            def __init__(self, ids, recorder):
                self.ids = ids
                self.recorder = recorder

            def open(self, request, timeout=None):
                path = request.full_url.rsplit("/", 1)[-1]
                self.recorder.append(path)
                if path == "health":
                    body = {"backend": "modal"}
                elif path == "models":
                    body = {"models": [{"id": model} for model in self.ids]}
                else:
                    raise AssertionError("Unexpected HTTP endpoint: " + path)
                return io.BytesIO(json.dumps(body).encode("utf-8"))

        opener = FakeOpener(available, seen)
        monkeypatch.setattr(urllib.request, "build_opener", lambda *args, obj=opener: obj)
        scope = dict(original)
        if len(available) == len(models):
            with redirect_stdout(io.StringIO()):
                exec(compile(code["client"], "client", "exec"), scope)
            assert seen == ["health", "models"]
            assert scope["request_body"](models[0], "text", "x")["model"] == models[0]
        else:
            with pytest.raises(RuntimeError, match="Both embedding models"):
                exec(compile(code["client"], "client", "exec"), scope)


def test_new_optional_performance_phases_with_offline_responses(tmp_path):
    """No Modal charges: exercise real notebook cells against fake response envelopes."""
    from contextlib import redirect_stdout
    scope = {}
    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    assert "model_switch" not in code["client"]
    with redirect_stdout(io.StringIO()):
        exec(compile(code["config"], "config", "exec"), scope)
        exec(compile(code["client"], "client", "exec"), scope)
    assert not scope["RUN_BENCHMARK"]
    assert not scope["RUN_CONCURRENT"] and not scope["RUN_MIXED_BATCH"]
    scope.update(RUN_BENCHMARK=True, RUN_CONCURRENT=True, RUN_MIXED_BATCH=True,
                 CONCURRENT_WORKERS=2, CONCURRENT_REQUESTS=2,
                 MIXED_BATCH_REPETITIONS=1, QWEN_INSTRUCTION_MODE="custom")
    scope["run_configuration_snapshot"] = scope["current_run_configuration"]()
    scope["run_completed_phases"].add("preflight")
    calls = []

    def response(body):
        model = body["model"]
        assert ("instruction" in body) == (model == scope["MODELS"][0])
        meta = embedding_metadata(model, "fixture-revision", 1024,
                                  body.get("instruction")).model_dump()
        return {
            "model": model, "backend": "modal", "request_id": body["request_id"],
            "revision": "fixture-revision", "dimension": 1024,
            "input_sha256": scope["hashes"]["text"],
            "vector": [1.0] + [0.0]*1023, "embedding_metadata": meta,
            "elapsed_ms": 3.0,
            "timing": {"model_state": "warm", "gpu_worker_first_request": False,
                       "gpu_worker_id": "0"*32 if model == scope["MODELS"][0] else "1"*32,
                       "batch_size": 1, "gpu_worker_ms": 2.0, "model_load_ms": 0.0,
                       "gpu_encode_ms": 1.0, "gpu_dispatch_overhead_ms": 0.1},
        }

    def fake_http(path, body=None):
        calls.append((path, body))
        if path == "/v1/embeddings":
            return response(body)
        assert path == "/v1/embeddings/batches"
        return {"count": 2, "items": [response(item) for item in body["items"]]}

    # Even an existing user report elsewhere must be unaffected by this test.
    sentinel = tmp_path / "preexisting-user-benchmark.json"
    original_report = b"USER_BENCHMARK_REPORT_MUST_SURVIVE"
    sentinel.write_bytes(original_report)
    scope["http_json"] = fake_http
    with redirect_stdout(io.StringIO()):
        exec(compile(code["concurrent-run"], "concurrent-run", "exec"), scope)
        exec(compile(code["mixed-batch-run"], "mixed-batch-run", "exec"), scope)
    assert len(scope["concurrent_trials"]) == 4
    assert all(t["status"] == "completed" for t in scope["concurrent_trials"])
    assert len(scope["concurrent_summary"]) == 2
    assert len(scope["batch_trials"]) == 1
    assert scope["batch_trials"][0]["status"] == "completed"
    assert scope["batch_trials"][0]["batch_size"] == 2
    assert sum(path == "/v1/embeddings/batches" for path, _ in calls) == 1
    assert sentinel.read_bytes() == original_report
    assert Path.cwd() == tmp_path
    assert not Path("/kaggle/working").is_dir()  # Test-only path isolation.



def test_benchmark_report_shows_bilingual_latency_table():
    final_code = "".join(notebook()["cells"][-1]["source"])
    compile(final_code, "benchmark-report-cell", "exec")
    assert "Confirmed latency / Độ trễ đã xác nhận" in final_code
    assert "Phase / Giai đoạn" in final_code
    assert "Confirmed / Xác nhận" in final_code
    assert "p50 (ms)" in final_code and "p95 (ms)" in final_code


def test_benchmark_optional_probes_precede_results_explanation():
    """The results section belongs after concurrent and mixed-batch probes."""
    cells = notebook()["cells"]
    ids = [cell["id"] for cell in cells]
    expected = [
        "optional-probes-intro", "concurrent-run", "mixed-batch-run",
        "report-intro", "report",
    ]
    assert all(item in ids for item in expected)
    assert [ids.index(item) for item in expected] == sorted(ids.index(item) for item in expected)
    assert ids[-1] == "report"
    intro = "".join(cells[ids.index("optional-probes-intro")]["source"])
    assert "### English" in intro and "### Tiếng Việt" in intro
    assert "throughput duy trì" in intro
    assert "## Results and limitations" in "".join(cells[-2]["source"])


def test_documented_modal_secret_commands_are_valid_bash():
    """Examples must not use shell redirection operators as pseudo-placeholders."""
    import subprocess
    root = NOTEBOOK.parents[1]
    # Check the published README examples when the README is available.
    if not (root / "README.md").is_file():
        import pytest
        pytest.skip("README usage examples unavailable in this checkout")
    for file in ("docs/OPERATIONS.md", "docs/OPERATIONS.vi.md"):
        content = (root / file).read_text(encoding="utf-8")
        assert "--qwen8-revision <" not in content
        assert "--wemm9-revision <" not in content
        fenced = chr(96) * 3
        snippets = re.findall(fenced + r"bash\n(.*?)" + fenced, content, re.S)
        commands = [x for x in snippets if "create_modal_secret.py" in x]
        assert len(commands) == 1, file
        checked = subprocess.run(["bash", "-n"], input=commands[0],
                                 capture_output=True, text=True, check=False)
        assert checked.returncode == 0, (file, checked.stderr)


def test_custom_benchmark_instruction_is_normalized_before_hashing(monkeypatch):
    import pytest

    monkeypatch.setenv("BENCHMARK_QWEN_INSTRUCTION", "  Retrieve sample images.  ")
    cells = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]}
    config = cells["config"].replace(
        'QWEN_INSTRUCTION_MODE = "default" #@param',
        'QWEN_INSTRUCTION_MODE = "custom" #@param',
    )
    scope = {}
    exec(compile(config, "config", "exec"), scope)
    exec(compile(cells["client"], "client", "exec"), scope)
    assert scope["QWEN_CUSTOM_INSTRUCTION"] == "Retrieve sample images."
    body = scope["request_body"](scope["MODELS"][0], "text", "sample-1")
    assert body["instruction"] == "Retrieve sample images."
    monkeypatch.setenv("BENCHMARK_QWEN_INSTRUCTION", "    ")
    with pytest.raises(AssertionError):
        exec(compile(config, "config", "exec"), {})


def test_uv_setup_installs_pip_for_documented_modal_command():
    root = NOTEBOOK.parents[1]
    # Check the published README examples when the README is available.
    if not (root / "README.md").is_file():
        import pytest
        pytest.skip("README usage examples unavailable in this checkout")
    setup = (root / "scripts" / "setup.sh").read_text(encoding="utf-8")
    assert 'uv pip install --python .venv/bin/python pip "${constraint_args[@]}"' in setup
    assert "constraints/cpu.txt" in setup
    for name in ("README.md", "README.vi.md",
                 "docs/OPERATIONS.md", "docs/OPERATIONS.vi.md"):
        assert "python -m pip install -e '.[modal]'" in (root / name).read_text()



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
    import re

    canonical = []
    for cell in nb["cells"]:
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
    cells = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
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
        for cell_id in ("client", "cold-run", "warm-run", "alternate-run",
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
