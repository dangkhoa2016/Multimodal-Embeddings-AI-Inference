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
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return book


@pytest.mark.parametrize("missing", ["both", "url", "token"])
def test_enabled_benchmark_missing_credentials_prompts_without_network(
    monkeypatch, missing,
):
    import sys
    import types

    code = {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}
    monkeypatch.delenv("VISION_EMBEDDINGS_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_EMBEDDINGS_API_TOKEN", raising=False)
    if missing == "url":
        monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "test-credentials-token")
    elif missing == "token":
        monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.invalid")

    module = types.ModuleType("kaggle_secrets")

    class UnavailableSecrets:
        def get_secret(self, _key):
            raise RuntimeError("Unavailable Kaggle secret")

    module.UserSecretsClient = UnavailableSecrets
    monkeypatch.setitem(sys.modules, "kaggle_secrets", module)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    prompts = []

    def cancelled(label):
        prompts.append(label)
        raise EOFError

    def forbidden(*_args, **_kwargs):
        pytest.fail("Missing credentials must not send HTTP")

    monkeypatch.setattr("getpass.getpass", cancelled)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    scope = {}
    exec(compile(code["config"], "config", "exec"), scope)
    scope["RUN_BENCHMARK"] = True
    with pytest.raises(RuntimeError, match="Interactive input unavailable"):
        exec(compile(code["client"], "client", "exec"), scope)
    assert len(prompts) == 1
    assert "Bearer token" in prompts[0] if missing == "token" else "HTTPS URL" in prompts[0]
    assert not Path("multimodal_embeddings_performance_summary.json").exists()


def benchmark_sync_checkout(tmp_path):
    """A tiny valid notebook whose digest has become stale after a code edit."""
    import shutil

    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "notebooks" / "source").mkdir(parents=True)
    tool = checkout / "scripts" / "sync_notebooks.py"
    shutil.copyfile(NOTEBOOK.parents[1] / "scripts" / tool.name, tool)
    config = 'TIMEOUT_SECONDS = 900\nNOTEBOOK_TEMPLATE_SHA256 = "' + "0" * 64 + '"\n'
    client = 'print("helper")\n'
    source = checkout / "notebooks" / "source" / "multimodal_embeddings_benchmark.py"
    source.write_text(
        "# Source for the standalone notebook code cells.\n"
        "# %% NOTEBOOK CELL config\n" + config
        + "# %% NOTEBOOK CELL client\n" + client
    )
    book = checkout / "notebooks" / "multimodal_embeddings_benchmark.ipynb"
    book.write_text(json.dumps({
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [
            {"id": key, "cell_type": "code", "metadata": {},
             "source": text.splitlines(keepends=True), "outputs": [], "execution_count": None}
            for key, text in (("config", config), ("client", client))
        ],
    }))
    return checkout, tool, source, book


def test_sync_check_rejects_a_stale_template_digest_without_writing(tmp_path):
    import subprocess
    import sys

    checkout, tool, source, book = benchmark_sync_checkout(tmp_path)
    before = source.read_bytes(), book.read_bytes()
    result = subprocess.run(
        [sys.executable, str(tool), "--check"], cwd=checkout, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "stale template fingerprint" in result.stderr
    assert (source.read_bytes(), book.read_bytes()) == before


def test_sync_write_refreshes_the_digest_and_is_idempotent(tmp_path):
    import hashlib
    import subprocess
    import sys

    checkout, tool, source, book = benchmark_sync_checkout(tmp_path)
    source.write_text(source.read_text().replace("TIMEOUT_SECONDS = 900", "TIMEOUT_SECONDS = 901"))
    result = subprocess.run(
        [sys.executable, str(tool), "--write"], cwd=checkout, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    # Expected bytes come from the two known fixture cells, with the self-hash excluded.
    expected = hashlib.sha256(
        b'TIMEOUT_SECONDS = 901\nNOTEBOOK_TEMPLATE_SHA256 = "EXCLUDED"\n\nprint("helper")\n'
    ).hexdigest()
    code = {c["id"]: "".join(c["source"]) for c in json.loads(book.read_text())["cells"]}
    assert f'NOTEBOOK_TEMPLATE_SHA256 = "{expected}"' in source.read_text()
    assert f'NOTEBOOK_TEMPLATE_SHA256 = "{expected}"' in code["config"]
    assert "TIMEOUT_SECONDS = 901" in code["config"]
    before = source.read_bytes(), book.read_bytes()
    for mode in ("--check", "--write"):
        result = subprocess.run(
            [sys.executable, str(tool), mode], cwd=checkout, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (source.read_bytes(), book.read_bytes()) == before


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
        exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
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
    assert "reversed(MODELS)" in warm and "Random(20261114 + block)" in warm
    assert "WARM_REPETITIONS // 2" in warm
    assert 'require_warm_observation(result, "alternate")' in code["alternate-run"]
    scope = {}
    exec(compile(code["config"], "config", "exec"), scope)
    scope.update(IMAGE_PATH="", TEXT_SAMPLE="sample", RUN_BENCHMARK=False)
    exec(compile(code["client"], "client", "exec"), scope)
    exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
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
    enabled_cold = edited.replace('BENCHMARK_MODE = "warm" #@param',
                                  'BENCHMARK_MODE = "cold" #@param')
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
        license_notes = (root / ("LICENSE-NOTES" + suffix)).read_text(encoding="utf-8")
        readme = (root / ("README" + suffix)).read_text(encoding="utf-8")
        guide = (root / "docs" / ("NOTEBOOK_GUIDE" + suffix)).read_text(encoding="utf-8")
        assert "routinging" not in changelog
        assert "two-request" not in changelog and "hai request" not in changelog
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
        benchmark_guide = (root / "benchmarks" / ("README" + suffix)).read_text(
            encoding="utf-8"
        )
        assert "BENCHMARK_MODE" in benchmark_guide


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
    exec(compile(cells["client-request-helpers"], "client-request-helpers", "exec"), scope)
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
            exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
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
                elif path in self.ids:
                    assert request.get_method() == "GET"
                    body = {"id": path}
                else:
                    raise AssertionError("Unexpected HTTP endpoint: " + path)
                return io.BytesIO(json.dumps(body).encode("utf-8"))

        opener = FakeOpener(available, seen)
        monkeypatch.setattr(urllib.request, "build_opener", lambda *args, obj=opener: obj)
        scope = dict(original)
        if len(available) == len(models):
            with redirect_stdout(io.StringIO()):
                exec(compile(code["client"], "client", "exec"), scope)
                exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"),
                     scope)
            assert seen == ["health", "models", *models]
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
        exec(compile(code["client-request-helpers"], "client-request-helpers", "exec"), scope)
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
    # Separate optional-phase response tests from async provisioning.
    def offline_prepare(model, modality, phase, round_number):
        worker = ("0" if model == scope["MODELS"][0] else "1") * 32
        scope["gpu_readiness"][model] = {
            "worker_id": worker, "observed_at": scope["time"].perf_counter(),
            "phase": phase, "round": round_number,
        }
        return {"gpu_worker_id": worker}
    scope["async_prepare_gpu"] = offline_prepare
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
    assert "English — Confirmed latency" in final_code
    assert "Tiếng Việt — Độ trễ đã xác nhận" in final_code
    assert "| Phase | Model | Input | Confirmed | Latency n |" in final_code
    assert "| Giai đoạn | Model | Đầu vào | Đã xác nhận |" in final_code
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
    exec(compile(cells["client-request-helpers"], "client-request-helpers", "exec"), scope)
    assert scope["QWEN_CUSTOM_INSTRUCTION"] == "Retrieve sample images."
    body = scope["request_body"](scope["MODELS"][0], "text", "sample-1")
    assert body["instruction"] == "Retrieve sample images."
    monkeypatch.setenv("BENCHMARK_QWEN_INSTRUCTION", "    ")
    with pytest.raises(ValueError, match="BENCHMARK_QWEN_INSTRUCTION") as error:
        exec(compile(config, "config", "exec"), {})
    assert "phải có" in str(error.value)


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



def test_unpublished_sources_require_explicit_opt_in_and_remain_strict_after_publication(tmp_path):
    import subprocess
    import sys

    checkout, tool, _source, book = benchmark_sync_checkout(tmp_path)
    published = book.read_bytes()
    book.unlink()

    def check(*extra):
        return subprocess.run([sys.executable, str(tool), "--check", *extra],
                              cwd=checkout, capture_output=True, text=True)

    assert check().returncode != 0
    allowed = check("--allow-unpublished")
    assert allowed.returncode == 0, allowed.stderr
    assert not book.exists()

    book.write_bytes(published)
    strict = check("--allow-unpublished")
    assert strict.returncode != 0 and "stale template fingerprint" in strict.stderr

    book.unlink()
    (checkout / "README.md").write_text("# Published project\n")
    assert check("--allow-unpublished").returncode != 0
