"""Validate the Kaggle/Colab HTTP-only notebook without Modal cloud access."""

import base64
import hashlib
import io
import json
import math
import os
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from test_notebook_template import load_notebook, seed_report_input_fixture

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.config import Settings

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "multimodal_embeddings_api_demo.ipynb"
MODELS = ("qwen3-vl-embedding-8b", "wemm-embedding-9b")


def test_custom_image_preview_requires_opt_in_and_hides_path(monkeypatch, tmp_path):
    import sys
    import types

    from PIL import Image

    private = tmp_path / "private-identity.png"
    Image.new("RGB", (2, 2), "purple").save(private)
    encoded = base64.b64encode(private.read_bytes()).decode("ascii")
    monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH", str(private))
    shown = []
    package = types.ModuleType("IPython")
    package.__path__ = []
    display = types.ModuleType("IPython.display")
    display.HTML = lambda data: data
    display.display = shown.append
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    for key in ("configure-code", "configure-helpers-code", "sample-options-code",
                "sample-preparation-code"):
        exec(compile(cells[key], key, "exec"), scope)
    assert not any(encoded in html for html in shown)
    scope["PREVIEW_CUSTOM_IMAGES"] = True
    exec(compile(cells["sample-preparation-code"], "opted-in-preview", "exec"), scope)
    previews = [html for html in shown if encoded in html]
    assert len(previews) == 1 and "data:image/png;base64," in previews[0]
    assert all(str(private) not in html and private.name not in html for html in shown)




def test_report_timing_totals_and_labels_are_localized():
    book = load_notebook()
    cell = next(c for c in book["cells"] if c["id"] == "results-code")
    source = "".join(cell["source"])
    assert cell["metadata"]["jupyter"]["source_hidden"] is True
    assert "API processing time (text / image)" in source
    assert "Thời gian xử lý API (văn bản / ảnh)" in source
    assert '{t["text"]}: {seconds(total_text)}' in source
    assert '{t["image"]}: {seconds(total_image)}' in source
    assert "matrix_html(language)" in source
    assert 'scope="row"' in source
    assert 'scope="col"' in source
    assert "điều phối/lập lịch" in source


def test_custom_inputs_are_labeled_and_never_previewed(monkeypatch, tmp_path):
    """A private image path must never be leaked into the HTML demo report."""
    from PIL import Image

    private = tmp_path / "private-photo.png"
    Image.new("RGB", (16, 16), (20, 40, 60)).save(private)
    monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH", str(private))
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_TEXT_2", "an unrelated custom text")
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_LABEL", "<script>pair</script>")
    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", "qwen3-vl-embedding-8b")

    cells = {
        cell["id"]: "".join(cell["source"])
        for cell in load_notebook()["cells"] if cell["cell_type"] == "code"
    }
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-code"], "configure-code", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
        exec(compile(cells["sample-preparation-code"], "sample-preparation-code", "exec"), scope)
    assert scope["image_sources"] == ["custom", "built-in", "built-in"]
    assert scope["text_sources"] == ["built-in", "custom", "built-in"]
    assert scope["default_pairs"] is False
    assert scope["image_samples"][0][:8] == bytes([137, 80, 78, 71, 13, 10, 26, 10])

    result = {
        "elapsed_ms": 2.0,
        "_client_elapsed_ms": 3.0,
        "embedding_metadata": {"preprocessing_version": "fixture",
                               "config_sha256": "c" * 64},
    }
    sample = {
        "sample": 1, "model": "qwen3-vl-embedding-8b",
        "revision": "fixture", "backend": "mock", "dimension": 1024,
        "text_norm": 1.0, "image_norm": 1.0, "cosine_text_image": 0.6,
        "text_elapsed_ms": 2.0, "image_elapsed_ms": 2.0,
    }
    scope.update(
        math=math,
        completed_run_config={
            "model": scope["model_id"],
            "dimension": scope["dimension"],
            "instruction_mode": scope["instruction_mode"],
            "custom_instruction_sha256": None,
            "config_sha256": "c" * 64,
        },
        text_results=[dict(result) for _ in range(3)],
        image_results=[dict(result) for _ in range(3)],
        summary=[dict(sample, sample=i + 1) for i in range(3)],
        similarity_matrix=[[0.6] * 3 for _ in range(3)],
    )
    seed_report_input_fixture(scope)
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["results-code"], "results-code", "exec"), scope)
    html = scope["report_html"]
    assert "Input sources" in html
    assert "Nguồn đầu vào" in html
    assert "Sample 1" in html
    assert "Mẫu 2" in html
    assert "custom" in html and "tùy chỉnh" in html
    assert "Matching pair 1" not in html
    assert "&lt;script&gt;pair&lt;/script&gt;" in html
    assert str(private) not in html
    assert "<script>" not in html


def test_http_error_codes_include_bilingual_sanitized_guidance(monkeypatch):
    import urllib.error

    code = next(
        "".join(c["source"]) for c in load_notebook()["cells"]
        if c["id"] == "http-verification-code"
    )
    helpers = code.split("verified_model_id = None  # Previous model", 1)[0]
    hints = {
        401: ("Bearer token", "Bearer token"),
        413: ("image/request", "dung lượng"),
        422: ("pixels", "pixel"),
        503: ("worker", "worker"),
    }
    for status, phrases in hints.items():
        class Opener:
            def __init__(self, http_status):
                self.http_status = http_status

            def open(self, request, timeout):
                raise urllib.error.HTTPError(
                    request.full_url, self.http_status,
                    "secret-error-body", {}, io.BytesIO(b"PRIVATE_RESPONSE_BODY")
                )

        fake_opener = Opener(status)
        monkeypatch.setattr(urllib.request, "build_opener",
                            lambda *args, opener=fake_opener: opener)
        scope = {
            "api_token": "secret-user-token",
            "validate_api_token": lambda value: value,
            "base_url": "https://sensitive.example",
            "connection_ready": True,
            "configured_connection_identity": (
                "https://sensitive.example",
                hashlib.sha256(b"secret-user-token").hexdigest(),
            ),
            "validated_base_url": lambda url: url,
            "timeout": 30,
            "message_pair": lambda en, vi: "English: " + en + " Tiếng Việt: " + vi,
        }
        scope["configuration_ready"] = True  # Isolated helper fixture has valid form config.
        scope["require_current_configuration"] = lambda: None
        exec(compile(helpers, "api-error-helpers", "exec"), scope)
        with pytest.raises(RuntimeError) as error:
            scope["api_request"]("/v1/embeddings", {"model": "test"})
        message = str(error.value)
        assert f"HTTP {status}" in message
        assert all(phrase in message for phrase in phrases)
        assert "PRIVATE_RESPONSE_BODY" not in message
        assert "secret-error-body" not in message
        assert "secret-user-token" not in message
        assert "sensitive.example" not in message


def test_api_request_timeout_is_bilingual_sanitized_and_never_retried(monkeypatch):
    """Socket open/read timeouts are not silently retried; no private data leaks."""
    import traceback
    import urllib.request

    code = next(
        "".join(c["source"]) for c in load_notebook()["cells"]
        if c["id"] == "http-verification-code"
    )
    helpers = code.split("verified_model_id = None  # Previous model", 1)[0]
    secret = "PRIVATE_TIMEOUT_TOKEN"
    endpoint = "https://private-timeout-endpoint.example"
    raw_timeout = "PRIVATE_TIMEOUT_INPUT_PAYLOAD"

    for phase in ("open", "read", "wrapped_url_error"):
        calls = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, *args):
                raise TimeoutError(raw_timeout)

        class FakeOpener:
            def __init__(self, timeout_phase, call_log):
                self.timeout_phase = timeout_phase
                self.call_log = call_log

            def open(self, request, timeout=None):
                self.call_log.append((request.get_method(), timeout))
                if self.timeout_phase == "open":
                    raise TimeoutError(raw_timeout)
                if self.timeout_phase == "wrapped_url_error":
                    raise urllib.error.URLError(TimeoutError(raw_timeout))
                return FakeResponse()

        opener = FakeOpener(phase, calls)
        monkeypatch.setattr(
            urllib.request, "build_opener", lambda *args, o=opener: o
        )
        scope = {
            "api_token": secret,
            "validate_api_token": lambda token: token,
            "base_url": endpoint,
            "connection_ready": True,
            "configured_connection_identity": (
                endpoint, hashlib.sha256(secret.encode()).hexdigest()
            ),
            "validated_base_url": lambda url: url,
            "timeout": 23,
            "message_pair": lambda en, vi: "English: " + en + " Tiếng Việt: " + vi,
        }
        scope["configuration_ready"] = True  # Isolated helper fixture has valid form config.
        scope["require_current_configuration"] = lambda: None
        exec(compile(helpers, "api-timeout-helpers", "exec"), scope)
        with pytest.raises(RuntimeError) as caught:
            scope["api_request"]("/v1/embeddings", {"model": "test"})
        rendered = "".join(traceback.format_exception(caught.value))
        assert "timed out" in rendered and "quá thời gian chờ" in rendered
        assert "GPU worker" in rendered and "không tự động gửi lại POST" in rendered
        for sensitive in (secret, endpoint, raw_timeout):
            assert sensitive not in rendered
        assert calls == [("POST", 23)], (phase, calls)


def test_invalid_bearer_token_cannot_leak_from_notebook_traceback(monkeypatch):
    """CR/LF in the middle of an entered token must not reach urllib or notebook output."""
    import traceback

    secret = "PRIVATE_FAKE_TOKEN_1234567890\nSECOND_LINE"
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.test")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", secret)
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-code"], "configure-code", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
    with pytest.raises(ValueError) as caught:
        exec(compile(cells["credentials-input"], "credentials-input", "exec"), scope)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "Bearer token" in rendered
    assert "PRIVATE_FAKE_TOKEN" not in rendered
    assert secret not in rendered

    # A re-run that replaces the token without running the credentials cell
    # must still be rejected before the HTTP transport sends a request.
    helper = cells["http-verification-code"].split(
        "verified_model_id = None  # Previous model", 1
    )[0]
    scope.update(api_token=secret, base_url="https://example.test", timeout=30)
    with pytest.raises(ValueError) as caught:
        exec(compile(helper, "http-helper", "exec"), scope)
        scope["api_request"]("/v1/embeddings", {"model": "test"})
    assert "PRIVATE_FAKE_TOKEN" not in "".join(traceback.format_exception(caught.value))

    # The transport itself can emit unsafe ValueError, which must be sanitized.
    safe = "safe-token-123456789"
    scope["api_token"] = safe
    scope["connection_ready"] = True
    scope["configured_connection_identity"] = (
        "https://example.test", hashlib.sha256(safe.encode()).hexdigest()
    )

    class UnsafeTransport:
        def open(self, request, timeout):
            raise ValueError("Invalid header value Bearer " + safe)

    scope["opener"] = UnsafeTransport()
    with pytest.raises(RuntimeError) as caught:
        scope["api_request"]("/v1/health")
    assert safe not in "".join(traceback.format_exception(caught.value))


def test_text_hash_uses_stripped_utf8_for_reusable_notebook_helper():
    """Match the real Pydantic request normalization using a local mock API."""
    import hashlib

    token = "offline-token-123456"
    app = TestClient(create_app(Settings(api_token=token, backend="mock")))
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    prefix = cells["embedding-code"].split("completed_run_config = None", 1)[0]
    calls = []

    def api_request(path, payload=None):
        calls.append((path, payload))
        response = app.post(path, json=payload, headers={"Authorization": "Bearer " + token})
        assert response.status_code == 200, response.status_code
        return response.json()

    scope = {
        "math": math, "json": json, "hashlib": hashlib, "base64": base64,
        "dimension": 1024, "instruction_mode": "default",
        "custom_instruction": "", "supports_custom_instruction": False,
        "verified_model_id": "wemm-embedding-9b",
        "connection_ready": True, "verified_connection_identity": "test-session",
        "current_connection_identity": lambda: "test-session",
        "QWEN_LEGACY_INSTRUCTION": "Represent the image or text for retrieval in a visual gallery.",
        "message_pair": lambda en, vi: en + " / " + vi,
        "require_real": lambda: False,
        "api_request": api_request,
    }
    scope["configuration_ready"] = True  # Isolated helper fixture has valid form config.
    scope.update(require_current_configuration=lambda: None, health={"backend": "mock"})
    exec(compile(prefix, "embedding-helpers", "exec"), scope)
    result, norm = scope["checked_embedding"](
        "wemm-embedding-9b", text="  hello world  "
    )
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert calls[0][1]["text"] == "hello world"
    assert result["input_sha256"] == expected
    assert result["input_sha256"] != hashlib.sha256(
        b"  hello world  "
    ).hexdigest()
    assert abs(norm - 1.0) < 0.001
    assert "input_sha256" in result

    with pytest.raises(ValueError, match="Text must not be empty"):
        scope["checked_embedding"]("wemm-embedding-9b", text="  \n  ")
    assert len(calls) == 1


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize(
    "variant",
    [
        "valid", "wrong_dimension", "missing_dimension", "boolean_dimension",
        "boolean_vector", "tuple_vector", "object_vector", "scalar_vector",
        "short_vector", "non_finite_vector",
    ],
)
def test_demo_rejects_malformed_embedding_response(model_id, variant):
    """Validate declared dimensions, actual vector shape and strict numeric types."""
    import hashlib

    token = "safe-offline-token-12345"
    api = TestClient(create_app(Settings(api_token=token, backend="mock")))
    source = next("".join(c["source"]) for c in load_notebook()["cells"]
                  if c["id"] == "embedding-code")
    helpers = source.split("completed_run_config = None", 1)[0]

    def request(path, body):
        response = api.post(
            path, json=body, headers={"Authorization": "Bearer " + token}
        )
        assert response.status_code == 200
        data = response.json()
        if variant == "wrong_dimension":
            data["dimension"] = 1
        elif variant == "missing_dimension":
            data.pop("dimension")
        elif variant == "boolean_dimension":
            data["dimension"] = True
        elif variant == "boolean_vector":
            data["vector"] = [True] + [False] * 1023
        elif variant == "tuple_vector":
            data["vector"] = tuple(data["vector"])
        elif variant == "object_vector":
            data["vector"] = {"first": 1}
        elif variant == "scalar_vector":
            data["vector"] = 1.0
        elif variant == "short_vector":
            data["vector"] = [1.0]
        elif variant == "non_finite_vector":
            data["vector"][0] = float("nan")
        return data

    scope = {
        "math": math, "json": json, "base64": base64, "hashlib": hashlib,
        "dimension": 1024, "instruction_mode": "default",
        "custom_instruction": "", "supports_custom_instruction": False,
        "verified_model_id": model_id,
        "connection_ready": True, "verified_connection_identity": "test-session",
        "current_connection_identity": lambda: "test-session",
        "QWEN_LEGACY_INSTRUCTION": (
            "Represent the image or text for retrieval in a visual gallery."
        ),
        "message_pair": lambda en, vi: en + " / " + vi,
        "require_real": lambda: False,
        "api_request": request,
    }
    scope["configuration_ready"] = True  # Isolated helper fixture has valid form config.
    scope.update(require_current_configuration=lambda: None, health={"backend": "mock"})
    exec(compile(helpers, "embedded-validation", "exec"), scope)
    check = scope["checked_embedding"]
    if variant == "valid":
        result, norm = check(model_id, text="a blue sphere")
        assert result["dimension"] == 1024
        assert abs(norm - 1.0) < 0.001
    else:
        with pytest.raises(RuntimeError):
            check(model_id, text="a blue sphere")


def test_report_refuses_input_and_label_edits_until_inference_rerun(monkeypatch, tmp_path):
    """An interactive notebook must not relabel old cosine scores as new inputs."""
    token = "safe-offline-test-token-123456"
    client = TestClient(create_app(Settings(api_token=token, backend="mock")))
    calls = []

    class FakeTransport:
        def open(self, request, timeout=None):
            path = urlsplit(request.full_url).path
            if request.data is not None:
                calls.append(json.loads(request.data.decode("utf-8")))
            response = client.request(
                request.get_method(), path,
                headers=dict(request.header_items()), content=request.data,
            )
            assert response.status_code == 200, (path, response.status_code)
            return io.BytesIO(response.content)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: FakeTransport())
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "http://127.0.0.1:18867")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", token)
    monkeypatch.setenv("VISION_EMBEDDINGS_REQUIRE_REAL", "false")
    for key in tuple(os.environ):
        if (key.startswith("VISION_EMBEDDINGS_SAMPLE_")
                or key.startswith("VISION_EMBEDDINGS_IMAGE_PATH")):
            monkeypatch.delenv(key)

    cells = {cell["id"]: "".join(cell["source"]) for cell in load_notebook()["cells"]
             if cell["cell_type"] == "code"}
    scope = {"__name__": "__main__"}
    with redirect_stdout(io.StringIO()):
        for cell in load_notebook()["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), cell["id"], "exec"), scope)

    assert len(calls) == 6
    initial_report = scope["report_html"]
    old_hashes = scope["completed_run_config"]["input_sha256"]
    image_path = tmp_path / "authorized-sample.webp"
    image_path.write_bytes(scope["SYNTHETIC_IMAGES"][1])
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_TEXT", "a newly described landscape")
    monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH", str(image_path))
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_LABEL", "New label")
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["sample-preparation-code"], "new-samples", "exec"), scope)
    with pytest.raises(RuntimeError, match="Inputs or labels changed since inference"):
        exec(compile(cells["results-code"], "stale-result", "exec"), scope)
    assert len(calls) == 6
    assert scope["report_html"] == initial_report

    with redirect_stdout(io.StringIO()):
        exec(compile(cells["embedding-code"], "new-inference", "exec"), scope)
        exec(compile(cells["results-code"], "new-results", "exec"), scope)
    assert len(calls) == 12
    assert scope["completed_run_config"]["input_sha256"] != old_hashes
    assert "New label" in scope["report_html"]

    # Changing *only* a label must invalidate the previous completed report.
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_LABEL", "Label edited again")
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["sample-preparation-code"], "new-label", "exec"), scope)
    with pytest.raises(RuntimeError, match="Inputs or labels changed since inference"):
        exec(compile(cells["results-code"], "stale-label", "exec"), scope)
    assert len(calls) == 12


def test_privacy_transfer_and_result_interpretation_match_both_languages():
    """Prevent a false claim that custom images never leave the runtime."""
    book = load_notebook()
    cells = {c["id"]: "".join(c["source"]) for c in book["cells"]}
    samples = cells["sample-options"]
    results = cells["results-explanation"]
    assert "the notebook does not upload or download your images" not in samples
    assert "notebook không tự tải lên hay tải ảnh về" not in samples
    assert "the notebook sends the image contents to the configured API over HTTPS" in samples
    assert "notebook gửi nội dung ảnh tới API đã cấu hình qua HTTPS" in samples
    assert "Neither API processing time nor cosine similarity" in results
    assert "retrieval accuracy or biometric identity" in results
    assert "độ chính xác retrieval hoặc danh tính sinh trắc học" in results
    assert "retrieval or retrieval accuracy" not in results


def test_demo_intro_keeps_only_essential_first_run_steps():
    book = load_notebook()
    cells = {c["id"]: "".join(c["source"]) for c in book["cells"]}
    intro = cells["intro"]
    assert intro.count("1. **") == intro.count("4. **") == 2
    assert "See **Usage notes**" in intro
    assert "Xem **Lưu ý sử dụng**" in intro
    assert "preserve legacy embeddings" not in cells["configure-options"]
    assert "giữ embedding cũ" not in cells["configure-options"]


def test_custom_image_read_error_keeps_preview_and_hides_path(monkeypatch, tmp_path):
    """Invalid private input must not prevent the independent built-in preview."""
    import sys
    from types import ModuleType

    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    private = tmp_path / "private-directory" / "nonexistent-sensitive-photo.webp"
    monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH_2", str(private))
    shown = []

    # Keep CPU tests runnable with only [dev] dependencies; IPython is an
    # optional hosted-notebook UI feature, not a package setup prerequisite.
    package = ModuleType("IPython")
    package.__path__ = []
    display_module = ModuleType("IPython.display")

    class FakeHTML:
        def __init__(self, data):
            self.data = data

    display_module.HTML = FakeHTML
    display_module.display = lambda markup: shown.append(markup)
    package.display = display_module
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-code"], "configure", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
        with pytest.raises(ValueError) as raised:
            exec(compile(cells["sample-preparation-code"], "prepare", "exec"), scope)
    error = str(raised.value)
    assert "Image 2" in error and "Ảnh 2" in error
    assert "Colab/Kaggle" in error
    assert str(private) not in error
    assert len(shown) >= 1
    previews = [obj.data for obj in shown if 'data:image/webp;base64,' in obj.data]
    assert len(previews) == 1
    assert previews[0].count('data:image/webp;base64,') == 3
    assert all("private-directory" not in obj.data for obj in shown)
    assert raised.value.__cause__ is None

    private.parent.mkdir()
    private.write_bytes(scope["SYNTHETIC_IMAGES"][1])
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["sample-preparation-code"], "corrected", "exec"), scope)
    assert scope["image_samples"][1] == private.read_bytes()
    assert len(scope["image_samples"]) == 3


def test_custom_image_permission_error_omits_private_path(monkeypatch, tmp_path):
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    private = tmp_path / "internal-image.webp"
    private.write_bytes(b"fake")
    monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH_3", str(private))
    original = Path.read_bytes

    def forbidden(path):
        if path == private:
            raise PermissionError("Permission denied: " + str(private))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-code"], "configure", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
        with pytest.raises(ValueError) as raised:
            exec(compile(cells["sample-preparation-code"], "prepare", "exec"), scope)
    assert "Image 3" in str(raised.value)
    assert str(private) not in str(raised.value)
    assert raised.value.__cause__ is None


def test_cosine_matrix_uses_semantic_html_table_headers():
    """3x3 scores must expose both axes to screen-reader table navigation."""
    source = next("".join(c["source"]) for c in load_notebook()["cells"]
                  if c["id"] == "results-code")
    assert "def matrix_html(language):" in source
    assert "role=\\\"grid\\\"" not in source
    assert "<table " in source and "<caption " in source
    assert 'scope="row"' in source and 'scope="col"' in source
    assert "enumerate(similarity_matrix)" in source
    assert "safe_text(f" in source


def test_switching_wemm_to_qwen_custom_requires_fresh_model_preflight(monkeypatch):
    """The previous model's instruction capability must never authorize a new model."""
    token = "offline-notebook-model-switch-token"
    client = TestClient(create_app(Settings(api_token=token, backend="mock")))
    sent = []

    class LocalTransport:
        def open(self, request, timeout=None):
            path = urlsplit(request.full_url).path
            if request.data is not None and path == "/v1/embeddings":
                sent.append(json.loads(request.data.decode("utf-8")))
            response = client.request(
                request.get_method(), path,
                headers=dict(request.header_items()), content=request.data,
            )
            assert response.status_code == 200, (path, response.status_code)
            return io.BytesIO(response.content)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: LocalTransport())
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "http://127.0.0.1:18867")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", token)
    monkeypatch.setenv("VISION_EMBEDDINGS_REQUIRE_REAL", "false")
    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", "wemm-embedding-9b")
    monkeypatch.setenv("VISION_EMBEDDINGS_INSTRUCTION_MODE", "default")
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        for name in ("configure-code", "configure-helpers-code", "sample-options-code",
                     "sample-preparation-code", "credentials-input",
                     "http-verification-code"):
            exec(compile(cells[name], name, "exec"), scope)
    assert scope["verified_model_id"] == "wemm-embedding-9b"
    assert scope["supports_custom_instruction"] is False
    assert sent == []

    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", "qwen3-vl-embedding-8b")
    monkeypatch.setenv("VISION_EMBEDDINGS_INSTRUCTION_MODE", "custom")
    monkeypatch.setenv(
        "VISION_EMBEDDINGS_CUSTOM_INSTRUCTION", "Retrieve matching sample images."
    )
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-code"], "qwen-config", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
    # This is precisely the stale WeMM capability reproduced by the review.
    assert scope["supports_custom_instruction"] is False
    assert scope["verified_model_id"] == "wemm-embedding-9b"
    with pytest.raises(RuntimeError, match="Model changed or API not checked"):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "stale-preflight", "exec"), scope)
    assert sent == []  # The entire inference phase was blocked before POST.

    with redirect_stdout(io.StringIO()):
        exec(compile(cells["http-verification-code"], "qwen-preflight", "exec"), scope)
    assert scope["verified_model_id"] == "qwen3-vl-embedding-8b"
    assert scope["supports_custom_instruction"] is True
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["embedding-code"], "qwen-inference", "exec"), scope)
    assert len(sent) == 6
    assert all(call["model"] == "qwen3-vl-embedding-8b" for call in sent)
    assert all(call["instruction"] == "Retrieve matching sample images."
               for call in sent)


def test_failed_third_image_reprepare_blocks_all_inference_posts(monkeypatch, tmp_path):
    """An attempted manual inference after failed preparation must send zero POSTs."""
    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        for name in ("configure-code", "configure-helpers-code", "sample-options-code",
                     "sample-preparation-code"):
            exec(compile(cells[name], name, "exec"), scope)
    assert scope["sample_prepared"] is True
    original = tuple(scope["image_samples"])
    bad_path = tmp_path / "confidential" / "missing-third.webp"
    monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH_3", str(bad_path))
    with pytest.raises(ValueError, match="Image 3"):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["sample-preparation-code"], "failed-reprepare", "exec"), scope)
    assert scope["sample_prepared"] is False
    assert scope["image_samples"] == list(original)  # No partial publish.

    sent = []
    scope["verified_model_id"] = scope["model_id"]
    scope["api_request"] = lambda *a, **k: sent.append((a, k))
    with pytest.raises(RuntimeError, match="preparation failed"):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "after-failed-preparation", "exec"), scope)
    assert sent == []
    assert str(bad_path) not in repr(sent)

    monkeypatch.delenv("VISION_EMBEDDINGS_IMAGE_PATH_3")
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["sample-preparation-code"], "recovered-preparation", "exec"), scope)
    assert scope["sample_prepared"] is True
    assert len(scope["text_samples"]) == len(scope["image_samples"]) == 3
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_TEXT", "changed after preparation")
    with pytest.raises(RuntimeError, match="Samples or settings changed"):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "stale-sample-selection", "exec"), scope)
    assert sent == []


def test_model_rerun_keeps_custom_text_and_image(monkeypatch, tmp_path):
    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        for name in (
            "configure-code", "configure-helpers-code",
            "sample-options-code", "sample-preparation-code",
        ):
            exec(compile(cells[name], name, "exec"), scope)
    path = tmp_path / "authorized.webp"
    path.write_bytes(scope["SYNTHETIC_IMAGES"][1])
    custom = cells["sample-options-code"].replace(
        'SAMPLE_TEXT_1 = "a flat bright-green triangle on a cream background"',
        'SAMPLE_TEXT_1 = "my custom search text"',
    ).replace('IMAGE_PATH_2 = ""', 'IMAGE_PATH_2 = ' + repr(str(path)))
    with redirect_stdout(io.StringIO()):
        exec(compile(custom, "custom-samples", "exec"), scope)
        exec(compile(cells["sample-preparation-code"], "custom-prepare", "exec"), scope)
    expected = (scope["text_samples"].copy(), scope["image_samples"].copy())
    assert expected[0][0] == "my custom search text"
    assert expected[1][1] == path.read_bytes()
    for model in ("wemm-embedding-9b", "qwen3-vl-embedding-8b"):
        monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", model)
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["configure-code"], "new-model", "exec"), scope)
            exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
            exec(compile(cells["sample-preparation-code"], "new-prepare", "exec"), scope)
        assert scope["text_samples"] == expected[0]
        assert scope["image_samples"] == expected[1]
        assert scope["image_sources"][1] == "custom"
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_TEXT", "environment wins")
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["sample-preparation-code"], "env", "exec"), scope)
    assert scope["text_samples"][0] == "environment wins"
    monkeypatch.delenv("VISION_EMBEDDINGS_SAMPLE_TEXT")
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["sample-preparation-code"], "restored", "exec"), scope)
    assert scope["text_samples"][0] == "my custom search text"



@pytest.mark.parametrize("url", [
    "https://user:PRIVATE_URL_PASSWORD@private.example：443",
    "https://private customer.example",
    "https://private.example:not-a-port",
    "https://private.example:99999",
    "https://private.\nexample",
])
def test_malformed_hidden_url_is_rejected_without_private_traceback(monkeypatch, url):
    import traceback

    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", url)
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-url-token-123456")
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    for key in ("configure-code", "configure-helpers-code"):
        exec(compile(cells[key], key, "exec"), scope)
    with pytest.raises(ValueError) as caught:
        exec(compile(cells["credentials-input"], "url-input", "exec"), scope)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "private.example" not in rendered and "PRIVATE_URL_PASSWORD" not in rendered
    assert "private customer.example" not in rendered
    assert scope.get("connection_ready") is False


def test_demo_sanitizes_invalid_url_transport_without_retry(notebook_connection):
    import http.client
    import traceback

    scope, _cells, _requests, token, _url = notebook_connection
    calls = []

    class InvalidTransport:
        def open(self, request, timeout):
            calls.append(request.get_method())
            raise http.client.InvalidURL("private-host " + token)

    scope["opener"] = InvalidTransport()
    with pytest.raises(RuntimeError) as caught:
        scope["api_request"]("/v1/embeddings", {"text": "public fixture"})
    rendered = "".join(traceback.format_exception(caught.value))
    assert token not in rendered and "private-host" not in rendered
    assert calls == ["POST"]
