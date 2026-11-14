"""Validate the Kaggle/Colab HTTP-only notebook without Modal cloud access."""

import base64
import hashlib
import io
import json
import math
import os
import re
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.config import Settings

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "multimodal_embeddings_api_demo.ipynb"
MODELS = ("qwen3-vl-embedding-8b", "wemm-embedding-9b")


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_code():
    return "\n".join(
        "".join(cell["source"]) for cell in load_notebook()["cells"] if cell["cell_type"] == "code"
    )


def seed_report_input_fixture(scope):
    """Attach a matching recorded-run envelope to offline HTML-only fixtures."""

    scope.setdefault("require_current_configuration", lambda: None)
    scope.setdefault("text_samples", ["fixture text"] * 3)
    scope.setdefault("image_samples", [b"fixture image"] * 3)
    scope.setdefault("sample_labels", [""] * 3)
    scope.setdefault("text_sources", ["built-in"] * 3)
    scope.setdefault("image_sources", ["built-in"] * 3)
    scope.setdefault("default_pairs", True)
    scope.setdefault("sample_prepared", True)
    scope.setdefault("sample_selection_snapshot", ("prepared HTML fixture",))
    scope.setdefault("current_sample_selection", lambda: scope["sample_selection_snapshot"])
    identity = ("https://offline-report.example.invalid", "a" * 64)
    scope.setdefault("current_connection_identity", lambda: identity)
    scope.setdefault("verified_model_id", scope.get("model_id", "qwen3-vl-embedding-8b"))
    scope.setdefault("verified_connection_identity", identity)
    hashes = {
        "text": tuple(hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
                      for text in scope["text_samples"]),
        "image": tuple(hashlib.sha256(image).hexdigest() for image in scope["image_samples"]),
    }
    for modality in ("text", "image"):
        for result, digest in zip(scope[modality + "_results"], hashes[modality], strict=True):
            result["input_sha256"] = digest
    scope.setdefault("completed_run_config", {
        "model": scope.get("model_id", "qwen3-vl-embedding-8b"),
        "dimension": scope.get("dimension", 1024),
        "instruction_mode": scope.get("instruction_mode", "default"),
        "custom_instruction_sha256": None,
        "config_sha256": scope["text_results"][0]["embedding_metadata"]["config_sha256"],
    })
    scope["completed_run_config"].update(
        connection_identity=identity,
        input_sha256=hashes,
        sample_labels=tuple(scope["sample_labels"]),
        text_sources=tuple(scope["text_sources"]),
        image_sources=tuple(scope["image_sources"]),
        default_pairs=scope["default_pairs"],
    )


def test_notebook_structure_and_clean_outputs():
    book = load_notebook()
    assert book["nbformat"] == 4 and book["nbformat_minor"] >= 5
    assert book["metadata"]["kernelspec"]["name"] == "python3"
    assert len(book["cells"]) >= 10
    ids = [cell["id"] for cell in book["cells"]]
    assert len(ids) == len(set(ids))
    assert all(isinstance(cell_id, str) and cell_id for cell_id in ids)
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), "<notebook-cell>", "exec")


def test_user_visible_notebook_diagnostics_are_bilingual():
    demo = notebook_code()
    benchmark = (ROOT / "notebooks" / "multimodal_embeddings_benchmark.ipynb").read_text()
    assert "English: Polling interval must be 1–60 seconds" in demo
    assert "Tiếng Việt: Khoảng chờ polling" in demo
    assert "English: Polling window must be 1–3600 seconds" in demo
    assert "Tiếng Việt: Thời gian polling" in demo
    assert "English: Invalid async cache; do not overwrite without backing it up" in demo
    assert "Tiếng Việt: Cache async không hợp lệ" in demo
    assert "Mapping request ID cũ phải là đối tượng JSON" in demo
    assert (
        "English: Verify your actual Modal settings; displayed defaults are examples only."
        in benchmark
    )
    assert "Tiếng Việt: Kiểm tra cấu hình Modal thực tế" in benchmark
    assert "English: Both embedding models must be available in catalog" in benchmark
    assert "Tiếng Việt: Danh mục API phải có đủ hai model embedding" in benchmark
    assert "English: Missing embedding_metadata" in benchmark
    assert "Tiếng Việt: Thiếu embedding_metadata" in benchmark


def test_notebook_intro_is_fully_visible_in_both_languages():
    intro = "".join(load_notebook()["cells"][0]["source"])
    assert intro.startswith("# Multimodal Embeddings AI Inference — Demo\n\n")
    assert "### Tiếng Việt" in intro and "### English" in intro
    assert intro.index("### English") < intro.index("\n---\n") < intro.index("### Tiếng Việt")
    assert intro.count("1. **") == 2
    assert intro.count("2. **") == 2
    assert intro.count("3. **") == 2
    assert intro.count("4. **") == 2
    assert "không cần chọn ngôn ngữ" not in intro
    assert "no language selection" not in intro
    assert "<details>" not in intro


def test_bundled_previews_run_without_credentials_before_api_connection(monkeypatch):
    """Fresh readers must see the samples before being asked for a URL/token."""
    import getpass

    book = load_notebook()
    ids = [cell["id"] for cell in book["cells"]]
    assert ids.index("configure-code") < ids.index("sample-preparation-code")
    assert ids.index("sample-preparation-code") < ids.index("credentials-input")
    assert ids.index("credentials-input") < ids.index("http-verification-code")
    assert ids.index("http-verification-code") < ids.index("embedding-code")

    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(
        getpass, "getpass",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Unexpected token prompt")),
    )
    scope = {"__name__": "__main__"}
    output = io.StringIO()
    with redirect_stdout(output):
        for cell in book["cells"][:ids.index("credentials-explanation")]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), cell["id"], "exec"), scope)

    assert len(scope["text_samples"]) == len(scope["image_samples"]) == 3
    assert scope["SYNTHETIC_IMAGES"] == tuple(scope["image_samples"])
    assert "api_token" not in scope and "base_url" not in scope
    assert 'src="data:image/webp;base64,' in scope["builtin_preview_html"]()


def test_custom_sample_fields_and_environment_priority(monkeypatch, tmp_path):
    """A beginner can edit one pair without touching embedded-image code."""
    cells = {c["id"]: "".join(c["source"])
             for c in load_notebook()["cells"] if c["cell_type"] == "code"}
    ids = [c["id"] for c in load_notebook()["cells"]]
    assert ids.index("configure-code") < ids.index("sample-options-code")
    assert ids.index("sample-options-code") < ids.index("sample-preparation-code")
    assert ids.index("sample-preparation-code") < ids.index("credentials-input")
    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)

    scope = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-code"], "config", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
        exec(compile(cells["sample-options-code"], "samples", "exec"), scope)
        exec(compile(cells["sample-preparation-code"], "prepare", "exec"), scope)
    assert scope["default_pairs"] is True
    path = tmp_path / "my-authorized-image.webp"
    path.write_bytes(scope["SYNTHETIC_IMAGES"][0])

    custom = cells["sample-options-code"].replace(
        'SAMPLE_TEXT_1 = "a flat bright-green triangle on a cream background"',
        'SAMPLE_TEXT_1 = "my user-supplied description"',
    ).replace('IMAGE_PATH_1 = ""', 'IMAGE_PATH_1 = ' + repr(str(path)))
    assert custom != cells["sample-options-code"]
    with redirect_stdout(io.StringIO()):
        exec(compile(custom, "custom-samples", "exec"), scope)
        exec(compile(cells["sample-preparation-code"], "new-samples", "exec"), scope)
    assert scope["text_samples"][0] == "my user-supplied description"
    assert scope["image_samples"][0] == path.read_bytes()
    assert scope["text_sources"] == ["custom", "built-in", "built-in"]
    assert scope["image_sources"] == ["custom", "built-in", "built-in"]
    assert scope["default_pairs"] is False

    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_TEXT", "environment wins")
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["sample-preparation-code"], "env-priority", "exec"), scope)
    assert scope["text_samples"][0] == "environment wins"
    assert scope["text_sources"][0] == "custom"


def test_notebook_has_one_default_model_and_no_modal_management():
    code = notebook_code()
    assert 'MODEL_ID = "qwen3-vl-embedding-8b" #@param' in code
    assert "wemm-embedding-9b" in code
    assert "VISION_EMBEDDINGS_MODEL" in code
    assert "VISION_EMBEDDINGS_DIMENSION" in code
    assert "VISION_EMBEDDINGS_REQUEST_TIMEOUT" in code
    assert "VISION_EMBEDDINGS_BASE_URL" in code
    assert "VISION_EMBEDDINGS_API_TOKEN" in code
    assert "getpass(" in code
    for forbidden in (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "VISION_EMBEDDINGS_MODELS",
        "VISION_EMBEDDINGS_MODAL_DEPLOY",
        "VISION_EMBEDDINGS_MODAL_DOWNLOAD",
        "subprocess",
        "modal_cmd(",
    ):
        assert forbidden not in code


def test_notebook_matches_bilingual_guides():
    book = load_notebook()
    assert any(cell["cell_type"] == "markdown" for cell in book["cells"])
    if not (ROOT / "docs" / "NOTEBOOK_GUIDE.md").exists():
        pytest.skip("notebook guide unavailable in this checkout")
    left = (ROOT / "docs" / "NOTEBOOK_GUIDE.md").read_text(encoding="utf-8")
    right = (ROOT / "docs" / "NOTEBOOK_GUIDE.vi.md").read_text(encoding="utf-8")
    assert "bilingual English/Vietnamese" in left
    assert "song ngữ Anh–Việt" in right
    for guide in (left, right):
        for key in (
            "VISION_EMBEDDINGS_MODEL",
            "VISION_EMBEDDINGS_BASE_URL",
            "VISION_EMBEDDINGS_API_TOKEN",
            "VISION_EMBEDDINGS_DIMENSION",
            "VISION_EMBEDDINGS_REQUEST_TIMEOUT",
        ):
            assert key in guide
        assert "Model configuration" in guide
        assert "API connection" in guide


def test_bilingual_sections_are_always_visible_and_technically_aligned():
    book = load_notebook()
    ids = [cell["id"] for cell in book["cells"]]
    required = (
        "intro", "configure-options", "configure-code", "configure-helpers-code",
        "sample-options", "sample-options-code", "sample-preparation-explanation",
        "builtin-512px-three-image-preview", "sample-preparation-code",
        "credentials-explanation", "credentials-input",
        "http-verification-explanation", "http-verification-code",
        "embedding-explanation", "embedding-code", "results-explanation",
        "results-code", "notes",
    )
    assert len(ids) == len(set(ids))
    assert set(required) <= set(ids)
    assert [ids.index(cell_id) for cell_id in required] == sorted(
        ids.index(cell_id) for cell_id in required
    )
    for cell in book["cells"]:
        content = "".join(cell["source"])
        if cell["cell_type"] != "markdown" or cell["id"] == "builtin-512px-three-image-preview":
            continue
        assert "<details>" not in content and "</details>" not in content, cell["id"]
        assert content.count("\n---\n") == 1, cell["id"]
        en, vi = content.split("\n---\n")
        assert en.count("### English") == 1, cell["id"]
        assert vi.count("### Tiếng Việt") == 1, cell["id"]
        assert re.search(r"[ăâđêôơưĂÂĐÊÔƠƯạ-ỹ]", vi), cell["id"]
        assert not re.search(r"[ăâđêôơưĂÂĐÊÔƠƯạ-ỹ]", en.split("### English", 1)[1]), cell["id"]
        assert sorted(re.findall(r"`([^`]+)`", vi)) == sorted(re.findall(r"`([^`]+)`", en)), cell[
            "id"
        ]
        assert len(re.findall(r"^\s*\| .* \|$", vi, re.M)) == len(
            re.findall(r"^\s*\| .* \|$", en, re.M)
        ), cell["id"]
        assert len(re.findall(r"^- ", vi, re.M)) == len(re.findall(r"^- ", en, re.M)), cell["id"]

    code = notebook_code()
    assert "VISION_EMBEDDINGS_LANGUAGE" not in code
    assert "LANGUAGE =" not in code
    assert "show_section(" not in code
    assert "def say(" not in code
    assert "def message_pair(" in code
    assert "Tiếng Việt:" in code and "English:" in code
    assert "Tiếng Việt — Kết quả" in code and "English — Results" in code
    assert code.index("English — Results") < code.index("Tiếng Việt — Kết quả")

    documents = (
        ROOT / "README.md",
        ROOT / "README.vi.md",
        ROOT / "docs" / "NOTEBOOK_GUIDE.md",
        ROOT / "docs" / "NOTEBOOK_GUIDE.vi.md",
    )
    for document in documents:
        if not document.exists():
            continue  # Optional documentation is unavailable in this checkout.
        content = document.read_text(encoding="utf-8")
        assert "expandable" not in content and "mở rộng riêng" not in content
        assert not re.search(r"(?i)\bcell\s+[0-9]+\b", content), document.name


def test_three_embedded_generated_images_are_exact_api_inputs():
    """Code-embedded WebP previews survive Colab dropping notebook attachments."""
    from html import escape as escape_html

    book = load_notebook()
    preview = next(cell for cell in book["cells"]
                   if cell["id"] == "builtin-512px-three-image-preview")
    markdown = "".join(preview["source"])
    assert "512 × 512" in markdown
    assert "attachment:" not in markdown
    assert not preview.get("attachments")

    source = next(cell for cell in book["cells"] if cell["id"] == "sample-preparation-code")
    prelude = "".join(source["source"]).split("TEXT_KEYS =", 1)[0]
    scope = {"escape_html": escape_html}
    exec(compile(prelude, "<embedded-samples>", "exec"), scope)
    decoded = scope["SYNTHETIC_IMAGES"]
    assert len(decoded) == len(set(decoded)) == 3
    assert all(data[:4] == b"RIFF" and data[8:12] == b"WEBP" for data in decoded)
    html = scope["builtin_preview_html"]()
    assert html.count("<img ") == 3
    assert html.count("data:image/webp;base64,") == 3
    for encoded, image in zip(scope["EMBEDDED_WEBP_BASE64"], decoded, strict=True):
        assert base64.b64decode(encoded) == image
        assert f'data:image/webp;base64,{encoded}' in html
    assert "Tam giác xanh 2D" in html and "Hộp quà màu cam" in html
    assert "image_samples" not in prelude  # Do not preview private custom images.
    assert "display(HTML(builtin_preview_html()))" in "".join(source["source"])


@pytest.mark.parametrize(("selected_model", "instruction_mode"), [
    (MODELS[0], "default"), (MODELS[0], "custom"), (MODELS[1], "default"),
])
def test_notebook_calls_only_selected_model_via_http(monkeypatch, selected_model, instruction_mode):
    token = "not-a-real-token-" + "x" * 32
    base_url = "http://127.0.0.1:18767"
    client = TestClient(create_app(Settings(api_token=token, backend="mock")))
    calls = []

    class ClientOnlyOpener:
        def open(self, request, timeout=None):
            path = urlsplit(request.full_url).path
            body = request.data
            headers = dict(request.header_items())
            if body is not None:
                calls.append(json.loads(body.decode("utf-8")))
            response = client.request(
                request.get_method(),
                path,
                headers=headers,
                content=body,
            )
            assert response.status_code == 200, (path, response.status_code)
            return io.BytesIO(response.content)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: ClientOnlyOpener())
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", base_url)
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", token)
    monkeypatch.setenv("VISION_EMBEDDINGS_REQUIRE_REAL", "false")
    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", selected_model)
    monkeypatch.setenv("VISION_EMBEDDINGS_INSTRUCTION_MODE", instruction_mode)
    monkeypatch.setenv(
        "VISION_EMBEDDINGS_CUSTOM_INSTRUCTION", "  Retrieve matching sample images.  "
    )
    monkeypatch.delenv("VISION_EMBEDDINGS_IMAGE_PATH", raising=False)

    scope = {"__name__": "__main__"}
    output = io.StringIO()
    with redirect_stdout(output):
        for index, cell in enumerate(load_notebook()["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"<cell-{index}>", "exec"), scope)

    assert len(calls) == 6
    assert all(call["model"] == selected_model for call in calls)
    if instruction_mode == "custom":
        assert all(call.get("instruction") == "Retrieve matching sample images." for call in calls)
    else:
        assert all("instruction" not in call for call in calls)
    assert all(
        "embedding_metadata" in result
        for result in scope["text_results"] + scope["image_results"]
    )
    assert len({result["embedding_metadata"]["config_sha256"]
                for result in scope["text_results"] + scope["image_results"]}) == 1
    assert [("text" in call, "image_base64" in call) for call in calls] == [
        (True, False), (False, True),
        (True, False), (False, True),
        (True, False), (False, True),
    ]
    assert len({call["text"] for call in calls if "text" in call}) == 3
    assert len({call["image_base64"] for call in calls if "image_base64" in call}) == 3
    assert len(scope["summary"]) == 3
    assert len(scope["similarity_matrix"]) == 3
    assert all(len(row) == 3 for row in scope["similarity_matrix"])
    for index, item in enumerate(scope["summary"]):
        assert item["cosine_text_image"] == scope["similarity_matrix"][index][index]
        assert math.isclose(item["text_norm"], 1.0, abs_tol=0.001)
        assert math.isclose(item["image_norm"], 1.0, abs_tol=0.001)
    assert scope["summary"][0]["model"] == selected_model
    output_text = output.getvalue()
    if "English — Results" in output_text:
        assert "Passed:" in output_text
        assert "Đạt:" in output_text
        assert output_text.index("English — Results") < output_text.index(
            "Tiếng Việt — Kết quả"
        )
    else:
        assert "IPython.core.display.HTML" in output_text
        assert "English — Results" in scope["report_html"]
        assert "Tiếng Việt — Kết quả" in scope["report_html"]
    expected = scope["SYNTHETIC_IMAGES"][0]
    assert base64.b64decode(calls[1]["image_base64"]) == expected
    assert base64.b64decode(calls[3]["image_base64"]) == scope["SYNTHETIC_IMAGES"][1]
    assert base64.b64decode(calls[5]["image_base64"]) == scope["SYNTHETIC_IMAGES"][2]
    assert token not in output.getvalue()
    assert base_url not in output.getvalue()
    report = scope["report_html"]
    assert report.index("English — Results") < report.index("Tiếng Việt — Kết quả")
    assert "All nine text ↔ image cosine scores" in report
    assert "Chín điểm cosine văn bản" in report
    assert report.count("<table ") == 2  # Semantic English and Vietnamese matrices
    assert report.count('scope="col"') == 8
    assert report.count('scope="row"') == 6
    assert "Matching pair 3" in report
    assert "not a percentage" in report
    assert "không phải tỷ lệ phần trăm" in report
    assert "1,024" in report
    assert "No raw vectors, private image contents" in report
    assert "Không hiển thị vector thô" in report
    assert token not in report
    assert base_url not in report
    assert scope["completed_run_config"]["instruction_mode"] == instruction_mode
    assert scope["completed_run_config"]["config_sha256"] == (
        scope["text_results"][0]["embedding_metadata"]["config_sha256"]
    )

    if selected_model == MODELS[0] and instruction_mode == "default":
        # A user can change only the configuration cell and then rerun the
        # report cell. Existing vectors must not be described as custom.
        cells = {cell["id"]: "".join(cell["source"]) for cell in load_notebook()["cells"]}
        old_config = dict(scope["completed_run_config"])
        old_report = scope["report_html"]
        monkeypatch.setenv("VISION_EMBEDDINGS_INSTRUCTION_MODE", "custom")
        exec(compile(cells["configure-code"], "<rerun-config>", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
        with pytest.raises(RuntimeError, match="Configuration changed since inference"):
            exec(compile(cells["results-code"], "<stale-report>", "exec"), scope)
        assert len(calls) == 6
        assert scope["report_html"] == old_report
        assert scope["completed_run_config"] == old_config

        # Re-executing the embedding cell produces a new, attributable report.
        exec(compile(cells["embedding-code"], "<rerun-embeddings>", "exec"), scope)
        exec(compile(cells["results-code"], "<fresh-report>", "exec"), scope)
        assert len(calls) == 12
        assert all(call.get("instruction") == "Retrieve matching sample images."
                   for call in calls[6:])
        assert scope["completed_run_config"]["instruction_mode"] == "custom"
        assert scope["completed_run_config"]["config_sha256"] != old_config["config_sha256"]
        assert "custom" in scope["report_html"]

        # Changing only the custom text also invalidates the prior report.
        monkeypatch.setenv(
            "VISION_EMBEDDINGS_CUSTOM_INSTRUCTION", "Retrieve another image description."
        )
        exec(compile(cells["configure-code"], "<rerun-custom-text>", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
        with pytest.raises(RuntimeError, match="Configuration changed since inference"):
            exec(compile(cells["results-code"], "<stale-custom-report>", "exec"), scope)
        assert len(calls) == 12


def test_beginner_report_escapes_untrusted_metadata_and_explains_score(capsys):
    book = load_notebook()
    src = next("".join(c["source"]) for c in book["cells"] if c["id"] == "results-code")
    response = {"elapsed_ms": 20025.488, "_client_elapsed_ms": 21000, "timing": None,
                "embedding_metadata": {"config_sha256": "a" * 64}}
    sample = {
        "model": "qwen3-vl-embedding-8b",
        "revision": '<script>alert("unsafe")</script>',
        "backend": "mock",
        "dimension": 1024,
        "text_norm": 1.0,
        "image_norm": 1.0,
        "cosine_text_image": 0.77275,
        "text_elapsed_ms": 20025.488,
        "image_elapsed_ms": 1147.72,
    }
    scope = {
        "math": math,
        "hashlib": __import__("hashlib"),
        "message_pair": lambda en, vi: "English: " + en + "\nTiếng Việt: " + vi,
        "model_id": "qwen3-vl-embedding-8b",
        "dimension": 1024,
        "instruction_mode": "default",
        "custom_instruction": "",
        "completed_run_config": {
            "model": "qwen3-vl-embedding-8b", "dimension": 1024,
            "instruction_mode": "default", "custom_instruction_sha256": None,
            "config_sha256": "a" * 64,
        },
        "default_pairs": True,
        "sample_labels": ["", "", ""],
        "text_sources": ["built-in"] * 3,
        "image_sources": ["built-in"] * 3,
        "text_results": [dict(response) for _ in range(3)],
        "image_results": [dict(response, elapsed_ms=1147.72) for _ in range(3)],
        "summary": [dict(sample, sample=i + 1) for i in range(3)],
        "similarity_matrix": [[0.77275, 0.2, -0.1],
                              [0.3, 0.77275, 0.1],
                              [-0.3, 0.5, 0.77275]],
    }
    seed_report_input_fixture(scope)
    scope["configuration_ready"] = True  # Isolated helper fixture has valid form config.
    exec(compile(src, "<friendly-report>", "exec"), scope)
    html = scope["report_html"]
    assert "English — Results" in html
    assert html.index("English — Results") < html.index("Tiếng Việt — Kết quả")
    assert html.count("0.7728") >= 6  # Three matched pairs, two languages.
    assert "not a percentage" in html
    assert "không phải tỷ lệ phần trăm" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "SIMULATION COMPLETE" in html
    assert "MÔ PHỎNG HOÀN TẤT" in html
    assert html.count("<table ") == 2
    assert html.count("<caption ") == 2
    assert 'role="grid"' not in html
    assert "60.08 s" in html  # Three server processing times summed.
    assert "3.44 s" in html
    output = capsys.readouterr().out
    if "English — Results" in output:
        assert output.index("English — Results") < output.index("Tiếng Việt — Kết quả")
    elif output:
        assert "IPython.core.display.HTML" in output


def test_every_stage_has_clear_bilingual_accessible_output():
    book = load_notebook()
    code_cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    cells_by_id = {cell["id"]: "".join(cell["source"]) for cell in code_cells}
    stage_ids = {
        "configure-helpers-code", "sample-preparation-code", "credentials-input",
        "http-verification-code", "embedding-code",
    }
    assert stage_ids <= cells_by_id.keys()
    assert {"sample-options-code", "results-code"} <= cells_by_id.keys()
    for cell_id in stage_ids:
        assert "show_stage(" in cells_by_id[cell_id], cell_id

    setup = cells_by_id["configure-helpers-code"]
    assert "def stage_markup(" in setup
    # The HTML markup in the first code cell is used by every subsequent stage.
    assert "background:#ffffff !important" in setup
    assert "color:#15283b !important" in setup
    assert "font-size:19px" in setup
    assert 'lang="en"' in setup
    assert 'lang="vi"' in setup
    assert 'role="status"' in setup
    assert "escape_html" in setup


def test_stage_card_escapes_metadata_and_dark_theme_is_legible(monkeypatch):
    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", "qwen3-vl-embedding-8b")
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]}
    scope = {}
    exec(compile(cells["configure-code"], "<form>", "exec"), scope)
    exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
    exec(compile(cells["configure-helpers-code"], "<setup>", "exec"), scope)
    markup = scope["stage_markup"](
        "3 / 5", "<script>unsafe</script>", "Connection OK", "Đã kết nối", "Thành công"
    )
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert markup.index('lang="en"') < markup.index('lang="vi"')
    assert "background:#ffffff !important" in markup
    assert "color:#15283b !important" in markup
    assert "font-size:15px" in markup
    assert "Thành công" in markup


def test_final_report_semantic_matrix_has_explicit_dark_mode_contrast():
    source = next(
        "".join(c["source"]) for c in load_notebook()["cells"] if c["id"] == "results-code"
    )
    assert "def detail_row(" in source
    assert "<table " in source and "<thead><tr>" in source and "</tbody></table>" in source
    assert "<caption " in source and 'scope="row"' in source and 'scope="col"' in source
    assert "background:#ffffff !important" in source
    assert "color:#15283b !important" in source
    assert "font-size:27px" in source
    assert "font-size:15px" in source
    assert "overflow-wrap:anywhere" in source
    assert 'role="region"' in source


def test_demo_has_no_benchmark_or_cross_model_requests():
    book = load_notebook()
    source = notebook_code()
    assert "BENCHMARK_MODE" not in source
    assert "VISION_EMBEDDINGS_BENCHMARK" not in source
    assert "benchmark_enabled(" not in source
    assert not any("performance-benchmark" in cell["id"] for cell in book["cells"])
    assert "multimodal_embeddings_benchmark.ipynb" in "".join(
        "".join(cell["source"]) for cell in book["cells"] if cell["id"] == "notes"
    )


def test_timing_breakdown_is_optional_for_old_server():
    book = load_notebook()
    result_source = next("".join(c["source"]) for c in book["cells"] if c["id"] == "results-code")
    assert "timing" in result_source
    assert "gpu_dispatch_overhead_ms" in result_source
    assert "not an isolated cold-start measurement" in result_source


def test_demo_sends_three_distinct_inputs_per_modality_with_no_benchmark_mode():
    code = notebook_code()
    assert "for index, (text, image) in enumerate(" in code
    assert "text_results.append(" in code
    assert "image_results.append(" in code
    assert "similarity_matrix" in code
    assert "for index in range(3)" in code
    assert "BENCHMARK_MODE" not in code


@pytest.mark.parametrize(
    "cell_id",
    (
        "intro",
        "configure-options",
        "credentials-explanation",
        "http-verification-explanation",
        "sample-preparation-explanation",
        "embedding-explanation",
        "results-explanation",
        "notes",
    ),
)
def test_all_bilingual_markdown_sections_have_matching_structure(cell_id):
    """Keep the two languages parallel when editing explanatory notebook text."""
    cell = next(item for item in load_notebook()["cells"] if item["id"] == cell_id)
    text = "".join(cell["source"])
    assert text.count("\n---\n") == 1
    english, vietnamese = text.split("\n---\n")
    assert english.count("### English") == 1
    assert vietnamese.count("### Tiếng Việt") == 1

    def line_types(section):
        kinds = []
        body = section.split("### English", 1)[1] if "### English" in section else (
            section.split("### Tiếng Việt", 1)[1]
        )
        for line in body.strip().splitlines():
            trimmed = line.strip()
            if not trimmed:
                kinds.append("blank")
            elif trimmed.startswith("|"):
                kinds.append("table")
            elif re.match(r"^\d+\. ", trimmed):
                kinds.append("ordered")
            elif trimmed.startswith("- "):
                kinds.append("list")
            elif trimmed.startswith("**"):
                kinds.append("label")
            else:
                kinds.append("paragraph")
        return kinds

    assert line_types(english) == line_types(vietnamese)
    assert chr(92) + chr(96) not in text
    if cell_id == "configure-options":
        def defaults(section):
            return [
                (line.split("|")[1].strip(), line.split("|")[2].strip().replace("trống", "empty"))
                for line in section.splitlines()
                if line.startswith("| ") and "VISION_EMBEDDINGS_" in line
            ]

        assert defaults(english) == defaults(vietnamese)


def test_preview_markdown_explains_runtime_images_in_both_languages():
    preview = next(
        cell for cell in load_notebook()["cells"]
        if cell["id"] == "builtin-512px-three-image-preview"
    )
    markdown = "".join(preview["source"])
    assert "### English" in markdown and "### Tiếng Việt" in markdown
    assert "512 × 512" in markdown
    assert "2D green triangle" in markdown
    assert "tam giác xanh 2D" in markdown
    assert "hình cầu xanh 3D" in markdown
    assert "attachment:" not in markdown
    assert not preview.get("attachments")


def test_user_facing_full_model_names():
    import json
    import re
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "notebooks"
    path /= "multimodal_embeddings_api_demo.ipynb"
    data = json.loads(path.read_text())
    md = "\n".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "markdown")
    assert "Qwen3-VL-Embedding-8B" in md
    assert "Tencent WeMM-Embedding-9B" in md
    assert not re.search(r"(?<![\w/])(?:Qwen|WeMM)(?![\w/-])", md)


def test_bundled_images_source_is_compact_and_collapsible():
    """Built-in images must not create unbounded horizontal scrolling."""
    cell = next(c for c in load_notebook()["cells"] if c["id"] == "sample-preparation-code")
    source = "".join(cell["source"])
    assert max(map(len, source.splitlines())) <= 130
    assert cell["metadata"].get("jupyter", {}).get("source_hidden") is True
    assert source.count("EMBEDDED_WEBP_BASE64 = (") == 1


def test_whitespace_only_custom_instruction_fails_before_api(monkeypatch):
    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", "qwen3-vl-embedding-8b")
    monkeypatch.setenv("VISION_EMBEDDINGS_INSTRUCTION_MODE", "custom")
    monkeypatch.setenv("VISION_EMBEDDINGS_CUSTOM_INSTRUCTION", "    ")
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]}
    scope = {}
    exec(compile(cells["configure-code"], "configure-code", "exec"), scope)
    with pytest.raises(ValueError, match="Instruction must be one line"):
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)


def test_bilingual_demo_describes_conditional_gpu_post_count():
    """Six evaluated embeddings must not be advertised as six unconditional POSTs."""
    cells = {cell["id"]: "".join(cell["source"]) for cell in load_notebook()["cells"]}
    intro = cells["intro"]
    setup = cells["configure-options"]
    inference = cells["embedding-explanation"]
    all_markdown = "\n".join(
        "".join(cell["source"]) for cell in load_notebook()["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "up to six new GPU POSTs" in intro
    assert "tối đa sáu GPU POST mới" in intro
    assert "up to six new GPU POSTs" in setup
    assert "tối đa sáu GPU POST mới" in setup
    assert "up to six new GPU POSTs" in inference
    assert "tối đa sáu GPU POST mới" in inference
    assert "six sequential POST requests" not in all_markdown
    assert "sáu POST tuần tự" not in all_markdown
    assert "APPROVE_NEW_GPU_JOBS = False" in cells["embedding-code"]




@pytest.mark.parametrize("partial_env", [None, "url", "token"])
def test_full_run_all_preview_only_without_credentials(monkeypatch, partial_env):
    """A fresh Kaggle Save & Run All must finish without prompts, HTTP or fake results."""
    import getpass

    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)
    if partial_env == "url":
        monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://private.example.invalid")
    if partial_env == "token":
        monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "offline-valid-test-token")

    def block_prompt(*_args, **_kwargs):
        raise AssertionError("Preview-only Run All requested a credential prompt")

    class BlockNetwork:
        def open(self, *_args, **_kwargs):
            raise AssertionError("Preview-only Run All made an HTTP request")

    monkeypatch.setattr(getpass, "getpass", block_prompt)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: BlockNetwork())
    scope = {"__name__": "__main__"}
    book = load_notebook()
    with redirect_stdout(io.StringIO()):
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), cell["id"], "exec"), scope)

    assert scope["preview_only_mode"] is True
    assert scope["connection_ready"] is False
    assert scope["verified_connection_identity"] is None
    assert scope["completed_run_config"] is None
    assert scope["report_html"] is None
    assert len(scope["text_samples"]) == len(scope["image_samples"]) == 3
    assert scope["SYNTHETIC_IMAGES"] == tuple(scope["image_samples"])


def test_user_can_explicitly_enable_hidden_api_prompts(monkeypatch):
    """Explicit RUN_API_DEMO opt-in may request secrets, but not inference yet."""
    import getpass

    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)
    prompts = iter(["https://private.example.invalid", "offline-valid-test-token"])
    monkeypatch.setattr(getpass, "getpass", lambda *_args, **_kwargs: next(prompts))
    cells = {
        c["id"]: "".join(c["source"])
        for c in load_notebook()["cells"] if c["cell_type"] == "code"
    }
    scope = {"__name__": "__main__"}
    with redirect_stdout(io.StringIO()):
        exec(cells["configure-code"], scope)
        scope["RUN_API_DEMO"] = True
        exec(cells["configure-helpers-code"], scope)
        exec(cells["credentials-input"], scope)
    assert scope["preview_only_mode"] is False
    assert scope["connection_ready"] is True
    assert scope["base_url"] == "https://private.example.invalid"
    assert scope["api_token"] == "offline-valid-test-token"
    assert scope["verified_connection_identity"] is None
