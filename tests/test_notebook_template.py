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
    import hashlib

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


@pytest.fixture
def verified_notebook_session(monkeypatch):
    """Offline authenticated client; no real network, token or GPU."""
    token = "offline-secure-reconnection-token"
    first_url = "https://first.example.invalid"
    second_url = "https://second.example.invalid"
    client = TestClient(create_app(Settings(api_token=token, backend="mock")))
    requests = []

    class LocalTransport:
        def open(self, request, timeout=None):
            requests.append((request.full_url, request.get_method()))
            response = client.request(
                request.get_method(), urlsplit(request.full_url).path,
                headers=dict(request.header_items()), content=request.data,
            )
            assert response.status_code == 200
            return io.BytesIO(response.content)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: LocalTransport())
    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", first_url)
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", token)
    monkeypatch.setenv("VISION_EMBEDDINGS_REQUIRE_REAL", "false")
    cells = {
        cell["id"]: "".join(cell["source"]) for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    }
    scope = {}
    with redirect_stdout(io.StringIO()):
        for cell_id in (
            "configure-code", "configure-helpers-code",
            "sample-options-code", "sample-preparation-code",
            "credentials-input", "http-verification-code", "embedding-code",
        ):
            exec(compile(cells[cell_id], cell_id, "exec"), scope)
    assert len([r for r in requests if r[1] == "POST"]) == 6
    assert all(r[0].startswith(first_url) for r in requests)
    return scope, cells, requests, token, second_url


@pytest.mark.parametrize("change", ["form", "failed_preparation", "environment", "image_file"])
def test_report_blocks_unprepared_or_changed_sample_selection(
    monkeypatch, verified_notebook_session, tmp_path, change,
):
    """Results must not publish old success after an input edit or preparation error."""
    scope, cells, requests, _token, _other_url = verified_notebook_session
    if change == "image_file":
        path = tmp_path / "authorized.webp"
        path.write_bytes(scope["SYNTHETIC_IMAGES"][0])
        monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH", str(path))
        with redirect_stdout(io.StringIO()):
            exec(cells["sample-preparation-code"], scope)
            exec(cells["embedding-code"], scope)
    with redirect_stdout(io.StringIO()):
        exec(cells["results-code"], scope)
    previous_report = scope["report_html"]
    request_count = len(requests)
    if change == "form":
        edited = cells["sample-options-code"].replace(
            'SAMPLE_TEXT_1 = "a flat bright-green triangle on a cream background"',
            'SAMPLE_TEXT_1 = "a red car parked beside a river"',
        )
        exec(edited, scope)
        assert scope["sample_prepared"] is False
    elif change == "failed_preparation":
        monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH_3", str(tmp_path / "missing.webp"))
        with redirect_stdout(io.StringIO()), pytest.raises(ValueError, match="Image 3"):
            exec(cells["sample-preparation-code"], scope)
        assert scope["sample_prepared"] is False
    elif change == "environment":
        monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_TEXT", "a newly selected description")
    else:
        path.write_bytes(scope["SYNTHETIC_IMAGES"][1])
    with redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="Prepare input samples"):
        exec(cells["results-code"], scope)
    assert scope["report_html"] == previous_report
    assert len(requests) == request_count


def test_invalid_reconnect_revokes_token_and_blocks_all_inference(
    monkeypatch, verified_notebook_session,
):
    """The old preflight cannot be reused after an invalid URL or token."""
    scope, cells, requests, token, second_url = verified_notebook_session
    baseline = len(requests)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "http://untrusted.example.invalid")
    with pytest.raises(ValueError, match="HTTPS API endpoint required"):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["credentials-input"], "rejected-url", "exec"), scope)
    assert scope["connection_ready"] is False
    assert scope["verified_model_id"] is None
    assert scope["verified_connection_identity"] is None
    assert scope["configured_connection_identity"] is None
    assert scope["base_url"] is None
    assert scope["api_token"] is None
    with pytest.raises(RuntimeError):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "after-bad-url", "exec"), scope)
    assert len(requests) == baseline  # No HTTP sent to the rejected endpoint.

    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", second_url)
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["credentials-input"], "new-credentials", "exec"), scope)
    with pytest.raises(RuntimeError, match="Model changed or API not checked"):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "before-preflight", "exec"), scope)
    assert len(requests) == baseline
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["http-verification-code"], "new-preflight", "exec"), scope)
        exec(compile(cells["embedding-code"], "verified-inference", "exec"), scope)
    assert len([r for r in requests if r[1] == "POST"]) == 12
    assert all(r[0].startswith(second_url) for r in requests[baseline:])

    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "short")
    with pytest.raises(ValueError, match="Bearer token"):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["credentials-input"], "rejected-token", "exec"), scope)
    with pytest.raises(RuntimeError):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "after-bad-token", "exec"), scope)
    assert len([r for r in requests if r[1] == "POST"]) == 12


def test_editing_or_cancelling_verified_connection_sends_zero_posts(
    monkeypatch, verified_notebook_session,
):
    """HTTPS URL/token edits or interrupted prompts must fail before HTTP."""
    scope, cells, requests, token, _ = verified_notebook_session
    baseline = len(requests)
    scope["base_url"] = "https://new-unverified.example.invalid"
    with pytest.raises(RuntimeError, match="connection"):
        scope["api_request"]("/v1/embeddings", {"model": scope["model_id"]})
    with pytest.raises(RuntimeError):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "edited-url", "exec"), scope)
    assert len(requests) == baseline

    scope["base_url"] = "https://first.example.invalid"
    scope["api_token"] = "another-otherwise-valid-test-token"
    with pytest.raises(RuntimeError, match="connection"):
        scope["api_request"]("/v1/embeddings", {"model": scope["model_id"]})
    with pytest.raises(RuntimeError):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "edited-token", "exec"), scope)
    assert len(requests) == baseline

    scope["api_token"] = token
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["embedding-code"], "restored-connection", "exec"), scope)
    assert len([r for r in requests if r[1] == "POST"]) == 12

    monkeypatch.delenv("VISION_EMBEDDINGS_BASE_URL")
    def interrupt_prompt(_prompt):
        raise KeyboardInterrupt()
    monkeypatch.setattr("getpass.getpass", interrupt_prompt)
    with pytest.raises(KeyboardInterrupt):
        exec(compile(cells["credentials-input"], "interrupted-credentials", "exec"), scope)
    assert scope["connection_ready"] is False
    assert scope["api_token"] is None
    assert scope["base_url"] is None
    with pytest.raises(RuntimeError):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "after-interruption", "exec"), scope)
    assert len([r for r in requests if r[1] == "POST"]) == 12


def test_bilingual_demo_restart_table():
    book = load_notebook()
    intro = "".join(book["cells"][0]["source"])
    assert "| Changed | Rerun |" in intro
    assert "| Đã thay đổi | Chạy lại |" in intro
    assert intro.count("**Secure API connection → Check API connection") == 2
    assert intro.count("**Customize samples → Prepare input samples") == 2
    assert intro.count("**Model configuration → Configuration helpers → Prepare input samples") == 2
    assert "do not automatically retry an already sent POST" in intro


def test_quickstart_explains_development_setup():
    en = ROOT / "README.md"
    vi = ROOT / "README.vi.md"
    if not en.exists() or not vi.exists():
        pytest.skip("Root README files unavailable in this checkout")
    en_text, vi_text = en.read_text(), vi.read_text()
    assert "**What setup does:**" in en_text
    assert "**Lệnh setup làm gì:**" in vi_text
    for doc in (en_text, vi_text):
        assert "scripts/setup.sh" in doc
        assert "scripts/quickstart_mock.sh" in doc
        assert "pytest" in doc and "Ruff" in doc


def test_effective_source_disclosure_hides_sample_values(monkeypatch, tmp_path):
    """The model/dimension source and sample overrides are visible, not raw inputs."""
    import sys
    from types import ModuleType, SimpleNamespace

    # Only the display contract is under test. Simulate a notebook renderer
    # so a fresh dev install does not require an unlisted IPython dependency.
    shown = []
    fake_ipython = ModuleType("IPython")
    fake_display = ModuleType("IPython.display")
    fake_display.HTML = lambda html: SimpleNamespace(data=html)
    fake_display.display = lambda item: shown.append(item.data)
    fake_ipython.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    with redirect_stdout(io.StringIO()):
        for cell in (
            "configure-code", "configure-helpers-code",
            "sample-options-code", "sample-preparation-code",
        ):
            exec(compile(cells[cell], cell, "exec"), scope)
    default_output = "\n".join(map(str, shown))
    assert "form: MODEL_ID" in default_output
    assert "form: DIMENSION" in default_output
    assert "No sample environment overrides" in default_output

    text_sample = "example-sample-value-r5"
    label = "example-label-value-r5"
    image_path = tmp_path / "example-private-image-r5.webp"
    image_path.write_bytes(scope["SYNTHETIC_IMAGES"][0])
    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", "wemm-embedding-9b")
    monkeypatch.setenv("VISION_EMBEDDINGS_DIMENSION", "512")
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_TEXT", text_sample)
    monkeypatch.setenv("VISION_EMBEDDINGS_IMAGE_PATH_2", str(image_path))
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_LABEL_3", label)
    shown.clear()
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-code"], "config-r5", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "configure-helpers-code", "exec"), scope)
        exec(compile(cells["sample-preparation-code"], "samples-r5", "exec"), scope)
    output = "\n".join(map(str, shown))
    assert scope["model_id"] == "wemm-embedding-9b"
    assert scope["dimension"] == 512
    assert scope["text_samples"][0] == text_sample
    assert scope["image_samples"][1] == image_path.read_bytes()
    assert scope["sample_labels"][2] == label
    for key in (
        "VISION_EMBEDDINGS_MODEL",
        "VISION_EMBEDDINGS_DIMENSION",
        "VISION_EMBEDDINGS_SAMPLE_TEXT",
        "VISION_EMBEDDINGS_IMAGE_PATH_2",
        "VISION_EMBEDDINGS_SAMPLE_LABEL_3",
    ):
        assert key in output
    assert "environment override" in output
    assert "biến môi trường" in output
    for value in (text_sample, label, str(image_path)):
        assert value not in output


def test_rerun_table_uses_actual_editable_and_helper_cell_titles():
    cells = {cell["id"]: "".join(cell["source"]) for cell in load_notebook()["cells"]}
    for cell_id, label in (
        ("configure-code", "Model configuration"),
        ("sample-options-code", "Customize samples"),
        ("sample-preparation-code", "Prepare input samples"),
        ("credentials-input", "Secure API connection"),
        ("http-verification-code", "Check API connection"),
        ("embedding-code", "Generate embeddings"),
        ("results-code", "Results"),
    ):
        assert label in cells["intro"], label
        assert label in cells[cell_id].splitlines()[0], cell_id
    assert "Helpers — prepare built-in images" not in cells["intro"]
    assert "Helpers — validate and render results" not in cells["intro"]


def test_config_helper_is_collapsed_and_form_only_contains_settings():
    cells = {cell["id"]: cell for cell in load_notebook()["cells"]}
    form = "".join(cells["configure-code"]["source"])
    helper = "".join(cells["configure-helpers-code"]["source"])
    assert len(form.splitlines()) <= 16
    assert form.count("#@param") == 5
    assert "def " not in form and "stage_markup" not in form
    assert "configuration_ready = False" in form
    assert cells["configure-helpers-code"]["metadata"]["jupyter"]["source_hidden"] is True
    assert "def stage_markup(" in helper
    assert "def validate_api_token(" in helper
    assert "configuration_ready = True" in helper
    ids = [c["id"] for c in load_notebook()["cells"]]
    assert ids.index("configure-helpers-code") == ids.index("configure-code") + 1


def test_markdown_insertions_allowed_while_required_cell_order_is_enforced(monkeypatch):
    """Additional bilingual documentation is safe; absent/reordered required cells aren't."""
    import copy
    import sys

    book = copy.deepcopy(load_notebook())
    extra = {
        "id": "optional-reader-note",
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Optional note / Ghi chú tùy chọn\n\n",
            "### English\n\n",
            "An additional explanatory cell does not affect requests.\n\n",
            "---\n\n",
            "### Tiếng Việt\n\n",
            "Một cell giải thích bổ sung không làm thay đổi request.\n",
        ],
    }
    index = next(i for i, c in enumerate(book["cells"]) if c["id"] == "notes")
    book["cells"].insert(index, extra)
    monkeypatch.setattr(sys.modules[__name__], "load_notebook", lambda: book)
    test_bilingual_sections_are_always_visible_and_technically_aligned()

    absent = copy.deepcopy(book)
    absent["cells"] = [c for c in absent["cells"] if c["id"] != "credentials-input"]
    monkeypatch.setattr(sys.modules[__name__], "load_notebook", lambda: absent)
    with pytest.raises(AssertionError):
        test_bilingual_sections_are_always_visible_and_technically_aligned()

    wrong_order = copy.deepcopy(book)
    ids = [c["id"] for c in wrong_order["cells"]]
    a, b = ids.index("configure-helpers-code"), ids.index("credentials-input")
    wrong_order["cells"][a], wrong_order["cells"][b] = (
        wrong_order["cells"][b], wrong_order["cells"][a]
    )
    monkeypatch.setattr(sys.modules[__name__], "load_notebook", lambda: wrong_order)
    with pytest.raises(AssertionError):
        test_bilingual_sections_are_always_visible_and_technically_aligned()


def test_rerunning_only_visible_form_blocks_stale_http_and_report(monkeypatch):
    """A changed form must not silently reuse the previous helper/preflight state."""
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}

    def forbid_network(*args, **kwargs):
        raise AssertionError("HTTP attempted after an incomplete model edit")

    monkeypatch.setattr(urllib.request, "build_opener", forbid_network)
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-code"], "form", "exec"), scope)
        exec(compile(cells["configure-helpers-code"], "helper", "exec"), scope)
    assert scope["configuration_ready"] is True

    exec(compile(cells["configure-code"], "form-rerun", "exec"), scope)
    assert scope["configuration_ready"] is False
    for cell_id in ("http-verification-code", "embedding-code", "results-code"):
        with pytest.raises(RuntimeError, match="Configuration helpers"):
            exec(compile(cells[cell_id], cell_id, "exec"), scope)

    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-helpers-code"], "helper-rerun", "exec"), scope)
    assert scope["configuration_ready"] is True


@pytest.mark.parametrize(
    ("variable", "invalid"),
    [
        ("VISION_EMBEDDINGS_DIMENSION", "invalid-dimension"),
        ("VISION_EMBEDDINGS_REQUEST_TIMEOUT", "invalid-timeout"),
        ("VISION_EMBEDDINGS_MODEL", "not-a-supported-model"),
        ("VISION_EMBEDDINGS_INSTRUCTION_MODE", "not-a-valid-mode"),
    ],
)
def test_failed_configuration_helper_revokes_stale_state_and_can_recover(
    monkeypatch, verified_notebook_session, variable, invalid,
):
    """F03: a failed helper rerun must not send any more inference POSTs."""
    scope, cells, requests, _token, _other_url = verified_notebook_session
    before_requests = len(requests)
    before_posts = sum(method == "POST" for _, method in requests)
    before_config = {
        key: scope[key] for key in
        ("model_id", "dimension", "timeout", "instruction_mode",
         "custom_instruction", "image_limit")
    }
    assert scope["configuration_ready"] is True
    assert before_posts == 6
    monkeypatch.setenv(variable, invalid)
    with pytest.raises(ValueError):
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["configure-helpers-code"], "invalid-helper", "exec"), scope)
    assert scope["configuration_ready"] is False
    assert {key: scope[key] for key in before_config} == before_config
    for cell_id in ("http-verification-code", "embedding-code", "results-code"):
        with pytest.raises(RuntimeError, match="Configuration helpers") as error:
            exec(compile(cells[cell_id], "blocked-" + cell_id, "exec"), scope)
        assert "English:" in str(error.value)
        assert "Tiếng Việt:" in str(error.value)
    assert len(requests) == before_requests

    monkeypatch.delenv(variable)
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["configure-helpers-code"], "recovered-helper", "exec"), scope)
        assert scope["configuration_ready"] is True
        exec(compile(cells["http-verification-code"], "recovered-preflight", "exec"), scope)
        exec(compile(cells["embedding-code"], "recovered-inference", "exec"), scope)
        exec(compile(cells["results-code"], "recovered-report", "exec"), scope)
    assert sum(method == "POST" for _, method in requests) == before_posts + 6
    assert "English — Results" in scope["report_html"]
    assert "Tiếng Việt — Kết quả" in scope["report_html"]


@pytest.mark.parametrize(
    "cell_id", ["http-verification-code", "embedding-code", "results-code"]
)
@pytest.mark.parametrize("execute_form", [False, True])
def test_first_run_without_helpers_has_safe_self_contained_recovery(
    monkeypatch, cell_id, execute_form,
):
    """F04: bypassing the helper cannot produce NameError or dispatch an HTTP call."""
    cells = {c["id"]: "".join(c["source"]) for c in load_notebook()["cells"]
             if c["cell_type"] == "code"}
    scope = {}
    if execute_form:
        exec(compile(cells["configure-code"], "new-runtime-form", "exec"), scope)

    def forbidden_http(*_args, **_kwargs):
        raise AssertionError("HTTP attempted without configuration helpers")

    monkeypatch.setattr(urllib.request, "build_opener", forbidden_http)
    with pytest.raises(RuntimeError) as error:
        exec(compile(cells[cell_id], "new-runtime-" + cell_id, "exec"), scope)
    message = str(error.value)
    assert "English: Run Configuration helpers" in message
    assert "Tiếng Việt: Hãy chạy Configuration helpers" in message
    assert "NameError" not in message


def test_recovery_cell_never_calls_http_and_identifies_next_step(
    monkeypatch, verified_notebook_session,
):
    scope, cells, requests, _token, _url = verified_notebook_session
    assert "recovery-code" in cells
    before = len(requests)
    recovery = cells["recovery-code"]
    fresh = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(recovery, "recovery-fresh", "exec"), fresh)
    assert "Configuration helpers" in fresh["recovery_message"]
    with redirect_stdout(io.StringIO()):
        exec(compile(recovery, "recovery-complete", "exec"), scope)
    assert scope["recovery_message"].startswith("English: Results")
    monkeypatch.setenv("VISION_EMBEDDINGS_SAMPLE_TEXT", "changed private sample")
    with redirect_stdout(io.StringIO()):
        exec(compile(recovery, "recovery-input", "exec"), scope)
    assert "Prepare input samples" in scope["recovery_message"]
    assert "changed private sample" not in scope["recovery_message"]
    monkeypatch.delenv("VISION_EMBEDDINGS_SAMPLE_TEXT")
    monkeypatch.setenv("VISION_EMBEDDINGS_DIMENSION", "invalid-dimension")
    with redirect_stdout(io.StringIO()):
        exec(compile(recovery, "recovery-config", "exec"), scope)
    assert "Configuration helpers" in scope["recovery_message"]
    monkeypatch.delenv("VISION_EMBEDDINGS_DIMENSION")
    scope["verified_model_id"] = None
    with redirect_stdout(io.StringIO()):
        exec(compile(recovery, "recovery-preflight", "exec"), scope)
    assert "Check API connection" in scope["recovery_message"]
    assert len(requests) == before


def test_recovery_requires_new_inference_after_reconnection(
    monkeypatch, verified_notebook_session,
):
    scope, cells, requests, _token, other_url = verified_notebook_session
    before_posts = sum(method == "POST" for _, method in requests)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", other_url)
    with redirect_stdout(io.StringIO()):
        for cell in ("credentials-input", "http-verification-code", "recovery-code"):
            exec(compile(cells[cell], cell, "exec"), scope)
    assert "Generate embeddings" in scope["recovery_message"]
    assert sum(method == "POST" for _, method in requests) == before_posts


@pytest.mark.parametrize(
    "change", ["endpoint", "token", "invalid_url", "invalid_token", "cancelled", "preflight"]
)
def test_report_rejects_stale_connection_without_sending_requests(
    monkeypatch, verified_notebook_session, change,
):
    """A prior successful report cannot stand in for a changed or failed connection."""
    scope, cells, requests, _token, other_url = verified_notebook_session
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["results-code"], "original-report", "exec"), scope)
    original_report = scope["report_html"]
    if change == "preflight":
        class Unavailable:
            def open(self, *_args, **_kwargs):
                raise urllib.error.URLError("private transport failure")
        monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Unavailable())
        with pytest.raises(RuntimeError, match="Cannot reach"):
            exec(compile(cells["http-verification-code"], "failed-preflight", "exec"), scope)
    else:
        if change in {"endpoint", "invalid_url"}:
            monkeypatch.setenv(
                "VISION_EMBEDDINGS_BASE_URL",
                other_url if change == "endpoint" else "http://private.example.invalid",
            )
        elif change in {"token", "invalid_token"}:
            monkeypatch.setenv(
                "VISION_EMBEDDINGS_API_TOKEN",
                "another-private-test-token" if change == "token" else "short",
            )
        else:
            monkeypatch.delenv("VISION_EMBEDDINGS_BASE_URL")
            def cancel(_prompt):
                raise KeyboardInterrupt
            monkeypatch.setattr("getpass.getpass", cancel)
        if change in {"invalid_url", "invalid_token", "cancelled"}:
            expected = KeyboardInterrupt if change == "cancelled" else ValueError
            with pytest.raises(expected):
                exec(compile(cells["credentials-input"], "failed-reconnect", "exec"), scope)
        else:
            with redirect_stdout(io.StringIO()):
                exec(compile(cells["credentials-input"], "reconnect", "exec"), scope)
                if change == "endpoint":
                    exec(compile(cells["http-verification-code"], "preflight", "exec"), scope)
    before = len(requests)
    with pytest.raises(RuntimeError, match="connection") as error:
        exec(compile(cells["results-code"], "stale-report", "exec"), scope)
    assert "Tiếng Việt:" in str(error.value)
    assert "private" not in str(error.value) and other_url not in str(error.value)
    assert scope["report_html"] == original_report
    assert len(requests) == before
    if change == "endpoint":
        with redirect_stdout(io.StringIO()):
            exec(compile(cells["embedding-code"], "fresh-inference", "exec"), scope)
            exec(compile(cells["results-code"], "fresh-report", "exec"), scope)
        assert sum(method == "POST" for _, method in requests) == 12
        assert scope["completed_run_config"]["connection_identity"][0] == other_url


def test_report_can_reuse_results_after_verifying_the_same_connection(
    verified_notebook_session,
):
    scope, cells, requests, _token, _url = verified_notebook_session
    with redirect_stdout(io.StringIO()):
        for name in ("credentials-input", "http-verification-code", "results-code"):
            exec(compile(cells[name], name, "exec"), scope)
    assert sum(method == "POST" for _, method in requests) == 6
    assert "English — Results" in scope["report_html"]


@pytest.mark.parametrize(
    ("variable", "value"),
    [("VISION_EMBEDDINGS_BASE_URL", "https://changed.example.invalid"),
     ("VISION_EMBEDDINGS_API_TOKEN", "changed-private-test-token")],
)
def test_recovery_detects_changed_connection_environment_without_http(
    monkeypatch, verified_notebook_session, variable, value,
):
    scope, cells, requests, _token, _url = verified_notebook_session
    before = len(requests)
    monkeypatch.setenv(variable, value)
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["recovery-code"], "changed-connection-env", "exec"), scope)
    assert "Secure API connection → Check API connection" in scope["recovery_message"]
    assert value not in scope["recovery_message"]
    assert len(requests) == before


def test_public_code_cell_sources_and_preview_are_synchronized():
    """The published notebook is standalone and has reviewable editable sources."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/sync_notebooks.py"), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    notebook = json.loads(
        (root / "notebooks/multimodal_embeddings_api_demo.ipynb").read_text()
    )
    sources = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    assert "Built-in samples only" in sources["sample-preparation-code"]
    assert "Chỉ ảnh mẫu nhúng sẵn" in sources["sample-preparation-code"]
    assert "not measured similarity scores" in sources["results-explanation"]
    assert "không phải điểm tương đồng đã đo" in sources["results-explanation"]


def test_actions_use_pinned_v7_revisions():
    """Prevent future accidental use of unpinned or obsolete major versions."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text()
    assert re.search(r"actions/checkout@[a-f0-9]{40} # v7", workflow)
    assert re.search(r"actions/setup-python@[a-f0-9]{40} # v7\.1\.0", workflow)
    assert "persist-credentials: false" in workflow
