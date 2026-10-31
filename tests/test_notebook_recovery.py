"""Validate the Kaggle/Colab HTTP-only notebook without Modal cloud access."""

import io
import json
import os
import re
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from test_notebook_template import load_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "multimodal_embeddings_api_demo.ipynb"
MODELS = ("qwen3-vl-embedding-8b", "wemm-embedding-9b")




@pytest.mark.parametrize("selected_model", MODELS)
def test_first_run_recovery_renders_the_published_report(
    monkeypatch, notebook_connection, selected_model,
):
    """Following recovery in a fresh runtime must include every report dependency."""
    scope, cells, requests, _token, _url = notebook_connection
    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL", selected_model)
    with redirect_stdout(io.StringIO()):
        for name in ("configure-code", "configure-helpers-code",
                     "sample-preparation-code", "http-verification-code", "recovery-code"):
            exec(compile(cells[name], name, "exec"), scope)
    assert not any(method == "POST" for _, method in requests)
    assert "LANGUAGES" not in scope

    titles = {
        "Generate embeddings": "embedding-code",
        "Results": "results-code",
    }
    with redirect_stdout(io.StringIO()):
        for title in scope["_next_step"].split(" → "):
            cell_id = titles[title]
            exec(compile(cells[cell_id], cell_id, "exec"), scope)
    assert sum(method == "POST" for _, method in requests) == 6
    assert "English — Results" in scope["report_html"]
    assert "Tiếng Việt — Kết quả" in scope["report_html"]
    assert scope["summary"][0]["model"] == selected_model

    # Rendering and recovery after inference must never dispatch another POST.
    with redirect_stdout(io.StringIO()):
        exec(cells["recovery-code"], scope)
        for title in scope["_next_step"].split(" → "):
            exec(cells[titles[title]], scope)
    assert sum(method == "POST" for _, method in requests) == 6
    assert "English — Results" in scope["report_html"]


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
    scope["RUN_API_DEMO"] = True  # Explicit opt-in enables missing-credential prompts.
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
    tables = "\n".join(line for line in intro.splitlines() if line.startswith("| "))
    assert "| Changed | Rerun |" in intro
    assert "| Đã thay đổi | Chạy lại |" in intro
    assert tables.count("**Secure API connection → Check API connection") == 2
    assert tables.count("**Customize samples → Prepare input samples") == 2
    assert (
        tables.count("**Model configuration → Configuration helpers → Prepare input samples") == 2
    )
    assert "do not automatically retry an already sent POST" in intro


def test_contributing_explains_development_setup():
    en = ROOT / "CONTRIBUTING.md"
    vi = ROOT / "CONTRIBUTING.vi.md"
    if not en.exists() or not vi.exists():
        pytest.skip("Contributor guides unavailable in this checkout")
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
    assert form.count("#@param") == 6
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

    import test_notebook_template as original_tests

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
    monkeypatch.setattr(original_tests, "load_notebook", lambda: book)
    original_tests.test_bilingual_sections_are_always_visible_and_technically_aligned()

    absent = copy.deepcopy(book)
    absent["cells"] = [c for c in absent["cells"] if c["id"] != "credentials-input"]
    monkeypatch.setattr(original_tests, "load_notebook", lambda: absent)
    with pytest.raises(AssertionError):
        original_tests.test_bilingual_sections_are_always_visible_and_technically_aligned()

    wrong_order = copy.deepcopy(book)
    ids = [c["id"] for c in wrong_order["cells"]]
    a, b = ids.index("configure-helpers-code"), ids.index("credentials-input")
    wrong_order["cells"][a], wrong_order["cells"][b] = (
        wrong_order["cells"][b], wrong_order["cells"][a]
    )
    monkeypatch.setattr(original_tests, "load_notebook", lambda: wrong_order)
    with pytest.raises(AssertionError):
        original_tests.test_bilingual_sections_are_always_visible_and_technically_aligned()


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
            scope["RUN_API_DEMO"] = True
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
    expected_hint = "Configuration helpers" if change == "cancelled" else "connection"
    with pytest.raises(RuntimeError, match=expected_hint) as error:
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
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/sync_notebooks.py"), "--check"]
    if not (root / "README.md").exists():
        command.append("--allow-unpublished")
    result = subprocess.run(
        command,
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
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text()
    assert re.search(r"actions/checkout@[a-f0-9]{40} # v7", workflow)
    assert re.search(r"actions/setup-python@[a-f0-9]{40} # v7\.1\.0", workflow)
    assert "persist-credentials: false" in workflow

