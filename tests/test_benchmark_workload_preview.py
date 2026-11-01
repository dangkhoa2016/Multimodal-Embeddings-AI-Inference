"""Offline UI/workload preview tests for the standalone performance benchmark."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "notebooks/multimodal_embeddings_benchmark.ipynb"


def code_cells():
    return {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(BOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    }


def preview(**overrides):
    source = code_cells()["config"]
    for name, value in overrides.items():
        before = next(
            line for line in source.splitlines()
            if line.startswith(name + " = ")
        )
        source = source.replace(before, before.replace(
            before.split(" = ", 1)[1].split(" #@param")[0],
            repr(value) if isinstance(value, str) else str(value),
        ), 1)
    values, output = {}, io.StringIO()
    with redirect_stdout(output):
        exec(compile(source, "benchmark-config-preview", "exec"), values)
    return values, output.getvalue()


def test_all_code_cells_have_bilingual_colab_titles_and_clean_outputs():
    book = json.loads(BOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 9
    for cell in cells:
        source = "".join(cell["source"])
        title = next(line for line in source.splitlines() if line.startswith("#@title "))
        assert " / " in title and "display-mode" in title
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


def test_default_preview_counts_warmup_and_measured_separately():
    values, output = preview()
    assert values["RUN_BENCHMARK"] is False
    assert values["PLANNED_POSTS"] == {
        "cold": 0, "warmup": 8, "warm_measured": 16,
        "alternate_setup": 0, "alternate_measured": 0,
        "concurrent": 0, "mixed_batch": 0,
    }
    assert values["PLANNED_POST_TOTAL"] == 24
    assert values["PLANNED_EMBEDDING_ITEMS"] == 24
    assert "Authorization / Cho phép chạy: OFF / TẮT" in output
    assert "Request làm nóng" in output
    assert "Total planned HTTP POST attempts" in output
    assert "không tự thử lại" in output


def test_all_phases_preview_distinguishes_post_count_batch_items_and_idle():
    values, output = preview(
        RUN_BENCHMARK=True, RUN_COLD=True, RUN_ALTERNATE=True,
        RUN_CONCURRENT=True, RUN_MIXED_BATCH=True,
    )
    assert values["PLANNED_POSTS"] == {
        "cold": 10, "warmup": 8, "warm_measured": 16,
        "alternate_setup": 2, "alternate_measured": 10,
        "concurrent": 8, "mixed_batch": 3,
    }
    assert values["PLANNED_POST_TOTAL"] == 57
    assert values["PLANNED_EMBEDDING_ITEMS"] == 60
    assert values["COLD_IDLE_WAIT_SECONDS"] == 210
    assert values["COLD_TOTAL_IDLE_SECONDS"] == 2100
    assert "Authorization / Cho phép chạy: ON / BẬT" in output
    assert "không gồm thời gian xử lý API/GPU" in output
    assert "two items each" in output
    assert "no automatic retry or cost estimate" in output


def test_single_model_cold_only_previews_without_network_or_auth():
    values, output = preview(
        RUN_COLD=True, RUN_WARM=False,
        COLD_TARGET_MODEL="qwen3-vl-embedding-8b",
    )
    assert values["PLANNED_POST_TOTAL"] == 5
    assert values["PLANNED_EMBEDDING_ITEMS"] == 5
    assert values["COLD_TOTAL_IDLE_SECONDS"] == 1050
    assert "OFF / TẮT" in output
    assert "5" in output and "Chỉ tính thời gian chờ cold" in output
