"""Offline UI/workload preview tests for the standalone performance benchmark."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

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
        "concurrent_setup": 0, "concurrent": 0,
        "mixed_batch_setup": 0, "mixed_batch": 0,
    }
    assert values["PLANNED_POST_TOTAL"] == 24
    assert values["PLANNED_EMBEDDING_ITEMS"] == 24
    assert "English: Authorization: OFF" in output
    assert "Tiếng Việt: Cho phép chạy: TẮT" in output
    assert "Request làm nóng" in output
    assert "Total planned HTTP POST attempts" in output
    assert "không tự thử lại" in output


@pytest.mark.parametrize(
    ("mode", "posts", "items", "idle_seconds"),
    [("warm", 24, 24, 0), ("cold", 10, 10, 2100),
     ("alternating", 12, 12, 0), ("concurrent", 10, 10, 0),
     ("mixed_batch", 9, 12, 0)],
)
def test_each_mode_previews_one_scenario(mode, posts, items, idle_seconds):
    values, output = preview(RUN_BENCHMARK=True, BENCHMARK_MODE=mode)
    assert values["BENCHMARK_MODE"] == mode
    assert sum(values[name] for name in (
        "RUN_COLD", "RUN_WARM", "RUN_ALTERNATE", "RUN_CONCURRENT",
        "RUN_MIXED_BATCH",
    )) == 1
    assert values["PLANNED_POST_TOTAL"] == posts
    assert values["PLANNED_EMBEDDING_ITEMS"] == items
    assert values["COLD_TOTAL_IDLE_SECONDS"] == idle_seconds
    assert "English: Selected benchmark mode: " + mode in output
    assert "Tiếng Việt: Chế độ benchmark đã chọn: " + mode in output
    assert "English: Authorization: ON" in output
    assert "no automatic retry or cost estimate" in output
    if mode == "cold":
        assert "không gồm thời gian xử lý API/GPU" in output
    if mode == "mixed_batch":
        assert "two items each" in output


def test_full_mode_preview_lists_all_five_phases_and_budget_without_http():
    values, output = preview(BENCHMARK_MODE="full")
    assert values["RUN_BENCHMARK"] is False
    assert all(values[name] for name in (
        "RUN_COLD", "RUN_WARM", "RUN_ALTERNATE", "RUN_CONCURRENT", "RUN_MIXED_BATCH"
    ))
    assert values["PLANNED_POSTS"] == {
        "cold": 10, "warmup": 8, "warm_measured": 16,
        "alternate_setup": 2, "alternate_measured": 10,
        "concurrent_setup": 2, "concurrent": 8,
        "mixed_batch_setup": 6, "mixed_batch": 3,
    }
    assert values["PLANNED_POST_TOTAL"] == 65
    assert values["PLANNED_EMBEDDING_ITEMS"] == 68
    assert values["COLD_IDLE_WAIT_SECONDS"] == 210
    assert values["COLD_TOTAL_IDLE_SECONDS"] == 2100
    assert "English: Selected benchmark mode: full" in output
    assert "English: Authorization: OFF" in output
    assert "Tiếng Việt: Cho phép chạy: TẮT" in output
    assert "35.0 minutes total" in output
    assert "hai model" in output
    assert "65" in output and "68" in output


def test_full_mode_requires_cold_scale_down_preconditions_before_dispatch():
    import pytest

    with pytest.raises(ValueError, match="GATEWAY_MIN_CONTAINERS"):
        preview(BENCHMARK_MODE="full", RUN_BENCHMARK=True,
                GATEWAY_MIN_CONTAINERS=1)
    with pytest.raises(ValueError, match="GPU_MIN_CONTAINERS"):
        preview(BENCHMARK_MODE="full", RUN_BENCHMARK=True,
                GPU_MIN_CONTAINERS=1)


def test_full_mode_phase_order_and_report_journal_contract():
    book = json.loads(BOOK.read_text(encoding="utf-8"))
    ids = [cell["id"] for cell in book["cells"]]
    phase_ids = ["cold-run", "warm-run", "alternate-run",
                 "concurrent-run", "mixed-batch-run"]
    assert all(p in ids for p in phase_ids)
    assert [ids.index(p) for p in phase_ids] == sorted(ids.index(p) for p in phase_ids)
    code = code_cells()
    assert '"benchmark_mode": BENCHMARK_MODE' in code["client"]
    assert '("cold", RUN_COLD)' in code["client"]
    assert '("mixed_batch", RUN_MIXED_BATCH)' in code["client"]
    assert "run_requested_phases" in code["report"]


@pytest.mark.parametrize("bad_mode", ["", "all", "WARM", "cold,warm", 123])
def test_invalid_mode_rejected_before_network(bad_mode):
    with pytest.raises(ValueError, match="BENCHMARK_MODE"):
        preview(BENCHMARK_MODE=bad_mode)




def test_single_model_cold_only_previews_without_network_or_auth():
    values, output = preview(
        BENCHMARK_MODE="cold",
        COLD_TARGET_MODEL="qwen3-vl-embedding-8b",
    )
    assert values["PLANNED_POST_TOTAL"] == 5
    assert values["PLANNED_EMBEDDING_ITEMS"] == 5
    assert values["COLD_TOTAL_IDLE_SECONDS"] == 1050
    assert "English: Authorization: OFF" in output
    assert "Tiếng Việt: Cho phép chạy: TẮT" in output
    assert "5" in output and "Chỉ tính thời gian chờ cold" in output
