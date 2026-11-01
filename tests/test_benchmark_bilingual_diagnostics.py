"""Offline regression coverage for bilingual performance-benchmark diagnostics."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "multimodal_embeddings_benchmark.ipynb"


def cells():
    return {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    }


@pytest.mark.parametrize(
    ("before", "after", "field", "vietnamese"),
    [
        ("DIMENSION = 1024", "DIMENSION = 256", "DIMENSION", "phải"),
        ("CONCURRENT_WORKERS = 2", "CONCURRENT_WORKERS = 0", "CONCURRENT_WORKERS", "phải"),
        ("WARM_REPETITIONS = 4", "WARM_REPETITIONS = 3", "WARM_REPETITIONS", "số chẵn"),
        ("COLD_REPETITIONS = 5", "COLD_REPETITIONS = 0", "COLD_REPETITIONS", "phải"),
        ("COLD_WINDOW_PADDING_SECONDS = 30", "COLD_WINDOW_PADDING_SECONDS = 0",
         "COLD_WINDOW_PADDING_SECONDS", "phải"),
        ('QWEN_INSTRUCTION_MODE = "default"', 'QWEN_INSTRUCTION_MODE = "wrong"',
         "QWEN_INSTRUCTION_MODE", "phải"),
        ('COLD_TARGET_MODEL = "both"', 'COLD_TARGET_MODEL = "invalid"',
         "COLD_TARGET_MODEL", "phải"),
    ],
)
def test_invalid_config_reports_both_languages(
    before, after, field, vietnamese, monkeypatch,
):
    monkeypatch.delenv("BENCHMARK_TEXT", raising=False)
    code = cells()["config"]
    assert before in code
    code = code.replace(before, after, 1)
    with pytest.raises(ValueError) as raised:
        exec(compile(code, "benchmark-config", "exec"), {})
    assert field in str(raised.value)
    assert " / " in str(raised.value)
    assert vietnamese in str(raised.value)


def test_blank_text_and_cold_container_mismatch_are_bilingual(monkeypatch):
    monkeypatch.setenv("BENCHMARK_TEXT", " ")
    with pytest.raises(ValueError, match="BENCHMARK_TEXT") as raised:
        exec(cells()["config"], {})
    assert "ký tự" in str(raised.value)
    monkeypatch.delenv("BENCHMARK_TEXT")
    code = cells()["config"].replace("RUN_BENCHMARK = False", "RUN_BENCHMARK = True", 1)
    code = code.replace('BENCHMARK_MODE = "warm"', 'BENCHMARK_MODE = "cold"', 1)
    code = code.replace("GATEWAY_MIN_CONTAINERS = 0", "GATEWAY_MIN_CONTAINERS = 1", 1)
    with pytest.raises(ValueError, match="GATEWAY_MIN_CONTAINERS") as raised:
        exec(code, {})
    assert "Chạy cold" in str(raised.value)


def test_runtime_response_errors_have_vietnamese_and_no_network(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BENCHMARK_IMAGE_PATH", raising=False)
    scope = {}
    snippets = cells()
    for part in ("config", "client", "client-request-helpers"):
        exec(compile(snippets[part], part, "exec"), scope)
    with pytest.raises(RuntimeError, match="Instrumented GPU timing required") as raised:
        scope["checked_gpu_timing"]({})
    assert "Thiếu dữ liệu" in str(raised.value)
    scope["register_provenance"]("qwen3-vl-embedding-8b", "revision-a", {})
    with pytest.raises(RuntimeError, match="Model revision changed") as raised:
        scope["register_provenance"]("qwen3-vl-embedding-8b", "revision-b", {})
    assert "thay đổi" in str(raised.value)
    scope["http_json"] = lambda *_args, **_kwargs: {"model": "incorrect"}
    with pytest.raises(RuntimeError, match="request ID/input digest mismatch") as raised:
        scope["_sample_checked"]("qwen3-vl-embedding-8b", "text", "warm", 1)
    assert "không khớp" in str(raised.value)
    assert not list(tmp_path.glob("*.json"))


def test_no_bare_config_assertions_and_no_retry_policy_unchanged():
    snippets = cells()
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(snippets["config"])))
    client = snippets["client"]
    assert "no retry" in client and "không thử lại" in client
    assert "Tiếng Việt:" in client
    assert "run_started_phases.add(phase)" in client
    assert "save_failed_report(phase, error_type, failed_trial)" in client
