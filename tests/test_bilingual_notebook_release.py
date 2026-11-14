"""Protect separate English/Vietnamese notebook outputs and release hashes."""
import hashlib
import io
import json
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def cells(filename: str) -> dict[str, str]:
    nb = json.loads((NOTEBOOKS / filename).read_text(encoding="utf-8"))
    return {cell["id"]: "".join(cell["source"]) for cell in nb["cells"]
            if cell["cell_type"] == "code"}


def test_benchmark_preview_outputs_independent_language_lines():
    config = cells("multimodal_embeddings_benchmark.ipynb")["config"]
    scope: dict[str, object] = {}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(config, "benchmark-preview", "exec"), scope)
    text = output.getvalue()
    assert scope["RUN_BENCHMARK"] is False
    assert scope["PLANNED_POST_TOTAL"] == 24
    assert "English: Planned workload (preview only)" in text
    assert "Tiếng Việt: Khối lượng dự kiến (chỉ xem trước)" in text
    assert "English: Authorization: OFF" in text
    assert "Tiếng Việt: Cho phép chạy: TẮT" in text
    assert "English: Total planned HTTP POST attempts" in text
    assert "Tiếng Việt: Tổng lượt HTTP POST dự kiến" in text
    assert " / Khối lượng" not in text
    assert " / Cho phép chạy" not in text
    assert " / Tổng lượt" not in text
    for phase in ("Warm-up requests", "Warm measured requests"):
        assert f"English: {phase}" in text


def test_benchmark_report_uses_two_explicit_latency_tables():
    report = cells("multimodal_embeddings_benchmark.ipynb")["report"]
    assert "| Phase | Model | Input | Confirmed | Latency n |" in report
    assert "| Giai đoạn | Model | Đầu vào | Đã xác nhận | Số mẫu latency |" in report
    assert "### English — Confirmed latency" in report
    assert "### Tiếng Việt — Độ trễ đã xác nhận" in report
    assert "English: No confirmed timing states" in report
    assert "Tiếng Việt: Chưa có mẫu đã xác nhận" in report
    assert "English: LIVE REPORT" in report
    assert "Tiếng Việt: BÁO CÁO THỰC TẾ" in report
    assert "Confirmed latency / Độ trễ đã xác nhận" not in report


def test_demo_previews_use_localized_html_regions():
    code = cells("multimodal_embeddings_api_demo.ipynb")
    samples = code["sample-preparation-code"]
    results = code["results-code"]
    assert '<div lang="en"><strong>Built-in samples only.' in samples
    assert '<div lang="vi"><strong>Chỉ ảnh mẫu nhúng sẵn.' in samples
    assert '<span lang="en" style="display:block">' in samples
    assert '<span lang="vi" style="display:block">' in samples
    assert '<p lang="en">Private image preview' in samples
    assert '<p lang="vi">Xem ảnh riêng' in samples
    assert '<div lang="en"><strong>Actual inference inputs' in results
    assert '<div lang="vi"><strong>Ảnh thực sự dùng cho inference' in results
    assert "Actual inference inputs / Ảnh thực sự" not in results
    assert 'aria-label="Built-in image previews / Hình ảnh nhúng sẵn"' not in samples
    assert 'aria-label="Built-in image previews"' in samples
    assert 'show_stage("Results / Kết quả"' not in results
    assert 'show_stage("5 / 5", "Preview only — no cosine scores"' in results
    assert '"cached / từ cache"' not in results
    assert '"cached" if language == "en" else "từ cache"' in results


def test_release_download_checksum_list_is_flat_and_nonrecursive(tmp_path):
    generated = tmp_path / "SHA256SUMS.release.txt"
    script = ROOT / "scripts/build_release_checksums.py"
    subprocess.run(
        ["python", str(script), "--output", str(generated)],
        check=True, capture_output=True, text=True,
    )
    lines = generated.read_text(encoding="utf-8").splitlines()
    paths = (
        ROOT / "notebooks/multimodal_embeddings_api_demo.ipynb",
        ROOT / "notebooks/multimodal_embeddings_benchmark.ipynb",
        ROOT / "SHA256SUMS.txt",
    )
    assert len(lines) == len(paths)
    for line, path in zip(lines, paths, strict=True):
        assert line == f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
    assert not any("SHA256SUMS.release.txt" in line for line in lines)
    subprocess.run(
        ["python", str(script), "--output", str(generated), "--check"],
        check=True, capture_output=True, text=True,
    )


def test_readme_stable_and_development_links_are_distinct():
    for name in ("README.md", "README.vi.md"):
        readme = (ROOT / name).read_text(encoding="utf-8")
        assert "git checkout v1.0.0" in readme
        notebooks = (
            "multimodal_embeddings_api_demo.ipynb",
            "multimodal_embeddings_benchmark.ipynb",
        )
        for notebook in notebooks:
            stable = f"/blob/v1.0.0/notebooks/{notebook}"
            development = f"/blob/main/notebooks/{notebook}"
            assert readme.count(stable) == 2
            assert readme.count(development) == 1
