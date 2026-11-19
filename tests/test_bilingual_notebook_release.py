"""Protect separate English/Vietnamese notebook outputs and release hashes."""
import hashlib
import io
import json
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import pytest

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
    if not any((ROOT / name).exists() for name in ("README.md", "README.vi.md")):
        pytest.skip("README pair is not published in this historical snapshot")
    for name in ("README.md", "README.vi.md"):
        readme = (ROOT / name).read_text(encoding="utf-8")
        final_navigation = ("**Notebook snapshots:**" in readme or
                            "**Hai snapshot notebook:**" in readme)
        if final_navigation:
            assert "git checkout v1.0.0" in readme
        notebooks = (
            "multimodal_embeddings_api_demo.ipynb",
            "multimodal_embeddings_benchmark.ipynb",
        )
        for notebook in notebooks:
            stable = f"/blob/v1.0.0/notebooks/{notebook}"
            development = f"/blob/main/notebooks/{notebook}"
            assert readme.count(stable) == (1 if final_navigation else 0)
            assert readme.count(development) == 2


def test_notebook_intro_uses_full_model_ids_and_viewer_guidance():
    for filename in (
        "multimodal_embeddings_api_demo.ipynb",
        "multimodal_embeddings_benchmark.ipynb",
    ):
        nb = json.loads((NOTEBOOKS / filename).read_text(encoding="utf-8"))
        intro = "".join(nb["cells"][0]["source"])
        for model in ("Qwen/Qwen3-VL-Embedding-8B", "tencent/WeMM-Embedding-9B"):
            assert intro.count(model) >= 2
        assert "Kaggle/Jupyter" in intro
        assert nb["metadata"]["multimodal_embeddings"]["source_version"] == "1.0.0"

def test_release_checksums_support_a_pinned_git_snapshot(tmp_path):
    generated = tmp_path / "release-ref.sha256"
    script = ROOT / "scripts/build_release_checksums.py"
    subprocess.run(
        ["python", str(script), "--git-ref", "HEAD", "--output", str(generated)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    paths = (
        "notebooks/multimodal_embeddings_api_demo.ipynb",
        "notebooks/multimodal_embeddings_benchmark.ipynb",
        "SHA256SUMS.txt",
    )
    for line, name in zip(generated.read_text().splitlines(), paths, strict=True):
        snapshot = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        assert line == f"{hashlib.sha256(snapshot).hexdigest()}  {Path(name).name}"

def test_project_metadata_uses_single_version_1_0_0():
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "1.0.0"
    package = (ROOT / "src/multimodal_embeddings_ai_inference/__init__.py").read_text()
    assert '__version__ = "1.0.0"' in package
    for name in ("README.md", "README.vi.md", "CHANGELOG.md", "CHANGELOG.vi.md"):
        if name.startswith("README") and not any(
            (ROOT / readme).exists() for readme in ("README.md", "README.vi.md")
        ):
            continue  # Still check package/changelog versions before README publication.
        text = (ROOT / name).read_text(encoding="utf-8")
        import re

        versions = re.findall(r"(?<![0-9])v?[0-9]+\.[0-9]+\.[0-9]+(?![0-9])", text)
        assert versions
        assert all(version.removeprefix("v") == "1.0.0" for version in versions)


def test_release_checksum_missing_git_ref_reports_actionable_error(tmp_path):
    script = ROOT / "scripts/build_release_checksums.py"
    result = subprocess.run(
        ["python", str(script), "--git-ref", "does-not-exist-v1-0-0",
         "--output", str(tmp_path / "checksums.txt")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "git fetch --tags" in result.stderr
    assert "Traceback" not in result.stderr


def test_changelogs_present_one_initial_release_not_patch_history():
    english = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    vietnamese = (ROOT / "CHANGELOG.vi.md").read_text(encoding="utf-8")
    assert english.startswith("# Release notes\n")
    assert vietnamese.startswith("# Thông tin phát hành\n")
    assert english.count("\n## 1.0.0 — Initial release\n") == 1
    assert vietnamese.count("\n## 1.0.0 — Bản phát hành đầu tiên\n") == 1
    assert english.count("\n## ") == vietnamese.count("\n## ") == 1
    for content in (english.lower(), vietnamese.lower()):
        for term in ("1.0.1", "bản vá này", "previous release", "earlier version"):
            assert term not in content
    for model in ("Qwen/Qwen3-VL-Embedding-8B", "tencent/WeMM-Embedding-9B"):
        assert model in english and model in vietnamese


def test_release_asset_gate_requires_four_verified_downloads(tmp_path):
    """Offline release gate checks exact asset list, pinned tag and integrity."""
    from scripts.build_release_checksums import ASSETS, generate
    from scripts.verify_release_assets import verify

    directory = tmp_path / "published"
    directory.mkdir()
    for asset in ASSETS:
        contents = subprocess.check_output(
            ["git", "show", f"HEAD:{asset.relative_to(ROOT).as_posix()}"], cwd=ROOT
        )
        (directory / asset.name).write_bytes(contents)
    (directory / "SHA256SUMS.release.txt").write_text(generate("HEAD"))

    verify(directory, "HEAD")
    missing = directory / "SHA256SUMS.txt"
    original = missing.read_bytes()
    missing.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify(directory, "HEAD")
    missing.write_bytes(original)
    notebook = directory / "multimodal_embeddings_api_demo.ipynb"
    notebook.write_bytes(notebook.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(directory, "HEAD")


def test_supplementary_archive_checksum_is_pinned(tmp_path):
    from scripts.build_release_checksums import ASSETS, SUPPLEMENTARY_ASSETS, generate
    from scripts.verify_release_assets import verify

    name, digest = next(iter(SUPPLEMENTARY_ASSETS.items()))
    assert name.endswith("2026-11-19.zip")
    assert f"{digest}  {name}" in generate("HEAD", include_supplementary=True)

    directory = tmp_path / "release"
    directory.mkdir()
    for asset in ASSETS:
        path = asset.relative_to(ROOT).as_posix()
        (directory / asset.name).write_bytes(
            subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
        )
    for supplementary_name in SUPPLEMENTARY_ASSETS:
        (directory / supplementary_name).write_bytes(b"invalid archive")
    (directory / "SHA256SUMS.release.txt").write_text(
        generate("HEAD", include_supplementary=True)
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(directory, "HEAD")
