"""Ensure public naming and deployment resources stay neutral and consistent."""

import subprocess
import tomllib
from pathlib import Path

from multimodal_embeddings_ai_inference.config import Settings

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "multimodal_embeddings_ai_inference"
ENV_PREFIX = "VISION_EMBEDDINGS_"
PROJECT_SLUG = "multimodal-embeddings-ai-inference"


def test_distribution_import_namespace():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == PROJECT_SLUG
    assert (ROOT / "src" / NAMESPACE / "__init__.py").is_file()


def test_environment_prefix_and_configuration(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "API_TOKEN", "test-token-for-public-release-000000")
    monkeypatch.setenv(ENV_PREFIX + "BACKEND", "mock")
    settings = Settings.from_env()
    assert settings.api_token == "test-token-for-public-release-000000"
    assert settings.backend == "mock"


def test_modal_names_and_dedicated_gpu_resources():
    app = (ROOT / "modal_app.py").read_text()
    models = (ROOT / "modal_models.py").read_text()
    assert f'modal.App("{PROJECT_SLUG}")' in app
    assert "multimodal-embeddings-api" in app
    for gpu_class in ("QwenL40SGpu", "WeMML40SGpu"):
        assert f"class {gpu_class}:" in app
    for name in ("multimodal-embeddings-qwen-weights", "multimodal-embeddings-wemm-weights"):
        assert name in app and name in models
    assert ENV_PREFIX + "QWEN_VOLUME_NAME" in app
    assert ENV_PREFIX + "WEMM_VOLUME_NAME" in app
    assert "SharedL40SGpu" not in app


def test_tracked_sources_have_no_legacy_project_names():
    names = [
        item.decode()
        for item in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
        if item
    ]
    forbidden = ("Compact" + "Faces-Vision-Embeddings-AI-Inference",
                 "compact" + "faces_vision_embeddings_ai_inference")
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_text(encoding="utf-8", errors="replace")
        for old in forbidden:
            assert old not in name and old not in data, (name, old)


def test_markdown_fences_fail_on_trailing_text_or_missing_closer(tmp_path):
    import runpy

    import pytest

    if not (ROOT / "README.md").exists():
        pytest.skip("README introduced in final source commit")
    validator = runpy.run_path(str(ROOT / "scripts" / "audit_release.py"))["markdown_fence_errors"]
    triple = chr(96) * 3
    example = tmp_path / "README.md"
    example.write_text("Intro\n\n" + triple + "bash\necho hello\n" + triple + " trailing\n")
    assert any("closing code fence" in issue for issue in validator(example))
    example.write_text("Intro\n\n" + triple + "bash\necho hello\n")
    assert any("unterminated" in issue for issue in validator(example))
    example.write_text("Intro\n\n" + triple + "bash\necho hello\n" + triple + "\nAfter\n")
    assert validator(example) == []


def test_bilingual_audit_allows_independent_paragraph_wrapping():
    """Bilingual docs may differ in line breaks while keeping structural parity."""
    import runpy

    import pytest

    if not (ROOT / "README.md").exists():
        pytest.skip("Bilingual README is added in the final source commit")
    signature = runpy.run_path(str(ROOT / "scripts" / "audit_release.py"))[
        "markdown_structure"
    ]
    english = ["# Usage", "", "A short explanation.", "", "| Name | Value |", "| --- | --- |"]
    vietnamese = [
        "# Hướng dẫn", "", "Một đoạn giải thích", "có thể xuống dòng khác nhau.",
        "", "| Tên | Giá trị |", "| --- | --- |",
    ]
    assert signature(english) == signature(vietnamese)
    assert signature(english) != signature(["## Wrong level", "| One | Two | Three |"])
    assert signature(english) != signature(["# Usage", "```bash", "echo hello", "```"])


def test_public_readmes_have_complete_code_fences():
    import runpy

    import pytest

    if not (ROOT / "README.md").exists():
        pytest.skip("README introduced in final source commit")
    validator = runpy.run_path(str(ROOT / "scripts" / "audit_release.py"))["markdown_fence_errors"]
    assert validator(ROOT / "README.md") == []
    assert validator(ROOT / "README.vi.md") == []
