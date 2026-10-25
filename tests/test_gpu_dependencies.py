"""Validate pinned GPU-worker requirements without downloading models."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirements(model: str) -> dict[str, str]:
    rows = (ROOT / "constraints" / f"gpu-{model}.txt").read_text().splitlines()
    pins = [row.strip() for row in rows if row.strip() and not row.lstrip().startswith("#")]
    result = {}
    for row in pins:
        assert re.fullmatch(r"[a-zA-Z0-9_.-]+==[a-zA-Z0-9_.+!-]+", row), row
        name, version = row.split("==")
        name = name.lower().replace("_", "-")
        assert name not in result, name
        result[name] = version
    assert len(result) >= 40
    return result


def test_gpu_requirements_use_distinct_pinned_model_stacks() -> None:
    qwen = _requirements("qwen")
    wemm = _requirements("wemm")
    assert qwen["torch"] == wemm["torch"] == "2.8.0"
    assert qwen["torchvision"] == wemm["torchvision"] == "0.23.0"
    assert qwen["transformers"] == "4.57.3"
    assert wemm["transformers"] == "5.2.0"
    assert qwen["huggingface-hub"] == "0.36.0"
    assert wemm["huggingface-hub"] == "1.4.1"


def test_modal_workers_install_committed_locks_in_separate_venvs() -> None:
    source = (ROOT / "modal_app.py").read_text()
    for model in ("qwen", "wemm"):
        assert f'ROOT / "constraints" / "gpu-{model}.txt"' in source
        assert (
            f'/opt/venv-{model}/bin/pip install --no-cache-dir -r /opt/gpu-{model}.txt'
            in source
        )
    assert source.count("add_local_file(") == 2
