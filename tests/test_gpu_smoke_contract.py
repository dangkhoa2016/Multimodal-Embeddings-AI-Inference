"""CPU checks for the explicitly invoked real-GPU smoke command.

All provider calls use a fake Modal object. Real GPU execution is opt-in only.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def smoke(monkeypatch):
    calls = []
    modal = types.ModuleType("modal")

    class FakeEndpoint:
        def __init__(self, model_id):
            self.model_id = model_id
            self.remote = self.encode

        def encode(self, model_id, items):
            calls.append((model_id, items))
            assert model_id == self.model_id
            return {
                "vectors": [[1.0] + [0.0] * (item["dimension"] - 1) for item in items],
                "checkpoint_identity": {
                    "model": model_id,
                    "revision": "a" * 40 if model_id == "qwen3-vl-embedding-8b" else "b" * 40,
                },
                "timing": {"gpu_worker_id": (
                    "a" if model_id == "qwen3-vl-embedding-8b" else "b"
                ) * 32, "model_state": "warm"},
            }

    class FakeClass:
        @staticmethod
        def from_name(app, cls):
            assert app == "multimodal-embeddings-ai-inference"
            model_id = {
                "QwenL40SGpu": "qwen3-vl-embedding-8b",
                "WeMML40SGpu": "wemm-embedding-9b",
            }[cls]
            return lambda: types.SimpleNamespace(encode_many=FakeEndpoint(model_id))

    modal.Cls = FakeClass
    monkeypatch.setitem(sys.modules, "modal", modal)
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_modal_gpu.py"
    spec = importlib.util.spec_from_file_location("smoke_modal_gpu_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_real_smoke_defaults_to_image_only_and_checks_both_models(
    smoke, tmp_path, monkeypatch, capsys,
):
    module, calls = smoke
    image = tmp_path / "authorized.png"
    image.write_bytes(b"authorized-image-data")
    monkeypatch.setattr(
        sys, "argv", ["smoke_modal_gpu.py", "--image", str(image), "--dimension", "2"]
    )
    module.main()
    assert len(calls) == 2
    assert all(len(items) == 1 and items[0]["kind"] == "image" for _, items in calls)
    assert capsys.readouterr().out.count("REAL_PASS") == 2


def test_real_smoke_text_is_explicit_and_runs_for_both_models(smoke, tmp_path, monkeypatch):
    module, calls = smoke
    image = tmp_path / "authorized.png"
    image.write_bytes(b"authorized-image-data")
    monkeypatch.setattr(
        sys, "argv",
        [
            "smoke_modal_gpu.py", "--image", str(image),
            "--text", "an orange gift box", "--dimension", "3",
        ],
    )
    module.main()
    assert len(calls) == 2
    for _, items in calls:
        assert [item["kind"] for item in items] == ["image", "text"]
        assert items[1]["payload"] == "an orange gift box"
        assert all(item["dimension"] == 3 for item in items)


def test_invalid_image_file_never_dispatches(smoke, tmp_path, monkeypatch):
    module, calls = smoke
    image = tmp_path / "empty.png"
    image.touch()
    monkeypatch.setattr(sys, "argv", ["smoke_modal_gpu.py", "--image", str(image)])
    with pytest.raises(SystemExit):
        module.main()
    assert not calls


def test_blank_text_never_dispatches(smoke, tmp_path, monkeypatch):
    module, calls = smoke
    image = tmp_path / "image.png"
    image.write_bytes(b"authorized-image-data")
    monkeypatch.setattr(sys, "argv", ["smoke_modal_gpu.py", "--image", str(image), "--text", " "])
    with pytest.raises(SystemExit):
        module.main()
    assert not calls

def test_real_smoke_rejects_revision_drift(smoke, tmp_path, monkeypatch):
    module, calls = smoke
    image = tmp_path / "authorized.png"
    image.write_bytes(b"authorized-image-data")
    monkeypatch.setattr(
        sys, "argv",
        ["smoke_modal_gpu.py", "--image", str(image),
         "--model", "qwen3-vl-embedding-8b", "--qwen-revision", "c" * 40],
    )
    with pytest.raises(RuntimeError, match="checkpoint revision mismatch"):
        module.main()
    assert len(calls) == 1


def test_real_smoke_rejects_bad_revision_before_gpu(smoke, tmp_path, monkeypatch):
    module, calls = smoke
    image = tmp_path / "authorized.png"
    image.write_bytes(b"authorized-image-data")
    monkeypatch.setattr(
        sys, "argv", ["smoke_modal_gpu.py", "--image", str(image),
                       "--qwen-revision", "latest"],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert calls == []


def test_real_smoke_accepts_matching_pinned_revision(smoke, tmp_path, monkeypatch):
    module, calls = smoke
    image = tmp_path / "authorized.png"
    image.write_bytes(b"authorized-image-data")
    monkeypatch.setattr(
        sys, "argv", ["smoke_modal_gpu.py", "--image", str(image),
                       "--model", "wemm-embedding-9b", "--wemm-revision", "b" * 40],
    )
    module.main()
    assert len(calls) == 1
