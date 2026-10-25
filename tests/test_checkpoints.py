import json
import sys
import types

import pytest

from multimodal_embeddings_ai_inference.checkpoints import (
    cached_checkpoint,
    checkpoint_info,
)


def test_verified_checkpoint(tmp_path):
    folder = tmp_path / "tencent" / "WeMM-Embedding-9B"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    sha = "a" * 40
    (folder / ".embedding-snapshot.json").write_text(
        json.dumps(
            {
                "repository": "tencent/WeMM-Embedding-9B",
                "requested_revision": "main",
                "commit_sha": sha,
            }
        )
    )
    assert cached_checkpoint("wemm-embedding-9b", "main", tmp_path) == str(folder)
    assert checkpoint_info("wemm-embedding-9b", sha, tmp_path)["commit_sha"] == sha
    assert cached_checkpoint("wemm-embedding-9b", "b" * 40, tmp_path) is None
    assert cached_checkpoint("qwen3-vl-embedding-8b", "main", tmp_path) is None


def test_refuse_corrupt_checkpoint_marker(tmp_path):
    folder = tmp_path / "Qwen" / "Qwen3-VL-Embedding-8B"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / ".embedding-snapshot.json").write_text("{broken")
    assert cached_checkpoint("qwen3-vl-embedding-8b", "main", tmp_path) is None

    (folder / ".embedding-snapshot.json").write_text(
        json.dumps(
            {
                "repository": "Qwen/Qwen3-VL-Embedding-8B",
                "requested_revision": "main",
                "commit_sha": "fake",
            }
        )
    )
    assert cached_checkpoint("qwen3-vl-embedding-8b", "main", tmp_path) is None


@pytest.mark.parametrize("model,repository,key,adapter_name,class_name", [
    ("wemm-embedding-9b", "tencent/WeMM-Embedding-9B", "WEMM9", "wemm", "WeMMEncoder"),
    ("qwen3-vl-embedding-8b", "Qwen/Qwen3-VL-Embedding-8B", "QWEN8", "qwen", "QwenVLEncoder"),
])
def test_encoder_keeps_identity_of_loaded_snapshot(
    tmp_path, monkeypatch, model, repository, key, adapter_name, class_name
):
    from importlib import import_module

    folder = tmp_path.joinpath(*repository.split("/"))
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    marker = folder / ".embedding-snapshot.json"
    marker.write_text(json.dumps({
        "repository": repository, "requested_revision": "a" * 40, "commit_sha": "a" * 40,
    }))
    monkeypatch.setenv("VISION_EMBEDDINGS_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv(f"VISION_EMBEDDINGS_{key}_REVISION", "a" * 40)
    monkeypatch.setenv("VISION_EMBEDDINGS_QWEN_SOURCE_PATH", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    loaded_paths = []

    class FakeModel:
        def __init__(self, model_name_or_path=None, **kwargs):
            if model_name_or_path:
                loaded_paths.append(model_name_or_path)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loaded_paths.append(path)
            return cls()

        def eval(self):
            return self

    modules = {
        "torch": {"bfloat16": object()},
        "transformers": {"AutoModel": FakeModel, "AutoProcessor": FakeModel},
        "src": {"__path__": []}, "src.models": {"__path__": []},
        "src.models.qwen3_vl_embedding": {"Qwen3VLEmbedder": FakeModel},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    encoder = getattr(import_module(
        f"multimodal_embeddings_ai_inference.engines.{adapter_name}"
    ), class_name)()
    monkeypatch.setenv(f"VISION_EMBEDDINGS_{key}_REVISION", "b" * 40)
    marker.write_text(json.dumps({
        "repository": repository, "requested_revision": "b" * 40, "commit_sha": "b" * 40,
    }))
    assert loaded_paths and set(loaded_paths) == {str(folder)}
    assert getattr(encoder, "checkpoint_identity", None) == {
        "model": model, "repository": repository, "revision": "a" * 40,
    }
