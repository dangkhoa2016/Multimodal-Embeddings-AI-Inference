import json

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
