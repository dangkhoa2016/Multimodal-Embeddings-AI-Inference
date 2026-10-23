"""Download each pinned model checkpoint to its own Modal Volume using CPU only."""

import json
import os
from pathlib import Path

import modal

app = modal.App("multimodal-embeddings-model-downloads")
qwen_volume_name = os.getenv(
    "VISION_EMBEDDINGS_QWEN_VOLUME_NAME", "multimodal-embeddings-qwen-weights"
)
wemm_volume_name = os.getenv(
    "VISION_EMBEDDINGS_WEMM_VOLUME_NAME", "multimodal-embeddings-wemm-weights"
)
qwen_volume = modal.Volume.from_name(qwen_volume_name, create_if_missing=True)
wemm_volume = modal.Volume.from_name(wemm_volume_name, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface-hub>=0.33,<2")


def _fetch_checkpoint(repository: str, revision: str, volume: modal.Volume) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    sha = HfApi().model_info(repository, revision=revision).sha
    if not isinstance(sha, str) or len(sha) != 40:
        raise RuntimeError(f"Cannot resolve immutable revision for {repository}")
    destination = Path("/models/checkpoints").joinpath(*repository.split("/"))
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".embedding-snapshot.json"
    marker.unlink(missing_ok=True)
    snapshot_download(repo_id=repository, revision=sha, local_dir=str(destination))
    if not (destination / "config.json").is_file():
        raise RuntimeError(f"Missing config.json for {repository}")
    marker.write_text(
        json.dumps(
            {"repository": repository, "requested_revision": revision, "commit_sha": sha},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    volume.commit()
    return {"repository": repository, "commit_sha": sha, "path": str(destination)}


@app.function(
    image=image, volumes={"/models": qwen_volume}, cpu=4, memory=16384, timeout=14400
)
def fetch_qwen(revision: str = "2c4565515e0f265c6511776e7193b22c0968ddc7") -> dict:
    return _fetch_checkpoint("Qwen/Qwen3-VL-Embedding-8B", revision, qwen_volume)


@app.function(
    image=image, volumes={"/models": wemm_volume}, cpu=4, memory=16384, timeout=14400
)
def fetch_wemm(revision: str = "00c52839de57a6d4fd5b78cf5522ccf0ac8ea482") -> dict:
    return _fetch_checkpoint("tencent/WeMM-Embedding-9B", revision, wemm_volume)


@app.local_entrypoint()
def main(
    selection: str = "both",
    qwen8_revision: str = "2c4565515e0f265c6511776e7193b22c0968ddc7",
    wemm9_revision: str = "00c52839de57a6d4fd5b78cf5522ccf0ac8ea482",
):
    if selection not in {"both", "qwen8", "wemm9"}:
        raise ValueError("selection must be both, qwen8 or wemm9")
    if selection in {"both", "qwen8"}:
        item = fetch_qwen.remote(qwen8_revision)
        print(f"READY {item['repository']} @ {item['commit_sha']} -> {qwen_volume_name}")
    if selection in {"both", "wemm9"}:
        item = fetch_wemm.remote(wemm9_revision)
        print(f"READY {item['repository']} @ {item['commit_sha']} -> {wemm_volume_name}")
    print("Pin the printed revisions in the application Secret for reproducible inference.")
