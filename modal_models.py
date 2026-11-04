"""Download BOTH official model snapshots to ONE shared Modal Volume (CPU only)."""

import json
import os
from pathlib import Path

import modal

app = modal.App("compactfaces-ai-model-downloads")
volume_name = os.getenv("CF_MODEL_VOLUME_NAME", "compactfaces-ai-weights")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)
mount = "/models"
checkpoints = Path(mount) / "checkpoints"
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface-hub>=0.33,<2")
SOURCES = {
    "wemm9": "tencent/WeMM-Embedding-9B",
    "qwen8": "Qwen/Qwen3-VL-Embedding-8B",
}


@app.function(image=image, volumes={mount: volume}, cpu=4, memory=16384, timeout=14400)
def fetch_models(selection: str = "both", revisions: dict[str, str] | None = None) -> list[dict]:
    from huggingface_hub import HfApi, snapshot_download

    groups = {"both": ("wemm9", "qwen8"), "wemm9": ("wemm9",), "qwen8": ("qwen8",)}
    if selection not in groups:
        raise ValueError(f"Invalid selection: {selection}")
    revisions = revisions or {}
    api = HfApi()
    results = []
    for key in groups[selection]:
        repo = SOURCES[key]
        rev = revisions.get(key, "main")
        commit = api.model_info(repo, revision=rev).sha
        if not commit or len(commit) != 40:
            raise RuntimeError(f"Cannot resolve SHA for {repo}")
        dest = checkpoints.joinpath(*repo.split("/"))
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".compactfaces-snapshot.json"
        marker.unlink(missing_ok=True)
        # Single writer, sequential commits. Public model weights not included in source release.
        snapshot_download(repo_id=repo, revision=commit, local_dir=str(dest))
        if not (dest / "config.json").is_file():
            raise RuntimeError(f"Missing config.json for {repo}")
        marker.write_text(
            json.dumps(
                {
                    "repository": repo,
                    "requested_revision": rev,
                    "commit_sha": commit,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        volume.commit()
        results.append({"repository": repo, "commit_sha": commit, "path": str(dest)})
    return results


@app.local_entrypoint()
def main(selection: str = "both", wemm9_revision: str = "main", qwen8_revision: str = "main"):
    for item in fetch_models.remote(
        selection,
        {
            "wemm9": wemm9_revision,
            "qwen8": qwen8_revision,
        },
    ):
        print(f"READY {item['repository']} @ {item['commit_sha']} -> {item['path']}")
    print(f"SHARED VOLUME: {volume_name}")
    print("Copy exact revision hashes to the Modal Secret for reproducible response metadata.")
