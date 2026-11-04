"""Resolve pre-downloaded, revision-checked checkpoints on the shared Modal Volume."""

import json
import os
import re
from pathlib import Path

MODEL_REPOSITORIES = {
    "wemm-embedding-9b": ("tencent/WeMM-Embedding-9B", "CF_WEMM9_REVISION"),
    "qwen3-vl-embedding-8b": ("Qwen/Qwen3-VL-Embedding-8B", "CF_QWEN8_REVISION"),
}
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def checkpoint_dir(model_id: str, root: str | Path | None = None) -> Path:
    repo, _ = MODEL_REPOSITORIES[model_id]
    base = Path(root or os.getenv("CF_MODEL_ROOT", "/models/checkpoints"))
    return base.joinpath(*repo.split("/"))


def checkpoint_info(
    model_id: str, requested_revision: str | None = None, root: str | Path | None = None
) -> dict | None:
    repo, env_name = MODEL_REPOSITORIES[model_id]
    revision = requested_revision or os.getenv(env_name, "main")
    folder = checkpoint_dir(model_id, root)
    marker = folder / ".compactfaces-snapshot.json"
    if not (marker.is_file() and (folder / "config.json").is_file()):
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if (
        data.get("repository") != repo
        or not isinstance(data.get("commit_sha"), str)
        or not SHA_PATTERN.fullmatch(data["commit_sha"])
    ):
        return None
    if revision not in (data.get("requested_revision"), data["commit_sha"]):
        return None
    return {"path": str(folder), "commit_sha": data["commit_sha"], "repository": repo}


def cached_checkpoint(
    model_id: str, requested_revision: str | None = None, root: str | Path | None = None
) -> str | None:
    info = checkpoint_info(model_id, requested_revision, root)
    return info["path"] if info else None


def require_checkpoint(model_id: str) -> str:
    result = cached_checkpoint(model_id)
    if not result:
        raise FileNotFoundError(
            f"Missing or mismatched {model_id} on shared Volume. "
            "Run: modal run modal_models.py --selection both"
        )
    return result
