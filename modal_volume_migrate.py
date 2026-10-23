"""Copy verified model checkpoints from a legacy Volume into separate Modal Volumes.

Run with a detached Modal App for transfers larger than a local CLI time window:

  VISION_EMBEDDINGS_SOURCE_VOLUME_NAME=old-volume modal run --detach modal_volume_migrate.py

The original Volume is never modified. Only a verified, completed destination
receives .embedding-snapshot.json; consumers fail closed without this marker.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import modal

app = modal.App("multimodal-embeddings-volume-migration")

source_name = os.getenv("VISION_EMBEDDINGS_SOURCE_VOLUME_NAME", "").strip()
if not source_name:
    raise ValueError("Set VISION_EMBEDDINGS_SOURCE_VOLUME_NAME before running this script")

qwen_name = os.getenv(
    "VISION_EMBEDDINGS_QWEN_VOLUME_NAME", "multimodal-embeddings-qwen-weights"
)
wemm_name = os.getenv(
    "VISION_EMBEDDINGS_WEMM_VOLUME_NAME", "multimodal-embeddings-wemm-weights"
)
if len({source_name, qwen_name, wemm_name}) != 3:
    raise ValueError("Source and destination Volume names must be distinct")

source_volume = modal.Volume.from_name(source_name, create_if_missing=False)
qwen_volume = modal.Volume.from_name(qwen_name, create_if_missing=True)
wemm_volume = modal.Volume.from_name(wemm_name, create_if_missing=True)
cpu_image = modal.Image.debian_slim(python_version="3.11").env(
    {
        "VISION_EMBEDDINGS_SOURCE_VOLUME_NAME": source_name,
        "VISION_EMBEDDINGS_QWEN_VOLUME_NAME": qwen_name,
        "VISION_EMBEDDINGS_WEMM_VOLUME_NAME": wemm_name,
    }
)

REVISIONS = {
    "Qwen/Qwen3-VL-Embedding-8B": "2c4565515e0f265c6511776e7193b22c0968ddc7",
    "tencent/WeMM-Embedding-9B": "00c52839de57a6d4fd5b78cf5522ccf0ac8ea482",
}
CHUNK = 8 * 1024 * 1024


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(source: Path, repo: str) -> dict:
    markers = [
        x for x in source.iterdir()
        if x.name.endswith("-snapshot.json") or x.name == ".embedding-snapshot.json"
    ]
    valid = []
    for marker in markers:
        try:
            item = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if item.get("repository") == repo and item.get("commit_sha") == REVISIONS[repo]:
            valid.append(item)
    if len(valid) != 1:
        raise RuntimeError(f"Expected one verified source snapshot marker for {repo}")
    if not (source / "config.json").is_file():
        raise RuntimeError(f"Missing model config.json for {repo}")
    return valid[0]


def _copy_model(repo: str, volume: modal.Volume, destination_name: str) -> dict:
    relative = Path("checkpoints").joinpath(*repo.split("/"))
    src = Path("/source") / relative
    dst = Path("/target") / relative
    if not src.is_dir():
        raise RuntimeError(f"Source checkpoint missing: {repo}")
    metadata = _metadata(src, repo)
    all_files = sorted(
        (
            file
            for file in src.rglob("*")
            if file.is_file() and not file.name.endswith("-snapshot.json")
        ),
        key=lambda file: str(file.relative_to(src)),
    )
    if not all_files:
        raise RuntimeError(f"No model files available for {repo}")
    print(f"START {repo} files={len(all_files)}", flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for number, file in enumerate(all_files, 1):
        rel = file.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = _sha256(file)
        if target.is_file() and target.stat().st_size == file.stat().st_size:
            if _sha256(target) == expected_hash:
                print(f"SKIP_VERIFIED {repo} {number}/{len(all_files)} {rel}", flush=True)
                total_bytes += file.stat().st_size
                continue
        started = time.monotonic()
        part = target.with_name(target.name + ".copying")
        if part.exists():
            part.unlink()
        with file.open("rb") as reader, part.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=CHUNK)
        if _sha256(part) != expected_hash:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch after transfer: {repo}/{rel}")
        part.replace(target)
        volume.commit()
        total_bytes += file.stat().st_size
        print(
            f"VERIFIED {repo} {number}/{len(all_files)} {rel} "
            f"bytes={file.stat().st_size} seconds={time.monotonic()-started:.1f}",
            flush=True,
        )
    marker = dst / ".embedding-snapshot.json"
    marker.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print(f"COMPLETE {repo} files={len(all_files)} bytes={total_bytes}", flush=True)
    return {
        "repository": repo,
        "target_volume": destination_name,
        "revision": metadata["commit_sha"],
        "verified_files": len(all_files),
        "verified_bytes": total_bytes,
    }


@app.function(
    image=cpu_image,
    volumes={"/source": source_volume, "/target": qwen_volume},
    cpu=4,
    memory=16384,
    timeout=14400,
    max_containers=1,
)
def migrate_qwen():
    return _copy_model("Qwen/Qwen3-VL-Embedding-8B", qwen_volume, qwen_name)


@app.function(
    image=cpu_image,
    volumes={"/source": source_volume, "/target": wemm_volume},
    cpu=4,
    memory=16384,
    timeout=14400,
    max_containers=1,
)
def migrate_wemm():
    return _copy_model("tencent/WeMM-Embedding-9B", wemm_volume, wemm_name)


@app.local_entrypoint()
def main(selection: str = "both"):
    if selection not in ("both", "qwen", "wemm"):
        raise ValueError("selection must be both, qwen or wemm")
    if selection in ("both", "qwen"):
        call = migrate_qwen.spawn()
        print(f"QWEN_SPAWNED {call.object_id}", flush=True)
    if selection in ("both", "wemm"):
        call = migrate_wemm.spawn()
        print(f"WEMM_SPAWNED {call.object_id}", flush=True)
    print("Detached transfers are resumable. Do not deploy until both volumes verify.", flush=True)
