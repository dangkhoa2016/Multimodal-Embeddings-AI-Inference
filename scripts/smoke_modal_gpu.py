"""Validate both deployed dedicated GPU classes using an authorized local image."""

import argparse
import math
import re
from pathlib import Path

import modal

WORKERS = {
    "qwen3-vl-embedding-8b": "QwenL40SGpu",
    "wemm-embedding-9b": "WeMML40SGpu",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Authorized local image path")
    parser.add_argument("--app", default="multimodal-embeddings-ai-inference")
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--model", choices=("both", *WORKERS), default="both")
    parser.add_argument("--text", help="Optional text query for a second real embedding")
    parser.add_argument("--qwen-revision", help="Require this exact pinned Qwen revision")
    parser.add_argument("--wemm-revision", help="Require this exact pinned WeMM revision")
    args = parser.parse_args()
    for name, revision in (("Qwen", args.qwen_revision), ("WeMM", args.wemm_revision)):
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
            parser.error(f"{name} expected revision must be a 40-character lowercase SHA")
    image_bytes = args.image.read_bytes()
    if not image_bytes:
        parser.error("The image is empty")
    if args.dimension <= 0:
        parser.error("dimension must be positive")

    models = WORKERS if args.model == "both" else (args.model,)
    worker_ids = {}
    for model_id in models:
        worker = modal.Cls.from_name(args.app, WORKERS[model_id])()
        print(f"START {model_id}", flush=True)
        items = [{"kind": "image", "payload": image_bytes, "dimension": args.dimension}]
        if args.text is not None:
            if not args.text.strip():
                parser.error("--text must contain a non-blank query")
            items.append({"kind": "text", "payload": args.text, "dimension": args.dimension})
        result = worker.encode_many.remote(model_id, items)
        if not isinstance(result, dict) or len(result.get("vectors", [])) != len(items):
            raise RuntimeError("Incorrect number of model embeddings")
        identity = result.get("checkpoint_identity")
        if not isinstance(identity, dict) or identity.get("model") != model_id:
            raise RuntimeError(f"Unverified model identity for {model_id}")
        expected_revision = (
            args.qwen_revision if model_id == "qwen3-vl-embedding-8b"
            else args.wemm_revision
        )
        if expected_revision and identity.get("revision") != expected_revision:
            raise RuntimeError(f"GPU checkpoint revision mismatch for {model_id}")
        norms = []
        for item, vector in zip(items, result["vectors"], strict=True):
            if len(vector) != args.dimension or not all(math.isfinite(x) for x in vector):
                raise RuntimeError(f"Invalid {item['kind']} embedding for {model_id}")
            norm = math.sqrt(sum(value * value for value in vector))
            if abs(norm - 1.0) >= 0.001:
                raise RuntimeError(f"Invalid {item['kind']} norm for {model_id}: {norm}")
            norms.append(norm)
        timing = result.get("timing", {})
        worker_id = timing.get("gpu_worker_id")
        if not isinstance(worker_id, str) or len(worker_id) != 32:
            raise RuntimeError(f"Missing GPU worker lifecycle ID for {model_id}")
        worker_ids[model_id] = worker_id
        print(
            f"REAL_PASS {model_id} modalities={','.join(item['kind'] for item in items)} "
            f"dim={args.dimension} norms={','.join(f'{norm:.5f}' for norm in norms)} "
            f"state={timing.get('model_state')}",
            flush=True,
        )
    if len(worker_ids) == 2 and len(set(worker_ids.values())) != 2:
        raise RuntimeError("Expected independent worker lifecycle IDs")


if __name__ == "__main__":
    main()
