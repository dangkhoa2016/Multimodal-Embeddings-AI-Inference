"""Validate both deployed dedicated GPU classes using an authorized local image."""

import argparse
import math
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
    args = parser.parse_args()
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
        result = worker.encode_many.remote(
            model_id, [{"kind": "image", "payload": image_bytes, "dimension": args.dimension}]
        )
        if not isinstance(result, dict) or len(result.get("vectors", [])) != 1:
            raise RuntimeError("Expected one profiled embedding")
        vector = result["vectors"][0]
        norm = math.sqrt(sum(value * value for value in vector))
        if len(vector) != args.dimension or not all(math.isfinite(x) for x in vector):
            raise RuntimeError(f"Invalid embedding for {model_id}")
        if abs(norm - 1.0) >= 0.001:
            raise RuntimeError(f"Invalid norm for {model_id}: {norm}")
        timing = result.get("timing", {})
        worker_id = timing.get("gpu_worker_id")
        if not isinstance(worker_id, str) or len(worker_id) != 32:
            raise RuntimeError(f"Missing GPU worker lifecycle ID for {model_id}")
        worker_ids[model_id] = worker_id
        print(
            f"REAL_PASS {model_id} dim={len(vector)} norm={norm:.5f} "
            f"state={timing.get('model_state')}",
            flush=True,
        )
    if len(worker_ids) == 2 and len(set(worker_ids.values())) != 2:
        raise RuntimeError("Expected independent worker lifecycle IDs")


if __name__ == "__main__":
    main()
