"""Validate deployed GPU embeddings with an authorized local image (no API token needed)."""

import argparse
import math
from pathlib import Path

import modal

MODELS = ("qwen3-vl-embedding-8b", "wemm-embedding-9b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Authorized local image path")
    parser.add_argument("--app", default="compactfaces-ai-inference")
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--model", choices=("both", *MODELS), default="both")
    parser.add_argument(
        "--switch-back",
        action="store_true",
        help="After both models, rerun Qwen to verify WeMM-to-Qwen switch",
    )
    args = parser.parse_args()

    image_bytes = args.image.read_bytes()
    if not image_bytes:
        parser.error("The image is empty")
    if args.dimension <= 0:
        parser.error("dimension must be positive")

    worker = modal.Cls.from_name(args.app, "SharedL40SGpu")()
    models = MODELS if args.model == "both" else (args.model,)
    if args.switch_back and args.model == "both":
        models = (*models, MODELS[0])

    for model_id in models:
        print(f"START {model_id}", flush=True)
        vectors = worker.encode_many.remote(
            model_id, [{"kind": "image", "payload": image_bytes, "dimension": args.dimension}]
        )
        if len(vectors) != 1:
            raise RuntimeError("Expected one embedding")
        vector = vectors[0]
        norm = math.sqrt(sum(value * value for value in vector))
        if len(vector) != args.dimension or not all(math.isfinite(x) for x in vector):
            raise RuntimeError(f"Invalid embedding for {model_id}")
        if abs(norm - 1.0) >= 0.001:
            raise RuntimeError(f"Invalid norm for {model_id}: {norm}")
        print(f"REAL_PASS {model_id} dim={len(vector)} norm={norm:.5f}", flush=True)


if __name__ == "__main__":
    main()
