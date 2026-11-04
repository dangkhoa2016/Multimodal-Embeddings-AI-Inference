#!/usr/bin/env python3
"""End-to-end API smoke for two real models, one image, no benchmark claims."""

import argparse
import base64
import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path

MODELS = ("qwen3-vl-embedding-8b", "wemm-embedding-9b")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--allow-mock", action="store_true", help="Local test mode only")
    args = parser.parse_args()
    token = os.getenv("CF_API_TOKEN", "")
    if len(token) < 16:
        parser.error("CF_API_TOKEN environment variable is required")
    image_bytes = args.image.read_bytes()
    digest = hashlib.sha256(image_bytes).hexdigest()
    data = base64.b64encode(image_bytes).decode("ascii")
    for model in MODELS:
        body = json.dumps(
            {"model": model, "image_base64": data, "dimension": args.dimension}
        ).encode("utf-8")
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/embeddings",
            body,
            {"Content-Type": "application/json", "Authorization": "Bearer " + token},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1800) as response:
            result = json.load(response)
        assert result["model"] == model
        assert args.allow_mock or result["backend"] == "modal", "MOCK backend is not real inference"
        assert result["input_sha256"] == digest
        assert len(result["vector"]) == args.dimension
        norm = math.sqrt(sum(float(v) ** 2 for v in result["vector"]))
        assert abs(norm - 1.0) < 0.001
        print(
            json.dumps(
                {
                    "model": model,
                    "backend": result["backend"],
                    "dimension": args.dimension,
                    "revision": result["revision"],
                    "elapsed_ms": result["elapsed_ms"],
                    "input_sha256": digest,
                    "status": "PASS",
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
