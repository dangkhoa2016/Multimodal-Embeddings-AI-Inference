#!/usr/bin/env python3
"""Check a live endpoint without ever claiming mock results are real embeddings."""

import argparse
import base64
import hashlib
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="wemm-embedding-9b")
    p.add_argument("--image", type=Path)
    p.add_argument("--text")
    p.add_argument("--dimension", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()
    if (args.image is None) == (args.text is None):
        p.error("Specify exactly one of --image or --text")
    token = os.environ.get("CF_API_TOKEN", "")
    if not token:
        print("CF_API_TOKEN is required", file=sys.stderr)
        return 2
    body: dict = {"model": args.model, "dimension": args.dimension}
    if args.image:
        data = args.image.read_bytes()
        body["image_base64"] = base64.b64encode(data).decode("ascii")
    else:
        data = args.text.encode("utf-8")
        body["text"] = args.text
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/embeddings",
        method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"HTTP error {exc.code}: {exc.read(512).decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    vec = result.get("vector", [])
    length = math.sqrt(sum(float(n) * float(n) for n in vec))
    assert len(vec) == args.dimension, "Wrong output dimension"
    assert abs(length - 1.0) < 0.001, "Vector is not L2-normalized"
    assert result["input_sha256"] == hashlib.sha256(data).hexdigest(), "Input digest mismatch"
    print(
        json.dumps(
            {
                "pass": True,
                "backend": result["backend"],
                "model": result["model"],
                "revision": result["revision"],
                "dimension": len(vec),
                "input_sha256": result["input_sha256"],
                "elapsed_ms": result["elapsed_ms"],
                "warning": "MOCK OUTPUT, NOT AN AI BENCHMARK"
                if result["backend"] == "mock"
                else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
