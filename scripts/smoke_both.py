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


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the application's Bearer token to a redirected endpoint.
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--allow-mock", action="store_true", help="Local test mode only")
    parser.add_argument(
        "--qwen-revision", default=os.getenv("VISION_EMBEDDINGS_QWEN8_REVISION", ""),
        help="Expected 40-character checkpoint SHA; defaults to VISION_EMBEDDINGS_QWEN8_REVISION",
    )
    parser.add_argument(
        "--wemm-revision", default=os.getenv("VISION_EMBEDDINGS_WEMM9_REVISION", ""),
        help="Expected 40-character checkpoint SHA; defaults to VISION_EMBEDDINGS_WEMM9_REVISION",
    )
    args = parser.parse_args()
    expected_revisions = {
        MODELS[0]: args.qwen_revision.strip().lower(),
        MODELS[1]: args.wemm_revision.strip().lower(),
    }
    for model, revision in expected_revisions.items():
        if (revision or not args.allow_mock) and (
            len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision)
        ):
            parser.error(f"Expected pinned 40-character checkpoint revision for {model}")
    token = os.getenv("VISION_EMBEDDINGS_API_TOKEN", "")
    if len(token) < 16:
        parser.error("VISION_EMBEDDINGS_API_TOKEN environment variable is required")
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
        with urllib.request.build_opener(RejectRedirects).open(req, timeout=1800) as response:
            result = json.load(response)
        expected_revision = expected_revisions[model]
        if expected_revision and result.get("revision") != expected_revision:
            raise ValueError(f"Unexpected checkpoint revision for {model}")
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
                    "revision_verified": bool(expected_revision),
                    "elapsed_ms": result["elapsed_ms"],
                    "input_sha256": digest,
                    "status": "PASS",
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
