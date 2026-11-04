"""Smoke test of already-deployed Modal GPU class, no rebuild or API secrets."""

import math
from pathlib import Path

import modal

APP = "compactfaces-ai-inference"
CLS = "SharedL40SGpu"
DATA = Path("/tmp/compactfaces-core-0002-source.png")


def main():
    worker = modal.Cls.from_name(APP, CLS)()
    blob = DATA.read_bytes()
    for model_id in ("qwen3-vl-embedding-8b", "wemm-embedding-9b"):
        print(f"START {model_id}", flush=True)
        vectors = worker.encode_many.remote(
            model_id, [{"kind": "image", "payload": blob, "dimension": 1024}]
        )
        assert len(vectors) == 1
        values = vectors[0]
        norm = math.sqrt(sum(float(v) ** 2 for v in values))
        assert len(values) == 1024
        assert all(math.isfinite(v) for v in values)
        assert abs(norm - 1.0) < 0.001
        print(f"REAL_PASS {model_id} dim={len(values)} norm={norm:.5f}", flush=True)


if __name__ == "__main__":
    main()
