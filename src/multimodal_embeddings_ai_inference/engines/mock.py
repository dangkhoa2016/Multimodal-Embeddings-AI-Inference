"""Deterministic fake engine for API-contract tests; NEVER for benchmark claims."""

import hashlib
import math

from ..embedding_config import QWEN_LEGACY_INSTRUCTION


class MockEncoder:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def encode(
        self, payload: bytes | str, kind: str, dimension: int, instruction: str | None = None,
        options: dict | None = None,
    ) -> list[float]:
        data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        seed_input = self.model_id.encode("utf-8") + b"\0" + data
        # Preserve exact historical mock embeddings when instruction is omitted.
        if instruction is not None and instruction != QWEN_LEGACY_INSTRUCTION:
            seed_input += b"\0instruction:" + instruction.encode("utf-8")
        if options:
            # Mock is deterministic and not a quality/latency proxy for CUDA.
            import json
            feature_options = {k: v for k, v in options.items() if k != "normalize"}
            if feature_options:
                seed_input += b"\0options:" + json.dumps(
                    feature_options, sort_keys=True
                ).encode("utf-8")
        seed = hashlib.sha256(seed_input).digest()
        numbers = []
        for i in range(dimension):
            block = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
            numbers.append(int.from_bytes(block[:4], "big") / 0xFFFFFFFF - 0.5)
        length = math.sqrt(sum(n * n for n in numbers))
        return [n / length for n in numbers] if (options or {}).get("normalize", True) else numbers
