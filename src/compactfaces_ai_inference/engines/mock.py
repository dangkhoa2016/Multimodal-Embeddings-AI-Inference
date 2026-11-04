"""Deterministic fake engine for API-contract tests; NEVER for benchmark claims."""

import hashlib
import math


class MockEncoder:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def encode(self, payload: bytes | str, kind: str, dimension: int) -> list[float]:
        data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        seed = hashlib.sha256(self.model_id.encode("utf-8") + b"\0" + data).digest()
        numbers = []
        for i in range(dimension):
            block = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
            numbers.append(int.from_bytes(block[:4], "big") / 0xFFFFFFFF - 0.5)
        length = math.sqrt(sum(n * n for n in numbers))
        return [n / length for n in numbers]
