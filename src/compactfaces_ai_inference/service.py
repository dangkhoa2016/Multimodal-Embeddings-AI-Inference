"""Model routing; the Modal remote worker is ONE L40S per deployment."""

import asyncio
import logging
import os
from collections import defaultdict

from .validation import Prepared
from .vector_utils import normalize_vector

logger = logging.getLogger(__name__)


def model_revision(model_id: str) -> str:
    key = {
        "wemm-embedding-9b": "CF_WEMM9_REVISION",
        "qwen3-vl-embedding-8b": "CF_QWEN8_REVISION",
    }[model_id]
    return os.getenv(key, "main")


class InferenceService:
    def __init__(self, backend: str, remote_worker=None):
        self.backend = backend
        self.remote_worker = remote_worker
        self._local_models: dict = {}
        self._locks = defaultdict(asyncio.Lock)

    def _load_local(self, model_id: str):
        if self.backend == "mock":
            from .engines.mock import MockEncoder

            return MockEncoder(model_id)
        if model_id == "wemm-embedding-9b":
            from .engines.wemm import WeMMEncoder

            return WeMMEncoder()
        from .engines.qwen import QwenVLEncoder

        return QwenVLEncoder()

    async def _encode_local(self, item: Prepared) -> list[float]:
        async with self._locks[item.model]:
            if item.model not in self._local_models:
                self._local_models[item.model] = await asyncio.to_thread(
                    self._load_local, item.model
                )
            return await asyncio.to_thread(
                self._local_models[item.model].encode,
                item.payload,
                item.kind,
                item.dimension,
            )

    async def encode_many(self, items: list[Prepared]) -> list[list[float]]:
        if not items:
            return []
        if self.backend != "modal":
            return [await self._encode_local(item) for item in items]
        if self.remote_worker is None:
            raise RuntimeError("Modal worker missing")
        output: list[list[float] | None] = [None] * len(items)
        by_model: dict[str, list[tuple[int, Prepared]]] = defaultdict(list)
        for index, item in enumerate(items):
            by_model[item.model].append((index, item))
        # Modal max_containers=1 makes these model groups execute on ONE GPU.
        for model_id, group in by_model.items():
            inputs = [
                {"payload": p.payload, "kind": p.kind, "dimension": p.dimension} for _, p in group
            ]
            vectors = await self.remote_worker.encode_many.remote.aio(model_id, inputs)
            if len(vectors) != len(group):
                raise ValueError("GPU returned incorrect number of embeddings")
            for (index, item), values in zip(group, vectors, strict=True):
                output[index] = normalize_vector(values, item.dimension)
        return [vector for vector in output if vector is not None]
