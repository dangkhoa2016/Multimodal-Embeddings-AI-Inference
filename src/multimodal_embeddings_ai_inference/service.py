"""Route model groups to separate dedicated Modal GPU workers."""

import asyncio
import logging
import os
import time
from collections import defaultdict

from .validation import Prepared
from .vector_utils import normalize_vector

logger = logging.getLogger(__name__)


def model_revision(model_id: str) -> str:
    key = {
        "wemm-embedding-9b": "VISION_EMBEDDINGS_WEMM9_REVISION",
        "qwen3-vl-embedding-8b": "VISION_EMBEDDINGS_QWEN8_REVISION",
    }[model_id]
    return os.getenv(key, "main")


class InferenceService:
    def __init__(self, backend: str, remote_workers=None):
        self.backend = backend
        self.remote_workers = remote_workers
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
            arguments = (item.payload, item.kind, item.dimension)
            if item.instruction is None:
                return await asyncio.to_thread(self._local_models[item.model].encode, *arguments)
            return await asyncio.to_thread(
                self._local_models[item.model].encode,
                *arguments,
                instruction=item.instruction,
            )

    async def encode_many_profiled(
        self, items: list[Prepared]
    ) -> tuple[list[list[float]], list[dict | None]]:
        if not items:
            return [], []
        if self.backend != "modal":
            return [await self._encode_local(item) for item in items], [None] * len(items)
        if not isinstance(self.remote_workers, dict):
            raise RuntimeError("Modal routing requires a model-to-worker mapping")
        output: list[list[float] | None] = [None] * len(items)
        timings: list[dict | None] = [None] * len(items)
        by_model: dict[str, list[tuple[int, Prepared]]] = defaultdict(list)
        for index, item in enumerate(items):
            by_model[item.model].append((index, item))

        async def dispatch(model_id: str, group: list[tuple[int, Prepared]]):
            if model_id not in self.remote_workers:
                raise ValueError(f"No Modal GPU worker is assigned to {model_id}")
            inputs = []
            for _, prepared in group:
                item = {
                    "payload": prepared.payload,
                    "kind": prepared.kind,
                    "dimension": prepared.dimension,
                }
                if prepared.instruction is not None:
                    item["instruction"] = prepared.instruction
                inputs.append(item)
            started = time.perf_counter()
            remote_result = await self.remote_workers[model_id].encode_many.remote.aio(
                model_id, inputs
            )
            round_trip_ms = (time.perf_counter() - started) * 1000.0
            vectors = (
                remote_result.get("vectors") if isinstance(remote_result, dict) else remote_result
            )
            detail = remote_result.get("timing") if isinstance(remote_result, dict) else None
            if not isinstance(vectors, list) or len(vectors) != len(group):
                raise ValueError("GPU returned incorrect number of embeddings")
            entries = []
            for (index, prepared), values in zip(group, vectors, strict=True):
                metrics = None
                if isinstance(detail, dict):
                    metrics = detail.copy()
                    worker_ms = metrics.get("gpu_worker_ms")
                    metrics["gpu_dispatch_overhead_ms"] = (
                        round(max(0.0, round_trip_ms - worker_ms), 3)
                        if isinstance(worker_ms, (int, float))
                        else None
                    )
                entries.append((index, normalize_vector(values, prepared.dimension), metrics))
            return entries

        # Independent GPU workers allow cross-model groups to run in parallel.
        results = await asyncio.gather(
            *(dispatch(model_id, group) for model_id, group in by_model.items())
        )
        for group in results:
            for index, vector, timing in group:
                output[index] = vector
                timings[index] = timing
        if any(values is None for values in output):
            raise ValueError("Missing GPU embeddings")
        return [values for values in output if values is not None], timings

    async def encode_many(self, items: list[Prepared]) -> list[list[float]]:
        vectors, _ = await self.encode_many_profiled(items)
        return vectors
