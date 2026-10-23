"""Route model groups to separate dedicated Modal GPU workers."""

import asyncio
import logging
import os
import time
from collections import defaultdict

from .checkpoints import SHA_PATTERN, validate_checkpoint_identity
from .model_options import effective_encoding_profile
from .validation import Prepared
from .vector_utils import normalize_vector

logger = logging.getLogger(__name__)


class InferenceOutcomeUnknown(RuntimeError):
    """Remote work may still exist; a cancellation request is not termination proof."""


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
        self._cleanup_tasks: set[asyncio.Task] = set()

    def _track_cleanup(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._cleanup_tasks.add(task)

        def finished(completed):
            self._cleanup_tasks.discard(completed)
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    logger.warning("GPU call cleanup failed (%s)", type(error).__name__)

        task.add_done_callback(finished)
        return task

    async def _cancel_call(self, call):
        # Workers accept one input at a time. Terminating this owned call's
        # container cannot interrupt a different concurrent input on that worker.
        task = self._track_cleanup(call.cancel.aio(terminate_containers=True))
        try:
            await asyncio.wait_for(asyncio.shield(task), 5)
        except (Exception, asyncio.CancelledError):
            # Keep a strong reference so a slow control-plane cancellation can
            # finish after the HTTP response, without extending its wait forever.
            pass

    async def _cancel_late_spawn(self, spawn):
        try:
            call = await asyncio.shield(spawn)
        except (Exception, asyncio.CancelledError):
            return
        await self._cancel_call(call)

    async def _remote_result(self, worker, model_id, inputs):
        # A cancelled spawn acknowledgement must not orphan an already-enqueued
        # input. Retain its handle and cancel it when the acknowledgement arrives.
        spawn = asyncio.create_task(worker.encode_many.spawn.aio(model_id, inputs))
        try:
            call = await asyncio.shield(spawn)
        except BaseException as error:
            self._track_cleanup(self._cancel_late_spawn(spawn))
            if isinstance(error, Exception):
                raise InferenceOutcomeUnknown("Remote inference outcome is unknown") from None
            raise
        try:
            return await call.get.aio()
        except BaseException as error:
            try:
                await self._cancel_call(call)
            except Exception:
                logger.warning("GPU call cancellation could not be requested")
            if isinstance(error, Exception):
                raise InferenceOutcomeUnknown("Remote inference outcome is unknown") from None
            raise

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
            kwargs = {}
            if item.instruction is not None:
                kwargs["instruction"] = item.instruction
            if item.options:
                kwargs["options"] = item.options
            return await asyncio.to_thread(
                self._local_models[item.model].encode, *arguments, **kwargs
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
            expected_revision = model_revision(model_id)
            if not SHA_PATTERN.fullmatch(expected_revision):
                raise ValueError("Modal inference requires a pinned checkpoint revision")
            inputs = []
            for _, prepared in group:
                item = {
                    "payload": prepared.payload,
                    "kind": prepared.kind,
                    "dimension": prepared.dimension,
                    "expected_revision": expected_revision,
                }
                if prepared.instruction is not None:
                    item["instruction"] = prepared.instruction
                if prepared.options:
                    item["model_config"] = prepared.options
                inputs.append(item)
            started = time.perf_counter()
            remote_result = await self._remote_result(
                self.remote_workers[model_id], model_id, inputs
            )
            round_trip_ms = (time.perf_counter() - started) * 1000.0
            validate_checkpoint_identity(
                (remote_result.get("checkpoint_identity")
                 if isinstance(remote_result, dict) else None),
                model_id, expected_revision,
            )
            vectors = (
                remote_result.get("vectors") if isinstance(remote_result, dict) else remote_result
            )
            detail = remote_result.get("timing") if isinstance(remote_result, dict) else None
            if not isinstance(vectors, list) or len(vectors) != len(group):
                raise ValueError("GPU returned incorrect number of embeddings")
            # Confirm exactly what this subprocess processed for every item.
            expected_profiles = [
                effective_encoding_profile(
                    model_id, kind=prepared.kind, dimension=prepared.dimension,
                    instruction=prepared.instruction, options=prepared.options,
                )
                for _, prepared in group
            ]
            applied = (
                remote_result.get("effective_profiles")
                if isinstance(remote_result, dict) else None
            )
            if not isinstance(applied, list) or applied != expected_profiles:
                raise ValueError("GPU worker effective encoding profile mismatch")
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
                entries.append((
                    index,
                    normalize_vector(
                        values, prepared.dimension,
                        normalize=prepared.options.get("normalize", True),
                    ),
                    metrics,
                ))
            return entries

        # Independent GPU workers allow cross-model groups to run in parallel.
        tasks = [asyncio.create_task(dispatch(model_id, group))
                 for model_id, group in by_model.items()]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            # A failed group must not leave another model's work running for an
            # HTTP batch that can no longer return a complete result.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
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
