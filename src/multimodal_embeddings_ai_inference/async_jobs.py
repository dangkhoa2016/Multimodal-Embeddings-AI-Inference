"""Durable, idempotent asynchronous embedding jobs.

The store holds no image payload or credentials; only metadata and a completed
response. A job is never automatically re-submitted after uncertain dispatch.
"""
import asyncio
import hashlib
import logging
import re
import time
from dataclasses import asdict

from fastapi import HTTPException

from .async_admission import AsyncSlotPool
from .catalog import CATALOG
from .embedding_config import embedding_metadata
from .schemas import EmbeddingResponse
from .service import InferenceOutcomeUnknown, model_revision
from .validation import Prepared

logger = logging.getLogger(__name__)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}$")

# Advisory states only: NEVER free an idempotency key or auto-dispatch again.
# Running jobs have a heartbeat; a cold/suspended worker may recover later.
JOB_HEARTBEAT_SECONDS = 30
RUNNING_STALE_SECONDS = 180
QUEUED_STALE_SECONDS = 24 * 60 * 60


def job_key(request_id: str, client_id: str = "owner") -> str:
    # Preserve legacy owner IDs across token rotation; other tenants are isolated.
    if client_id == "owner":
        material = "multimodal-embeddings-job-v1:" + request_id
    else:
        material = "multimodal-embeddings-job-v2:" + client_id + ":" + request_id
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fingerprint(prepared: Prepared, *, revision: str | None = None) -> str:
    """Bind one request ID to exactly one model, input, and encoding profile."""
    revision = model_revision(prepared.model) if revision is None else revision
    config = embedding_metadata(
        prepared.model, revision,
        prepared.dimension, prepared.instruction, prepared.options,
    )
    text = "|".join((
        prepared.model, revision,
        prepared.input_sha256, prepared.kind, str(prepared.dimension),
        config.config_sha256,
    ))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_response(prepared: Prepared, vector: list[float],
                   timing: dict | None, elapsed_ms: float, backend: str,
                   *, revision: str | None = None) -> dict:
    revision = model_revision(prepared.model) if revision is None else revision
    return EmbeddingResponse(
        model=prepared.model,
        repository=CATALOG[prepared.model].repository,
        revision=revision,
        backend=backend,
        dimension=prepared.dimension,
        vector=vector,
        input_sha256=prepared.input_sha256,
        request_id=prepared.request_id,
        elapsed_ms=round(elapsed_ms, 3),
        timing=timing,
        normalized=prepared.options.get("normalize", True),
        embedding_metadata=embedding_metadata(
            prepared.model, revision,
            prepared.dimension, prepared.instruction, prepared.options,
        ),
    ).model_dump(mode="json")


def public_job(job_id: str, record: dict) -> dict:
    state = record["state"]
    now = int(time.time())
    if ((state == "RUNNING" and (record.get("outcome_uncertain")
                                or now - record["updated_at"] > RUNNING_STALE_SECONDS))
            or (state == "QUEUED" and now - record["created_at"] > QUEUED_STALE_SECONDS)):
        # A stalled worker is NOT proof that GPU work stopped. This is a
        # read-only, reversible warning, not a destructive state transition.
        state = "INDETERMINATE"
    result = {
        "job_id": job_id,
        "state": state,
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }
    if record["state"] == "SUCCEEDED":
        result["result"] = record["result"]
    elif record["state"] == "FAILED":
        result["error"] = "Inference unavailable; this job will not be retried automatically"
    elif state == "QUEUED":
        result["note"] = "Dispatch may be pending. Poll this job; do not use a new request_id."
    elif state == "INDETERMINATE":
        result["note"] = (
            "Remote inference outcome is unknown. Inspect Modal logs; retain the job ID "
            "and its permits. Never auto-resubmit with a new ID."
        ) if record.get("outcome_uncertain") else (
            "Coordinator heartbeat or dispatch is overdue. Inspect Modal logs; "
            "the original GPU work might still run. Never auto-resubmit with a new ID."
        )
    return result


class AsyncEmbeddingJobs:
    """Dependency-injected durable store and one-shot Modal spawn.

    Store must support async get(key) and put(key, value, skip_if_exists).
    Spawn must be an async callable accepting (job_id, prepared_dict).
    """
    def __init__(self, store, spawn, admission: AsyncSlotPool | None = None):
        self.store = store
        self.spawn = spawn
        # Explicit production limits are injected by modal_app. Liberal local
        # defaults preserve existing isolated CPU harnesses.
        self.admission = admission or AsyncSlotPool(
            store, global_limit=64, per_client_limit=16
        )

    @staticmethod
    def _owned(record: dict, client_id: str) -> bool:
        # Pre-migration records belong exclusively to the legacy owner.
        return record.get("client_id", "owner") == client_id

    async def submit(
        self, prepared: Prepared, quota, client_id: str = "owner",
        *, distributed_quota=None,
    ) -> tuple[dict, int]:
        rid = prepared.request_id
        if not rid or not _ID_RE.fullmatch(rid):
            raise HTTPException(
                status_code=422,
                detail="Async inference requires a stable request_id (8-128 URL-safe characters)",
            )
        key = job_key(rid, client_id)
        revision = model_revision(prepared.model)
        fp = fingerprint(prepared, revision=revision)
        try:
            existing = await self.store.get(key)
        except Exception:
            logger.error("Async job registry lookup failed; details suppressed")
            raise HTTPException(
                status_code=503, detail="Job registry temporarily unavailable"
            ) from None
        if existing is not None:
            return self._existing(key, fp, existing, client_id)

        # Quota still applies to new async work, but duplicate reads are free.
        quota.acquire(1, client_id)
        try:
            # A duplicate already accepted above does not consume rate budget.
            # Debit before the durable job claim, otherwise a failed Redis
            # acknowledgement could strand a claimed job and cause unsafe retry.
            if distributed_quota is not None:
                await distributed_quota.admit(1, client_id, sync=False)
            # Conditional inserts guard actual outstanding work, not merely
            # submissions in progress. An uncertain slot remains occupied.
            try:
                slots = await self.admission.reserve(key, client_id)
            except HTTPException:
                raise
            except Exception:
                logger.error("Async permit backend unavailable; details suppressed")
                raise HTTPException(
                    status_code=503, detail="Job admission temporarily unavailable"
                ) from None
            now = int(time.time())
            record = {
                "fingerprint": fp, "revision": revision, "state": "QUEUED",
                "created_at": now, "updated_at": now,
                "client_id": client_id, "slots": slots,
            }
            try:
                claimed = await self.store.put(key, record, skip_if_exists=True)
            except Exception:
                # Claim ACK may be uncertain: preserve slots; do not dispatch.
                logger.error("Async registry claim uncertain; permits preserved")
                raise HTTPException(
                    status_code=503, detail="Job registry temporarily unavailable"
                ) from None
            if not claimed:
                # Job record belongs to another successful claimant. Release
                # ONLY this request's extra, never-dispatched permit pair.
                await self.admission.release(slots, key)
                existing = await self.store.get(key)
                if existing is None:
                    raise HTTPException(
                        status_code=503, detail="Job registry temporarily unavailable"
                    )
                return self._existing(key, fp, existing, client_id)

            try:
                # The only dispatch of this request_id. Retain admission while
                # waiting for ACK so slow control-plane calls cannot fill the gateway.
                await self.spawn(key, asdict(prepared))
            except Exception:
                logger.error("Async job dispatch acknowledgement uncertain; job retained")
        finally:
            quota.release(client_id)
        try:
            record = await self.store.get(key) or record
        except Exception:
            # Claim and dispatch already exist; client can poll stable ID.
            record = record
        result = public_job(key, record)
        return result, 200 if result["state"] in ("SUCCEEDED", "FAILED", "INDETERMINATE") else 202

    @staticmethod
    def _existing(key: str, fp: str, record: dict,
                  client_id: str = "owner") -> tuple[dict, int]:
        if not AsyncEmbeddingJobs._owned(record, client_id):
            raise HTTPException(status_code=404, detail="Job not found")
        if record.get("fingerprint") != fp:
            raise HTTPException(
                status_code=409, detail="request_id already used for different input or profile"
            )
        result = public_job(key, record)
        return result, 200 if result["state"] in ("SUCCEEDED", "FAILED", "INDETERMINATE") else 202

    async def status_by_request(
        self, request_id: str, client_id: str = "owner",
    ) -> tuple[dict, int]:
        """Read-only recovery after lost POST acknowledgement; never spawn GPU.

        The server derives the tenant-scoped job ID, so clients need not know
        their internal client ID or the job-key hashing implementation.
        """
        if not _ID_RE.fullmatch(request_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return await self.status(job_key(request_id, client_id), client_id)

    async def status(self, key: str, client_id: str = "owner") -> tuple[dict, int]:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            record = await self.store.get(key)
        except Exception:
            logger.error("Async status store unavailable; details suppressed")
            raise HTTPException(
                status_code=503, detail="Job registry temporarily unavailable"
            ) from None
        if record is None or not self._owned(record, client_id):
            raise HTTPException(status_code=404, detail="Job not found or expired")
        result = public_job(key, record)
        return result, 200 if result["state"] in ("SUCCEEDED", "FAILED", "INDETERMINATE") else 202


async def execute_job(job_id: str, prepared_data: dict, store, service) -> None:
    """Executed inside a separate CPU Modal function, not the HTTP gateway."""
    record = await store.get(job_id)
    if record is None or record.get("state") != "QUEUED":
        return
    # Claim a coordinator start exactly once, even if the provider spawns
    # duplicate workers for a single job. A lost acknowledgement is uncertain:
    # never let another coordinator create a second GPU dispatch.
    start_key = "job:start:v1:" + job_id
    if not await store.put(
        start_key, {"job_id": job_id, "state": "STARTED", "created_at": int(time.time())},
        skip_if_exists=True,
    ):
        return
    record = {**record, "state": "RUNNING", "updated_at": int(time.time())}
    await store.put(job_id, record)

    stop = asyncio.Event()

    async def heartbeat():
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=JOB_HEARTBEAT_SECONDS)
                break
            except TimeoutError:
                pass
            try:
                current = await store.get(job_id)
                if not current or current.get("state") != "RUNNING":
                    break
                await store.put(job_id, {
                    **current, "updated_at": int(time.time()),
                })
            except Exception:
                logger.error("Async heartbeat unavailable; details suppressed")

    monitor = asyncio.create_task(heartbeat())
    outcome = "SUCCEEDED"
    try:
        prepared = Prepared(**prepared_data)
        revision = model_revision(prepared.model)
        if (record.get("revision", revision) != revision
                or record.get("fingerprint") != fingerprint(prepared, revision=revision)):
            raise ValueError("Coordinator profile differs from the accepted job")
        started = time.perf_counter()
        vectors, timings = await service.encode_many_profiled([prepared])
        result = build_response(
            prepared, vectors[0], timings[0],
            (time.perf_counter() - started) * 1000, service.backend, revision=revision,
        )
    except InferenceOutcomeUnknown:
        logger.error("Async remote outcome unknown; permits retained")
        outcome = "INDETERMINATE"
    except Exception:
        logger.error("Async inference failed; provider details suppressed")
        outcome = "FAILED"
    finally:
        # Join any in-flight heartbeat write BEFORE persisting a terminal
        # state: a heartbeat may hold a stale RUNNING snapshot from store.get.
        # On cancellation the outcome remains uncertain, not falsely FAILED.
        stop.set()
        await monitor

    if outcome == "INDETERMINATE":
        await store.put(job_id, {
            **record, "outcome_uncertain": True, "updated_at": int(time.time())
        })
        return
    if outcome == "FAILED":
        await store.put(job_id, {
            **record, "state": "FAILED", "updated_at": int(time.time())
        })
        if record.get("slots"):
            await AsyncSlotPool(store).release(record["slots"], job_id)
        return
    await store.put(job_id, {
        **record, "state": "SUCCEEDED",
        "result": result, "updated_at": int(time.time()),
    })
    if record.get("slots"):
        await AsyncSlotPool(store).release(record["slots"], job_id)
