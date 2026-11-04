"""Unknown remote outcomes retain durable admission without another GPU dispatch."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from test_multitenant_admission import DurableMemory, active_permit_claims

from multimodal_embeddings_ai_inference.api import _InferenceQuota
from multimodal_embeddings_ai_inference.async_admission import AsyncSlotPool
from multimodal_embeddings_ai_inference.async_jobs import AsyncEmbeddingJobs, execute_job
from multimodal_embeddings_ai_inference.service import InferenceService
from multimodal_embeddings_ai_inference.validation import Prepared

MODEL = "wemm-embedding-9b"


def item(request_id):
    return Prepared(MODEL, "public example", "text", 64, "a" * 64, request_id)


@pytest.mark.parametrize("failure,cancel_ack", [
    ("spawn", "unavailable"), ("result", "error"), ("result", "accepted"),
])
def test_unknown_remote_outcome_keeps_slots_across_gateway_restart(
    monkeypatch, failure, cancel_ack,
):
    monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", "a" * 40)

    async def scenario():
        store, queued, remote_work = DurableMemory(), [], []

        async def coordinator_spawn(key, data):
            queued.append((key, data))

        async def gpu_spawn(model, inputs):
            remote_work.append({"active": True, "cancel_requested": False})
            work = remote_work[-1]
            if failure == "spawn":
                raise OSError("GPU queued but spawn acknowledgement lost")

            async def get():
                raise OSError("Result transport lost while GPU may run")

            async def cancel(**kwargs):
                work["cancel_requested"] = True
                if cancel_ack == "error":
                    raise OSError("Cancellation transport unavailable")
                # An acknowledgement accepts a cancellation request; it does
                # not establish that the remote computation has stopped.

            return SimpleNamespace(get=SimpleNamespace(aio=get),
                                   cancel=SimpleNamespace(aio=cancel))

        worker = SimpleNamespace(encode_many=SimpleNamespace(
            spawn=SimpleNamespace(aio=gpu_spawn),
        ))
        service = InferenceService("modal", {MODEL: worker})
        jobs = AsyncEmbeddingJobs(store, coordinator_spawn, AsyncSlotPool(
            store, global_limit=1, per_client_limit=1,
        ))
        quota = _InferenceQuota(120, 4)
        accepted, _ = await jobs.submit(item("unknown-remote-job-001"), quota, "alice")
        key = accepted["job_id"]
        await execute_job(*queued[0], store, service)
        record = store.data[key]
        assert record["state"] == "RUNNING" and record["outcome_uncertain"] is True
        assert "result" not in record
        assert all([await store.get(record["slots"][kind]) is not None
                    for kind in ("global", "client")])
        public, status = await jobs.status(key, "alice")
        assert public["state"] == "INDETERMINATE" and status == 200
        assert "result" not in public and "unknown" in public["note"]
        restarted = AsyncEmbeddingJobs(store, coordinator_spawn, AsyncSlotPool(
            store, global_limit=1, per_client_limit=1,
        ))
        duplicate, _ = await restarted.submit(item("unknown-remote-job-001"), quota, "alice")
        assert duplicate["state"] == "INDETERMINATE"
        with pytest.raises(HTTPException) as error:
            await restarted.submit(item("unknown-remote-job-002"), quota, "alice")
        assert error.value.status_code == 429
        await execute_job(*queued[0], store, service)
        assert len(queued) == len(remote_work) == 1 and remote_work[0]["active"]

    asyncio.run(scenario())


def test_confirmed_failure_before_remote_dispatch_releases_slots():
    async def scenario():
        store, queued = DurableMemory(), []

        async def spawn(key, data):
            queued.append((key, data))

        class InvalidProfile:
            backend = "modal"

            async def encode_many_profiled(self, items):
                raise ValueError("Rejected before any remote work")

        jobs = AsyncEmbeddingJobs(store, spawn, AsyncSlotPool(
            store, global_limit=1, per_client_limit=1,
        ))
        accepted, _ = await jobs.submit(item("confirmed-rejection-001"), _InferenceQuota(120, 4))
        await execute_job(*queued[0], store, InvalidProfile())
        assert store.data[accepted["job_id"]]["state"] == "FAILED"
        assert not active_permit_claims(store)

    asyncio.run(scenario())
