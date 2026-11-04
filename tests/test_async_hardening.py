"""CPU-only stress and chaos tests for durable paid-job admission."""
import asyncio
import time

import httpx
import pytest
from test_multitenant_admission import (
    ALICE,
    BOB,
    DurableMemory,
    active_permit_claims,
    app_with,
    auth,
    payload,
    settings,
)

from multimodal_embeddings_ai_inference.async_admission import AsyncSlotPool
from multimodal_embeddings_ai_inference.async_jobs import (
    AsyncEmbeddingJobs,
    execute_job,
    fingerprint,
    job_key,
)
from multimodal_embeddings_ai_inference.service import InferenceService
from multimodal_embeddings_ai_inference.validation import Prepared


def test_repeated_same_id_429_never_strands_global_capacity():
    from fastapi import HTTPException

    from multimodal_embeddings_ai_inference.api import _InferenceQuota

    async def scenario():
        store, queued = DurableMemory(), []

        async def spawn(key, data):
            queued.append((key, data))

        jobs = AsyncEmbeddingJobs(store, spawn, AsyncSlotPool(
            store, global_limit=2, per_client_limit=1,
        ))
        quota = _InferenceQuota(120, 4)
        def sample(rid):
            return Prepared("wemm-embedding-9b", "hello", "text", 64, "a" * 64, rid)
        await jobs.submit(sample("accepted-before-429"), quota, "alice")
        for _ in range(4):
            with pytest.raises(HTTPException) as error:
                await jobs.submit(sample("same-rejected-id-001"), quota, "alice")
            assert error.value.status_code == 429
            assert sum(k.startswith("permit:global:") for k in active_permit_claims(store)) == 1
        await execute_job(*queued[0], store, InferenceService("mock"))
        assert not active_permit_claims(store)
        await jobs.submit(sample("same-rejected-id-001"), quota, "alice")
        await execute_job(*queued[1], store, InferenceService("mock"))
        assert not active_permit_claims(store)

    asyncio.run(scenario())


def test_same_job_reacquisition_has_independent_once_only_release():
    async def scenario():
        store = DurableMemory()
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        first = await pool.reserve("a" * 64, "alice")
        await pool.release(first, "a" * 64)
        second = await pool.reserve("a" * 64, "alice")
        await pool.release(first, "a" * 64)
        assert await store.get(second["global"]) is not None
        await pool.release(second, "a" * 64)
        assert not active_permit_claims(store)

    asyncio.run(scenario())


def test_legacy_permit_records_fail_closed_without_atomic_compare_delete():
    async def scenario():
        store = DurableMemory()
        slots = {"global": "permit:global:0", "client": "permit:client:legacy:0"}
        for key in slots.values():
            store.data[key] = {"job_id": "a" * 64, "claimed_at": 1}
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        await pool.release(slots, "a" * 64)
        assert all(key in store.data for key in slots.values())
        with pytest.raises(Exception) as error:
            await pool.reserve("b" * 64, "bob")
        assert getattr(error.value, "status_code", None) == 429

    asyncio.run(scenario())


def test_duplicate_detached_coordinators_never_encode_twice():
    async def scenario():
        store = DurableMemory()
        rid = "coordinator-double-start-001"
        p = Prepared(
            "wemm-embedding-9b", "hello", "text", 64, "a" * 64, rid,
        )
        key = job_key(rid)
        now = int(time.time())
        store.data[key] = {
            "state": "QUEUED", "created_at": now, "updated_at": now,
            "fingerprint": fingerprint(p),
        }
        calls = 0

        class SlowService:
            backend = "mock"

            async def encode_many_profiled(self, items):
                nonlocal calls
                calls += 1
                await asyncio.sleep(0.02)
                return await InferenceService("mock").encode_many_profiled(items)

        await asyncio.gather(*(
            execute_job(key, {
                "model": p.model, "payload": p.payload, "kind": p.kind,
                "dimension": p.dimension, "input_sha256": p.input_sha256,
                "request_id": p.request_id,
            }, store, SlowService()) for _ in range(10)
        ))
        assert calls == 1
        assert store.data[key]["state"] == "SUCCEEDED"
        assert len([k for k in store.data if k.startswith("job:start:v1:")]) == 1

    asyncio.run(scenario())


def test_start_claim_lost_ack_never_starts_or_retries_gpu():
    async def scenario():
        rid = "start-claim-lost-ack-001"
        key = job_key(rid)
        p = Prepared("wemm-embedding-9b", "hello", "text", 64, "a" * 64, rid)

        class UncertainStartStore(DurableMemory):
            async def put(self, key, value, skip_if_exists=False):
                result = await super().put(key, value, skip_if_exists)
                if key.startswith("job:start:v1:") and result:
                    raise OSError("ACK lost after durable start claim")
                return result

        store = UncertainStartStore()
        store.data[key] = {
            "state": "QUEUED", "created_at": 1, "updated_at": 1,
            "fingerprint": fingerprint(p),
        }
        called = []

        class Service:
            async def encode_many_profiled(self, items):
                called.append(1)
                raise AssertionError("Should not dispatch")

        prepared = dict(
            model=p.model, payload=p.payload, kind=p.kind, dimension=p.dimension,
            input_sha256=p.input_sha256, request_id=p.request_id,
        )
        with pytest.raises(OSError, match="ACK lost"):
            await execute_job(key, prepared, store, Service())
        # A second coordinator must fail the durable start claim.
        await execute_job(key, prepared, store, Service())
        assert called == []
        assert store.data[key]["state"] == "QUEUED"

    asyncio.run(scenario())


def test_parallel_release_marker_does_not_delete_reallocated_permit():
    async def scenario():
        class PausedRelease(DurableMemory):
            def __init__(self):
                super().__init__()
                self.hold = asyncio.Event()
                self.resume = asyncio.Event()
                self.wait_once = True
                self.target = None

            async def get(self, key):
                value = await super().get(key)
                if key == self.target and self.wait_once:
                    self.wait_once = False
                    self.hold.set()
                    await self.resume.wait()
                return value

        store = PausedRelease()
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        slots = await pool.reserve("a" * 64, "alice")
        store.target = slots["global"]
        release_one = asyncio.create_task(pool.release(slots, "a" * 64))
        await asyncio.wait_for(store.hold.wait(), 2)
        # Second call is safely rejected by its durable once-only marker.
        await asyncio.wait_for(pool.release(slots, "a" * 64), 2)
        store.resume.set()
        await asyncio.wait_for(release_one, 2)
        new_slots = await pool.reserve("b" * 64, "bob")
        # Old release must not be allowed to pop the new owner's slot.
        await pool.release(slots, "a" * 64)
        assert (await store.get(new_slots["global"]))["job_id"] == "b" * 64

    asyncio.run(scenario())


def test_expired_claim_and_late_release_cannot_delete_reassigned_slot():
    """An expired claim can strand capacity but must never double-admit."""
    async def scenario():
        class ExpireDuringRelease(DurableMemory):
            def __init__(self):
                super().__init__()
                self.hold = asyncio.Event()
                self.resume = asyncio.Event()
                self.pause = True
                self.target = ""

            async def get(self, key):
                value = await super().get(key)
                if key == self.target and self.pause:
                    self.pause = False
                    self.hold.set()
                    await self.resume.wait()
                return value

        store = ExpireDuringRelease()
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        first = await pool.reserve("a" * 64, "alice")
        store.target = first["global"]
        old_release = asyncio.create_task(pool.release(first, "a" * 64))
        await asyncio.wait_for(store.hold.wait(), 2)
        store.data.pop(first["global"])  # Simulate an external expiry.
        newer = await pool.reserve("b" * 64, "bob")
        store.resume.set()
        await asyncio.wait_for(old_release, 2)
        assert (await store.get(newer["global"]))["job_id"] == "b" * 64
        with pytest.raises(Exception) as error:
            await pool.reserve("c" * 64, "charlie")
        assert getattr(error.value, "status_code", None) == 429

    asyncio.run(scenario())


def test_terminal_store_failure_retains_capacity():
    async def scenario():
        class FailedTerminal(DurableMemory):
            async def put(self, key, value, skip_if_exists=False):
                if isinstance(value, dict) and value.get("state") in ("FAILED", "SUCCEEDED"):
                    raise OSError("terminal persistence unavailable")
                return await super().put(key, value, skip_if_exists)

        store = FailedTerminal()
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        p = Prepared("wemm-embedding-9b", "hello", "text", 64, "a" * 64, "terminal-failure-001")
        job_id = job_key(p.request_id)
        slots = await pool.reserve(job_id, "owner")
        now = int(time.time())
        store.data[job_id] = {
            "state": "QUEUED", "created_at": now, "updated_at": now,
            "fingerprint": fingerprint(p), "slots": slots,
        }
        with pytest.raises(OSError, match="terminal"):
            await execute_job(job_id, vars(p).copy(), store, InferenceService("mock"))
        assert store.data[job_id]["state"] == "RUNNING"
        assert await store.get(slots["global"]) is not None
        assert await store.get(slots["client"]) is not None

    asyncio.run(scenario())


def test_one_hundred_parallel_clients_bounded_after_restarts():
    async def scenario():
        store, dispatched = DurableMemory(), []
        options = settings(global_slots=8, client_slots=4)
        app = app_with(store, dispatched, options)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://local") as client:
            replies = await asyncio.gather(*(
                client.post("/v1/embeddings/async",
                            headers=auth(ALICE if i % 2 else BOB),
                            json=payload(f"parallel-load-{i:04d}"))
                for i in range(100)
            ))
        assert all(r.status_code in (202, 429) for r in replies)
        assert sum(r.status_code == 202 for r in replies) <= 8
        assert len(dispatched) == sum(r.status_code == 202 for r in replies)
        assert sum(k.startswith("permit:global:") for k in active_permit_claims(store)) <= 8
        assert sum(k.startswith("permit:client:") for k in active_permit_claims(store)) <= 8
        restarted = app_with(store, dispatched, options)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(restarted),
                                     base_url="http://local") as client:
            for i in range(10):
                r = await client.post("/v1/embeddings/async", headers=auth(ALICE),
                                      json=payload(f"restart-overload-{i:04d}"))
                assert r.status_code == 429
        assert len(dispatched) <= 8

    asyncio.run(scenario())
