"""CPU-only multi-client security and durable GPU admission regressions."""
import asyncio
import hashlib

import httpx
import pytest

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.async_admission import AsyncSlotPool
from multimodal_embeddings_ai_inference.async_jobs import AsyncEmbeddingJobs, execute_job
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.service import InferenceService

OWNER = "owner-gateway-token-123456789"
ALICE = "alice-client-key-123456789"
ALICE_NEW = "alice-rotated-key-123456789"
BOB = "bob-client-key-123456789"
MODEL = "wemm-embedding-9b"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def auth(value):
    return {"Authorization": "Bearer " + value}


def payload(request_id):
    return {"model": MODEL, "text": "face", "dimension": 64, "request_id": request_id}


class DurableMemory:
    def __init__(self):
        self.data = {}
        self.lock = asyncio.Lock()

    async def get(self, key):
        return self.data.get(key)

    async def put(self, key, value, skip_if_exists=False):
        async with self.lock:
            if skip_if_exists and key in self.data:
                return False
            self.data[key] = value
            return True

    async def pop(self, key, default=None):
        async with self.lock:
            return self.data.pop(key, default)


def active_permit_claims(store):
    """Count logical active claims; released v2 records remain immutable."""
    claims = []
    for key, owner in store.data.items():
        if not key.startswith(("permit:global:", "permit:client:")):
            continue
        if ":v2:" not in key:
            claims.append(key)  # Existing legacy permits remain occupied.
            continue
        if not key.rsplit(":v2:", 1)[-1].isdigit():
            continue
        receipt = store.data.get(AsyncSlotPool._receipt_key(key))
        if not AsyncSlotPool._matches(owner, receipt):
            claims.append(key)
    return claims


def settings(*, alice=True, bob=True, global_slots=3, client_slots=2, client_rate=60):
    keys = {}
    if alice:
        keys["alice"] = (digest(alice),) if isinstance(alice, str) else (digest(ALICE),)
    if bob:
        keys["bob"] = (digest(BOB),)
    return Settings(
        api_token=OWNER, backend="mock", client_key_hashes=keys,
        max_async_jobs_global=global_slots,
        max_async_jobs_per_client=client_slots,
        max_embeddings_per_client_per_minute=client_rate,
    )


def app_with(store, spawned, opts):
    async def spawn(key, data):
        spawned.append((key, data))
    jobs = AsyncEmbeddingJobs(
        store, spawn, AsyncSlotPool(
            store, global_limit=opts.max_async_jobs_global,
            per_client_limit=opts.max_async_jobs_per_client,
        ),
    )
    return create_app(opts, InferenceService("mock"), async_jobs=jobs)


def test_tenant_isolation_rotation_revocation_and_idempotency():
    async def scenario():
        store, spawned = DurableMemory(), []
        app = app_with(store, spawned, settings())
        rid = "same-request-id-0001"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://test") as client:
            a = await client.post("/v1/embeddings/async", headers=auth(ALICE), json=payload(rid))
            b = await client.post("/v1/embeddings/async", headers=auth(BOB), json=payload(rid))
            assert a.status_code == b.status_code == 202
            assert a.json()["job_id"] != b.json()["job_id"]
            ak, bk = a.json()["job_id"], b.json()["job_id"]
            assert (await client.get("/v1/embeddings/jobs/" + ak,
                                     headers=auth(BOB))).status_code == 404
            assert (await client.get("/v1/embeddings/jobs/" + bk,
                                     headers=auth(ALICE))).status_code == 404
            assert (await client.get("/v1/embeddings/jobs/" + ak,
                                     headers=auth(OWNER))).status_code == 404
        # Rotate only Alice; existing job ID must not change.
        rotated = app_with(store, spawned, settings(alice=ALICE_NEW))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(rotated),
                                     base_url="http://test") as client:
            assert (await client.get("/v1/embeddings/jobs/" + ak,
                                     headers=auth(ALICE_NEW))).status_code == 202
            duplicate = await client.post("/v1/embeddings/async",
                                          headers=auth(ALICE_NEW), json=payload(rid))
            assert duplicate.status_code == 202
            assert duplicate.json()["job_id"] == ak
            assert (await client.get("/v1/embeddings/jobs/" + ak,
                                     headers=auth(ALICE))).status_code == 401
        assert len(spawned) == 2

    asyncio.run(scenario())


def test_global_and_per_tenant_slots_survive_gateway_restart_and_flood():
    async def scenario():
        store, spawned = DurableMemory(), []
        opts = settings(global_slots=3, client_slots=2)
        app = app_with(store, spawned, opts)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://test") as client:
            replies = await asyncio.gather(*(
                client.post("/v1/embeddings/async",
                            headers=auth(ALICE if i < 6 else BOB),
                            json=payload(f"flood-request-{i:04d}"))
                for i in range(12)
            ))
            assert sum(r.status_code == 202 for r in replies) == 3
            assert sum(r.status_code == 429 for r in replies) == 9
            assert all("Retry-After" in r.headers for r in replies if r.status_code == 429)
            assert len(spawned) == 3
            # Duplicate is a read-only operation even when global capacity is full.
            chosen = next(i for i in range(12) if replies[i].status_code == 202)
            same = await client.post("/v1/embeddings/async",
                                     headers=auth(ALICE if chosen < 6 else BOB),
                                     json=payload(f"flood-request-{chosen:04d}"))
            assert same.status_code == 202
        restarted = app_with(store, spawned, opts)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(restarted),
                                     base_url="http://test") as client:
            blocked = await client.post("/v1/embeddings/async",
                                        headers=auth(BOB),
                                        json=payload("after-restart-0001"))
            assert blocked.status_code == 429
        assert sum(key.startswith("permit:global:")
                   for key in active_permit_claims(store)) == 3

    asyncio.run(scenario())


def test_definitive_completion_releases_slots_but_uncertain_ack_does_not():
    async def scenario():
        store, spawned = DurableMemory(), []

        async def uncertain_spawn(key, prepared):
            spawned.append((key, prepared))
            raise OSError("ack may have been lost")

        jobs = AsyncEmbeddingJobs(
            store, uncertain_spawn,
            AsyncSlotPool(store, global_limit=1, per_client_limit=1),
        )
        app = create_app(settings(global_slots=1, client_slots=1),
                         InferenceService("mock"), async_jobs=jobs)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://test") as client:
            first = await client.post("/v1/embeddings/async", headers=auth(ALICE),
                                      json=payload("uncertain-ack-001"))
            assert first.status_code == 202
            blocked = await client.post("/v1/embeddings/async", headers=auth(BOB),
                                        json=payload("uncertain-ack-002"))
            assert blocked.status_code == 429
            assert len(spawned) == 1
            # Simulate an explicitly confirmed single coordinator, never
            # automatically re-submit after an uncertain acknowledgement.
            await execute_job(*spawned[0], store, InferenceService("mock"))
            assert store.data[first.json()["job_id"]]["state"] == "SUCCEEDED"
            next_call = await client.post("/v1/embeddings/async", headers=auth(BOB),
                                          json=payload("uncertain-ack-002"))
            assert next_call.status_code == 202

    asyncio.run(scenario())


def test_per_client_embedding_rate_limit_and_legacy_owner():
    from fastapi.testclient import TestClient
    app = app_with(DurableMemory(), [], settings(client_rate=1))
    with TestClient(app) as client:
        body = {"model": MODEL, "text": "demo", "dimension": 64}
        assert client.post("/v1/embeddings", headers=auth(ALICE),
                           json=body).status_code == 200
        throttled = client.post("/v1/embeddings", headers=auth(ALICE), json=body)
        assert throttled.status_code == 429
        assert "Retry-After" in throttled.headers
        assert client.post("/v1/embeddings", headers=auth(BOB),
                           json=body).status_code == 200
        assert client.post("/v1/embeddings", headers=auth(OWNER),
                           json=body).status_code == 200


def test_reject_bad_client_registry_and_shared_tokens():
    with pytest.raises(ValueError, match="unique"):
        Settings(api_token=OWNER, client_key_hashes={
            "alice": (digest(ALICE),), "bob": (digest(ALICE),),
        })
    with pytest.raises(ValueError, match="reserved"):
        Settings(api_token=OWNER, client_key_hashes={"owner": (digest(ALICE),)})


def test_store_unavailable_rejects_without_gpu_dispatch():
    async def scenario():
        class OfflineStore(DurableMemory):
            async def get(self, key):
                raise OSError("offline")

        spawned = []
        app = app_with(OfflineStore(), spawned, settings())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://test") as client:
            result = await client.post("/v1/embeddings/async",
                                       headers=auth(ALICE),
                                       json=payload("registry-offline-001"))
            assert result.status_code == 503
        assert not spawned

    asyncio.run(scenario())


def test_client_hash_registry_env_parsing(monkeypatch):
    import json

    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", OWNER)
    monkeypatch.setenv("VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON", json.dumps({
        "client_a": [digest(ALICE), digest(ALICE_NEW)]
    }))
    monkeypatch.setenv("VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL", "7")
    monkeypatch.setenv("VISION_EMBEDDINGS_MAX_ASYNC_JOBS_PER_CLIENT", "2")
    configured = Settings.from_env()
    assert len(configured.client_key_hashes["client_a"]) == 2
    assert configured.max_async_jobs_global == 7
    monkeypatch.setenv("VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON", "{")
    with pytest.raises(ValueError, match="CLIENT_KEY_HASHES"):
        Settings.from_env()


def test_delayed_cursor_writer_cannot_permanently_exhaust_released_slot():
    """A stale release cursor must not trap scans in the first 1024 generations."""
    async def scenario():
        class DelayedCursor(DurableMemory):
            def __init__(self):
                super().__init__()
                self.entered, self.resume = asyncio.Event(), asyncio.Event()
                self.pause_once = True
                self.get_calls = 0

            async def get(self, key):
                self.get_calls += 1
                return await super().get(key)

            async def put(self, key, value, skip_if_exists=False):
                if (self.pause_once and key == "permit:global:0:v2:cursor"
                        and value == 1):
                    self.pause_once = False
                    self.entered.set()
                    await self.resume.wait()
                return await super().put(key, value, skip_if_exists)

        store = DelayedCursor()
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        old = await pool.reserve("first", "alice")
        held = asyncio.create_task(pool.release(old, "first"))
        await store.entered.wait()
        for generation in range(AsyncSlotPool._MAX_SCAN + 1):
            job_id = f"later-{generation}"
            slots = await pool.reserve(job_id, "alice")
            await pool.release(slots, job_id)
        store.resume.set()
        await held
        assert store.data["permit:global:0:v2:cursor"] == 1
        before_reads = store.get_calls
        recovered = await pool.reserve("after-stale-write", "alice")
        assert store.get_calls - before_reads < 64  # O(log N) checkpoint recovery.
        assert recovered["global"].endswith(
            f":v2:{AsyncSlotPool._MAX_SCAN + 2}"
        )
        await pool.release(old, "first")
        assert recovered["global"] in active_permit_claims(store)
        await pool.release(recovered, "after-stale-write")
        assert not active_permit_claims(store)

    asyncio.run(scenario())


def test_v2_admission_blocks_active_legacy_permits_during_drain():
    """One-way fail-closed protection is not support for mixed-version rollout."""
    async def scenario():
        store = DurableMemory()
        store.data["permit:global:0"] = {"job_id": "legacy-in-progress"}
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as rejected:
            await pool.reserve("new-work", "alice")
        assert rejected.value.status_code == 429
        assert not any(":v2:" in key for key in store.data)

    asyncio.run(scenario())


def test_mixed_version_admission_documentation_requires_full_drain():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("ASYNC_RELIABILITY.md", "ASYNC_RELIABILITY.vi.md"):
        text = (root / "docs" / name).read_text()
        assert "v1/v2" in text
        assert "drain" in text.lower()
        assert "Redis/Valkey" in text


def _seed_released_generations(store, *, count, base="permit:global:0",
                               missing_checkpoints=()):
    """Synthetic intact release chain, not a claim of live GPU execution."""
    for generation in range(count):
        key = AsyncSlotPool._claim_key(base, generation)
        owner = {"job_id": f"done-{generation}", "acquisition_id": f"a-{generation}"}
        store.data[key] = owner
        store.data[AsyncSlotPool._receipt_key(key)] = dict(owner)
    for boundary in range(AsyncSlotPool._MAX_SCAN, count + 1, AsyncSlotPool._MAX_SCAN):
        if boundary not in missing_checkpoints:
            store.data[f"{base}:v2:checkpoint:{boundary}"] = {
                "slot": AsyncSlotPool._claim_key(base, boundary - 1)
            }
    store.data[base + ":v2:cursor"] = 1


@pytest.mark.parametrize("released_count", [4095, 16383, 65535])
def test_cursor_recovery_has_logarithmic_store_read_bound(released_count):
    """A stale cursor must not require reading the intervening released claims."""
    async def scenario():
        class Reads(DurableMemory):
            def __init__(self):
                super().__init__()
                self.reads = 0

            async def get(self, key):
                self.reads += 1
                return await super().get(key)

        store = Reads()
        _seed_released_generations(store, count=released_count)
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        slots = await pool.reserve("next-owner", "fresh-client")
        assert slots["global"] == AsyncSlotPool._claim_key(
            "permit:global:0", released_count
        )
        assert store.reads < 110
        await pool.release(slots, "next-owner")
        assert slots["global"] not in active_permit_claims(store)

    asyncio.run(scenario())


def test_cursor_recovery_with_missing_checkpoint_does_not_scan_history():
    """The proof is the receipt chain; sparse old checkpoints can be delayed."""
    async def scenario():
        class Reads(DurableMemory):
            def __init__(self):
                super().__init__()
                self.reads = 0

            async def get(self, key):
                self.reads += 1
                return await super().get(key)

        store = Reads()
        _seed_released_generations(
            store, count=16383, missing_checkpoints=(2048, 8192, 14336),
        )
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        claimed = await pool.reserve("after-gap", "new-client")
        assert claimed["global"].endswith(":v2:16383")
        assert store.reads < 110
        await pool.release(claimed, "after-gap")

    asyncio.run(scenario())


def test_cursor_recovery_tries_second_slot_after_long_active_history():
    async def scenario():
        store = DurableMemory()
        _seed_released_generations(store, count=16383)
        active_key = AsyncSlotPool._claim_key("permit:global:0", 16383)
        store.data[active_key] = {"job_id": "busy", "acquisition_id": "ongoing"}
        pool = AsyncSlotPool(store, global_limit=2, per_client_limit=2)
        claimed = await pool.reserve("free-slot", "fresh-client")
        assert claimed["global"] == AsyncSlotPool._claim_key("permit:global:1", 0)
        assert active_key in active_permit_claims(store)
        await pool.release(claimed, "free-slot")
        assert active_key in active_permit_claims(store)

    asyncio.run(scenario())


def test_cursor_recovery_missing_predecessor_receipt_fails_closed():
    async def scenario():
        store = DurableMemory()
        _seed_released_generations(store, count=4095)
        key = AsyncSlotPool._claim_key("permit:global:0", 4094)
        store.data.pop(AsyncSlotPool._receipt_key(key))
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await pool.reserve("must-not-claim", "new-client")
        assert exc.value.status_code == 429
        assert AsyncSlotPool._claim_key("permit:global:0", 4095) not in store.data

    asyncio.run(scenario())


def test_cursor_recovery_concurrent_reservations_keep_single_owner():
    async def scenario():
        store = DurableMemory()
        _seed_released_generations(store, count=4095)
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)

        async def request(number):
            try:
                return await pool.reserve(f"contender-{number}", "alice")
            except Exception as exc:
                return exc

        results = await asyncio.gather(*(request(i) for i in range(12)))
        wins = [result for result in results if isinstance(result, dict)]
        from fastapi import HTTPException
        losses = [result for result in results if isinstance(result, HTTPException)]
        assert len(wins) == 1
        assert len(losses) == 11
        assert all(result.status_code == 429 for result in losses)
        assert wins[0]["global"].endswith(":v2:4095")
        assert wins[0]["global"] in active_permit_claims(store)
        await pool.release(wins[0], "contender-" + str(results.index(wins[0])))
        assert not active_permit_claims(store)

    asyncio.run(scenario())


def test_cursor_recovery_emits_sanitized_admission_diagnostics(caplog):
    import logging

    async def scenario():
        store = DurableMemory()
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        with caplog.at_level(
            logging.DEBUG, logger="multimodal_embeddings_ai_inference.async_admission"
        ):
            claimed = await pool.reserve("private-id", "private-client")
        messages = [rec.message for rec in caplog.records
                    if "Async admission reserve store_reads=" in rec.message]
        assert messages
        assert "elapsed_ms=" in messages[-1]
        assert "private-id" not in messages[-1]
        assert "private-client" not in messages[-1]
        await pool.release(claimed, "private-id")

    asyncio.run(scenario())


def test_delayed_checkpoint_writer_and_stale_cursor_recover_in_bounded_reads():
    """Real reserve/release history; an old checkpoint ACK can arrive very late."""
    async def scenario():
        class DelayedCheckpoint(DurableMemory):
            def __init__(self):
                super().__init__()
                self.paused = asyncio.Event()
                self.resume = asyncio.Event()
                self.pause_once = True
                self.reads = 0

            async def get(self, key):
                self.reads += 1
                return await super().get(key)

            async def put(self, key, value, skip_if_exists=False):
                if (self.pause_once and key == "permit:global:0:v2:checkpoint:1024"):
                    self.pause_once = False
                    self.paused.set()
                    await self.resume.wait()
                return await super().put(key, value, skip_if_exists)

        store = DelayedCheckpoint()
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        for generation in range(1023):
            slots = await pool.reserve(f"early-{generation}", "alice")
            await pool.release(slots, f"early-{generation}")

        held = await pool.reserve("checkpoint-writer", "alice")
        delayed = asyncio.create_task(pool.release(held, "checkpoint-writer"))
        await store.paused.wait()

        for generation in range(3071):
            job = f"later-{generation}"
            slots = await pool.reserve(job, "alice")
            await pool.release(slots, job)

        store.resume.set()
        await delayed
        assert store.data["permit:global:0:v2:cursor"] == 1024
        before = store.reads
        recovered = await pool.reserve("late-checkpoint-recovery", "alice")
        assert recovered["global"].endswith(":v2:4095")
        assert store.reads - before < 110
        await pool.release(held, "checkpoint-writer")
        assert recovered["global"] in active_permit_claims(store)
        await pool.release(recovered, "late-checkpoint-recovery")
        assert not active_permit_claims(store)

    asyncio.run(scenario())


def test_expired_claim_with_newer_checkpoint_never_gets_reused():
    async def scenario():
        store = DurableMemory()
        _seed_released_generations(store, count=5120)
        missing = AsyncSlotPool._claim_key("permit:global:0", 4100)
        store.data.pop(missing)
        store.data.pop(AsyncSlotPool._receipt_key(missing))
        pool = AsyncSlotPool(store, global_limit=1, per_client_limit=1)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as error:
            await pool.reserve("new-work", "new-client")
        assert error.value.status_code == 429
        assert missing not in store.data
    asyncio.run(scenario())
