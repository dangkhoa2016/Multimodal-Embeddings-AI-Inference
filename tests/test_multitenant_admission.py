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
        assert len([key for key in store.data if key.startswith("permit:global:")]) == 3

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
