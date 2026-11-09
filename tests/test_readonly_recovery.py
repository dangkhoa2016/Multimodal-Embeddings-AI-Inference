"""Read-only async lost-ACK recovery for owner and per-client API keys."""
import asyncio
import hashlib

import httpx
from test_multitenant_admission import (
    ALICE,
    ALICE_NEW,
    BOB,
    OWNER,
    DurableMemory,
    app_with,
    auth,
    payload,
    settings,
)

from multimodal_embeddings_ai_inference.async_jobs import job_key


def test_tenant_scoped_lookup_by_request_id_and_rotation():
    async def scenario():
        rid = "same-request-across-clients-001"
        store, spawned = DurableMemory(), []
        api = app_with(store, spawned, settings(global_slots=4))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(api), base_url="http://test",
        ) as client:
            before = await client.get("/v1/embeddings/jobs/by-request/" + rid,
                                      headers=auth(ALICE))
            assert before.status_code == 404
            first = await client.post(
                "/v1/embeddings/async", headers=auth(ALICE), json=payload(rid),
            )
            second = await client.post(
                "/v1/embeddings/async", headers=auth(BOB), json=payload(rid),
            )
            assert first.status_code == second.status_code == 202
            alice_id, bob_id = first.json()["job_id"], second.json()["job_id"]
            assert alice_id == job_key(rid, "alice")
            assert bob_id == job_key(rid, "bob")
            assert alice_id != bob_id
            assert (await client.get("/v1/embeddings/jobs/by-request/" + rid,
                                     headers=auth(ALICE))).json()["job_id"] == alice_id
            assert (await client.get("/v1/embeddings/jobs/by-request/" + rid,
                                     headers=auth(BOB))).json()["job_id"] == bob_id
            owner = await client.get("/v1/embeddings/jobs/by-request/" + rid,
                                     headers=auth(OWNER))
            assert owner.status_code == 404
            invalid = await client.get(
                "/v1/embeddings/jobs/by-request/bad", headers=auth(ALICE),
            )
            assert invalid.status_code == 404
            assert len(spawned) == 2

        # A new API key for the same client must recover its original job.
        rotated = app_with(store, spawned, settings(alice=ALICE_NEW, global_slots=4))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(rotated), base_url="http://test",
        ) as client:
            result = await client.get(
                "/v1/embeddings/jobs/by-request/" + rid, headers=auth(ALICE_NEW),
            )
            assert result.status_code == 202
            assert result.json()["job_id"] == alice_id
            assert (await client.get("/v1/embeddings/jobs/by-request/" + rid,
                                     headers=auth(ALICE))).status_code == 401
        assert len(spawned) == 2
    asyncio.run(scenario())


def test_legacy_owner_preexisting_job_is_recovered_without_dispatch():
    async def scenario():
        store, spawned = DurableMemory(), []
        rid = "legacy-owner-cache-0001"
        key = job_key(rid)
        store.data[key] = {
            "fingerprint": hashlib.sha256(b"old-known-job").hexdigest(),
            "state": "SUCCEEDED", "created_at": 10, "updated_at": 11,
            "result": {"model": "wemm-embedding-9b", "vector": [1.0]},
        }
        api = app_with(store, spawned, settings())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(api), base_url="http://test",
        ) as client:
            recovered = await client.get(
                "/v1/embeddings/jobs/by-request/" + rid, headers=auth(OWNER),
            )
            assert recovered.status_code == 200
            assert recovered.json()["job_id"] == key
            assert recovered.json()["state"] == "SUCCEEDED"
            forbidden = await client.get(
                "/v1/embeddings/jobs/by-request/" + rid, headers=auth(ALICE),
            )
            assert forbidden.status_code == 404
        assert not spawned
    asyncio.run(scenario())
