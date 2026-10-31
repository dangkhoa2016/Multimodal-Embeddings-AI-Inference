"""CPU-only Lua-backed shared quota integration: two gateways, one Redis store."""
import asyncio
import hashlib

import fakeredis.aioredis
import httpx
import pytest
from test_multitenant_admission import DurableMemory

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.async_jobs import AsyncEmbeddingJobs
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.distributed_quota import RedisDistributedQuota
from multimodal_embeddings_ai_inference.service import InferenceService

OWNER = "offline-distributed-owner-token-1234"
A = "offline-distributed-client-a-token"
B = "offline-distributed-client-b-token"
BODY = {"model": "wemm-embedding-9b", "text": "example", "dimension": 64}


def authorization(token):
    return {"Authorization": "Bearer " + token}


def options(**overrides):
    values = dict(
        api_token=OWNER, backend="mock",
        client_key_hashes={
            "alice": (hashlib.sha256(A.encode()).hexdigest(),),
            "bob": (hashlib.sha256(B.encode()).hexdigest(),),
        },
        distributed_rate_enabled=True,
        distributed_rate_redis_url="redis://localhost/0",
        max_embeddings_per_minute=10,
        max_embeddings_per_client_per_minute=6,
        max_inflight_requests=20,
        max_body_requests=16,
    )
    values.update(overrides)
    return Settings(**values)


def make_quota(server, opts):
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    quota = RedisDistributedQuota(
        client, namespace="test:distributed:quota",
        global_units_per_minute=opts.max_embeddings_per_minute,
        client_units_per_minute=opts.max_embeddings_per_client_per_minute,
        global_sync_inflight=opts.max_distributed_sync_inflight,
        client_sync_inflight=opts.max_distributed_client_sync_inflight,
        lease_seconds=60,
    )
    return client, quota


def test_ten_tokens_shared_atomically_across_four_gateways():
    async def scenario():
        server = fakeredis.FakeServer()
        # This case isolates shared *rate* fairness from the independently
        # tested distributed sync-concurrency and local body admission caps.
        opts = options(
            max_distributed_sync_inflight=64,
            max_distributed_client_sync_inflight=64,
        )
        clients = []
        for _ in range(4):
            redis, quota = make_quota(server, opts)
            app = create_app(opts, InferenceService("mock"), distributed_quota=quota)
            clients.append((redis, httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test",
            )))
        try:
            replies = await asyncio.gather(*(
                clients[i % 4][1].post(
                    "/v1/embeddings", headers=authorization(A if i % 2 else B), json=BODY,
                ) for i in range(64)
            ))
            allowed = [r for r in replies if r.status_code == 200]
            denied = [r for r in replies if r.status_code == 429]
            assert len(allowed) == 10, {r.status_code: r.text for r in replies}
            assert len(denied) == 54
            assert all("Retry-After" in r.headers for r in denied)
            # The same server holds two token bucket entries and a bounded
            # number of expiring concurrency keys, not one key per request.
            keys = await clients[0][0].keys("{test:distributed:quota}:*")
            assert len(keys) <= 6
            ttl_values = [await clients[0][0].ttl(key) for key in keys]
            assert all(ttl > 0 for ttl in ttl_values)
        finally:
            for redis, client in clients:
                await client.aclose()
                await redis.aclose()

    asyncio.run(scenario())


def test_atomic_two_budget_check_never_debits_global_for_denied_client():
    async def scenario():
        server = fakeredis.FakeServer()
        opts = options(max_embeddings_per_minute=6,
                       max_embeddings_per_client_per_minute=3)
        db, quota = make_quota(server, opts)
        app = create_app(opts, InferenceService("mock"), distributed_quota=quota)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local",
        ) as client:
            for _ in range(3):
                assert (await client.post("/v1/embeddings",
                                          headers=authorization(A), json=BODY)).status_code == 200
            assert (await client.post("/v1/embeddings",
                                      headers=authorization(A), json=BODY)).status_code == 429
            for _ in range(3):
                assert (await client.post("/v1/embeddings",
                                          headers=authorization(B), json=BODY)).status_code == 200
        await db.aclose()

    asyncio.run(scenario())


def test_batch_debits_by_embedding_units_not_http_call():
    async def scenario():
        server = fakeredis.FakeServer()
        opts = options(max_embeddings_per_minute=6,
                       max_embeddings_per_client_per_minute=4)
        redis, quota = make_quota(server, opts)
        app = create_app(opts, InferenceService("mock"), distributed_quota=quota)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local",
        ) as client:
            first = await client.post("/v1/embeddings/batches",
                                      headers=authorization(A),
                                      json={"items": [BODY] * 4})
            assert first.status_code == 200
            assert (await client.post("/v1/embeddings",
                                      headers=authorization(A), json=BODY)).status_code == 429
            assert (await client.post("/v1/embeddings/batches",
                                      headers=authorization(B),
                                      json={"items": [BODY] * 2})).status_code == 200
        await redis.aclose()

    asyncio.run(scenario())


def test_global_sync_lease_blocks_second_gateway_without_consuming_rate():
    async def scenario():
        server = fakeredis.FakeServer()
        opts = options(
            max_distributed_sync_inflight=1,
            max_distributed_client_sync_inflight=1,
            max_embeddings_per_minute=5, max_embeddings_per_client_per_minute=5,
        )
        entered = asyncio.Event()
        proceed = asyncio.Event()
        calls = []

        class SlowMock:
            backend = "mock"

            async def encode_many_profiled(self, inputs):
                calls.append(1)
                entered.set()
                await proceed.wait()
                return await InferenceService("mock").encode_many_profiled(inputs)

        db1, q1 = make_quota(server, opts)
        db2, q2 = make_quota(server, opts)
        a = create_app(opts, SlowMock(), distributed_quota=q1)
        b = create_app(opts, SlowMock(), distributed_quota=q2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(a),
                                     base_url="http://one") as first, httpx.AsyncClient(
                transport=httpx.ASGITransport(b), base_url="http://two") as second:
            task = asyncio.create_task(first.post(
                "/v1/embeddings", headers=authorization(A), json=BODY))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                denied = await second.post("/v1/embeddings",
                                           headers=authorization(B), json=BODY)
                assert denied.status_code == 429
                assert "concurrency" in denied.json()["detail"].lower()
                health = await second.get("/health", headers=authorization(B))
                assert health.status_code == 200
                assert len(calls) == 1
            finally:
                proceed.set()
                assert (await task).status_code == 200
            # Releasing after a confirmed success makes a different gateway
            # eligible without waiting for a lease TTL.
            follow_up = await second.post("/v1/embeddings",
                                          headers=authorization(B), json=BODY)
            assert follow_up.status_code == 200
        await db1.aclose()
        await db2.aclose()

    asyncio.run(scenario())


def test_redis_timeout_fails_closed_and_never_dispatches_mock_gpu():
    async def scenario():
        calls = []

        class Offline:
            async def eval(self, *args):
                raise OSError("Redis unavailable with private details")

        opts = options()
        quota = RedisDistributedQuota(
            Offline(), namespace="test:offline",
            global_units_per_minute=10, client_units_per_minute=6,
            global_sync_inflight=4, client_sync_inflight=2, lease_seconds=60,
        )

        class CountingService:
            backend = "mock"

            async def encode_many_profiled(self, items):
                calls.append(items)
                raise AssertionError("Should not dispatch after Redis outage")

        app = create_app(opts, CountingService(), distributed_quota=quota)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local",
        ) as client:
            denied = await client.post("/v1/embeddings",
                                       headers=authorization(A), json=BODY)
            assert denied.status_code == 503
            assert denied.headers["Retry-After"] == "5"
            assert "private" not in denied.text.lower()
            assert (await client.get("/health",
                                     headers=authorization(A))).status_code == 200
        assert calls == []

    asyncio.run(scenario())


def test_uncertain_sync_gpu_failure_keeps_distributed_lease():
    async def scenario():
        server = fakeredis.FakeServer()
        opts = options(
            backend="modal", inference_timeout_seconds=0.01,
            max_distributed_sync_inflight=1,
            max_distributed_client_sync_inflight=1,
        )
        redis, quota = make_quota(server, opts)

        class LongCall:
            async def encode_many_profiled(self, inputs):
                await asyncio.Event().wait()

        app = create_app(opts, LongCall(), distributed_quota=quota)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local",
        ) as client:
            timed_out = await client.post(
                "/v1/embeddings", headers=authorization(A), json=BODY,
            )
            assert timed_out.status_code == 504
            key = "{test:distributed:quota}:sync:global"
            assert await redis.zcard(key) == 1
            again = await client.post(
                "/v1/embeddings", headers=authorization(B), json=BODY,
            )
            assert again.status_code == 429
        await redis.aclose()

    asyncio.run(scenario())


def test_shared_async_rate_budget_and_duplicate_does_not_double_debit():
    async def scenario():
        server = fakeredis.FakeServer()
        opts = options(
            max_embeddings_per_minute=2, max_embeddings_per_client_per_minute=2,
        )
        store = DurableMemory()
        spawned = []
        async def spawn(key, data):
            spawned.append(key)

        db, quota = make_quota(server, opts)
        jobs = AsyncEmbeddingJobs(store, spawn)
        app = create_app(opts, InferenceService("mock"), async_jobs=jobs,
                         distributed_quota=quota)
        body = dict(BODY, request_id="redis-async-idempotent-0001")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local",
        ) as client:
            first = await client.post("/v1/embeddings/async",
                                      headers=authorization(A), json=body)
            assert first.status_code == 202
            for _ in range(5):
                duplicate = await client.post("/v1/embeddings/async",
                                              headers=authorization(A), json=body)
                assert duplicate.status_code == 202
            assert len(spawned) == 1
            another = await client.post("/v1/embeddings/async",
                                        headers=authorization(A),
                                        json=dict(BODY, request_id="redis-async-second-0001"))
            assert another.status_code == 202
            denied = await client.post("/v1/embeddings/async",
                                       headers=authorization(A),
                                       json=dict(BODY, request_id="redis-async-third-0001"))
            assert denied.status_code == 429
            assert len(spawned) == 2
        await db.aclose()

    asyncio.run(scenario())


def test_multi_gateway_without_distributed_backend_is_refused():
    with pytest.raises(ValueError, match="Multi-gateway"):
        options(distributed_rate_enabled=False, gateway_max_containers=2)
    with pytest.raises(ValueError, match="not configured"):
        create_app(options(), InferenceService("mock"))
    with pytest.raises(ValueError, match="TLS"):
        options(distributed_rate_redis_url="redis://public.redis.test:6379/0")
    assert options(gateway_max_containers=4).gateway_max_containers == 4


def test_request_body_semaphore_bounds_predecode_work():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class Controlled:
            backend = "mock"
            async def encode_many_profiled(self, items):
                entered.set()
                await release.wait()
                return await InferenceService("mock").encode_many_profiled(items)

        opts = Settings(api_token=OWNER, max_body_requests=1,
                        max_inflight_requests=8)
        app = create_app(opts, Controlled())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local",
        ) as client:
            held = asyncio.create_task(client.post(
                "/v1/embeddings", json=BODY, headers=authorization(OWNER)))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                rejected = await client.post(
                    "/v1/embeddings", json=BODY, headers=authorization(OWNER),
                )
                assert rejected.status_code == 429
                assert rejected.headers["Retry-After"] == "1"
                assert (await client.get("/health",
                                         headers=authorization(OWNER))).status_code == 200
            finally:
                release.set()
                assert (await held).status_code == 200

    asyncio.run(scenario())


def test_lost_redis_reply_after_debit_is_fail_closed():
    async def scenario():
        server = fakeredis.FakeServer()
        redis, actual = make_quota(server, options())

        class AckLost:
            async def eval(self, *args):
                await redis.eval(*args)
                raise OSError("Artificial lost acknowledgement")

        quota = RedisDistributedQuota(
            AckLost(), namespace="test:distributed:quota",
            global_units_per_minute=10, client_units_per_minute=6,
            global_sync_inflight=4, client_sync_inflight=2, lease_seconds=60,
        )
        calls = []

        class Counting:
            backend = "mock"

            async def encode_many_profiled(self, items):
                calls.append(items)
                raise AssertionError("No inference after lost rate ACK")

        app = create_app(options(), Counting(), distributed_quota=quota)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://local") as client:
            response = await client.post("/v1/embeddings",
                                         headers=authorization(A), json=BODY)
            assert response.status_code == 503
            assert not calls
        global_key = actual._keys("alice")[0]
        assert float(await redis.hget(global_key, "tokens")) < 10
        await redis.aclose()

    asyncio.run(scenario())


def test_distributed_env_bootstrap_and_tls_validation(monkeypatch):
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", OWNER)
    monkeypatch.setenv("VISION_EMBEDDINGS_BACKEND", "modal")
    monkeypatch.setenv("VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS", "3")
    monkeypatch.setenv("VISION_EMBEDDINGS_RATE_REDIS_URL", "rediss://example.invalid:6380/0")
    config = Settings.from_env()
    assert config.distributed_rate_enabled is True
    assert config.gateway_max_containers == 3
    assert "rediss://" not in repr(config)
    monkeypatch.setenv("VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT", "0")
    with pytest.raises(ValueError, match="Multi-gateway"):
        Settings.from_env()
    monkeypatch.setenv("VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT", "1")
    monkeypatch.setenv("VISION_EMBEDDINGS_RATE_REDIS_URL", "redis://example.invalid:6379/0")
    with pytest.raises(ValueError, match="TLS"):
        Settings.from_env()


def test_redis_cluster_hash_tag_and_hashed_client_keys():
    quota = RedisDistributedQuota(
        object(), namespace="safe:namespace", global_units_per_minute=10,
        client_units_per_minute=5, global_sync_inflight=4,
        client_sync_inflight=2, lease_seconds=60,
    )
    keys = quota._keys("alice")
    assert all(key.startswith("{safe:namespace}:") for key in keys)
    assert all("alice" not in key for key in keys)
    assert len(set(keys)) == 4


def test_sample_env_rate_flag_is_an_explicit_override(monkeypatch):
    """The safe sample's 0 must not be mistaken for Settings auto mode."""
    from pathlib import Path

    sample = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    # The sample override was introduced later than the limiter itself.
    if "VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT=" in sample:
        assert "VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT=0" in sample
        assert "VISION_EMBEDDINGS_RATE_REDIS_URL=" in sample
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", OWNER)
    monkeypatch.setenv("VISION_EMBEDDINGS_BACKEND", "modal")
    monkeypatch.setenv("VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS", "1")
    monkeypatch.setenv("VISION_EMBEDDINGS_RATE_REDIS_URL", "rediss://example.invalid:6380/0")
    monkeypatch.setenv("VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT", "0")
    assert Settings.from_env().distributed_rate_enabled is False
    monkeypatch.setenv("VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS", "2")
    with pytest.raises(ValueError, match="Multi-gateway"):
        Settings.from_env()
    monkeypatch.setenv("VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT", "1")
    assert Settings.from_env().distributed_rate_enabled is True
    monkeypatch.setenv("VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT", "")
    assert Settings.from_env().distributed_rate_enabled is True
    monkeypatch.delenv("VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT")
    assert Settings.from_env().distributed_rate_enabled is True
    monkeypatch.delenv("VISION_EMBEDDINGS_RATE_REDIS_URL")
    with pytest.raises(ValueError, match="Multi-gateway"):
        Settings.from_env()


def test_bilingual_operations_explains_sample_override():
    from pathlib import Path

    docs = Path(__file__).resolve().parents[1] / "docs"
    for name in ("OPERATIONS.md", "OPERATIONS.vi.md"):
        text = (docs / name).read_text(encoding="utf-8")
        assert "`.env.example`" in text
        assert "`Settings.from_env()`" in text
        assert "DISTRIBUTED_RATE_LIMIT" in text
        assert "rediss://" in text
