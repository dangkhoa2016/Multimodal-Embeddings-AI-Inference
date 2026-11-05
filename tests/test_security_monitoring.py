"""CPU-only privacy and authorization tests for operational telemetry."""
import asyncio
import base64
import hashlib
import logging

import httpx
from PIL import Image
from test_multitenant_admission import DurableMemory

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.async_jobs import AsyncEmbeddingJobs
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.security_monitor import GatewaySecurityMonitor
from multimodal_embeddings_ai_inference.service import InferenceService

OWNER = "owner-sensitive-token-do-not-expose-123"
CLIENT = "client-sensitive-token-do-not-expose-123"
SENTINEL = "PRIVATE-IMAGE-AND-TEXT-DO-NOT-ECHO"
AUTH = {"Authorization": "Bearer " + OWNER}
OTHER = {"Authorization": "Bearer " + CLIENT}
BODY = {"model": "wemm-embedding-9b", "text": "safe", "dimension": 64}


def make_settings(**changes):
    defaults = dict(
        api_token=OWNER,
        client_key_hashes={"client": (hashlib.sha256(CLIENT.encode()).hexdigest(),)},
        backend="mock", max_embeddings_per_minute=1, auth_failure_limit=20,
    )
    defaults.update(changes)
    return Settings(**defaults)


def test_owner_only_metrics_private_and_count_statuses():
    async def scenario():
        app = create_app(make_settings(), InferenceService("mock"))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://local") as client:
            assert (await client.get("/v1/ops/metrics")).status_code == 401
            forbidden = await client.get("/v1/ops/metrics", headers=OTHER)
            assert forbidden.status_code == 404
            assert (await client.get("/health", headers=AUTH)).status_code == 200
            assert (await client.post("/v1/embeddings",
                                      headers=AUTH, json=BODY)).status_code == 200
            assert (await client.post("/v1/embeddings",
                                      headers=AUTH, json=BODY)).status_code == 429
            invalid = await client.post("/v1/embeddings", headers=AUTH,
                                        json={"model": "wemm-embedding-9b",
                                              "text": SENTINEL, "dimension": "no"})
            assert invalid.status_code == 422
            assert SENTINEL not in invalid.text
            stats = await client.get("/v1/ops/metrics", headers=AUTH)
            assert stats.status_code == 200
            assert stats.headers["Cache-Control"] == "no-store"
            counts = stats.json()["http_responses"]
            assert counts["ops_metrics"]["401"] == 1
            assert counts["ops_metrics"]["404"] == 1
            assert counts["embedding"]["429"] == 1
            assert counts["embedding"]["422"] == 1
            assert stats.json()["scope"] == "gateway_process_only"
            content = stats.text
            for secret in (OWNER, CLIENT, SENTINEL, "client-sensitive", "/v1/embeddings"):
                assert secret not in content

    asyncio.run(scenario())


def test_settings_repr_hides_owner_and_redis_credentials():
    settings = make_settings(
        distributed_rate_enabled=True,
        distributed_rate_redis_url="rediss://operator:super-private-value@redis.example.org:6380/0",
    )
    rendered = repr(settings)
    for secret in (OWNER, CLIENT, "super-private-value", "rediss://"):
        assert secret not in rendered
    assert "max_inflight_requests" in rendered


def test_validation_response_never_echoes_raw_input_or_pillow_exception(monkeypatch):
    async def scenario():
        app = create_app(make_settings(), InferenceService("mock"))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://local") as client:
            malformed = await client.post(
                "/v1/embeddings", headers=AUTH,
                json={"model": "wemm-embedding-9b", "text": SENTINEL,
                      "dimension": {"leaked": SENTINEL}},
            )
            assert malformed.status_code == 422
            assert SENTINEL not in malformed.text
            assert "input" not in malformed.text
            assert "ctx" not in malformed.text
            def noisy_image(_data):
                raise OSError("Pillow leaked: " + SENTINEL)
            monkeypatch.setattr(Image, "open", noisy_image)
            image = await client.post(
                "/v1/embeddings", headers=AUTH,
                json={"model": "wemm-embedding-9b",
                      "image_base64": base64.b64encode(b"safe").decode()},
            )
            assert image.status_code == 422
            assert "Invalid image" in image.text
            assert SENTINEL not in image.text

    asyncio.run(scenario())


def test_inference_failure_logs_no_provider_message(caplog):
    async def scenario():
        class EvilProvider:
            backend = "mock"
            async def encode_many_profiled(self, items):
                raise RuntimeError("Bearer " + OWNER + " " + SENTINEL)

        app = create_app(make_settings(), EvilProvider())
        with caplog.at_level(logging.ERROR):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://local",
            ) as client:
                failed = await client.post("/v1/embeddings", headers=AUTH, json=BODY)
                assert failed.status_code == 503
                assert OWNER not in failed.text and SENTINEL not in failed.text
        assert OWNER not in caplog.text
        assert SENTINEL not in caplog.text

    asyncio.run(scenario())


def test_security_monitor_cardinality_is_constant_and_unknown_values_ignored():
    stats = GatewaySecurityMonitor()
    for i in range(1000):
        stats.http(f"/private/customer/{i}", 998, detail="Bearer " + OWNER)
        stats.job("BEARER:" + OWNER)
    snapshot = stats.snapshot()
    assert len(snapshot["http_responses"]) == 1
    assert len(snapshot["http_responses"]["other"]) == 1
    assert snapshot["http_responses"]["other"]["500"] == 1000
    assert not any(snapshot["async_job_state_observations"].values())
    assert OWNER not in repr(snapshot)


def test_per_gateway_metrics_not_global():
    async def scenario():
        a = create_app(make_settings(), InferenceService("mock"))
        b = create_app(make_settings(), InferenceService("mock"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(a), base_url="http://a",
        ) as ca, httpx.AsyncClient(
            transport=httpx.ASGITransport(b), base_url="http://b",
        ) as cb:
            # Public /health is outside protected /v1 middleware metrics;
            # use the protected per-model metadata endpoint for isolation.
            assert (
                await ca.get("/v1/models/qwen3-vl-embedding-8b", headers=AUTH)
            ).status_code == 200
            one = (await ca.get("/v1/ops/metrics", headers=AUTH)).json()
            two = (await cb.get("/v1/ops/metrics", headers=AUTH)).json()
            assert one["http_responses"]["models"]["200"] == 1
            assert "models" not in two["http_responses"]

    asyncio.run(scenario())


def test_async_state_observations_and_store_outage_never_expose_exception(caplog):
    async def scenario():
        rows = DurableMemory()
        spawned = []

        async def spawn(job_id, _prepared):
            spawned.append(job_id)

        jobs = AsyncEmbeddingJobs(rows, spawn)
        app = create_app(make_settings(max_embeddings_per_minute=5),
                         InferenceService("mock"), async_jobs=jobs)
        with caplog.at_level(logging.ERROR):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://local",
            ) as client:
                created = await client.post(
                    "/v1/embeddings/async", headers=AUTH,
                    json={**BODY, "request_id": "monitor-test-job-001"},
                )
                assert created.status_code == 202
                assert len(spawned) == 1
                response = await client.get(
                    f"/v1/embeddings/jobs/{created.json()['job_id']}", headers=AUTH,
                )
                assert response.status_code == 202
                stats = (await client.get("/v1/ops/metrics", headers=AUTH)).json()
                assert stats["async_job_state_observations"]["QUEUED"] == 2

                async def failed_get(key):
                    raise RuntimeError("Redis password " + SENTINEL + " " + OWNER)

                rows.get = failed_get
                failure = await client.get(
                    f"/v1/embeddings/jobs/{created.json()['job_id']}", headers=AUTH,
                )
                assert failure.status_code == 503
                assert SENTINEL not in failure.text and OWNER not in failure.text
                current = (await client.get("/v1/ops/metrics", headers=AUTH)).json()
                assert current["http_responses"]["async_status"]["503"] == 1
        assert SENTINEL not in caplog.text and OWNER not in caplog.text

    asyncio.run(scenario())
