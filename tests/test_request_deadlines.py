"""Bound overload, scheduling waits and cleanup of HTTP-owned Modal calls."""

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.model_options import effective_encoding_profile
from multimodal_embeddings_ai_inference.service import InferenceOutcomeUnknown, InferenceService
from multimodal_embeddings_ai_inference.validation import Prepared

TOKEN = "offline-deadline-auth-token"
MODEL = "wemm-embedding-9b"
AUTH = {"Authorization": "Bearer " + TOKEN}


@pytest.fixture(autouse=True)
def pinned_revisions(monkeypatch):
    for key in ("WEMM9", "QWEN8"):
        monkeypatch.setenv(f"VISION_EMBEDDINGS_{key}_REVISION", "a" * 40)


class ControlledWorker:
    def __init__(self, *, fail=False, spawn_ack=None):
        self.calls = []
        self.began, self.release = asyncio.Event(), asyncio.Event()
        self.fail, self.spawn_ack = fail, spawn_ack

        async def spawn(model, items):
            cancelled = asyncio.Event()
            record = {"cancelled": cancelled, "terminate_containers": None}
            self.calls.append(record)
            self.began.set()

            async def get():
                if self.fail:
                    raise RuntimeError("Controlled worker failure")
                await self.release.wait()
                repository = ("tencent/WeMM-Embedding-9B" if model == MODEL
                              else "Qwen/Qwen3-VL-Embedding-8B")
                return {"checkpoint_identity": {
                    "model": model, "repository": repository, "revision": "a" * 40,
                }, "effective_profiles": [
                    effective_encoding_profile(
                        model, kind=item["kind"], dimension=item["dimension"],
                        instruction=item.get("instruction"),
                        options=item.get("model_config"),
                    ) for item in items
                ], "vectors": [[1.0] + [0.0] * (item["dimension"] - 1)
                                    for item in items]}

            async def cancel(terminate_containers=False):
                record["terminate_containers"] = terminate_containers
                cancelled.set()

            handle = SimpleNamespace(get=SimpleNamespace(aio=get),
                                     cancel=SimpleNamespace(aio=cancel))
            if self.spawn_ack is not None:
                await self.spawn_ack.wait()
            return handle

        async def remote(model, items):
            return await (await spawn(model, items)).get.aio()

        self.encode_many = SimpleNamespace(spawn=SimpleNamespace(aio=spawn),
                                           remote=SimpleNamespace(aio=remote))


def sample(model=MODEL):
    return Prepared(model=model, payload="public test", kind="text", dimension=64,
                    input_sha256="", request_id=None)


def test_ten_concurrent_inputs_dispatch_four_and_reject_six_without_blocking_health():
    source = ast.parse((Path(__file__).resolve().parents[1] / "modal_app.py").read_text())
    gateway = next(node for node in source.body
                   if isinstance(node, ast.FunctionDef) and node.name == "api_entrypoint")
    decorator = next(dec for dec in gateway.decorator_list if isinstance(dec, ast.Call)
                     and isinstance(dec.func, ast.Attribute) and dec.func.attr == "concurrent")
    capacity = next(ast.literal_eval(kw.value) for kw in decorator.keywords
                    if kw.arg == "max_inputs")
    assert capacity >= 11, "Ten HTTP inputs must leave gateway capacity for a health request"

    async def scenario():
        worker = ControlledWorker()
        app = create_app(Settings(api_token=TOKEN, backend="modal"),
                         InferenceService("modal", {MODEL: worker}))
        slots = asyncio.Semaphore(capacity)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://gateway.test") as client:
            async def post():
                async with slots:
                    return await client.post("/v1/embeddings", headers=AUTH, json={
                        "model": MODEL, "text": "public burst", "dimension": 64,
                    })

            tasks = [asyncio.create_task(post()) for _ in range(10)]
            try:
                async def rejected():
                    while len([task for task in tasks if task.done()]) < 6:
                        await asyncio.sleep(0)

                await asyncio.wait_for(rejected(), 1)
                done = [task.result() for task in tasks if task.done()]
                assert len(done) == 6 and all(reply.status_code == 429 for reply in done)
                assert all("Retry-After" in reply.headers for reply in done)
                async with slots:
                    health = await asyncio.wait_for(client.get("/health", headers=AUTH), 1)
                assert health.status_code == 200
                assert len(worker.calls) == 4
            finally:
                worker.release.set()
                replies = await asyncio.wait_for(asyncio.gather(*tasks), 2)
                assert sorted(reply.status_code for reply in replies) == [200] * 4 + [429] * 6

    asyncio.run(scenario())


def test_inference_deadline_cancels_remote_input_returns_504_and_releases_admission():
    async def scenario():
        worker = ControlledWorker()
        settings = Settings(api_token=TOKEN, backend="modal", max_inflight_requests=1,
                            inference_timeout_seconds=0.05)
        app = create_app(settings, InferenceService("modal", {MODEL: worker}))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://gateway.test") as client:
            body = {"model": MODEL, "text": "public test", "dimension": 64}
            result = await asyncio.wait_for(
                client.post("/v1/embeddings", headers=AUTH, json=body), 1
            )
            assert result.status_code == 504 and result.headers["Retry-After"]
            assert worker.calls[0]["cancelled"].is_set()
            assert worker.calls[0]["terminate_containers"] is True
            worker.release.set()
            recovered = await client.post("/v1/embeddings", headers=AUTH, json=body)
            assert recovered.status_code == 200

    asyncio.run(scenario())


def test_modal_deadline_does_not_interrupt_serialized_local_inference():
    async def scenario():
        began, release = threading.Event(), threading.Event()
        active, peak = 0, 0
        guard = threading.Lock()

        class BlockingEncoder:
            def encode(self, payload, kind, dimension):
                nonlocal active, peak
                with guard:
                    active += 1
                    peak = max(peak, active)
                began.set()
                try:
                    assert release.wait(2), "Test did not release local inference"
                    return [1.0] + [0.0] * (dimension - 1)
                finally:
                    with guard:
                        active -= 1

        service = InferenceService("local")
        service._local_models[MODEL] = BlockingEncoder()
        app = create_app(Settings(api_token=TOKEN, backend="local",
                                  inference_timeout_seconds=0.01), service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://gateway.test") as client:
            body = {"model": MODEL, "text": "public test", "dimension": 64}
            first = asyncio.create_task(client.post("/v1/embeddings", headers=AUTH, json=body))
            requests = [first]
            try:
                assert await asyncio.to_thread(began.wait, 1)
                requests.append(asyncio.create_task(
                    client.post("/v1/embeddings", headers=AUTH, json=body)
                ))
                done, _ = await asyncio.wait(requests, timeout=0.07)
                assert not done, "A Modal-only deadline must not cancel a local encoder thread"
            finally:
                release.set()
                replies = await asyncio.gather(*requests)
            assert [reply.status_code for reply in replies] == [200, 200]
            assert peak == 1

    asyncio.run(scenario())


def test_caller_cancellation_cancels_the_owned_remote_call():
    async def scenario():
        worker = ControlledWorker()
        service = InferenceService("modal", {MODEL: worker})
        task = asyncio.create_task(service.encode_many([sample()]))
        await asyncio.wait_for(worker.began.wait(), 1)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(worker.calls[0]["cancelled"].wait(), 1)
        assert worker.calls[0]["terminate_containers"] is True

    asyncio.run(scenario())


def test_failed_batch_group_cancels_other_model_work():
    async def scenario():
        first, second = ControlledWorker(), ControlledWorker(fail=True)
        qwen = "qwen3-vl-embedding-8b"
        service = InferenceService("modal", {MODEL: first, qwen: second})
        with pytest.raises(InferenceOutcomeUnknown, match="unknown"):
            await asyncio.wait_for(service.encode_many([sample(), sample(qwen)]), 1)
        await asyncio.wait_for(first.calls[0]["cancelled"].wait(), 1)

    asyncio.run(scenario())


def test_cancellation_during_spawn_cleans_up_the_late_call_handle():
    async def scenario():
        acknowledged = asyncio.Event()
        worker = ControlledWorker(spawn_ack=acknowledged)
        service = InferenceService("modal", {MODEL: worker})
        task = asyncio.create_task(service.encode_many([sample()]))
        await asyncio.wait_for(worker.began.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        acknowledged.set()
        await asyncio.wait_for(worker.calls[0]["cancelled"].wait(), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["0", "-1", "121", "nan", "inf"])
def test_invalid_inference_deadline_is_rejected(monkeypatch, value):
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", TOKEN)
    monkeypatch.setenv("VISION_EMBEDDINGS_INFERENCE_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        Settings.from_env()
