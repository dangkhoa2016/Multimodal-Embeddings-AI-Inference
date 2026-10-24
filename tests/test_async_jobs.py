"""CPU-only checks for idempotent Modal-style embedding jobs and polling."""
import asyncio

import httpx
import pytest

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.async_jobs import (
    AsyncEmbeddingJobs,
    execute_job,
    fingerprint,
)
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.service import InferenceService
from multimodal_embeddings_ai_inference.validation import Prepared

TOKEN = "test-async-bearer-123456789"
MODEL = "wemm-embedding-9b"
AUTH = {"Authorization": "Bearer " + TOKEN}


class MemoryDict:
    def __init__(self):
        self.rows = {}
        self.lock = asyncio.Lock()

    async def get(self, key):
        return self.rows.get(key)

    async def put(self, key, value, skip_if_exists=False):
        async with self.lock:
            if skip_if_exists and key in self.rows:
                return False
            self.rows[key] = value
            return True

    async def pop(self, key, default=None):
        async with self.lock:
            return self.rows.pop(key, default)


class FakeRunner:
    def __init__(self, store, delay=0.1, uncertain_ack=False):
        self.store = store
        self.delay = delay
        self.calls = []
        self.tasks = []
        self.uncertain_ack = uncertain_ack

    async def __call__(self, key, prepared_data):
        self.calls.append(key)
        parent = self

        class SlowService:
            backend = "mock"

            async def encode_many_profiled(self, items):
                await asyncio.sleep(parent.delay)
                service = InferenceService("mock")
                return await service.encode_many_profiled(items)

        task = asyncio.create_task(execute_job(
            key, prepared_data, self.store, SlowService(),
        ))
        self.tasks.append(task)
        if self.uncertain_ack:
            raise RuntimeError("ack lost")


def request(text="face", request_id="wemm-case-0031"):
    return {
        "model": MODEL, "text": text, "dimension": 64,
        "request_id": request_id,
    }


def heartbeat_fingerprint(request_id):
    return fingerprint(Prepared(MODEL, "hello", "text", 64, "a" * 64, request_id))


def test_async_submit_poll_dedupe_and_no_60_second_deadline():
    async def scenario():
        store = MemoryDict()
        runner = FakeRunner(store, delay=0.5)
        settings = Settings(api_token=TOKEN, backend="modal",
                            inference_timeout_seconds=0.01)
        jobs = AsyncEmbeddingJobs(store, runner)
        app = create_app(settings, InferenceService("mock"), async_jobs=jobs)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway.test",
        ) as client:
            first = await client.post("/v1/embeddings/async", headers=AUTH, json=request())
            assert first.status_code == 202, first.text
            assert runner.tasks and not runner.tasks[0].done(), "HTTP must return before inference"
            key = first.json()["job_id"]
            assert len(key) == 64
            assert first.headers["Cache-Control"] == "no-store"
            assert "Retry-After" in first.headers

            duplicate = await client.post("/v1/embeddings/async", headers=AUTH, json=request())
            assert duplicate.status_code == 202
            assert duplicate.json()["job_id"] == key
            assert len(runner.calls) == 1

            conflict = await client.post("/v1/embeddings/async", headers=AUTH,
                                         json=request(text="different"))
            assert conflict.status_code == 409
            await asyncio.gather(*runner.tasks)
            completed = await client.get(f"/v1/embeddings/jobs/{key}", headers=AUTH)
            assert completed.status_code == 200
            assert completed.json()["state"] == "SUCCEEDED"
            data = completed.json()["result"]
            assert len(data["vector"]) == 64
            assert data["request_id"] == "wemm-case-0031"
            assert "image_base64" not in str(completed.json())
            again = await client.post("/v1/embeddings/async", headers=AUTH, json=request())
            assert again.status_code == 200
            assert len(runner.calls) == 1
            assert again.json()["result"]["vector"] == data["vector"]

    asyncio.run(scenario())


def test_concurrent_admission_claims_one_job():
    async def scenario():
        store = MemoryDict()
        runner = FakeRunner(store, delay=0.08)
        jobs = AsyncEmbeddingJobs(store, runner)
        app = create_app(Settings(api_token=TOKEN), InferenceService("mock"),
                         async_jobs=jobs)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://gateway.test") as client:
            replies = await asyncio.gather(*(client.post(
                "/v1/embeddings/async", headers=AUTH,
                json=request(request_id="wemm-concurrent-001"),
            ) for _ in range(8)))
            assert all(r.status_code == 202 for r in replies)
            assert len({r.json()["job_id"] for r in replies}) == 1
            assert len(runner.calls) == 1
        await asyncio.gather(*runner.tasks)
    asyncio.run(scenario())


def test_uncertain_spawn_ack_never_double_dispatches():
    async def scenario():
        store = MemoryDict()
        runner = FakeRunner(store, uncertain_ack=True, delay=0.02)
        jobs = AsyncEmbeddingJobs(store, runner)
        app = create_app(Settings(api_token=TOKEN), InferenceService("mock"), async_jobs=jobs)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://gateway.test") as client:
            first = await client.post("/v1/embeddings/async", headers=AUTH, json=request())
            assert first.status_code == 202
            again = await client.post("/v1/embeddings/async", headers=AUTH, json=request())
            assert again.status_code == 202
            assert len(runner.calls) == 1
            await asyncio.gather(*runner.tasks)
            result = await client.get("/v1/embeddings/jobs/" + first.json()["job_id"],
                                      headers=AUTH)
            assert result.json()["state"] == "SUCCEEDED"
    asyncio.run(scenario())


def test_same_job_survives_bearer_rotation():
    async def scenario():
        store = MemoryDict()
        runner = FakeRunner(store, delay=0.01)
        old = create_app(
            Settings(api_token=TOKEN), InferenceService("mock"),
            async_jobs=AsyncEmbeddingJobs(store, runner),
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(old),
                                     base_url="http://gateway.test") as client:
            started = await client.post("/v1/embeddings/async", headers=AUTH, json=request())
            assert started.status_code == 202
        await asyncio.gather(*runner.tasks)

        rotated = "new-rotated-bearer-token-123456"
        new = create_app(
            Settings(api_token=rotated), InferenceService("mock"),
            async_jobs=AsyncEmbeddingJobs(store, runner),
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(new),
                                     base_url="http://gateway.test") as client:
            again = await client.post(
                "/v1/embeddings/async",
                headers={"Authorization": "Bearer " + rotated},
                json=request(),
            )
            assert again.status_code == 200
            assert again.json()["job_id"] == started.json()["job_id"]
            assert len(runner.calls) == 1
            assert (await client.get(
                "/v1/embeddings/jobs/" + again.json()["job_id"],
                headers={"Authorization": "Bearer " + rotated},
            )).status_code == 200
    asyncio.run(scenario())


def test_async_endpoints_auth_validation_not_found():
    async def scenario():
        store = MemoryDict()
        runner = FakeRunner(store)
        jobs = AsyncEmbeddingJobs(store, runner)
        app = create_app(Settings(api_token=TOKEN), InferenceService("mock"), async_jobs=jobs)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                     base_url="http://gateway.test") as client:
            assert (await client.post("/v1/embeddings/async", json=request())).status_code == 401
            assert (await client.get("/v1/embeddings/jobs/" + "a" * 64)).status_code == 401
            invalid = request(request_id="x")
            assert (await client.post("/v1/embeddings/async", json=invalid,
                                      headers=AUTH)).status_code == 422
            unknown = await client.get("/v1/embeddings/jobs/" + "a" * 64, headers=AUTH)
            assert unknown.status_code == 404
            assert len(runner.calls) == 0
    asyncio.run(scenario())


def test_stale_running_job_warns_without_releasing_idempotency(monkeypatch):
    """A worker interruption is not proof that GPU work stopped."""
    from multimodal_embeddings_ai_inference import async_jobs as module

    async def scenario():
        store = MemoryDict()
        runner = FakeRunner(store)
        jobs = AsyncEmbeddingJobs(store, runner)
        key = module.job_key("stale-test-case-001")
        now = 2_000_000
        monkeypatch.setattr(module.time, "time", lambda: now)
        store.rows[key] = {
            "state": "RUNNING", "created_at": now - 7200,
            "updated_at": now - module.RUNNING_STALE_SECONDS - 1,
            "fingerprint": "opaque-test-fingerprint",
        }
        status, code = await jobs.status(key)
        assert code == 200 and status["state"] == "INDETERMINATE"
        assert "might still run" in status["note"]
        assert store.rows[key]["state"] == "RUNNING"
        assert runner.calls == []
        store.rows[key] = {**store.rows[key], "state": "SUCCEEDED",
                           "result": {"vector": [1.0]}, "updated_at": now}
        status, code = await jobs.status(key)
        assert code == 200 and status["state"] == "SUCCEEDED"
        assert runner.calls == []

    asyncio.run(scenario())



class PausedHeartbeatDict(MemoryDict):
    """Hold a heartbeat between its RUNNING read and stale RUNNING write."""
    def __init__(self):
        super().__init__()
        self.pending = asyncio.Event()
        self.release = asyncio.Event()
        self.writes = []

    async def put(self, key, value, skip_if_exists=False):
        if (value["state"] == "RUNNING"
                and self.rows.get(key, {}).get("state") == "RUNNING"):
            self.pending.set()
            await asyncio.wait_for(self.release.wait(), timeout=2)
        self.writes.append(value["state"])
        return await super().put(key, value, skip_if_exists=skip_if_exists)


def test_heartbeat_cannot_overwrite_terminal_state(monkeypatch):
    """Exercise the previously reproduced FAILED -> RUNNING race on CPU."""
    from multimodal_embeddings_ai_inference import async_jobs as module

    monkeypatch.setattr(module, "JOB_HEARTBEAT_SECONDS", 0.001)

    async def scenario(failing):
        store = PausedHeartbeatDict()
        job_id = "f" * 64
        store.rows[job_id] = {
            "fingerprint": heartbeat_fingerprint("heartbeat-race-test-001"), "state": "QUEUED",
            "created_at": 1, "updated_at": 1,
        }
        finished = asyncio.Event()

        class ControlledService:
            backend = "mock"

            async def encode_many_profiled(self, items):
                await asyncio.wait_for(store.pending.wait(), timeout=2)
                finished.set()
                if failing:
                    raise RuntimeError("intentional CPU-only failure")
                return await InferenceService("mock").encode_many_profiled(items)

        prepared = {
            "model": MODEL, "payload": "hello", "kind": "text",
            "dimension": 64, "input_sha256": "a" * 64,
            "request_id": "heartbeat-race-test-001",
        }
        job = asyncio.create_task(execute_job(
            job_id, prepared, store, ControlledService(),
        ))
        try:
            await asyncio.wait_for(finished.wait(), timeout=2)
            await asyncio.sleep(0)
        finally:
            store.release.set()
        await asyncio.wait_for(job, timeout=2)
        terminal = "FAILED" if failing else "SUCCEEDED"
        assert store.rows[job_id]["state"] == terminal
        assert store.writes[-1] == terminal
        assert store.writes.count(terminal) == 1
        assert "RUNNING" not in store.writes[store.writes.index(terminal) + 1:]

    async def run():
        await scenario(True)
        await scenario(False)

    asyncio.run(run())


def test_heartbeat_write_exception_does_not_mask_success(monkeypatch):
    from multimodal_embeddings_ai_inference import async_jobs as module

    monkeypatch.setattr(module, "JOB_HEARTBEAT_SECONDS", 0.001)

    async def scenario():
        heartbeat_attempted = asyncio.Event()

        class FlakyStore(MemoryDict):
            async def put(self, key, value, skip_if_exists=False):
                if (value["state"] == "RUNNING"
                        and self.rows.get(key, {}).get("state") == "RUNNING"):
                    heartbeat_attempted.set()
                    raise OSError("injected heartbeat error")
                return await super().put(key, value, skip_if_exists=skip_if_exists)

        store = FlakyStore()
        job_id = "e" * 64
        store.rows[job_id] = {
            "fingerprint": heartbeat_fingerprint("heartbeat-error-test-001"), "state": "QUEUED",
            "created_at": 1, "updated_at": 1,
        }

        class MockService:
            backend = "mock"

            async def encode_many_profiled(self, items):
                await asyncio.wait_for(heartbeat_attempted.wait(), timeout=2)
                return await InferenceService("mock").encode_many_profiled(items)

        await asyncio.wait_for(execute_job(job_id, {
            "model": MODEL, "payload": "hello", "kind": "text",
            "dimension": 64, "input_sha256": "a" * 64,
            "request_id": "heartbeat-error-test-001",
        }, store, MockService()), timeout=2)
        assert store.rows[job_id]["state"] == "SUCCEEDED"
        assert len(store.rows[job_id]["result"]["vector"]) == 64

    asyncio.run(scenario())


def test_cancelled_coordinator_keeps_gpu_outcome_unknown(monkeypatch):
    """Cancellation must not invent a terminal state or restart GPU work."""
    from multimodal_embeddings_ai_inference import async_jobs as module

    monkeypatch.setattr(module, "JOB_HEARTBEAT_SECONDS", 0.001)

    async def scenario():
        store = MemoryDict()
        job_id = "d" * 64
        store.rows[job_id] = {
            "fingerprint": heartbeat_fingerprint("heartbeat-cancel-test-001"), "state": "QUEUED",
            "created_at": 1, "updated_at": 1,
        }
        entered = asyncio.Event()

        class BlockedService:
            async def encode_many_profiled(self, items):
                entered.set()
                await asyncio.Event().wait()

        job = asyncio.create_task(execute_job(job_id, {
            "model": MODEL, "payload": "hello", "kind": "text",
            "dimension": 64, "input_sha256": "a" * 64,
            "request_id": "heartbeat-cancel-test-001",
        }, store, BlockedService()))
        await asyncio.wait_for(entered.wait(), timeout=2)
        job.cancel()
        try:
            await asyncio.wait_for(job, timeout=2)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation must propagate")
        assert store.rows[job_id]["state"] == "RUNNING"
        previous = store.rows[job_id]["updated_at"]
        await asyncio.sleep(0.01)
        assert store.rows[job_id]["updated_at"] == previous

    asyncio.run(scenario())


def test_stale_queue_warns_without_dispatching(monkeypatch):
    from multimodal_embeddings_ai_inference import async_jobs as module

    async def scenario():
        store = MemoryDict()
        runner = FakeRunner(store)
        key = module.job_key("queued-test-case-001")
        now = 2_000_000
        monkeypatch.setattr(module.time, "time", lambda: now)
        store.rows[key] = {
            "state": "QUEUED", "created_at": now - module.QUEUED_STALE_SECONDS - 1,
            "updated_at": now - module.QUEUED_STALE_SECONDS - 1,
            "fingerprint": "opaque-test-fingerprint",
        }
        status, code = await AsyncEmbeddingJobs(store, runner).status(key)
        assert code == 200 and status["state"] == "INDETERMINATE"
        assert store.rows[key]["state"] == "QUEUED"
        assert runner.calls == []

    asyncio.run(scenario())


def test_detached_coordinator_uses_its_declared_revision_secret(monkeypatch):
    """Gateway variables do not automatically reach a detached Modal function."""
    import ast
    import hashlib
    import json
    import os
    from pathlib import Path
    from unittest.mock import patch

    from multimodal_embeddings_ai_inference import service as service_module
    from multimodal_embeddings_ai_inference.api import _InferenceQuota
    from multimodal_embeddings_ai_inference.schemas import EmbeddingRequest
    from multimodal_embeddings_ai_inference.validation import prepare

    pin = "a" * 40
    options = {}

    class App:
        def function(self, **kwargs):
            options.update(kwargs)
            return lambda function: function

    class Encoder:
        def __init__(self, backend, workers):
            self.backend = backend

        async def encode_many_profiled(self, items):
            return [[1.0] + [0.0] * 63], [None]

    monkeypatch.setattr(service_module, "InferenceService", Encoder)
    tree = ast.parse((Path(__file__).resolve().parents[1] / "modal_app.py").read_text())
    entrypoint = next(node for node in tree.body
                      if isinstance(node, ast.AsyncFunctionDef)
                      and node.name == "run_embedding_job")
    secret = {"VISION_EMBEDDINGS_WEMM9_REVISION": pin}

    async def scenario():
        store, queue = MemoryDict(), []

        async def spawn(key, data):
            queue.append((key, data))

        with patch.dict(os.environ, secret, clear=True):
            item = prepare(EmbeddingRequest(**request()), Settings(api_token=TOKEN))
            await AsyncEmbeddingJobs(store, spawn).submit(item, _InferenceQuota(120, 4))
        scope = {"app": App(), "gateway_image": object(), "model_revision_secret": secret,
                 "QwenL40SGpu": object, "WeMML40SGpu": object,
                 "_ModalJobStore": lambda: store}
        exec(compile(ast.Module(body=[entrypoint], type_ignores=[]), "modal_app.py", "exec"),
             scope)
        coordinator_env = {key: value for values in options.get("secrets", [])
                           for key, value in values.items()}
        with patch.dict(os.environ, coordinator_env, clear=True):
            await scope["run_embedding_job"](*queue[0])
        result = store.rows[queue[0][0]]["result"]
        assert result["revision"] == pin
        expected = {"dimension": 64, "instruction_sha256": None, "model": MODEL,
                    "preprocessing_version": "wemm-adapter-v1", "revision": pin}
        digest = hashlib.sha256(json.dumps(
            expected, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        assert result["embedding_metadata"]["config_sha256"] == digest

    asyncio.run(scenario())


def test_changed_coordinator_revision_fails_before_encoding(monkeypatch):
    from multimodal_embeddings_ai_inference.api import _InferenceQuota
    from multimodal_embeddings_ai_inference.schemas import EmbeddingRequest
    from multimodal_embeddings_ai_inference.validation import prepare

    async def scenario():
        store, queue, encoded = MemoryDict(), [], []

        async def spawn(key, data):
            queue.append((key, data))

        monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", "a" * 40)
        item = prepare(EmbeddingRequest(**request()), Settings(api_token=TOKEN))
        await AsyncEmbeddingJobs(store, spawn).submit(item, _InferenceQuota(120, 4))
        # Also protect queued records created before explicit revision was stored.
        store.rows[queue[0][0]].pop("revision", None)
        monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", "b" * 40)

        class Encoder:
            backend = "modal"

            async def encode_many_profiled(self, items):
                encoded.extend(items)
                return [[1.0] + [0.0] * 63], [None]

        await execute_job(*queue[0], store, Encoder())
        assert store.rows[queue[0][0]]["state"] == "FAILED"
        assert encoded == []

    asyncio.run(scenario())


def test_async_admission_holds_slot_until_dispatch_acknowledgement():
    from fastapi import HTTPException

    from multimodal_embeddings_ai_inference.api import _InferenceQuota
    from multimodal_embeddings_ai_inference.schemas import EmbeddingRequest
    from multimodal_embeddings_ai_inference.validation import prepare

    async def scenario():
        store, entered, release = MemoryDict(), asyncio.Event(), asyncio.Event()

        async def spawn(key, data):
            if not entered.is_set():
                entered.set()
                await release.wait()

        jobs, quota = AsyncEmbeddingJobs(store, spawn), _InferenceQuota(120, 1)
        first = prepare(EmbeddingRequest(**request()), Settings(api_token=TOKEN))
        second = prepare(EmbeddingRequest(**request(request_id="held-ack-second-001")),
                         Settings(api_token=TOKEN))
        task = asyncio.create_task(jobs.submit(first, quota))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            duplicate, status = await jobs.submit(first, quota)
            assert status == 202 and duplicate["state"] == "QUEUED"
            with pytest.raises(HTTPException) as error:
                await jobs.submit(second, quota)
            assert error.value.status_code == 429
        finally:
            release.set()
            await task
        _, status = await jobs.submit(second, quota)
        assert status == 202

    asyncio.run(scenario())


def test_cancelled_dispatch_releases_admission_without_redispatch():
    from multimodal_embeddings_ai_inference.api import _InferenceQuota
    from multimodal_embeddings_ai_inference.schemas import EmbeddingRequest
    from multimodal_embeddings_ai_inference.validation import prepare

    async def scenario():
        store, entered, calls = MemoryDict(), asyncio.Event(), []

        async def spawn(key, data):
            calls.append(key)
            if len(calls) == 1:
                entered.set()
                await asyncio.Event().wait()

        jobs, quota = AsyncEmbeddingJobs(store, spawn), _InferenceQuota(120, 1)
        item = prepare(EmbeddingRequest(**request()), Settings(api_token=TOKEN))
        task = asyncio.create_task(jobs.submit(item, quota))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        duplicate, status = await jobs.submit(item, quota)
        assert status == 202 and duplicate["state"] == "QUEUED" and len(calls) == 1
        other = prepare(EmbeddingRequest(**request(request_id="after-cancel-test-001")),
                        Settings(api_token=TOKEN))
        _, status = await jobs.submit(other, quota)
        assert status == 202 and len(calls) == 2

    asyncio.run(scenario())
