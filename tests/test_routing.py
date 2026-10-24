import asyncio
import math
from types import SimpleNamespace

import pytest

from multimodal_embeddings_ai_inference.service import InferenceService, model_revision
from multimodal_embeddings_ai_inference.validation import Prepared

MODELS = {
    "wemm-embedding-9b": ("tencent/WeMM-Embedding-9B", "WEMM9", "a" * 40),
    "qwen3-vl-embedding-8b": ("Qwen/Qwen3-VL-Embedding-8B", "QWEN8", "b" * 40),
}


@pytest.fixture(autouse=True)
def pinned_revisions(monkeypatch):
    for _, key, revision in MODELS.values():
        monkeypatch.setenv(f"VISION_EMBEDDINGS_{key}_REVISION", revision)


class FakeWorker:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.checkpoint_identity = {
            "model": name, "repository": MODELS[name][0], "revision": MODELS[name][2],
        }
        parent = self

        class Remote:
            async def aio(self, model_id, items):
                assert model_id == parent.name
                parent.calls.append((model_id, len(items)))
                assert all(item["expected_revision"] == MODELS[model_id][2] for item in items)
                sign = 1 if model_id == "wemm-embedding-9b" else -1
                await asyncio.sleep(0)
                return {
                    "checkpoint_identity": parent.checkpoint_identity,
                    "vectors": [[float(sign)] * item["dimension"] for item in items],
                    "timing": {
                        "model_state": "initial_load",
                        "gpu_worker_first_request": True,
                        "model_load_ms": 200.0,
                        "gpu_encode_ms": 13.5,
                        "gpu_worker_ms": 220.0,
                        "gpu_worker_id": "a" * 32,
                        "batch_size": len(items),
                    },
                }

        class Method:
            remote = Remote()

            class spawn:
                @staticmethod
                async def aio(model_id, items):
                    async def get():
                        return await Remote().aio(model_id, items)

                    async def cancel(terminate_containers=False):
                        pass

                    return SimpleNamespace(get=SimpleNamespace(aio=get),
                                           cancel=SimpleNamespace(aio=cancel))

        self.encode_many = Method()


def prepared(model, payload="hello"):
    return Prepared(
        model=model,
        payload=payload,
        kind="text",
        dimension=64,
        input_sha256="",
        request_id=None,
    )


def test_each_model_routes_to_its_own_gpu_worker():
    qwen = FakeWorker("qwen3-vl-embedding-8b")
    wemm = FakeWorker("wemm-embedding-9b")
    service = InferenceService("modal", {qwen.name: qwen, wemm.name: wemm})
    output = asyncio.run(service.encode_many([prepared(wemm.name), prepared(qwen.name)]))
    assert len(output) == 2
    assert output[0][0] > 0 and output[1][0] < 0
    assert qwen.calls == [(qwen.name, 1)]
    assert wemm.calls == [(wemm.name, 1)]
    assert all(abs(math.sqrt(sum(x * x for x in o)) - 1) < 1e-4 for o in output)


def test_batch_grouping_stays_on_dedicated_workers():
    qwen = FakeWorker("qwen3-vl-embedding-8b")
    wemm = FakeWorker("wemm-embedding-9b")
    service = InferenceService("modal", {qwen.name: qwen, wemm.name: wemm})
    vectors, metrics = asyncio.run(
        service.encode_many_profiled(
            [prepared(qwen.name), prepared(wemm.name), prepared(qwen.name)]
        )
    )
    assert len(vectors) == len(metrics) == 3
    assert qwen.calls == [(qwen.name, 2)]
    assert wemm.calls == [(wemm.name, 1)]
    assert all(m["gpu_dispatch_overhead_ms"] >= 0 for m in metrics)


def test_unknown_gpu_route_fails_closed():
    service = InferenceService("modal", {})
    with pytest.raises(ValueError, match="No Modal GPU worker"):
        asyncio.run(service.encode_many([prepared("wemm-embedding-9b")]))


def test_revision_keys(monkeypatch):
    monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", "w" * 40)
    monkeypatch.setenv("VISION_EMBEDDINGS_QWEN8_REVISION", "q" * 40)
    assert model_revision("wemm-embedding-9b") == "w" * 40
    assert model_revision("qwen3-vl-embedding-8b") == "q" * 40


@pytest.mark.parametrize("bad_identity", [
    None, {}, {"model": "wemm-embedding-9b"},
    {"model": "qwen3-vl-embedding-8b", "repository": "tencent/WeMM-Embedding-9B",
     "revision": "a" * 40},
    {"model": "wemm-embedding-9b", "repository": "another/repository", "revision": "a" * 40},
    {"model": "wemm-embedding-9b", "repository": "tencent/WeMM-Embedding-9B",
     "revision": "b" * 40},
    {"model": "wemm-embedding-9b", "repository": "tencent/WeMM-Embedding-9B",
     "revision": "main"},
])
def test_service_rejects_unverified_worker_vectors(bad_identity):
    worker = FakeWorker("wemm-embedding-9b")
    worker.checkpoint_identity = bad_identity
    service = InferenceService("modal", {worker.name: worker})
    with pytest.raises(ValueError, match="checkpoint identity"):
        asyncio.run(service.encode_many([prepared(worker.name)]))


@pytest.mark.parametrize("revision", ["main", "abc", "w" * 40, ""])
def test_service_rejects_unpinned_expected_revision_before_dispatch(monkeypatch, revision):
    worker = FakeWorker("wemm-embedding-9b")
    monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", revision)
    service = InferenceService("modal", {worker.name: worker})
    with pytest.raises(ValueError, match="revision"):
        asyncio.run(service.encode_many([prepared(worker.name)]))
    assert not worker.calls


def test_async_job_rejects_worker_drift_instead_of_mislabeling_result():
    from test_async_jobs import MemoryDict

    from multimodal_embeddings_ai_inference.api import _InferenceQuota
    from multimodal_embeddings_ai_inference.async_jobs import AsyncEmbeddingJobs, execute_job

    async def scenario():
        store, queued = MemoryDict(), []

        async def spawn(key, data):
            queued.append((key, data))

        worker = FakeWorker("wemm-embedding-9b")
        worker.checkpoint_identity["revision"] = "b" * 40
        service = InferenceService("modal", {worker.name: worker})
        item = Prepared(worker.name, "hello", "text", 64, "c" * 64, "worker-drift-001")
        jobs = AsyncEmbeddingJobs(store, spawn)
        await jobs.submit(item, _InferenceQuota(120, 4))
        await execute_job(*queued[0], store, service)
        record = store.rows[queued[0][0]]
        assert worker.calls == [(worker.name, 1)]
        assert record["state"] == "FAILED" and "result" not in record

    asyncio.run(scenario())
