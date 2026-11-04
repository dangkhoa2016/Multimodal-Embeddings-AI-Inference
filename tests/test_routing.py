import asyncio
import math

import pytest

from multimodal_embeddings_ai_inference.service import InferenceService, model_revision
from multimodal_embeddings_ai_inference.validation import Prepared


class FakeWorker:
    def __init__(self, name):
        self.name = name
        self.calls = []
        parent = self

        class Remote:
            async def aio(self, model_id, items):
                assert model_id == parent.name
                parent.calls.append((model_id, len(items)))
                sign = 1 if model_id == "wemm-embedding-9b" else -1
                await asyncio.sleep(0)
                return {
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
