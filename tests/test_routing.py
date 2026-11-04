import asyncio
import math

from compactfaces_ai_inference.service import InferenceService, model_revision
from compactfaces_ai_inference.validation import Prepared


class FakeRemoteMethod:
    class encode_many:
        class remote:
            @staticmethod
            async def aio(model_id, items):
                n = 1 if model_id == "wemm-embedding-9b" else -1
                return [[float(n)] * item["dimension"] for item in items]


def test_both_models_go_to_one_remote_worker():
    service = InferenceService("modal", FakeRemoteMethod())
    inputs = [
        Prepared(
            model="wemm-embedding-9b",
            payload="a",
            kind="text",
            dimension=64,
            input_sha256="",
            request_id=None,
        ),
        Prepared(
            model="qwen3-vl-embedding-8b",
            payload="b",
            kind="text",
            dimension=64,
            input_sha256="",
            request_id=None,
        ),
    ]
    output = asyncio.run(service.encode_many(inputs))
    assert len(output) == 2
    assert output[0][0] > 0 and output[1][0] < 0
    assert all(abs(math.sqrt(sum(x * x for x in o)) - 1) < 1e-4 for o in output)


def test_revision_keys(monkeypatch):
    monkeypatch.setenv("CF_WEMM9_REVISION", "w" * 40)
    monkeypatch.setenv("CF_QWEN8_REVISION", "q" * 40)
    assert model_revision("wemm-embedding-9b") == "w" * 40
    assert model_revision("qwen3-vl-embedding-8b") == "q" * 40
