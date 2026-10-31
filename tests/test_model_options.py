"""CPU-only per-model request configuration, transport and safety regressions."""
import asyncio
import base64
import io
import math
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.async_jobs import fingerprint
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.model_options import effective_encoding_profile, schema_for
from multimodal_embeddings_ai_inference.service import InferenceService
from multimodal_embeddings_ai_inference.validation import prepare

QWEN = "qwen3-vl-embedding-8b"
WEMM = "wemm-embedding-9b"
TOKEN = "model-config-local-test-token-123456789"
AUTH = {"Authorization": "Bearer " + TOKEN}
SETTINGS = Settings(api_token=TOKEN, backend="mock")


@pytest.fixture
def client():
    with TestClient(create_app(SETTINGS)) as api:
        yield api


def send(client, model, *, cfg=None, **kwargs):
    data = {"model": model, "text": "Compare the target image.", **kwargs}
    if cfg is not None:
        data["model_config"] = cfg
    return client.post("/v1/embeddings", headers=AUTH, json=data)


def image():
    img = Image.new("RGB", (48, 48), (15, 30, 60))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_catalog_advertises_per_model_options_and_defaults(client):
    qwen = client.get(f"/v1/models/{QWEN}", headers=AUTH).json()
    wemm = client.get(f"/v1/models/{WEMM}", headers=AUTH).json()
    qa = qwen["model_config_schema"]
    wa = wemm["model_config_schema"]
    assert qa["additionalProperties"] is False
    assert qa["properties"]["dimension"]["default"] == 1024
    assert qa["properties"]["max_length"]["default"] == 8192
    assert qa["properties"]["normalize"]["default"] is True
    assert "instruction" in qa["properties"]
    assert "instruction" not in wa["properties"]
    assert "max_length" not in wa["properties"]
    assert set(("min_pixels", "max_pixels", "normalize")) <= set(wa["properties"])
    assert schema_for(QWEN) == qa


@pytest.mark.parametrize("model,dimension", [(QWEN, 4096), (QWEN, 127),
                                               (WEMM, 512), (WEMM, 4096)])
def test_dimension_from_model_config_equals_legacy_field(client, model, dimension):
    old = send(client, model, dimension=dimension)
    new = send(client, model, cfg={"dimension": dimension})
    assert old.status_code == new.status_code == 200
    assert old.json()["vector"] == new.json()["vector"]
    assert old.json()["embedding_metadata"] == new.json()["embedding_metadata"]


def test_optional_defaults_and_explicit_defaults_are_identical(client):
    implicit = send(client, QWEN)
    explicit = send(client, QWEN, cfg={
        "normalize": True, "max_length": 8192, "dimension": 1024,
    })
    assert implicit.status_code == explicit.status_code == 200
    assert {k: v for k, v in implicit.json().items() if k != "elapsed_ms"} == (
        {k: v for k, v in explicit.json().items() if k != "elapsed_ms"}
    )


def test_qwen_instruction_alias_preserves_legacy_behavior(client):
    a = send(client, QWEN, instruction="Retrieve image descriptions.")
    b = send(client, QWEN, cfg={"instruction": "Retrieve image descriptions."})
    assert a.status_code == b.status_code == 200
    assert {k: v for k, v in a.json().items() if k != "elapsed_ms"} == (
        {k: v for k, v in b.json().items() if k != "elapsed_ms"}
    )


def test_per_request_options_change_profile_and_no_gpu_dispatch(client):
    default = send(client, QWEN)
    tuned = send(client, QWEN, cfg={"max_length": 4096})
    assert default.status_code == tuned.status_code == 200
    assert default.json()["embedding_metadata"]["config_sha256"] != (
        tuned.json()["embedding_metadata"]["config_sha256"]
    )
    assert default.json()["input_sha256"] == tuned.json()["input_sha256"]
    assert tuned.json()["normalized"] is True


@pytest.mark.parametrize("model", [QWEN, WEMM])
def test_normalize_false_is_real_and_records_provenance(client, model):
    default = send(client, model, cfg={"dimension": 64}).json()
    raw = send(client, model, cfg={"dimension": 64, "normalize": False}).json()
    assert raw["normalized"] is False
    assert default["normalized"] is True
    assert abs(math.sqrt(sum(x*x for x in default["vector"])) - 1.0) < 1e-5
    assert abs(math.sqrt(sum(x*x for x in raw["vector"])) - 1.0) > 0.01
    assert raw["embedding_metadata"]["config_sha256"] != (
        default["embedding_metadata"]["config_sha256"]
    )


@pytest.mark.parametrize("cfg", [
    {"extra": 1}, {"normalize": "false"}, {"normalize": 0},
    {"dimension": True}, {"max_length": -1}, {"max_length": 999999},
    {"max_length": 1.5}, {"min_pixels": -1}, {"max_pixels": False},
    {"instruction": ""}, {"instruction": "a\nb"},
    {"max_length": [1]}, {"unknown": {"arbitrary": [1]}},
    {f"option_{i}": i for i in range(17)},
])
def test_unknown_bad_or_unsafe_qwen_options_are_rejected(client, cfg):
    response = send(client, QWEN, cfg=cfg)
    assert response.status_code == 422, response.text
    assert "model-config-local-test-token" not in response.text


@pytest.mark.parametrize("cfg", [
    {"instruction": "retrieve"}, {"max_length": 512},
    {"dimension": 123}, {"normalize": None},
])
def test_wrong_wemm_options_rejected(client, cfg):
    assert send(client, WEMM, cfg=cfg).status_code == 422


def test_conflicting_top_level_and_model_config_are_rejected(client):
    assert send(client, QWEN, dimension=128,
                cfg={"dimension": 256}).status_code == 422
    assert send(client, QWEN, instruction="one",
                cfg={"instruction": "two"}).status_code == 422
    assert send(client, QWEN, dimension=128,
                cfg={"dimension": 128}).status_code == 200


def test_pixel_knobs_rejected_for_text_and_accepted_for_images(client):
    for model in (QWEN, WEMM):
        assert send(client, model, cfg={"max_pixels": 100000}).status_code == 422
        picture = {
            "model": model,
            "image_base64": image(),
            "model_config": {"min_pixels": 4096, "max_pixels": 8192},
        }
        resp = client.post("/v1/embeddings", headers=AUTH, json=picture)
        assert resp.status_code == 200, resp.text
    assert client.post("/v1/embeddings", headers=AUTH, json={
        "model": QWEN, "image_base64": image(),
        "model_config": {"min_pixels": 10000, "max_pixels": 4096},
    }).status_code == 422


def test_mixed_batch_has_independent_model_configs(client):
    items = [
        {"model": QWEN, "text": "alpha",
         "model_config": {"dimension": 128, "max_length": 2048,
                          "instruction": "Find matching images."}},
        {"model": WEMM, "text": "beta",
         "model_config": {"dimension": 256, "normalize": False}},
    ]
    response = client.post("/v1/embeddings/batches", headers=AUTH,
                           json={"items": items})
    assert response.status_code == 200, response.text
    a, b = response.json()["items"]
    assert len(a["vector"]) == 128 and a["normalized"] is True
    assert len(b["vector"]) == 256 and b["normalized"] is False


def test_async_fingerprint_binds_model_config_and_same_defaults():
    from multimodal_embeddings_ai_inference.schemas import EmbeddingRequest

    base = {"model": QWEN, "text": "semantic search", "request_id": "unique-config-001"}
    p1 = prepare(EmbeddingRequest.model_validate(base), SETTINGS)
    p2 = prepare(EmbeddingRequest.model_validate({
        **base, "model_config": {"max_length": 4096},
    }), SETTINGS)
    p3 = prepare(EmbeddingRequest.model_validate({
        **base, "model_config": {"normalize": True, "max_length": 8192},
    }), SETTINGS)
    assert fingerprint(p1) != fingerprint(p2)
    assert fingerprint(p1) == fingerprint(p3)


def test_modal_routing_keeps_model_options_separate(monkeypatch):
    for key in ("WEMM9", "QWEN8"):
        monkeypatch.setenv(f"VISION_EMBEDDINGS_{key}_REVISION", "a" * 40)

    class Worker:
        def __init__(self, model):
            self.inputs = []

            async def aio(model_id, items):
                assert model_id == model
                self.inputs.extend(items)
                return {
                    "vectors": [[float(i + 1) for i in range(x["dimension"])]
                                for x in items],
                    "effective_profiles": [
                        effective_encoding_profile(
                            model, kind=x["kind"], dimension=x["dimension"],
                            instruction=x.get("instruction"), options=x.get("model_config"),
                        ) for x in items
                    ],
                    "checkpoint_identity": {
                        "model": model, "repository": (
                            "Qwen/Qwen3-VL-Embedding-8B" if model == QWEN
                            else "tencent/WeMM-Embedding-9B"
                        ), "revision": "a"*40,
                    },
                }

            async def spawn(model_id, items):
                async def get():
                    return await aio(model_id, items)

                async def cancel(terminate_containers=False):
                    return None

                return types.SimpleNamespace(
                    get=types.SimpleNamespace(aio=get),
                    cancel=types.SimpleNamespace(aio=cancel),
                )

            self.encode_many = types.SimpleNamespace(
                spawn=types.SimpleNamespace(aio=spawn)
            )

    from multimodal_embeddings_ai_inference.schemas import EmbeddingRequest
    q, w = Worker(QWEN), Worker(WEMM)
    prepare_q = prepare(EmbeddingRequest.model_validate({
        "model": QWEN, "text": "test", "model_config": {
            "dimension": 64, "max_length": 4096, "normalize": False,
        },
    }), SETTINGS)
    prepare_w = prepare(EmbeddingRequest.model_validate({
        "model": WEMM, "text": "test", "model_config": {
            "dimension": 128, "normalize": False,
        },
    }), SETTINGS)
    service = InferenceService("modal", {QWEN: q, WEMM: w})
    vectors = asyncio.run(service.encode_many([prepare_q, prepare_w]))
    assert len(vectors) == 2
    assert q.inputs[0]["model_config"] == {"max_length": 4096, "normalize": False}
    assert w.inputs[0]["model_config"] == {"normalize": False}
    assert math.sqrt(sum(v*v for v in vectors[0])) > 1


@pytest.mark.parametrize("ack", ["missing", "wrong", "matching"])
def test_modal_worker_profile_ack_is_verified_for_each_batch_item(monkeypatch, ack):
    """Old/misaligned workers cannot label vectors with requested config hash."""
    from multimodal_embeddings_ai_inference.model_options import effective_encoding_profile
    from multimodal_embeddings_ai_inference.schemas import EmbeddingRequest

    monkeypatch.setenv("VISION_EMBEDDINGS_QWEN8_REVISION", "a" * 40)
    source = [
        prepare(EmbeddingRequest.model_validate({
            "model": QWEN, "text": "alpha",
            "model_config": {"dimension": 64, "max_length": 64},
        }), SETTINGS),
        prepare(EmbeddingRequest.model_validate({
            "model": QWEN, "text": "beta",
            "model_config": {"dimension": 128, "normalize": False},
        }), SETTINGS),
    ]

    async def spawn(model_id, items):
        async def get():
            result = {
                "vectors": [[1.0] + [0.0] * (item["dimension"] - 1)
                            for item in items],
                "checkpoint_identity": {
                    "model": model_id,
                    "repository": "Qwen/Qwen3-VL-Embedding-8B",
                    "revision": "a" * 40,
                },
            }
            profiles = [
                effective_encoding_profile(
                    model_id, kind=item["kind"], dimension=item["dimension"],
                    instruction=item.get("instruction"), options=item.get("model_config"),
                ) for item in items
            ]
            if ack == "wrong":
                profiles[1]["model_config"] = {}
            if ack != "missing":
                result["effective_profiles"] = profiles
            return result

        async def cancel(terminate_containers=False):
            return None

        return types.SimpleNamespace(
            get=types.SimpleNamespace(aio=get),
            cancel=types.SimpleNamespace(aio=cancel),
        )

    worker = types.SimpleNamespace(
        encode_many=types.SimpleNamespace(spawn=types.SimpleNamespace(aio=spawn))
    )
    service = InferenceService("modal", {QWEN: worker})
    if ack == "matching":
        result = asyncio.run(service.encode_many(source))
        assert [len(vector) for vector in result] == [64, 128]
    else:
        with pytest.raises(ValueError, match="effective encoding profile"):
            asyncio.run(service.encode_many(source))
