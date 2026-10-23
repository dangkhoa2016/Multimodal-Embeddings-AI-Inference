"""CPU integration contracts for the public embedding path (fake vectors, real API)."""

import base64
import hashlib
import io
import math

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.service import InferenceService

AUTH = {"Authorization": "Bearer offline-integration-token-123456"}


@pytest.fixture
def api():
    app = create_app(
        Settings(api_token="offline-integration-token-123456", backend="mock"),
        InferenceService("mock"),
    )
    with TestClient(app) as client:
        yield client


def image_payload():
    image = Image.new("RGB", (5, 7), (48, 96, 144))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("model", ["qwen3-vl-embedding-8b", "wemm-embedding-9b"])
@pytest.mark.parametrize("kind", ["text", "image"])
def test_embedding_contract_preserves_inputs_identity_and_vector_shape(api, model, kind):
    body = {"model": model, "dimension": 64, "request_id": "integration-vector"}
    if kind == "text":
        body["text"] = "  tìm ảnh tương tự  "
        input_digest = hashlib.sha256("tìm ảnh tương tự".encode()).hexdigest()
    else:
        body["image_base64"], input_digest = image_payload()
    response = api.post("/v1/embeddings", headers=AUTH, json=body)
    assert response.status_code == 200, response.text
    output = response.json()
    assert output["model"] == model
    assert output["request_id"] == body["request_id"]
    assert output["input_sha256"] == input_digest
    assert output["backend"] == "mock"
    assert output["dimension"] == len(output["vector"]) == 64
    assert all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in output["vector"])
    assert math.sqrt(sum(value * value for value in output["vector"])) == pytest.approx(
        1.0, abs=1e-5
    )
    assert len(output["embedding_metadata"]["config_sha256"]) == 64
    assert output["timing"] is None
    duplicate = api.post("/v1/embeddings", headers=AUTH, json=body)
    assert duplicate.status_code == 200
    assert duplicate.json()["vector"] == output["vector"]


def test_mixed_batch_preserves_model_and_request_correspondence(api):
    inputs = [
        {"model": "qwen3-vl-embedding-8b", "text": "first", "request_id": "one"},
        {"model": "wemm-embedding-9b", "text": "second", "request_id": "two"},
        {"model": "qwen3-vl-embedding-8b", "text": "third", "request_id": "three"},
    ]
    reply = api.post("/v1/embeddings/batches", headers=AUTH, json={"items": inputs})
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["count"] == 3
    assert [(i["model"], i["request_id"]) for i in body["items"]] == [
        (i["model"], i["request_id"]) for i in inputs
    ]
    for item, sent in zip(body["items"], inputs, strict=True):
        assert item["input_sha256"] == hashlib.sha256(sent["text"].encode()).hexdigest()


@pytest.mark.parametrize("value", ["", "   ", "\u0000", "\u0007"])
def test_invalid_text_never_returns_embeddings(api, value):
    response = api.post(
        "/v1/embeddings", headers=AUTH,
        json={"model": "qwen3-vl-embedding-8b", "text": value},
    )
    assert response.status_code == 422
    assert "vector" not in response.text
