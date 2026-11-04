import base64
import io
import math

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from compactfaces_ai_inference.api import create_app
from compactfaces_ai_inference.config import Settings
from compactfaces_ai_inference.service import InferenceService
from compactfaces_ai_inference.validation import normalize_vector

TOKEN = "test-secret-at-least-16-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client():
    settings = Settings(api_token=TOKEN, backend="mock")
    with TestClient(create_app(settings, InferenceService("mock"))) as api:
        yield api


def sample_image():
    img = Image.new("RGB", (16, 16), (48, 70, 92))
    output = io.BytesIO()
    img.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


def test_auth_protects_all_routes(client):
    for path in ["/v1/health", "/v1/models", "/v1/models/wemm-embedding-9b"]:
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer invalid"}).status_code == 401
    assert client.post("/v1/embeddings", json={}).status_code == 401


def test_health_explicitly_says_mock(client):
    result = client.get("/v1/health", headers=AUTH)
    assert result.status_code == 200
    assert result.json()["backend"] == "mock"
    assert "does not prove" in result.json()["note"]


def test_model_catalog_has_no_fake_qwen4b(client):
    items = client.get("/v1/models", headers=AUTH).json()["models"]
    ids = {x["id"] for x in items}
    assert ids == {"wemm-embedding-9b", "qwen3-vl-embedding-8b"}
    assert client.get("/v1/models/qwen3-vl-embedding-4b", headers=AUTH).status_code == 404
    assert client.get("/v1/models/wemm-embedding-8b", headers=AUTH).status_code == 404


def test_image_embedding_contract_and_norm(client):
    request = {
        "model": "wemm-embedding-9b",
        "image_base64": sample_image(),
        "dimension": 128,
        "request_id": "CORE:0002:P1",
    }
    resp = client.post("/v1/embeddings", headers=AUTH, json=request)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["request_id"] == "CORE:0002:P1"
    assert data["repository"] == "tencent/WeMM-Embedding-9B"
    assert data["backend"] == "mock"
    assert data["dimension"] == len(data["vector"]) == 128
    assert len(data["input_sha256"]) == 64
    assert abs(math.sqrt(sum(x * x for x in data["vector"])) - 1) < 1e-5
    assert (
        client.post("/v1/embeddings", headers=AUTH, json=request).json()["vector"] == data["vector"]
    )


def test_text_and_qwen_dimension(client):
    resp = client.post(
        "/v1/embeddings",
        headers=AUTH,
        json={"model": "qwen3-vl-embedding-8b", "text": "a face smiling", "dimension": 256},
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["vector"]) == 256


def test_batches_are_ordered(client):
    items = [
        {"model": "wemm-embedding-9b", "text": "one", "dimension": 64, "request_id": "a"},
        {"model": "qwen3-vl-embedding-8b", "text": "two", "dimension": 64, "request_id": "b"},
    ]
    r = client.post("/v1/embeddings/batches", headers=AUTH, json={"items": items})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2
    assert [it["request_id"] for it in r.json()["items"]] == ["a", "b"]


@pytest.mark.parametrize(
    "body",
    [
        {"model": "qwen3-vl-embedding-4b", "text": "hello"},
        {"model": "wemm-embedding-9b", "text": "hello", "dimension": 2000},
        {"model": "wemm-embedding-9b", "text": "hello", "dimension": 5000},
        {"model": "wemm-embedding-9b", "image_base64": "@@NOT-BASE64@@"},
        {"model": "wemm-embedding-9b", "image_base64": "data:image/png;base64,abc"},
        {"model": "wemm-embedding-9b", "image_base64": base64.b64encode(b"not-image").decode()},
        {"model": "wemm-embedding-9b", "text": "hello", "image_base64": "abc"},
        {"model": "wemm-embedding-9b"},
    ],
)
def test_reject_bad_inputs(client, body):
    response = client.post("/v1/embeddings", headers=AUTH, json=body)
    assert response.status_code in {413, 422}


def test_batch_limit(client):
    items = [{"model": "wemm-embedding-9b", "text": f"t{i}"} for i in range(9)]
    assert (
        client.post("/v1/embeddings/batches", headers=AUTH, json={"items": items}).status_code
        == 422
    )


def test_image_pixels_limit():
    settings = Settings(api_token=TOKEN, backend="mock", max_image_pixels=10)
    with TestClient(create_app(settings, InferenceService("mock"))) as client:
        response = client.post(
            "/v1/embeddings",
            headers=AUTH,
            json={"model": "wemm-embedding-9b", "image_base64": sample_image()},
        )
        assert response.status_code == 422


def test_vector_failure_cases():
    for values in ([0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], [1.0]):
        with pytest.raises(ValueError):
            normalize_vector(values, 2)


def test_missing_production_token_is_rejected(monkeypatch):
    monkeypatch.delenv("CF_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="CF_API_TOKEN"):
        Settings.from_env()


def test_different_mock_model_spaces(client):
    a = client.post(
        "/v1/embeddings",
        headers=AUTH,
        json={"model": "wemm-embedding-9b", "text": "face", "dimension": 64},
    ).json()["vector"]
    b = client.post(
        "/v1/embeddings",
        headers=AUTH,
        json={"model": "qwen3-vl-embedding-8b", "text": "face", "dimension": 64},
    ).json()["vector"]
    assert a != b


def test_model_route_never_loads_gpu_for_listing(client):
    r = client.get("/v1/models/wemm-embedding-9b", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["runtime_status"] == "not_checked"


def test_reject_file_url_injection(client):
    r = client.post(
        "/v1/embeddings",
        headers=AUTH,
        json={
            "model": "wemm-embedding-9b",
            "image_url": "http://169.254.169.254/latest/",
            "text": "hello",
        },
    )
    assert r.status_code == 422
