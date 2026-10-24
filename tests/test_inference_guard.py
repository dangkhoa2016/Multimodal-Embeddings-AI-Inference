"""Offline checks for gateway-instance quota and least-privilege Modal configuration."""

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from multimodal_embeddings_ai_inference.api import _InferenceQuota, create_app
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.service import InferenceService

TOKEN = "offline-auth-token-123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ROOT = Path(__file__).resolve().parents[1]


def test_worker_and_coordinator_secrets_exclude_gateway_token():
    import ast

    roles = {"QwenL40SGpu": "model_revision_secret", "WeMML40SGpu": "model_revision_secret",
             "run_embedding_job": "model_revision_secret", "api_entrypoint": "gateway_secret"}
    found = set()
    for node in ast.parse((ROOT / "modal_app.py").read_text()).body:
        if getattr(node, "name", None) not in roles:
            continue
        found.add(node.name)
        values = [kw.value for dec in node.decorator_list if isinstance(dec, ast.Call)
                  for kw in dec.keywords if kw.arg == "secrets"]
        assert len(values) == 1 and isinstance(values[0], ast.List)
        assert len(values[0].elts) == 1 and isinstance(values[0].elts[0], ast.Name)
        assert values[0].elts[0].id == roles[node.name]
    assert found == set(roles)


def test_quota_counts_embedding_units_instead_of_http_requests():
    settings = Settings(api_token=TOKEN, backend="mock", max_embeddings_per_minute=2)
    with TestClient(create_app(settings, InferenceService("mock"))) as client:
        items = [
            {"model": "wemm-embedding-9b", "text": "one"},
            {"model": "wemm-embedding-9b", "text": "two"},
        ]
        result = client.post("/v1/embeddings/batches", headers=AUTH, json={"items": items})
        assert result.status_code == 200
        assert client.get("/v1/health", headers=AUTH).status_code == 200
        denied = client.post(
            "/v1/embeddings",
            headers=AUTH,
            json={"model": "wemm-embedding-9b", "text": "three"},
        )
        assert denied.status_code == 429
        assert int(denied.headers["Retry-After"]) >= 1
        assert "budget" in denied.json()["detail"].lower()


def test_inflight_limit_rejects_and_recovers():
    quota = _InferenceQuota(max_units_per_minute=3, max_inflight=1)
    quota.acquire(1)
    with pytest.raises(HTTPException) as exc:
        quota.acquire(1)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "1"
    quota.release()
    quota.acquire(1)
    quota.release()


def test_failed_authentication_does_not_spend_quota():
    settings = Settings(api_token=TOKEN, backend="mock", max_embeddings_per_minute=1)
    with TestClient(create_app(settings, InferenceService("mock"))) as client:
        assert client.post("/v1/embeddings", json={}).status_code == 401
        allowed = client.post(
            "/v1/embeddings",
            headers=AUTH,
            json={"model": "wemm-embedding-9b", "text": "hello"},
        )
        assert allowed.status_code == 200


def test_quota_settings_are_configurable_with_safe_minimum(monkeypatch):
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", TOKEN)
    monkeypatch.setenv("VISION_EMBEDDINGS_MAX_EMBEDDINGS_PER_MINUTE", "40")
    monkeypatch.setenv("VISION_EMBEDDINGS_MAX_INFLIGHT_REQUESTS", "2")
    config = Settings.from_env()
    assert config.max_embeddings_per_minute == 40
    assert config.max_inflight_requests == 2


def test_quota_rejects_invalid_direct_settings():
    with pytest.raises(ValueError, match="positive"):
        _InferenceQuota(max_units_per_minute=0, max_inflight=2)
    with pytest.raises(ValueError, match="positive"):
        _InferenceQuota(max_units_per_minute=10, max_inflight=0)


def test_modal_gateway_defaults_singleton_and_scaling_requires_distributed_backend():
    """Deployment defaults to one gateway; scaling is opt-in and guarded."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "modal_app.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    gateway = next(node for node in module.body
                   if isinstance(node, ast.FunctionDef) and node.name == "api_entrypoint")
    modal_function = next(dec for dec in gateway.decorator_list
                          if isinstance(dec, ast.Call)
                          and isinstance(dec.func, ast.Attribute)
                          and dec.func.attr == "function")
    count = next(kw.value for kw in modal_function.keywords
                 if kw.arg == "max_containers")
    assert isinstance(count, ast.Name) and count.id == "gateway_max_containers"
    code = source.read_text(encoding="utf-8")
    assert 'os.getenv("VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS", "1")' in code
    assert "RedisDistributedQuota" in code
    with pytest.raises(ValueError, match="Multi-gateway"):
        Settings(api_token=TOKEN, gateway_max_containers=2)
