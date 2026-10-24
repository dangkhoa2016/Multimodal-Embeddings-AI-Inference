"""Instruction-aware API regression: defaults, validation, provenance and routing."""

import asyncio
import hashlib
import sys
import types

import pytest
from fastapi.testclient import TestClient

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.embedding_config import (
    QWEN_LEGACY_INSTRUCTION,
)
from multimodal_embeddings_ai_inference.model_options import effective_encoding_profile
from multimodal_embeddings_ai_inference.service import InferenceService
from multimodal_embeddings_ai_inference.validation import Prepared

TOKEN = "test-only-instruction-token-with-enough-length"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
QWEN = "qwen3-vl-embedding-8b"
WEMM = "wemm-embedding-9b"


@pytest.fixture
def client():
    with TestClient(create_app(Settings(api_token=TOKEN, backend="mock"))) as app:
        yield app


def request(client, model=QWEN, **extra):
    return client.post(
        "/v1/embeddings",
        headers=AUTH,
        json={"model": model, "text": "How do I reset my password?", "dimension": 64, **extra},
    )


def test_legacy_omission_matches_explicit_original_instruction(client):
    implicit = request(client)
    explicit = request(client, instruction=QWEN_LEGACY_INSTRUCTION)
    assert implicit.status_code == explicit.status_code == 200
    a, b = implicit.json(), explicit.json()
    assert a["vector"] == b["vector"]
    assert a["input_sha256"] == b["input_sha256"]
    assert a["embedding_metadata"] == b["embedding_metadata"]
    assert a["embedding_metadata"]["preprocessing_version"] == "qwen3-vl-adapter-v1"
    assert a["embedding_metadata"]["instruction_sha256"] == hashlib.sha256(
        QWEN_LEGACY_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    assert len(a["embedding_metadata"]["config_sha256"]) == 64


def test_custom_instruction_changes_config_and_keeps_raw_input_hash(client):
    legacy = request(client).json()
    instruction = "Retrieve troubleshooting documents that answer the user's question."
    custom = request(client, instruction=instruction)
    assert custom.status_code == 200, custom.text
    response = custom.json()
    assert response["input_sha256"] == legacy["input_sha256"]
    assert response["embedding_metadata"]["config_sha256"] != (
        legacy["embedding_metadata"]["config_sha256"]
    )
    assert response["embedding_metadata"]["instruction_sha256"] == hashlib.sha256(
        instruction.encode("utf-8")
    ).hexdigest()
    assert response["vector"] != legacy["vector"]  # Synthetic mock only, not an accuracy claim
    assert instruction not in custom.text


def test_instruction_hash_is_stable_across_requests(client):
    values = [
        request(client, instruction="Retrieve relevant documents.").json()["embedding_metadata"]
        for _ in range(2)
    ]
    assert values[0] == values[1]


@pytest.mark.parametrize("instruction", ["", "   ", "\n", "\t", "hi\nthere", "x" * 513])
def test_reject_empty_control_and_oversized_instruction(client, instruction):
    assert request(client, instruction=instruction).status_code == 422


def test_wemm_explicit_instruction_is_rejected_not_ignored(client):
    response = request(client, model=WEMM, instruction="Search for relevant images.")
    assert response.status_code == 422
    assert "Custom instruction" in response.text


def test_wemm_default_is_unchanged_and_has_no_instruction_hash(client):
    response = request(client, model=WEMM)
    assert response.status_code == 200
    metadata = response.json()["embedding_metadata"]
    assert metadata["instruction_sha256"] is None
    assert metadata["preprocessing_version"] == "wemm-adapter-v1"
    assert len(metadata["config_sha256"]) == 64


def test_model_catalog_declares_capability(client):
    assert client.get(f"/v1/models/{QWEN}", headers=AUTH).json()[
        "supports_custom_instruction"
    ] is True
    assert client.get(f"/v1/models/{WEMM}", headers=AUTH).json()[
        "supports_custom_instruction"
    ] is False


def test_mixed_batch_retains_per_item_policy_and_order(client):
    items = [
        {"model": QWEN, "text": "alpha", "instruction": "Retrieve similar documents."},
        {"model": WEMM, "text": "beta"},
        {"model": QWEN, "text": "gamma"},
    ]
    result = client.post("/v1/embeddings/batches", headers=AUTH, json={"items": items})
    assert result.status_code == 200, result.text
    out = result.json()
    assert out["count"] == 3
    assert [it["model"] for it in out["items"]] == [QWEN, WEMM, QWEN]
    assert out["items"][0]["embedding_metadata"]["instruction_sha256"] != (
        out["items"][2]["embedding_metadata"]["instruction_sha256"]
    )
    items[1]["instruction"] = "unsupported"
    rejected = client.post("/v1/embeddings/batches", headers=AUTH, json={"items": items})
    assert rejected.status_code == 422


def test_modal_route_only_transports_explicit_instruction(monkeypatch):
    for key in ("WEMM9", "QWEN8"):
        monkeypatch.setenv(f"VISION_EMBEDDINGS_{key}_REVISION", "a" * 40)
    class Worker:
        def __init__(self, model):
            self.received = []
            outer = self

            class Remote:
                async def aio(self, model_id, items):
                    assert model_id == model
                    outer.received.append(items)
                    repository = ("Qwen/Qwen3-VL-Embedding-8B" if model == QWEN
                                  else "tencent/WeMM-Embedding-9B")
                    return {
                            "vectors": [[1.0] * item["dimension"] for item in items],
                            "effective_profiles": [
                                effective_encoding_profile(
                                    model_id, kind=item["kind"], dimension=item["dimension"],
                                    instruction=item.get("instruction"),
                                    options=item.get("model_config"),
                                ) for item in items
                            ],
                            "checkpoint_identity": {
                                "model": model, "repository": repository, "revision": "a" * 40,
                            }}

            class Spawn:
                async def aio(self, model_id, items):
                    async def get():
                        return await Remote().aio(model_id, items)

                    async def cancel(terminate_containers=False):
                        pass

                    return types.SimpleNamespace(get=types.SimpleNamespace(aio=get),
                                                 cancel=types.SimpleNamespace(aio=cancel))

            self.encode_many = types.SimpleNamespace(spawn=Spawn())

    qwen, wemm = Worker(QWEN), Worker(WEMM)
    service = InferenceService("modal", {QWEN: qwen, WEMM: wemm})
    inputs = [
        Prepared(QWEN, "hi", "text", 64, "a", None, "Retrieve articles."),
        Prepared(QWEN, "hello", "text", 64, "b", None),
        Prepared(WEMM, "ok", "text", 64, "c", None),
    ]
    vectors = asyncio.run(service.encode_many(inputs))
    assert len(vectors) == 3
    assert qwen.received[0][0]["instruction"] == "Retrieve articles."
    assert "instruction" not in qwen.received[0][1]
    assert "instruction" not in wemm.received[0][0]


def test_qwen_adapter_uses_exact_legacy_or_custom_instruction(monkeypatch):
    from multimodal_embeddings_ai_inference.engines import qwen as adapter

    observed = []

    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def process(self, items):
            observed.extend(items)
            return [[1.0] + [0.0] * 63]

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    mock_code = types.ModuleType("src.models.qwen3_vl_embedding")
    mock_code.Qwen3VLEmbedder = FakeModel
    pkg = types.ModuleType("src")
    pkg.__path__ = []
    models = types.ModuleType("src.models")
    models.__path__ = []
    for name, module in (
        ("torch", fake_torch),
        ("src", pkg),
        ("src.models", models),
        ("src.models.qwen3_vl_embedding", mock_code),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(adapter, "require_checkpoint_info", lambda _: {
        "path": "/stub/model", "repository": "Qwen/Qwen3-VL-Embedding-8B", "commit_sha": "a" * 40,
    })
    encoder = adapter.QwenVLEncoder()
    encoder.encode("document", "text", 64)
    encoder.encode("question", "text", 64, instruction="Retrieve relevant answers.")
    assert observed[0]["instruction"] == QWEN_LEGACY_INSTRUCTION
    assert observed[1]["instruction"] == "Retrieve relevant answers."


def test_gpu_import_without_pydantic():
    """GPU model venv intentionally does not install FastAPI or Pydantic."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = """
import builtins
original = builtins.__import__
def reject(name, *args, **kwargs):
    if (name == 'pydantic' or name.startswith('pydantic.')
        or name == 'fastapi' or name.startswith('fastapi.')):
        raise ModuleNotFoundError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = reject
from multimodal_embeddings_ai_inference.embedding_config import QWEN_LEGACY_INSTRUCTION
assert QWEN_LEGACY_INSTRUCTION == 'Represent the image or text for retrieval in a visual gallery.'
"""
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env, capture_output=True, text=True,
        timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
