"""Shared session fixtures for the independent API demo notebook tests."""

import io
import json
import os
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.config import Settings

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/multimodal_embeddings_api_demo.ipynb"


@pytest.fixture
def notebook_connection(monkeypatch):
    """Run published cells through preflight against the real local mock API."""
    token = "offline-secure-reconnection-token"
    first_url = "https://first.example.invalid"
    second_url = "https://second.example.invalid"
    client = TestClient(create_app(Settings(api_token=token, backend="mock")))
    requests = []

    class LocalTransport:
        def open(self, request, timeout=None):
            requests.append((request.full_url, request.get_method()))
            response = client.request(
                request.get_method(), urlsplit(request.full_url).path,
                headers=dict(request.header_items()), content=request.data,
            )
            assert response.status_code == 200
            return io.BytesIO(response.content)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: LocalTransport())
    for key in tuple(os.environ):
        if key.startswith("VISION_EMBEDDINGS_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", first_url)
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", token)
    monkeypatch.setenv("VISION_EMBEDDINGS_REQUIRE_REAL", "false")
    cells = {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    }
    scope = {}
    with redirect_stdout(io.StringIO()):
        for cell_id in (
            "configure-code", "configure-helpers-code",
            "sample-options-code", "sample-preparation-code",
            "credentials-input", "http-verification-code",
        ):
            exec(compile(cells[cell_id], cell_id, "exec"), scope)
            if cell_id == "configure-code":
                scope["RUN_API_DEMO"] = True  # Offline mock session explicitly enables API mode.
    assert not any(method == "POST" for _, method in requests)
    assert all(r[0].startswith(first_url) for r in requests)
    return scope, cells, requests, token, second_url


@pytest.fixture
def verified_notebook_session(notebook_connection):
    """A completed six-request run, with no report cells executed yet."""
    scope, cells, requests, _token, _url = notebook_connection
    with redirect_stdout(io.StringIO()):
        exec(compile(cells["embedding-code"], "embedding-code", "exec"), scope)
    assert sum(method == "POST" for _, method in requests) == 6
    return notebook_connection



CORE_TESTS = {
    "test_api", "test_encoder_inference", "test_retrieval_quality",
    "test_embedding_pipeline", "test_retrieval_evidence",
    "test_routing", "test_checkpoints", "test_instructions", "test_model_options",
    "test_gpu_smoke_contract", "test_smoke_both_async", "test_smoke_instruction_compat",
}
SECURITY_TESTS = {
    "test_auth_guard", "test_distributed_quota", "test_inference_guard",
    "test_multitenant_admission", "test_security_monitoring", "test_modal_secret",
    "test_client_key_tools", "test_request_deadlines", "test_async_hardening",
    "test_async_jobs", "test_async_outcomes", "test_async_job_inspector",
    "test_readonly_recovery", "test_gpu_supervisor",
}
PUBLICATION_TESTS = {
    "test_acceptance_record", "test_client_documentation",
    "test_client_integration_redirects", "test_gpu_dependencies",
    "test_landing", "test_project_namespace", "test_sha256_manifest",
    "test_bilingual_notebook_release", "test_benchmark_provenance",
    "test_evidence_chronology",
}


NOTEBOOK_TESTS = {
    "test_benchmark_async_readiness", "test_benchmark_attempt_persistence",
    "test_benchmark_readiness_refresh",
    "test_benchmark_bilingual_diagnostics", "test_benchmark_measurements",
    "test_benchmark_preflight_auth", "test_benchmark_session_recovery",
    "test_benchmark_workload_preview", "test_cold_diagnostics",
    "test_notebook_async_resume", "test_notebook_recovery",
    "test_notebook_recovery_validation", "test_notebook_security",
    "test_notebook_template", "test_performance_notebook",
}


def pytest_collection_modifyitems(items):
    """Classify existing tests without skipping or duplicating execution."""
    for item in items:
        stem = item.path.stem
        if stem in CORE_TESTS:
            marker = "core"
        elif stem in SECURITY_TESTS:
            marker = "security"
        elif stem in PUBLICATION_TESTS:
            marker = "publication"
        elif stem in NOTEBOOK_TESTS:
            marker = "notebook"
        else:
            raise pytest.UsageError(f"Classify {stem} before adding a test module")
        item.add_marker(getattr(pytest.mark, marker))
