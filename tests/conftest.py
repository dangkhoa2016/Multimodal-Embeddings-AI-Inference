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

