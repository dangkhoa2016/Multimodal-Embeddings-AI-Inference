"""Public landing-page regressions; no network or GPU is required."""

import pytest
from fastapi.testclient import TestClient

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.config import Settings
from multimodal_embeddings_ai_inference.service import InferenceService

OWNER = "test-owner-bearer-credential-1234"


@pytest.fixture
def api():
    with TestClient(create_app(Settings(api_token=OWNER, backend="mock"),
                               InferenceService("mock"))) as client:
        yield client


@pytest.mark.parametrize(
    ("path", "locale", "heading"),
    [
        ("/", "en", "Image and text embeddings,"),
        ("/?lang=en", "en", "Image and text embeddings,"),
        ("/?lang=vi", "vi", "Embedding ảnh và văn bản,"),
        ("/?lang=%3Cscript%3E", "en", "Image and text embeddings,"),
    ],
)
def test_root_is_accessible_and_bilingual_without_credentials(api, path, locale, heading):
    response = api.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert f'<html lang="{locale}">' in response.text
    assert heading in response.text
    assert "Qwen/Qwen3-VL-Embedding-8B" in response.text
    assert "tencent/WeMM-Embedding-9B" in response.text
    assert "https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference" in response.text
    assert "<script>" not in response.text
    assert OWNER not in response.text


def test_root_is_static_with_security_headers(api):
    response = api.get("/")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "GET /health" in response.text
    assert "GET /v1/models" in response.text
    assert 'href="/health"' in response.text
    assert 'href="/v1/models"' in response.text
    assert 'href="/?lang=vi"' in response.text
    assert response.text.count('href="/health"') == 2
    assert response.text.count('href="/v1/models"') == 2
    assert '<a href="/health">GET /health</a>' in response.text
    assert '<a href="/v1/models">GET /v1/models</a>' in response.text


def test_public_routes_do_not_bypass_private_api_auth_or_enable_framework_docs(api):
    assert api.get("/").status_code == 200
    assert api.get("/health").status_code == 200
    assert api.get("/v1/models").status_code == 200
    assert api.get("/v1/models/qwen3-vl-embedding-8b").status_code == 401
    assert api.get("/v1/ops/metrics").status_code == 401
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert api.get(path).status_code == 404


@pytest.mark.parametrize(
    ("locale", "health_text", "models_text"),
    [
        ("en", "API health", "Model catalog"),
        ("vi", "Trạng thái API", "Danh sách model"),
    ],
)
def test_navigation_links_are_local_useful_and_distinct_from_hero(
    api, locale, health_text, models_text,
):
    from html.parser import HTMLParser

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.inside_nav = False
            self.active_link = None
            self.links = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "nav":
                self.inside_nav = True
            if self.inside_nav and tag == "a":
                self.active_link = attrs.get("href")

        def handle_endtag(self, tag):
            if tag == "a":
                self.active_link = None
            if tag == "nav":
                self.inside_nav = False

        def handle_data(self, data):
            if self.active_link is not None:
                self.links.append((self.active_link, data.strip()))

    response = api.get("/", params={"lang": locale})
    assert response.status_code == 200
    parser = Links()
    parser.feed(response.text)
    assert ("/health", health_text) in parser.links
    assert ("/v1/models", models_text) in parser.links
    assert not any("github.com" in url for url, _ in parser.links)
    assert api.get("/health").status_code == 200
    assert api.get("/v1/models").status_code == 200
