"""Offline security regression tests for the CPU authentication gateway."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from multimodal_embeddings_ai_inference.api import create_app
from multimodal_embeddings_ai_inference.auth_guard import (
    AuthenticationFailureGuard,
    client_identity,
)
from multimodal_embeddings_ai_inference.config import Settings

TOKEN = "offline-strong-token-value-0123456789"
CORRECT = {"Authorization": f"Bearer {TOKEN}"}
WRONG = {"Authorization": "Bearer guessed-wrong-token"}


def build_client(ip="203.0.113.10", **settings):
    return TestClient(
        create_app(Settings(api_token=TOKEN, backend="mock", **settings)),
        client=(ip, 50505),
    )


def test_three_failures_lock_peer_then_return_retry_after():
    with build_client() as client:
        for _ in range(2):
            response = client.get("/v1/health", headers=WRONG)
            assert response.status_code == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"
            assert response.headers["Cache-Control"] == "no-store"
        third = client.get("/v1/health", headers=WRONG)
        assert third.status_code == 429
        assert third.json()["detail"] == "Too many failed authentication attempts"
        assert third.headers["Retry-After"] == "900"
        fourth = client.post("/v1/embeddings", headers=WRONG, json={})
        assert fourth.status_code == 429
        assert int(fourth.headers["Retry-After"]) >= 1


def test_lockout_skips_credential_checks_and_expires_without_extending(monkeypatch):
    from multimodal_embeddings_ai_inference import api

    clock = [100.0]
    guard = AuthenticationFailureGuard(clock=lambda: clock[0])
    monkeypatch.setattr(api, "AuthenticationFailureGuard", lambda **kwargs: guard)
    compare = api.hmac.compare_digest
    comparisons = []

    def counted(left, right):
        comparisons.append(1)
        return compare(left, right)

    monkeypatch.setattr(api.hmac, "compare_digest", counted)
    with build_client() as client:
        assert [client.get("/v1/health", headers=WRONG).status_code for _ in range(3)] == [
            401, 401, 429,
        ]
        assert len(comparisons) == 3
        clock[0] += 100
        for headers in (WRONG, CORRECT):
            response = client.get("/v1/health", headers=headers)
            assert response.status_code == 429 and response.headers["Retry-After"] == "800"
        assert len(comparisons) == 3
        clock[0] += 800
        assert client.get("/v1/health", headers=CORRECT).status_code == 200
        assert len(comparisons) == 4
        assert client.get("/v1/models", headers=WRONG).status_code == 401


def test_attempts_share_state_across_protected_routes():
    with build_client() as client:
        assert client.get("/v1/health").status_code == 401
        assert client.get("/v1/models", headers=WRONG).status_code == 401
        assert client.post("/v1/embeddings", json={}).status_code == 429
        assert client.get("/v1/models/wemm-embedding-9b", headers=WRONG).status_code == 429


def test_distinct_asgi_peer_addresses_have_independent_counters():
    app = create_app(Settings(api_token=TOKEN, backend="mock"))
    with TestClient(app, client=("203.0.113.10", 1101)) as alice:
        with TestClient(app, client=("198.51.100.12", 2202)) as bob:
            assert [alice.get("/v1/health", headers=WRONG).status_code for _ in range(3)] == [
                401, 401, 429
            ]
            assert bob.get("/v1/health", headers=WRONG).status_code == 401
            assert bob.get("/v1/health", headers=CORRECT).status_code == 200
            assert alice.get("/v1/health", headers=WRONG).status_code == 429


def test_spoofed_forwarded_headers_do_not_change_authentication_identity():
    app = create_app(Settings(api_token=TOKEN, backend="mock"))
    with TestClient(app, client=("203.0.113.22", 5050)) as client:
        for i in range(3):
            response = client.get(
                "/v1/health",
                headers={**WRONG, "X-Forwarded-For": f"198.51.100.{i + 1}"},
            )
        assert response.status_code == 429
        assert client.get(
            "/v1/health",
            headers={**WRONG, "X-Forwarded-For": "192.0.2.45"},
        ).status_code == 429


def test_authentication_precedes_large_or_malformed_body_buffering():
    app = create_app(Settings(api_token=TOKEN, backend="mock", max_image_bytes=1))
    with TestClient(app) as client:
        reply = client.post(
            "/v1/embeddings",
            headers={**WRONG, "Content-Length": "99999999"},
            content=b"not-json",
        )
        assert reply.status_code == 401
        assert client.get("/v1/health", headers=CORRECT).status_code == 200


def test_unknown_non_api_endpoints_remain_unavailable_without_token():
    with build_client() as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_failure_window_expires_and_lockout_recovers():
    clock = [100.0]
    guard = AuthenticationFailureGuard(
        max_failures=3, window_seconds=60, lockout_seconds=120,
        clock=lambda: clock[0],
    )
    peer = "203.0.113.7"
    assert guard.record_failure(peer) is None
    assert guard.record_failure(peer) is None
    clock[0] += 61
    assert guard.record_failure(peer) is None
    assert guard.record_failure(peer) is None
    assert guard.record_failure(peer) == 120
    clock[0] += 10
    assert guard.record_failure(peer) == 110
    clock[0] += 110
    assert guard.record_failure(peer) is None


def test_guard_counter_is_thread_safe_and_bounded():
    guard = AuthenticationFailureGuard(max_failures=3, max_clients=2)
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(guard.record_failure, ["192.0.2.10"] * 40))
    assert results.count(None) == 2
    assert len(results) - results.count(None) == 38
    guard.record_failure("192.0.2.11")
    guard.record_failure("192.0.2.12")
    assert len(guard._clients) == 2


def test_bad_configuration_rejected(monkeypatch):
    for key, field in (
        ("VISION_EMBEDDINGS_AUTH_FAILURE_LIMIT", "auth_failure_limit"),
        ("VISION_EMBEDDINGS_AUTH_FAILURE_WINDOW_SECONDS", "auth_failure_window_seconds"),
        ("VISION_EMBEDDINGS_AUTH_LOCKOUT_SECONDS", "auth_lockout_seconds"),
        ("VISION_EMBEDDINGS_AUTH_MAX_TRACKED_CLIENTS", "auth_max_tracked_clients"),
    ):
        with pytest.raises(ValueError, match=key):
            Settings(api_token=TOKEN, **{field: 0})
        monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", TOKEN)
        monkeypatch.setenv(key, "0")
        with pytest.raises(ValueError, match=key):
            Settings.from_env()
        monkeypatch.delenv(key)


def test_peer_resolver_ignores_raw_forwarded_header():
    from starlette.requests import Request
    scope = {
        "type": "http", "method": "GET", "path": "/v1/health",
        "query_string": b"", "server": ("example.com", 80),
        "client": ("198.51.100.42", 1000),
        "headers": [(b"x-forwarded-for", b"203.0.113.50")],
    }
    assert client_identity(Request(scope)) == "198.51.100.42"


def test_mounted_apps_enforce_their_own_credentials():
    parent_token = "offline-parent-token-123456789"
    child_token = "offline-child-token-987654321"
    parent = create_app(Settings(api_token=parent_token, backend="mock"))
    parent.mount("/v1/private", create_app(Settings(api_token=child_token, backend="mock")))
    with TestClient(parent) as client:
        path = "/v1/private/v1/health"
        parent_headers = {"Authorization": "Bearer " + parent_token}
        child_headers = {"Authorization": "Bearer " + child_token}
        assert client.get(path, headers=parent_headers).status_code == 401
        assert client.get(path, headers=child_headers).status_code == 200
        reply = client.post("/v1/private/v1/embeddings", headers=child_headers, json={
            "model": "wemm-embedding-9b", "text": "public fixture", "dimension": 64,
        })
        assert reply.status_code == 200 and len(reply.json()["vector"]) == 64


@pytest.mark.parametrize("topology", ["mount", "root_path", "nested_api", "included_mount"])
def test_prefixed_unauthorized_request_never_reads_body(topology):
    import asyncio

    from fastapi import FastAPI

    inner = create_app(Settings(api_token=TOKEN, backend="mock"))
    if topology == "mount":
        app = FastAPI()
        app.mount("/gateway", inner)
        root_path, path = "", "/gateway/v1/embeddings"
    elif topology in {"nested_api", "included_mount"}:
        app = create_app(Settings(api_token="offline-parent-token-123456789", backend="mock"))
        if topology == "included_mount":
            from fastapi import APIRouter

            router = APIRouter()
            router.mount("/private", inner)
            app.include_router(router, prefix="/v1")
        else:
            app.mount("/v1/private", inner)
        root_path, path = "", "/v1/private/v1/embeddings"
    else:
        app = inner
        root_path, path = "/gateway", "/gateway/v1/embeddings"
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path,
        "root_path": root_path, "query_string": b"", "server": ("example.test", 80),
        "client": ("203.0.113.10", 5050),
        "headers": [(b"content-type", b"application/json")],
    }
    replies = []

    async def receive():
        pytest.fail("Unauthorized prefixed requests must not consume the body")

    async def send(message):
        replies.append(message)

    asyncio.run(app(scope, receive, send))
    response = next(reply for reply in replies if reply["type"] == "http.response.start")
    assert response["status"] == 401


def test_parent_body_limit_does_not_override_mounted_app_limit():
    parent = create_app(Settings(api_token=TOKEN, backend="mock", max_image_bytes=1))
    child_token = "offline-child-token-987654321"
    child = create_app(Settings(api_token=child_token, backend="mock",
                                max_image_bytes=2 * 1024 * 1024, max_batch_size=1))
    parent.mount("/v1/private", child)
    with TestClient(parent) as client:
        response = client.post(
            "/v1/private/v1/embeddings", content=b"invalid-json" * 100000,
            headers={"Authorization": "Bearer " + child_token, "Content-Type": "application/json"},
        )
    assert response.status_code == 422


@pytest.mark.parametrize("token", ["x" * 4090, "x" * 15, "x" * 8 + "\n" + "x" * 8, "x" * 16 + "é"])
def test_invalid_server_token_configuration_is_rejected_at_startup(monkeypatch, token):
    with pytest.raises(ValueError, match="VISION_EMBEDDINGS_API_TOKEN"):
        Settings(api_token=token)
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", token)
    with pytest.raises(ValueError, match="VISION_EMBEDDINGS_API_TOKEN"):
        Settings.from_env()


def test_longest_supported_server_token_can_authenticate():
    token = "x" * 4089
    with TestClient(create_app(Settings(api_token=token, backend="mock"))) as client:
        reply = client.get("/v1/health", headers={"Authorization": "Bearer " + token})
        assert reply.status_code == 200


def test_concurrent_guard_admission_checks_only_the_first_three_candidates():
    guard = AuthenticationFailureGuard(clock=lambda: 100.0)
    checked = []

    def incorrect():
        checked.append(1)
        return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: guard.authenticate("203.0.113.10", incorrect), range(40)))
    assert len(checked) == 3
    assert results.count((False, None)) == 2
    assert results.count((False, 900)) == 38


def test_added_non_api_routes_keep_their_public_behavior():
    app = create_app(Settings(api_token=TOKEN, backend="mock"))

    @app.get("/status")
    def status():
        return {"status": "public"}

    with TestClient(app) as client:
        assert client.get("/status").status_code == 200


def test_included_api_router_inherits_authentication_and_body_limits():
    from fastapi import APIRouter, Request

    app = create_app(Settings(api_token=TOKEN, backend="mock", max_image_bytes=1))
    router = APIRouter()

    @router.post("/extension")
    async def extension(request: Request):
        return {"size": len(await request.body())}

    app.include_router(router, prefix="/v1")
    with TestClient(app) as client:
        assert client.post("/v1/extension", content=b"public").status_code == 401
        reply = client.post("/v1/extension", content=b"public", headers=CORRECT)
        assert reply.status_code == 200 and reply.json() == {"size": 6}
        oversized = client.post("/v1/extension", content=b"x" * (1024 * 1024 + 1), headers=CORRECT)
        assert oversized.status_code == 413


@pytest.mark.parametrize("depth", [1, 2])
def test_apps_mounted_through_included_routers_keep_their_credentials(depth):
    from fastapi import APIRouter

    parent_token = "offline-parent-token-123456789"
    child_token = "offline-child-token-987654321"
    parent = create_app(Settings(api_token=parent_token, backend="mock"))
    router = APIRouter()
    router.mount("/private", create_app(Settings(api_token=child_token, backend="mock")))
    prefix = "/v1"
    if depth == 2:
        group = APIRouter()
        group.include_router(router, prefix="/group")
        router = group
        prefix += "/group"
    parent.include_router(router, prefix="/v1")
    with TestClient(parent) as client:
        path = prefix + "/private/v1/health"
        wrong = client.get(path, headers={"Authorization": "Bearer " + parent_token})
        assert wrong.status_code == 401
        correct = client.get(path, headers={"Authorization": "Bearer " + child_token})
        assert correct.status_code == 200
