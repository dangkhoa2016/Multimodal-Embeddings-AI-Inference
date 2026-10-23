"""Framework-independent REST API with Bearer authentication for inference."""

import asyncio
import hmac as hmac  # Compatibility: external auth regressions patch api.hmac.
import logging
import threading
import time
from collections import deque

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.routing import Match, Mount, get_route_path

from . import __version__
from .async_jobs import AsyncEmbeddingJobs
from .auth_guard import AuthenticationFailureGuard, client_identity
from .catalog import CATALOG, public_spec
from .client_keys import identify_bearer
from .config import Settings
from .distributed_quota import RedisDistributedQuota
from .embedding_config import embedding_metadata
from .landing import render_landing
from .schemas import BatchRequest, BatchResponse, EmbeddingRequest, EmbeddingResponse
from .security_monitor import GatewaySecurityMonitor, route_label
from .service import InferenceService, model_revision
from .validation import prepare

logger = logging.getLogger(__name__)



class _InferenceQuota:
    """Best-effort per-gateway-instance GPU request guard; not a global quota."""

    def __init__(self, max_units_per_minute: int, max_inflight: int,
                 max_client_units_per_minute: int | None = None) -> None:
        if max_units_per_minute < 1 or max_inflight < 1:
            raise ValueError("Inference quota limits must be positive")
        self.max_units_per_minute = max_units_per_minute
        self.max_inflight = max_inflight
        self.max_client_units_per_minute = max_client_units_per_minute or max_units_per_minute
        self._client_recent: dict[str, deque[tuple[float, int]]] = {}
        self._lock = threading.Lock()
        self._recent: deque[tuple[float, int]] = deque()
        self._used = 0
        self._inflight = 0

    def acquire(self, units: int, client_id: str = "owner") -> None:
        now = time.monotonic()
        with self._lock:
            while self._recent and now - self._recent[0][0] >= 60:
                _, previous = self._recent.popleft()
                self._used -= previous
            if self._inflight >= self.max_inflight:
                raise HTTPException(
                    status_code=429,
                    detail="Inference concurrency limit reached",
                    headers={"Retry-After": "1"},
                )
            if self._used + units > self.max_units_per_minute:
                wait = (
                    max(1, int(61 - (now - self._recent[0][0])))
                    if self._recent else 60
                )
                raise HTTPException(
                    status_code=429,
                    detail="Inference request budget exceeded",
                    headers={"Retry-After": str(wait)},
                )
            if client_id != "owner":
                recent = self._client_recent.setdefault(client_id, deque())
                while recent and now - recent[0][0] >= 60:
                    recent.popleft()
                if units + sum(amount for _, amount in recent) > self.max_client_units_per_minute:
                    raise HTTPException(
                        status_code=429, detail="Client embedding rate limit reached",
                        headers={"Retry-After": "60"},
                    )
                recent.append((now, units))
            self._recent.append((now, units))
            self._used += units
            self._inflight += 1

    def release(self, client_id: str = "owner") -> None:
        with self._lock:
            self._inflight -= 1


def create_app(
    settings: Settings | None = None, service: InferenceService | None = None,
    *, async_jobs: AsyncEmbeddingJobs | None = None,
    distributed_quota: RedisDistributedQuota | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or InferenceService(settings.backend)
    if settings.distributed_rate_enabled and distributed_quota is None:
        raise ValueError("Distributed admission is enabled but its backend is not configured")
    body_slots = asyncio.BoundedSemaphore(settings.max_body_requests)
    quota = _InferenceQuota(
        settings.max_embeddings_per_minute, settings.max_inflight_requests,
        settings.max_embeddings_per_client_per_minute,
    )
    security_monitor = GatewaySecurityMonitor()
    auth_guard = AuthenticationFailureGuard(
        max_failures=settings.auth_failure_limit,
        window_seconds=settings.auth_failure_window_seconds,
        lockout_seconds=settings.auth_lockout_seconds,
        max_clients=settings.auth_max_tracked_clients,
    )
    app = FastAPI(
        title="Multimodal Embeddings AI Inference API",
        version=__version__,
        description="Authenticated image/text embeddings for research, not biometric verification.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def public_landing(lang: str = "en") -> HTMLResponse:
        """Public, static introduction; never creates a GPU inference call."""
        return HTMLResponse(
            render_landing(lang),
            headers={
                "Cache-Control": "public, max-age=300",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'unsafe-inline'; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
                ),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _private_request_validation(request: Request, error: RequestValidationError):
        # FastAPI/Pydantic may echo attacker-controlled input in error["input"]
        # or error["ctx"]; never return or log those fields.
        details = []
        allowed = {"model", "text", "image_base64", "dimension", "instruction",
                   "model_config",
                   "request_id", "items", "body"}
        for entry in error.errors()[:8]:
            fields = [
                str(part) for part in entry.get("loc", ())
                if isinstance(part, str) and part in allowed
            ]
            details.append({"loc": fields, "type": "invalid"})
        return JSONResponse(
            status_code=422,
            content={"detail": details or [{"type": "invalid"}]},
            headers={"Cache-Control": "no-store"},
        )

    def _authenticate_request(request: Request) -> None:
        if (request.scope.get("_embedding_bearer_verified") is auth_guard
                and "_embedding_client_id" in request.scope):
            return
        authorization = request.headers.get("Authorization")
        prefix = "Bearer "
        identified: str | None = None

        def valid_credentials() -> bool:
            nonlocal identified
            if (authorization is None or not authorization.startswith(prefix)
                    or len(authorization) > 4096):
                return False
            identified = identify_bearer(
                authorization[len(prefix):],
                owner_token=settings.api_token,
                client_key_hashes=settings.client_key_hashes,
            )
            return identified is not None

        valid, retry_after = auth_guard.authenticate(client_identity(request), valid_credentials)
        if valid:
            request.scope["_embedding_bearer_verified"] = auth_guard
            request.scope["_embedding_client_id"] = identified
            return

        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many failed authentication attempts",
                headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
            )
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
        )

    async def _read_checked_body(request: Request) -> None:
        # JSON/base64 overhead; decoded image limits are enforced in prepare().
        limit = max(1024 * 1024, settings.max_image_bytes * settings.max_batch_size * 2)
        try:
            n = int(request.headers.get("content-length", "0"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from None
        if n > limit:
            raise HTTPException(status_code=413, detail="Request body too large")
        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                raise HTTPException(status_code=413, detail="Request body too large")
            chunks.append(chunk)
        request._body = b"".join(chunks)

    def _routing_candidates(routes):
        for route in routes:
            # Newer FastAPI versions retain include_router branches lazily;
            # their effective contexts preserve the actual prefixed match order.
            expand = getattr(route, "effective_candidates", None)
            if callable(expand):
                yield from _routing_candidates(expand())
            else:
                yield route

    @app.middleware("http")
    async def protect_owned_api_routes(request: Request, call_next):
        # Mirror the router's first-full-match/first-partial-match selection.
        # Match selection identifies mounted apps; the router-relative path
        # also handles lazy include_router wrappers without a public path field.
        matched = partial = None
        for route in _routing_candidates(app.router.routes):
            match, _ = route.matches(request.scope)
            if match is Match.FULL:
                matched = route
                break
            if match is Match.PARTIAL and partial is None:
                partial = route
        if matched is None:
            matched = partial
        owner = getattr(matched, "original_route", matched)
        if (owner is None or isinstance(owner, Mount)
                or not get_route_path(request.scope).startswith("/v1/")
                or (request.method == "GET"
                    and get_route_path(request.scope) == "/v1/models")):
            # Mounted applications own their credentials and body limits.
            return await call_next(request)
        label = route_label(get_route_path(request.scope))
        try:
            _authenticate_request(request)
            # Limit simultaneous authenticated POST body buffering / decoding,
            # independently of GPU inflight. GET health/status remains responsive.
            hold_body = request.method in ("POST", "PUT", "PATCH")
            if hold_body:
                if body_slots.locked():
                    raise HTTPException(
                        status_code=429, detail="Gateway request body capacity reached",
                        headers={"Retry-After": "1", "Cache-Control": "no-store"},
                    )
                await body_slots.acquire()
            try:
                await _read_checked_body(request)
                response = await call_next(request)
                security_monitor.http(label, response.status_code)
                return response
            finally:
                if hold_body:
                    body_slots.release()
        except HTTPException as error:
            security_monitor.http(
                label, error.status_code,
                detail=error.detail if isinstance(error.detail, str) else None,
            )
            return JSONResponse(
                status_code=error.status_code, content={"detail": error.detail},
                headers=error.headers,
            )

    def authenticate(request: Request) -> None:
        # Defence in depth: routes still declare an authentication dependency.
        _authenticate_request(request)

    protected = [Depends(authenticate)]

    @app.get("/v1/ops/metrics", dependencies=protected)
    def operator_metrics(request: Request):
        # A valid non-owner client learns nothing about operational telemetry.
        if request.scope.get("_embedding_client_id") != "owner":
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(
            content=security_monitor.snapshot(),
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "backend": settings.backend,
            "note": "API liveness only; this does not prove that models are loaded or correct",
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "models": [
                {
                    **public_spec(spec),
                    "revision": model_revision(spec.id),
                    "runtime_status": "not_checked",
                }
                for spec in CATALOG.values()
            ],
        }

    @app.get("/v1/models/{model_id}", dependencies=protected)
    def model_info(model_id: str):
        spec = CATALOG.get(model_id)
        if spec is None:
            raise HTTPException(status_code=404, detail="Unknown model")
        return {
            **public_spec(spec),
            "revision": model_revision(spec.id),
            "runtime_status": "not_checked",
        }

    async def predict(
        items: list[EmbeddingRequest], client_id: str = "owner"
    ) -> list[EmbeddingResponse]:
        prepared = [prepare(item, settings) for item in items]
        # Count embedding units, not HTTP calls: a batch of 8 consumes 8 units.
        # Runs only after endpoint authentication and input validation.
        quota.acquire(len(prepared), client_id)
        lease = None
        complete = False
        try:
            if distributed_quota is not None:
                lease = await distributed_quota.admit(
                    len(prepared), client_id, sync=True
                )
            started = time.perf_counter()
            deadline = asyncio.timeout(
                settings.inference_timeout_seconds if settings.backend == "modal" else None
            )
            try:
                # Bound scheduling as well as GPU execution; a provider queue can
                # otherwise outlive both the client and Modal's execution timeout.
                # Local encoder threads cannot be cancelled safely this way.
                async with deadline:
                    vectors, timings = await service.encode_many_profiled(prepared)
            except Exception:
                if deadline.expired():
                    raise HTTPException(
                        status_code=504,
                        detail="Inference deadline exceeded; GPU cancellation requested",
                        headers={"Retry-After": "5"},
                    ) from None
                # Do not expose internals, credentials, model paths or image data.
                logger.error("Inference failed; provider error details suppressed")
                raise HTTPException(
                    status_code=503, detail="Inference temporarily unavailable"
                ) from None
            elapsed = 1000.0 * (time.perf_counter() - started)
            if len(vectors) != len(prepared) or len(timings) != len(prepared):
                raise HTTPException(status_code=503, detail="Inference returned incomplete results")
            responses = [
                EmbeddingResponse(
                    model=p.model,
                    repository=CATALOG[p.model].repository,
                    revision=model_revision(p.model),
                    backend=settings.backend,
                    dimension=p.dimension,
                    vector=vector,
                    input_sha256=p.input_sha256,
                    request_id=p.request_id,
                    elapsed_ms=round(elapsed, 3),
                    timing=timings[index],
                    normalized=p.options.get("normalize", True),
                    embedding_metadata=embedding_metadata(
                        p.model, model_revision(p.model), p.dimension, p.instruction,
                        p.options,
                    ),
                )
                for index, (p, vector) in enumerate(zip(prepared, vectors, strict=True))
            ]
            complete = True
            return responses

        finally:
            try:
                if complete and lease is not None:
                    # Only confirmed responses release early. An interrupted
                    # release is safe: the distributed lease will expire.
                    await asyncio.shield(distributed_quota.release(lease))
            finally:
                quota.release(client_id)

    @app.post("/v1/embeddings/async", dependencies=protected)
    async def submit_async_embedding(req: EmbeddingRequest, request: Request):
        if async_jobs is None:
            raise HTTPException(status_code=503, detail="Async inference is not configured")
        prepared = prepare(req, settings)
        payload, status_code = await async_jobs.submit(
            prepared, quota, request.scope["_embedding_client_id"],
            distributed_quota=distributed_quota,
        )
        security_monitor.job(payload.get("state", ""))
        headers = {"Cache-Control": "no-store"}
        if status_code == 202:
            headers["Retry-After"] = "3"
        return JSONResponse(content=payload, status_code=status_code, headers=headers)

    @app.get("/v1/embeddings/jobs/by-request/{request_id}", dependencies=protected)
    async def find_async_job_by_request_id(request_id: str, request: Request):
        """Authenticated, read-only lookup for lost async POST acknowledgements."""
        if async_jobs is None:
            raise HTTPException(status_code=503, detail="Async inference is not configured")
        payload, status_code = await async_jobs.status_by_request(
            request_id, request.scope["_embedding_client_id"],
        )
        security_monitor.job(payload.get("state", ""))
        headers = {"Cache-Control": "no-store"}
        if status_code == 202:
            headers["Retry-After"] = "3"
        return JSONResponse(content=payload, status_code=status_code, headers=headers)

    @app.get("/v1/embeddings/jobs/{job_id}", dependencies=protected)
    async def get_async_embedding(job_id: str, request: Request):
        if async_jobs is None:
            raise HTTPException(status_code=503, detail="Async inference is not configured")
        payload, status_code = await async_jobs.status(
            job_id, request.scope["_embedding_client_id"]
        )
        security_monitor.job(payload.get("state", ""))
        headers = {"Cache-Control": "no-store"}
        if status_code == 202:
            headers["Retry-After"] = "3"
        return JSONResponse(content=payload, status_code=status_code, headers=headers)

    @app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=protected)
    async def embedding(req: EmbeddingRequest, request: Request):
        return (await predict([req], request.scope["_embedding_client_id"]))[0]

    @app.post("/v1/embeddings/batches", response_model=BatchResponse, dependencies=protected)
    async def batch(req: BatchRequest, request: Request):
        if len(req.items) > settings.max_batch_size:
            raise HTTPException(status_code=422, detail="Batch size exceeds configured limit")
        results = await predict(req.items, request.scope["_embedding_client_id"])
        return BatchResponse(count=len(results), items=results)

    return app


def local_app() -> FastAPI:
    """Uvicorn factory; require authentication settings at startup."""
    return create_app()
