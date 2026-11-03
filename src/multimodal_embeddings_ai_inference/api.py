"""Framework-independent REST API with Bearer authentication for inference."""

import hmac
import logging
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .catalog import CATALOG, public_spec
from .config import Settings
from .embedding_config import embedding_metadata
from .schemas import BatchRequest, BatchResponse, EmbeddingRequest, EmbeddingResponse
from .service import InferenceService, model_revision
from .validation import prepare

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, service: InferenceService | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or InferenceService(settings.backend)
    app = FastAPI(
        title="Multimodal Embeddings AI Inference API",
        version=__version__,
        description="Authenticated image/text embeddings for research, not biometric verification.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def check_body_size(request: Request, call_next):
        # JSON/base64 overhead; decoded image limit is additionally enforced in prepare().
        limit = max(1024 * 1024, settings.max_image_bytes * settings.max_batch_size * 2)
        try:
            n = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        if n > limit:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        # Enforce size even for chunked transfers without Content-Length.
        # Starlette caches the body for downstream FastAPI parsing.
        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
            chunks.append(chunk)
        request._body = b"".join(chunks)
        return await call_next(request)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        prefix = "Bearer "
        if (
            authorization is None
            or not authorization.startswith(prefix)
            or not hmac.compare_digest(authorization[len(prefix) :], settings.api_token)
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    protected = [Depends(authenticate)]

    @app.get("/v1/health", dependencies=protected)
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "backend": settings.backend,
            "note": "API liveness only; this does not prove that models are loaded or correct",
        }

    @app.get("/v1/models", dependencies=protected)
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

    async def predict(items: list[EmbeddingRequest]) -> list[EmbeddingResponse]:
        prepared = [prepare(item, settings) for item in items]
        started = time.perf_counter()
        try:
            vectors, timings = await service.encode_many_profiled(prepared)
        except Exception:
            # Never expose framework tracebacks, model paths, Hugging Face tokens or image contents.
            logger.exception("Inference failed")
            raise HTTPException(
                status_code=503, detail="Inference temporarily unavailable"
            ) from None
        elapsed = 1000.0 * (time.perf_counter() - started)
        if len(vectors) != len(prepared) or len(timings) != len(prepared):
            raise HTTPException(status_code=503, detail="Inference returned incomplete results")
        return [
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
                embedding_metadata=embedding_metadata(
                    p.model, model_revision(p.model), p.dimension, p.instruction
                ),
            )
            for index, (p, vector) in enumerate(zip(prepared, vectors, strict=True))
        ]

    @app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=protected)
    async def embedding(req: EmbeddingRequest):
        return (await predict([req]))[0]

    @app.post("/v1/embeddings/batches", response_model=BatchResponse, dependencies=protected)
    async def batch(req: BatchRequest):
        if len(req.items) > settings.max_batch_size:
            raise HTTPException(status_code=422, detail="Batch size exceeds configured limit")
        results = await predict(req.items)
        return BatchResponse(count=len(results), items=results)

    return app


def local_app() -> FastAPI:
    """Uvicorn factory; require authentication settings at startup."""
    return create_app()
