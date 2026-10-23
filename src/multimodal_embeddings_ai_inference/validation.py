import base64
import binascii
import hashlib
import io
from dataclasses import dataclass, field

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from .catalog import CATALOG
from .config import Settings
from .model_options import resolve_request_options
from .schemas import EmbeddingRequest


@dataclass
class Prepared:
    model: str
    payload: bytes | str
    kind: str
    dimension: int
    input_sha256: str
    request_id: str | None
    instruction: str | None = None
    options: dict = field(default_factory=dict)


def prepare(req: EmbeddingRequest, settings: Settings) -> Prepared:
    spec = CATALOG.get(req.model)
    if spec is None:
        raise HTTPException(status_code=422, detail="Unsupported model ID (see GET /v1/models)")
    kind = "text" if req.text is not None else "image"
    try:
        dim, instruction, options = resolve_request_options(
            req.model, req.options, dimension=req.dimension,
            instruction=req.instruction, kind=kind,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if req.text is not None:
        content: bytes | str = req.text
        digest = hashlib.sha256(req.text.encode("utf-8")).hexdigest()
        kind = "text"
    else:
        raw = req.image_base64 or ""
        if raw.startswith("data:") or len(raw) > (settings.max_image_bytes + 2) * 4 // 3 + 8:
            raise HTTPException(
                status_code=413, detail="Use raw base64 within the image size limit"
            )
        try:
            content = base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error):
            raise HTTPException(status_code=422, detail="Invalid base64 image") from None
        if not content or len(content) > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="Image exceeds size limit")
        try:
            with Image.open(io.BytesIO(content)) as im:
                if im.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Supported formats: JPEG, PNG, WEBP")
                if im.width * im.height > settings.max_image_pixels:
                    raise ValueError("Image exceeds maximum pixel count")
                im.verify()
        except (ValueError, Image.DecompressionBombError, UnidentifiedImageError, OSError):
            # Pillow exception text may contain user-controlled image metadata
            # or filesystem details; never echo or log it to an API client.
            raise HTTPException(status_code=422, detail="Invalid image") from None
        digest = hashlib.sha256(content).hexdigest()
        kind = "image"
    return Prepared(
        model=req.model,
        payload=content,
        kind=kind,
        dimension=dim,
        input_sha256=digest,
        request_id=req.request_id,
        instruction=instruction,
        options=options,
    )
