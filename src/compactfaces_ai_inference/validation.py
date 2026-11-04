import base64
import binascii
import hashlib
import io
import math
from dataclasses import dataclass

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

from .catalog import CATALOG, supported_dimension
from .config import Settings
from .schemas import EmbeddingRequest


@dataclass
class Prepared:
    model: str
    payload: bytes | str
    kind: str
    dimension: int
    input_sha256: str
    request_id: str | None


def prepare(req: EmbeddingRequest, settings: Settings) -> Prepared:
    spec = CATALOG.get(req.model)
    if spec is None:
        raise HTTPException(status_code=422, detail="Unsupported model ID (see GET /v1/models)")
    dim = req.dimension or spec.default_dimension
    if not supported_dimension(spec, dim):
        raise HTTPException(status_code=422, detail=f"Unsupported dimension for {req.model}")
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
        except (ValueError, Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from None
        digest = hashlib.sha256(content).hexdigest()
        kind = "image"
    return Prepared(
        model=req.model,
        payload=content,
        kind=kind,
        dimension=dim,
        input_sha256=digest,
        request_id=req.request_id,
    )


def image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def normalize_vector(values, required_dimension: int) -> list[float]:
    """Reject NaNs, non-finite and zero vectors; L2-normalize after truncation."""
    try:
        import numpy as np

        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size < required_dimension:
            raise ValueError("Model returned vector shorter than the requested dimension")
        arr = arr[:required_dimension]
        if not np.all(np.isfinite(arr)):
            raise ValueError("Model returned NaN/Inf vector")
        mag = float(np.linalg.norm(arr))
        if not math.isfinite(mag) or mag <= 1e-9:
            raise ValueError("Model returned zero vector")
        return (arr / mag).astype(float).tolist()
    except (TypeError, OverflowError) as exc:
        raise ValueError("Model returned an invalid vector") from exc
