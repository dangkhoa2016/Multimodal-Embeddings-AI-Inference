"""GPU-safe image decoding and vector normalization without FastAPI dependencies."""

import io
import math

from PIL import Image, ImageOps


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
