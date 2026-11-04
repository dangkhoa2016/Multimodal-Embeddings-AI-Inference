import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_token: str
    backend: str = "mock"
    max_image_bytes: int = 5 * 1024 * 1024
    max_image_pixels: int = 12_000_000
    max_batch_size: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.getenv("CF_BACKEND", "mock").lower().strip()
        if backend not in {"mock", "local", "modal"}:
            raise ValueError("CF_BACKEND must be mock, local, or modal")
        token = os.getenv("CF_API_TOKEN", "").strip()
        if len(token) < 16:
            raise ValueError("CF_API_TOKEN must be set to a secret of at least 16 characters")
        return cls(
            api_token=token,
            backend=backend,
            max_image_bytes=int(os.getenv("CF_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))),
            max_image_pixels=int(os.getenv("CF_MAX_IMAGE_PIXELS", "12000000")),
            max_batch_size=min(32, max(1, int(os.getenv("CF_MAX_BATCH_SIZE", "8")))),
        )
