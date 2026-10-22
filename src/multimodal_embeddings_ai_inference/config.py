import hashlib
import json
import os
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    api_token: str = field(repr=False)
    backend: str = "mock"
    client_key_hashes: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)
    max_async_jobs_global: int = 8
    max_async_jobs_per_client: int = 2
    max_embeddings_per_client_per_minute: int = 60
    distributed_rate_enabled: bool = False
    distributed_rate_redis_url: str = field(default="", repr=False)
    distributed_rate_namespace: str = "multimodal-embeddings-v1"
    max_distributed_sync_inflight: int = 4
    max_distributed_client_sync_inflight: int = 2
    gateway_max_containers: int = 1
    max_body_requests: int = 8
    max_image_bytes: int = 5 * 1024 * 1024
    max_image_pixels: int = 12_000_000
    max_batch_size: int = 8
    max_embeddings_per_minute: int = 120
    max_inflight_requests: int = 4
    inference_timeout_seconds: float = 60
    auth_failure_limit: int = 3
    auth_failure_window_seconds: int = 300
    auth_lockout_seconds: int = 900
    auth_max_tracked_clients: int = 4096

    def __post_init__(self):
        if (not isinstance(self.api_token, str) or not 16 <= len(self.api_token) <= 4089
                or not all(33 <= ord(ch) <= 126 for ch in self.api_token)):
            raise ValueError(
                "VISION_EMBEDDINGS_API_TOKEN must contain 16–4089 visible ASCII characters"
            )
        if not isinstance(self.client_key_hashes, dict) or len(self.client_key_hashes) > 64:
            raise ValueError("Client key registry must contain at most 64 clients")
        seen = {hashlib.sha256(self.api_token.encode("ascii")).hexdigest()}
        for client_id, hashes in self.client_key_hashes.items():
            if (not isinstance(client_id, str) or client_id == "owner"
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,63}", client_id)):
                raise ValueError("Invalid or reserved client ID")
            if not isinstance(hashes, (list, tuple)) or not 1 <= len(hashes) <= 4:
                raise ValueError("Each client needs 1-4 API key digests")
            for digest in hashes:
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Client key digest must be lowercase SHA-256")
                if digest in seen:
                    raise ValueError("API keys must be unique across clients")
                seen.add(digest)
        if not 1 <= self.max_async_jobs_global <= 256:
            raise ValueError("Invalid global async outstanding limit")
        if not 1 <= self.max_async_jobs_per_client <= self.max_async_jobs_global:
            raise ValueError("Invalid per-client async outstanding limit")
        if not 1 <= self.max_embeddings_per_client_per_minute <= 100000:
            raise ValueError("Invalid per-client embedding rate limit")
        if not 1 <= self.max_embeddings_per_minute <= 100000:
            raise ValueError("Invalid global embedding rate limit")
        if not (
            1 <= self.max_distributed_client_sync_inflight
            <= self.max_distributed_sync_inflight <= 64
        ):
            raise ValueError("Invalid distributed sync concurrency limits")
        if (
            type(self.gateway_max_containers) is not int
            or not 1 <= self.gateway_max_containers <= 8
        ):
            raise ValueError("Invalid gateway container limit")
        if type(self.max_body_requests) is not int or not 1 <= self.max_body_requests <= 16:
            raise ValueError("Invalid request body concurrency limit")
        if self.gateway_max_containers > 1 and not self.distributed_rate_enabled:
            raise ValueError("Multi-gateway deployment requires distributed rate limiting")
        if self.distributed_rate_enabled:
            from urllib.parse import urlsplit

            url = urlsplit(self.distributed_rate_redis_url)
            if url.scheme not in ("redis", "rediss") or not url.hostname:
                raise ValueError("Distributed rate limiting needs a Redis URL")
            if url.scheme == "redis" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError("Remote Redis connections must use TLS (rediss://)")
            if not re.fullmatch(r"[A-Za-z0-9:_-]{1,96}", self.distributed_rate_namespace):
                raise ValueError("Invalid Redis rate-limit namespace")
        for key, value, ceiling in (
            ("VISION_EMBEDDINGS_AUTH_FAILURE_LIMIT", self.auth_failure_limit, 100),
            (
                "VISION_EMBEDDINGS_AUTH_FAILURE_WINDOW_SECONDS",
                self.auth_failure_window_seconds,
                86400,
            ),
            ("VISION_EMBEDDINGS_AUTH_LOCKOUT_SECONDS", self.auth_lockout_seconds, 86400),
            ("VISION_EMBEDDINGS_AUTH_MAX_TRACKED_CLIENTS", self.auth_max_tracked_clients, 100000),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{key} must be an integer between 1 and {ceiling}")
        if (type(self.inference_timeout_seconds) not in (int, float)
                or not 0 < self.inference_timeout_seconds <= 120):
            raise ValueError("VISION_EMBEDDINGS_INFERENCE_TIMEOUT_SECONDS must be > 0 and <= 120")

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.getenv("VISION_EMBEDDINGS_BACKEND", "mock").lower().strip()
        if backend not in {"mock", "local", "modal"}:
            raise ValueError("VISION_EMBEDDINGS_BACKEND must be mock, local, or modal")
        token = os.getenv("VISION_EMBEDDINGS_API_TOKEN", "").strip()
        if len(token) < 16:
            raise ValueError(
                "VISION_EMBEDDINGS_API_TOKEN must be set to a secret of at least 16 characters"
            )
        raw_keys = os.getenv("VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON", "{}")
        try:
            client_keys = json.loads(raw_keys)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON") from error
        redis_url = os.getenv("VISION_EMBEDDINGS_RATE_REDIS_URL", "")
        flag = os.getenv("VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT", "").strip()
        if flag not in ("", "0", "1"):
            raise ValueError("VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT must be 0 or 1")
        distributed = bool(redis_url) if not flag else flag == "1"
        return cls(
            distributed_rate_enabled=distributed,
            distributed_rate_redis_url=redis_url,
            distributed_rate_namespace=os.getenv(
                "VISION_EMBEDDINGS_RATE_REDIS_NAMESPACE", "multimodal-embeddings-v1"
            ),
            gateway_max_containers=int(
                os.getenv("VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS", "1")
            ),
            max_body_requests=int(os.getenv("VISION_EMBEDDINGS_MAX_BODY_REQUESTS", "8")),
            max_distributed_sync_inflight=int(
                os.getenv("VISION_EMBEDDINGS_MAX_DISTRIBUTED_SYNC_INFLIGHT", "4")
            ),
            max_distributed_client_sync_inflight=int(
                os.getenv("VISION_EMBEDDINGS_MAX_DISTRIBUTED_CLIENT_SYNC_INFLIGHT", "2")
            ),
            client_key_hashes=client_keys,
            max_async_jobs_global=int(os.getenv("VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL", "8")),
            max_async_jobs_per_client=int(
                os.getenv("VISION_EMBEDDINGS_MAX_ASYNC_JOBS_PER_CLIENT", "2")
            ),
            max_embeddings_per_client_per_minute=int(
                os.getenv("VISION_EMBEDDINGS_MAX_CLIENT_EMBEDDINGS_PER_MINUTE", "60")
            ),
            api_token=token,
            backend=backend,
            max_image_bytes=int(
                os.getenv("VISION_EMBEDDINGS_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))
            ),
            max_image_pixels=int(os.getenv("VISION_EMBEDDINGS_MAX_IMAGE_PIXELS", "12000000")),
            max_batch_size=min(32, max(1, int(os.getenv("VISION_EMBEDDINGS_MAX_BATCH_SIZE", "8")))),
            max_embeddings_per_minute=max(
                1, int(os.getenv("VISION_EMBEDDINGS_MAX_EMBEDDINGS_PER_MINUTE", "120"))
            ),
            max_inflight_requests=max(
                1, int(os.getenv("VISION_EMBEDDINGS_MAX_INFLIGHT_REQUESTS", "4"))
            ),
            inference_timeout_seconds=float(
                os.getenv("VISION_EMBEDDINGS_INFERENCE_TIMEOUT_SECONDS", "60")
            ),
            auth_failure_limit=int(os.getenv("VISION_EMBEDDINGS_AUTH_FAILURE_LIMIT", "3")),
            auth_failure_window_seconds=int(
                os.getenv("VISION_EMBEDDINGS_AUTH_FAILURE_WINDOW_SECONDS", "300")
            ),
            auth_lockout_seconds=int(os.getenv("VISION_EMBEDDINGS_AUTH_LOCKOUT_SECONDS", "900")),
            auth_max_tracked_clients=int(
                os.getenv("VISION_EMBEDDINGS_AUTH_MAX_TRACKED_CLIENTS", "4096")
            ),
        )
