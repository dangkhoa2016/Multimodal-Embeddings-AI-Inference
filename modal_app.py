"""Modal deployment with one CPU API gateway and two dedicated L40S model workers."""

import os
import time
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
# Deployment-time scaling choice (not a GPU setting). Require Redis admission
# in every gateway whenever more than one CPU gateway container is allowed.
gateway_max_containers = int(os.getenv("VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS", "1"))
if not 1 <= gateway_max_containers <= 8:
    raise ValueError("Gateway container count must be between 1 and 8")
app = modal.App("multimodal-embeddings-ai-inference")
gateway_secret = modal.Secret.from_name(
    os.getenv("VISION_EMBEDDINGS_MODAL_SECRET_NAME", "multimodal-embeddings-api")
)

# Workers receive checkpoint revisions only, never the gateway bearer token.
model_revision_secret = modal.Secret.from_name(
    os.getenv("VISION_EMBEDDINGS_MODAL_MODEL_SECRET_NAME", "multimodal-embeddings-model-config")
)

qwen_volume = modal.Volume.from_name(
    os.getenv("VISION_EMBEDDINGS_QWEN_VOLUME_NAME", "multimodal-embeddings-qwen-weights"),
    create_if_missing=False,
)
wemm_volume = modal.Volume.from_name(
    os.getenv("VISION_EMBEDDINGS_WEMM_VOLUME_NAME", "multimodal-embeddings-wemm-weights"),
    create_if_missing=False,
)

# Job metadata and completed vectors only: never store tokens or raw images.
job_records = modal.Dict.from_name(
    os.getenv("VISION_EMBEDDINGS_JOB_DICT_NAME", "multimodal-embeddings-async-jobs"),
    create_if_missing=True,
)


class _ModalJobStore:
    async def get(self, key: str):
        return await job_records.get.aio(key)

    async def put(self, key: str, record: dict, skip_if_exists: bool = False):
        return await job_records.put.aio(key, record, skip_if_exists=skip_if_exists)

    async def pop(self, key: str, default=None):
        return await job_records.pop.aio(key, default)


base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.162.2",
    "pydantic==2.14.3",
    "numpy==2.3.5",
    "pillow==12.3.0",
)
common_env = {
    "PYTHONPATH": "/opt/embedding-src",
    "VISION_EMBEDDINGS_BACKEND": "modal",
    "VISION_EMBEDDINGS_MODEL_ROOT": "/models/checkpoints",
    "HF_HUB_OFFLINE": "1",
}
gateway_image = base_image.pip_install("redis==6.4.0").env({
    **common_env,
    "VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS": str(gateway_max_containers),
}).add_local_dir(
    str(ROOT / "src"), remote_path="/opt/embedding-src"
)

qwen_image = (
    base_image.add_local_file(
        str(ROOT / "constraints" / "gpu-qwen.txt"), remote_path="/opt/gpu-qwen.txt", copy=True
    )
    .apt_install("git")
    .run_commands(
        "python -m venv /opt/venv-qwen",
        "/opt/venv-qwen/bin/pip install --no-cache-dir -r /opt/gpu-qwen.txt",
        (
            "git init /opt/Qwen3-VL-Embedding && cd /opt/Qwen3-VL-Embedding && "
            "git remote add origin https://github.com/QwenLM/Qwen3-VL-Embedding.git && "
            "git fetch --depth 1 origin 393e2978d27852b0d0230d6994f37f9c15bed73c && "
            "git checkout --detach FETCH_HEAD"
        ),
    )
    .env({**common_env, "VISION_EMBEDDINGS_QWEN_SOURCE_PATH": "/opt/Qwen3-VL-Embedding"})
    .add_local_dir(str(ROOT / "src"), remote_path="/opt/embedding-src")
)
wemm_image = (
    base_image.add_local_file(
        str(ROOT / "constraints" / "gpu-wemm.txt"), remote_path="/opt/gpu-wemm.txt", copy=True
    ).run_commands(
        "python -m venv /opt/venv-wemm",
        "/opt/venv-wemm/bin/pip install --no-cache-dir -r /opt/gpu-wemm.txt",
    )
    .env(common_env)
    .add_local_dir(str(ROOT / "src"), remote_path="/opt/embedding-src")
)


@app.cls(
    gpu="L40S",
    cpu=8,
    memory=65536,
    image=qwen_image,
    secrets=[model_revision_secret],
    volumes={"/models": qwen_volume},
    min_containers=0,
    max_containers=1,
    scaledown_window=180,
    timeout=1800,
)
class QwenL40SGpu:
    @modal.enter()
    def initialize(self):
        from multimodal_embeddings_ai_inference.gpu_supervisor import ModelSupervisor

        self.supervisor = ModelSupervisor(
            python_paths={"qwen3-vl-embedding-8b": "/opt/venv-qwen/bin/python"}
        )
        self.gpu_worker_id = uuid.uuid4().hex

    @modal.method()
    def encode_many(self, model_id: str, items: list[dict]) -> dict:
        if model_id != "qwen3-vl-embedding-8b":
            raise ValueError("This worker only serves Qwen3-VL-Embedding-8B")
        started = time.perf_counter()
        result = self.supervisor.encode_many_profiled(model_id, items)
        result["timing"]["gpu_worker_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 3
        )
        result["timing"]["gpu_worker_id"] = self.gpu_worker_id
        return result

    @modal.exit()
    def cleanup(self):
        self.supervisor.close()


@app.cls(
    gpu="L40S",
    cpu=8,
    memory=65536,
    image=wemm_image,
    secrets=[model_revision_secret],
    volumes={"/models": wemm_volume},
    min_containers=0,
    max_containers=1,
    scaledown_window=180,
    timeout=1800,
)
class WeMML40SGpu:
    @modal.enter()
    def initialize(self):
        from multimodal_embeddings_ai_inference.gpu_supervisor import ModelSupervisor

        self.supervisor = ModelSupervisor(
            python_paths={"wemm-embedding-9b": "/opt/venv-wemm/bin/python"}
        )
        self.gpu_worker_id = uuid.uuid4().hex

    @modal.method()
    def encode_many(self, model_id: str, items: list[dict]) -> dict:
        if model_id != "wemm-embedding-9b":
            raise ValueError("This worker only serves WeMM-Embedding-9B")
        started = time.perf_counter()
        result = self.supervisor.encode_many_profiled(model_id, items)
        result["timing"]["gpu_worker_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 3
        )
        result["timing"]["gpu_worker_id"] = self.gpu_worker_id
        return result

    @modal.exit()
    def cleanup(self):
        self.supervisor.close()


@app.function(
    image=gateway_image,
    secrets=[model_revision_secret],
    min_containers=0,
    max_containers=2,
    timeout=3600,
)
async def run_embedding_job(job_id: str, prepared_data: dict):
    """Detached CPU coordinator; GPU scheduling can exceed an HTTP timeout."""
    from multimodal_embeddings_ai_inference.async_jobs import execute_job
    from multimodal_embeddings_ai_inference.service import InferenceService

    workers = {
        "qwen3-vl-embedding-8b": QwenL40SGpu(),
        "wemm-embedding-9b": WeMML40SGpu(),
    }
    await execute_job(
        job_id, prepared_data, _ModalJobStore(), InferenceService("modal", workers)
    )


@app.function(
    image=gateway_image,
    secrets=[gateway_secret],
    min_containers=0,
    # Multi-gateway requires a shared atomic Redis limiter configured at runtime.
    max_containers=gateway_max_containers,
    timeout=1800,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def api_entrypoint():
    from multimodal_embeddings_ai_inference.api import create_app
    from multimodal_embeddings_ai_inference.async_admission import AsyncSlotPool
    from multimodal_embeddings_ai_inference.async_jobs import AsyncEmbeddingJobs
    from multimodal_embeddings_ai_inference.config import Settings
    from multimodal_embeddings_ai_inference.distributed_quota import RedisDistributedQuota
    from multimodal_embeddings_ai_inference.service import InferenceService

    workers = {
        "qwen3-vl-embedding-8b": QwenL40SGpu(),
        "wemm-embedding-9b": WeMML40SGpu(),
    }
    settings = Settings.from_env()
    if settings.max_inflight_requests >= 16:
        raise ValueError("Inference limit must leave gateway capacity for health and quota replies")
    store = _ModalJobStore()
    distributed_quota = None
    if settings.distributed_rate_enabled:
        import redis.asyncio as redis_async

        redis_client = redis_async.from_url(
            settings.distributed_rate_redis_url,
            decode_responses=True, socket_connect_timeout=1.0,
            socket_timeout=1.5, retry_on_timeout=False,
        )
        distributed_quota = RedisDistributedQuota(
            redis_client, namespace=settings.distributed_rate_namespace,
            global_units_per_minute=settings.max_embeddings_per_minute,
            client_units_per_minute=settings.max_embeddings_per_client_per_minute,
            global_sync_inflight=settings.max_distributed_sync_inflight,
            client_sync_inflight=settings.max_distributed_client_sync_inflight,
            lease_seconds=int(settings.inference_timeout_seconds) + 60,
        )
    jobs = AsyncEmbeddingJobs(
        store, run_embedding_job.spawn.aio,
        admission=AsyncSlotPool(
            store, global_limit=settings.max_async_jobs_global,
            per_client_limit=settings.max_async_jobs_per_client,
        ),
    )
    return create_app(
        settings, InferenceService("modal", workers), async_jobs=jobs,
        distributed_quota=distributed_quota,
    )
