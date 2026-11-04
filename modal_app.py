"""Modal deployment with one CPU API gateway and two dedicated L40S model workers."""

import os
import time
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("multimodal-embeddings-ai-inference")
secret = modal.Secret.from_name(
    os.getenv("VISION_EMBEDDINGS_MODAL_SECRET_NAME", "multimodal-embeddings-api")
)

qwen_volume = modal.Volume.from_name(
    os.getenv("VISION_EMBEDDINGS_QWEN_VOLUME_NAME", "multimodal-embeddings-qwen-weights"),
    create_if_missing=False,
)
wemm_volume = modal.Volume.from_name(
    os.getenv("VISION_EMBEDDINGS_WEMM_VOLUME_NAME", "multimodal-embeddings-wemm-weights"),
    create_if_missing=False,
)

base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi>=0.110,<1",
    "pydantic>=2.6,<3",
    "numpy>=1.26,<3",
    "pillow>=10.2,<13",
)
common_env = {
    "PYTHONPATH": "/opt/embedding-src",
    "VISION_EMBEDDINGS_BACKEND": "modal",
    "VISION_EMBEDDINGS_MODEL_ROOT": "/models/checkpoints",
    "HF_HUB_OFFLINE": "1",
}
gateway_image = base_image.env(common_env).add_local_dir(
    str(ROOT / "src"), remote_path="/opt/embedding-src"
)

qwen_image = (
    base_image.apt_install("git")
    .run_commands(
        "python -m venv /opt/venv-qwen",
        (
            "/opt/venv-qwen/bin/pip install --no-cache-dir "
            "'torch==2.8.*' 'torchvision>=0.23,<0.24' "
            "'transformers>=4.57.3,<5' 'accelerate>=1.12,<3' "
            "'decord>=0.6' 'qwen-vl-utils==0.0.14' "
            "'huggingface-hub>=0.33,<2' numpy pillow"
        ),
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
    base_image.run_commands(
        "python -m venv /opt/venv-wemm",
        (
            "/opt/venv-wemm/bin/pip install --no-cache-dir "
            "'torch==2.8.*' 'torchvision>=0.23,<0.24' "
            "'transformers==5.2.0' 'accelerate>=1.1,<3' "
            "'qwen-vl-utils==0.0.14' "
            "'huggingface-hub>=0.33,<2' numpy pillow"
        ),
    )
    .env(common_env)
    .add_local_dir(str(ROOT / "src"), remote_path="/opt/embedding-src")
)


@app.cls(
    gpu="L40S",
    cpu=8,
    memory=65536,
    image=qwen_image,
    secrets=[secret],
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
    secrets=[secret],
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
    secrets=[secret],
    min_containers=0,
    max_containers=2,
    timeout=1800,
)
@modal.asgi_app()
def api_entrypoint():
    from multimodal_embeddings_ai_inference.api import create_app
    from multimodal_embeddings_ai_inference.config import Settings
    from multimodal_embeddings_ai_inference.service import InferenceService

    workers = {
        "qwen3-vl-embedding-8b": QwenL40SGpu(),
        "wemm-embedding-9b": WeMML40SGpu(),
    }
    return create_app(Settings.from_env(), InferenceService("modal", workers))
