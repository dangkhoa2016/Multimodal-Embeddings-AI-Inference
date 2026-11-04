"""Modal deployment: ONE L40S GPU worker, two models on ONE shared Volume.

To reduce VRAM usage, the GPU worker launches ONE isolated model process at a time.
When the requested model changes, the previous process exits and releases CUDA.

Commands:
  modal run modal_models.py --selection both
  modal deploy modal_app.py
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("compactfaces-ai-inference")
secret = modal.Secret.from_name("compactfaces-api")
volume = modal.Volume.from_name(
    os.getenv("CF_MODEL_VOLUME_NAME", "compactfaces-ai-weights"),
    create_if_missing=True,
)
common = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.110,<1", "pydantic>=2.6,<3", "numpy>=1.26,<3", "pillow>=10.2,<13")
    .add_local_dir(str(ROOT / "src"), remote_path="/opt/cf-src", copy=True)
    .env({"PYTHONPATH": "/opt/cf-src", "CF_BACKEND": "modal"})
)
# Two isolated Python environments inside ONE GPU container. They are deliberately
# not two Modal functions; deploying separate GPU functions could reserve two GPUs.
# This image is large because both Python environments include CUDA torch packages.
gpu_image = (
    common.apt_install("git")
    .run_commands(
        "python -m venv /opt/venv-qwen",
        (
            "/opt/venv-qwen/bin/pip install --no-cache-dir "
            "'torch==2.8.*' 'torchvision>=0.23,<0.24' "
            "'transformers>=4.57.3,<5' 'accelerate>=1.12,<3' "
            "'decord>=0.6' 'qwen-vl-utils==0.0.14' "
            "'huggingface-hub>=0.33,<2' numpy pillow"
        ),
        "python -m venv /opt/venv-wemm",
        (
            "/opt/venv-wemm/bin/pip install --no-cache-dir "
            "'torch==2.8.*' 'transformers==5.2.0' "
            "'accelerate>=1.1,<3' 'qwen-vl-utils==0.0.14' "
            "'huggingface-hub>=0.33,<2' numpy pillow"
        ),
        (
            "git init /opt/Qwen3-VL-Embedding && cd /opt/Qwen3-VL-Embedding && "
            "git remote add origin https://github.com/QwenLM/Qwen3-VL-Embedding.git && "
            "git fetch --depth 1 origin 393e2978d27852b0d0230d6994f37f9c15bed73c && "
            "git checkout --detach FETCH_HEAD"
        ),
        "cd /opt/Qwen3-VL-Embedding && git rev-parse HEAD > /opt/qwen-upstream-commit.txt",
    )
    .env(
        {
            "CF_MODEL_ROOT": "/models/checkpoints",
            "HF_HUB_OFFLINE": "1",
            "CF_QWEN_SOURCE_PATH": "/opt/Qwen3-VL-Embedding",
        }
    )
)


@app.cls(
    gpu="L40S",
    image=gpu_image,
    secrets=[secret],
    volumes={"/models": volume},
    min_containers=0,
    max_containers=1,
    scaledown_window=180,
    timeout=1800,
)
class SharedL40SGpu:
    @modal.enter()
    def initialize(self):
        from compactfaces_ai_inference.gpu_supervisor import ModelSupervisor

        self.supervisor = ModelSupervisor()

    @modal.method()
    def encode_many(self, model_id: str, items: list[dict]) -> list[list[float]]:
        return self.supervisor.encode_many(model_id, items)

    @modal.exit()
    def cleanup(self):
        self.supervisor.close()


@app.function(image=common, secrets=[secret], min_containers=0, max_containers=2, timeout=1800)
@modal.asgi_app()
def api_entrypoint():
    from compactfaces_ai_inference.api import create_app
    from compactfaces_ai_inference.config import Settings
    from compactfaces_ai_inference.service import InferenceService

    return create_app(Settings.from_env(), InferenceService("modal", SharedL40SGpu()))
