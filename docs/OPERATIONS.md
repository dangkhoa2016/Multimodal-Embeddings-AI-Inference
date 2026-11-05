# Modal deployment and operations

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](OPERATIONS.vi.md)

Follow the steps below to deploy and operate the **Multimodal Embeddings AI Inference API** on Modal.

## Runtime layout

- A CPU FastAPI gateway accepts HTTPS requests authenticated with a Bearer token.
- Two independent GPU workers each use one NVIDIA L40S with `max_containers=1`; no cross-model subprocess switching.
- Each model checkpoint resides in its own Modal Volume, separate from the GPU's VRAM.

## Names and resource ownership

- Environment variables use the `VISION_EMBEDDINGS_` prefix (for example `VISION_EMBEDDINGS_API_TOKEN`).
- Python import package: `multimodal_embeddings_ai_inference`.
- Modal App: `multimodal-embeddings-ai-inference`.
- Modal Secret: `multimodal-embeddings-api`.
- Model Volumes: `multimodal-embeddings-qwen-weights` and `multimodal-embeddings-wemm-weights`.

Configure `VISION_EMBEDDINGS_QWEN_VOLUME_NAME` and `VISION_EMBEDDINGS_WEMM_VOLUME_NAME` independently for checkpoint downloads and GPU inference. Run `scripts/create_modal_secret.py` to create the API authentication Secret.

## Setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m pip install -e '.[modal]'
modal setup
```

This command sequence runs locally without a GPU; Modal handles remote model downloads and inference.

## Download and pin checkpoints

```bash
modal run modal_models.py --selection both
```

Record the immutable revisions for [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B). The currently verified revisions are:

- `Qwen/Qwen3-VL-Embedding-8B`: `2c4565515e0f265c6511776e7193b22c0968ddc7`.
- `tencent/WeMM-Embedding-9B`: `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`.

The SHAs identify public upstream model snapshots; they are **not access tokens**. Do not silently replace them with moving `main` when reproducing results.

## Configure authentication

```bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482
```

The example pins specific model checkpoint revisions. Verify the downloaded checkpoint revisions and replace the example SHAs when they differ before creating the Secret.

The helper creates an API Bearer token, registers a Modal Secret, and saves the same values on **the computer running the command**, with owner-only file permissions. The completion message prints the local file path, not the token. Treat this local file as the source for client onboarding; do not print, commit or copy it into public notebook code.

For an authenticated smoke test on that same computer, load the file without displaying its contents:

```bash
TOKEN_FILE="/path/printed/by/the/helper"
set -a
source "$TOKEN_FILE"
set +a
```

For Colab/Kaggle or another machine, share **only the API token** through a trusted secret manager or a hidden notebook prompt, never a full local credential file.

Loading the file also exports `VISION_EMBEDDINGS_QWEN8_REVISION` and `VISION_EMBEDDINGS_WEMM9_REVISION`. The HTTPS smoke test in the README requires these expected checkpoint SHAs; they can also be supplied with `--qwen-revision` and `--wemm-revision`. It rejects redirects and revision mismatches before reporting PASS for the affected model.

## Deliberate API token rotation

`--force` allows replacing the existing Modal Secret but **reuses a saved token**. `--rotate-token --force` generates a new token, updates Modal and replaces the local file only after Modal CLI succeeds. Do not delete the existing token file to trigger rotation.

~~~bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482 \
  --force --rotate-token
~~~

Prepare client updates before rotating. Depending on Modal's running-container lifecycle, restart or redeploy the API so existing instances use the updated Secret. After rollout, verify the new token works and the old token receives HTTP 401. The helper retains the old local token file if Modal CLI fails. If the local file is lost, --force generates a new token.


## Deploy

```bash
modal deploy modal_app.py
```

Use the deployment URL printed by your own Modal account. Configure it in your API caller or server-side secret manager; do not hardcode a live service URL in the repository.

## Acceptance checks

```bash
.venv/bin/ruff check src modal_app.py modal_models.py scripts tests
.venv/bin/python -m pytest -q
python scripts/smoke_modal_gpu.py --image /path/to/authorized-image.png 
```

A real GPU smoke test must return finite, L2-normalized vectors for both full model IDs and successfully route to their dedicated GPU workers. The HTTPS path also needs an authenticated request to `/v1/embeddings`.

## Access control and spending safeguards

For deployments serving multiple clients, configure **per-client rate limits and quotas at an authenticated ingress/proxy**. Allow only authorized callers, monitor usage and billing alerts, and restrict GPU scaling/concurrency to your budget. Keep a way to disable the deployment promptly. Provider budget alerts are **not** guaranteed hard spending caps. This service does **not** enforce global or per-client quotas, so do not distribute one shared Bearer token to untrusted clients.

## Logs, reliability and cost

```bash
modal app list
modal app logs multimodal-embeddings-ai-inference
```

Treat provider logs as private operational data. Avoid logging tokens, uploaded images, raw embeddings or identifiable input metadata.
Requests can incur cold-start and queueing latency. Account for GPU, CPU, storage and network charges; configure budget protection and rate limits before inviting external users.

For API fields, see [API contract](API_CONTRACT.md); for verification procedures, see [Verification](VERIFICATION.md).
