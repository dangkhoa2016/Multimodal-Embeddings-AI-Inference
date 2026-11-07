
# Multimodal Embeddings AI Inference

**An API that turns images and text into embedding vectors**, powered by [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B). Designed for self-deployment with a FastAPI gateway and two dedicated Modal NVIDIA L40S GPU workers.

[![CI](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/actions/workflows/ci.yml) ![Version](https://img.shields.io/badge/version-1.0.0-2563EB) [![License: MIT](https://img.shields.io/badge/license-MIT-16A34A)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-inference-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![Modal L40S](https://img.shields.io/badge/Modal-NVIDIA%20L40S-111827)](https://modal.com/) [![Image and text](https://img.shields.io/badge/input-image%20%2B%20text-0F766E)](docs/API_CONTRACT.md)

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

**Choose how to start**

- **Try locally:** Run the [mock quickstart](#quickstart-local-contract-tests) without a GPU or Modal account.
- **Call an existing API:** Use the [CPU notebook](#interactive-notebook-for-kaggle-and-google-colab) or the [minimal HTTP client](docs/CLIENT_INTEGRATION.md) with your own HTTPS endpoint and application token.
- **Host the models:** Follow the [Modal deployment guide](#deploy-real-inference-on-modal) to download checkpoints and serve both models.

## Overview

Clients written in **Python, Node.js, C#/.NET, Java, Go or any other language** can request image/text embeddings using standard HTTP and JSON.

- **Input:** a PNG/JPEG/WebP image encoded as Base64, or UTF-8 text.
- **Output:** normalized numeric vector, model identifier, checkpoint revision, input SHA256, dimension and processing time.
- **Serving:** CPU FastAPI gateway and two dedicated NVIDIA L40S GPU workers, one model per worker.
- **Authentication:** application-level Bearer token. Production clients should keep it in their backend; personal Colab/Kaggle demos use a private runtime and hidden prompts, never a published notebook cell.

## Call an existing API

This repository provides the source code for deploying an AI inference service over HTTPS on Modal. Once deployed, the API can be accessed from Python, Ruby on Rails, Node.js, or any HTTP client using the deployment's HTTPS endpoint and application Bearer token (`VISION_EMBEDDINGS_API_TOKEN`). The following request uses the API ID of **tencent/WeMM-Embedding-9B**:

```bash
curl -X POST 'https://YOUR_DEPLOYMENT.modal.run/v1/embeddings' \
  -H "Authorization: Bearer $VISION_EMBEDDINGS_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"wemm-embedding-9b","text":"a mountain landscape","dimension":1024}'
```

For image input, send `image_base64` instead of `text`. See the [API contract](docs/API_CONTRACT.md) for response fields, limits and errors.

## Example use cases

- **Text-to-image retrieval:** search an indexed image collection using text and image vectors produced by the same model and revision.
- **Image similarity:** find related images within an authorized collection.
- **Application integrations:** call the authenticated HTTPS API from a backend and store vectors in a dedicated search index.

These are potential use cases rather than measured accuracy claims. Evaluate retrieval quality on representative data for your task.

## Supported models

| Full Hugging Face model name | API ID | Parameters | Supported dimensions |
| --- | --- | ---: | --- |
| [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | `qwen3-vl-embedding-8b` | 8B | 64–4096 |
| [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) | `wemm-embedding-9b` | 9B | 64, 128, 256, 512, 1024, 2048, 4096 |

The default output size is **1,024 dimensions** for both models. Their vector spaces are independent: do not mix embeddings across different models, revisions or preprocessing policies.

## Architecture

```text
Any HTTPS client ---> FastAPI Gateway (CPU / Bearer)
                           |
                       route(model)
                       /          \
           Qwen worker             WeMM worker
             (L40S)                   (L40S)
               |                        |
           Qwen Volume               WeMM Volume
               |                        |
   Qwen3-VL-Embedding-8B      Tencent WeMM-Embedding-9B
```

Each GPU worker loads only its assigned model; each worker scales to zero independently.

## Requirements

- **Local mock:** Python 3.11+, Git, curl and Bash (WSL on Windows). No GPU, Modal account or API token required.
- **Hosted notebook client:** A Python 3 CPU runtime on Kaggle/Colab, Internet access and an already deployed HTTPS API URL with its application Bearer token. No Modal CLI or model weights required.
- **Self-hosted GPU inference:** The local prerequisites above, a Modal account with L40S and storage quota, access to the official Hugging Face model repositories and a secure secret manager for credentials.

## Quickstart: local contract tests

From a Bash terminal (including Google Colab or Kaggle), run:

```bash
git clone https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference.git
cd Multimodal-Embeddings-AI-Inference
bash scripts/setup.sh
bash scripts/quickstart_mock.sh
```

**What setup does:** `scripts/setup.sh` creates a Python virtual environment, installs development dependencies, compiles Python sources, runs Ruff and executes the available pytest suite. It is a contributor/development setup and can produce substantial output; it does **not** start inference or use a GPU. After setup, `scripts/quickstart_mock.sh` is the short local contract smoke test.

Already cloned the repo? Skip `git clone` and `cd`; run the last two commands from the repository root. The smoke script creates one temporary token, starts the local mock API, verifies an authenticated embedding response, and stops the server. No second terminal or manual token setup is needed.

A passing test shows `"pass": true` and `"backend": "mock"`. This is **not real Qwen3-VL-Embedding-8B/Tencent WeMM-Embedding-9B inference**; testing a deployed Modal endpoint requires its separate HTTPS URL and API Bearer token.

For the CPU dependency versions used by setup and CI, see [constraints/cpu.txt](constraints/cpu.txt). The install path and GPU dependencies are documented in [Contributing](CONTRIBUTING.md).

## Interactive notebook for Kaggle and Google Colab

The [bilingual English/Vietnamese API notebook](notebooks/multimodal_embeddings_api_demo.ipynb) runs on CPU and does not manage Modal resources.

- **Required:** The HTTPS endpoint and Bearer token of a deployment you operate or are authorized to access. The built-in preview works without an API URL or token.
- **Run:** Select Qwen3-VL-Embedding-8B (default) or Tencent WeMM-Embedding-9B, run **Model configuration** and the collapsed **Configuration helpers** cell, optionally edit the three text/image pairs in **Customize samples**, preview the bundled images, then enter credentials in **API connection**.
- **Result:** Six validated requests and an annotated 3 × 3 cosine matrix. For a single-request implementation without the notebook helpers, use the [minimal Python HTTP client](docs/CLIENT_INTEGRATION.md#python-311-standard-library). See the [English notebook guide](docs/NOTEBOOK_GUIDE.md) or [Vietnamese guide](docs/NOTEBOOK_GUIDE.vi.md).

**For contributors:** The notebook code cells also have [reviewable Python sources](notebooks/source/) synchronized by `python scripts/sync_notebooks.py --check`. The public notebooks remain standalone. The demo includes an empty report-layout preview, not fabricated similarity results.

**Open the API notebook:** [Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_api_demo.ipynb) · [GitHub notebook](notebooks/multimodal_embeddings_api_demo.ipynb). Kaggle users can import the notebook file.

## Inference performance benchmark

The [benchmark notebook](notebooks/multimodal_embeddings_benchmark.ipynb) compares serving behavior for Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B.

- **Measures:** Balanced warm latency, opt-in cold idle trials and alternating GPU worker routing; not retrieval quality or recognition accuracy.
- **Requires:** Your own authenticated HTTPS inference deployment; no dataset or local GPU.
- **By default:** `RUN_BENCHMARK=False` sends no requests. Enabling it with safe defaults sends **24 POST requests** (8 warm-ups + 16 measured text/image calls; at most 4 samples per model/modality, a smoke check rather than reliable p95) without cold waits. See the [benchmark protocol](benchmarks/README.md) for configuration and limitations.

**Reruns and Stop:** Interrupted phases cannot resume in the same run. Start a new run at **Configuration → API client/preflight**. See [benchmark recovery and limitations](benchmarks/README.md).

**Open the benchmark notebook:** [Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_benchmark.ipynb) · [GitHub notebook](notebooks/multimodal_embeddings_benchmark.ipynb).

## Deploy real inference on Modal

Requires your own Modal account, L40S/storage quota and access to both model repositories. Downloads and inference can incur charges.

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m pip install -e '.[modal]'
modal setup
modal run modal_models.py --selection both
```

Before deploying, follow [Configure authentication](docs/OPERATIONS.md#configure-authentication) to create the application Secret using the downloaded immutable revisions. That guide also explains how to load the owner-only token file without displaying it.

```bash
modal deploy modal_app.py
```

Use your deployment's HTTPS URL. After loading the token and pinned revisions from the owner-only file described above, verify both models with an authorized image:

```bash
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png --dimension 1024
```

The script reads expected checkpoint SHAs from `VISION_EMBEDDINGS_QWEN8_REVISION` and `VISION_EMBEDDINGS_WEMM9_REVISION`; alternatively pass `--qwen-revision` and `--wemm-revision`. Missing or mismatched revisions stop verification. HTTP redirects are rejected.

Success means a `PASS` record for each model with `revision_verified: true` and normalized vectors. See [Operations](docs/OPERATIONS.md) for token rotation, revocation, redeployment and client onboarding; see [Verification](docs/VERIFICATION.md#acceptance-source-provenance) to retain attributable evidence.

## REST endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/v1/health` | Gateway liveness only |
| `GET` | `/v1/models` | Available models and revisions |
| `GET` | `/v1/models/{model_id}` | One model specification |
| `POST` | `/v1/embeddings` | Encode one image or text |
| `POST` | `/v1/embeddings/batches` | Encode a bounded batch |

### Example response (abbreviated)

Illustrative, abbreviated JSON—not a captured API result. The vector and latency numbers are invented; a real 1,024-dimensional vector has 1,024 entries.

```json
{
  "model": "wemm-embedding-9b",
  "revision": "<checkpoint-revision>",
  "dimension": 1024,
  "vector": [0.0123, -0.0456, 0.0789],
  "normalized": true,
  "input_sha256": "<computed SHA-256 of the input>",
  "elapsed_ms": 152.4,
  "embedding_metadata": {
    "config_sha256": "<embedding configuration SHA-256>"
  }
}
```

See the [complete API schema](docs/API_CONTRACT.md).

## Task-specific embedding instructions

**Qwen3-VL-Embedding-8B** accepts an optional instruction for semantic document search, image retrieval and multimodal RAG. When omitted, the API applies its default Qwen instruction, “Represent the image or text for retrieval in a visual gallery.” Reproducible vectors require the same instruction, model revision and preprocessing policy.
**Tencent WeMM-Embedding-9B** does not accept the custom `instruction` field in this API; sending it returns HTTP 422. The response includes hashed encoding provenance without exposing the raw instruction. See the [API contract](docs/API_CONTRACT.md).

## Integration, privacy and limitations

- See [Python, Node.js and C#/.NET integration examples](docs/CLIENT_INTEGRATION.md) for runnable HTTP clients.
- In production, keep tokens in the backend. For personal Kaggle/Colab notebooks, enter them via hidden prompts in a private runtime; clear outputs before sharing. Never log raw images, Base64 inputs, embeddings or authorization headers.
- The service has no built-in rate limiting or spending cap. Restrict callers and configure platform-level protection before wider exposure.
- Cosine similarity **is not a calibrated confidence score**. Validate retrieval quality with representative labeled examples.

## Documentation

> **Testing and verification:** The repository includes automated API tests, inference validation tools, and post-deployment verification guidance. See [Verification documentation](docs/VERIFICATION.md) for details.

- [Operations](docs/OPERATIONS.md) · [Verification](docs/VERIFICATION.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [HTTP clients](docs/CLIENT_INTEGRATION.md)
- [API contract](docs/API_CONTRACT.md) · [Model Volumes](docs/MODEL_VOLUMES.md) · [Security](SECURITY.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [License notes](LICENSE-NOTES.md)
- [Notebook guide](docs/NOTEBOOK_GUIDE.md) · [Support](.github/SUPPORT.md) · [Code of conduct](.github/CODE_OF_CONDUCT.md) · [Open an issue](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/issues/new/choose)

## License and maintainer

The project's own code and documentation are MIT-licensed; third-party model weights and dependencies retain their own terms. No model weights or user datasets are distributed here.

**Maintainer:** Đăng Khoa · `i.am@dangkhoa.dev`.
