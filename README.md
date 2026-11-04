# CompactFaces AI Inference

**A research-oriented multimodal embedding REST API using Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B on one Modal NVIDIA L40S.** Built for [CompactFaces Synthetic Face Dataset](https://github.com/dangkhoa2016/CompactFaces-Synthetic-Face-Pipeline) and a future Rails 8 evaluation application.

**Language:** English | [Tiếng Việt](README.vi.md)

> **Release status — v0.2.1:** Deployed on Modal L40S; both Qwen3-VL-Embedding-8B and WeMM-Embedding-9B **passed real image inference** with finite 1,024-dimensional normalized vectors. Gateway authentication (401/200), Ruff and 26 offline tests also passed. This establishes API/model execution, **not face-recognition accuracy or a biometric benchmark**. Mock vectors must never be used for AI evaluation.

**Live verification evidence:** [2026-11-04 REST/GPU results](docs/VERIFICATION.md).

## What it does

- Exposes authenticated `GET /v1/health`, `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/embeddings`, `POST /v1/embeddings/batches` endpoints.
- Takes **image bytes as Base64** or UTF-8 text and returns L2-normalized numeric vectors, content SHA256, model ID, revision and elapsed time.
- Uses **one CPU FastAPI gateway** and **one GPU `SharedL40SGpu` Modal class** (`gpu="L40S"`, `max_containers=1`); it switches between isolated model subprocesses instead of loading both simultaneously into VRAM.
- Stores both model snapshots on **one persistent Modal Volume** named `compactfaces-ai-weights`.
- Provides offline mock-mode tests, Modal prefetch/deploy scripts, secure token setup, CI and Rails integration examples.

**This is an image-retrieval experiment, not a validated biometric identification system.** CompactFaces synthetic identity labels do not prove facial identity consistency, and multimodal image embedding similarity is not equivalent to biometric identity verification.

## Models

| API model ID | Official Hugging Face repository | Nominal size | Supported output dims |
|---|---|---:|---|
| `qwen3-vl-embedding-8b` | [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 8B | 64–4096 (MRL) |
| `wemm-embedding-9b` | [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) | 9B | 64, 128, 256, 512, 1024, 2048, 4096 |

Tencent has **no official WeMM-Embedding-8B checkpoint** in its published model lineup; we use the closest official larger model, 9B. Both default to **1024 output dimensions**. Embeddings from different models must be stored and searched separately; do not compare their vectors directly.

## Architecture

```text
Rails 8 (future) -- HTTPS / Bearer ---> FastAPI Gateway (CPU)
                                          |
                                          v
                                SharedL40SGpu (L40S 48 GB)
                               max_containers=1, min=0
                                          |
                              +-----------+------------+
                              | subprocess supervisor |
                              +-----------+------------+
                                   /              \
                        isolated Qwen env    isolated WeMM env
                        Qwen 8B *or*        Tencent 9B
                        (never simultaneously resident)
                                          |
                                          v
                              Modal Volume (checkpoints)
```

The first request for a model incurs load time; switching models unloads the previous model. The repository **does not claim** two BF16 models can co-reside on 48 GB VRAM. CUDA memory, cold starts, image sizes and batch limits must be verified on Modal.

## Prerequisites

- Python **3.11+** and Git on your local machine.
- A Modal account with **L40S access and sufficient budget/quota**.
- Internet access to Hugging Face for one-time checkpoint downloads.
- A trusted machine to create and store the API token. You do not need a local GPU for tests or deployment commands.

## Quickstart A — local API + offline tests (no GPU, no model downloads)

```bash
bash scripts/setup.sh
export CF_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
CF_BACKEND=mock .venv/bin/uvicorn compactfaces_ai_inference.api:local_app \
  --factory --host 127.0.0.1 --port 8000
```

In a second terminal, using the same `CF_API_TOKEN`:

```bash
curl -H "Authorization: Bearer $CF_API_TOKEN" http://127.0.0.1:8000/v1/models
python scripts/smoke_api.py --base-url http://127.0.0.1:8000 \
  --model qwen3-vl-embedding-8b --text "synthetic portrait" --dimension 1024
```

**MOCK is deterministic fake data** for contract tests. It is unsuitable for similarity/accuracy reports and never starts a GPU.

## Quickstart B — Modal deploy, step by step

### 1. Install and authenticate Modal

```bash
source .venv/bin/activate
python -m pip install -e '.[modal]'
modal setup
```

If using Windows, run these commands in WSL/Linux/macOS or an equivalent Bash environment.

### 2. Download BOTH snapshots to ONE Volume (CPU job)

```bash
modal run modal_models.py --selection both
modal volume ls compactfaces-ai-weights /checkpoints
```

The first command prints **two real Hugging Face commit SHA values**. Retain them for the next step. Download time and storage depend on available bandwidth and Modal resources. The model weights are **not** bundled in Git.

### 3. Set the Modal Secret and Bearer token

Use the exact 40-character commit SHAs printed by step 2:

```bash
python scripts/create_modal_secret.py \
  --qwen8-revision <QWEN_40_CHARACTER_SHA> \
  --wemm9-revision <WEMM_40_CHARACTER_SHA>
```

This helper creates a cryptographically random `CF_API_TOKEN`, stores it with **0600 file permissions** at `~/.config/compactfaces-ai-inference/modal-secret.env`, and uploads `CF_API_TOKEN`, `CF_QWEN8_REVISION` and `CF_WEMM9_REVISION` to the Modal Secret named **`compactfaces-api`**. It never prints the token. Never commit or publish this local file. To replace the existing Modal secret while preserving your local token, run the helper with `--force`.

Alternatively, use `modal secret create --from-dotenv` with a properly protected file. For public Hugging Face repositories, an HF token is not normally required; do not place personal access tokens in source control.

### 4. Deploy API Gateway + single L40S worker

```bash
modal deploy modal_app.py
```

Modal prints the actual FastAPI HTTPS endpoint. Record **the exact URL**; it is not predictable from this README. Model loading is lazy, so deployment success by itself is not proof that weights load or inference works.

### 5. Test both REAL models with an authorized image

Retrieve the token locally without printing it:

```bash
export CF_API_TOKEN="$(python3 - <<'PY'
from pathlib import Path
p = Path.home()/'.config/compactfaces-ai-inference/modal-secret.env'
print(next(x.partition('=')[2] for x in p.read_text().splitlines() if x.startswith('CF_API_TOKEN=')))
PY
)"
```

Then invoke the smoke test with an image you have the right to use:

```bash
python scripts/smoke_both.py \
  --base-url 'https://YOUR-REAL-DEPLOYED-URL.modal.run' \
  --image /path/to/authorized_image.png --dimension 1024
```

The script rejects `mock` as a successful remote test, checks expected model IDs, vector lengths, L2 norm and SHA256. Run additional tests for cold/warm loads, model switching and GPU memory. **Do not claim FULL PASS until both models produce actual vectors.**

### 6. Connect Rails 8

Store only the gateway URL and `CF_API_TOKEN` in Rails encrypted credentials (or your secret manager). Use server-side requests to `/v1/models`, `/v1/embeddings` and `/v1/embeddings/batches`. See [Rails integration](docs/RAILS_INTEGRATION.md). Never put the Bearer token in browser JavaScript.

## API example

```bash
curl -X POST 'https://YOUR-REAL-DEPLOYED-URL.modal.run/v1/embeddings' \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"wemm-embedding-9b","text":"a smiling face","dimension":1024}'
```

For images, use `image_base64` (plain Base64, not a `data:` URI). The response contains `vector`, `dimension`, `model`, `repository`, `revision`, `backend`, `input_sha256`, `elapsed_ms`. See [API contract](docs/API_CONTRACT.md). Endpoint authorization is application-level Bearer; Modal Proxy Authentication is **not** enabled in this release, to keep Rails integration as URL + one token.

## Operations and costs

- GPU configuration: Modal NVIDIA L40S (48 GB), **at most one container** at once, **scales to zero** when idle. This caps simultaneous GPU containers for this deployment, not the total accumulated bill.
- All requests may incur GPU cold-start and model load latency. Avoid alternating models for every image; group batches by model.
- The API only supports synchronous requests today (maximum eight items per batch by default). Long-running 935-image runs should be split into small Rails/Solid Queue jobs with retry and checkpoint handling.
- **No model training is performed.** Runtime results do not support face-identification accuracy claims until you independently validate the dataset and benchmark protocol.
- Modal Volume and CPU gateway usage may incur charges independently of GPU. Set Modal budget alerts and review usage before exposing a public endpoint.

See [GitHub publication guide](docs/PUBLISH_GITHUB.md) for creating the actual remote repository.

See [Volume lifecycle](docs/SHARED_VOLUME.md), [deployment troubleshooting](docs/TROUBLESHOOTING.md), [security](SECURITY.md), and [model/data license boundaries](LICENSE-NOTES.md).

## Repository structure

```text
compactfaces-ai-inference/
├── src/compactfaces_ai_inference/       # FastAPI, validation, engines, GPU process manager
├── modal_app.py                         # One L40S GPU class and CPU API gateway
├── modal_models.py                      # Two checkpoints -> one Modal Volume
├── scripts/                             # Setup, secrets, API smoke tests
├── tests/                               # Offline tests, including subprocess switching
├── docs/                                # API contract, Rails, Volume, operations
├── .github/workflows/ci.yml            # CPU-only CI
├── .env.example                         # Safe placeholders only
├── README.md / README.vi.md             # Bilingual publication
└── LICENSE / LICENSE-NOTES.md
```

## Contributing, license and provenance

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md). Original repository-owned source/docs are MIT licensed. Pretrained model weights and upstream code are **not included** and retain their own licenses. CompactFaces dataset images are distributed separately under their stated noncommercial conditions; the code MIT license does not grant image usage rights. See [LICENSE-NOTES.md](LICENSE-NOTES.md).

**Maintainer:** Đăng Khoa · contact: `i.am@dangkhoa.dev`.

See also: [Modal operations and release verification](docs/OPERATIONS.md).
