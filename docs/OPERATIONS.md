# Modal operations and release verification

This is the operational runbook for deploying **CompactFaces AI Inference** from a clean machine. It deliberately does not contain credentials.

## Scope

- One public HTTPS FastAPI gateway with application-level Bearer authentication.
- One L40S GPU class, `max_containers=1` and `min_containers=0`, processing requests serially per container.
- Two pinned checkpoint snapshots on the shared Modal Volume `compactfaces-ai-weights`.
- Exactly one GPU model subprocess active at any time; changing models incurs a reload.

## Setup

```bash
bash scripts/setup.sh
.venv/bin/python -m pip install -e '.[modal]'
modal setup
```

For Python environments without `ensurepip`, `scripts/setup.sh` uses `uv` when installed. No GPU is needed locally.

## Download and verify checkpoint snapshots

```bash
modal run modal_models.py --selection both
modal volume ls compactfaces-ai-weights /checkpoints
```

Pin those exact Git revisions; do not blindly substitute `main` when reproducing a benchmark.

- Qwen3-VL-Embedding-8B: `2c4565515e0f265c6511776e7193b22c0968ddc7`
- Tencent WeMM-Embedding-9B: `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`

## Configure authentication

```bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482
```

The local token copy is kept in `~/.config/compactfaces-ai-inference/modal-secret.env` with owner-only permissions. Never echo the token, commit that file, or publish the secret. Rotate credentials before deploying into an untrusted environment.

## Deploy

```bash
modal deploy modal_app.py
```

Obtain the HTTPS deployment URL from Modal and configure it outside version control.

The URL can differ by workspace and environment. Store it in Rails settings, not hardcoded business logic.

## Acceptance checks

```bash
.venv/bin/ruff check src modal_app.py modal_models.py scripts tests
.venv/bin/python -m pytest -q
python scripts/smoke_modal_gpu.py --image /path/to/authorized_image.png --switch-back
```

The GPU smoke script invokes the existing deployed Modal class directly and checks each output for 1,024 finite dimensions and L2 norm near 1. It uses a synthetic image, not a human biometric record. For an HTTP-path test, see `scripts/smoke_both.py` and the main README.

Do not mark either model production-ready until that specific model has a passing real-GPU smoke test. A successful deploy or mocked contract test is insufficient.

## Logs, scaling and costs

```bash
modal app logs compactfaces-ai-inference --since 30m --tail 100
modal app list
modal volume ls compactfaces-ai-weights /checkpoints
```

The L40S worker is configured to scale to zero. Model switching unloads the old model and reloads the new one; avoid interleaved batches when latency matters. Account for GPU, CPU, RAM, volume and network charges. The Model Supervisor logs library tracebacks to private Modal logs while exposing generic errors to public API clients.

## Publishing policy

Commit only source, configuration examples, docs, and tests. Never commit model weights, dataset imagery, `.env`, the generated token, local caches or downloadable archives. After deployment, validate CI and both real model paths before declaring a release.
