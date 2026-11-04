# Troubleshooting and deployment validation

| Symptom | Meaning | Action |
|---|---|---|
| `401` | Wrong/missing app Bearer | Check Rails secret equals local `CF_API_TOKEN`; do not paste into logs |
| `503` first GPU call | Loader failed, OOM, dependency mismatch, missing snapshot | Inspect protected Modal app/function logs; verify model/revision directory |
| `GPU model worker exited unexpectedly` | Child crashed during load/inference | Review `/tmp/compactfaces-gpu-worker.log` via protected Modal logs/shell; no public tracebacks |
| `Missing or mismatched checkpoint` | Volume incomplete or wrong SHA in Modal Secret | Rerun `modal run modal_models.py --selection both` with selected SHA, then update secret and deploy |
| Qwen import/version error | Official upstream code changed or missing dependency | Record upstream git commit and pinned requirements; rebuild worker image deliberately |
| WeMM model import error | Transformers/qwen-vl-utils mismatch | Confirm `transformers==5.2.0` in isolated WeMM environment |
| Long response time | Cold start or model-switch reload | Group requests by model, lower batch size, measure cold/warm separately |
| GPU OOM | BF16 weights + vision activations exceed memory | Reduce image resolution/batch, use one active process; record peak VRAM; don't assume 48 GB always sufficient |
| `413/422` | Unsupported image/dimension/batch | Use allowed PNG/JPEG/WebP, max bytes/pixels and model dimension policy |
| Vector size wrong | Invalid model output or stale client | Fail closed; do not pad/truncate without model-specific evaluation |

## Suggested acceptance gates

1. **CPU/unit:** `pytest -q`, all tests PASS, no secrets/model weights tracked.
2. **Prefetch:** both snapshot markers exist in ONE Volume; SHA values recorded.
3. **Deployment:** endpoint URL exists, missing token gets `401`, valid token returns `/v1/models`.
4. **Model A:** Qwen 8B produces 1024 finite L2-normalized values for a permitted PNG.
5. **Model B:** WeMM 9B produces 1024 finite L2-normalized values for the SAME PNG.
6. **Switching:** A → B → A succeeds on one L40S; record load time and peak VRAM.
7. **Failure:** invalid image rejected; missing snapshot or failed model gives sanitized error.
8. **Benchmark:** real CompactFaces dataset validation and independently scoped research metrics; avoid biometric accuracy overclaims.

**No GPU evidence is included with the offline release.** Attach a dated report with exact commit SHA, model revisions, GPU, memory peak and timing before publishing benchmark claims.
