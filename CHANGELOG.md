# Changelog

## 0.2.1 — 2026-11-04

- Deployed authenticated Modal FastAPI gateway and one NVIDIA L40S GPU class.
- Verified both official checkpoints from a shared Modal Volume, returning 1,024-dimensional finite, normalized embeddings with a real CompactFaces synthetic image.
- Fixed isolated subprocess dependencies: split GPU-safe vector utilities from FastAPI and added torchvision for WeMM Transformers 5.2.0 video processor initialization.
- Added explicit 8 CPU cores/64 GiB RAM, GPU diagnostic logs, source-only mount for rapid redeploy, and cross-model switch smoke script.
- Fixed local setup for environments without ensurepip, added bilingual operations runbooks and Ruff enforcement in CI.
- Verified Ruff clean and 26/26 offline tests. Biometric recognition accuracy remains unmeasured.

## 0.2.0 — 2026-11-04

- Select two officially available large multimodal embedding models: Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B.
- Route both through **one Modal L40S GPU worker** using process-level model switching.
- Share one Modal Volume for both pinned checkpoint downloads.
- Add owner-only local secret generation and Modal CLI upload helper.
- Add offline model-switching tests, local REST contract tests, GitHub Actions CI and bilingual documentation.

**Limit:** GPU inference, VRAM use and cold-start performance have not been validated on Modal.
