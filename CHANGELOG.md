# Changelog

## 0.2.1 — 2026-11-04

- Deployed authenticated FastAPI embeddings on a shared NVIDIA L40S worker with isolated Qwen and WeMM runtimes.
- Verified real GPU inference, Qwen → WeMM → Qwen model switching, and authenticated HTTPS embedding requests for both models.
- Added bilingual operations and verification documentation, API contract tests, linting and source integrity checks.
- Hardened public-source defaults by disabling interactive OpenAPI documentation and removing deployment-specific identifiers.
- Clarified that image retrieval research evidence does not establish biometric accuracy or production readiness.

## 0.2.0 — 2026-11-04

- Introduced independent Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B inference adapters.
- Implemented one Modal L40S GPU worker with model subprocess switching and a shared Volume of pinned checkpoints.
- Added authenticated API routes, deterministic local mocks, setup scripts and Rails 8 integration examples.
