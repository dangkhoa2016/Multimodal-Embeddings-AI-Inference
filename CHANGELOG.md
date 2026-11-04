# Changelog

## 0.2.0 — 2026-11-04

- Select two officially available large multimodal embedding models: Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B.
- Route both through **one Modal L40S GPU worker** using process-level model switching.
- Share one Modal Volume for both pinned checkpoint downloads.
- Add owner-only local secret generation and Modal CLI upload helper.
- Add offline model-switching tests, local REST contract tests, GitHub Actions CI and bilingual documentation.

**Limit:** GPU inference, VRAM use and cold-start performance have not been validated on Modal.
