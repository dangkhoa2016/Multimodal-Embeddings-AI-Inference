# Changelog

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## 1.0.0 — initial source version

Initial source version of Multimodal Embeddings AI Inference.

- Provide authenticated image and text embedding APIs using Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B.
- Serve each checkpoint on a dedicated Modal NVIDIA L40S worker with its own model Volume.
- Include setup, checkpoint management, deployment, security controls, language-agnostic HTTP client examples and bilingual documentation.
- Provide reproducible, authenticated HTTPS smoke-test procedures for both models, including checkpoint-revision checks and vector normalization.
- Provide Python 3.11/3.12 CI, API contract tests, linting and a source SHA256 integrity manifest.
- Include a bilingual six-request API Demo Notebook with built-in sample previews available without credentials and a separate opt-in performance notebook.
- Support balanced warm latency, optional GPU cold trials, alternating worker routing, bounded concurrency and mixed-model batches. No measured benchmark statistics are published here.

For deployment-specific checks, see [Verification and acceptance](docs/VERIFICATION.md). This source release does not claim measured facial identity accuracy or production-level performance.
