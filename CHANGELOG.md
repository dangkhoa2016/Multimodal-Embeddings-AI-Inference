# Changelog

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## 1.0.0

Image and text embeddings, deployment tools and notebooks for building multimodal applications.

- Provide authenticated image and text embedding APIs using Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B.
- Serve each checkpoint on a dedicated Modal NVIDIA L40S worker with its own model Volume.
- Include setup, checkpoint management, deployment, security controls, language-agnostic HTTP client examples and bilingual documentation.
- Support single and batch requests, custom Qwen instructions, and response metadata for tracking model and encoding settings.
- Add opt-in idempotent async GPU jobs with CPU coordination, persistent status/result polling and a resumable, cost-aware bilingual API demo notebook.
- Add coordinator heartbeats, read-only stale-job warnings, token-rotation-safe notebook IDs, tenant-scoped read-only request-ID recovery and accurate client-time/cache-hit reporting without automatic GPU retries.
- Support optional per-client API keys, persisted outstanding-job admission, and opt-in Redis/Valkey distributed rate and synchronous concurrency limits.
- Provide offline key lifecycle tools, owner-only privacy-bounded operational metrics, fail-closed backend errors and security regression audits.
- Bound Modal inference waits with cancellable deadlines, keep health responsive under load and reject malformed credentials safely.
- Enforce peer lockouts before credential checks and isolate authentication/body limits across mounted API apps.
- Sanitize notebook connection errors and persist sequential benchmark attempts before dispatch.
- Include a bilingual six-request API Demo Notebook with built-in sample previews available without credentials and a separate opt-in performance notebook.
- Support balanced warm latency, optional GPU cold trials, alternating worker routing, bounded concurrency and mixed-model batches. The notebook exports measured statistics for the operator's selected deployment and configuration.

See [Operations](docs/OPERATIONS.md) to deploy the service and the [notebook guide](docs/NOTEBOOK_GUIDE.md) to try the examples.
