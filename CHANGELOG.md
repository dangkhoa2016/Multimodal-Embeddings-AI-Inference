# Release notes

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## 1.0.0 — Initial release

**Multimodal Embeddings AI Inference 1.0.0** provides image and text embedding APIs, deployment tools, integration examples, and bilingual notebooks for multimodal search applications.

### API and models

- Supports [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B), each with an independent embedding space.
- Provides authenticated HTTPS endpoints for images, text, single and batch requests, custom Qwen instructions, and model-supported vector dimensions.
- Returns metadata to help applications verify model identity, embedding settings, and processing details.

### Security and asynchronous processing

- Supports async GPU jobs with `request_id`, status polling, read-only result recovery, and safeguards against accidental duplicate GPU requests.
- Offers optional per-client API keys, request rate limits, concurrency admission, and shared Redis/Valkey controls for multi-gateway deployments.
- Includes offline key management, operator-restricted metrics, and protections for invalid credentials, backend failures, and overdue jobs.

### Notebooks and integration guides

- **Bilingual Demo Notebook:** previews three built-in image/text pairs without credentials; explicitly enabled live inference produces a measured 3 × 3 cosine-similarity matrix.
- **Bilingual Benchmark Notebook:** measures warm latency, optional cold starts, alternating routing, bounded concurrency, and mixed-model batches; exports deployment- and configuration-specific JSON reports.
- Notebooks run as CPU-side API clients; potentially billable inference calls must be explicitly enabled.
- Includes HTTP client examples and English/Vietnamese integration, deployment, security, and troubleshooting guides.

### Deployment and verification

- Supports Modal deployment with a dedicated NVIDIA L40S GPU worker per model and Model Volumes for checkpoints.
- Includes a GPU-free local mock mode, automated tests, offline security checks, and SHA-256 manifests for source files and release assets.

**Benchmark scope:** Included GPU results reflect specific measurement environments and configurations. They do not guarantee production SLAs or embedding quality on user data.

**Get started:** See the [operations guide](docs/OPERATIONS.md), [notebook guide](docs/NOTEBOOK_GUIDE.md), and [license notes](LICENSE-NOTES.md).
