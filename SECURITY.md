# Security policy

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

Please report vulnerabilities privately to **i.am@dangkhoa.dev** rather than posting access credentials or private data in a public issue.

## Authentication and data handling

- All `/v1/*` endpoints require an application Bearer token (`VISION_EMBEDDINGS_API_TOKEN`). Keep it in your own deployment secret manager and never embed it in browser JavaScript.
- The API accepts Base64-encoded JPEG, PNG or WebP image bytes, or text. It does not fetch arbitrary input URLs.
- Input size and pixel limits reduce resource misuse. The gateway applies process-local request protections and optional per-client SHA-256-digest Bearer keys, tenant-isolated async jobs, durable outstanding-job permits, and optional Redis/Valkey distributed rate/concurrency limits (HTTP 429). These are **not** billing guarantees or a substitute for a Modal Workspace budget and spend limit. See [security implementation](docs/SECURITY.md) and [operations](docs/OPERATIONS.md).
- Uploaded image bytes are processed in memory. Do not log request bodies, raw images, tokens, authorization headers or sensitive embeddings.
- Interactive API documentation endpoints (`/docs`, `/redoc` and `/openapi.json`) are disabled in the deployed source.

The Modal deployment separates the gateway Secret (containing the API Bearer token) from the GPU-worker Secret (containing only public pinned model revisions). Do not attach gateway authentication credentials to model workers. Redeploying an existing service requires provisioning the worker Secret first; see [Operations](docs/OPERATIONS.md).

## Operational safeguards

- Keep model weights, private inputs and deployment credentials outside the source repository. Production clients must keep API tokens in a backend; personal hosted notebooks should use private runtimes and hidden prompts, with outputs cleared before sharing.
- Apply TLS, access controls and provider-level request/budget limits before opening the service to untrusted users.
- Review upstream model code, dependencies and licenses before redeploying or redistributing artifacts.
- A successful GPU inference test does **not** demonstrate accuracy for biometric identity verification. Treat such use cases as requiring independent evaluation, consent and privacy review.

This project provides a research-oriented inference API, not a production security certification.
