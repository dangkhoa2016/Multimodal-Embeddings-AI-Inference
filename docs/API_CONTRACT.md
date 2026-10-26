# API contract

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](API_CONTRACT.vi.md)

Use the endpoint assigned to **your own deployment**; this repository does not host a shared public inference endpoint.

Public `GET /health` and `GET /v1/models` need no token and do not invoke GPUs. All other `/v1/*` endpoints, including model detail, jobs, and inference, require `Authorization: Bearer <API_TOKEN>`. By default this is the single owner token (`VISION_EMBEDDINGS_API_TOKEN`); operators may optionally configure SHA-256 API key digests per client. Use TLS outside localhost and never expose tokens in public notebooks.

## Read-only endpoints

- `GET /health` (**public**): gateway liveness and version, **not** GPU model readiness.
- `GET /v1/ops/metrics`: **owner Bearer token only**; per-gateway, process-local
  counters for HTTP statuses, observed async job states, uptime, and
  selected middleware rejection reasons. Non-owner clients receive HTTP 404;
  unauthenticated clients receive 401/lockout. No tokens, raw URL paths,
  request IDs, client identities, images, text, or embeddings are included.
  `async_job_state_observations` are **not** live global GPU job counts;
  counters reset on container restart and have `Cache-Control: no-store`.
  See [Security](SECURITY.md).
- `GET /v1/models` (**public**): supported model IDs, output dimensions, pinned checkpoint revisions and each model's model_config_schema (option names, limits and defaults).
- `GET /v1/models/{model_id}` (**authenticated**): metadata for one model or HTTP 404.

## POST /v1/embeddings

Supply exactly one of `image_base64` (PNG/JPEG/WebP) and `text` (UTF-8). Provide `model`; optionally specify `dimension` and `request_id` (up to 128 characters). For text, the schema strips leading/trailing whitespace before inference; input_sha256 hashes the UTF-8 bytes of the stripped text, not the original text. For example, two spaces surrounding hello world do not change its text digest. For images, SHA256 hashes the decoded image bytes, unaffected by text normalization.

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "a landscape photograph",
  "dimension": 1024,
  "request_id": "request-example-001"
}
```

Illustrative response (the shortened vector is **not** a real model result):

```json
{
  "model": "qwen3-vl-embedding-8b",
  "repository": "Qwen/Qwen3-VL-Embedding-8B",
  "revision": "<PINNED_MODEL_REVISION>",
  "backend": "modal",
  "dimension": 1024,
  "vector": [0.001, -0.002],
  "input_sha256": "<SHA256_OF_INPUT>",
  "request_id": "request-example-001",
  "elapsed_ms": 123.0,
  "timing": {"model_state": "warm", "gpu_worker_first_request": false, "model_load_ms": 0.0, "gpu_encode_ms": 76.0, "gpu_worker_ms": 80.0, "gpu_dispatch_overhead_ms": 3.0, "batch_size": 1},
  "embedding_metadata": {"preprocessing_version": "qwen3-vl-adapter-v1", "instruction_sha256": "<HASH>", "config_sha256": "<HASH>"},
  "normalized": true
}
```

With the default `normalize=true`, response vectors contain exactly `dimension` finite numbers and have L2 norm approximately 1. `normalize=false` returns finite vectors without guaranteeing unit norm. `elapsed_ms` reports API inference processing for the request or grouped batch, **not** end-to-end network latency.

## Per-model configurable inference

The optional `model_config` object allows per-request overrides. The list of
allowed options is advertised by each model's
`GET /v1/models/{model_id}` → `model_config_schema.properties`.
It supports output dimension, optional normalization, image pixel limits,
and Qwen-specific instruction / max_length. Arbitrary unknown model parameters
are rejected with HTTP 422, never silently passed to GPU. Existing
top-level `dimension` and `instruction` remain supported. Refer to the
[per-model configuration guide](MODEL_CONFIGURATION.md) for examples and limits.

## Optional Qwen instruction and encoding metadata

For **Qwen3-VL-Embedding-8B** only, the optional `instruction` field is a one-line task description of 1–512 characters. Empty, control-character or overlong values return HTTP 422. The model list advertises `supports_custom_instruction`.

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "How can I reset my password?",
  "dimension": 1024,
  "instruction": "Retrieve help articles that answer the user's question."
}
```

When omitted, the service preserves its previous gallery-retrieval Qwen instruction, including the resulting vector. WeMM does not accept custom instructions; supplying one returns HTTP 422.

The additive `embedding_metadata` response object contains `preprocessing_version`, `instruction_sha256` (null for WeMM) and `config_sha256`. The raw instruction is not repeated in the response. The original `input_sha256` continues to hash only the input, so it is not a unique embedding cache key.

Use the model, revision, dimension and preprocessing version to separate vector indexes. Use `(input_sha256, config_sha256)` together for caching individual embeddings. Query and document instructions may intentionally differ; identical instruction hashes are not a prerequisite for similarity. Re-embed affected indexed items if their encoding configuration changes.

## Performance timing breakdown

Authenticated inference responses preserve the original elapsed_ms (gateway-side inference wait) and now include optional timing data when the Modal GPU worker supports instrumentation. Older clients can ignore timing; mock/local backends return null.

- model_state: initial_load (first observed model on that worker), model_reload (restarting a dead process), or warm (same process reused). gpu_worker_first_request indicates the first request observed by that GPU container. gpu_worker_id is an optional opaque random 32-character hex identifier for one GPU worker lifetime on updated deployments, not a hostname, container ID, or proof of autoscaling. Older deployments may omit it.
- model_load_ms: elapsed time for subprocess startup, checkpoint/model initialization, and replacement. It is 0 on a warm request. gpu_encode_ms: subprocess batch preprocessing and vector computation time.
- gpu_worker_ms: time spent in the Modal GPU method; gpu_dispatch_overhead_ms: gateway-to-worker round-trip minus worker time, including scheduling, transport, and possible container provisioning. It is **not pure cold-start time**.
- batch_size states how many inputs share these measurements. Timing data is diagnostic, not a latency SLA; for full client latency, measure a wall-clock timer in Kaggle, Colab, or Rails.

All fields are measured in milliseconds. A first observed model request may be warm if the service is already running; concurrent clients may affect state and timings.

## Model identifiers and dimensions

- [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B): `qwen3-vl-embedding-8b`, integer dimensions 64–4096.
- [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B): `wemm-embedding-9b`, dimensions 64, 128, 256, 512, 1024, 2048 or 4096.

Omitting `dimension` or sending `null` selects the default 1024 dimensions. Sending `0` or an unsupported dimension returns HTTP 422.

Different model IDs, revisions, dimensions and preprocessing configurations produce noninterchangeable vectors.

## Async GPU jobs (cold starts and resume)

Use **POST /v1/embeddings/async** when GPU scheduling or model loading can exceed
the synchronous HTTP deadline. It accepts one normal embedding request and
requires a stable `request_id` (8–128 ASCII letters, digits, `_`, `-`, `.`,
or `:`; the first character must be alphanumeric).

- HTTP **202** returns `job_id`, `state=QUEUED` or `RUNNING`, and
  `Retry-After: 3`. This is *not* a completed embedding.
- Poll **GET /v1/embeddings/jobs/{job_id}** with a currently valid Bearer token;
  after token rotation, use the new token and keep the same job ID.
- `SUCCEEDED` returns HTTP **200** with `result` in the same format as
  `POST /v1/embeddings`. `FAILED` is terminal; never retry automatically.
- When the async POST acknowledgement is missing, use **GET
  /v1/embeddings/jobs/by-request/{request_id}** with the same authenticated
  client key; the server derives the tenant-scoped job ID without dispatching
  GPU work. HTTP 404 may mean the job never existed **or** the record expired.
  The demo notebook uses this GET first, then a legacy owner-scoped GET if a
  pre-upgrade server does not support it. Manually resubmitting the same ID
  and profile is deduplicated **only while the backend record exists**; an
  expired record can start new billable GPU work. Never automatically POST
  after losing the ACK. A reused ID with different model, input, dimension
  or instruction / model_config returns **409**.
- New submissions retain a short-lived gateway admission slot until dispatch
  acknowledgement and durable global/per-client outstanding-job permits until
  definitive completion. A shared Redis/Valkey rate budget (when configured)
  applies once to new work; polling and identical accepted async IDs do not
  consume another distributed embedding unit. This is not a monetary cap.
- Jobs are bound to the accepted model revision and encoding profile. A
  coordinator with a different profile fails the job before GPU dispatch.
- The detached CPU coordinator waits for L40S scheduling while the client
  polls. It has a finite 3,600-second execution limit; jobs can still fail.
- The coordinator updates a heartbeat about every 30 seconds. A `RUNNING`
  job with no heartbeat for over 180 seconds, or a `QUEUED` job older than
  24 hours, is reported as `INDETERMINATE` (HTTP **200**) with a warning.
  This is a **read-only advisory**, not proof that GPU execution stopped.
  The underlying record remains reserved; a later success can be recovered
  via the same job ID. Do not automatically create another request ID.

A Modal Dict (default `multimodal-embeddings-async-jobs`; configurable via
`VISION_EMBEDDINGS_JOB_DICT_NAME`) stores status and completed vectors,
but not raw input bytes or tokens. Modal Dict records expire after seven days
of inactivity. Copy completed embeddings into your own durable vector store.
Job IDs and vectors are sensitive to your dataset; keep responses private.
A disconnected client does not automatically stop a submitted GPU job.
The `INDETERMINATE` response is reversible if a coordinator resumes or
finishes. Inspect the Modal logs/job store before any manual attempt to
recover work; no automatic retry, re-dispatch or deletion occurs.

The synchronous endpoint remains unchanged and **does not** provide
idempotency. Its `request_id` is a correlation label only. The async
route is optional and currently handles one embedding at a time.

A lost GPU dispatch/result acknowledgement is also exposed as `INDETERMINATE`.
A cancellation acknowledgement only confirms receipt of a cancellation request;
it does not prove that GPU execution stopped. The coordinator retains the
underlying `RUNNING` record and both outstanding-job permits, even when a
cancellation request succeeds. Confirmed pre-dispatch rejection or a completed
result can release permits. Inspect the provider before manual reconciliation;
never automatically retry an uncertain job.

### Loaded checkpoint verification

For Modal inference, the service requires the GPU worker's model ID, repository
and 40-character checkpoint SHA to match the pinned request revision before
accepting vectors. The subprocess captures this identity from the verified
snapshot marker at model load; warm responses retain that identity. The worker
checks the expected revision before encoding. Missing identity, an unpinned
revision or a mismatch fails the request; async jobs become `FAILED` without a
result. Legacy workers without this protocol are rejected. This is snapshot
metadata verification, not an independent attestation of weight-file contents.

## POST /v1/embeddings/batches

```json
{"items": [
  {"model":"qwen3-vl-embedding-8b", "text":"one", "dimension":1024},
  {"model":"wemm-embedding-9b", "text":"two", "dimension":1024}
]}
```

Returns `{"count": 2, "items": [...]}` in input order. The gateway groups requests by model and dispatches independent model groups to separate GPU workers. The default batch limit is **8** (schema maximum **32**).

## Shared rate limiting (optional)

The reference single-gateway deployment uses a local request budget. When
a shared Redis/Valkey limiter is configured, all gateway containers atomically
debit global and per-client **embedding-unit** token buckets using Redis server
time. A batch of four spends four units; the async route charges only new
jobs, not authenticated GET polling or repeat reads of an existing job.
Independent synchronous GPU requests also share expiring concurrency
leases across gateways. Rate/concurrency/body overload responds with HTTP
**429** and `Retry-After`; a shared backend failure responds with **503**,
never a fallback to a weaker local-only budget. Rate buckets refill evenly
over 60 seconds and allow bursts up to their capacity. Synchronous calls
remain non-idempotent even when they carry a `request_id`.

## Errors and input limits

- HTTP 401: Bearer token missing or invalid.
- HTTP 404: model identifier not found in model metadata route.
- HTTP 413: request body or image exceeds configured limits.
- HTTP 422: invalid Base64/image, dimension, model, batch size or request schema.
- HTTP 429: temporary authentication lockout or a per-gateway inference unit/concurrency limit; `Retry-After` provides the delay. The `detail` field identifies authentication lockout (`Too many failed authentication attempts`) versus inference limits.
- HTTP 503: inference worker unavailable; private error details are not returned to the client.
- HTTP 504: the Modal inference deadline expired, including GPU scheduling time; remote cancellation is requested and `Retry-After` is provided.

API tokens contain **16–4089 visible ASCII characters**. By default, three failed authentications within five minutes lock the ASGI peer for 15 minutes; all credentials from that peer are blocked until expiry. See [authentication lockouts](OPERATIONS.md#failed-bearer-token-attempts-and-temporary-lockouts).

Defaults: **5 MiB** per image, **12 million** decoded pixels, **8** items per batch and **8192** text characters. Images must be JPEG, PNG or WebP; remote image URLs are not fetched.

Default: 120 embedding units/minute and 4 concurrent inference requests per gateway container. Batch items count individually. Limits reset on container restart; they are not global quotas or spending caps.

For the Modal backend, the default inference deadline is **60 seconds**, configurable through `VISION_EMBEDDINGS_INFERENCE_TIMEOUT_SECONDS` (> 0, maximum 120). A timeout releases admission and requests cancellation of that request's GPU calls; cancellation cleanup waits at most 5 additional seconds. Request IDs are correlation labels, not idempotency keys. Do not automatically resend a POST after a timeout.

## Compatibility

The path version remains `/v1`. Reindex if the model revision, embedding preprocessing or dimension changes; cache embeddings using the pair `(input_sha256, embedding_metadata.config_sha256)`.
