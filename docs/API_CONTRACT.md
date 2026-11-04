# API contract

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](API_CONTRACT.vi.md)

Use the endpoint assigned to **your own deployment**; this repository does not host a shared public inference endpoint.

All `/v1/*` endpoints require `Authorization: Bearer <VISION_EMBEDDINGS_API_TOKEN>`. Use TLS outside localhost, and keep the token server-side.

## Read-only endpoints

- `GET /v1/health`: gateway liveness and version, **not** GPU model readiness.
- `GET /v1/models`: supported model IDs, output dimensions and pinned checkpoint revisions.
- `GET /v1/models/{model_id}`: metadata for one model or HTTP 404.

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

Real response vectors contain exactly `dimension` finite numbers and have L2 norm approximately 1. `elapsed_ms` reports API inference processing for the request or grouped batch, **not** end-to-end network latency.

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

## POST /v1/embeddings/batches

```json
{"items": [
  {"model":"qwen3-vl-embedding-8b", "text":"one", "dimension":1024},
  {"model":"wemm-embedding-9b", "text":"two", "dimension":1024}
]}
```

Returns `{"count": 2, "items": [...]}` in input order. The gateway groups requests by model and dispatches independent model groups to separate GPU workers. The default batch limit is **8** (schema maximum **32**).

## Errors and input limits

- HTTP 401: Bearer token missing or invalid.
- HTTP 404: model identifier not found in model metadata route.
- HTTP 413: request body or image exceeds configured limits.
- HTTP 422: invalid Base64/image, dimension, model, batch size or request schema.
- HTTP 503: inference worker unavailable; private error details are not returned to the client.

Defaults: **5 MiB** per image, **12 million** decoded pixels, **8** items per batch and **8192** text characters. Images must be JPEG, PNG or WebP; remote image URLs are not fetched.

## Compatibility

The path version remains `/v1`. Reindex if the model revision, embedding preprocessing or dimension changes; cache embeddings using the pair `(input_sha256, embedding_metadata.config_sha256)`.
