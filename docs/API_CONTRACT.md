# API Contract v1 — CompactFaces AI Inference

Base URL: `https://<Modal-provided-URL>.modal.run` (replace with actual URL after deployment).

Every endpoint requires `Authorization: Bearer <CF_API_TOKEN>`. TLS is required outside localhost. Authentication uses the app token from Modal Secret `compactfaces-api` (not your Modal administrative API token). This version **does not enable Modal Proxy Authentication**.

## Read-only endpoints

- `GET /v1/health`: gateway liveness only, returns `status`, `version`, `backend`, and a warning that model availability is not proven.
- `GET /v1/models`: two allowlisted model specs, nominal parameter size, supported dimensions, revision and `runtime_status: not_checked`.
- `GET /v1/models/{model_id}`: single spec or 404.

## POST /v1/embeddings

Send exactly one of `image_base64` (raw Base64 of PNG/JPEG/WebP) and `text` (UTF-8). `model` is required, `dimension` optional (default 1024), `request_id` optional, max 128 chars. No arbitrary image URL fetching.

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "a synthetic portrait",
  "dimension": 1024,
  "request_id": "local-example"
}
```

Response structure (vector truncated in this *schema example*, not a real embedding):

```json
{
  "model": "qwen3-vl-embedding-8b",
  "repository": "Qwen/Qwen3-VL-Embedding-8B",
  "revision": "<40-char-pinned-model-commit>",
  "backend": "modal",
  "dimension": 1024,
  "vector": [0.001, -0.002],
  "input_sha256": "<sha256-of-input-bytes>",
  "request_id": "local-example",
  "elapsed_ms": 123.0,
  "normalized": true
}
```

A real response includes **exactly `dimension` finite floating-point values** with L2 norm approximately 1.0. `elapsed_ms` is the measured end-to-end batch request time (not per-image GPU kernel time); in batch responses the same group-wide elapsed time is currently repeated for each item. `revision` comes from Modal Secret; pin to the exact snapshot commit. `backend=mock` means **fake vectors**, not real inference.

### Dimension policy

- `qwen3-vl-embedding-8b`: integer 64–4096.
- `wemm-embedding-9b`: only 64, 128, 256, 512, 1024, 2048, 4096.

Do not compare vectors of different model IDs, preprocessing policies, dimensions or checkpoint revisions directly.

## POST /v1/embeddings/batches

```json
{"items": [
  {"model":"wemm-embedding-9b", "text":"one", "dimension":1024},
  {"model":"qwen3-vl-embedding-8b", "text":"two", "dimension":1024}
]}
```

Returns `{ "count": 2, "items": [<full responses in original request order>] }`. The API groups work by model for fewer GPU switches and then restores input order. Max **8** items by default (up to **32** if CF_MAX_BATCH_SIZE is raised). Large embeddings produce large JSON responses; prefer modest batch sizes. For 935 images, the Rails caller should provide checkpointed jobs rather than one giant synchronous HTTP request.

## Errors and limits

- `401`: missing/incorrect Bearer token.
- `404`: unknown model in model-info path.
- `413`: image too large or request body exceeds configured size.
- `422`: malformed Base64/image, invalid dimension/model, missing input, batch over limit, invalid field.
- `503`: internal model inference unavailable; details intentionally hidden from the client, check protected Modal logs.

Defaults: **5 MiB per image**, **12 million decoded pixels**, **8 images per batch**, **8192-character text max**, allowlisted JPEG/PNG/WebP only. No image URLs, no browser-side tokens, no image persistence by the gateway.

## Compatibility

API version stays `/v1`. Adding new model IDs may be compatible; changing dimension policy, embedding preprocessing, response shape, or auth requires updated clients. Treat `model + revision + dimension + preprocessing configuration` as the embedding cache key, together with input SHA256.
