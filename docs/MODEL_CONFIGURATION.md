# Per-model request configuration

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](MODEL_CONFIGURATION.vi.md)

The embedding API supports optional model-specific overrides through the `model_config` JSON object. Leaving it out preserves the existing defaults (1024 output dimensions and the earlier Qwen instruction). The legacy top-level `dimension` and Qwen `instruction` fields remain supported.

Consult authenticated `GET /v1/models/{model_id}` for `model_config_schema.properties`. It advertises supported option names, types, constraints, and defaults so clients can generate forms without hard-coded model settings.

## Available options

| Option | Qwen/Qwen3-VL-Embedding-8B | tencent/WeMM-Embedding-9B |
| --- | --- | --- |
| `dimension` | Any integer 64–4096 | 64, 128, 256, 512, 1024, 2048, 4096 |
| `instruction` | One-line text, 1–512 characters | Unsupported |
| `normalize` | Boolean, default true | Boolean, default true |
| `max_length` | Token limit 64–32768, default 8192 | Unsupported by this adapter |
| `min_pixels`, `max_pixels` | Image processor pixel limits | Image processor pixel limits |

Image options apply **only** to images. Large pixel/token limits may increase GPU cost and memory use; a valid request is not a guarantee it will fit the device. Arbitrary model loader, GPU, dtype, filesystem and checkpoint kwargs are never accepted.

### Single request

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "How can I reset my password?",
  "model_config": {
    "dimension": 2048,
    "instruction": "Retrieve helpful support articles.",
    "max_length": 4096,
    "normalize": true
  }
}
```

### Batch with different model settings

```json
{
  "items": [
    {"model": "qwen3-vl-embedding-8b", "text": "a landscape", "model_config": {"dimension": 2048, "instruction": "Retrieve similar photographs."}},
    {"model": "wemm-embedding-9b", "text": "a landscape", "model_config": {"dimension": 1024, "normalize": true}}
  ]
}
```

Identical values supplied both as top-level fields and inside `model_config` are accepted. Conflicting values, unsupported options, invalid types and unsupported dimensions return HTTP 422 **before** GPU dispatch. Both synchronous and asynchronous endpoints support this object; async `request_id` idempotency includes the effective option profile.

When `normalize=false`, vectors are finite but are **not** guaranteed to have unit L2 norm; the response reports `normalized=false`. The result's `embedding_metadata.config_sha256` includes all effective nondefault options, so cache by `(input_sha256, config_sha256)` and keep indexes separated by encoding profile.

The standalone Demo/Benchmark notebooks continue to use reproducible default configurations. Advanced overrides may be submitted by authenticated HTTP clients; the notebooks never implicitly authorize billable GPU jobs. These option paths are tested with CPU mock/CI and still require actual GPU acceptance testing before quality or performance claims.
