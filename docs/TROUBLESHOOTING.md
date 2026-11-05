# Troubleshooting and acceptance checks

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](TROUBLESHOOTING.vi.md)

## Common issues

| Symptom | Likely cause | Recommended action |
| --- | --- | --- |
| HTTP `401` | Missing or invalid Bearer token | Verify server-side `VISION_EMBEDDINGS_API_TOKEN` without printing its value |
| HTTP `413`/`422` | Oversized input or invalid schema | Check Base64, image limits, model ID, dimension and batch size |
| HTTP `503` | Inference worker unavailable | Inspect protected provider logs, checkpoint revision and available GPU memory |
| Model loading error | Dependency or checkpoint mismatch | Rebuild pinned environments and verify checkpoint integrity |
| High latency | GPU worker cold start or dispatch queueing | Batch by model and measure cold/warm requests separately |
| GPU out of memory | Model activations or oversized inputs | Reduce batch/image dimensions; keep only one model process active |
| Unexpected vector dimensions | Wrong model revision or client assumptions | Reject mismatches; never silently pad or truncate vectors |

## Acceptance checklist

1. Run offline tests, lint and the SHA256 source manifest check.
2. Download and verify both full model checkpoints: `Qwen/Qwen3-VL-Embedding-8B` and `tencent/WeMM-Embedding-9B`.
3. Confirm that unauthenticated API requests are denied, and valid authenticated catalog requests succeed.
4. Confirm both models return finite 1,024-dimensional, normalized vectors on the same authorized image.
5. Confirm model routing and sanitized error responses, and monitor GPU memory under load.
6. Measure retrieval quality independently before claiming accuracy or suitability for biometric identification.

For functional smoke-test procedures, see [Verification](VERIFICATION.md).
