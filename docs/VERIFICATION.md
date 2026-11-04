# Inference verification — 2026-11-04

This document summarizes controlled execution evidence for CompactFaces AI Inference. It **does not measure facial identity recognition accuracy**.

## Environment and reproducibility

- Hardware: one NVIDIA L40S GPU, with a shared checkpoint Volume and sequential model switching.
- Input: an authorized synthetic CompactFaces image converted to 400 × 400 RGB PNG. Input SHA256: `f7f65de628690f58ff40770f0a1f689aacb7b77618ebc149f35c6db8e63283a7`. The image is not part of this source repository.
- Qwen model: `Qwen/Qwen3-VL-Embedding-8B`, revision `2c4565515e0f265c6511776e7193b22c0968ddc7`.
- Tencent model: `tencent/WeMM-Embedding-9B`, revision `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`.
- Qwen inference adapter source revision: `393e2978d27852b0d0230d6994f37f9c15bed73c`.

## Observed results

| Test | Outcome |
| --- | --- |
| Direct GPU invocation, Qwen → WeMM → Qwen | PASS; finite 1024-dimensional, L2-normalized vectors |
| Authenticated HTTPS `POST /v1/embeddings` with both models | PASS; `backend=modal`, expected revisions and matching input SHA256 |
| Authenticated and unauthenticated REST contract checks | 9/9 PASS, with expected HTTP 200, 401 and 404 statuses |
| Offline API tests and lint | PASS; one nonblocking Starlette deprecation warning |

The API-reported sample `elapsed_ms` values were **531.685 ms** (Qwen) and **11160.632 ms** (WeMM). They are individual request samples affected by model loading and switching, **not** end-to-end network times, p95 latency, throughput or a service-level guarantee.

## Interpretation and limits

The models run successfully but face-matching accuracy, negative-pair errors, recall, dataset-wide retrieval performance, sustained concurrency, peak VRAM and per-request costs remain unmeasured. Store embeddings and indexes separately for each model and revision. Synthetic identity labels do not validate biometric consistency. See [Operations](OPERATIONS.md) and [Security](../SECURITY.md).
