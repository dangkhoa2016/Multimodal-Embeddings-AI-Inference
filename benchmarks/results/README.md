# Historical GPU measurements — 2026-11-07

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

These results were captured from a real Modal deployment on **2026-11-07 UTC**, using Qwen/Qwen3-VL-Embedding-8B and tencent/WeMM-Embedding-9B. The performance notebook's code cells ran on a Linux CPU client with Python 3.13.18, calling the remote HTTPS API.

The deployment used one L40S worker per model, one CPU gateway with 16 concurrent HTTP inputs, a four-request inference limit and a 60-second inference deadline. Checkpoint revisions and source content digests are recorded in the JSON files. GPU hardware and source identity are operator configuration; response timing does not independently attest them.

Git commit IDs were not captured for this run. The serving and overload JSON files record the runtime source files' SHA-256 digests for comparison with the measured snapshot. The reports contain no API credentials, deployment URLs or raw embedding vectors.

## Warm latency

The [serving report](serving.json) contains **35 POST requests / 38 embedding items**: 8 warm-ups, 16 measured warm requests, 8 concurrent requests and 3 two-model batches. Embeddings use 1,024 dimensions. Inputs are the notebook's public text and deterministic 128 × 128 RGB gradient PNG.

| Model | Input | Samples | Client p50 (ms) | Client p95 (ms) | GPU encode p50 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-Embedding-8B | Text | 4 | 2680.75 | 2697.13 | 25.77 |
| Qwen3-VL-Embedding-8B | Image | 4 | 3535.12 | 3887.78 | 40.34 |
| WeMM-Embedding-9B | Text | 4 | 2674.98 | 2720.28 | 128.90 |
| WeMM-Embedding-9B | Image | 4 | 3459.50 | 3483.03 | 148.19 |

Client time includes transport, gateway handling, remote dispatch and inference. GPU encode time is the component reported by the worker. Each p95 describes only four observations in this run.

## Concurrent requests and mixed batches

Each model received a separate four-request burst with two client workers:

| Model | Completed / attempted | Client workers | Burst (ms) | Client p50 (ms) | Client p95 (ms) | Observed items/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-Embedding-8B | 4/4 | 2 | 5636.87 | 2703.76 | 2948.13 | 0.71 |
| WeMM-Embedding-9B | 4/4 | 2 | 5566.91 | 2696.45 | 2832.68 | 0.72 |

The three mixed-model batches each contained one item per model:

| Completed / attempted | Validated items | Client p50 (ms) | Client p95 (ms) | Mean (ms) | Observed items/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3/3 | 6 | 2902.64 | 2918.23 | 2903.40 | 0.69 |

Burst duration includes client dispatch and journal writes. Mixed-batch latency covers a whole HTTP batch; its items/s divides validated items by summed batch wall time. These measurements describe short runs, with the selected client concurrency and inputs. See the [benchmark guide](../README.md) for calculation and classification rules.

## Ten-request overload check

A separate [overload report](overload.json) records ten simultaneous text POSTs with 64-dimensional output and no automatic POST retries:

| HTTP status | Requests | Client elapsed range (s) |
| --- | ---: | ---: |
| 200 | 4 | 19.391–28.215 |
| 429 | 6 | 1.377–1.791 |

Health returned HTTP 200 in **1.108 seconds** during the burst. Provider call graphs showed four successful GPU calls and no GPU calls for the six rejected requests. The admitted requests included two initial model loads. This check exercises admission and health responsiveness; automated regression tests separately cover deadline expiry and remote-call cancellation.

## Text-to-image search

The [image-search report](image-search.json) uses the three bundled WebP samples, Qwen3-VL-Embedding-8B and 1,024 dimensions. Three image POSTs created one reusable index; two text POSTs queried it:

| Query | Top image | Cosine |
| --- | --- | ---: |
| an orange gift box with a ribbon | gift.webp | 0.7600 |
| a blue sphere | sphere.webp | 0.6648 |

This is a small historical retrieval example. Evaluate ranking quality on representative data, using the same model revision and encoding settings. The [API demo notebook](../../notebooks/multimodal_embeddings_api_demo.ipynb) includes the three sample images; the [benchmark guide](../README.md) explains how to measure your own deployment.
