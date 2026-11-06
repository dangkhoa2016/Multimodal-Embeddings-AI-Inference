# Inference performance benchmark

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

This directory documents the [performance benchmark notebook](../notebooks/multimodal_embeddings_benchmark.ipynb) for the authenticated Multimodal Embeddings API. A Python 3 **CPU** notebook calls an existing HTTPS deployment serving [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) on dedicated NVIDIA L40S workers. No local GPU, CUDA runtime, checkpoint download or Modal CLI is required.

**Models under test:** Qwen3-VL-Embedding-8B (Qwen3, about 8 billion parameters) and Tencent WeMM-Embedding-9B (Tencent, about 9 billion parameters) turn text and images into numerical vectors. This notebook measures API serving time, not vector quality or face-recognition accuracy.

## Run on Kaggle or Google Colab

1. Use a Python 3 CPU notebook with Internet enabled: [open in Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_benchmark.ipynb) or import the [benchmark notebook](../notebooks/multimodal_embeddings_benchmark.ipynb) into Kaggle.
2. **Enter your deployment's actual settings** for gateway/GPU scaledown and `min_containers` in the notebook. The prefilled **gateway 60s / GPU 180s / min-containers 0** values are editable examples from one deployment, **not universal Modal defaults**. Avoid competing traffic during cold idle. Ordinary API timings cannot automatically verify the deployment's autoscaler settings.
3. Safe defaults are `RUN_BENCHMARK=False`, `RUN_COLD=False`, `RUN_WARM=True`, `WARM_REPETITIONS=4` and `WARMUP_REQUESTS=1`: enabling only `RUN_BENCHMARK` triggers 24 warm-up/measurement POST requests total (both modalities warmed up) and no cold idle waits. For a cold-only rerun, explicitly set `RUN_COLD=True`, `RUN_WARM=False` and `RUN_ALTERNATE=False`, then `RUN_BENCHMARK = True`. Supply `VISION_EMBEDDINGS_BASE_URL` (HTTPS base URL without `/v1`) and `VISION_EMBEDDINGS_API_TOKEN` (application-level Bearer token), or enter them using hidden prompts. The defaults do **not** send requests.
4. Run all cells and inspect the notebook summary and the generated `multimodal_embeddings_performance_summary.json`. No API credential, raw vector, URL or input content is stored in this report.

**First-run order:** Set `RUN_BENCHMARK=True` in Configuration **before** the API client/preflight cell, then run the enabled phases and Report. Saved default `False` sends no POSTs. If the client was run with the flag off, rerun both Configuration and client/preflight after enabling it.

**Mixed-batch attempt journal:** Each mixed-model batch attempt is written to the redacted JSON as `in_progress` before its POST. A caught KeyboardInterrupt marks it `interrupted` and preserves earlier completed attempts; forced UI termination may leave the last attempt `in_progress` without proving server acceptance. Neither state is a confirmed measurement and no POST is retried.

**Concurrent attempt journal:** Each scheduled request is recorded as `in_progress` before dispatch. A caught KeyboardInterrupt immediately saves an `INCOMPLETE_INTERRUPTED` report, cancels work that has not started, and retains validated responses from requests already in flight without scheduling replacements. Forced UI termination can leave an `in_progress` attempt; this does not prove server acceptance or completion. The burst wall time includes notebook dispatch and journal-writing overhead.

**Rerun safety:** One API client/preflight execution creates a new run ID and a SHA-256 fingerprint of the measurement configuration and input digests. Rerunning any already-started phase, including an interrupted phase, or changing settings/inputs without redoing preflight is rejected before another POST. Caught KeyboardInterrupt writes an INCOMPLETE_INTERRUPTED report; forced notebook cancellation may leave IN_PROGRESS until a rerun is rejected. Already-dispatched requests may still execute on the server. Start a fresh run in order: Configuration → API client/preflight → enabled phases → Report. Final reports validate expected trial counts; no stale or duplicated samples are counted. Run IDs/fingerprints are not proof of server source identity.

**Provenance:** Optional BENCHMARK_CLIENT_SOURCE_SHA and BENCHMARK_SERVER_SOURCE_SHA accept 40-character Git commits, identified as operator-provided rather than API-verified. The embedded notebook template SHA256 is checked by repository tests, not re-attested after user edits. Success and failure JSON reports include source attribution and redacted measurement configuration without tokens, endpoints, private images or raw instructions.

## Balanced protocol

| Case | Qwen3-VL-Embedding-8B | Tencent WeMM-Embedding-9B | Accepted when |
| --- | ---: | ---: | --- |
| GPU cold | Configurable attempts and computed idle time | Same count and idle time | `gpu_worker_first_request=true` and `model_state=initial_load` |
| Warm-up | Requests per modality × 2 modalities × 2 blocks | Same number | Excluded |
| Warm text | Configurable equal count | Same count | `model_state=warm` |
| Warm image | Configurable equal count | Same count | `model_state=warm` |
| Alternating routes | Configurable attempts | Same number | `model_state=warm` |

Both models receive identical text, identical PNG bytes, equal vector dimensions (1,024) and one item per request. Cold runs alternate model order by repetition and **do not probe either function** during the calculated idle interval. Warm runs use two blocks with reversed order (Qwen3-VL-Embedding-8B→Tencent WeMM-Embedding-9B, Tencent WeMM-Embedding-9B→Qwen3-VL-Embedding-8B), with a text and an image warm-up before the measured randomized modality sequences. Defaults per model: two text and two image warm-ups, followed by eight measured requests. The report counts both blocks. Alternating calls measure warm routing across two dedicated GPU workers. No measured POST is retried; unconfirmed/failed observations cannot silently become warm retry samples.

**Important:** The calculated waiting interval may allow Modal to scale down, but sleep is not proof. The notebook computes `COLD_IDLE_WAIT_SECONDS = max(GATEWAY_SCALEDOWN_SECONDS, GPU_SCALEDOWN_SECONDS) + COLD_WINDOW_PADDING_SECONDS`. Example: gateway 50s, GPU 120s and 30s padding produce 150s per trial. The GPU worker reports its first-request indicator; the API gateway's cold state **cannot be independently confirmed** from the current API. Client time includes HTTP/network, gateway processing, GPU scheduling, model load and inference. The gateway cold/provisioning component is not isolated. The notebook displays a **calculated estimate of total idle waiting time** from the configured repetition count, number of models and per-trial idle interval; processing time is additional and GPU usage may be billed.

## Optional concurrent and mixed-model measurements

The notebook keeps cold, warm and alternating requests **sequential**. Cold trials are disabled by default; enable them only after confirming deployment idle settings and costs. Two additional phases are **disabled by default**: set `RUN_CONCURRENT=True` for bounded HTTP request bursts, or `RUN_MIXED_BATCH=True` for one item per model using `/v1/embeddings/batches`. They run only when `RUN_BENCHMARK=True`. Concurrency is capped at four client workers and 16 requests per model; mixed batches at ten repetitions. Every response checks `embedding_metadata`, `instruction_sha256` and `config_sha256`. `QWEN_INSTRUCTION_MODE="default"` omits instruction; `custom` is available only for Qwen3-VL-Embedding-8B. Tencent WeMM-Embedding-9B never receives instruction.

Alternating warm routing requires different, stable `gpu_worker_id` values for the two models. Concurrent trials require valid GPU instrumentation (model state, first-request flag, batch size, nonnegative timing fields), while GPU worker ID remains optional. Only validated completed trials count toward the observed **short-burst** items/second; client wall-clock latency is measured separately; a batch records one HTTP wall time and actual size. Neither establishes steady-state capacity or GPU autoscaling limits.

Even concurrent-only or mixed-batch-only runs record each verified model revision and embedding configuration; inconsistent revisions or configurations cause failed attempts instead of a success-only report.

## Results and boundaries

The JSON records all trial attempts, UTC timestamps, actual idle duration, GPU state, optional random worker IDs (if supported by the API), and confirmed/unconfirmed status, model revisions, sample hashes and descriptive p50/p95/mean for confirmed samples only. Core cold/warm/alternating results measure **sequential endpoint latency**. Optional concurrent bursts and two-model batches provide bounded API serving observations, **not sustained throughput**. These phases do not measure model semantic quality, face verification, retrieval accuracy, cross-modal ranking, GPU utilization, deployment SLAs or production readiness. Never mix embeddings from Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B in one vector space. Use separate benchmark datasets for downstream retrieval-quality assessment. On a failed HTTP request, metadata check or routing validation, the notebook atomically replaces the prior JSON report with a redacted INCOMPLETE_FAILED partial report, then stops without scheduling later phases or finalizing aggregate statistics. For concurrency, at most the already-dispatched bounded in-flight POSTs may finish and are included; pending/new work is not submitted. No POST is retried. A run in progress replaces the previous report with an IN_PROGRESS marker before contacting the API.

**Sample-size caution:** With the default warm-only configuration, 24 POSTs include 8 warm-ups and 16 measured requests, at most 4 measurements per model/modality. The rendered table includes `Latency n` and the exported JSON contains `sample_size_context`. A small-sample p95 is descriptive, not a production tail-latency estimate.

There are no published measured performance numbers from this notebook yet.
