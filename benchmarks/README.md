# Inference performance benchmark

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

This directory documents the [performance benchmark notebook](../notebooks/multimodal_embeddings_benchmark.ipynb) for the authenticated Multimodal Embeddings API. A Python 3 **CPU** notebook calls an existing HTTPS deployment serving [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) on dedicated NVIDIA L40S workers. No local GPU, CUDA runtime, checkpoint download or Modal CLI is required.

**Models under test:** Qwen3-VL-Embedding-8B (Qwen3, about 8 billion parameters) and Tencent WeMM-Embedding-9B (Tencent, about 9 billion parameters) turn text and images into numerical vectors. This notebook measures API serving time, not vector quality or face-recognition accuracy.

See the [GPU benchmark results](results/README.md) for warm latency, bounded concurrency, mixed-model batches and a ten-request overload check, with the original JSON reports. These are historical measurements of that deployment snapshot.

## Run on Kaggle or Google Colab

1. Use a Python 3 CPU notebook with Internet enabled: [open in Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_benchmark.ipynb) or import the [benchmark notebook](../notebooks/multimodal_embeddings_benchmark.ipynb) into Kaggle.
2. **Enter your deployment's actual settings** for gateway/GPU scaledown and `min_containers` in the notebook. The prefilled **gateway 60s / GPU 180s / min-containers 0** values are editable examples from one deployment, **not universal Modal defaults**. Avoid competing traffic during cold idle. Ordinary API timings cannot automatically verify the deployment's autoscaler settings.
3. Safe defaults are `RUN_BENCHMARK=False`, `BENCHMARK_MODE="warm"`, `WARM_REPETITIONS=4` and `WARMUP_REQUESTS=1`: enabling only `RUN_BENCHMARK` triggers up to 24 warm-up/measurement POST requests total (both modalities warmed up), without cold idle waits. For a cold-only run, select `BENCHMARK_MODE="cold"` and then `RUN_BENCHMARK=True`. The dropdown also offers `alternating`, `concurrent`, `mixed_batch`, and `full`, which runs all five in a single session. Supply `VISION_EMBEDDINGS_BASE_URL` (HTTPS base URL without `/v1`) and `VISION_EMBEDDINGS_API_TOKEN` (application-level Bearer token), or add both names to Kaggle Secrets. If one or both values are unavailable, the notebook automatically opens a hidden input prompt for each missing value when `RUN_BENCHMARK=True`. The defaults do **not** send requests.
4. Run all cells and inspect the notebook summary and the generated `multimodal_embeddings_performance_summary.json`. No API credential, raw vector, URL or input content is stored in this report.

**Choose a benchmark mode:** The form exposes one `BENCHMARK_MODE` dropdown. Selecting `full` runs all five phases sequentially in one session and one JSON report; other selections run only the chosen phase. Internal phase flags are derived automatically.

| Mode | What it measures | Example POST cap (defaults) |
| --- | --- | ---: |
| `warm` (default) | Text/image latency after warm-up | 24 (8 warm-up + 16 measured) |
| `cold` | Async job cold-start after idle, polled with GET | 10 (5/model; GET polls excluded) |
| `alternating` | Routing latency when switching Qwen/WeMM | 13 (2 setup + up to 1 refresh + 10 measured) |
| `concurrent` | Async GPU preparation + bounded sync burst | 10 (2 preparation + 8 measured) |
| `mixed_batch` | Async GPU preparation before each two-model batch | up to 12 POST / 15 embedding items |
| `full` | Cold → warm → alternating → concurrent → mixed_batch | **69 POST / 72 embedding items** |

These are **request limits for the example configuration**, not cost or speed estimates. Full mode also has **35 minutes of calculated cold idle** with the example scale-down settings, *plus* API/GPU execution time; cold work may be billed. Confirm both `min_containers=0` and actual scale-down settings. The selected mode is recorded in the JSON report. Full mode stops at the first failed phase and preserves a partial report; do not rerun or automatically retry a possibly dispatched POST. To run a different mode, start a fresh session/preflight rather than rerunning a completed phase.

The Configuration cell previews the planned maximum HTTP POST attempts, with warm-up
and measured trials separated, mixed-batch item counts, and calculated cold idle
waiting time. This bilingual preview sends no requests; actual attempts may stop
early on error. It does not estimate GPU processing time or monetary cost.

**Cold async protocol:** Each cold trial records a stable random `async_request_id` before sending **one** `POST /v1/embeddings/async`. The notebook polls `GET /v1/embeddings/jobs/{job_id}` until `SUCCEEDED`, `FAILED`, `INDETERMINATE` or a deadline. Set `COLD_JOB_TIMEOUT_SECONDS` (default 1800) and `COLD_POLL_INTERVAL_SECONDS` (default 15); at most 120 GET polls per trial at these defaults, **additional to** the 69 planned full-mode POSTs. A lost POST acknowledgement can be investigated with the read-only `GET /v1/embeddings/jobs/by-request/{request_id}`; never automatically retry the POST. A job registry must be operational.

**Async warm-up and GPU readiness:** Each text/image warm-up, alternating setup, concurrent setup and per-batch mixed-model setup now submits one journaled `POST /v1/embeddings/async`, followed by bounded GET polling. They reuse `COLD_JOB_TIMEOUT_SECONDS` and `COLD_POLL_INTERVAL_SECONDS` to keep the form small. Measured warm, alternating, concurrent and mixed-batch requests stay on the real synchronous consumer endpoints. A confirmed warm measurement requires `model_state=warm`, first-request false and the same opaque GPU worker ID as its recent preparation. A stale model observed after two sequential preparations permits at most one separately journaled, fresh async readiness request per model pair; if either worker remains stale, missing or changed during measurement, the notebook fails closed. This is new preparation, never a retry of a possibly dispatched POST or an unbounded keep-alive. Gateway `/health` alone does not prove GPU readiness. Declared scale-down values are safety assumptions, not verified autoscaler telemetry. Default full workload cap: 69 POSTs (10 cold, 8 warm-up, 16 warm measured, 2 alternating setup, up to 1 alternating readiness refresh, 10 alternating measured, 2 concurrent setup, 8 concurrent measured, 6 mixed-batch setup, up to 3 mixed-batch readiness refreshes and 3 batches); GET polling is additional.

**Timing evidence:** Cold `client_elapsed_ms` includes job submission, queue/provisioning, GPU model loading, execution, polling and transport. The report records `async_ack_elapsed_ms` and `async_after_ack_ms`; GPU model load comes from validated worker timing. **Standalone GPU allocation wait is not exposed by the job API** (`gpu_allocation_wait_ms=null`). Failure labels such as `http_504_unconfirmed_cause`, `async_indeterminate` and `async_job_wait_timeout_unconfirmed` require Modal logs to determine the cause. Full mode stops safely after a failed cold trial and does not resend GPU POSTs.

**Connection behavior:** The notebook checks environment variables, then Kaggle Secrets, then hidden interactive prompts for **only missing values**. It does not prompt or access secrets when `RUN_BENCHMARK=False`. For unattended Kaggle Save & Run All, authorize both secrets first; an unattended runtime may not support interactive prompts. If credential entry fails, no benchmark POST is sent, and the previous report is not overwritten. Never place a Bearer token in notebook source or outputs.

**First-run order:** Set `RUN_BENCHMARK=True` in Configuration **before** the API client/preflight cell, then run **Benchmark configuration → API client/preflight → Request and validation helpers → enabled phases → Report** in order. Saved default `False` sends no POSTs. If the client was run with the flag off, rerun both Configuration and client/preflight after enabling it.

**Sequential attempt journal:** Each cold, warm-up, warm or alternating attempt is saved as `in_progress` before dispatch and updated in place after its response. A caught Stop preserves the attempt even during response bookkeeping; a forced stop may retain the pending marker. Pending markers do not prove server acceptance, and no POST is retried.

**Mixed-batch attempt journal:** Each mixed-model batch attempt is written to the redacted JSON as `in_progress` before its POST. A caught KeyboardInterrupt marks it `interrupted` and preserves earlier completed attempts; forced UI termination may leave the last attempt `in_progress` without proving server acceptance. Neither state is a confirmed measurement and no POST is retried.

**Concurrent attempt journal:** Each scheduled request is recorded as `in_progress` before dispatch. A caught KeyboardInterrupt immediately saves an `INCOMPLETE_INTERRUPTED` report, cancels work that has not started, and retains validated responses from requests already in flight without scheduling replacements. Forced UI termination can leave an `in_progress` attempt; this does not prove server acceptance or completion. The burst wall time includes notebook dispatch and journal-writing overhead.

**Rerun safety:** One API client/preflight execution creates a new run ID and a SHA-256 fingerprint of the measurement configuration and input digests. Rerunning any already-started phase, including an interrupted phase, or changing settings/inputs without redoing preflight is rejected before another POST. Caught KeyboardInterrupt writes an INCOMPLETE_INTERRUPTED report; forced notebook cancellation may leave IN_PROGRESS until a rerun is rejected. Already-dispatched requests may still execute on the server. Start a fresh run in order: Benchmark configuration → API client/preflight → Request and validation helpers → enabled phases → Report. Final reports validate expected trial counts; no stale or duplicated samples are counted. Run IDs/fingerprints are not proof of server source identity.

**Provenance:** Optional BENCHMARK_CLIENT_SOURCE_SHA and BENCHMARK_SERVER_SOURCE_SHA accept 40-character Git commits, identified as operator-provided rather than API-verified. The embedded notebook template SHA256 is checked by repository tests, not re-attested after user edits. Success and failure JSON reports include source attribution and redacted measurement configuration without tokens, endpoints, private images or raw instructions.

### Capturing reproducible deployment evidence

For a new measured run, record the **client Git commit** from the checkout used to export the notebook, the **server deployment commit** as verified by the operator, model checkpoint revisions and licenses, dependency lockfile digests, runtime Python/CUDA versions, relevant Modal worker settings, UTC session timestamps and input dataset SHA-256 values. Match the deployment file hashes to an archived source snapshot. A manually entered commit ID is a declaration, **not** independent server attestation; keep this distinction explicit in the report. The included November 2026 JSON remains unchanged and cannot be retrospectively upgraded to a verified source commit.

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

The notebook keeps cold, warm and alternating requests **sequential**. Cold trials are not the default; select `BENCHMARK_MODE="cold"` only after confirming deployment idle settings and costs. Two additional scenarios are available in the dropdown: select `BENCHMARK_MODE="concurrent"` for bounded HTTP request bursts, or `BENCHMARK_MODE="mixed_batch"` for one item per model using `/v1/embeddings/batches`. Only the selected scenario runs, and only when `RUN_BENCHMARK=True`. Concurrency is capped at four client workers and 16 requests per model; mixed batches at ten repetitions. Every response checks `embedding_metadata`, `instruction_sha256` and `config_sha256`. `QWEN_INSTRUCTION_MODE="default"` omits instruction; `custom` is available only for Qwen3-VL-Embedding-8B. Tencent WeMM-Embedding-9B never receives instruction.

Alternating warm routing requires different, stable `gpu_worker_id` values for the two models. Concurrent measured trials require validated GPU telemetry and a worker ID matching their recent preparation; unconfirmed warm states stop the phase. Only validated completed trials count toward the observed **short-burst** items/second; client wall-clock latency is measured separately; a batch records one HTTP wall time and actual size. Neither establishes steady-state capacity or GPU autoscaling limits.

Even concurrent-only or mixed-batch-only runs record each verified model revision and embedding configuration; inconsistent revisions or configurations cause failed attempts instead of a success-only report.

## Results and boundaries

The JSON records all trial attempts, UTC timestamps, actual idle duration, GPU state, optional random worker IDs (if supported by the API), and confirmed/unconfirmed status, model revisions, sample hashes and descriptive p50/p95/mean for confirmed samples only. Core cold/warm/alternating results measure **sequential endpoint latency**. Optional concurrent bursts and two-model batches provide bounded API serving observations, **not sustained throughput**. These phases do not measure model semantic quality, face verification, retrieval accuracy, cross-modal ranking, GPU utilization, deployment SLAs or production readiness. Never mix embeddings from Qwen3-VL-Embedding-8B and Tencent WeMM-Embedding-9B in one vector space. Use separate benchmark datasets for downstream retrieval-quality assessment. On a failed HTTP request, metadata check or routing validation, the notebook atomically replaces the prior JSON report with a redacted INCOMPLETE_FAILED partial report, then stops without scheduling later phases or finalizing aggregate statistics. For concurrency, at most the already-dispatched bounded in-flight POSTs may finish and are included; pending/new work is not submitted. No POST is retried. A run in progress replaces the previous report with an IN_PROGRESS marker before contacting the API.

**Sample-size caution:** With the default warm-only configuration, 24 POSTs include 8 warm-ups and 16 measured requests, at most 4 measurements per model/modality. The rendered table includes `Latency n` and the exported JSON contains `sample_size_context`. A small-sample p95 is descriptive, not a production tail-latency estimate.

Run the notebook against an authorized deployment to generate a report with its own measured latency, sample counts and configuration provenance.

## Offline text-to-image retrieval evaluation

Use [the offline evaluator](../scripts/evaluate_retrieval.py) to calculate
**hit@1, recall@K, and mean reciprocal rank (MRR)** from saved embeddings:

```bash
python scripts/evaluate_retrieval.py --input saved-embeddings.json --top-k 3
```

JSON declares `model_id`, pinned `revision` and `config_sha256` at dataset
level **and on every image/query record**. Each image has `id` and `vector`;
each query has `id`, `vector` and ground-truth `relevant_images` IDs.
The evaluator rejects mixed identities, invalid vectors, dimension mismatches
and unknown ground-truth IDs. It makes no API or GPU calls. Metrics describe
only the labeled dataset, not general semantic accuracy.
