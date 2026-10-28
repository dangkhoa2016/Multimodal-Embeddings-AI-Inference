# Using the notebooks on Google Colab and Kaggle

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](NOTEBOOK_GUIDE.vi.md)

Explore image and text embeddings with the [bilingual English/Vietnamese API notebook](../notebooks/multimodal_embeddings_api_demo.ipynb). Choose [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) or [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B), try the bundled examples, and replace them with your own data.

## What you need

- A Python 3 **CPU** runtime on Colab or Kaggle with Internet enabled. The notebook includes its client code and sample images; you can open it directly without cloning the repository or installing model weights.
- For inference, the **HTTPS base URL and application Bearer token** of a deployed API. You can use your own [Modal deployment](OPERATIONS.md) or one you have permission to access.
- To preview the bundled images, just run the sample preparation steps below. Credentials are needed only when you connect to the API.

## Open the notebook

[Open in Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_api_demo.ipynb) or [download the notebook](../notebooks/multimodal_embeddings_api_demo.ipynb). On Kaggle, import that file into a Python notebook and enable Internet access.

**Unattended Kaggle Save & Run All:** With `RUN_API_DEMO=True`, configure the URL/token securely through environment variables first. Missing credentials require interactive hidden prompts; never hard-code tokens into a public notebook.

**Safe automated execution:** `RUN_API_DEMO=False` always runs preview only, without prompts, HTTP/GPU calls or fabricated cosine scores, **even when credentials are in the environment**. Set `RUN_API_DEMO=True` to enable the live API and potentially billable GPU work. Credentials can be entered through hidden prompts or preconfigured; no second approval switch is needed.

### Reverse proxies and Cloudflare

When using Cloudflare Worker or another reverse proxy, confirm that its upstream points to the intended Modal workspace before sending credentials or billable GPU requests. Both notebooks identify their requests with a dedicated User-Agent instead of the Python urllib default. This avoids certain client-signature blocks (such as Cloudflare error 1010), but does not bypass proxy access policy. Public GET /health and GET /v1/models do not send Bearer tokens; private API routes retain authentication. A successful public response cannot prove which upstream served it. For remaining HTTP 403 errors, inspect proxy rules and origin configuration. Do not retry uncertain GPU POSTs.

In Kaggle, run the named configuration and action cells from top to bottom. Some Jupyter viewers show long helper code expanded despite Colab folding metadata; these cells do not require editing. Review configuration with `RUN_API_DEMO=False` or `RUN_BENCHMARK=False` first; explicitly enable live requests only when you intend to use GPU inference.

## Run the demo

1. **Choose a model.** In **Model configuration**, choose `MODEL_ID`, `DIMENSION` and `TIMEOUT_SECONDS`. Run **Model configuration → Configuration helpers**. The default is Qwen3-VL-Embedding-8B with 1,024 dimensions. Colab provides form controls; on Kaggle, edit the values in the code cell.
2. **Prepare your examples.** Run **Customize samples → Prepare input samples**. The defaults pair three short descriptions with a green triangle, blue sphere and orange gift box. This step shows the bundled previews without calling the API.
3. **Connect your API.** Under **API connection**, run **Secure API connection → Check API connection**. Enter the base HTTPS URL without `/v1` and your API token in the hidden prompts.
4. **Generate and compare.** With `RUN_API_DEMO=True`, run **Generate embeddings → Results** for three texts and three images. Modal requests use resumable async jobs automatically, with no extra approval switch. Up to six potentially billable new GPU POSTs are possible if there is no reusable cache/job; previously validated results or pending jobs may reduce that count to zero.

You can stop after sample preparation to explore the preview, or continue to inference when you have connection details. API calls may start billable GPU workers in the deployment account.

## Use your own data

Edit `SAMPLE_TEXT_1` through `SAMPLE_TEXT_3` in **Customize samples**. To use your own images, set `IMAGE_PATH_1` through `IMAGE_PATH_3` to existing PNG, JPEG or WebP files in the runtime; leave a path empty to keep its bundled image.

For example, upload a file to Colab and use `/content/my-image.png`, or attach a Kaggle dataset and use `/kaggle/input/my-dataset/my-image.png`. These are example paths; choose files you are authorized to send to the API.

The built-in gallery is a reference only, **not proof of the submitted input**. After preparing custom files, the notebook shows a bilingual warning identifying each custom image slot, and Results repeats the warning. For privacy, custom pixels and file paths are never displayed automatically. A missing or unreadable custom image produces a message identifying the affected sample.

To preview your selected custom images, explicitly set `PREVIEW_CUSTOM_IMAGES=True`
in **Customize samples**, then rerun **Prepare input samples**. This adds their
pixels to notebook outputs; file paths stay hidden. Clear outputs before sharing.

## Read the results

The 3 × 3 matrix has **images in rows and texts in columns**. Entry `(i, j)` is `cosine(image_i, text_j)`. The diagonal compares samples with the same index; when you replace inputs, you decide whether those pairs describe the same subject.

Cosine scores describe vector similarity, not confidence percentages. The report also shows the model, checkpoint revision, vector dimension, L2 norms, instruction mode and request processing times. API processing time differs from the complete network round trip. Use the performance notebook below for structured latency measurements.

## Make changes or recover from an interruption

The final **What should I run next?** cell runs after Results and summarizes the latest state. In preview mode, it confirms that no HTTP/GPU calls were made in that preview run; an unresolved-job warning appears only if the private cache contains pending work. Rerunning this guidance cell sends no requests. The rerun table at the top of the notebook covers changes to samples, model and connection details.

If an inference request times out or is interrupted, check the deployment before starting another run: GPU work may still be running. The notebook does not automatically retry POST requests. For connection errors, see [Troubleshooting](TROUBLESHOOTING.md).

## Resume each embedding without duplicate GPU work

The Demo Notebook uses resumable async jobs for live Modal backends. `RUN_API_DEMO=True` is the single explicit opt-in for real API/GPU work. Validated vectors and job IDs are saved after each result, so repeated **Run All** runs reuse verified cached vectors or poll existing jobs, without a duplicate POST.

The local file `multimodal_embeddings_api_demo_cache.json` persists the
checkpoint between cell runs. It contains vectors, request IDs and job IDs,
**not raw images, API addresses or Bearer tokens**. Vectors can still be
sensitive: do not publish this file with your notebook outputs. Override its
path using `VISION_EMBEDDINGS_NOTEBOOK_CACHE`, or restore a private backup
when the runtime resets. If the file is lost, submitting **the same inputs** to the **same deployment**
reconstructs the same idempotent request ID, even after a Bearer-token rotation. The server can return an existing result **only while its backend job record remains available**. Modal Dict records are documented to expire after approximately seven days of inactivity. If the checkpoint and server record are both missing, the same request ID can trigger new, billable GPU work. Inspect the original job before starting new work. The deployment URL, model revision, dimension, instruction and
input digest remain part of the key. The notebook can also read older
credential-bound checkpoints from the same token/session after upgrading.
Never invent a new ID to retry a timeout.

**Lost async POST acknowledgement:** For an `UNCERTAIN` checkpoint, the
notebook first sends a **GET-only** lookup to
`/v1/embeddings/jobs/by-request/{request_id}`. The API derives the
authenticated client's job key, including for per-client keys after rotation.
If that route is missing on an older deployment (HTTP 404), the notebook
falls back to the legacy owner-scoped deterministic key using **GET only**.
No second GPU POST is sent from uncertain cached work. A missing/expired
job record does not authorize retry. For manual review, see
[controlled reconciliation](OPERATIONS.md#controlled-reconciliation-for-overdue-async-jobs).

**Legacy recovery after token rotation (read-only):** Keep the older private cache file. When a saved job uses the pre-upgrade credential-bound ID, set `VISION_EMBEDDINGS_LEGACY_REQUEST_ID_MAP` to a JSON object mapping the *current stable* `demo-...` request ID to its *old* `demo-...` cache key. Obtain both IDs from the configured notebook and your own saved cache; do not paste tokens or private inputs into the mapping. The old cache entry must already contain a completed result or a 64-character job ID. The notebook then uses the old ID and polls via **GET only**, without submitting new work; it fails closed if the mapped cache entry is missing or uncertain. A recovered vector is still checked against input digest, model, revision and dimension before use. An expired backend job with no completed local vector cannot be recovered by this mapping. Never guess an ID or change credentials merely to recreate it.

`QUEUED` or `RUNNING` jobs are polled for up to `MAX_POLL_SECONDS`
(default 900); rerun the cell to continue later. A `FAILED` job is not
retried automatically. `INDETERMINATE` means the dispatch or coordinator
heartbeat is overdue, **not** that the GPU job stopped; inspect Modal logs
and do not submit with a new ID automatically. Results distinguish cached
vectors (no newly measured client duration) from new/resumed jobs; client
wall time includes submission, network polling and wait, not GPU-only latency.
The notebook expects a Modal backend with async support. Update older
deployments before running the notebook rather than disabling idempotency.

**Acceptance testing after a redeploy:** A cached vector can still be valid with the same URL and model revision, but is **not proof that the new GPU deployment ran**. For a fresh GPU acceptance test, intentionally change one sample text or image (with permission) and confirm that the corresponding result is not marked as cached; check Modal GPU calls separately. Do not delete an uncertain/pending job journal to force another POST.

## Model settings

Qwen3-VL-Embedding-8B accepts integer dimensions 64–4096. Tencent WeMM-Embedding-9B accepts 64, 128, 256, 512, 1024, 2048 or 4096.

Use the default instruction for the bundled demo, or choose custom instruction mode for Qwen3-VL-Embedding-8B to describe your retrieval task. WeMM uses default mode. See the [API reference](API_CONTRACT.md#optional-qwen-instruction-and-encoding-metadata) for instruction and embedding metadata details.

## Optional environment settings

You can supply connection details through `VISION_EMBEDDINGS_BASE_URL` and `VISION_EMBEDDINGS_API_TOKEN` instead of the prompts. Existing environment variables take precedence over the visible form fields. Remove an override if you want to use a form value instead.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `VISION_EMBEDDINGS_MODEL` | `qwen3-vl-embedding-8b` | One model: `qwen3-vl-embedding-8b` or `wemm-embedding-9b` |
| `VISION_EMBEDDINGS_DIMENSION` | `1024` | Output vector dimensions |
| `VISION_EMBEDDINGS_REQUEST_TIMEOUT` | `900` | HTTP request timeout in seconds |
| `VISION_EMBEDDINGS_INSTRUCTION_MODE` | `default` | Omit the optional field to use the API default; `custom` only for Qwen3-VL-Embedding-8B |
| `VISION_EMBEDDINGS_CUSTOM_INSTRUCTION` | Example retrieval instruction | Used only when custom mode is selected (one line, 1–512 characters) |
| `VISION_EMBEDDINGS_SAMPLE_TEXT` | `a flat bright-green triangle on a cream background` | Text sample 1 |
| `VISION_EMBEDDINGS_SAMPLE_TEXT_2` | `a glossy cobalt-blue 3D sphere on a pale background` | Text sample 2 |
| `VISION_EMBEDDINGS_SAMPLE_TEXT_3` | `a realistic orange gift box tied with a ribbon on a wooden table` | Text sample 3 |
| `VISION_EMBEDDINGS_IMAGE_PATH` | empty | Image 1: authorized file or 2D green triangle |
| `VISION_EMBEDDINGS_IMAGE_PATH_2` | empty | Image 2: authorized file or 3D blue sphere |
| `VISION_EMBEDDINGS_IMAGE_PATH_3` | empty | Image 3: authorized file or realistic orange gift |
| `VISION_EMBEDDINGS_MAX_IMAGE_BYTES` | `5242880` | Client-side image limit (5 MiB maximum) |
| `VISION_EMBEDDINGS_REQUIRE_REAL` | `true` | Reject mock/local API backends by default |

Optional `VISION_EMBEDDINGS_SAMPLE_LABEL`, `VISION_EMBEDDINGS_SAMPLE_LABEL_2` and `VISION_EMBEDDINGS_SAMPLE_LABEL_3` set display names for your examples. The notebook reports which settings use environment overrides while keeping credentials and private paths out of status messages.

## Measure serving performance

Use the [performance notebook](../notebooks/multimodal_embeddings_benchmark.ipynb) to compare warm latency, optional cold starts, alternating model routes and bounded concurrent requests. It runs on CPU against your deployed API and exports a JSON report. Read the [benchmark guide](../benchmarks/README.md) before enabling measurements; the default `RUN_BENCHMARK=False` sends no API calls.

## Share and get help

Keep your runtime private, enter credentials through the hidden prompts, and clear outputs and restart the runtime before sharing. Use data you are authorized to process.

For questions or reproducible problems, see [Support](../.github/SUPPORT.md). Examples, translations and notebook improvements are welcome through [Contributing](../CONTRIBUTING.md).
