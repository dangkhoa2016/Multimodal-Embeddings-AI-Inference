
# Notebook guide — Google Colab and Kaggle

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](NOTEBOOK_GUIDE.vi.md)

The [bilingual English/Vietnamese Multimodal Embeddings API notebook](../notebooks/multimodal_embeddings_api_demo.ipynb) is an **HTTP client**. It works on a CPU runtime without model weights, CUDA binaries or the Modal CLI. The inference server is deployed and managed **separately**.

The API supports [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B). **Qwen3-VL-Embedding-8B is selected by default; only one model runs per notebook execution.**

## Open the notebook

[Open directly in Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_api_demo.ipynb) or [view the notebook file](../notebooks/multimodal_embeddings_api_demo.ipynb). For Kaggle, import the notebook file into a new Python 3 CPU notebook.

**Inference prerequisite:** An already deployed HTTPS API URL and an application Bearer token.

## Required connection details

| Environment variable | Required? | Description |
| --- | --- | --- |
| `VISION_EMBEDDINGS_BASE_URL` | Yes | Already-deployed HTTPS endpoint, e.g. `https://YOUR_DEPLOYMENT.modal.run` (without `/v1`) |
| `VISION_EMBEDDINGS_API_TOKEN` | Yes | Application Bearer token issued by the API deployment operator |

The three bundled images can be previewed before supplying either value. Both connection values can then be entered with hidden prompts under **API connection**. Values already set in the notebook runtime's environment take precedence. The notebook does not request `MODAL_TOKEN_ID` or `MODAL_TOKEN_SECRET`: those credentials are for managing Modal resources, not calling this API.

**Maintaining notebook code:** Code cells are synchronized from [reviewable Python sources](../notebooks/source/) using `python scripts/sync_notebooks.py --check` (or `--write` after an intentional source edit). The published notebooks remain self-contained: Kaggle and Colab do not need a clone or those source files.

## How to run

1. Open the notebook in **Google Colab** or **Kaggle Notebooks** with a Python 3 **CPU** runtime; no repository checkout is needed for API-only use.
2. Under **Model configuration**, choose `MODEL_ID` (default **Qwen3-VL-Embedding-8B**), `DIMENSION` and `TIMEOUT_SECONDS`. Colab presents a model dropdown; on Kaggle edit the configuration form. Run the collapsed **Configuration helpers** cell next. Matching environment variables override the visible defaults.
3. Optionally edit **Customize samples** to change the three texts and three image paths in a dedicated cell. Then run **Prepare input samples** to view the three embedded 512 × 512 px images under **Built-in image previews** (2D illustration, 3D render and photorealistic scene). This step does not require credentials or send network requests.
4. Under **API connection**, enter the deployed HTTPS URL and API Bearer token using hidden prompts, unless already set as environment variables. Run the authentication/model catalog checks, then the three text and three image examples. The notebook sends **exactly six sequential inference requests** for the **one selected model**.
5. Read the accessible English-first, Vietnamese-second HTML report: a 3 × 3 cosine matrix (not percentages), vector size, six API times, L2 norms and revision. Each earlier stage also has a readable status panel. The report uses explicit high-contrast colors to remain legible in Kaggle dark mode.
6. Clear all outputs and restart the runtime before sharing the notebook.

## Editable cells and helpers

Edit **Model configuration**, **Customize samples**, and the **Secure API connection** prompts. Run the collapsed **Configuration helpers**, **Prepare input samples** and **Results** cells without editing validation, Base64 data or HTML/CSS. The HTTP integration steps are in **Check API connection** and **Generate embeddings**. Helpers remain embedded so the notebook works independently on Kaggle and Colab; no package download is required.

## Edit your own inputs

After **Model configuration** and **Configuration helpers**, open **Customize samples / Tùy chỉnh dữ liệu mẫu**. Edit `SAMPLE_TEXT_1` through `SAMPLE_TEXT_3` and optionally `IMAGE_PATH_1` through `IMAGE_PATH_3`. Empty paths keep the bundled WebP images. Nonempty paths must identify an authorized PNG/JPEG/WebP file already in the runtime. Example locations: Colab `/content/my-image.png` after uploading a file, or Kaggle `/kaggle/input/my-dataset/my-image.png` after attaching a dataset. These are path examples, not preinstalled files.

**Connection changes:** Follow the notebook recovery table. A failed reconnection invalidates the previous preflight and blocks inference. Results also refuses an unverified connection or vectors from a different endpoint/token. After changing connection details, run **Secure API connection → Check API connection → Generate embeddings → Results**.

**Re-run order:** Use **What should I run next?** below **Configuration helpers** for local recovery guidance, or the bilingual **After an edit or interruption** table at its top. The recovery cell sends no requests and never retries a POST. After a model edit, rerun **Configuration helpers**, prepare samples and perform a fresh **Check API connection** before inference. Edited sample fields persist, but old model capability checks cannot be reused.

**Effective configuration sources:** The demo displays whether the selected model and vector dimension came from the form or an environment override. After preparing samples, it lists only the names of environment variables overriding text, image paths or labels. Private text, image paths, label values and credentials are never displayed in these status messages. To use a form value, unset the corresponding environment variable and rerun the relevant cells.

**Precedence:** existing `VISION_EMBEDDINGS_SAMPLE_TEXT`, `VISION_EMBEDDINGS_IMAGE_PATH`, and their `_2` / `_3` environment variables override the visible cell fields. Remove a conflicting environment variable to use the edited field. Selected samples persist after a model change; the recovery table lists the required steps. Old vectors are never relabeled as new results; previews always show bundled images rather than private inputs.

Bundled image previews are displayed before custom-file validation. For an unreadable or missing custom image, the notebook shows its image number and a bilingual recovery hint without exposing the path. Confirm the file exists in this runtime, correct its path, and rerun sample preparation before inference.

## Optional environment variables

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

The model detail endpoint advertises instruction capability. The notebook requires a matching input SHA-256, preprocessing version, instruction hash and configuration SHA-256. For text, the digest uses UTF-8 of the value after stripping surrounding whitespace, matching the server schema; image digests use original image bytes. Its six examples use one instruction configuration; real retrieval applications may use different instructions for query and document embeddings. Custom instruction is rejected with Tencent WeMM-Embedding-9B before an inference request. Leading/trailing whitespace in Qwen custom instruction is trimmed before validation and SHA-256 calculation; empty instructions after trimming are rejected.

Qwen3-VL-Embedding-8B accepts integer dimensions 64–4096. Tencent WeMM-Embedding-9B accepts 64, 128, 256, 512, 1024, 2048 and 4096. Invalid model/dimension combinations are rejected in the configuration cell before sending requests.

**Kaggle/Jupyter tip:** The notebooks are standalone; no repository clone, Kaggle Input dataset or local model weights are required. Edit short configuration and sample cells only. Some viewers show implementation cells even when marked hidden in Jupyter metadata; collapse them using the editor controls if supported, then run in order. Never paste credentials into a saved code cell.

**Illustrative output layout, not measured results:** The demo reports a 3 × 3 cosine matrix with three text rows and three image columns. Each entry is `cosine(text_i, image_j)`, **not** identification confidence or benchmark evidence. Values depend on your inputs and model.

## Performance benchmark notebook

The API demo sends **exactly six sequential inference requests for one selected model** (three texts, three images); there is no model-switch or benchmark mode. For repeatable cold GPU tests (idle time derived from editable deployment values), balanced warm text/image latency and two-direction independent GPU worker warmup, use the [separate performance notebook](../notebooks/multimodal_embeddings_benchmark.ipynb) and [benchmark protocol](../benchmarks/README.md). The benchmark is explicitly opt-in and has cold trials disabled by default. [Open the benchmark in Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_benchmark.ipynb).

## What successful output looks like

The notebook displays a bilingual results panel with **instruction mode and embedding configuration SHA-256**, **selected model**, **three matched cosine scores**, a **3 × 3 text/image cosine matrix**, **vector dimension**, **checkpoint revision**, **L2 norms** and **six API processing times**. Values are computed by the selected live deployment; no fabricated score or GPU measurement is shown here.

## API vs. infrastructure operations

This notebook **does not install Modal CLI, manage a Modal App/Secret/Volume, download checkpoints, or deploy the inference service**. Operators who need their own deployment should follow [Operations](OPERATIONS.md); the notebook is only an authenticated client for an existing endpoint.

## Privacy, costs and limits

- Use images you are authorized to process. The notebook embeds three generated 512 × 512 px WebP images (no downloads): a 2D green triangle, a 3D glossy blue sphere, and a photorealistic orange gift box. Without custom paths, these exact three embedded images are sent to the API. These are not a retrieval benchmark. Private user-supplied images are never previewed automatically.
- The API token and endpoint are not printed or written to a file. Hidden inputs do not make the runtime memory private: run only trusted code and restart after use.
- Do not share raw embeddings, image inputs, private URLs or authentication headers. Untrusted notebook viewers should not inherit access to a private inference service.
- Even six small inference requests can start a billable L40S container; API times are shown in seconds, may include cold start or independent GPU worker warmup, and are not end-to-end network latency.
- `GET /v1/health` proves gateway liveness, not GPU readiness. The default configuration rejects mock backends for live demonstrations.
- The report explains cosine scores (approximately −1 to +1) and L2 norms near 1. **Cosine is not a percentage or confidence level**, and it does not establish biometric identity or retrieval accuracy.
- Mock execution can validate the notebook logic, but successful Kaggle/Colab execution and live Modal GPU inference require separate verification.

See [API contract](API_CONTRACT.md), [Operations](OPERATIONS.md), and [Verification](VERIFICATION.md).

**Custom inputs:** Optional VISION_EMBEDDINGS_SAMPLE_LABEL (_2, _3) adds display labels to the three input pairs. With a custom text or image sample the report shows built-in/custom status per item without revealing image paths or previewing private images. The cosine diagonal represents same-index comparisons, not assumed semantic matches. HTTP errors show bilingual hints for 401, 413, 422 and 503 without response bodies.
