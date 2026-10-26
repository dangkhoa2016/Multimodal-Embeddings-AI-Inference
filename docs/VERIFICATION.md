# Check your deployment

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](VERIFICATION.vi.md)

Use these checks after deploying or updating the Multimodal Embeddings AI Inference API. They help confirm authentication, model responses and notebook connections before you connect an application.

## Models and connection details

The service provides [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) as `qwen3-vl-embedding-8b` and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) as `wemm-embedding-9b`. Each model has its own vector space.

Have your deployment's HTTPS URL, API Bearer token and expected checkpoint revisions available. If you have not deployed yet, start with [Operations](OPERATIONS.md). That guide also explains how to load the saved token and revisions into your shell.

## Verify resources and gateway before GPU work

In the **same Modal Workspace and Environment** used for checkpoint downloads,
Secrets and deployment, inspect the resource names without running inference:

```bash
modal app list
modal volume list
modal secret list
```

Confirm the App, both model Volumes, and the separate API and model-revision
Secrets are present. Their existence alone does not prove that either model
loaded. See [first-time deployment](OPERATIONS.md#first-time-deployment-prerequisites-and-order)
for profile and Environment selection.

Load the owner-only local credential file produced by
`scripts/create_modal_secret.py` as described in
[Operations](OPERATIONS.md#configure-authentication). Keep its contents out
of logs and shared notebook outputs. With the real URL returned by
`modal deploy`, check gateway health without a GPU embedding request:

```bash
curl --fail-with-body --silent --show-error \
  -H "Authorization: Bearer $VISION_EMBEDDINGS_API_TOKEN" \
  'https://YOUR_DEPLOYMENT.modal.run/v1/health'
```

An authenticated healthy gateway response checks the CPU route only, not
model availability. A missing/invalid token should be denied; do not make
repeated bad-token attempts because the gateway can temporarily lock the
client out.

## Check model responses

**The following checks invoke real L40S inference and may cost money.**
Choose an image you are authorized to process. They test different layers:

```bash
# 1. Direct Modal SDK call to Qwen and WeMM GPU worker classes.
#    This bypasses the HTTPS gateway and its Bearer authentication.
python scripts/smoke_modal_gpu.py --image /path/to/authorized-image.png

# 2. Authenticated HTTPS end-to-end call to both model routes.
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png --dimension 1024
```

The direct worker test must print `REAL_PASS` for **both** models and validate
finite, L2-normalized vectors and distinct worker IDs. The HTTPS test must
print `"status": "PASS"` and `"revision_verified": true` for **both**
models. It checks model ID, `backend=modal`, expected pinned revision, input
digest, dimension 1,024 and L2 norm. Set
`VISION_EMBEDDINGS_QWEN8_REVISION` / `VISION_EMBEDDINGS_WEMM9_REVISION`
from the owner-only file or pass `--qwen-revision` / `--wemm-revision`.

Run the HTTPS test promptly after direct worker checks while GPU containers
may still be warm. Its synchronous endpoint has a **60-second default server
deadline**; on the first cold start an HTTP 504 may mean scheduling/loading
exceeded that deadline. A timed-out POST can still have triggered billable
work. **Do not automatically retry**: inspect your Modal App logs and
[troubleshooting guide](TROUBLESHOOTING.md), then use the
[async API with a stable request ID](API_CONTRACT.md#async-gpu-jobs-cold-starts-and-resume)
for new, explicitly approved cold-start inference. Neither smoke test proves
retrieval accuracy, availability SLAs or that Modal currently runs the
latest Git commit.

## Check a notebook connection

Open the [API demo](../notebooks/multimodal_embeddings_api_demo.ipynb), prepare the samples, connect your deployment and run **Generate embeddings → Results**. You should see six validated responses and the bilingual cosine report. Follow the [notebook guide](NOTEBOOK_GUIDE.md) to change inputs or reconnect.

If you use both Colab and Kaggle, check each platform you intend to support. You can use the [performance notebook](../notebooks/multimodal_embeddings_benchmark.ipynb) for latency measurements; it starts with `RUN_BENCHMARK=False` and makes calls only when enabled.

## Explore the API locally

To try the HTTP workflow before deploying:

```bash
bash scripts/setup.sh --quick
bash scripts/quickstart_mock.sh
```

The local mock starts and stops automatically. A successful run prints `"pass": true` and `"backend": "mock"`. It uses synthetic vectors; connect to your deployed service to check model-generated embeddings.

Development checks, notebook source synchronization and checksum maintenance are documented in [Contributing](../CONTRIBUTING.md).

## Share useful observations

For each deployment session, record GPU or hosted-UI observations with the environment and version you used when reporting a problem or sharing performance results. The optional [deployment worksheet](ACCEPTANCE_TEMPLATE.md) can help organize them.

Include the affected model, steps to reproduce, expected behavior and sanitized errors. Keep tokens, private URLs, input images and raw embeddings out of public reports. See [Support](../.github/SUPPORT.md) for the reporting channels.

Functional checks cover the deployment and configuration you exercised. Evaluate retrieval quality separately on examples representative of your application.
