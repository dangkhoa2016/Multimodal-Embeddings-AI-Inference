# Verification and acceptance

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](VERIFICATION.vi.md)

This guide explains how to verify a deployment of the Multimodal Embeddings AI Inference API. Run the checks on your own environment. Example commands and expected outcomes are **procedures**, not published performance results or a certification of any deployment.

## Supported embedding models

The service supports two image-and-text embedding models, each with its own vector space:

| Full model name on Hugging Face | Model ID used in API requests | Purpose |
| --- | --- | --- |
| [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | `qwen3-vl-embedding-8b` | Create image or text embeddings for semantic and cross-modal retrieval |
| [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) | `wemm-embedding-9b` | Create image or text embeddings for image similarity and text-to-image retrieval |

“Qwen” and “WeMM” are short names in this guide; the **model ID** is the exact value for the API `model` field. Keep embeddings produced by different models in separate indexes.

## Local CPU checks

From the repository root, run setup **on first installation** or again when the Python environment or dependencies need refreshing:

```bash
bash scripts/setup.sh
```

Setup creates or reuses `.venv`, installs/refreshes CPU dependencies and runs Ruff and pytest. Rerunning is normally supported, but it can access the package index, take time and update dependencies within project constraints; **it is not needed before every mock check**.

After setup, repeat the local mock smoke test whenever needed:

```bash
bash scripts/quickstart_mock.sh
python scripts/audit_release.py
python scripts/update_sha256_manifest.py --check
sha256sum -c SHA256SUMS.txt
```

Each quickstart run generates a temporary Bearer token, selects an available localhost port, starts the mock API, validates its response and stops the server. It removes its temporary server log on success; on failure, it preserves the log and prints its path for troubleshooting. The mock does **not** load either model checkpoint or use a GPU. A successful mock check verifies local API contracts, not real embedding quality.

| Check | Expected outcome and verification |
| --- | --- |
| CPU validation (pytest, Ruff) | Record **PASS** only when checks complete successfully. The [CI workflow](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/actions/workflows/ci.yml) must be inspected for the exact source commit. |

Without direct run evidence, mark the corresponding GPU or hosted-UI environment **not independently verified**.

## Verify deployed inference

Use an HTTPS endpoint you operate or are authorized to access. Configure the application Bearer token and expected checkpoint revisions as described in [Operations](OPERATIONS.md#configure-authentication). Run the smoke test with an authorized image:

```bash
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png --dimension 1024
```

Set `VISION_EMBEDDINGS_QWEN8_REVISION` and `VISION_EMBEDDINGS_WEMM9_REVISION`, or provide the equivalent `--qwen-revision` and `--wemm-revision` arguments.

For each model, verify that:

- Requests without valid authentication are rejected; authorized requests succeed.
- Model ID and checkpoint revision match the expected configuration (`revision_verified: true`).
- Vectors contain finite numbers, have the requested dimension and are L2-normalized.
- Text and authorized image inputs route to the selected model, with sanitized errors.

A mocked PASS does not establish real GPU availability or retrieval quality.

## Verify hosted notebooks

The [API demo](../notebooks/multimodal_embeddings_api_demo.ipynb) and [performance notebook](../notebooks/multimodal_embeddings_benchmark.ipynb) use an existing HTTPS API; they do not deploy GPU models.

On Kaggle, Colab or another Jupyter host, validate editable inputs, previews, protected credential prompts, model capability checks, response validation and report rendering. Confirm recovery after invalid configuration or interrupted requests without automatically retrying a potentially completed POST.

The performance notebook defaults to `RUN_BENCHMARK=False`. Enable only authorized checks, account for GPU charges and label each measured phase accurately.

## Acceptance source provenance

Record the actual application source commit, model checkpoint revisions, UTC time, environment, tested cases and redacted results. Client and deployed server source revisions may differ; document each when applicable.

```bash
git rev-parse HEAD
sha256sum -c SHA256SUMS.txt
python scripts/record_acceptance.py --run-checks --output /tmp/acceptance-session.json
```

The acceptance recorder captures source and CPU checks. It does **not** contact an inference endpoint or validate GPU execution or hosted notebook UI. Associate CI results with the exact commit, not a general CI badge. Use the [acceptance template](ACCEPTANCE_TEMPLATE.md) when helpful.

Never publish access tokens, private service URLs, private images, raw inputs, raw embeddings or identifying worker details.

## Scope and limitations

Functional checks verify behavior only for the configuration tested. They do not establish retrieval accuracy, biometric confidence, production availability, sustained throughput, p95 latency or cost per request. Evaluate retrieval quality separately using authorized representative data and keep embeddings from distinct models or configurations separate.

See [Operations](OPERATIONS.md) and [Security](../SECURITY.md).
