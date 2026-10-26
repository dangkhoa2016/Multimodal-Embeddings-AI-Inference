# Dedicated model Volumes and GPU lifecycle

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](MODEL_VOLUMES.vi.md)

The deployment assigns one NVIDIA L40S worker and one persistent Modal Volume to each model. Neither worker loads or unloads the other model.

```text
FastAPI gateway (CPU)
├── QwenL40SGpu: 1 × L40S
│   └── multimodal-embeddings-qwen-weights
│       └── checkpoints/Qwen/Qwen3-VL-Embedding-8B/
└── WeMML40SGpu: 1 × L40S
    └── multimodal-embeddings-wemm-weights
        └── checkpoints/tencent/WeMM-Embedding-9B/
```

## Workspace, Environment and checkpoint ownership

Both Volumes are created by `modal_models.py` in the **active Modal
Workspace and Environment**. The deployed `modal_app.py` looks up these
same Volume names with `create_if_missing=False`: a different Environment
or mismatched `VISION_EMBEDDINGS_QWEN_VOLUME_NAME` /
`VISION_EMBEDDINGS_WEMM_VOLUME_NAME` will cause a missing resource rather
than reuse the intended checkpoints. Check `modal profile list`,
`modal environment list` and `MODAL_ENVIRONMENT` before both download
and deployment; see [Operations](OPERATIONS.md#select-a-consistent-workspace-and-environment).

`READY <repository> @ <sha> -> <volume>` printed by each download function
confirms that the snapshot and Volume commit completed. Carry those exact
SHA revisions into the two Secrets using
[scripts/create_modal_secret.py](OPERATIONS.md#configure-authentication).
A Volume is **persistent disk storage**, not a GPU reservation or proof
that loading and inference work; use the separate tests in
[Verification](VERIFICATION.md).

For a fresh installation, run `modal run modal_models.py --selection both` to download [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) into separate Volumes. Do not download again when existing checkpoints can be copied and verified.

Specify Volume names with `VISION_EMBEDDINGS_QWEN_VOLUME_NAME` and `VISION_EMBEDDINGS_WEMM_VOLUME_NAME`. GPU classes use `min_containers=0`, `max_containers=1` and `scaledown_window=180` by default. Both can run simultaneously, requiring up to two L40S GPUs when both are active.

A Volume stores downloaded weights on persistent disk; **it does not reserve GPU VRAM**. Cold start remains possible after scale-to-zero, but alternating API calls between warm workers do not cause model switching. Keep model and checkpoint revision metadata separate per worker.

Pin revisions using `VISION_EMBEDDINGS_QWEN8_REVISION` and `VISION_EMBEDDINGS_WEMM9_REVISION`; never mix vectors produced by distinct models or revisions in one index.

Before retiring an existing deployment, verify both Volume contents and test both dedicated GPU workers through the authenticated HTTPS API.
