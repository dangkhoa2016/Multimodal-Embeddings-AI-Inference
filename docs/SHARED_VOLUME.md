# Shared Volume and single-GPU lifecycle

Both HF repositories are downloaded to `compactfaces-ai-weights`:

```
/models/checkpoints/
├── Qwen/Qwen3-VL-Embedding-8B/
└── tencent/WeMM-Embedding-9B/
```

`modal run modal_models.py --selection both` downloads on CPU, resolves immutable Hugging Face commit SHA values and writes `.compactfaces-snapshot.json` after each successful download. Modal Volume commits make downloaded files visible to subsequently started containers. A Volume is persistent model storage **not GPU VRAM**. Avoid two download jobs writing the same checkpoint directory concurrently.

`SharedL40SGpu` is the **only GPU class** in `modal_app.py` and has `gpu="L40S"`, `max_containers=1`, `min_containers=0`. Its supervisor maintains at most one CUDA subprocess per container. Requests for the active model reuse the process. Switching models terminates the old process, waits for it to exit and starts a second process in the corresponding Python virtual environment. Both models are never intentionally resident simultaneously.

This trades throughput/latency on model switches for VRAM headroom and predictable cost. Batches grouped by model minimize switch frequency; compare cold/warm latency when available. Real memory footprint and Modal scheduling behavior **are not verified by offline tests**. On GPU OOM, reduce image dimensions and batch size, then inspect logs and actual model requirements; do not silently quantize and claim BF16 parity.

Deployment pins checkpoint revision via `CF_WEMM9_REVISION` and `CF_QWEN8_REVISION` in the `compactfaces-api` Secret. Set them from the SHA output of the prefetch command; never leave `main` for publishable benchmarks.
