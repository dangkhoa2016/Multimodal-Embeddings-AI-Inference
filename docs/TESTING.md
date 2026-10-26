# Verification and test strategy

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](TESTING.vi.md)

This project separates **CPU correctness tests** from **explicitly authorized
real GPU inference**. A passing CPU suite does not demonstrate retrieval
quality from either deployed checkpoint.

## CPU suites

| Command | Purpose |
| --- | --- |
| `PYTHONPATH=src python -m pytest -q` | Complete CPU suite |
| `PYTHONPATH=src python -m pytest -q -m core` | API, vectors, encoder adapters, retrieval and routing |
| `PYTHONPATH=src python -m pytest -q -m security` | Auth, rate limits, async jobs and admission |
| `PYTHONPATH=src python -m pytest -q -m notebook` | Demo and benchmark notebooks |
| `PYTHONPATH=src python -m pytest -q -m publication` | Release, evidence and docs |
| `PYTHONPATH=src python -m pytest -p pytest_cov -q --cov=multimodal_embeddings_ai_inference --cov-report=term-missing` | Source coverage |

Install development dependencies with `python -m pip install -e '.[dev]'`.
CI runs all CPU tests on Python 3.11, 3.12 and 3.13. Evidence whose Git
addition timestamp precedes its latest recorded event fails CI.
Comparing source hashes does not independently attest deployed GPU source.

## Retrieval evaluation from real saved embeddings

Prepare separate model-specific JSON datasets with a pinned revision,
config_sha256, image/query vectors and explicit ground-truth `relevant_images`.
Both models must use matching labeled workloads. Never mix vectors from
Qwen and WeMM in one embedding space.

    python scripts/evaluate_retrieval.py       --input qwen-saved-embeddings.json       --input wemm-saved-embeddings.json       --top-k 5 --require-both-models

The script calculates Recall@K, Hit@1 and MRR without GPU calls and never
prints raw vectors. Synthetic vectors in unit tests verify arithmetic only.
The existing saved image-search evidence covers two Qwen queries, not a
matched, comprehensive WeMM quality evaluation.

## Explicit real GPU smoke test

Only an authorized deployment operator should invoke this billable command:

    python scripts/smoke_modal_gpu.py       --image /path/to/authorized-image.png       --text "an orange gift box"       --qwen-revision "$VISION_EMBEDDINGS_QWEN8_REVISION"       --wemm-revision "$VISION_EMBEDDINGS_WEMM9_REVISION"

It checks returned checkpoint identity, expected revision, vector shape and
norm, and distinct worker lifecycle IDs. CI does not run this command.
It cannot prove retrieval quality or production reliability.

## Published release asset verification

After downloading all five v1.0.0 assets into one empty directory:

    python scripts/verify_release_assets.py       --directory /path/to/downloaded-assets       --git-ref v1.0.0 --require-supplementary

This verifies the actual downloaded files against the pinned Git tag and
SHA-256 manifest. A green CI badge cannot replace this check.
