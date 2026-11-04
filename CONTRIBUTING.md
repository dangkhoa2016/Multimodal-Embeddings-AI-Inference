# Contributing

We welcome improvements that preserve the project's research scope.

1. Open an issue describing a bug or proposed feature; avoid uploading private face images.
2. Create a branch, keep changes small, add tests and update both `README.md` and `README.vi.md` when user-facing behavior changes.
3. Run `bash scripts/setup.sh`, `pytest -q`, and `python -m compileall -q src` before a PR.
4. Use descriptive commit messages such as `fix: reject malformed embedding responses` or `docs: clarify Modal secret setup`.
5. Do not commit `.env`, credentials, models, user images or generated embeddings.

No GPU is required for CI. A PR that changes real model inference must include a separate GPU smoke report with exact model revisions, GPU type, peak VRAM and representative images from authorized datasets.
