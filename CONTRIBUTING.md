# Contributing

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Contributions to this multimodal inference API are welcome.

1. Open an issue describing the bug or proposed improvement. Never upload private images, tokens or model weights.
2. Work in a focused branch. Add offline tests for behavior changes.
3. Keep English and Vietnamese documentation synchronized for every user-facing change.
4. Run `bash scripts/setup.sh` to install the project's virtual environment. The script does not activate it in your shell. Run `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check src modal_app.py modal_models.py scripts tests` and `sha256sum -c SHA256SUMS.txt`.
5. After changing files, stage only the intended additions, edits and deletions with `git add <paths>` (and `git add -u` for deletions). Run `.venv/bin/python scripts/update_sha256_manifest.py`, stage `SHA256SUMS.txt`, then run `.venv/bin/python scripts/update_sha256_manifest.py --check` and `sha256sum -c SHA256SUMS.txt`. New files must be staged before generating the manifest; it follows the Git index like CI.
6. Use a concise Conventional Commit subject and bullet points in the commit body.

The CI workflow runs without GPU credentials. Changes to real inference should include a separate, sanitized GPU smoke-test report and pinned upstream model revisions. No full-resolution input images or credentials belong in the report.

## Reproduce the CPU validation environment

Setup and CI install `.[dev]` using `constraints/cpu.txt`. These pins cover CPU runtime/test dependencies, with a separate NumPy pin for Python 3.11. They do not pin the Modal CLI, GPU images or model checkpoints; those have separate deployment requirements. Project metadata keeps its supported dependency ranges.

```bash
python -m pip install -c constraints/cpu.txt -e '.[dev]'
```

Refresh pins deliberately: resolve in fresh Python 3.11 and 3.12 environments, run the full suite and Ruff in each, run the mock quickstart and documentation/manifest checks, and retain an acceptance record. Install build tools and pip separately; this constraints file is not a wheel archive or a GPU lockfile. Use [Verification](docs/VERIFICATION.md) for evidence and the hosted-platform checklist.
