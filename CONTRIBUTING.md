# Contributing

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Contributions are welcome, whether you are using the API in an application, trying a notebook, improving a translation or working on inference code. You do not need GPU access to contribute documentation, examples or CPU-tested changes.

## Ways to contribute

- Report a reproducible bug or describe a use case in the appropriate [issue form](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/issues/new/choose).
- Improve English/Vietnamese documentation and notebook instructions.
- Share client examples, integrations or ideas for model support.
- Fix code, improve validation or add tests for a behavior change.

For usage questions, see [Support](.github/SUPPORT.md). Report security vulnerabilities privately through [Security](SECURITY.md). Keep tokens, private images and model weights out of issues and pull requests.

## Your first pull request

1. For a substantial change, open an issue first so its scope and API compatibility can be discussed.
2. Fork the repository and work in a focused branch.
3. Make the change and keep English/Vietnamese user documentation aligned. Add offline coverage for changed behavior.
4. Run the relevant checks below, then open a pull request explaining the user-facing change and how you verified it.

Use a concise Conventional Commit subject and a short body explaining the change and its reason. Community participation follows the [Code of conduct](.github/CODE_OF_CONDUCT.md).

## Development setup

From a cloned repository:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

**What setup does:** It creates or reuses `.venv`, installs CPU development dependencies from `constraints/cpu.txt`, and runs Ruff and pytest. GPU deployment tools and model checkpoints are installed separately; see [Operations](docs/OPERATIONS.md).

## Editing notebook code

Code cells are maintained in `notebooks/source/` and embedded in the published `.ipynb` files. Edit the Python source, then run:

```bash
python scripts/sync_notebooks.py --write
```

For the benchmark notebook, this command also refreshes `NOTEBOOK_TEMPLATE_SHA256` in the Python source and published code cells. `--check` verifies synchronization and the fingerprint without modifying either file.

Edit explanatory Markdown in the notebook itself. Keep both languages aligned and the README preview consistent with the bundled samples. The resulting notebooks must remain self-contained for Colab and Kaggle.

## Preparing and checking changes

Stage the intended additions, edits and deletions before refreshing the checksum manifest. The generator uses paths from the Git index, so stage new files and removed paths first. The manifest excludes README.md and README.vi.md, which are introduced in the final README-only commit and remain integrity-protected by Git.

```bash
python scripts/update_sha256_manifest.py
git add SHA256SUMS.txt
python scripts/update_sha256_manifest.py --check
```

Run these checks before submitting:

```bash
python -m pytest -q
bash scripts/quickstart_mock.sh
ruff check src modal_app.py modal_models.py scripts tests
ruff check notebooks/source --select E4,E7,E9,F63,F7,F823 --ignore E402
python scripts/sync_notebooks.py --check
python scripts/audit_release.py
sha256sum -c SHA256SUMS.txt
```

CI runs CPU tests on Python 3.11–3.13 and HTTP client tests with Node.js 24. GPU access is not required for these checks.

## Reporting inference and notebook changes

For changes to real inference, include a sanitized test report with model revisions, dependency versions and the cases you exercised. For notebook changes, note the host and whether you checked inputs, previews, authentication, results and interruption recovery. The [deployment worksheet](docs/ACCEPTANCE_TEMPLATE.md) is available for recording observations.

To capture source identity and CPU checks for a report:

```bash
git rev-parse HEAD
python scripts/record_acceptance.py --run-checks --output /tmp/acceptance-session.json
```

This recorder collects source and CPU information; GPU and hosted UI observations need separate runs. Record client and deployed server revisions separately when they differ. Share only redacted results.

## Updating dependencies

CPU runtime and test versions are pinned in `constraints/cpu.txt`; GPU dependencies have separate Qwen and WeMM locks. When refreshing CPU pins, resolve in fresh Python 3.11, 3.12 and 3.13 environments, run the full suite and Ruff in each, and exercise the mock quickstart. Preserve the supported ranges in `pyproject.toml`. See [Operations](docs/OPERATIONS.md#reproducible-gpu-dependencies) for GPU dependency updates.
