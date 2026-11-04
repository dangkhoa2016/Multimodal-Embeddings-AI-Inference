#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v uv >/dev/null 2>&1; then
  if [ ! -x .venv/bin/python ]; then uv venv --python "${CF_PYTHON:-python3}" .venv; fi
  uv pip install --python .venv/bin/python -e '.[dev]'
else
  python3 -m venv .venv || { echo 'Python venv/ensurepip unavailable: install uv (https://docs.astral.sh/uv/)'; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e '.[dev]'
fi
.venv/bin/python -m compileall -q src modal_app.py modal_models.py scripts
.venv/bin/ruff check src modal_app.py modal_models.py scripts tests
.venv/bin/python -m pytest -q
printf '\nSetup complete. Configure CF_API_TOKEN before running mock API.\n'
