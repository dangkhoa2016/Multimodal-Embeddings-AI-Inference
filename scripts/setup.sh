#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# --quick installs only CPU runtime dependencies for a local mock smoke test.
# No flag: full contributor setup and validation.
if [[ $# -gt 1 ]] || { [[ $# -eq 1 ]] && [[ "$1" != "--quick" ]]; }; then
  printf 'Usage: bash scripts/setup.sh [--quick]\n' >&2
  exit 2
fi
setup_mode=dev
package_spec='.[dev]'
if [[ $# -eq 1 ]]; then
  setup_mode=quick
  package_spec='.'
fi
# Provide a temporary README only when package metadata requires one;
# clean up the temporary file when setup exits.
temporary_readme=false
if [[ ! -e README.md ]]; then
  printf '# Multimodal Embeddings AI Inference\n' > README.md
  temporary_readme=true
fi
cleanup_temporary_readme() {
  if [[ "$temporary_readme" == true ]]; then rm -f README.md; fi
}
trap cleanup_temporary_readme EXIT

constraint_args=()
if [[ -f constraints/cpu.txt ]]; then constraint_args=(-c constraints/cpu.txt); fi

if command -v uv >/dev/null 2>&1; then
  if [ ! -x .venv/bin/python ]; then uv venv --python "${VISION_EMBEDDINGS_PYTHON:-python3}" .venv; fi
  uv pip install --python .venv/bin/python pip "${constraint_args[@]}" -e "$package_spec"
else
  python3 -m venv .venv || { echo 'Python venv/ensurepip unavailable: install uv (https://docs.astral.sh/uv/)'; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install "${constraint_args[@]}" -e "$package_spec"
fi
.venv/bin/python -m compileall -q src modal_app.py modal_models.py scripts
if [[ "$setup_mode" == quick ]]; then
  printf '\nCPU runtime ready. Run bash scripts/quickstart_mock.sh (no GPU required).\n'
  exit 0
fi
.venv/bin/ruff check src modal_app.py modal_models.py scripts
# Run the test suite when test files are available.
if [[ -d tests ]] && find tests -type f -name 'test_*.py' -print -quit | grep -q .; then
  .venv/bin/ruff check tests
  .venv/bin/python -m pytest -q
else
  printf 'No tests at this source revision; skipping pytest.\n'
fi
printf '\nSetup complete. Run bash scripts/quickstart_mock.sh for a local mock test (token generated automatically).\n'
