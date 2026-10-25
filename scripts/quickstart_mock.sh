#!/usr/bin/env bash
# Local API contract smoke test; does not run either real embedding model.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/uvicorn || ! -x .venv/bin/python ]]; then
  echo "Install dependencies first: bash scripts/setup.sh" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the local API smoke test." >&2
  exit 1
fi

# The mock server and client share one temporary token in this process.
export VISION_EMBEDDINGS_API_TOKEN="$(
  .venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(48))'
)"
port="$(
  .venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
)"
base_url="http://127.0.0.1:$port"
log_file="$(mktemp /tmp/multimodal-embeddings-mock-XXXXXX.log)"
VISION_EMBEDDINGS_BACKEND=mock .venv/bin/uvicorn   multimodal_embeddings_ai_inference.api:local_app   --factory --host 127.0.0.1 --port "$port" > "$log_file" 2>&1 &
server_pid=$!

cleanup() {
  local status=$?
  trap - EXIT
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  if [[ "$status" -eq 0 ]]; then
    rm -f -- "$log_file"
  else
    printf 'Mock server log preserved for troubleshooting: %s\n' "$log_file" >&2
  fi
  exit "$status"
}
trap cleanup EXIT

ready=false
for _ in {1..30}; do
  if curl --silent --show-error --fail --output /dev/null     "$base_url/health" 2>/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 1
done

if [[ "$ready" != true ]]; then
  echo "Mock server did not become ready. Check: $log_file" >&2
  exit 1
fi

echo "Local mock API is ready (no real model or GPU)."
.venv/bin/python scripts/smoke_api.py   --base-url "$base_url"   --model qwen3-vl-embedding-8b   --text "sample image"   --dimension 1024
echo "Local mock contract test complete."
