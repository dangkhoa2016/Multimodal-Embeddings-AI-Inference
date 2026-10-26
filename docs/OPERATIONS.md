# Modal deployment and operations

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](OPERATIONS.vi.md)

Follow the steps below to deploy and operate the **Multimodal Embeddings AI Inference API** on Modal.

### Operator quick navigation

| Your task | Start with | GPU/Secret impact |
| --- | --- | --- |
| Deploy from scratch | [First-time checklist](#first-time-deployment-prerequisites-and-order) | Downloads/builds may cost; GPU checks require explicit approval |
| Prepare model checkpoints | [Download and pin checkpoints](#download-and-pin-checkpoints) | Remote CPU/storage |
| Configure owner token and model revisions | [Configure authentication](#configure-authentication) | Updates Modal Secrets only when operator runs helper |
| Add or rotate client keys | [Per-client keys](#optional-multi-client-api-keys-and-durable-async-admission) | Offline staging first; Secret rollout is manual |
| Scale CPU gateways | [Distributed rate limiting](#distributed-rate-limiting-and-gateway-overload-protection) | Requires shared Redis/Valkey and operator-controlled deploy |
| Verify a deployment | [Deployment checks](#check-the-deployment) | Includes separately approved billable GPU tests |
| Investigate uncertain async work | [Controlled reconciliation](#controlled-reconciliation-for-overdue-async-jobs) | GET/read-only first; never automatically retry GPU |

The [root README](../README.md) explains the API and local mock quickstart; this guide is for **operators**. The [API contract](API_CONTRACT.md) and [client integration guide](CLIENT_INTEGRATION.md) are for applications consuming a deployed endpoint.

### Configuration reference

The included `.env.example` is a local mock starter, not a complete production
configuration. For multi-client or multi-gateway deployments, consult the
[client key setup](#optional-multi-client-api-keys-and-durable-async-admission)
and [distributed rate limit setup](#distributed-rate-limiting-and-gateway-overload-protection).
The corresponding defaults in `Settings.from_env()` are:

| Setting | Default |
| --- | --- |
| `VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON` | `{}` |
| `VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL` | `8` |
| `VISION_EMBEDDINGS_MAX_ASYNC_JOBS_PER_CLIENT` | `2` |
| `VISION_EMBEDDINGS_MAX_CLIENT_EMBEDDINGS_PER_MINUTE` | `60` |
| `VISION_EMBEDDINGS_MAX_BODY_REQUESTS` | `8` |
| `VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT` | Auto when unset/empty; explicitly `0` in `.env.example` |
| `VISION_EMBEDDINGS_RATE_REDIS_URL` | Unset |
| `VISION_EMBEDDINGS_RATE_REDIS_NAMESPACE` | `multimodal-embeddings-v1` |
| `VISION_EMBEDDINGS_MAX_DISTRIBUTED_SYNC_INFLIGHT` | `4` |
| `VISION_EMBEDDINGS_MAX_DISTRIBUTED_CLIENT_SYNC_INFLIGHT` | `2` |
| `VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS` | `1` (deploy-time) |

## Security and operational telemetry

Use authenticated `GET /v1/ops/metrics` with the **owner token** for
per-gateway response counts (especially HTTP 401/429/503), uptime and
observed async job states. These counters reset after restart and do
not query Redis or Modal Dict. They do **not** give the number of active
GPU jobs across all workers; for that, reconcile the job store and Modal
execution logs with spending records. Keep responses private. Run the
offline audit without GPU or cloud credentials:

```bash
python scripts/security_audit.py
python -m pytest -q tests/test_security_monitoring.py
```

The audit checks known source invariants, and tests cover owner-only
access, token redaction, sanitized request errors and no secret-bearing
exception logs. See the bilingual [Security guide](SECURITY.md).
This does not replace Redis provider, network, dependency or live Modal
security assessments.

## Runtime layout

- A CPU FastAPI gateway accepts HTTPS requests authenticated with a Bearer token.
- Two independent GPU workers each use one NVIDIA L40S with `max_containers=1`; no cross-model subprocess switching.
- Each model checkpoint resides in its own Modal Volume, separate from the GPU's VRAM.
- Async mode uses a detached CPU coordinator and a Modal Dict for idempotent job status and completed vectors. The coordinator refreshes a heartbeat every 30 seconds; a client disconnect does not cancel submitted GPU work.

## First-time deployment: prerequisites and order

**Who can follow these steps?** A user with access to this repository, Python
3.11–3.13, Bash (Linux/macOS or Windows WSL), Git, and a Modal account with
permission to create Apps, Secrets and Volumes. The workspace must support
storage, CPU downloads and **up to two NVIDIA L40S workers** (one per model).
A local GPU is unnecessary. Model checkpoints are downloaded from Hugging Face
by Modal CPU functions; public model access and network availability are also
required.

**Before any billable step**, check your Modal workspace's billing setup, GPU
availability and [Workspace usage budget and spend limit](https://modal.com/docs/guide/budgets).
Downloads/builds consume remote CPU and storage resources; direct GPU checks
and inference can incur GPU charges. Your account's limits and GPU
availability cannot be validated by this repository or CI.

For a fresh installation, follow this order:

1. [Set up the local environment](#setup) and run `modal setup`.
2. [Select a single Workspace and Environment](#select-a-consistent-workspace-and-environment)
   for **all** downloads, Secret creation, deployment and verification.
3. [Download the two pinned checkpoints](#download-and-pin-checkpoints); confirm
   both `READY` lines and their revision SHAs.
4. [Create the two independent Secrets](#configure-authentication) using
   those same revision SHAs.
5. [Deploy](#deploy) and capture the HTTPS URL printed for your own account.
6. [Check the deployment](#check-the-deployment): control-plane checks first,
   then explicitly approved billable GPU and HTTPS tests.

A successful CLI deploy shows that the App was registered, **not** that either
large model has initialized or that the HTTPS inference route has passed.
See [Verification](VERIFICATION.md) and [Troubleshooting](TROUBLESHOOTING.md).

## Interpreting delayed async jobs

The CPU coordinator has a 3,600-second execution limit. When a job stays
`RUNNING` without a heartbeat for over 180 seconds, or stays `QUEUED` for
more than 24 hours, read-only status requests return `INDETERMINATE` (HTTP
200). This is **not** an automatic failure, cancellation or retry: the
existing job ID stays reserved and a delayed completion can still be read.
Inspect the original Modal function logs, job Dict and GPU activity before
taking manual action. Never submit with a fresh ID merely because a client
timed out; this can double GPU charges. The job status store is not a
permanent vector database: persist verified results in your application.

Bearer-token rotation does not alter the backend job key and does not change
new notebook-generated IDs for the same deployment/model/input profile.
Do not treat credential rotation as a reason to re-embed inputs.

## Controlled reconciliation for overdue async jobs

This is a **manual operator procedure**, not an automatic retry or a claim that
the currently deployed Modal app includes the latest source revision:

1. Record the existing job ID, state, creation/update timestamps, and source
   deployment revision without recording tokens, image bytes or full vectors.
2. Use authenticated **GET** of the original job status and inspect the CPU
   coordinator/Modal Dict and function logs for its exact job ID. Check GPU
   worker logs only to establish whether the original operation was accepted,
   is still active, or reached a terminal outcome; do not send inference POST.
3. If a terminal result is present, verify model, revision, dimension and input
   digest before persisting it in a durable application database.
4. If dispatch acknowledgement was lost or the job remains uncertain, retain
   its idempotency key and escalate for operator review. Never automatically
   delete/requeue it, mint a different request ID, or infer GPU termination
   from an expired heartbeat.
5. Modal Dict job records are documented to expire after approximately seven
   days without activity. Copy verified completed results to durable storage.
   A missing cache plus expired backend record can require explicit approval
   for a new billable inference; it is not a safe automatic recovery.

For an authorized read-only check, export `VISION_EMBEDDINGS_API_TOKEN` from
an approved secret manager (do not put the token in command history), then run:

```bash
python scripts/inspect_async_job.py --base-url https://YOUR_DEPLOYMENT.modal.run \
  --request-id demo-YOUR_STABLE_REQUEST_ID
# Alternatively use --job-id with the existing 64-character job ID.
```

The inspector performs **one authenticated GET**, never POST, and reports
redacted state and result metadata without vector bytes, token or raw input.
HTTP 404 means the job record is absent or expired, not that GPU stopped.
Verify source/deployment compatibility before deriving a job key from a
request ID, and inspect Modal logs before any manual new work.

The source coordinator joins its heartbeat task **before** writing either
terminal `SUCCEEDED` or `FAILED`, preventing stale `RUNNING` writes from
reversing a terminal result. This ordering does **not** solve force-kills,
remote GPU uncertainty, or deployment-version drift. No administrative
reconciliation endpoint or automatic redispatch is implemented.

## Names and resource ownership

- Environment variables use the `VISION_EMBEDDINGS_` prefix (for example `VISION_EMBEDDINGS_API_TOKEN`).
- Python import package: `multimodal_embeddings_ai_inference`.
- Modal App: `multimodal-embeddings-ai-inference`.
- Modal Secret: `multimodal-embeddings-api`.
- Model Volumes: `multimodal-embeddings-qwen-weights` and `multimodal-embeddings-wemm-weights`.

Configure `VISION_EMBEDDINGS_QWEN_VOLUME_NAME` and `VISION_EMBEDDINGS_WEMM_VOLUME_NAME` independently for checkpoint downloads and GPU inference. Run `scripts/create_modal_secret.py` to create separate gateway authentication and GPU-worker revision Secrets.

## Setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m pip install -e '.[modal]'
modal setup
```

This command sequence runs locally without a GPU; Modal handles remote model downloads and inference.

## Select a consistent Workspace and Environment

After `modal setup`, inspect the active Modal profile/Workspace and available
Environments. Pick one Environment **before** creating resources. The
`MODAL_ENVIRONMENT` variable is inherited by the Modal CLI, including the
`modal secret create` commands launched by the authentication helper:

```bash
modal profile list
modal environment list
export MODAL_ENVIRONMENT="YOUR_ENVIRONMENT"
modal config show --redact
```

Replace `YOUR_ENVIRONMENT` with an existing Environment shown by Modal.
The active profile selects the Workspace; do not switch profiles or change
`MODAL_ENVIRONMENT` midway. Apps, Secrets and Volumes are looked up in their
respective Environment. Creating checkpoints in one Environment and deploying
into another will leave expected resources missing. See the
[Modal Environments guide](https://modal.com/docs/guide/environments).

## Download and pin checkpoints

```bash
modal run modal_models.py --selection both
```

Record the immutable revisions for [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) and [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B). The currently verified revisions are:

- `Qwen/Qwen3-VL-Embedding-8B`: `2c4565515e0f265c6511776e7193b22c0968ddc7`.
- `tencent/WeMM-Embedding-9B`: `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`.

The SHAs identify public upstream model snapshots; they are **not access tokens**. Do not silently replace them with moving `main` when reproducing results.

Both model-download functions print `READY <repository> @ <40-character-SHA> -> <volume>`
only after a snapshot download and Volume commit succeed. Confirm **both**
outputs, and use the **same exact SHAs** below when creating Secrets; do not
guess from `main`. This step does **not** start an L40S GPU or prove that
the model can load. If a download fails, inspect the Modal function logs,
available storage and model access before changing revision or retrying.

## Reproducible GPU dependencies

The CPU gateway and the Qwen/WeMM GPU workers use separate Python environments. Worker dependencies are pinned in [Qwen lock](../constraints/gpu-qwen.txt) and [WeMM lock](../constraints/gpu-wemm.txt); their editable direct requirements are in the corresponding `.in` files. The two models intentionally use different Transformers and Hugging Face Hub versions.

To intentionally refresh a GPU lock, run the following commands with `uv` on a trusted machine, review the diff, then check GPU inference on the updated environment to associate performance reports with the correct dependency versions:

```bash
uv pip compile constraints/gpu-qwen.in --python-version 3.11 --python-platform x86_64-manylinux_2_28 --no-header --no-annotate -o constraints/gpu-qwen.txt
uv pip compile constraints/gpu-wemm.in --python-version 3.11 --python-platform x86_64-manylinux_2_28 --no-header --no-annotate -o constraints/gpu-wemm.txt
```

These locks target **Python 3.11 / Linux x86_64**. Package resolution is not GPU validation or a guarantee of identical wheel artifacts across package indexes. After deployment, record the actual installed packages (`pip freeze`), CUDA/GPU environment, source SHA and model revisions with the results of your GPU checks so you can reproduce the environment when troubleshooting. Never publish API tokens or private inputs.

## Configure authentication

```bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482
```

The example pins specific model checkpoint revisions. Verify the downloaded checkpoint revisions and replace the example SHAs when they differ before creating the Secret.

The helper creates an API Bearer token, registers two Modal Secrets, and saves the corresponding values on **the computer running the command**, with owner-only file permissions. The completion message prints the local file path, not the token. Treat this local file as the source for client onboarding; do not print, commit or copy it into public notebook code.

**Least-privilege layout:** The gateway attaches `VISION_EMBEDDINGS_MODAL_SECRET_NAME` (`multimodal-embeddings-api`), including the Bearer token and model revisions. The GPU workers attach `VISION_EMBEDDINGS_MODAL_MODEL_SECRET_NAME` (`multimodal-embeddings-model-config`), containing only the two revision SHAs and **no bearer token**. Secret names must be distinct. The helper creates the worker Secret first, but Modal CLI operations are not transactional.

**Existing deployment migration:** Run the helper with the original pinned model revisions and `--force` (without `--rotate-token`) before redeploying `modal_app.py`. This preserves the saved token, creates the new worker Secret and updates gateway configuration. Verify both models after redeployment. Only rotate the token when the clients can be updated.

For an authenticated smoke test on that same computer, load the file without displaying its contents:

```bash
TOKEN_FILE="/path/printed/by/the/helper"
set -a
source "$TOKEN_FILE"
set +a
```

For Colab/Kaggle or another machine, share **only the API token** through a trusted secret manager or a hidden notebook prompt, never a full local credential file.

Loading the file also exports `VISION_EMBEDDINGS_QWEN8_REVISION` and `VISION_EMBEDDINGS_WEMM9_REVISION`. The [HTTPS model check](VERIFICATION.md) requires these expected checkpoint SHAs; they can also be supplied with `--qwen-revision` and `--wemm-revision`. It rejects redirects and revision mismatches before reporting PASS for the affected model.

## Optional multi-client API keys and durable async admission

The original owner Bearer token remains supported as client `owner`. For independent clients, configure the **gateway-only** environment variable `VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON` as a JSON object mapping opaque client IDs (2–64 allowed letters/digits/underscore/hyphen) to 1–4 lowercase SHA-256 hex digests. For example, `{"client_a":["<64-HEX-SHA256>"]}`. Generate independent random tokens with a cryptographically secure generator; compute SHA-256 locally, distribute the **raw token only to its intended client** via a secret manager, and keep just the digest in the gateway configuration. Never put tokens, digests, or the owner secret in GPU worker Secrets. Duplicate digests, reserved client ID `owner`, and malformed registries fail startup.

**Offline client key lifecycle (no network/GPU):** Two CPU-only CLI helpers create and stage the complete configuration. `scripts/manage_client_keys.py` keeps a 0600 JSON digest registry and writes each newly generated raw token to a different, non-overwriting 0600 file. It never prints raw tokens; client IDs remain stable through rotation. The private directories must have mode 0700.

```bash
# The previous setup helper displays the private owner credential file path.
OWNER_CREDENTIAL_FILE="/path/previously/printed/by/setup/helper"
KEY_DIR="$(dirname "$OWNER_CREDENTIAL_FILE")"
# Examples only: execute manually on your trusted operator computer.
python scripts/manage_client_keys.py create --client-id client_a \
  --registry "$KEY_DIR/client-keys.json" --token-out "$KEY_DIR/client-a.key"
python scripts/manage_client_keys.py rotate --client-id client_a \
  --registry "$KEY_DIR/client-keys.json" --token-out "$KEY_DIR/client-a-new.key"
python scripts/manage_client_keys.py inspect --registry "$KEY_DIR/client-keys.json"
# After securely distributing and activating the new key, retire the old digest:
python scripts/manage_client_keys.py revoke-key --client-id client_a \
  --registry "$KEY_DIR/client-keys.json" --digest '<OLD_KEY_SHA256_HEX>'
# To revoke every key: revoke-client --client-id client_a --confirm-revoke-all
```

`scripts/client_registry_bundle.py` takes the original owner-only credential file from `scripts/create_modal_secret.py` (Bearer token and both pinned revisions), merges the current client registry, preserves any other `VISION_EMBEDDINGS_*` owner settings and stages a **new** complete 0600 dotenv outside the repository. It does not contact Modal, overwrite a running Secret or reveal credentials. If neither owner file nor registry declares any clients, staging fails closed unless `--confirm-empty-registry` is supplied. A missing registry is **not** silently treated as a revocation.

```bash
python scripts/client_registry_bundle.py --owner-file "$OWNER_CREDENTIAL_FILE" \
  --registry "$KEY_DIR/client-keys.json" --output "$KEY_DIR/gateway-next.env"
# Review the staged file PRIVATELY, checking the original token and pinned
# revisions before applying it with the authenticated Modal CLI.
# When approved by the operator only, update the existing gateway Secret:
# modal secret create --from-dotenv "$KEY_DIR/gateway-next.env" \
#   --force multimodal-embeddings-api
```

The basic `scripts/create_modal_secret.py --force` helper does **not** automatically retain an externally managed client registry: do not invoke it after onboarding unless its complete registry is also restored. Changing a Modal Secret does not update existing running containers immediately; the operator must deploy/restart when appropriate and verify new credentials without triggering GPU inference. **Do not `source` the staged JSON-bearing dotenv in Bash**; use it as input to a dotenv-aware Secret tool, not as a shell script. Never commit the owner file, staged dotenv or raw tokens.

Rotation: add a second digest for the same client, distribute its new raw token privately, then remove the old digest and update the gateway Secret. Revocation removes that client's digest(s). Existing authorized jobs remain owned by the **stable client ID**, not the particular key, and therefore do not cause GPU re-embedding when a key rotates. A revoked client has no further access to its jobs. Legacy records without `client_id` are accessible only to `owner`.

Async limits: `VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL=8` and `VISION_EMBEDDINGS_MAX_ASYNC_JOBS_PER_CLIENT=2` bound outstanding accepted jobs via persisted Modal Dict permit claims; `VISION_EMBEDDINGS_MAX_CLIENT_EMBEDDINGS_PER_MINUTE=60` bounds process-local embedding units per minute for each non-owner client (owner retains the legacy global limit of 120). The global process-local limit and synchronous inflight limit still apply. HTTP 429 with `Retry-After` means capacity/rate exhaustion; do not change request IDs to bypass limits. Job completion releases permits only after definitive terminal persistence. A once-only durable coordinator-start claim prevents duplicate detached coordinators from issuing two GPU calls for one job; a separate per-slot release marker prevents concurrent releases from deleting a newly assigned permit. Failure during either claim or release may strand capacity and requires operator reconciliation.

**Failure model:** Modal Dict has atomic conditional puts for individual permit keys, but not an all-or-nothing transaction across global + client permits. An interruption may leak permits, intentionally denying further paid jobs rather than admitting unbounded work. Uncertain queued/running jobs **must not be automatically freed, deleted, resubmitted or retried**. An operator must first reconcile job status, Modal execution logs and billing before carefully recovering capacity. Modal Dict entries expire after inactivity (currently documented as seven days), so these permits are not a permanent financial cap. Keep workspace budget and spend limit enabled. The gateway and coordinator must both use the same named job Dict in the same Modal Workspace/Environment; local per-process limits still apply as defense in depth; multi-gateway deployments additionally require the opt-in Redis/Valkey shared limiter described below.

## Distributed rate limiting and gateway overload protection

**Opt-in for existing single-gateway deployments; mandatory for multiple gateways.**
The default gateway still has `max_containers=1`. To enable independent CPU
gateways (up to eight), configure a Redis/Valkey service with TLS, authentication,
network restrictions and persistence, then set the following values on the
**gateway only**, never on detached coordinators or the two GPU workers:

- `VISION_EMBEDDINGS_RATE_REDIS_URL`: private `rediss://...` connection URI,
  provisioned in the gateway Secret alongside the existing owner Bearer token
  and pinned revisions. TLS is mandatory except for local loopback tests.
- `VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT=1`: explicitly enable the shared
  limiter (requires a valid Redis URL). If the flag is **unset or empty**,
  `Settings.from_env()` enables it automatically **only when** a Redis URL
  is configured. Explicit `0` disables it even if the URL is present;
  multiple gateways then fail closed at configuration validation.
  **The supplied `.env.example` sets `0`**, so copying it and adding only
  a Redis URL will not enable distributed limiting. For multiple gateways,
  change the flag to `1` (recommended, explicit), or unset/empty it for auto.
- `VISION_EMBEDDINGS_RATE_REDIS_NAMESPACE`: shared namespace (default
  `multimodal-embeddings-v1`). All gateway containers must use the same value
  and Redis database; changing it resets admission history.
- `VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS=2`: set in the **deployment
  shell before building/deploying** `modal_app.py`, not merely inside a
  runtime Secret. The image carries the same public setting into each gateway.
  Without distributed admission, startup fails closed if this exceeds one.
- `VISION_EMBEDDINGS_MAX_EMBEDDINGS_PER_MINUTE=120` and
  `VISION_EMBEDDINGS_MAX_CLIENT_EMBEDDINGS_PER_MINUTE=60`: global and
  per-client capacities, measured in embedding **units**, not HTTP calls.
- `VISION_EMBEDDINGS_MAX_DISTRIBUTED_SYNC_INFLIGHT=4` and
  `VISION_EMBEDDINGS_MAX_DISTRIBUTED_CLIENT_SYNC_INFLIGHT=2`: shared
  concurrent *synchronous* GPU HTTP admission across all gateways.
- `VISION_EMBEDDINGS_MAX_BODY_REQUESTS=8`: per-gateway bounded concurrent
  authenticated POST-body processing. Health/status GETs remain available.

**Algorithm:** One Redis Lua `EVAL` checks the global and client token buckets
and optional synchronous concurrency leases **atomically**, using Redis server
time. On acceptance it deducts `batch_size` units once from each applicable
bucket. On rejection it deducts nothing. Buckets allow bursts up to capacity and
refill steadily over 60 seconds: this is **not** a strict rolling-window
counter. Keys use opaque hashed client IDs, TTLs and a common Redis Cluster
hash-tag for one atomic shard; no tokens, images or vectors are written there.
A synchronous lease is released early only after a confirmed result. On
timeout, disconnection or uncertain GPU failure, it remains held until the
inference deadline plus 60 seconds. Lease expiry is not proof GPU work stopped.
Async jobs instead use the existing durable global/per-client outstanding
permits; both sync and async *new* work consume shared rate units.

**Failure behavior:** If Redis is unreachable, its reply is malformed or
an acknowledgement is lost, inference returns **HTTP 503** and never falls
back to local rate accounting. Budget/concurrency rejection returns **429**
with `Retry-After`, and bounded body capacity also returns 429. The metadata
and health endpoints continue to work after successful Bearer authentication.
The legacy in-process guard remains an additional per-gateway layer; it is
not the distributed accounting authority.

**Operational constraints:** The Redis/Valkey account must permit `EVAL`,
`TIME`, `HGET/HSET`, `EXPIRE`, `ZADD/ZCARD/ZREM/ZREMRANGEBYSCORE`.
Provision secure persistent storage, backups and monitoring for resets/failover.
A datastore reset can reset active quota and concurrency history; this is **not**
a billing hard stop. Keep Modal workspace budget and spend limits enabled.
Apply the updated gateway Secret through the private staged-file workflow
above; do not print the Redis connection URI or include it in the repository,
notebooks or worker Secrets. This release provides CPU-only tests with a
shared in-memory Redis emulator; actual provider latency and HA/failover
behavior must be validated separately before increasing gateway replicas.

## Failed Bearer-token attempts and temporary lockouts

Protected API routes validate authentication before reading the request body or dispatching GPU work, including mounted apps and path prefixes. For each ASGI client peer, the first two invalid or missing tokens receive HTTP 401; the third failure within five minutes returns HTTP 429 and blocks every further authentication attempt for 15 minutes. Locked peers are rejected before token comparison, including requests with the correct token. `Retry-After` states the remaining delay; requests during the lock do not extend it. Successful authentication before a lockout clears prior failures. Clients sharing a NAT or proxy address share the lockout.

The policy is configurable in the gateway environment. Default values are: VISION_EMBEDDINGS_AUTH_FAILURE_LIMIT=3, VISION_EMBEDDINGS_AUTH_FAILURE_WINDOW_SECONDS=300, VISION_EMBEDDINGS_AUTH_LOCKOUT_SECONDS=900, VISION_EMBEDDINGS_AUTH_MAX_TRACKED_CLIENTS=4096.

**Proxy trust:** The application deliberately does not read user-provided X-Forwarded-For or Forwarded headers for access control. It uses the ASGI peer address, which must only be modified by a verified trusted proxy. If Modal or another proxy presents the same peer address for many clients, they share this counter and lockout: original-client IP attribution is not guaranteed without verifying the deployment's proxy chain.

**Scope:** The thread-safe, memory-bounded limiter is local to each CPU gateway process; counters and locks reset on restart or tracked-peer eviction. The current Modal gateway limits container count to one, but horizontal scaling requires a shared atomic store or trusted edge/WAF rate limiter. A rotating-IP distributed attack can still evade per-peer protection; use edge controls where needed. Use long random tokens and rotate exposed credentials. Modal proxy authentication is an optional additional layer that requires client changes.

## Deliberate API token rotation

`--force` allows replacing the existing Modal Secret but **reuses a saved token**. `--rotate-token --force` generates a new token, updates Modal and replaces the local file only after Modal CLI succeeds. Do not delete the existing token file to trigger rotation.

~~~bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482 \
  --force --rotate-token
~~~

Prepare client updates before rotating. Depending on Modal's running-container lifecycle, restart or redeploy the API so existing instances use the updated Secret. After rollout, verify the new token works and the old token receives HTTP 401. The helper retains the old local token file if Modal CLI fails. If the local file is lost, --force generates a new token.


## Deploy

```bash
modal deploy modal_app.py
```

Use the deployment URL printed by your own Modal account. Configure it in your API caller or server-side secret manager; do not hardcode a live service URL in the repository.

## Check the deployment

First confirm the App and resource names in the selected Environment without
starting GPU work:

```bash
modal app list
modal volume list
modal secret list
```

The following commands are **separate, billable inference checks**, not
read-only setup tests. Use an image you are permitted to process and approve
their possible GPU charges:

```bash
# Direct Modal SDK to both GPU workers (does not test HTTPS authentication).
python scripts/smoke_modal_gpu.py --image /path/to/authorized-image.png

# HTTPS end-to-end: load the owner-only token/revision file as shown above.
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png --dimension 1024
```

Replace the URL with the **HTTPS base URL of your own deployed App**, not an
author's endpoint. The direct worker check validates model execution but
bypasses the gateway's Bearer authentication. The HTTPS check exercises the
gateway, both model routes, expected pinned SHA and normalized vectors.
Run it promptly after the direct checks while workers may still be warm:
the synchronous HTTPS inference route has a **60-second default server
deadline**, so an initial cold start may return HTTP 504. A timeout is **not
permission to retry automatically**; inspect Modal logs and, for cold-start
clients, use the explicitly approved
[async flow](API_CONTRACT.md#async-gpu-jobs-cold-starts-and-resume)
with a stable request ID. Follow [Verification](VERIFICATION.md) for PASS
criteria and recovery.

## Access control and spending safeguards

**Before opening a hosted API to untrusted clients, the operator must set both a Workspace budget (monthly total usage cap before credits) and a Workspace spend limit (monthly out-of-pocket charge cap after credits)** in Modal's **Usage & Billing** settings. These are enforced by Modal, unlike billing alerts. Remaining applicable credits may continue to fund workloads after the spend limit is reached, until the Workspace usage budget is exhausted. Owners/Managers configure the limits; source code and CI cannot verify their account-specific values. See the [official Modal budgets guide](https://modal.com/docs/guide/budgets). On supported Team/Enterprise workspaces, an Environment compute budget adds isolation but does not replace the Workspace caps.

The reference deployment limits the CPU gateway to **one container** and each GPU model worker to one container, with `min_containers=0`. The gateway applies best-effort limits of **120 embedding items/minute** and **4 in-flight embedding requests** by default (HTTP 429 and `Retry-After`). Each batch item consumes one unit. The counters reset after gateway restart and are **not durable distributed quotas, per-client quotas or monetary caps**; other environments and overlapping deployments can still incur charges.

The CPU gateway accepts **16 HTTP inputs concurrently** in that container, leaving room for health checks and immediate quota responses while GPU calls wait. The synchronous route admits at most 4 inference requests by default; each GPU worker processes one input at a time. With ten overlapping synchronous requests while four slots are occupied, the other six receive HTTP 429 without entering the GPU queue. If you raise the inference limit, keep it below 16 so control requests have capacity; an invalid configuration is rejected at startup. See [Modal input concurrency](https://modal.com/docs/guide/concurrent-inputs).

`VISION_EMBEDDINGS_INFERENCE_TIMEOUT_SECONDS` defaults to 60 (maximum 120), including GPU scheduling and execution. Expiry returns HTTP 504 with `Retry-After`, releases the inference slot and requests cancellation of the owned remote inputs. The workers process a single input at a time, allowing cancellation to terminate the affected container without interrupting another concurrent input. A failed batch group also cancels unfinished sibling work. A short, bounded cleanup may add up to 5 seconds to the response.

For multiple clients, configure **per-client rate limits and quotas at an authenticated ingress/proxy**, and never share the application Bearer token with untrusted callers. Monitor resource usage, ensure billing guards are configured, and retain an operator shutdown path (`modal app stop multimodal-embeddings-ai-inference`). Before sharing the service with others, verify the Workspace budget, spend limit and ingress controls in your Modal account.

## Durable async jobs and cold GPU scheduling

For new clients, use the [async endpoint](API_CONTRACT.md#async-gpu-jobs-cold-starts-and-resume) instead of repeatedly POSTing to the 60-second
synchronous route. The detached `run_embedding_job` CPU function can wait
for GPU scheduling and has `timeout=3600`; the two L40S GPU workers remain
`min_containers=0` to avoid paying for permanently warm containers.
Job metadata and completed vectors live in a dedicated Modal Dict named
`multimodal-embeddings-async-jobs` by default
(`VISION_EMBEDDINGS_JOB_DICT_NAME` overrides the name).

New async submissions use a short-lived gateway request slot while waiting
for dispatch acknowledgement, plus **durable global and per-client
outstanding-job permits** retained until the definitive terminal result.
These are separate from synchronous inflight limits. The detached coordinator
receives the model revision Secret and checks the accepted encoding profile
before dispatching GPU work. Uncertain jobs may keep permits indefinitely
until carefully reconciled; do not clear them merely because a heartbeat ages.

Do not delete the Dict until results have been copied to durable application
storage. Dict records expire after seven days of inactivity. If dispatch or
client acknowledgement is uncertain, use authenticated **GET
/v1/embeddings/jobs/by-request/{request_id}** to find its tenant-scoped job ID,
or GET the saved job ID if already known. Inspect the deployment. Do not POST
again merely because the acknowledgement was lost:
an identical request can start new billable work if its record is missing or
expired. Missing records require investigation and an explicit decision before
submitting new work. There is no guaranteed GPU capacity or automatic retry for
terminal failures. Async jobs may incur GPU costs after the client disconnects.
A local mock server does not configure the Modal job coordinator; use the
synchronous endpoint for CPU-only mock tests.

## Updating checkpoint revisions and workers

Deploy the gateway, detached coordinator and both GPU worker classes from the
same source after setting pinned revision Secrets and downloading the matching
snapshots. Replace old worker containers as part of that deployment; changing a
Secret does not relabel weights already in memory. Do not replace model Volume
contents while active workers are loading from them.

The supervisor compares the request's expected revision with the subprocess's
load-time identity before encoding, and the service validates the returned
model/repository/SHA before accepting vectors. A warm worker with an older SHA
or a legacy worker without identity fails closed. Redeploy the complete app
rather than updating only the coordinator. Existing failed async IDs remain
terminal: inspect the failure before explicitly authorizing any new billable job.

## Logs, reliability and cost

```bash
modal app list
modal app logs multimodal-embeddings-ai-inference
```

Treat provider logs as private operational data. Avoid logging tokens, uploaded images, raw embeddings or identifiable input metadata.
Requests can incur cold-start and queueing latency. Account for GPU, CPU, storage and network charges; configure budget protection and rate limits before inviting external users.

For API fields, see [API contract](API_CONTRACT.md); for verification procedures, see [Verification](VERIFICATION.md).
