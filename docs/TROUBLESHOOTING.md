# Troubleshooting

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](TROUBLESHOOTING.vi.md)

## Multi-client and async admission diagnostics

If one client gets HTTP 404 for another client's job ID, this is intentional tenant isolation, not evidence that the job disappeared. A rotated key must retain the same client ID; removing all its digests revokes polling rights. HTTP 429 with `Async outstanding job capacity reached` indicates durable global/per-client job permits (not the 4-request gateway inflight guard). HTTP 503 with `Job registry temporarily unavailable` or `Job admission temporarily unavailable` must be treated as **uncertain**: poll the original stable request/job ID after recovery and inspect operator logs; do not generate a different ID or automatically resubmit. Leaked permits may need operator reconciliation; never remove a permit solely based on heartbeat age. Durable `job:start:v1:` and `permit:release:` markers are deliberate once-only safeguards. Never clear the entire Modal Dict to recover capacity, since it also contains idempotency records and completed vectors. Use `scripts/manage_client_keys.py` and `scripts/client_registry_bundle.py` for offline credential rotation; stage privately and apply manually. When replacing the Modal gateway Secret, preserve `VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON`; the basic Secret creation helper does not preserve externally managed client keys.

## Shared Redis/Valkey admission diagnostics

When distributed admission is enabled, an inference POST can return HTTP
**429** for `Distributed embedding budget exceeded`,
`Distributed synchronous concurrency limit reached`, or
`Gateway request body capacity reached`. Follow the `Retry-After`
header; do not automatically re-POST synchronous inference or change an
accepted async request ID to bypass the budget. HTTP **503**
`Distributed inference admission unavailable` indicates that Redis was
unreachable or its acknowledgement was uncertain. Inspect the operator's
Redis connectivity/TLS, ACLs, failover and namespace; do not switch to a
different per-process limiter while troubleshooting. Health GET can still
respond even when Redis is down.

An expired sync lease does not prove that GPU work stopped. A Redis reset,
namespace change or removed rate keys can unexpectedly reset quotas:
reconcile Redis state with Modal logs and the Workspace budget/spend limit.
All gateway replicas must point to the same Redis database and namespace;
multi-gateway operation is disallowed when shared admission is disabled.
Further configuration is described in [Operations](OPERATIONS.md#distributed-rate-limiting-and-gateway-overload-protection).

## Common issues

| Symptom | Likely cause | Recommended action |
| --- | --- | --- |
| Missing Secret/Volume or App not found | Download, Secret creation and deployment used different Workspaces/Environments, or a resource was not created | Verify `modal profile list`, `modal environment list`, `MODAL_ENVIRONMENT`, then inspect `modal app list`, `modal volume list` and `modal secret list` in the same Environment; do not switch context between steps |
| Checkpoint download fails | Model access, network, CPU execution time or storage quota is insufficient | Inspect the specific `modal_models.py` function log; verify Hugging Face access, pinned SHA and available Volume storage before retrying |
| L40S unavailable or GPU allocation denied | Workspace account/billing state, L40S quota or temporary GPU capacity | Check workspace billing and GPU quota in Modal; deployment success alone does not reserve a GPU |
| HTTP `401` | Missing or invalid Bearer token | Verify server-side `VISION_EMBEDDINGS_API_TOKEN` without printing its value |
| HTTP `413`/`422` | Oversized input or invalid schema | Check Base64, image limits, model ID, dimension and batch size |
| HTTP `503` | Inference worker unavailable | Inspect protected provider logs, checkpoint revision and available GPU memory |
| HTTP `429` | Authentication lockout, occupied inference slots or exhausted unit budget | Check `detail`; for an auth lockout verify the token and wait for `Retry-After`, otherwise reduce overlapping requests |
| HTTP `504` | GPU scheduling or execution exceeded the deadline | Inspect Modal queue/capacity and worker logs before any new billable request; never automatically retry a timed-out POST |
| Model loading error | Dependency or checkpoint mismatch | Rebuild pinned environments and verify checkpoint integrity |
| High latency | GPU worker cold start or dispatch queueing | Batch by model and measure cold/warm requests separately |
| GPU out of memory | Model activations or oversized inputs | Reduce batch/image dimensions; keep only one model process active |
| Unexpected vector dimensions | Wrong model revision or client assumptions | Reject mismatches; never silently pad or truncate vectors |

## First-time installation recovery

1. Start from [Operations](OPERATIONS.md#first-time-deployment-prerequisites-and-order) and confirm billing safeguards and the selected Workspace/Environment.
2. Check for **two** completed `READY` checkpoint downloads with the **same SHAs** used in the two Secrets. Do not mix a default/moving `main` with pinned revisions.
3. Use [Verification](VERIFICATION.md#verify-resources-and-gateway-before-gpu-work) to check resources and public CPU health **before** invoking any GPU job.
4. When a direct GPU smoke test passes but HTTPS fails, investigate gateway URL, authentication, quota and deployment logs; direct Modal SDK calls bypass the HTTPS gateway.
5. If the first synchronous HTTPS request returns `504`, inspect whether work was accepted before deciding on any new billable request. **Never auto-retry a timed-out POST**; prefer the async endpoint with a stable request ID for cold starts.

Do not expose Bearer tokens, full embeddings or private input images in issues or diagnostics.

## After deployment

1. Run offline tests, lint and the SHA256 source manifest check.
2. Download and verify both full model checkpoints: `Qwen/Qwen3-VL-Embedding-8B` and `tencent/WeMM-Embedding-9B`.
3. Confirm public `GET /health` and `GET /v1/models` work without a token; verify private inference and job routes still reject unauthenticated requests.
4. Confirm both models return finite 1,024-dimensional, normalized vectors on the same authorized image.
5. Confirm model routing and sanitized error responses, and monitor GPU memory under load.
6. Evaluate retrieval quality with examples representative of your application.

For functional smoke-test procedures, see [Verification](VERIFICATION.md).
