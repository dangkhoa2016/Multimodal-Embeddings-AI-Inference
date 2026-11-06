# Security and monitoring

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

This page documents the controls in source version **1.0.0**. It does **not**
certify an actual Modal deployment, Redis operator configuration, external
infrastructure, key storage, or model inference result.

## Scope of security controls

The project implements API-level safeguards for owner and optional independent
clients. These controls do not constitute a complete end-user authentication,
organization management or enterprise multi-tenant authorization system.
Per-client tokens and async job ownership must not be presented as full
business tenant isolation, billing controls or a security certification.

A shared/business deployment requires its operator to assess end-to-end
tenant separation, user authorization, secret management, abuse protection,
auditability, incident response and provider spending limits. See the
[deployment scope](OPERATIONS.md#deployment-scope-and-business-adoption) and
the operator checklist below.

## Trust boundaries

- Every owned `/v1/*` API route **except public GET /v1/models** checks a Bearer token before reading a
  request body. An owner key is managed separately from optional SHA-256
  digest-based per-client keys; stable client IDs scope async job access.
- `GET /v1/ops/metrics` is **owner-only**. A non-owner with a valid
  client key receives HTTP 404; unauthenticated requests receive HTTP 401
  (or lockout). Never share the owner key with untrusted clients.
- Redis/Valkey is an optional shared rate/concurrency ledger for multiple
  CPU gateways. Its credential stays in the **gateway-only** Secret;
  GPU workers receive model revisions but no API tokens/Redis URL.
- Modal Dict retains async job metadata and completed vectors. Neither
  a Redis rate bucket nor a Dict permit is a guaranteed billing ceiling;
  configure the Modal Workspace budget and spend limit.

## Privacy-safe metrics and error handling

The owner-only metrics route exposes only **bounded aggregates**: counts
by fixed route category and HTTP status, observed async response states,
known middleware rejection reasons, and gateway uptime. It deliberately
does **not** expose raw URLs, source IPs, request IDs, tenant identifiers,
Bearer tokens/digests, prompts, images, vectors or stack traces. Responses
have `Cache-Control: no-store`.

Metrics have scope `gateway_process_only`: they reset on container restart
and are **not** aggregated across Redis gateways or detached coordinators.
`async_job_state_observations` increments when a job is submitted or polled;
it is **not** the number of currently running GPU jobs. For infrastructure-wide
counts, use your protected provider logs and external metrics service with
redaction and narrowly scoped access. Do not scan the full Modal Dict on each
metrics request.

Pydantic request validation responses suppress the attacker-supplied
`input`/`ctx` payload, and invalid-image errors suppress Pillow exception
text. Store/provider failures use fixed operator log messages without
exception contents, image payloads or credentials. Avoid logging raw HTTP
request/response bodies at reverse proxies as well.

## Offline security checks (CPU only)

```bash
python scripts/security_audit.py
python -m pytest -q tests/test_security_monitoring.py
```

The script validates selected static security invariants and rejects
known sensitive tracked filenames. Runtime regression tests exercise
owner-only telemetry, token redaction, sanitized 422 responses, backend
failure behavior and finite label cardinality. CI runs these checks without
Modal credentials, network calls or GPU. Source checks are **defense in
depth**, not a substitute for dependency CVE review, formal penetration
testing, live Redis failure-injection, security incident response, or a
review of Cloud/Modal networking and identity policies.

## Production operator checklist

1. Keep the owner token in an owner-controlled secret manager and client keys
   separate; rotate or revoke keys by client ID using the offline tools in
   [Operations](OPERATIONS.md#optional-multi-client-api-keys-and-durable-async-admission).
2. Require TLS and least-privilege access to Redis/Valkey; protect Modal
   Secrets, Volumes, service URLs and logs. Preserve the client registry and
   pinned checkpoint revisions across Secret updates.
3. Monitor HTTP 401/429/503, async `INDETERMINATE`, admission outages,
   stored-lease reconciliation, Redis resets, worker failures and actual
   spending in Modal. An in-process metric is not proof of global job state.
4. For an uncertain POST/dispatch/timeout, use the stable async ID and
   authenticate GET polling; never automatically retry GPU inference or
   clear permits solely due to elapsed heartbeat age.
5. Validate deployment-specific failure and credential rotation procedures
   **before** increasing gateway replicas. No live security certification
   is implied by a passing CPU test suite.
