# Durable asynchronous inference and recovery

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ASYNC_RELIABILITY.vi.md)

This guide explains how to operate the optional async embedding API when a
Modal GPU invocation might outlive an HTTP request. It does not replace the
[API contract](API_CONTRACT.md), [deployment guide](OPERATIONS.md) or the
Modal workspace's billing budget.

## Request identity and the GPU boundary

A newly accepted async request binds its `request_id` to the model, input,
checkpoint revision and embedding configuration. Poll by `job_id`, or use
authenticated `GET /v1/embeddings/jobs/by-request/{request_id}` if the POST
acknowledgement was lost. These GET calls do not submit GPU work.

`QUEUED` and `RUNNING` are not evidence of success. `INDETERMINATE`
indicates that a deadline, heartbeat or remote acknowledgement cannot establish
the outcome. In particular, accepting a cancellation request does **not**
prove that the GPU stopped. Do not create a replacement request ID or
automatically submit again. A missing/expired job record also does not prove
that work was never dispatched.

## How outstanding capacity is reserved

The coordinator uses two logical permit pools: a global pool and one pool per
authenticated client. Each slot has a monotonically advancing sequence of
**versioned claims**. Once a definitive result is durably saved, the owner
writes an immutable release receipt for its exact claim and acquisition ID.
The next generation can be admitted only after that receipt matches the
preceding claim. No automatic release removes a reusable Modal Dict key.

This add-only protocol avoids a get-then-pop race: a delayed release cannot
delete another request's claim after key expiry/reallocation. The cursor is an
advisory optimization only; missing predecessors, mismatched receipts, lost
storage acknowledgements and old unversioned permits fail closed, possibly
reducing available capacity. An individual Dict entry can expire after seven
days of inactivity, so this is not an indefinite spending ceiling.

Recovery of a stale advisory cursor uses exponential and binary probes over matching immutable claim/release receipts, rather than scanning every intermediate generation. A missing or expired predecessor receipt is **not** evidence of free capacity: admission fails closed. Positive checkpoints from earlier releases can anchor recovery across old data. Gateway diagnostics record only the aggregate number of store reads and reserve duration (store_reads, elapsed_ms); no credential, request payload, vector or client identifier is logged. The read bound applies to intact contiguous release evidence; it is not a guarantee against missing/expired Modal Dict records or backend latency.

## Versioned admission migration: drain required

**Mixed v1/v2 admission writers are not supported.** A v1 gateway only
recognizes unversioned `permit:global:0` and `permit:client:*` keys, while v2
uses immutable `:v2:` generation claims. The v2 check for legacy permits is
one-way; a v1 writer can still accept work while a v2 permit is active.
Redis/Valkey rate limits do not fix this durable admission incompatibility.

Use a **drain-only / maintenance-window** migration: (1) stop all new async
admissions on the old version, including in-flight HTTP requests awaiting
admission; (2) confirm that all old gateway/coordinator instances have
terminated and cannot write more legacy permits; (3) reconcile each outstanding
legacy job/permit with authoritative read-only status and Modal logs, keeping
uncertain work occupied; (4) bring up only the new v2 admission writer, then
verify no legacy writer can resume. A rolling deployment with both old and new
writers is unsafe. Never bulk-delete the job Dict or remove unresolved permits.
If the old admission component cannot be fully drained, **do not enable v2**
against the same store.

## Operator reconciliation

1. Locate the original job using read-only status APIs and the
   [job inspector](../scripts/inspect_async_job.py). Keep its request ID,
   model and revision unchanged.
2. Correlate the job with Modal gateway, CPU coordinator and GPU worker logs.
   Verify whether a remote call has actually terminated before changing any
   capacity record; an expired heartbeat or accepted cancellation is not proof.
3. Preserve successful vectors in durable application storage. Treat jobs with
   unknown outcomes and their permits as active until reconciled.
4. For migration from unversioned permits, review outstanding jobs before
   replacing/retiring any legacy records. The service intentionally does not
   delete old slot keys. Never clear the whole Dict as a routine repair.

For deployments that run multiple HTTP gateways, configure shared
Redis/Valkey rate limiting and synchronous concurrency control separately;
this is distinct from durable async permits. Keep a workspace budget and
spend limit because rate limits and admission do not cap provider charges.

## Notebook recovery

The Demo Notebook keeps an optional private checkpoint journal. With `RUN_API_DEMO=False`, the notebook performs preview only, even with
credentials configured. `RUN_API_DEMO=True` permits potentially billable
GPU jobs, but uncertain work is recovered by GET only, not resubmitted.
Malformed read-only recovery responses halt the run while retaining the
journal; they never turn into an automatic POST. Keep the journal private
because successful records can contain embedding vectors.
