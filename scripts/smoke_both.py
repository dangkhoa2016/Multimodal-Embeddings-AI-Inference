#!/usr/bin/env python3
"""Authenticated HTTPS embedding smoke test with read-only async recovery."""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MODELS = ("qwen3-vl-embedding-8b", "wemm-embedding-9b")
REPOSITORIES = {
    MODELS[0]: "Qwen/Qwen3-VL-Embedding-8B",
    MODELS[1]: "tencent/WeMM-Embedding-9B",
}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
TERMINAL_STATES = frozenset(("SUCCEEDED", "FAILED", "INDETERMINATE"))


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a Bearer token to another origin, even on a 307/308.
        return None


def _emit(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


def _request(opener, base_url, token, path, *, payload=None, timeout=30):
    headers = {"Authorization": "Bearer " + token}
    method = "GET"
    body = None
    if payload is not None:
        method = "POST"
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + path, data=body, headers=headers, method=method,
    )
    with opener.open(request, timeout=timeout) as response:
        return json.load(response), response.status, response.headers


def _preflight(opener, base_url, token, models, revisions, allow_mock, timeout):
    health, status, _ = _request(
        opener, base_url, token, "/health", timeout=timeout,
    )
    if status != 200 or health.get("status") != "ok":
        raise ValueError("Gateway health preflight failed")
    if not allow_mock and health.get("backend") != "modal":
        raise ValueError("Preflight backend is not Modal")
    catalog, status, _ = _request(
        opener, base_url, token, "/v1/models", timeout=timeout,
    )
    if status != 200 or not isinstance(catalog.get("models"), list):
        raise ValueError("Model catalog preflight failed")
    by_id = {entry.get("id"): entry for entry in catalog["models"]}
    for model in models:
        entry = by_id.get(model)
        if entry is None:
            raise ValueError("Requested model absent from the catalog: " + model)
        expected = revisions[model]
        if expected and entry.get("revision") != expected:
            raise ValueError("Unexpected catalog revision for " + model)
    _emit(
        stage="preflight", status="PASS", backend=health["backend"],
        models=list(models), gpu_readiness="not_checked",
    )


def _stable_request_id(base_url, model, digest, dimension, revision):
    material = json.dumps(
        [base_url, model, digest, dimension, revision], separators=(",", ":"),
    ).encode("utf-8")
    return "smoke-" + hashlib.sha256(material).hexdigest()[:48]


def _request_id(args, model, digest, revision):
    if args.request_id:
        suffix = "" if args.model != "both" else (
            "-qwen" if model == MODELS[0] else "-wemm"
        )
        request_id = args.request_id + suffix
    else:
        request_id = _stable_request_id(
            args.base_url, model, digest, args.dimension, revision,
        )
    if not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise ValueError(
            "Invalid request_id (8-128 URL-safe characters, beginning with a letter or digit)"
        )
    return request_id


def _validate(result, model, digest, dimension, revision, allow_mock):
    if not isinstance(result, dict):
        raise ValueError("Embedding response is not an object")
    if result.get("model") != model:
        raise ValueError("Unexpected model in embedding response")
    if not allow_mock and result.get("backend") != "modal":
        raise ValueError("MOCK backend is not real inference")
    if revision and result.get("revision") != revision:
        raise ValueError("Unexpected checkpoint revision for " + model)
    if "repository" in result and result["repository"] != REPOSITORIES[model]:
        raise ValueError("Unexpected checkpoint repository for " + model)
    if result.get("dimension", dimension) != dimension:
        raise ValueError("Unexpected response dimension")
    if result.get("input_sha256") != digest:
        raise ValueError("Input digest mismatch")
    vector = result.get("vector")
    if not isinstance(vector, list) or len(vector) != dimension:
        raise ValueError("Embedding dimension mismatch")
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in vector):
        raise ValueError("Embedding contains nonfinite or invalid values")
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if not math.isfinite(norm) or abs(norm - 1.0) >= 0.001:
        raise ValueError("Embedding is not L2-normalized")
    return {
        "model": model,
        "backend": result["backend"],
        "dimension": dimension,
        "revision": result.get("revision"),
        "revision_verified": bool(revision),
        "input_sha256_verified": True,
        "elapsed_ms": result.get("elapsed_ms"),
        "timing": result.get("timing"),
        "status": "PASS",
    }


def _lookup(opener, args, token, request_id, *, job_id=None):
    if job_id:
        path = "/v1/embeddings/jobs/" + urllib.parse.quote(job_id, safe="")
    else:
        path = (
            "/v1/embeddings/jobs/by-request/"
            + urllib.parse.quote(request_id, safe="")
        )
    return _request(
        opener, args.base_url, token, path, timeout=args.http_timeout,
    )


def _poll(opener, args, token, model, request_id, initial, job_id, started,
          digest, revision):
    deadline = time.monotonic() + args.poll_timeout
    record = initial
    while True:
        if record is not None:
            state = record.get("state")
            if state == "SUCCEEDED":
                result = _validate(
                    record.get("result"), model, digest, args.dimension,
                    revision, args.allow_mock,
                )
                result.update(
                    mode="async", request_id=request_id, job_id=job_id,
                    wall_elapsed_ms=round(1000 * (time.monotonic() - started), 3),
                )
                _emit(**result)
                return 0
            if state in ("FAILED", "INDETERMINATE"):
                _emit(
                    model=model, mode="async", state=state, request_id=request_id,
                    job_id=job_id, status="STOP", action="inspect_job_read_only",
                )
                return 1
            if state not in ("QUEUED", "RUNNING"):
                raise ValueError("Unexpected async job state")
            if record.get("job_id"):
                if job_id and job_id != record["job_id"]:
                    raise ValueError("Async job ID changed")
                job_id = record["job_id"]
            _emit(
                model=model, mode="async", state=state,
                request_id=request_id, job_id=job_id,
                wall_elapsed_ms=round(1000 * (time.monotonic() - started), 3),
            )
        if time.monotonic() >= deadline:
            _emit(
                model=model, mode="async", state="UNKNOWN",
                request_id=request_id, job_id=job_id,
                status="STOP", action="resume_with_--resume_GET_only",
            )
            return 2
        time.sleep(min(args.poll_interval, max(0.0, deadline - time.monotonic())))
        try:
            record, _status, _headers = _lookup(
                opener, args, token, request_id, job_id=job_id,
            )
        except urllib.error.HTTPError as error:
            if error.code == 404 and job_id:
                # Read-only fallback, never another POST.
                record, _status, _headers = _lookup(
                    opener, args, token, request_id,
                )
            else:
                raise
        if record.get("job_id") and job_id and record["job_id"] != job_id:
            raise ValueError("Async lookup returned a different job")


def _async_run(opener, args, token, model, image_data, digest, revision):
    started = time.monotonic()
    request_id = _request_id(args, model, digest, revision)
    _emit(
        model=model, mode="async", request_id=request_id,
        status="LOOKUP", action="read_only_before_submission",
    )
    try:
        existing, _status, _headers = _lookup(
            opener, args, token, request_id,
        )
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        existing = None
    if existing is not None:
        _emit(model=model, mode="async", request_id=request_id,
              status="RESUME", action="GET_only")
        return _poll(
            opener, args, token, model, request_id, existing,
            existing.get("job_id"), started, digest, revision,
        )
    if args.resume:
        _emit(
            model=model, mode="async", request_id=request_id,
            status="NOT_FOUND_OR_EXPIRED", action="STOP_no_POST",
        )
        return 2

    # The ID is printed above before the ONE new billable POST. A lost ACK
    # must never trigger automatic resubmission, even if lookup returns 404.
    body = {
        "model": model, "image_base64": image_data,
        "dimension": args.dimension, "request_id": request_id,
    }
    try:
        ack, status, _headers = _request(
            opener, args.base_url, token, "/v1/embeddings/async",
            payload=body, timeout=args.http_timeout,
        )
        if status not in (200, 202):
            raise ValueError("Unexpected async submission HTTP status")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
        _emit(
            model=model, mode="async", request_id=request_id,
            status="ACK_UNCERTAIN", error_type=type(error).__name__,
            action="GET_by_request_only",
        )
        try:
            ack, _status, _headers = _lookup(
                opener, args, token, request_id,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            _emit(
                model=model, mode="async", request_id=request_id,
                status="UNKNOWN", action="STOP_no_POST_use_--resume",
            )
            return 2
    return _poll(
        opener, args, token, model, request_id, ack, ack.get("job_id"),
        started, digest, revision,
    )


def _sync_run(opener, args, token, model, image_data, digest, revision):
    started = time.monotonic()
    payload = {
        "model": model, "image_base64": image_data, "dimension": args.dimension,
    }
    try:
        response, status, _headers = _request(
            opener, args.base_url, token, "/v1/embeddings",
            payload=payload, timeout=args.http_timeout,
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
        _emit(
            model=model, mode="sync", status="UNKNOWN",
            error_type=type(error).__name__,
            action="STOP_no_retry_sync_is_not_idempotent",
        )
        return 2
    if status != 200:
        raise ValueError("Unexpected synchronous HTTP status")
    result = _validate(response, model, digest, args.dimension, revision, args.allow_mock)
    result.update(
        mode="sync",
        wall_elapsed_ms=round(1000 * (time.monotonic() - started), 3),
    )
    _emit(**result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HTTPS smoke: async is the cold-start-safe default; no automatic GPU retries.",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--model", choices=("both", *MODELS), default="both")
    parser.add_argument("--mode", choices=("async", "sync"), default="async")
    parser.add_argument("--resume", action="store_true",
                        help="Async read-only GET recovery; never submit a GPU job")
    parser.add_argument(
        "--request-id", help="Stable custom ID; adds a per-model suffix for both",
    )
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--poll-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--http-timeout", type=float, default=None,
        help="Timeout per HTTP call (default: async 30s, sync 1800s)",
    )
    parser.add_argument("--allow-mock", action="store_true", help="Local test mode only")
    parser.add_argument(
        "--qwen-revision", default=os.getenv("VISION_EMBEDDINGS_QWEN8_REVISION", ""),
        help="Pinned 40-character checkpoint SHA (or VISION_EMBEDDINGS_QWEN8_REVISION)",
    )
    parser.add_argument(
        "--wemm-revision", default=os.getenv("VISION_EMBEDDINGS_WEMM9_REVISION", ""),
        help="Pinned 40-character checkpoint SHA (or VISION_EMBEDDINGS_WEMM9_REVISION)",
    )
    args = parser.parse_args()
    if args.resume and args.mode != "async":
        parser.error("--resume requires --mode async")
    if args.dimension <= 0 or args.poll_interval <= 0 or args.poll_timeout < 0:
        parser.error("Invalid dimension or polling interval/timeout")
    if args.http_timeout is None:
        args.http_timeout = 30.0 if args.mode == "async" else 1800.0
    if args.http_timeout <= 0:
        parser.error("--http-timeout must be positive")
    selected = MODELS if args.model == "both" else (args.model,)
    revisions = {
        MODELS[0]: args.qwen_revision.strip().lower(),
        MODELS[1]: args.wemm_revision.strip().lower(),
    }
    for model in selected:
        revision = revisions[model]
        if (revision or not args.allow_mock) and not REVISION_PATTERN.fullmatch(revision):
            parser.error("Expected pinned 40-character checkpoint revision for " + model)
    token = os.getenv("VISION_EMBEDDINGS_API_TOKEN", "")
    if len(token) < 16:
        parser.error("VISION_EMBEDDINGS_API_TOKEN environment variable is required")
    parsed = urllib.parse.urlsplit(args.base_url)
    if (parsed.scheme != "https" and
            not (args.allow_mock and parsed.scheme == "http" and
                 parsed.hostname in ("127.0.0.1", "localhost"))):
        parser.error("Use HTTPS, or --allow-mock for an HTTP localhost test")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("Base URL cannot contain credentials, query or fragment")
    args.base_url = args.base_url.rstrip("/")
    image = args.image.read_bytes()
    digest = hashlib.sha256(image).hexdigest()
    image_data = base64.b64encode(image).decode("ascii")
    opener = urllib.request.build_opener(RejectRedirects)
    try:
        _preflight(
            opener, args.base_url, token, selected, revisions,
            args.allow_mock, args.http_timeout,
        )
        for model in selected:
            if args.mode == "async":
                outcome = _async_run(
                    opener, args, token, model, image_data, digest, revisions[model],
                )
            else:
                outcome = _sync_run(
                    opener, args, token, model, image_data, digest, revisions[model],
                )
            if outcome != 0:
                return outcome
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
        _emit(status="STOP", error_type=type(error).__name__,
              http_status=getattr(error, "code", None),
              action="inspect_or_resume_GET_only_no_auto_retry")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
