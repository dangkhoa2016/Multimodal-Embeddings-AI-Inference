#!/usr/bin/env python3
"""Read-only async job reconciliation. Never submits work or displays vector bytes."""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def job_id_for_request(request_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}", request_id):
        raise ValueError("Invalid request ID")
    return hashlib.sha256(
        ("multimodal-embeddings-job-v1:" + request_id).encode("utf-8")
    ).hexdigest()


def valid_base_url(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    if any(ord(char) < 33 or ord(char) == 127 for char in raw):
        raise ValueError("Invalid HTTPS base URL")
    try:
        parts = urlsplit(raw)
        if parts.port == 0:
            raise ValueError
        if (parts.scheme != "https" or not parts.hostname
                or parts.username or parts.password or parts.path
                or parts.query or parts.fragment):
            raise ValueError
    except ValueError:
        raise ValueError("Use an HTTPS base URL without /v1 or credentials") from None
    return raw


def inspect(base_url: str, bearer: str, job_id: str, *, opener=None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", job_id):
        raise ValueError("Invalid job ID")
    if len(bearer) < 16 or any(ord(ch) <= 32 or ord(ch) >= 127 for ch in bearer):
        raise ValueError("Missing/invalid API token")
    url = valid_base_url(base_url) + "/v1/embeddings/jobs/" + job_id
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + bearer}, method="GET"
    )
    opener = opener or urllib.request.build_opener(NoRedirects)
    with opener.open(request, timeout=30) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Unexpectedly large response; refusing to show vectors")
    result = json.loads(raw)
    if not isinstance(result, dict) or result.get("job_id") != job_id:
        raise ValueError("Unexpected job response")
    state = result.get("state")
    if state not in {"QUEUED", "RUNNING", "INDETERMINATE", "SUCCEEDED", "FAILED"}:
        raise ValueError("Unknown job state")
    redacted = {
        "job_id": job_id,
        "state": state,
        "created_at": result.get("created_at"),
        "updated_at": result.get("updated_at"),
        # Describes this inspector only, never the original GPU job.
        "inspector_sent_inference_post": False,
    }
    if state == "SUCCEEDED":
        info = result.get("result")
        if isinstance(info, dict):
            redacted["result_metadata"] = {
                "model": info.get("model"),
                "revision": info.get("revision"),
                "dimension": info.get("dimension"),
                "vector_length": (
                    len(info["vector"]) if isinstance(info.get("vector"), list) else None
                ),
            }
    elif state in {"QUEUED", "RUNNING", "INDETERMINATE"}:
        redacted["advice"] = (
            "Inspect coordinator and Modal logs for the SAME job. "
            "Do not create a new request ID or retry GPU work automatically."
        )
    else:
        redacted["advice"] = "Failed; inspect logs before any manually approved new work."
    return redacted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Authorized HTTPS API base URL")
    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument("--job-id")
    id_group.add_argument("--request-id")
    args = parser.parse_args(argv)
    try:
        job_id = args.job_id if args.job_id else job_id_for_request(args.request_id)
        result = inspect(
            args.base_url,
            os.environ.get("VISION_EMBEDDINGS_API_TOKEN", ""),
            job_id,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(
                "Job not found or record expired. This is NOT proof that GPU stopped. "
                "No inference POST was sent; inspect existing logs before any new job.",
                file=sys.stderr,
            )
            return 3
        print(f"Job GET returned HTTP {exc.code}; no POST sent.", file=sys.stderr)
        return 2
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Read-only inspection unavailable ({type(exc).__name__}); no POST sent.",
              file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
