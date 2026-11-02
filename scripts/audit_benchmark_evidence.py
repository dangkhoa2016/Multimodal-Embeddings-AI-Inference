#!/usr/bin/env python3
"""Compare archived benchmark evidence with a checkout without network/GPU calls.

A matching source digest alone never verifies which code ran on a remote worker.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPORTS = (
    "benchmarks/results/serving.json",
    "benchmarks/results/overload.json",
    "benchmarks/results/image-search.json",
)


def utc_time(value):
    if not isinstance(value, str):
        raise ValueError("Measurement timestamp must be a string")
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Measurement timestamp must include a UTC offset")
    return timestamp.astimezone(UTC)


def first_addition_date(root, name):
    result = subprocess.run(
        ["git", "log", "--reverse", "--diff-filter=A", "--format=%aI", "--", name],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode or not result.stdout.strip():
        return None
    return utc_time(result.stdout.splitlines()[0])


def recorded_times(report):
    """Find nested UTC event timestamps, not just the report header."""
    if isinstance(report, dict):
        for key, value in report.items():
            if key.endswith("_utc") and isinstance(value, str):
                yield utc_time(value)
            else:
                yield from recorded_times(value)
    elif isinstance(report, list):
        for value in report:
            yield from recorded_times(value)


def inspect(root, file_name):
    report = json.loads((root / file_name).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{file_name}: expected a JSON object")
    timestamp = utc_time(report.get("created_utc") or report.get("started_at_utc"))
    latest_event = max((timestamp, *recorded_times(report)))
    digests = report.get("source_sha256", {})
    if not isinstance(digests, dict):
        raise ValueError(f"{file_name}: source_sha256 must be a mapping")
    checks = []
    for name, expected in sorted(digests.items()):
        if (
            not isinstance(name, str) or not isinstance(expected, str)
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise ValueError(f"{file_name}: invalid source digest")
        source = (root / name).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            status = "unavailable"
        else:
            observed = hashlib.sha256(source.read_bytes()).hexdigest()
            status = (
                "matches_checkout" if observed == expected
                else "differs_from_checkout"
            )
        checks.append({"path": name, "status": status})
    addition_date = first_addition_date(root, file_name)
    chronology = (
        "unavailable" if addition_date is None
        else "git_addition_predates_recorded_measurement"
        if addition_date < latest_event
        else "git_addition_not_before_recorded_measurement"
    )
    return {
        "report": file_name,
        "recorded_at_utc": timestamp.isoformat(),
        "latest_event_utc": latest_event.isoformat(),
        "source_digest_comparisons": checks,
        "source_snapshot_verified": False,
        "source_snapshot_verification_note": (
            "Comparing file hashes with this checkout cannot attest the code "
            "that ran on a remote GPU worker. The deployment source commit "
            "was not independently verified."
        ),
        "git_metadata_chronology": chronology,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--strict-checkout", action="store_true",
        help="Fail if any recorded digest differs from this checkout",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        results = [inspect(root, name) for name in REPORTS]
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Invalid benchmark evidence: {exc}\n")
    print(json.dumps({
        "audit_kind": "offline_benchmark_evidence_audit",
        "network_or_gpu_requests": 0,
        "results": results,
    }, indent=2))
    if any(
        item["git_metadata_chronology"] != "git_addition_not_before_recorded_measurement"
        for item in results
    ):
        return 1
    if args.strict_checkout and any(
        item["status"] != "matches_checkout"
        for result in results
        for item in result["source_digest_comparisons"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
