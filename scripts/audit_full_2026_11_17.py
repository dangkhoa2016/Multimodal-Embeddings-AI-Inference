#!/usr/bin/env python3
"""Verify the 2026-11-17 Full Benchmark archive offline (no GPU/network)."""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PIN = "44f1f0c30c6c5ff77b590df530bf9968cc177c24"
EXPECTED = {
    "cold": 10,
    "warmup": 8,
    "warm": 16,
    "alternate_setup": 2,
    "alternate": 10,
    "concurrent_setup": 2,
    "mixed_batch_setup": 6,
}
REVISIONS = {
    "qwen3-vl-embedding-8b": "2c4565515e0f265c6511776e7193b22c0968ddc7",
    "wemm-embedding-9b": "00c52839de57a6d4fd5b78cf5522ccf0ac8ea482",
}


def require(ok, description):
    if not ok:
        raise ValueError(description)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def when(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def stats(rows, key):
    vals = sorted(
        row[key] for row in rows if type(row.get(key)) in (float, int) and math.isfinite(row[key])
    )
    if not vals:
        return None

    def pct(p):
        rank = (len(vals) - 1) * p / 100
        a, b = math.floor(rank), math.ceil(rank)
        return vals[a] + (vals[b] - vals[a]) * (rank - a)

    return {
        "n": len(vals),
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "mean_ms": round(statistics.mean(vals), 3),
    }


def verify(root):
    manifest = root / "EVIDENCE-SHA256SUMS.txt"
    require(manifest.is_file(), "Evidence checksum manifest missing")
    lines = manifest.read_text().splitlines()
    require(lines, "Empty evidence manifest")
    checked = set()
    for line in lines:
        digest, name = line.split("  ", 1)
        require(
            name not in checked and not name.startswith("/") and ".." not in Path(name).parts,
            "Unsafe or duplicate filename in manifest",
        )
        checked.add(name)
        file = root / name
        require(file.is_file() and sha(file.read_bytes()) == digest, "SHA-256 mismatch: " + name)
    report = json.loads((root / "benchmarks/results/full-2026-11-17.json").read_text())
    p = json.loads((root / "benchmarks/results/full-2026-11-17.provenance.json").read_text())
    run = p["runner_provenance"]
    modal = p["modal_deployment"]
    require(report.get("status") == "LIVE_BENCHMARK_MEASURED", "Report not completed")
    require(report["config"]["benchmark_mode"] == "full", "Wrong benchmark mode")
    require(report["config"]["retries"] == 0, "Retry count mismatch")
    require(set(report["models"]) == set(REVISIONS), "Unexpected model set")
    require(run["git_source_commit"] == PIN, "Source commit mismatch")
    require(modal["history"][0]["commit"] == PIN[:7], "Modal history mismatch")
    require(run["modal_app_id"] == modal["info"]["app_id"], "Modal app ID mismatch")
    require(
        run["modal_app_version"] == modal["info"]["lifecycle"]["version"], "Modal version mismatch"
    )
    require(
        run["report_sha256"]
        == sha((root / "benchmarks/results/full-2026-11-17.json").read_bytes()),
        "Report digest mismatch",
    )
    for key in ("client_source_commit", "server_source_commit"):
        require(report["provenance"][key]["value"] == PIN, "Run source mismatch: " + key)
    for model, rev in REVISIONS.items():
        require(
            report["models"][model]["revision"] == rev, "Checkpoint revision mismatch: " + model
        )
        require(report["models"][model]["dimension"] == 1024, "Vector dimension mismatch: " + model)
    require(
        when(run["started_utc"]) <= when(report["created_utc"]) <= when(run["finished_utc"]),
        "Run chronology mismatch",
    )
    require(when(report["created_utc"]).date().isoformat() == "2026-11-17", "Wrong report date")
    rows = report["trials"]
    concurrent = report["concurrent"]["trials"]
    mixed = report["mixed_model_batches"]
    require(dict(Counter(r["phase"] for r in rows)) == EXPECTED, "Phase counts mismatch")
    require(len(rows) == 54 and len(concurrent) == 8 and len(mixed) == 3, "Trial counts mismatch")
    require(
        all(
            t.get("status") == "completed" and t.get("confirmed") is True
            for t in rows + concurrent + mixed
        ),
        "Unconfirmed/failed trial",
    )
    require(sum(t.get("batch_size", 0) for t in mixed) == 6, "Mixed batch counts mismatch")
    summaries = {}
    for trial in rows:
        if trial["phase"] in ("cold", "warm", "alternate"):
            name = " / ".join((trial["phase"], trial["model"], trial["modality"]))
            summaries.setdefault(name, []).append(trial)
    require(set(summaries) == set(report["summary"]), "Summary groups mismatch")
    for name, samples in summaries.items():
        value = report["summary"][name]
        require(
            value["attempts"] == len(samples) and value["confirmed"] == len(samples),
            "Summary sample count mismatch",
        )
        for metric, key in [
            ("client", "client_elapsed_ms"),
            ("api", "api_elapsed_ms"),
            ("model_load", "model_load_ms"),
            ("gpu_encode", "gpu_encode_ms"),
        ]:
            require(
                value[metric] == stats(samples, key),
                "Summary statistics mismatch: " + metric + " / " + name,
            )
    require(
        run["executed_code_sha256"]
        == sha((root / "evidence/benchmark_execute_2026-11-17.py").read_bytes()),
        "Executed notebook script mismatch",
    )
    require(
        run["execution_log_sha256"]
        == sha((root / "evidence/benchmark_execution_2026-11-17.log").read_bytes()),
        "Execution log mismatch",
    )
    require(
        run["exit_code"] == 0 and run["status"] == "COMPLETED_UNAUDITED", "Execution incomplete"
    )
    return {
        "status": "PASS",
        "verified_files": len(checked),
        "post_requests": 65,
        "embedding_items": 68,
        "summary_groups": len(summaries),
        "provenance": (
            "Modal deployment history + operator-provided commit; not hardware attestation"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root), indent=2))
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)[:500]}, indent=2))
        sys.exit(2)
