"""Verify integrity and measurement accounting for the published Full Benchmark."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks/results/full-2026-11-19.json"
PROVENANCE = ROOT / "benchmarks/results/full-2026-11-19.provenance.json"
EXPECTED_SHA256 = "5667fdb6712cfc02df5c9a2546a2f1f0a52506393a1cb941c3edc2b7a9f28728"


def test_full_benchmark_original_report_digest_and_source_provenance():
    raw = REPORT.read_bytes()
    assert len(raw) == 87413
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    data = json.loads(raw)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["measured_report_sha256"] == EXPECTED_SHA256
    assert provenance["measured_report_bytes"] == len(raw)
    assert data["status"] == "LIVE_BENCHMARK_MEASURED"
    assert data["config"]["benchmark_mode"] == "full"
    assert data["config"]["run_id"] == "a6130023950a4d20b4be767efd79406f"
    assert data["provenance"]["client_source_commit"]["value"] is None
    assert data["provenance"]["server_source_commit"]["value"] is None
    assert provenance["source_attestation"]["server_source_commit_verified"] is False


def test_full_benchmark_every_measurement_and_request_is_confirmed():
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = data["trials"]
    concurrency = data["concurrent"]["trials"]
    batches = data["mixed_model_batches"]
    assert Counter(row["phase"] for row in rows) == {
        "cold": 10, "warmup": 8, "warm": 16,
        "alternate_setup": 2, "alternate": 10,
        "concurrent_setup": 2, "mixed_batch_setup": 6,
    }
    assert len(rows) == 54
    assert len(concurrency) == 8
    assert len(batches) == 3
    assert all(row["status"] == "completed" and row["confirmed"] is True
               for row in rows + concurrency + batches)
    async_rows = [row for row in rows if "async_request_id" in row]
    ids = [row["async_request_id"] for row in async_rows]
    assert len(ids) == len(set(ids)) == 28
    assert all(row["async_state"] == "SUCCEEDED"
               and row["async_post_attempts"] == 1 for row in async_rows)
    assert not any(row.get("readiness_refresh") for row in rows)
    assert len(rows) + len(concurrency) + len(batches) == 65
    assert len(rows) + len(concurrency) + sum(row["batch_size"]
                                              for row in batches) == 68
    assert all(row["model_state"] == "initial_load"
               and row["gpu_worker_first_request"] is True
               and row["idle_elapsed_seconds"] >= 210
               for row in rows if row["phase"] == "cold")
    assert all(row["model_state"] == "warm"
               and row["gpu_worker_first_request"] is False
               for row in rows if row["phase"] in {"warm", "alternate"})
    assert all(row["model_state"] == "warm"
               and row["gpu_worker_first_request"] is False
               for row in concurrency)
    assert all(len(set(batch["gpu_worker_ids"].values())) == 2
               and all(state["model_state"] == "warm"
                       and state["gpu_worker_first_request"] is False
                       for state in batch["gpu_states"].values())
               for batch in batches)


if __name__ == "__main__":
    test_full_benchmark_original_report_digest_and_source_provenance()
    test_full_benchmark_every_measurement_and_request_is_confirmed()
    print("PASS Full Benchmark 19 November 2026: SHA-256, provenance and 65 POST accounting")
