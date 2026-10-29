"""Acceptance records attribute CPU checks without inventing hosted/GPU evidence."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_acceptance.py"


def load_recorder():
    spec = importlib.util.spec_from_file_location("record_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_record_contains_only_source_and_cpu_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "private-token-fixture")
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://private.example.invalid")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "acceptance.json")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    raw = (tmp_path / "acceptance.json").read_text()
    record = json.loads(raw)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert record["client_source"]["commit"] == sha
    assert len(record["client_source"]["tree"]) == 40
    assert type(record["client_source"]["clean"]) is bool
    assert record["server_source"]["status"] == "unavailable"
    assert record["cpu_checks"] == {}
    assert "gpu" not in record
    assert "hosted_ui" not in record
    assert record["environment"]["packages"]["pytest"]
    assert "private-token-fixture" not in raw + result.stdout + result.stderr
    assert "private.example.invalid" not in raw + result.stdout + result.stderr


def test_failed_checks_are_recorded_without_raw_output(monkeypatch):
    recorder = load_recorder()
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "private response", "private path")

    monkeypatch.setattr(recorder.subprocess, "run", failed)
    checks = recorder.run_cpu_checks(ROOT)
    assert checks and calls
    assert all(item["status"] == "FAIL" for item in checks.values())
    assert "private" not in json.dumps(checks)


def test_declared_server_commit_is_never_inferred_from_client(tmp_path):
    output = tmp_path / "acceptance.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output),
         "--server-source-sha", "a" * 40], capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    assert record["server_source"] == {"commit": "a" * 40, "status": "operator_provided"}
    assert "gpu" not in record and "hosted_ui" not in record
    output.unlink()
    rejected = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output),
         "--server-source-sha", "not-a-sha"], capture_output=True, text=True, cwd=ROOT,
    )
    assert rejected.returncode != 0
    assert not output.exists()


def test_dirty_checkout_cannot_claim_unchanged_source_during_checks(monkeypatch, tmp_path):
    recorder = load_recorder()
    output = tmp_path / "dirty-acceptance.json"
    monkeypatch.setattr(recorder, "source_identity", lambda root: {
        "commit": "a" * 40, "tree": "b" * 40, "clean": False,
    })
    monkeypatch.setattr(recorder, "run_cpu_checks", lambda root: {
        "pytest": {"status": "PASS", "exit_code": 0},
    })
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--run-checks", "--output", str(output)])
    assert recorder.main() == 1
    record = json.loads(output.read_text())
    assert record["source_unchanged_during_checks"] is False
    assert record["cpu_checks"]["pytest"]["status"] == "PASS"
