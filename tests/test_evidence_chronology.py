"""Regression gate: false Git/benchmark chronology must fail CI."""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/audit_benchmark_evidence.py"
SPEC = importlib.util.spec_from_file_location("audit_benchmark_evidence", SOURCE)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize(
    "committed,expected",
    [
        ("2026-11-08T22:00:00+00:00", "git_addition_predates_recorded_measurement"),
        ("2026-11-09T01:00:00+00:00", "git_addition_not_before_recorded_measurement"),
    ],
)
def test_latest_nested_event_must_precede_file_commit(tmp_path, monkeypatch, committed, expected):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({
        "created_utc": "2026-11-08T20:00:00+00:00",
        "trials": [{"completed_at_utc": "2026-11-08T23:15:00+00:00"}],
        "source_sha256": {},
    }))
    monkeypatch.setattr(
        audit, "first_addition_date",
        lambda _root, _name: datetime.fromisoformat(committed),
    )
    result = audit.inspect(tmp_path, path.name)
    assert result["git_metadata_chronology"] == expected
    assert result["latest_event_utc"].startswith("2026-11-08T23:15:00")


@pytest.mark.parametrize(
    "committed,expected_exit",
    [
        ("2026-11-08T22:00:00+00:00", 1),
        ("2026-11-09T01:00:00+00:00", 0),
        (None, 1),
    ],
)
def test_chronology_gate_exit_code_prevents_false_green_ci(
    tmp_path, monkeypatch, committed, expected_exit, capsys,
):
    (tmp_path / "sample.json").write_text(json.dumps({
        "created_utc": "2026-11-08T23:15:00+00:00",
        "source_sha256": {},
    }))
    monkeypatch.setattr(audit, "REPORTS", ("sample.json",))
    monkeypatch.setattr(
        audit, "first_addition_date",
        lambda _root, _name: datetime.fromisoformat(committed) if committed else None,
    )
    monkeypatch.setattr(sys, "argv", ["audit", "--root", str(tmp_path)])
    assert audit.main() == expected_exit
    result = json.loads(capsys.readouterr().out)
    assert result["results"][0]["source_snapshot_verified"] is False


def test_invalid_nested_timestamp_fails_closed(tmp_path, monkeypatch):
    (tmp_path / "sample.json").write_text(json.dumps({
        "created_utc": "2026-11-08T20:00:00+00:00",
        "trials": [{"completed_at_utc": "not-a-timestamp"}],
        "source_sha256": {},
    }))
    monkeypatch.setattr(audit, "REPORTS", ("sample.json",))
    monkeypatch.setattr(sys, "argv", ["audit", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        audit.main()
    assert exc.value.code == 2
