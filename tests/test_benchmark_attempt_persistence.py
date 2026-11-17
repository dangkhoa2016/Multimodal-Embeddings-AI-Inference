"""Verify benchmark attempt journaling before and after HTTP dispatch."""

import json

import pytest


def test_sequential_attempt_is_persisted_before_dispatch(monkeypatch, tmp_path):
    from test_benchmark_session_recovery import run_warm_fixture

    _, scope, sent, _start, run_warm, _report = run_warm_fixture(monkeypatch, tmp_path)
    checked = scope["_sample_checked"]
    output = tmp_path / "multimodal_embeddings_performance_summary.json"

    def inspect_before_http(*args):
        journal = json.loads(output.read_text())
        assert len(journal["trials"]) == len(sent) + 1
        assert journal["trials"][-1]["status"] == "in_progress"
        return checked(*args)

    scope["_sample_checked"] = inspect_before_http
    run_warm()
    assert len(sent) == 24
    journal = json.loads(output.read_text())
    assert len(journal["trials"]) == 24
    assert all(trial["status"] == "completed" for trial in journal["trials"])


def test_stop_after_response_preserves_the_dispatched_attempt(monkeypatch, tmp_path):
    from test_benchmark_session_recovery import run_warm_fixture

    _, scope, sent, _start, run_warm, _report = run_warm_fixture(monkeypatch, tmp_path)
    clock = scope["utc_now"]
    interrupted = False

    def stop_after_http():
        nonlocal interrupted
        if sent and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return clock()

    scope["utc_now"] = stop_after_http
    with pytest.raises(KeyboardInterrupt):
        run_warm()
    assert len(sent) == 1
    report = json.loads((tmp_path / "multimodal_embeddings_performance_summary.json").read_text())
    assert report["status"] == "INCOMPLETE_INTERRUPTED"
    assert len(report["trials"]) == 1
    assert report["trials"][0]["status"] == "interrupted"
    assert report["trials"][0]["confirmed"] is False
    with pytest.raises(RuntimeError, match="Earlier benchmark failure"):
        run_warm()
    assert len(sent) == 1
