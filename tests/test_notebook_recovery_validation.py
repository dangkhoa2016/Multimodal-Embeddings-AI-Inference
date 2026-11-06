"""Reject malformed read-only notebook recovery without new GPU work."""

import pytest
from test_notebook_async_resume import BODY, RAW, load_helpers


@pytest.mark.parametrize("response", [None, [], "bad", {},
    {"job_id": "invalid", "state": "RUNNING"},
    {"job_id": "a" * 64, "state": "INVALID"},
    {"job_id": "z" * 64, "state": "RUNNING"},
])
def test_uncertain_invalid_lookup_never_submits_or_changes_saved_cache(
    monkeypatch, tmp_path, response,
):
    calls = []

    def lookup(path, payload=None):
        calls.append((path, payload))
        if payload is not None:
            pytest.fail("Malformed recovery response authorized another POST")
        return response

    ctx = load_helpers(monkeypatch, tmp_path, lookup)
    rid = ctx["_async_identity"](BODY, RAW)
    ctx["_ASYNC_CACHE"]["records"][rid] = {"state": "UNCERTAIN", "job_id": None}
    ctx["_save_async_cache"]()
    before = ctx["_ASYNC_CACHE_PATH"].read_bytes()
    ctx["MAX_POLL_SECONDS"] = 0  # A malformed lookup must fail before polling starts.
    with pytest.raises(RuntimeError, match="[Rr]ecovery"):
        ctx["_async_result"](BODY, RAW)
    assert len(calls) == 1 and calls[0][1] is None
    assert ctx["_ASYNC_CACHE_PATH"].read_bytes() == before
