"""Bounded, privacy-minimizing per-gateway operational counters.

No URL paths, raw client identifiers, tokens, request IDs, prompts, images,
vectors or exception strings are accepted. These observations reset with the
process and do NOT represent global live Modal GPU job state.
"""
import threading
import time

_ROUTES = frozenset((
    "health", "models", "embedding", "batch", "async_submit",
    "async_status", "ops_metrics", "other",
))
_STATES = frozenset(("QUEUED", "RUNNING", "INDETERMINATE", "SUCCEEDED", "FAILED"))
_CODES = frozenset((200, 202, 400, 401, 403, 404, 409, 413, 422, 429, 500, 503, 504))
_RATE_DETAILS = {
    "Too many failed authentication attempts": "auth_lockout",
    "Client embedding rate limit reached": "client_rate",
    "Inference request budget exceeded": "local_rate",
    "Inference concurrency limit reached": "local_concurrency",
    "Async outstanding job capacity reached": "async_capacity",
    "Distributed embedding budget exceeded": "distributed_rate",
    "Distributed synchronous concurrency limit reached": "distributed_sync",
    "Gateway request body capacity reached": "body_capacity",
}
_OUTAGE_DETAILS = frozenset((
    "Distributed inference admission unavailable",
    "Job registry temporarily unavailable",
    "Job admission temporarily unavailable",
    "Invalid distributed admission response",
))


class GatewaySecurityMonitor:
    """Constant-cardinality, thread-safe counters for authenticated operators."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._requests = {route: {str(code): 0 for code in sorted(_CODES)}
                          for route in sorted(_ROUTES)}
        self._states = {state: 0 for state in sorted(_STATES)}
        # These classify only errors handled directly by the outer middleware;
        # all response statuses, including route-level 429/503, are counted above.
        self._rate_denials = {reason: 0 for reason in sorted(set(_RATE_DETAILS.values()))}
        self._outages = 0

    def http(self, route: str, status: int, *, detail: str | None = None) -> None:
        safe_route = route if route in _ROUTES else "other"
        safe_status = str(status if status in _CODES else 500)
        with self._lock:
            self._requests[safe_route][safe_status] += 1
            if status == 429 and detail in _RATE_DETAILS:
                self._rate_denials[_RATE_DETAILS[detail]] += 1
            if status == 503 and detail in _OUTAGE_DETAILS:
                self._outages += 1

    def job(self, state: str) -> None:
        if state not in _STATES:
            return
        with self._lock:
            self._states[state] += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "scope": "gateway_process_only",
                "semantics": "cumulative_observations_not_live_modal_job_counts",
                "uptime_seconds": max(0, round(time.monotonic() - self._started)),
                "http_responses": {
                    route: {code: count for code, count in values.items() if count}
                    for route, values in self._requests.items() if any(values.values())
                },
                "async_job_state_observations": dict(self._states),
                "pre_route_rejection_reasons": dict(self._rate_denials),
                "admission_outage_responses": self._outages,
            }


def route_label(path: str) -> str:
    """No dynamic path or request identifier is included in metrics."""
    if path == "/v1/health":
        return "health"
    if path in ("/v1/models",) or path.startswith("/v1/models/"):
        return "models"
    if path == "/v1/embeddings":
        return "embedding"
    if path == "/v1/embeddings/batches":
        return "batch"
    if path == "/v1/embeddings/async":
        return "async_submit"
    if path.startswith("/v1/embeddings/jobs/"):
        return "async_status"
    if path == "/v1/ops/metrics":
        return "ops_metrics"
    return "other"
