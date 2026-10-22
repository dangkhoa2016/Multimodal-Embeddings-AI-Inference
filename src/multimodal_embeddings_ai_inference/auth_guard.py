"""Bounded, thread-safe protection against repeated Bearer-token guesses.

State is local to a single gateway process. Deployments with multiple gateway
replicas need a shared atomic store or a managed edge rate limiter.
"""

import ipaddress
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from starlette.requests import Request


@dataclass
class _ClientState:
    failures: deque[float] = field(default_factory=deque)
    blocked_until: float = 0.0


def client_identity(request: Request) -> str:
    """Use ASGI peer identity; never trust user-supplied forwarding headers.

    An upstream ASGI server may already rewrite the scope's client identity
    according to its own trusted-proxy policy. Correct proxy configuration is a
    deployment requirement, not something to infer from arbitrary headers.
    """
    client = request.scope.get("client")
    host = client[0] if client and isinstance(client[0], str) else ""
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        # A single conservative bucket is safer than attacker-selected header
        # values when the upstream provides no verifiable client address.
        return "unidentified-peer"


class AuthenticationFailureGuard:
    """Per-peer three-strike lockout with a sliding window and bounded memory."""

    def __init__(
        self,
        max_failures: int = 3,
        window_seconds: int = 300,
        lockout_seconds: int = 900,
        max_clients: int = 4096,
        *,
        clock=None,
    ) -> None:
        if min(max_failures, window_seconds, lockout_seconds, max_clients) < 1:
            raise ValueError("Authentication guard limits must be positive")
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self.max_clients = max_clients
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.Lock()
        self._clients: OrderedDict[str, _ClientState] = OrderedDict()

    def _record_failure_locked(self, peer: str, now: float) -> int | None:
        entry = self._clients.get(peer)
        if entry is None:
            if len(self._clients) >= self.max_clients:
                self._clients.popitem(last=False)
            entry = _ClientState()
            self._clients[peer] = entry
        self._clients.move_to_end(peer)
        if entry.blocked_until > now:
            return max(1, math.ceil(entry.blocked_until - now))
        if entry.blocked_until:
            entry.failures.clear()
            entry.blocked_until = 0.0
        while entry.failures and now - entry.failures[0] >= self.window_seconds:
            entry.failures.popleft()
        entry.failures.append(now)
        if len(entry.failures) >= self.max_failures:
            entry.blocked_until = now + self.lockout_seconds
            entry.failures.clear()
            return self.lockout_seconds
        return None

    def record_failure(self, peer: str) -> int | None:
        """Record an already-known failure atomically (without extending a lock)."""
        with self._lock:
            return self._record_failure_locked(peer, self._clock())

    def authenticate(self, peer: str, verify: Callable[[], bool]) -> tuple[bool, int | None]:
        """Admit and verify atomically; locked peers never reach credential checks.

        The verifier is a short local credential comparison, without I/O.
        Holding the lock avoids a success racing with and clearing a new lockout.
        """
        with self._lock:
            now = self._clock()
            entry = self._clients.get(peer)
            if entry is not None and entry.blocked_until > now:
                self._clients.move_to_end(peer)
                return False, max(1, math.ceil(entry.blocked_until - now))
            if verify():
                self._clients.pop(peer, None)
                return True, None
            return False, self._record_failure_locked(peer, now)
