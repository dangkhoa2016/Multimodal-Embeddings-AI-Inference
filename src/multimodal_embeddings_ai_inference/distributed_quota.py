"""Atomic Redis/Valkey rate budgets and bounded synchronous GPU reservations.

Only gateway containers have access to the Redis URL. Lua executes rate and
concurrency decisions in one Redis command, so admission is shared across all
gateways and denied work never consumes another budget. Uncertain replies fail
closed; synchronous GPU failures retain their leases until expiry.
"""
import hashlib
import logging
import math
import uuid
from dataclasses import dataclass

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# The Redis server clock, not container clocks, determines the refill rate.
# A shared distributed token bucket permits bursts up to the bucket capacity
# and refills at capacity/60 per second (not a strict sliding 60-second window).
_ADMIT_LUA = """
local now_parts = redis.call('TIME')
local now = tonumber(now_parts[1]) + tonumber(now_parts[2]) / 1000000
local units = tonumber(ARGV[1])
local global_capacity = tonumber(ARGV[2])
local client_capacity = tonumber(ARGV[3])
local max_global_active = tonumber(ARGV[4])
local max_client_active = tonumber(ARGV[5])
local sync = tonumber(ARGV[6])
local lease_id = ARGV[7]
local lease_seconds = tonumber(ARGV[8])

local function tokens(key, capacity)
    local saved = tonumber(redis.call('HGET', key, 'tokens'))
    local prior = tonumber(redis.call('HGET', key, 'time'))
    if saved == nil or prior == nil then
        return capacity
    end
    return math.min(capacity, saved + math.max(0, now - prior) * capacity / 60)
end

if units < 1 or units > global_capacity or units > client_capacity then
    return {'RATE', '60'}
end

if sync == 1 then
    redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
    redis.call('ZREMRANGEBYSCORE', KEYS[4], '-inf', now)
    if redis.call('ZCARD', KEYS[3]) >= max_global_active or
       redis.call('ZCARD', KEYS[4]) >= max_client_active then
        return {'CONCURRENCY', '2'}
    end
end

local global_tokens = tokens(KEYS[1], global_capacity)
local client_tokens = tokens(KEYS[2], client_capacity)
if global_tokens < units or client_tokens < units then
    local global_wait = (units - global_tokens) * 60 / global_capacity
    local client_wait = (units - client_tokens) * 60 / client_capacity
    return {'RATE', tostring(math.max(1, math.ceil(
        math.max(global_wait, client_wait, 0))))}
end

redis.call('HSET', KEYS[1], 'tokens',
    string.format('%.9f', global_tokens - units), 'time',
    string.format('%.6f', now))
redis.call('HSET', KEYS[2], 'tokens',
    string.format('%.9f', client_tokens - units), 'time',
    string.format('%.6f', now))
redis.call('EXPIRE', KEYS[1], 180)
redis.call('EXPIRE', KEYS[2], 180)

if sync == 1 then
    redis.call('ZADD', KEYS[3], now + lease_seconds, lease_id)
    redis.call('ZADD', KEYS[4], now + lease_seconds, lease_id)
    redis.call('EXPIRE', KEYS[3], lease_seconds + 60)
    redis.call('EXPIRE', KEYS[4], lease_seconds + 60)
end
return {'OK', '0'}
"""

_RELEASE_LUA = """
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
return 1
"""


@dataclass(frozen=True)
class SyncLease:
    """Opaque lease returned only for a successfully admitted sync request."""

    lease_id: str
    global_key: str
    client_key: str


class RedisDistributedQuota:
    def __init__(
        self, redis_client, *, namespace: str,
        global_units_per_minute: int, client_units_per_minute: int,
        global_sync_inflight: int, client_sync_inflight: int,
        lease_seconds: int,
    ):
        if not 1 <= global_units_per_minute <= 100000:
            raise ValueError("Invalid distributed global rate capacity")
        if not 1 <= client_units_per_minute <= global_units_per_minute:
            raise ValueError("Invalid distributed client rate capacity")
        if not 1 <= client_sync_inflight <= global_sync_inflight <= 64:
            raise ValueError("Invalid distributed sync concurrency")
        if not 5 <= lease_seconds <= 3600:
            raise ValueError("Invalid distributed sync lease lifetime")
        if not namespace or len(namespace) > 96 or not all(
            ch.isalnum() or ch in ":-_" for ch in namespace
        ):
            raise ValueError("Invalid distributed rate namespace")
        self.redis = redis_client
        self.namespace = namespace
        self.global_capacity = global_units_per_minute
        self.client_capacity = client_units_per_minute
        self.global_sync_inflight = global_sync_inflight
        self.client_sync_inflight = client_sync_inflight
        self.lease_seconds = lease_seconds

    def _keys(self, client_id: str) -> list[str]:
        identity = hashlib.sha256(client_id.encode("utf-8")).hexdigest()
        # One Redis Cluster hash slot is mandatory for the atomic multi-key
        # script; no client-identifying token or raw ID is embedded in keys.
        prefix = "{" + self.namespace + "}:"
        return [
            prefix + "rate:global",
            prefix + "rate:client:" + identity,
            prefix + "sync:global",
            prefix + "sync:client:" + identity,
        ]

    async def admit(self, units: int, client_id: str, *, sync: bool) -> SyncLease | None:
        if type(units) is not int or units < 1:
            raise ValueError("Embedding units must be a positive integer")
        keys = self._keys(client_id)
        lease_id = uuid.uuid4().hex if sync else ""
        try:
            result = await self.redis.eval(
                _ADMIT_LUA, 4, *keys, units, self.global_capacity,
                self.client_capacity, self.global_sync_inflight,
                self.client_sync_inflight, int(sync), lease_id,
                self.lease_seconds,
            )
        except Exception:
            # A Redis acknowledgement can be lost after a successful debit.
            # Never retry or fall back to a process-local rate limit.
            logger.error("Shared inference admission unavailable")
            raise HTTPException(
                status_code=503, detail="Distributed inference admission unavailable",
                headers={"Cache-Control": "no-store", "Retry-After": "5"},
            ) from None
        try:
            status, raw_wait = (
                value.decode("ascii") if isinstance(value, bytes) else str(value)
                for value in result
            )
            wait = max(1, min(60, math.ceil(float(raw_wait))))
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(
                status_code=503, detail="Invalid distributed admission response",
                headers={"Cache-Control": "no-store", "Retry-After": "5"},
            ) from None
        if status == "RATE":
            raise HTTPException(
                status_code=429, detail="Distributed embedding budget exceeded",
                headers={"Cache-Control": "no-store", "Retry-After": str(wait)},
            )
        if status == "CONCURRENCY":
            raise HTTPException(
                status_code=429, detail="Distributed synchronous concurrency limit reached",
                headers={"Cache-Control": "no-store", "Retry-After": str(wait)},
            )
        if status != "OK":
            raise HTTPException(
                status_code=503, detail="Invalid distributed admission response",
                headers={"Cache-Control": "no-store", "Retry-After": "5"},
            )
        if sync:
            return SyncLease(lease_id, keys[2], keys[3])
        return None

    async def release(self, lease: SyncLease) -> None:
        try:
            await self.redis.eval(
                _RELEASE_LUA, 2, lease.global_key, lease.client_key, lease.lease_id
            )
        except Exception:
            # The lease expires; do not mask a completed inference response.
            logger.error("Shared sync lease release unavailable; retained until expiry")
