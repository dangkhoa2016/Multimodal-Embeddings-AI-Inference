"""Conservative durable async GPU admission based on Modal Dict key claims.

Modal Dict has atomic per-key conditional puts, but no multi-key transaction
and no compare-and-swap delete. A crash may strand capacity: never assume a
stalled job has stopped, and never automatically retry paid inference.
"""
import hashlib
import time

from fastapi import HTTPException


class AsyncSlotPool:
    """Persist outstanding global and per-client permits across gateway restarts.

    Permits are NOT a permanent monetary ceiling: Modal Dict keys expire after
    prolonged inactivity. Use a workspace billing spend limit as the hard cap.
    """

    def __init__(self, store, *, global_limit: int = 8, per_client_limit: int = 2):
        if not 1 <= global_limit <= 256 or not 1 <= per_client_limit <= global_limit:
            raise ValueError("Invalid async outstanding job limits")
        self.store = store
        self.global_limit = global_limit
        self.per_client_limit = per_client_limit

    @staticmethod
    def _client_prefix(client_id: str) -> str:
        return "permit:client:" + hashlib.sha256(client_id.encode("utf-8")).hexdigest() + ":"

    async def _claim(self, prefix: str, count: int, job_id: str) -> str | None:
        for index in range(count):
            key = prefix + str(index)
            if await self.store.put(
                key, {"job_id": job_id, "claimed_at": int(time.time())},
                skip_if_exists=True,
            ):
                return key
        return None

    @staticmethod
    def _full() -> HTTPException:
        return HTTPException(
            status_code=429, detail="Async outstanding job capacity reached",
            headers={"Retry-After": "30", "Cache-Control": "no-store"},
        )

    async def reserve(self, job_id: str, client_id: str) -> dict[str, str]:
        global_key = await self._claim("permit:global:", self.global_limit, job_id)
        if global_key is None:
            raise self._full()
        try:
            client_key = await self._claim(
                self._client_prefix(client_id), self.per_client_limit, job_id,
            )
        except BaseException:
            # Unknown write acknowledgement: retain global slot, never reuse.
            raise
        if client_key is None:
            # This request never dispatched work and owns its sole global claim.
            await self.release({"global": global_key}, job_id)
            raise self._full()
        return {"global": global_key, "client": client_key}

    @staticmethod
    def _release_claim_key(slot_key: str, job_id: str) -> str:
        material = (slot_key + ":" + job_id).encode("utf-8")
        return "permit:release:" + hashlib.sha256(material).hexdigest()

    async def release(self, slots: dict, job_id: str) -> None:
        """Release a known-owned permit at most once, not via a racing get/pop.

        The once-only release marker must be acquired *before* checking or
        deleting a slot. A concurrent second release cannot pop a reallocated
        slot. An interrupted winner leaves a permit stranded, safely denying
        future work until operator reconciliation.
        """
        for kind in ("client", "global"):
            key = slots.get(kind)
            if not key:
                continue
            marker = self._release_claim_key(key, job_id)
            if not await self.store.put(
                marker, {"job_id": job_id, "slot": key, "created_at": int(time.time())},
                skip_if_exists=True,
            ):
                continue
            owner = await self.store.get(key)
            if owner is None:
                continue
            if owner.get("job_id") != job_id:
                # Never delete someone else's slot; fail closed on mismatch.
                raise RuntimeError("Async permit ownership mismatch")
            removed = await self.store.pop(key, None)
            if removed is not None and removed.get("job_id") != job_id:
                raise RuntimeError("Async permit changed during release")
