"""Conservative durable async GPU admission based on Modal Dict key claims.

Modal Dict provides atomic conditional insertion, but no compare-and-delete.
Versioned, append-only claims make each release refer to exactly one
acquisition: a delayed releaser never deletes another job's reallocated slot.
Records eventually expire after inactivity; enforce a workspace spend limit.
"""
import hashlib
import logging
import time
import uuid

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class AsyncSlotPool:
    """Persist per-client and global outstanding permits across restarts.

    Each of a fixed number of logical slots has sequential, immutable claims.
    Claim generation N+1 is permitted only after the matching release receipt
    for generation N. Neither claim nor receipt is ever deleted automatically.
    An expired/inconsistent chain fails closed until operator reconciliation.
    """

    _MAX_SCAN = 1024

    def __init__(self, store, *, global_limit: int = 8, per_client_limit: int = 2):
        if not 1 <= global_limit <= 256 or not 1 <= per_client_limit <= global_limit:
            raise ValueError("Invalid async outstanding job limits")
        self.store = store
        self.global_limit = global_limit
        self.per_client_limit = per_client_limit

    @staticmethod
    def _client_prefix(client_id: str) -> str:
        return "permit:client:" + hashlib.sha256(client_id.encode("utf-8")).hexdigest() + ":"

    @staticmethod
    def _claim_key(base: str, generation: int) -> str:
        return f"{base}:v2:{generation}"

    @staticmethod
    def _receipt_key(claim_key: str) -> str:
        return "permit:release:" + hashlib.sha256(
            ("v2:" + claim_key).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _matches(owner: dict | None, receipt: dict | None) -> bool:
        return (
            isinstance(owner, dict)
            and isinstance(receipt, dict)
            and owner.get("job_id") == receipt.get("job_id")
            and owner.get("acquisition_id") is not None
            and owner.get("acquisition_id") == receipt.get("acquisition_id")
        )

    async def _claim(
        self, prefix: str, count: int, job_id: str, acquisition_id: str,
        reads: list[int] | None = None,
    ) -> str | None:
        """Find the first unreleased generation using logarithmic probes.

        Matching immutable receipts certify releases. Generation N+1 cannot be
        acquired until generation N is released, so the release predicate is
        monotone while claim/receipt evidence remains intact. Missing or
        expired evidence is never accepted as proof of release.
        """
        async def read(key: str):
            if reads is not None:
                reads[0] += 1
            return await self.store.get(key)

        async def released(base: str, generation: int) -> bool:
            key = self._claim_key(base, generation)
            owner = await read(key)
            if owner is None:
                return False
            receipt = await read(self._receipt_key(key))
            if receipt is None:
                return False
            if not self._matches(owner, receipt):
                raise RuntimeError("Async permit release receipt mismatch")
            return True

        for index in range(count):
            base = prefix + str(index)
            if await read(base) is not None:
                continue  # Pre-migration permit remains occupied.
            cursor = await read(base + ":v2:cursor")
            generation = cursor if type(cursor) is int and cursor >= 0 else 0

            # Positive checkpoints may skip expired older claim records.
            # A delayed cursor update must never move this anchor backwards.
            checkpoint = self._MAX_SCAN
            while checkpoint <= generation:
                checkpoint *= 2
            while await read(f"{base}:v2:checkpoint:{checkpoint}") is not None:
                generation = checkpoint
                checkpoint *= 2

            # Exponential search for a negative release probe, then binary
            # search for the frontier. No linear page scan or hard 1024 limit.
            if await released(base, generation):
                low, distance = generation, 1
                while True:
                    upper = generation + distance
                    if upper >= (1 << 63):
                        raise RuntimeError("Async permit generation limit reached")
                    if not await released(base, upper):
                        break
                    low, distance = upper, distance * 2
                while upper - low > 1:
                    middle = low + (upper - low) // 2
                    if await released(base, middle):
                        low = middle
                    else:
                        upper = middle
                generation = upper

            # Ownership proof for the direct predecessor is mandatory.
            if generation:
                predecessor = self._claim_key(base, generation - 1)
                if not self._matches(
                    await read(predecessor),
                    await read(self._receipt_key(predecessor)),
                ):
                    continue

            key = self._claim_key(base, generation)
            owner = await read(key)
            if owner is None:
                # A later immutable checkpoint proves this is an expired hole,
                # not an unused generation. Never re-acquire its old number.
                next_boundary = (generation // self._MAX_SCAN + 1) * self._MAX_SCAN
                if await read(f"{base}:v2:checkpoint:{next_boundary}") is not None:
                    logger.warning("Async admission detected a historical claim gap")
                    continue
                if await self.store.put(
                    key, {
                        "job_id": job_id, "acquisition_id": acquisition_id,
                        "claimed_at": int(time.time()),
                    },
                    skip_if_exists=True,
                ):
                    return key
                # A concurrent contender acquired this candidate. Do not
                # re-use its slot; the next request can rescan after release.
                continue
            receipt = await read(self._receipt_key(key))
            if receipt is not None and not self._matches(owner, receipt):
                raise RuntimeError("Async permit release receipt mismatch")
            # An active claim keeps the slot occupied. Concurrent release
            # cannot justify acquisition without a fresh search.
        return None

    @staticmethod
    def _full() -> HTTPException:
        return HTTPException(
            status_code=429, detail="Async outstanding job capacity reached",
            headers={"Retry-After": "30", "Cache-Control": "no-store"},
        )

    async def reserve(self, job_id: str, client_id: str) -> dict[str, str]:
        acquisition_id = uuid.uuid4().hex
        reads = [0]
        started = time.perf_counter()
        try:
            global_key = await self._claim(
                "permit:global:", self.global_limit, job_id, acquisition_id, reads,
            )
            if global_key is None:
                raise self._full()
            try:
                client_key = await self._claim(
                    self._client_prefix(client_id), self.per_client_limit,
                    job_id, acquisition_id, reads,
                )
            except BaseException:
                # An acknowledgement may be lost; preserve global capacity.
                raise
            if client_key is None:
                await self.release(
                    {"global": global_key, "acquisition_id": acquisition_id}, job_id,
                )
                raise self._full()
            return {
                "global": global_key, "client": client_key,
                "acquisition_id": acquisition_id,
            }
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            log = logger.warning if reads[0] >= 128 or elapsed_ms >= 1000 else logger.debug
            log("Async admission reserve store_reads=%s elapsed_ms=%s",
                reads[0], elapsed_ms)

    @staticmethod
    def _release_claim_key(
        slot_key: str, job_id: str, acquisition_id: str | None = None,
    ) -> str:
        # Retained for pre-migration record inspection.
        material = slot_key + ":" + job_id
        if acquisition_id is not None:
            material = "acquisition:" + acquisition_id + ":" + material
        return "permit:release:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    async def release(self, slots: dict, job_id: str) -> None:
        """Record an immutable release; never pop a reusable Modal Dict key."""
        acquisition_id = slots.get("acquisition_id")
        for kind in ("client", "global"):
            key = slots.get(kind)
            if not key:
                continue

            if ":v2:" not in key:
                # No conditional delete exists for a legacy Modal Dict slot.
                # Preserve it rather than risk deleting a different claimant.
                logger.warning("Legacy async permit requires operator reconciliation")
                continue

            owner = await self.store.get(key)
            receipt_key = self._receipt_key(key)
            receipt = await self.store.get(receipt_key)
            expected = {"job_id": job_id, "acquisition_id": acquisition_id}
            if self._matches(owner, receipt):
                # Already released by an earlier invocation.
                continue
            if not isinstance(owner, dict) or any(
                owner.get(field) != value for field, value in expected.items()
            ):
                raise RuntimeError("Async permit ownership mismatch")
            if receipt is not None:
                raise RuntimeError("Async permit release receipt mismatch")

            accepted = await self.store.put(
                receipt_key, {
                    **expected, "slot": key, "created_at": int(time.time()),
                },
                skip_if_exists=True,
            )
            if not accepted:
                receipt = await self.store.get(receipt_key)
                if not self._matches(owner, receipt):
                    raise RuntimeError("Async permit release acknowledgement uncertain")

            # An advisory cursor reduces scans. Ownership still depends only on
            # the immutable claim + matching receipt, not on this pointer.
            base, generation_text = key.rsplit(":v2:", 1)
            generation = int(generation_text)
            if (generation + 1) % self._MAX_SCAN == 0:
                await self.store.put(
                    f"{base}:v2:checkpoint:{generation + 1}",
                    {"slot": key, "job_id": job_id, "acquisition_id": acquisition_id},
                    skip_if_exists=True,
                )
            await self.store.put(base + ":v2:cursor", generation + 1)
