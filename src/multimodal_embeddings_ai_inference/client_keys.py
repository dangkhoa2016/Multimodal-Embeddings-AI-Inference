"""Operator-managed, hashed per-client Bearer keys; secrets never enter job records."""
import hashlib
import hmac


def identify_bearer(token: str, *, owner_token: str, client_key_hashes: dict) -> str | None:
    """Resolve a stable client ID across API key rotation."""
    if not 16 <= len(token) <= 4089 or not token.isascii():
        return None
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    owner_match = hmac.compare_digest(token, owner_token)
    match = None
    # No early match: avoid disclosing which of the configured keys matched.
    for client_id, digests in client_key_hashes.items():
        for expected in digests:
            if hmac.compare_digest(digest, expected):
                match = client_id
    return "owner" if owner_match else match
