#!/usr/bin/env python3
"""Offline client key lifecycle; no network/Modal calls and no token stdout.

Store high-entropy credentials only in owner-only files, store SHA-256 digests
in a separate client registry, and never embed tokens in a Git working tree.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path

CLIENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
BASE = Path.home() / ".config" / "multimodal-embeddings-ai-inference"


def safe_dir(directory: Path) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("Credential directory is not an owner-controlled directory")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Credential directory must have permissions 0700")


def read_registry(path: Path) -> dict[str, list[str]]:
    if not path.exists() and not path.is_symlink():
        return {}
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("Registry must be a regular owner-only file (0600)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or len(data) > 64:
        raise ValueError("Client registry must be a JSON object with at most 64 clients")
    seen = set()
    for client, digests in data.items():
        if client == "owner" or not isinstance(client, str) or not CLIENT_ID.fullmatch(client):
            raise ValueError("Invalid client ID in registry")
        if (not isinstance(digests, list) or not 1 <= len(digests) <= 4):
            raise ValueError("Each client must have between one and four keys")
        for digest in digests:
            if not isinstance(digest, str) or not DIGEST.fullmatch(digest) or digest in seen:
                raise ValueError("Malformed or duplicated API key digest")
            seen.add(digest)
    return data


@contextlib.contextmanager
def locked_registry(path: Path):
    safe_dir(path.parent)
    lock = path.parent / ".client-keys.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if stat.S_IMODE(os.fstat(fd).st_mode) & 0o077:
            raise ValueError("Lock must be owner-only")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def write_registry(path: Path, entries: dict[str, list[str]]) -> None:
    # Atomic replace of an owner-only file; fsync before exposing the new name.
    safe_dir(path.parent)
    fd, staged = tempfile.mkstemp(prefix=".registry-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(entries, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        Path(staged).unlink(missing_ok=True)


def write_token(path: Path, token: str) -> None:
    safe_dir(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("create", "rotate", "revoke-key", "revoke-client", "inspect")
    )
    parser.add_argument("--registry", type=Path, default=BASE / "client-keys.json")
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--token-out", type=Path)
    parser.add_argument("--digest", default=None, help="Exact SHA-256 digest to retire")
    parser.add_argument("--confirm-revoke-all", action="store_true")
    args = parser.parse_args()
    if args.action != "inspect" and (not args.client_id or not CLIENT_ID.fullmatch(args.client_id)
                                     or args.client_id == "owner"):
        parser.error("--client-id must be a valid non-owner client ID")
    if args.action in ("create", "rotate") and not args.token_out:
        parser.error("create/rotate requires --token-out; the raw token is never printed")
    if args.token_out and args.token_out.expanduser().resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ):
        parser.error("Refusing to write a raw API key inside the Git repository")
    if args.action not in ("create", "rotate") and args.token_out:
        parser.error("--token-out is only for create/rotate")
    if args.action == "revoke-key" and (not args.digest or not DIGEST.fullmatch(args.digest)):
        parser.error("revoke-key requires exact --digest SHA-256")
    if args.action == "revoke-client" and not args.confirm_revoke_all:
        parser.error("revoke-client requires --confirm-revoke-all")
    registry = args.registry.expanduser()
    with locked_registry(registry):
        entries = read_registry(registry)
        cid = args.client_id
        if args.action == "inspect":
            print("Client registry:", registry)
            for client_id, hashes in sorted(entries.items()):
                print(f"{client_id}: {len(hashes)} active key digest(s)")
            return 0
        if args.action == "create" and cid in entries:
            parser.error("Client already exists; use rotate to add an additional key")
        if args.action == "rotate" and cid not in entries:
            parser.error("Unknown client; use create first")
        if args.action in ("create", "rotate"):
            if len(entries.get(cid, [])) >= 4:
                parser.error("At most four active key digests per client")
            if args.action == "create" and len(entries) >= 64:
                parser.error("At most 64 clients")
            token = secrets.token_urlsafe(48)
            digest = hashlib.sha256(token.encode("ascii")).hexdigest()
            if any(digest in hashes for hashes in entries.values()):
                raise RuntimeError("Token digest collision")
            write_token(args.token_out.expanduser(), token)
            entries.setdefault(cid, []).append(digest)
            write_registry(registry, entries)
            print(f"Key registered for {cid}. Private token saved to {args.token_out}.")
            print("Distribute privately, then stage and apply the complete gateway Secret.")
            return 0
        if cid not in entries:
            parser.error("Unknown client ID")
        if args.action == "revoke-key":
            if args.digest not in entries[cid]:
                parser.error("Digest not found under this client")
            if len(entries[cid]) == 1:
                parser.error("Cannot leave a client with zero keys; use revoke-client")
            entries[cid].remove(args.digest)
        elif args.action == "revoke-client":
            del entries[cid]
        write_registry(registry, entries)
    print(f"Updated registry for {cid}; deploy updated gateway Secret to take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
