#!/usr/bin/env python3
"""Safely generate and upload Modal gateway secret (owner-only local file).

IMPORTANT: Print neither CF_API_TOKEN nor the generated file contents.
Rails may use the same CF_API_TOKEN securely from the owner-only local file.
"""

import argparse
import os
import re
import secrets
import subprocess
from pathlib import Path

SHA = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen8-revision", required=True)
    parser.add_argument("--wemm9-revision", required=True)
    parser.add_argument("--force", action="store_true", help="Replace existing Modal Secret")
    args = parser.parse_args()
    for revision in (args.qwen8_revision, args.wemm9_revision):
        if not SHA.fullmatch(revision):
            parser.error("Each revision must be an immutable 40-character lowercase SHA")
    path = Path.home() / ".config" / "compactfaces-ai-inference" / "modal-secret.env"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    token = None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("CF_API_TOKEN="):
                token = line.partition("=")[2]
        if token is not None and len(token) < 32:
            raise RuntimeError("Existing token too short; refuse to reuse")
    if not token:
        token = secrets.token_urlsafe(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"CF_API_TOKEN={token}\n")
        f.write(f"CF_QWEN8_REVISION={args.qwen8_revision}\n")
        f.write(f"CF_WEMM9_REVISION={args.wemm9_revision}\n")
    path.chmod(0o600)
    command = ["modal", "secret", "create", "--from-dotenv", str(path)]
    if args.force:
        command.append("--force")
    command.append("compactfaces-api")
    subprocess.run(command, check=True)
    print(f"Secret uploaded. Owner-only local token copy: {path}")
    print("Never commit or share this file. Configure the same CF_API_TOKEN in Rails credentials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
