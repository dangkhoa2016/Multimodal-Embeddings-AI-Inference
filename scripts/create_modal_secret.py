#!/usr/bin/env python3
"""Create or deliberately rotate the gateway Bearer token without printing secrets.

Model revisions and token stay in an owner-only local dotenv file. During a
rotation, do not replace the known-good local token if Modal rejects the update.
"""

import argparse
import os
import re
import secrets
import subprocess
import tempfile
from pathlib import Path

SHA = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen8-revision", required=True)
    parser.add_argument("--wemm9-revision", required=True)
    parser.add_argument("--force", action="store_true", help="Replace existing Modal Secret")
    parser.add_argument(
        "--rotate-token", action="store_true",
        help="Generate a fresh API token; requires --force to update the Modal Secret",
    )
    args = parser.parse_args()
    if args.rotate_token and not args.force:
        parser.error("--rotate-token requires --force")
    for revision in (args.qwen8_revision, args.wemm9_revision):
        if not SHA.fullmatch(revision):
            parser.error("Each revision must be an immutable 40-character lowercase SHA")

    path = Path.home() / ".config" / "multimodal-embeddings-ai-inference" / "modal-secret.env"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass

    token = None
    if path.exists() and not args.rotate_token:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VISION_EMBEDDINGS_API_TOKEN="):
                token = line.partition("=")[2]
        if token is not None and len(token) < 32:
            raise RuntimeError("Existing token too short; refuse to reuse")
    if not token:
        token = secrets.token_urlsafe(48)

    # Upload from a private staged file; preserve the previous local token on CLI failure.
    fd, staged = tempfile.mkstemp(prefix=".modal-secret-", suffix=".env", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"VISION_EMBEDDINGS_API_TOKEN={token}\n")
            stream.write(f"VISION_EMBEDDINGS_QWEN8_REVISION={args.qwen8_revision}\n")
            stream.write(f"VISION_EMBEDDINGS_WEMM9_REVISION={args.wemm9_revision}\n")
        command = ["modal", "secret", "create", "--from-dotenv", staged]
        if args.force:
            command.append("--force")
        command.append("multimodal-embeddings-api")
        subprocess.run(command, check=True)
        os.replace(staged, path)
        path.chmod(0o600)
    finally:
        Path(staged).unlink(missing_ok=True)

    print(f"Secret uploaded. Owner-only local token copy: {path}")
    print("Keep this file private. Store the API token in your client secret manager.")
    if args.rotate_token:
        print("Token rotated. Update all clients and restart/redeploy active services as needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
