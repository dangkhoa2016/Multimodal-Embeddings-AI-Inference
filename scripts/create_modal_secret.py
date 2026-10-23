#!/usr/bin/env python3
"""Create gateway auth and worker-only revision Secrets without printing credentials.

One owner-only local dotenv keeps the token and public revision SHAs for smoke
tests. Deployment uses two independent Modal Secrets; GPU workers never
receive the API bearer token.
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

    gateway_name = os.getenv("VISION_EMBEDDINGS_MODAL_SECRET_NAME", "multimodal-embeddings-api")
    model_name = os.getenv(
        "VISION_EMBEDDINGS_MODAL_MODEL_SECRET_NAME", "multimodal-embeddings-model-config"
    )
    if not gateway_name or not model_name or gateway_name == model_name:
        parser.error("Gateway and worker revision Secrets must have different nonempty names")

    def upload(values: dict[str, str], name: str) -> None:
        # A separate private staging file prevents the gateway token reaching workers.
        fd, temporary = tempfile.mkstemp(
            prefix=".modal-secret-", suffix=".env", dir=path.parent
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                for key, value in values.items():
                    output.write(f"{key}={value}\n")
            command = ["modal", "secret", "create", "--from-dotenv", temporary]
            if args.force:
                command.append("--force")
            command.append(name)
            subprocess.run(command, check=True)
        finally:
            Path(temporary).unlink(missing_ok=True)

    # Keep existing gateway credentials until worker configuration is ready.
    # Modal operations are not transactional; operators must redeploy and verify.
    upload(
        {
            "VISION_EMBEDDINGS_QWEN8_REVISION": args.qwen8_revision,
            "VISION_EMBEDDINGS_WEMM9_REVISION": args.wemm9_revision,
        },
        model_name,
    )
    upload(
        {
            "VISION_EMBEDDINGS_API_TOKEN": token,
            "VISION_EMBEDDINGS_QWEN8_REVISION": args.qwen8_revision,
            "VISION_EMBEDDINGS_WEMM9_REVISION": args.wemm9_revision,
        },
        gateway_name,
    )
    fd, staged = tempfile.mkstemp(prefix=".modal-owner-", suffix=".env", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"VISION_EMBEDDINGS_API_TOKEN={token}\n")
            stream.write(f"VISION_EMBEDDINGS_QWEN8_REVISION={args.qwen8_revision}\n")
            stream.write(f"VISION_EMBEDDINGS_WEMM9_REVISION={args.wemm9_revision}\n")
        os.replace(staged, path)
        path.chmod(0o600)
    finally:
        Path(staged).unlink(missing_ok=True)

    print(f"Gateway and worker revision Secrets uploaded. Owner-only local token copy: {path}")
    print("Keep this file private. Store the API token in your client secret manager.")
    if args.rotate_token:
        print("Token rotated. Update all clients and restart/redeploy active services as needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
