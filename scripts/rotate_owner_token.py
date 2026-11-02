#!/usr/bin/env python3
"""Stage a new owner token without modifying secrets, registry or Modal."""

import argparse
import hashlib
import json
import secrets
from pathlib import Path

from client_registry_bundle import REVISION, ROOT, load_owner
from manage_client_keys import BASE, read_registry, write_token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "gateway-next.env")
    parser.add_argument("--registry", type=Path, default=BASE / "client-keys.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-empty-registry", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser()
    registry_file = args.registry.expanduser()
    output = args.output.expanduser()
    if source.resolve().is_relative_to(ROOT) or output.resolve().is_relative_to(ROOT):
        parser.error("Credentials must remain outside the Git repository")
    if source.resolve() == output.resolve():
        parser.error("Source and output must differ")
    if output.exists() or output.is_symlink():
        parser.error("Output already exists; choose a new private path")

    try:
        values = load_owner(source)
    except (OSError, ValueError) as exc:
        parser.error(f"Unsafe source configuration: {exc}")
    original = values.get("VISION_EMBEDDINGS_API_TOKEN", "")
    if not 32 <= len(original) <= 4089 or not original.isascii():
        parser.error("Owner token missing or invalid")
    for name in ("VISION_EMBEDDINGS_QWEN8_REVISION", "VISION_EMBEDDINGS_WEMM9_REVISION"):
        if not REVISION.fullmatch(values.get(name, "")):
            parser.error("Pinned model revision missing or invalid")
    if "VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON" not in values:
        parser.error("Missing client registry; first use scripts/client_registry_bundle.py")

    try:
        clients = json.loads(values["VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON"])
        if not isinstance(clients, dict):
            raise ValueError("Client registry must be a JSON object")
        if registry_file.exists() or registry_file.is_symlink():
            if read_registry(registry_file) != clients:
                parser.error("Gateway and canonical client registries differ")
        elif clients:
            parser.error("Canonical client registry is missing")
        if not clients and not args.confirm_empty_registry:
            parser.error("Empty registry requires --confirm-empty-registry")
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from multimodal_embeddings_ai_inference.config import Settings

        Settings(api_token=original, client_key_hashes=clients)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(f"Invalid client registry: {exc}")

    replacement = secrets.token_urlsafe(48)
    digest = hashlib.sha256(replacement.encode("ascii")).hexdigest()
    if replacement == original or any(digest in values for values in clients.values()):
        parser.error("Token collision; try again with a new output")
    values["VISION_EMBEDDINGS_API_TOKEN"] = replacement
    body = "".join(f"{name}={value}\n" for name, value in sorted(values.items()))
    write_token(output, body.rstrip("\n"))
    print(f"Complete gateway configuration staged privately: {output}")
    print("No Modal calls, Secret updates or deployments performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
