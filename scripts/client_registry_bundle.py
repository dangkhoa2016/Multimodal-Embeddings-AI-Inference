#!/usr/bin/env python3
"""Build a private complete Modal gateway dotenv from trusted owner files.

Never calls Modal, logs tokens or overwrites output. Owner later reviews and
applies the complete file explicitly to the existing Modal gateway Secret.
"""
import argparse
import json
import os
import re
import stat
from pathlib import Path

from manage_client_keys import BASE, read_registry, safe_dir, write_token

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = re.compile(r"^[A-Z][A-Z0-9_]*$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


def load_owner(path: Path) -> dict[str, str]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("Owner configuration must be a regular 0600 file")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if (not sep or not VARIABLE.fullmatch(key) or key in values
                or not key.startswith("VISION_EMBEDDINGS_")):
            raise ValueError("Malformed or duplicate owner configuration entry")
        values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-file", type=Path, default=BASE / "modal-secret.env")
    parser.add_argument("--registry", type=Path, default=BASE / "client-keys.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-empty-registry", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.resolve().is_relative_to(ROOT):
        parser.error("Never write credential files within Git repository")
    values = load_owner(args.owner_file.expanduser())
    token = values.get("VISION_EMBEDDINGS_API_TOKEN", "")
    if not isinstance(token, str) or not 16 <= len(token) <= 4089 or not token.isascii():
        parser.error("Owner API token missing or invalid")
    for name in ("VISION_EMBEDDINGS_QWEN8_REVISION", "VISION_EMBEDDINGS_WEMM9_REVISION"):
        if not REVISION.fullmatch(values.get(name, "")):
            parser.error("Pinned model revision missing or invalid")
    source = args.registry.expanduser()
    if source.exists() or source.is_symlink():
        registry = read_registry(source)
        if not registry and not args.confirm_empty_registry:
            parser.error("Empty registry needs --confirm-empty-registry")
        values["VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON"] = json.dumps(
            registry, separators=(",", ":"), sort_keys=True,
        )
    elif "VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON" in values:
        registry = json.loads(values["VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON"])
    else:
        if not args.confirm_empty_registry:
            parser.error("Registry unknown; supply one or --confirm-empty-registry")
        registry = {}
        values["VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON"] = "{}"
    # Validate the merged registry against the actual gateway rules.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from multimodal_embeddings_ai_inference.config import Settings
    Settings(api_token=token, client_key_hashes=registry)
    if output.exists() or output.is_symlink():
        parser.error("Output already exists; choose a fresh path")
    safe_dir(output.parent)
    write_token(output, "".join(f"{k}={v}\n" for k, v in sorted(values.items())).rstrip("\n"))
    print(f"Gateway configuration staged privately: {output}")
    print("No Modal API calls or deployment performed.")


if __name__ == "__main__":
    main()
