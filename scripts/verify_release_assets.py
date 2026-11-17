#!/usr/bin/env python3
"""Verify a completely downloaded GitHub Release against one immutable Git tag.

Run after downloading all assets into an otherwise empty directory.
No network calls, Git changes, credential access or GPU execution.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

try:
    from .build_release_checksums import ASSETS, ROOT, SUPPLEMENTARY_ASSETS, generate
except ImportError:  # Direct CLI invocation from the scripts directory.
    from build_release_checksums import ASSETS, ROOT, SUPPLEMENTARY_ASSETS, generate

EXPECTED = {p.name for p in ASSETS} | {"SHA256SUMS.release.txt"}
WITH_SUPPLEMENTARY = EXPECTED | set(SUPPLEMENTARY_ASSETS)


def verify(directory: Path, git_ref: str, *, require_supplementary=False) -> None:
    if not directory.is_dir():
        raise ValueError("Release asset directory does not exist")
    actual = {p.name for p in directory.iterdir()}
    if require_supplementary and actual != WITH_SUPPLEMENTARY:
        raise ValueError("Complete public release must include the evidence archive")
    if actual not in (EXPECTED, WITH_SUPPLEMENTARY):
        raise ValueError(
            f"Wrong release assets: missing={sorted(EXPECTED - actual)}, "
            f"unexpected={sorted(actual - WITH_SUPPLEMENTARY)}"
        )
    for name in actual:
        asset = directory / name
        if asset.is_symlink() or not asset.is_file():
            raise ValueError(f"Release asset is not a regular file: {name}")

    expected_manifest = generate(
        git_ref, include_supplementary=(actual == WITH_SUPPLEMENTARY)
    )
    if (directory / "SHA256SUMS.release.txt").read_text() != expected_manifest:
        raise ValueError("Release checksums differ from the pinned Git tag")
    for line in expected_manifest.splitlines():
        digest, filename = line.split("  ", 1)
        observed = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if digest != observed:
            raise ValueError(f"Asset checksum mismatch: {filename}")

    source_manifest = subprocess.check_output(
        ["git", "show", f"{git_ref}:SHA256SUMS.txt"], cwd=ROOT
    )
    if (directory / "SHA256SUMS.txt").read_bytes() != source_manifest:
        raise ValueError("Repository checksum manifest differs from pinned tag")
    print(f"PASS: {len(actual)} release assets match {git_ref} and SHA-256")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--git-ref", default="v1.0.0")
    parser.add_argument("--require-supplementary", action="store_true",
                        help="Require the pinned benchmark evidence ZIP")
    args = parser.parse_args()
    if args.git_ref.startswith("-") or ":" in args.git_ref:
        parser.error("Invalid release Git ref")
    try:
        verify(args.directory, args.git_ref,
               require_supplementary=args.require_supplementary)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
