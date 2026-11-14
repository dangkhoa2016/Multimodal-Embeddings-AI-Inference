#!/usr/bin/env python3
"""Generate a separate checksum list for downloadable GitHub Release assets.

Source SHA256SUMS.txt tracks repository-relative files. This generator creates
a list keyed by the flat filenames users download from the release page.
It deliberately does not include its own output file (no checksum cycle).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    ROOT / "notebooks/multimodal_embeddings_api_demo.ipynb",
    ROOT / "notebooks/multimodal_embeddings_benchmark.ipynb",
    ROOT / "SHA256SUMS.txt",
)
DEFAULT_NAME = "SHA256SUMS.release.txt"


def generate() -> str:
    lines: list[str] = []
    for path in ASSETS:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(DEFAULT_NAME),
        help="Destination for the release-only checksums (default: current directory)",
    )
    parser.add_argument("--check", action="store_true", help="Check an existing generated manifest")
    args = parser.parse_args()
    if args.output.resolve() in (p.resolve() for p in ASSETS):
        parser.error("Destination may not overwrite a source asset")
    expected = generate()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            parser.error("Release checksums are missing or out of date")
        print(f"PASS release checksums: {len(ASSETS)} assets")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected, encoding="utf-8")
        print(f"Wrote {args.output} ({len(ASSETS)} downloadable asset checksums)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
