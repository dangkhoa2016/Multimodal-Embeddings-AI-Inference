#!/usr/bin/env python3
"""Generate a separate checksum list for downloadable GitHub Release assets.

Source SHA256SUMS.txt tracks repository-relative files. This generator creates
a list keyed by the flat filenames users download from the release page.
It deliberately does not include its own output file (no checksum cycle).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    ROOT / "notebooks/multimodal_embeddings_api_demo.ipynb",
    ROOT / "notebooks/multimodal_embeddings_benchmark.ipynb",
    ROOT / "SHA256SUMS.txt",
)
SUPPLEMENTARY_ASSETS = {
    "multimodal_embeddings_benchmark_evidence_2026-11-19.zip": (
        "f64795f82d3b67947d4bad7c28ee0af625455b432b7881ee6ba8858528df1e82"
    ),
}
DEFAULT_NAME = "SHA256SUMS.release.txt"


def generate(git_ref: str | None = None, *, include_supplementary: bool = False) -> str:
    lines: list[str] = []
    for path in ASSETS:
        if git_ref is None:
            content = path.read_bytes()
        else:
            try:
                content = subprocess.check_output(
                    ["git", "show", f"{git_ref}:{path.relative_to(ROOT).as_posix()}"],
                    cwd=ROOT, stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as exc:
                raise ValueError(
                    f"Cannot read {path.name} from Git ref {git_ref!r}. "
                    "Check that the ref exists locally (git fetch --tags)."
                ) from exc
        digest = hashlib.sha256(content).hexdigest()
        lines.append(f"{digest}  {path.name}")
    if include_supplementary:
        for name, digest in sorted(SUPPLEMENTARY_ASSETS.items()):
            lines.append(f"{digest}  {name}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(DEFAULT_NAME),
        help="Destination for the release-only checksums (default: current directory)",
    )
    parser.add_argument("--check", action="store_true", help="Check an existing generated manifest")
    parser.add_argument(
        "--git-ref",
        help="Hash files from a pinned Git tree instead of the working tree (e.g. v1.0.0)",
    )
    parser.add_argument(
        "--include-supplementary", action="store_true",
        help="Include the pinned benchmark evidence archive",
    )
    args = parser.parse_args()
    if args.git_ref and (args.git_ref.startswith("-") or ":" in args.git_ref):
        parser.error("Invalid Git ref for asset checksum generation")
    if args.output.resolve() in (p.resolve() for p in ASSETS):
        parser.error("Destination may not overwrite a source asset")
    try:
        expected = generate(args.git_ref, include_supplementary=args.include_supplementary)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
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
