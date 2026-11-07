"""Maintain source checksums, excluding the final documentation-only README pair."""

import argparse
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"
# The final README-only commit cannot also change a checksum manifest.
# Git commits protect these README bytes; checksum all other tracked sources.
EXCLUDED = {MANIFEST.name, "README.md", "README.vi.md"}


def render_manifest():
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "-z"], cwd=ROOT
    )
    paths = sorted(
        (value.decode("utf-8", errors="surrogateescape") for value in raw.split(b"\0") if value),
        key=lambda path: path.encode("utf-8", errors="surrogateescape"),
    )
    lines = []
    for name in paths:
        if name in EXCLUDED:
            continue
        file = ROOT / name
        if not file.is_file():
            raise RuntimeError("Tracked file missing; stage the deletion first: " + name)
        if "\n" in name or "\r" in name:
            raise RuntimeError("Unsupported newline in tracked filename")
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without rewriting")
    args = parser.parse_args()
    expected = render_manifest()
    if args.check:
        if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != expected:
            parser.exit(1, "SHA256SUMS.txt is stale; stage changed files and regenerate.\n")
        print("PASS SHA256SUMS.txt matches the Git index and file contents")
    else:
        MANIFEST.write_text(expected, encoding="utf-8")
        print("Updated SHA256SUMS.txt for staged/tracked files")


if __name__ == "__main__":
    main()
