#!/usr/bin/env python3
"""Record source/environment evidence; never infer GPU or hosted UI acceptance."""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("fastapi", "uvicorn", "pydantic", "pillow", "numpy", "pytest", "httpx", "ruff")


def run_cpu_checks(root: Path) -> dict:
    """Persist only exit status: check output may contain private environment data."""
    commands = {
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "ruff": [sys.executable, "-m", "ruff", "check", "src", "modal_app.py",
                 "modal_models.py", "scripts", "tests"],
        "documentation": [sys.executable, "scripts/audit_release.py"],
        "manifest": [sys.executable, "scripts/update_sha256_manifest.py", "--check"],
    }
    checks = {}
    for name, command in commands.items():
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        checks[name] = {"status": "PASS" if result.returncode == 0 else "FAIL",
                        "exit_code": result.returncode}
    return checks


def source_identity(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "clean": not bool(git("status", "--porcelain", "--untracked-files=normal"))}


def build_record(root: Path, server_sha: str | None, checks: dict) -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "client_source": source_identity(root),
        "server_source": {"commit": server_sha,
                          "status": "operator_provided" if server_sha else "unavailable"},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "packages": packages},
        "cpu_checks": checks,
        "limitations": [
            "Client source identity does not verify the deployed server source.",
            "A dirty checkout is not fully identified by its commit/tree.",
            "This record contains source, environment and optional CPU checks only.",
            "Use docs/VERIFICATION.md for GPU and hosted notebook run procedures.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON file outside the source checkout; never overwrite")
    parser.add_argument("--server-source-sha", help="Optional operator-declared server Git SHA")
    parser.add_argument("--run-checks", action="store_true", help="Run CPU checks; no API calls")
    args = parser.parse_args()
    if args.server_source_sha and not re.fullmatch(r"[0-9a-fA-F]{40}", args.server_source_sha):
        parser.error("--server-source-sha must be a 40-character Git SHA")
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the source checkout")
    if output.exists():
        parser.error("--output already exists; choose a new evidence file")
    before = source_identity(ROOT)
    checks = run_cpu_checks(ROOT) if args.run_checks else {}
    record = build_record(ROOT, args.server_source_sha, checks)
    record["source_unchanged_during_checks"] = (
        bool(before["clean"] and record["client_source"] == before)
        if args.run_checks else None
    )
    # An owner-only new file, so evidence cannot accidentally replace a prior run.
    try:
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    except OSError:
        parser.exit(1, "Cannot write acceptance record; check output directory/permissions.\n")
    print("Source and CPU environment record saved.")
    return int(any(check["status"] != "PASS" for check in checks.values())
               or record["source_unchanged_during_checks"] is False)


if __name__ == "__main__":
    raise SystemExit(main())
