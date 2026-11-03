#!/usr/bin/env python3
"""Offline, fail-closed security invariant review; no GPU or provider calls.

This is a static guardrail, not a substitute for dependency vulnerability
scanning, provider penetration testing or review of Redis/Modal configuration.
"""
import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main():
    issues: list[str] = []
    api = source("src/multimodal_embeddings_ai_inference/api.py")
    jobs = source("src/multimodal_embeddings_ai_inference/async_jobs.py")
    validation = source("src/multimodal_embeddings_ai_inference/validation.py")
    settings = source("src/multimodal_embeddings_ai_inference/config.py")
    guard = source("src/multimodal_embeddings_ai_inference/auth_guard.py")
    metrics = source("src/multimodal_embeddings_ai_inference/security_monitor.py")
    quota = source("src/multimodal_embeddings_ai_inference/distributed_quota.py")
    modal = source("modal_app.py")

    for path in (
        "src/multimodal_embeddings_ai_inference/api.py",
        "src/multimodal_embeddings_ai_inference/async_jobs.py",
        "src/multimodal_embeddings_ai_inference/config.py",
        "src/multimodal_embeddings_ai_inference/validation.py",
        "src/multimodal_embeddings_ai_inference/security_monitor.py",
    ):
        ast.parse(source(path), filename=path)

    expected = [
        ('owner token redacted from Settings repr',
         'api_token: str = field(repr=False)' in settings),
        ('Redis URI redacted from Settings repr',
         'distributed_rate_redis_url: str = field(default="", repr=False)' in settings),
        ('per-client hashes redacted from Settings repr',
         'client_key_hashes: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)'
         in settings),
        ('owner-only metrics authorization',
         'request.scope.get("_embedding_client_id") != "owner"' in api),
        ('raw HTTP validation errors suppressed',
         '@app.exception_handler(RequestValidationError)' in api
         and '"input"' in api and '"ctx"' in api),
        ('no echo of Pillow error text',
         'detail=f"Invalid image: {exc}"' not in validation
         and 'detail="Invalid image"' in validation),
        ('no exception trace from gateway inference failures',
         'logger.exception(' not in api),
        ('no exception trace from async coordinator',
         'logger.exception(' not in jobs),
        ('no forwarded header peer spoofing',
         'request.scope.get("client")' in guard
         and 'request.headers.get("X-Forwarded-For")' not in guard),
        ('bounded telemetry vocabulary',
         '_ROUTES = frozenset' in metrics and '_STATES = frozenset' in metrics),
        ('shared rate accounting has no local fallback',
         'Distributed inference admission unavailable' in quota
         and 'redis.call(\'TIME\')' in quota),
        ('gateway Secret separate from GPU worker Secret',
         'secrets=[gateway_secret]' in modal
         and 'secrets=[model_revision_secret]' in modal),
    ]
    for description, passed in expected:
        if not passed:
            issues.append(description)

    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT,
    ).split(b"\x00")
    forbidden = re.compile(
        r"(^|/)(\.secrets|\.env(?!\.example$)|modal-secret\.env|"
        r"client-keys\.json|gateway-next\.env)(/|$)"
    )
    for item in tracked:
        if item and forbidden.search(item.decode("utf-8")):
            issues.append("sensitive tracked filename")
    if issues:
        for issue in issues:
            print("FAIL:", issue)
        raise SystemExit(1)
    print(f"PASS offline security invariants ({len(expected)} controls, no provider calls)")


if __name__ == "__main__":
    main()
