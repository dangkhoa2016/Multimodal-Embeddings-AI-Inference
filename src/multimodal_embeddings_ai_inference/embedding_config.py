"""Instruction policy and deterministic, non-sensitive embedding configuration hashes.

The omitted-instruction behavior is intentionally frozen for backward compatibility.
Instruction hashes are provenance, not a requirement for query/document hashes to match.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import EmbeddingMetadata


QWEN_LEGACY_INSTRUCTION = "Represent the image or text for retrieval in a visual gallery."

PREPROCESSING_VERSIONS = {
    "qwen3-vl-embedding-8b": "qwen3-vl-adapter-v1",
    "wemm-embedding-9b": "wemm-adapter-v1",
}


def embedding_metadata(
    model_id: str, revision: str, dimension: int, instruction: str | None = None,
    options: dict | None = None,
) -> EmbeddingMetadata:
    from .schemas import EmbeddingMetadata

    if model_id not in PREPROCESSING_VERSIONS:
        raise ValueError("Unknown model for embedding metadata")
    if model_id == "qwen3-vl-embedding-8b":
        effective = QWEN_LEGACY_INSTRUCTION if instruction is None else instruction
        instruction_hash = hashlib.sha256(effective.encode("utf-8")).hexdigest()
    else:
        if instruction is not None:
            raise ValueError("Instruction is unsupported by this model")
        instruction_hash = None
    data = {
        "dimension": dimension,
        "instruction_sha256": instruction_hash,
        "model": model_id,
        "preprocessing_version": PREPROCESSING_VERSIONS[model_id],
        "revision": revision,
    }
    if options:
        # Normalized, validated nondefault options affect vector identity.
        data["model_config"] = options
    config_hash = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EmbeddingMetadata(
        preprocessing_version=PREPROCESSING_VERSIONS[model_id],
        instruction_sha256=instruction_hash,
        config_sha256=config_hash,
    )
