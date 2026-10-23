"""Declarative per-model request knobs; no GPU or web imports.

Defaults describe the current gateway adapter, not upstream checkpoint
capabilities. Unknown options fail closed rather than being passed to CUDA.
"""
import json

from .catalog import CATALOG
from .embedding_config import QWEN_LEGACY_INSTRUCTION

_IMAGE_OPTIONS = {
    "min_pixels": {"type": "integer", "minimum": 1024, "maximum": 12_000_000},
    "max_pixels": {"type": "integer", "minimum": 4096, "maximum": 12_000_000},
}
_COMMON_OPTIONS = {
    "normalize": {"type": "boolean", "default": True},
}
_MODEL_OPTIONS = {
    "qwen3-vl-embedding-8b": {
        "instruction": {"type": "string", "minLength": 1, "maxLength": 512,
                        "default": QWEN_LEGACY_INSTRUCTION},
        "max_length": {"type": "integer", "minimum": 64, "maximum": 32768,
                       "default": 8192},
        "min_pixels": {**_IMAGE_OPTIONS["min_pixels"], "default": 4096},
        "max_pixels": {**_IMAGE_OPTIONS["max_pixels"], "default": 1843200},
    },
    "wemm-embedding-9b": _IMAGE_OPTIONS,
}
MAX_OPTIONS = 12
MAX_BYTES = 4096


def schema_for(model_id: str) -> dict:
    if model_id not in CATALOG:
        raise ValueError("Unknown model")
    spec = CATALOG[model_id]
    dimension = (
        {"type": "integer", "minimum": 64, "maximum": 4096}
        if spec.family == "qwen3-vl"
        else {"type": "integer", "enum": list(spec.dimensions)}
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "maxProperties": MAX_OPTIONS,
        "properties": {
            "dimension": {**dimension, "default": spec.default_dimension},
            **_COMMON_OPTIONS, **_MODEL_OPTIONS[model_id],
        },
    }


def validate_extra_options(model_id: str, options: dict | None, *,
                           kind: str | None = None) -> dict:
    """Return validated nondefault processor/output options.

    Dimension and instruction are resolved by validation.prepare, while this
    function is also safe to use in a GPU-only Python environment.
    """
    if options is None:
        return {}
    if not isinstance(options, dict) or len(options) > MAX_OPTIONS:
        raise ValueError("Invalid model configuration")
    try:
        encoded = json.dumps(options, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid model configuration") from None
    if len(encoded.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Model configuration too large")
    rules = {**_COMMON_OPTIONS, **_MODEL_OPTIONS.get(model_id, {})}
    if model_id not in CATALOG:
        raise ValueError("Unsupported model ID")
    cleaned = {}
    for key, value in options.items():
        if key not in rules:
            raise ValueError("Unsupported model configuration option")
        rule = rules[key]
        if rule["type"] == "integer":
            valid = type(value) is int and rule["minimum"] <= value <= rule["maximum"]
        elif rule["type"] == "boolean":
            valid = type(value) is bool
        else:
            valid = (
                isinstance(value, str) and bool(value.strip())
                and len(value) <= rule["maxLength"]
                and not any(ord(c) < 32 or ord(c) == 127 for c in value)
            )
        if not valid:
            raise ValueError("Invalid model configuration value: " + key)
        if value != rule.get("default"):
            cleaned[key] = value
    if kind == "text" and any(k in options for k in _IMAGE_OPTIONS):
        raise ValueError("Image pixel options require image input")
    if kind == "image":
        defaults = _MODEL_OPTIONS[model_id]
        low = cleaned.get("min_pixels", defaults.get("min_pixels", {}).get("default", 4096))
        high = cleaned.get("max_pixels", defaults.get("max_pixels", {}).get("default", 1843200))
        if low > high:
            raise ValueError("min_pixels must not exceed max_pixels")
    return cleaned


def resolve_request_options(
    model_id: str, config: dict | None, *,
    dimension: int | None, instruction: str | None, kind: str,
) -> tuple[int, str | None, dict]:
    if not isinstance(config, (dict, type(None))):
        raise ValueError("model_config must be an object")
    raw = dict(config or {})
    # Include aliases in the JSON-size and key-count limits.
    if len(raw) > MAX_OPTIONS or len(json.dumps(raw, default=str)) > MAX_BYTES:
        raise ValueError("Model configuration too large")
    spec = CATALOG[model_id]
    if "dimension" in raw:
        configured_dimension = raw.pop("dimension")
        if dimension is not None and dimension != configured_dimension:
            raise ValueError("Conflicting dimension values")
        dimension = configured_dimension
    if "instruction" in raw:
        configured_instruction = raw.pop("instruction")
        if instruction is not None and instruction != configured_instruction:
            raise ValueError("Conflicting instruction values")
        instruction = configured_instruction
    effective_dimension = spec.default_dimension if dimension is None else dimension
    from .catalog import supported_dimension
    if type(effective_dimension) is not int or not supported_dimension(
        spec, effective_dimension
    ):
        raise ValueError("Unsupported dimension for " + model_id)
    if instruction is not None:
        if model_id != "qwen3-vl-embedding-8b":
            raise ValueError("Custom instruction is supported only for Qwen")
        instruction_rule = _MODEL_OPTIONS[model_id]["instruction"]
        if (
            not isinstance(instruction, str) or not instruction.strip()
            or len(instruction) > instruction_rule["maxLength"]
            or any(ord(c) < 32 or ord(c) == 127 for c in instruction)
        ):
            raise ValueError("Invalid model instruction")
    extra = validate_extra_options(model_id, raw, kind=kind)
    return effective_dimension, instruction, extra



def effective_encoding_profile(
    model_id: str, *, kind: str, dimension: int,
    instruction: str | None, options: dict | None,
) -> dict:
    """Exact per-item processing contract acknowledged by the GPU subprocess."""
    return {
        "kind": kind,
        "dimension": dimension,
        "instruction": (
            instruction if instruction is not None else QWEN_LEGACY_INSTRUCTION
        ) if model_id == "qwen3-vl-embedding-8b" else None,
        "model_config": dict(options or {}),
    }
