"""Supported multimodal embedding models with independent vector spaces."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repository: str
    family: str
    dimensions: tuple[int, ...]
    default_dimension: int
    nominal_parameters: str
    modalities: tuple[str, ...] = ("image", "text")

    def public(self) -> dict:
        return asdict(self)


CATALOG = {
    "qwen3-vl-embedding-8b": ModelSpec(
        id="qwen3-vl-embedding-8b",
        repository="Qwen/Qwen3-VL-Embedding-8B",
        family="qwen3-vl",
        dimensions=tuple(range(64, 4097)),
        default_dimension=1024,
        nominal_parameters="8B",
    ),
    "wemm-embedding-9b": ModelSpec(
        id="wemm-embedding-9b",
        repository="tencent/WeMM-Embedding-9B",
        family="tencent-wemm",
        dimensions=(64, 128, 256, 512, 1024, 2048, 4096),
        default_dimension=1024,
        nominal_parameters="9B",
    ),
}


def public_spec(spec: ModelSpec) -> dict:
    from .model_options import schema_for

    data = spec.public()
    data["model_config_schema"] = schema_for(spec.id)
    data["supports_custom_instruction"] = spec.id == "qwen3-vl-embedding-8b"
    if spec.family == "qwen3-vl":
        data.pop("dimensions")
        data["dimension_range"] = {"min": 64, "max": 4096}
    return data


def supported_dimension(spec: ModelSpec, value: int) -> bool:
    return 64 <= value <= 4096 if spec.family == "qwen3-vl" else value in spec.dimensions
