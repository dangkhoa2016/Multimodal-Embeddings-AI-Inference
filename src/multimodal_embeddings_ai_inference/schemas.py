from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: str
    image_base64: str | None = None
    text: Annotated[str | None, Field(max_length=8192)] = None
    dimension: int | None = None
    instruction: Annotated[
        str | None, Field(min_length=1, max_length=512, pattern=r"^[^\x00-\x1f\x7f]+$")
    ] = None
    request_id: Annotated[str | None, Field(max_length=128)] = None

    @model_validator(mode="after")
    def exactly_one_input(self):
        if (self.image_base64 is None) == (self.text is None):
            raise ValueError("Supply exactly one of image_base64 or text")
        if self.text is not None and not self.text:
            raise ValueError("text must not be empty")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[EmbeddingRequest] = Field(min_length=1, max_length=32)


class InferenceTiming(BaseModel):
    """Observed stage times. None means unavailable; overhead is not pure cold-start."""

    model_state: Literal["initial_load", "model_switch", "model_reload", "warm"]
    gpu_worker_first_request: bool
    gpu_worker_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    model_load_ms: float = Field(ge=0)
    gpu_encode_ms: float | None = Field(default=None, ge=0)
    gpu_worker_ms: float = Field(ge=0)
    gpu_dispatch_overhead_ms: float | None = Field(default=None, ge=0)
    batch_size: int = Field(ge=1)


class EmbeddingMetadata(BaseModel):
    """Non-sensitive provenance for a configured vector encoding."""

    preprocessing_version: str
    instruction_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EmbeddingResponse(BaseModel):
    model: str
    repository: str
    revision: str
    backend: str
    dimension: int
    vector: list[float]
    input_sha256: str
    request_id: str | None
    elapsed_ms: float
    timing: InferenceTiming | None = None
    embedding_metadata: EmbeddingMetadata
    normalized: bool = True


class BatchResponse(BaseModel):
    count: int
    items: list[EmbeddingResponse]
