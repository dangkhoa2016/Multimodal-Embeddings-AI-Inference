from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: str
    image_base64: str | None = None
    text: Annotated[str | None, Field(max_length=8192)] = None
    dimension: int | None = None
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
    normalized: bool = True


class BatchResponse(BaseModel):
    count: int
    items: list[EmbeddingResponse]
