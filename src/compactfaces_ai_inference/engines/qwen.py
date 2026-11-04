"""Official Qwen embedding source adapter (8B only, CUDA/BF16)."""

import os
import sys

from ..checkpoints import require_checkpoint
from ..validation import image_from_bytes, normalize_vector


class QwenVLEncoder:
    def __init__(self):
        source = os.getenv("CF_QWEN_SOURCE_PATH", "/opt/Qwen3-VL-Embedding")
        if source not in sys.path:
            sys.path.insert(0, source)
        import torch
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

        self.model = Qwen3VLEmbedder(
            model_name_or_path=require_checkpoint("qwen3-vl-embedding-8b"),
            torch_dtype=torch.bfloat16,
        )

    def encode(self, payload: bytes | str, kind: str, dimension: int) -> list[float]:
        item = {"image": image_from_bytes(payload)} if kind == "image" else {"text": str(payload)}
        item["instruction"] = "Represent the image or text for retrieval in a visual gallery."
        outputs = self.model.process([item])
        if hasattr(outputs, "detach"):
            outputs = outputs.detach().float().cpu().numpy()
        return normalize_vector(outputs[0], dimension)
