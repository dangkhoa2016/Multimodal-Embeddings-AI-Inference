"""Official Qwen embedding source adapter (8B only, CUDA/BF16)."""

import os
import sys

from ..checkpoints import require_checkpoint_info
from ..embedding_config import QWEN_LEGACY_INSTRUCTION
from ..vector_utils import image_from_bytes, normalize_vector


class QwenVLEncoder:
    def __init__(self):
        info = require_checkpoint_info("qwen3-vl-embedding-8b")
        source = os.getenv("VISION_EMBEDDINGS_QWEN_SOURCE_PATH", "/opt/Qwen3-VL-Embedding")
        if source not in sys.path:
            sys.path.insert(0, source)
        import torch
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

        self.model = Qwen3VLEmbedder(
            model_name_or_path=info["path"],
            torch_dtype=torch.bfloat16,
        )
        self.checkpoint_identity = {
            "model": "qwen3-vl-embedding-8b", "repository": info["repository"],
            "revision": info["commit_sha"],
        }

    def encode(
        self, payload: bytes | str, kind: str, dimension: int, instruction: str | None = None
    ) -> list[float]:
        item = {"image": image_from_bytes(payload)} if kind == "image" else {"text": str(payload)}
        item["instruction"] = (
            QWEN_LEGACY_INSTRUCTION if instruction is None else instruction
        )
        outputs = self.model.process([item])
        if hasattr(outputs, "detach"):
            outputs = outputs.detach().float().cpu().numpy()
        return normalize_vector(outputs[0], dimension)
