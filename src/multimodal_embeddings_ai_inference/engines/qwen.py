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
        self, payload: bytes | str, kind: str, dimension: int, instruction: str | None = None,
        options: dict | None = None,
    ) -> list[float]:
        item = {"image": image_from_bytes(payload)} if kind == "image" else {"text": str(payload)}
        item["instruction"] = (
            QWEN_LEGACY_INSTRUCTION if instruction is None else instruction
        )
        options = options or {}
        # Qwen upstream stores max_length/min_pixels/max_pixels on the embedder.
        # The worker serves calls serially; restore old settings even on error.
        old_values = {}
        try:
            for key in ("max_length", "min_pixels", "max_pixels"):
                if key in options:
                    old_values[key] = getattr(self.model, key, None)
                    setattr(self.model, key, options[key])
            if options.get("normalize") is False:
                outputs = self.model.process([item], normalize=False)
            else:
                outputs = self.model.process([item])
        finally:
            for key, value in old_values.items():
                if value is None:
                    delattr(self.model, key)
                else:
                    setattr(self.model, key, value)
        if hasattr(outputs, "detach"):
            outputs = outputs.detach().float().cpu().numpy()
        return normalize_vector(
            outputs[0], dimension, normalize=options.get("normalize", True)
        )
