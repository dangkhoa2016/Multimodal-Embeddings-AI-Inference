"""Tencent WeMM-9B model adapter using official Transformers + Qwen-VL processing."""

from ..checkpoints import require_checkpoint
from ..validation import image_from_bytes, normalize_vector


class WeMMEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        path = require_checkpoint("wemm-embedding-9b")
        self.processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

    def encode(self, payload: bytes | str, kind: str, dimension: int) -> list[float]:
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils import process_vision_info

        content = (
            [{"type": "image", "image": image_from_bytes(payload)}]
            if kind == "image"
            else [{"type": "text", "text": str(payload)}]
        )
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        images, videos, kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, metadata = zip(*videos, strict=True)
            videos, metadata = list(videos), list(metadata)
        else:
            metadata = None
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=metadata,
            return_tensors="pt",
            **kwargs,
        ).to("cuda")
        with torch.inference_mode():
            value = F.normalize(self.model.embedding(**inputs).float(), dim=-1)
        return normalize_vector(value[0].detach().cpu().numpy(), dimension)
