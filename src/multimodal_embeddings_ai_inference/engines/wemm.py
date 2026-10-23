"""Tencent WeMM-9B model adapter using official Transformers + Qwen-VL processing."""

from ..checkpoints import require_checkpoint_info
from ..vector_utils import image_from_bytes, normalize_vector


class WeMMEncoder:
    def __init__(self):
        info = require_checkpoint_info("wemm-embedding-9b")
        import torch
        from transformers import AutoModel, AutoProcessor

        path = info["path"]
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
        self.checkpoint_identity = {
            "model": "wemm-embedding-9b", "repository": info["repository"],
            "revision": info["commit_sha"],
        }

    def encode(
        self, payload: bytes | str, kind: str, dimension: int,
        options: dict | None = None,
    ) -> list[float]:
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils import process_vision_info

        options = options or {}
        image_options = {
            key: options[key] for key in ("min_pixels", "max_pixels") if key in options
        }
        content = (
            [{"type": "image", "image": image_from_bytes(payload), **image_options}]
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
            value = self.model.embedding(**inputs).float()
            if options.get("normalize", True):
                value = F.normalize(value, dim=-1)
        return normalize_vector(
            value[0].detach().cpu().numpy(), dimension,
            normalize=options.get("normalize", True),
        )
