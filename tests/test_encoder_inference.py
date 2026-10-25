"""Behavioral CPU tests for Qwen and WeMM encoder adapters (not a GPU quality claim)."""
import sys
import types
from contextlib import nullcontext
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from multimodal_embeddings_ai_inference.embedding_config import QWEN_LEGACY_INSTRUCTION
from multimodal_embeddings_ai_inference.engines.qwen import QwenVLEncoder
from multimodal_embeddings_ai_inference.engines.wemm import WeMMEncoder


def image_bytes():
    out = BytesIO()
    Image.new("RGBA", (8, 6), (20, 80, 100, 180)).save(out, format="PNG")
    return out.getvalue()


class QwenModel:
    max_length = 1024
    min_pixels = 256
    max_pixels = 4096

    def __init__(self):
        self.calls = []
        self.vector = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)
        self.error = None

    def process(self, items, **kwargs):
        self.calls.append((items, kwargs, (self.max_length, self.min_pixels, self.max_pixels)))
        if self.error:
            raise self.error
        return self.vector


@pytest.fixture
def qwen():
    encoder = QwenVLEncoder.__new__(QwenVLEncoder)
    encoder.model = QwenModel()
    return encoder


def test_qwen_text_default_instruction_and_truncate_before_normalization(qwen):
    assert qwen.encode("query", "text", 2) == pytest.approx([0.6, 0.8])
    items, kwargs, _ = qwen.model.calls[0]
    assert items == [{"text": "query", "instruction": QWEN_LEGACY_INSTRUCTION}]
    assert kwargs == {}


def test_qwen_custom_instruction_non_unit_vector_and_temporary_options(qwen):
    result = qwen.encode("query", "text", 2, instruction="Find similar photos.",
                         options={"normalize": False, "max_length": 128})
    assert result == pytest.approx([3.0, 4.0])
    items, kwargs, setting = qwen.model.calls[0]
    assert items[0]["instruction"] == "Find similar photos."
    assert kwargs == {"normalize": False}
    assert setting == (128, 256, 4096)
    assert qwen.model.max_length == 1024


def test_qwen_image_decoding_and_temporary_pixel_options(qwen):
    result = qwen.encode(image_bytes(), "image", 3,
                         options={"min_pixels": 512, "max_pixels": 2048})
    assert result == pytest.approx([3 / 13, 4 / 13, 12 / 13])
    items, _, setting = qwen.model.calls[0]
    assert items[0]["image"].mode == "RGB"
    assert items[0]["image"].size == (8, 6)
    assert setting == (1024, 512, 2048)
    assert (qwen.model.min_pixels, qwen.model.max_pixels) == (256, 4096)


def test_qwen_restores_options_on_failure(qwen):
    qwen.model.error = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        qwen.encode("query", "text", 2,
                    options={"max_length": 64, "min_pixels": 300, "max_pixels": 900})
    assert (qwen.model.max_length, qwen.model.min_pixels, qwen.model.max_pixels) == (
        1024, 256, 4096)


@pytest.mark.parametrize("values,dim", [
    ([[float("nan"), 1.0]], 2),
    ([[float("inf"), 1.0]], 2),
    ([[0.0, 0.0]], 2),
    ([[1.0, 2.0]], 3),
])
def test_qwen_invalid_model_vectors_raise(qwen, values, dim):
    qwen.model.vector = np.asarray(values, dtype=np.float32)
    with pytest.raises(ValueError):
        qwen.encode("query", "text", dim)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, i):
        return Tensor(self.value[i])


class Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class Processor:
    def __init__(self):
        self.messages = []
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": False}
        self.messages = messages
        return "serialized-prompt"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.inputs = Inputs(input_ids="encoded")
        return self.inputs


class WeMMModel:
    def __init__(self):
        self.calls = []
        self.vector = [[3.0, 4.0, 0.0]]

    def embedding(self, **kwargs):
        self.calls.append(kwargs)
        return Tensor(self.vector)


@pytest.fixture
def wemm(monkeypatch):
    encoder = WeMMEncoder.__new__(WeMMEncoder)
    encoder.processor, encoder.model = Processor(), WeMMModel()
    calls = {"normalize": 0, "vision": []}
    torch = types.ModuleType("torch")
    torch.inference_mode = nullcontext
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")

    def normalize(tensor, dim):
        assert dim == -1
        calls["normalize"] += 1
        return Tensor(tensor.value / np.linalg.norm(tensor.value, axis=dim, keepdims=True))

    functional.normalize = normalize
    torch.nn, nn.functional = nn, functional
    vision = types.ModuleType("qwen_vl_utils")

    def process_vision_info(messages, **kwargs):
        calls["vision"].append(kwargs)
        images = [part["image"] for m in messages for part in m["content"]
                  if part["type"] == "image"]
        return images or None, None, {"vision_flag": 1}

    vision.process_vision_info = process_vision_info
    for name, module in [
        ("torch", torch), ("torch.nn", nn), ("torch.nn.functional", functional),
        ("qwen_vl_utils", vision),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    return encoder, calls


def test_wemm_text_goes_through_processor_and_gpu_device(wemm):
    encoder, calls = wemm
    assert encoder.encode("query", "text", 2) == pytest.approx([0.6, 0.8])
    assert encoder.processor.messages[0]["content"] == [{"type": "text", "text": "query"}]
    assert encoder.processor.inputs.device == "cuda"
    assert encoder.model.calls == [{"input_ids": "encoded"}]
    assert calls["normalize"] == 1
    assert calls["vision"][0]["image_patch_size"] == 16


def test_wemm_no_normalization_preserves_magnitude(wemm):
    encoder, calls = wemm
    assert encoder.encode("query", "text", 2, options={"normalize": False}) == [3.0, 4.0]
    assert calls["normalize"] == 0


def test_wemm_image_options_and_rgb_decoding(wemm):
    encoder, calls = wemm
    result = encoder.encode(image_bytes(), "image", 2,
                            options={"min_pixels": 512, "max_pixels": 8192})
    assert result == pytest.approx([0.6, 0.8])
    item = encoder.processor.messages[0]["content"][0]
    assert (item["type"], item["image"].mode, item["image"].size) == (
        "image", "RGB", (8, 6))
    assert (item["min_pixels"], item["max_pixels"]) == (512, 8192)
    assert encoder.processor.calls[0]["vision_flag"] == 1
    assert calls["normalize"] == 1


@pytest.mark.parametrize("values,dim", [
    ([[float("nan"), 1.0]], 2), ([[0.0, 0.0]], 2), ([[1.0, 2.0]], 3),
])
def test_wemm_rejects_invalid_model_output(wemm, values, dim):
    encoder, _ = wemm
    encoder.model.vector = values
    with pytest.raises(ValueError):
        encoder.encode("query", "text", dim, options={"normalize": False})


def test_wemm_rejects_invalid_image_before_inference(wemm):
    encoder, _ = wemm
    with pytest.raises((OSError, ValueError)):
        encoder.encode(b"not an image", "image", 2)
    assert encoder.model.calls == []
