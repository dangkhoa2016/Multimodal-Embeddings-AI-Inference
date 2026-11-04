"""Exercise dedicated model subprocess startup and warm reuse without CUDA."""

import sys

import pytest

from multimodal_embeddings_ai_inference.gpu_supervisor import ModelSupervisor


def test_dedicated_model_worker_never_switches(tmp_path, monkeypatch):
    fake_package = tmp_path / "multimodal_embeddings_ai_inference"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("")
    (fake_package / "gpu_stdio.py").write_text("""
import json, os, sys
model = sys.argv[1]
out = os.fdopen(int(os.environ['VISION_EMBEDDINGS_RESPONSE_FD']), 'w', buffering=1)
out.write(json.dumps({'ready': True, 'model': model}) + '\\n')
for line in sys.stdin:
    items = json.loads(line)['items']
    result = {'ok': True, 'vectors': [[1.0] * x['dimension'] for x in items], 'gpu_encode_ms': 0.3}
    out.write(json.dumps(result) + '\\n')
""")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("VISION_EMBEDDINGS_GPU_LOG", str(tmp_path / "gpu.log"))
    model = "wemm-embedding-9b"
    supervisor = ModelSupervisor({model: sys.executable}, startup_timeout=10)
    try:
        first = supervisor.encode_many_profiled(
            model, [{"kind": "text", "payload": "a", "dimension": 4}]
        )
        first_pid = supervisor.process.pid
        assert first["vectors"] == [[1.0] * 4]
        assert first["timing"]["model_state"] == "initial_load"
        assert first["timing"]["gpu_worker_first_request"] is True
        assert first["timing"]["model_load_ms"] > 0
        assert first["timing"]["gpu_encode_ms"] == 0.3
        warm = supervisor.encode_many_profiled(
            model, [{"kind": "text", "payload": "a", "dimension": 2}]
        )
        assert supervisor.process.pid == first_pid
        assert warm["timing"]["model_state"] == "warm"
        assert warm["timing"]["gpu_worker_first_request"] is False
        assert warm["timing"]["model_load_ms"] == 0
        with pytest.raises(ValueError, match="Wrong model"):
            supervisor.encode_many_profiled(
                "qwen3-vl-embedding-8b", [{"kind": "text", "payload": "b", "dimension": 4}]
            )
        assert supervisor.process.pid == first_pid
    finally:
        supervisor.close()
    assert supervisor.process is None


def test_refuse_multi_model_supervisor():
    with pytest.raises(ValueError, match="exactly one"):
        ModelSupervisor({"a": sys.executable, "b": sys.executable})
