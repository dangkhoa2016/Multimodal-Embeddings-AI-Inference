"""Exercise dedicated model subprocess startup and warm reuse without CUDA."""

import io
import json
import os
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
identity = {'model': model, 'repository': 'tencent/WeMM-Embedding-9B', 'revision': 'a' * 40}
out = os.fdopen(int(os.environ['VISION_EMBEDDINGS_RESPONSE_FD']), 'w', buffering=1)
out.write(json.dumps({'ready': True, 'model': model, 'checkpoint_identity': identity}) + '\\n')
for line in sys.stdin:
    items = json.loads(line)['items']
    with open(os.environ['ENCODE_LOG'], 'a') as log:
        log.write('encode\\n')
    result = {'ok': True, 'vectors': [[1.0] * x['dimension'] for x in items], 'gpu_encode_ms': 0.3}
    result['checkpoint_identity'] = identity
    out.write(json.dumps(result) + '\\n')
""")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("VISION_EMBEDDINGS_GPU_LOG", str(tmp_path / "gpu.log"))
    monkeypatch.setenv("ENCODE_LOG", str(tmp_path / "encode.log"))
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
        assert first["checkpoint_identity"] == {
            "model": model, "repository": "tencent/WeMM-Embedding-9B", "revision": "a" * 40,
        }
        monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", "b" * 40)
        with pytest.raises(ValueError, match="checkpoint identity"):
            supervisor.encode_many_profiled(model, [{
                "kind": "text", "payload": "reject before encoding", "dimension": 2,
                "expected_revision": "b" * 40,
            }])
        assert (tmp_path / "encode.log").read_text().splitlines() == ["encode"]
        warm = supervisor.encode_many_profiled(
            model, [{"kind": "text", "payload": "a", "dimension": 2}]
        )
        assert supervisor.process.pid == first_pid
        assert warm["timing"]["model_state"] == "warm"
        assert warm["timing"]["gpu_worker_first_request"] is False
        assert warm["timing"]["model_load_ms"] == 0
        assert warm["checkpoint_identity"] == first["checkpoint_identity"]
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


def test_stdio_freezes_verified_encoder_identity(monkeypatch):
    from multimodal_embeddings_ai_inference import gpu_stdio
    from multimodal_embeddings_ai_inference.engines import wemm

    model = "wemm-embedding-9b"
    identity = {"model": model, "repository": "tencent/WeMM-Embedding-9B", "revision": "a" * 40}

    class FakeEncoder:
        def __init__(self):
            self.checkpoint_identity = dict(identity, path="/private/checkpoint")

        def encode(self, *args):
            self.checkpoint_identity["revision"] = "b" * 40
            monkeypatch.setenv("VISION_EMBEDDINGS_WEMM9_REVISION", "b" * 40)
            return [1.0, 0.0]

    rfd, wfd = os.pipe()
    monkeypatch.setenv("VISION_EMBEDDINGS_RESPONSE_FD", str(wfd))
    monkeypatch.setattr(sys, "argv", ["gpu_stdio", model])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "items": [{"kind": "text", "text": "hi", "dimension": 2}],
    }) + "\n"))
    monkeypatch.setattr(wemm, "WeMMEncoder", FakeEncoder)
    with os.fdopen(rfd) as reader:
        assert gpu_stdio.main() == 0
        frames = [json.loads(line) for line in reader]
    assert len(frames) == 2
    assert frames[0]["checkpoint_identity"] == frames[1]["checkpoint_identity"] == identity


@pytest.mark.parametrize("failure", ["missing_ready_identity", "changed_result_identity"])
def test_supervisor_closes_unverified_process(tmp_path, monkeypatch, failure):
    package = tmp_path / "multimodal_embeddings_ai_inference"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "gpu_stdio.py").write_text("""
import json, os, sys
identity = {'model': sys.argv[1], 'repository': 'tencent/WeMM-Embedding-9B', 'revision': 'a' * 40}
out = os.fdopen(int(os.environ['VISION_EMBEDDINGS_RESPONSE_FD']), 'w', buffering=1)
ready = {'ready': True, 'model': sys.argv[1]}
if os.environ['PROTOCOL_FAILURE'] != 'missing_ready_identity':
    ready['checkpoint_identity'] = identity
out.write(json.dumps(ready) + '\\n')
for line in sys.stdin:
    identity['revision'] = 'b' * 40
    out.write(json.dumps({'ok': True, 'vectors': [[1.0]], 'checkpoint_identity': identity}) + '\\n')
""")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PROTOCOL_FAILURE", failure)
    monkeypatch.setenv("VISION_EMBEDDINGS_GPU_LOG", str(tmp_path / "gpu.log"))
    model = "wemm-embedding-9b"
    supervisor = ModelSupervisor({model: sys.executable}, startup_timeout=10)
    try:
        with pytest.raises(ValueError, match="checkpoint identity"):
            supervisor.encode_many(model, [{"kind": "text", "payload": "hi", "dimension": 1}])
        assert supervisor.process is supervisor.checkpoint_identity is None
    finally:
        supervisor.close()
