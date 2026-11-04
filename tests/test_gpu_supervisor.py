"""Exercise actual process startup/switching without CUDA or pretrained weights."""

import sys

from compactfaces_ai_inference.gpu_supervisor import ModelSupervisor


def test_single_active_child_and_model_switch(tmp_path, monkeypatch):
    fake_package = tmp_path / "compactfaces_ai_inference"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("")
    (fake_package / "gpu_stdio.py").write_text("""
import json, os, sys
model = sys.argv[1]
out = os.fdopen(int(os.environ['CF_RESPONSE_FD']), 'w', buffering=1)
out.write(json.dumps({'ready': True, 'model': model}) + '\\n')
for line in sys.stdin:
    items = json.loads(line)['items']
    sign = float(1 if model.startswith('wemm') else -1)
    result = {'ok': True, 'vectors': [[sign] * x['dimension'] for x in items]}
    out.write(json.dumps(result)+'\\n')
""")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("CF_GPU_LOG", str(tmp_path / "gpu.log"))
    supervisor = ModelSupervisor(
        {
            "wemm-embedding-9b": sys.executable,
            "qwen3-vl-embedding-8b": sys.executable,
        },
        startup_timeout=10,
    )
    try:
        result = supervisor.encode_many(
            "wemm-embedding-9b",
            [
                {"kind": "text", "payload": "a", "dimension": 4},
            ],
        )
        first_pid = supervisor.process.pid
        assert result == [[1.0] * 4]
        result = supervisor.encode_many(
            "wemm-embedding-9b",
            [
                {"kind": "text", "payload": "a", "dimension": 2},
            ],
        )
        assert supervisor.process.pid == first_pid
        assert result == [[1.0] * 2]
        result = supervisor.encode_many(
            "qwen3-vl-embedding-8b",
            [
                {"kind": "text", "payload": "b", "dimension": 4},
            ],
        )
        assert supervisor.process.pid != first_pid
        assert result == [[-1.0] * 4]
    finally:
        supervisor.close()
    assert supervisor.process is None
