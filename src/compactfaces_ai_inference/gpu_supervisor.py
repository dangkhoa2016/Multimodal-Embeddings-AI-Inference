"""Single-GPU multi-model supervisor with deterministic unload/reload semantics."""

import base64
import json
import os
import select
import subprocess
import threading
from pathlib import Path

PYTHONS = {
    "wemm-embedding-9b": "/opt/venv-wemm/bin/python",
    "qwen3-vl-embedding-8b": "/opt/venv-qwen/bin/python",
}


class ModelSupervisor:
    def __init__(self, python_paths: dict[str, str] | None = None, startup_timeout=900):
        self.python_paths = python_paths or PYTHONS
        self.startup_timeout = startup_timeout
        self.model_id = None
        self.process = None
        self.pipe = None
        self.log_handle = None
        self._lock = threading.Lock()

    def close(self):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            if self.process.stdin:
                self.process.stdin.close()
            self.process = None
        if self.pipe:
            self.pipe.close()
            self.pipe = None
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
        self.model_id = None

    def _read(self, seconds: int) -> dict:
        if self.pipe is None:
            raise RuntimeError("Worker not started")
        ready, _, _ = select.select([self.pipe], [], [], seconds)
        if not ready:
            self.close()
            raise TimeoutError("Model worker response timed out")
        line = self.pipe.readline()
        if not line:
            if self.process is not None:
                try:
                    self.process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    pass
            # Log the model import traceback to Modal server logs, never to API clients.
            try:
                if self.log_handle is not None:
                    self.log_handle.flush()
                    path = Path(self.log_handle.name)
                    with path.open("rb") as output:
                        output.seek(max(0, path.stat().st_size - 12000))
                        diagnostic = output.read().decode("utf-8", errors="replace")
                    print("GPU worker exited. Last diagnostic lines:\n" + diagnostic, flush=True)
            except OSError:
                print("GPU worker exited; diagnostic log unavailable", flush=True)
            self.close()
            raise RuntimeError("GPU model worker exited unexpectedly")
        return json.loads(line)

    def _switch(self, model_id: str):
        if model_id == self.model_id and self.process and self.process.poll() is None:
            return
        self.close()
        if model_id not in self.python_paths:
            raise ValueError("Unknown configured model")
        # Only one model subprocess is alive, hence one CUDA context at a time.
        log_path = Path(os.getenv("CF_GPU_LOG", "/tmp/compactfaces-gpu-worker.log"))
        self.log_handle = log_path.open("ab")
        rfd, wfd = os.pipe()
        env = os.environ.copy()
        env["CF_RESPONSE_FD"] = str(wfd)
        try:
            self.process = subprocess.Popen(
                [
                    self.python_paths[model_id],
                    "-m",
                    "compactfaces_ai_inference.gpu_stdio",
                    model_id,
                ],
                stdin=subprocess.PIPE,
                stdout=self.log_handle,
                stderr=self.log_handle,
                env=env,
                pass_fds=(wfd,),
                bufsize=1,
                text=True,
            )
        except Exception:
            os.close(rfd)
            os.close(wfd)
            self.close()
            raise
        os.close(wfd)
        self.pipe = os.fdopen(rfd, "r", encoding="utf-8")
        result = self._read(self.startup_timeout)
        if result.get("ready") is not True or result.get("model") != model_id:
            self.close()
            raise RuntimeError("GPU model initialization failed")
        self.model_id = model_id

    def encode_many(self, model_id: str, items: list[dict]) -> list[list[float]]:
        with self._lock:
            self._switch(model_id)
            request_items = []
            for item in items:
                request = {"kind": item["kind"], "dimension": item["dimension"]}
                if item["kind"] == "image":
                    request["image_base64"] = base64.b64encode(item["payload"]).decode("ascii")
                else:
                    request["text"] = item["payload"]
                request_items.append(request)
            self.process.stdin.write(json.dumps({"items": request_items}) + "\n")
            self.process.stdin.flush()
            result = self._read(900)
            if not result.get("ok"):
                raise RuntimeError("GPU model returned an error")
            vectors = result.get("vectors")
            if not isinstance(vectors, list) or len(vectors) != len(items):
                raise ValueError("Incomplete GPU embedding response")
            return vectors
