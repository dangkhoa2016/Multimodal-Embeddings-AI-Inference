"""A dedicated single-model GPU subprocess supervisor with restart semantics."""

import base64
import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path

from .checkpoints import validate_checkpoint_identity
from .model_options import effective_encoding_profile

PYTHONS = {
    "wemm-embedding-9b": "/opt/venv-wemm/bin/python",
    "qwen3-vl-embedding-8b": "/opt/venv-qwen/bin/python",
}


class ModelSupervisor:
    def __init__(self, python_paths: dict[str, str] | None = None, startup_timeout=900):
        self.python_paths = python_paths or PYTHONS
        if len(self.python_paths) != 1:
            raise ValueError("Dedicated GPU worker must have exactly one configured model")
        self.startup_timeout = startup_timeout
        self.model_id = None
        self.checkpoint_identity = None
        self.process = None
        self.pipe = None
        self.log_handle = None
        self._lock = threading.Lock()
        self._request_count = 0

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
        self.checkpoint_identity = None

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

    def _start(self, model_id: str):
        if model_id == self.model_id and self.process and self.process.poll() is None:
            return
        self.close()
        if model_id not in self.python_paths:
            raise ValueError("Unknown configured model")
        # A dedicated worker only starts its assigned model process.
        log_path = Path(
            os.getenv(
                "VISION_EMBEDDINGS_GPU_LOG", "/tmp/multimodal-embeddings-gpu-worker.log"
            )
        )
        self.log_handle = log_path.open("ab")
        rfd, wfd = os.pipe()
        env = os.environ.copy()
        env["VISION_EMBEDDINGS_RESPONSE_FD"] = str(wfd)
        try:
            self.process = subprocess.Popen(
                [
                    self.python_paths[model_id],
                    "-m",
                    "multimodal_embeddings_ai_inference.gpu_stdio",
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
        try:
            result = self._read(self.startup_timeout)
            if result.get("ready") is not True or result.get("model") != model_id:
                raise RuntimeError("GPU model initialization failed")
            self.checkpoint_identity = validate_checkpoint_identity(
                result.get("checkpoint_identity"), model_id
            )
        except Exception:
            self.close()
            raise
        self.model_id = model_id

    def encode_many_profiled(self, model_id: str, items: list[dict]) -> dict:
        """Return vectors and *measured* subprocess stages, never checkpoint paths."""
        with self._lock:
            first_request = self._request_count == 0
            active_same_model = (
                self.model_id == model_id
                and self.process is not None
                and self.process.poll() is None
            )
            if model_id not in self.python_paths:
                raise ValueError("Wrong model for this dedicated GPU worker")
            state = (
                "warm" if active_same_model else "initial_load" if first_request else "model_reload"
            )
            load_started = time.perf_counter()
            self._start(model_id)
            load_ms = (
                round((time.perf_counter() - load_started) * 1000.0, 3) if state != "warm" else 0.0
            )
            request_items = []
            for item in items:
                if "expected_revision" in item:
                    if not isinstance(item["expected_revision"], str):
                        raise ValueError("Invalid expected checkpoint revision")
                    validate_checkpoint_identity(
                        self.checkpoint_identity, model_id, item["expected_revision"]
                    )
                request = {"kind": item["kind"], "dimension": item["dimension"]}
                if item.get("model_config"):
                    request["model_config"] = item["model_config"]
                if item["kind"] == "image":
                    request["image_base64"] = base64.b64encode(item["payload"]).decode("ascii")
                else:
                    request["text"] = item["payload"]
                if "instruction" in item:
                    if model_id != "qwen3-vl-embedding-8b":
                        raise ValueError("Instruction is unsupported by this model")
                    request["instruction"] = item["instruction"]
                request_items.append(request)
            self.process.stdin.write(json.dumps({"items": request_items}) + "\n")
            self.process.stdin.flush()
            result = self._read(900)
            if not result.get("ok"):
                raise RuntimeError("GPU model returned an error")
            try:
                identity = validate_checkpoint_identity(
                    result.get("checkpoint_identity"), model_id,
                    self.checkpoint_identity["revision"],
                )
            except ValueError:
                self.close()
                raise
            vectors = result.get("vectors")
            if not isinstance(vectors, list) or len(vectors) != len(items):
                raise ValueError("Incomplete GPU embedding response")
            expected_profiles = [
                effective_encoding_profile(
                    model_id, kind=item["kind"], dimension=item["dimension"],
                    instruction=item.get("instruction"),
                    options=item.get("model_config"),
                )
                for item in items
            ]
            applied = result.get("effective_profiles")
            if not isinstance(applied, list) or applied != expected_profiles:
                raise ValueError("GPU worker effective encoding profile mismatch")
            self._request_count += 1
            encode_ms = result.get("gpu_encode_ms")
            if not isinstance(encode_ms, (int, float)) or encode_ms < 0:
                encode_ms = None
            return {
                "checkpoint_identity": identity,
                "effective_profiles": applied,
                "vectors": vectors,
                "timing": {
                    "model_state": state,
                    "gpu_worker_first_request": first_request,
                    "model_load_ms": load_ms,
                    "gpu_encode_ms": encode_ms,
                    "batch_size": len(items),
                },
            }

    def encode_many(self, model_id: str, items: list[dict]) -> list[list[float]]:
        """Preserve the original supervisor API for existing clients and tests."""
        return self.encode_many_profiled(model_id, items)["vectors"]
