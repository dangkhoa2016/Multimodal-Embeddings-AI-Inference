"""Isolated CUDA model process. Communicates with parent over private FD (not stdout).

One process exists at a time. An explicit process exit frees CUDA context on
model switch, avoiding simultaneous 8B + 9B VRAM residency.
"""

import base64
import json
import os
import sys
import time


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("qwen3-vl-embedding-8b", "wemm-embedding-9b"):
        raise ValueError("Expected one configured model id")
    model_id = sys.argv[1]
    fd = int(os.environ["VISION_EMBEDDINGS_RESPONSE_FD"])
    # The protocol uses FD rather than stdout: torch/transformers may print at startup.
    stream = os.fdopen(fd, "w", encoding="utf-8", buffering=1)

    def send(payload):
        stream.write(json.dumps(payload, allow_nan=False) + "\n")
        stream.flush()

    try:
        if model_id == "wemm-embedding-9b":
            from .engines.wemm import WeMMEncoder

            engine = WeMMEncoder()
        else:
            from .engines.qwen import QwenVLEncoder

            engine = QwenVLEncoder()
        send({"ready": True, "model": model_id})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                vectors = []
                encode_started = time.perf_counter()
                for item in request["items"]:
                    data = (
                        base64.b64decode(item["image_base64"], validate=True)
                        if item["kind"] == "image"
                        else item["text"]
                    )
                    if "instruction" in item:
                        if model_id != "qwen3-vl-embedding-8b":
                            raise ValueError("Instruction is unsupported by this model")
                        vectors.append(
                            engine.encode(
                                data, item["kind"], item["dimension"],
                                instruction=item["instruction"],
                            )
                        )
                    else:
                        vectors.append(engine.encode(data, item["kind"], item["dimension"]))
                # Includes preprocessing and synchronization inside each encoder call.
                encode_ms = max(0.0, (time.perf_counter() - encode_started) * 1000.0)
                send({"ok": True, "vectors": vectors, "gpu_encode_ms": round(encode_ms, 3)})
            except Exception as exc:
                # No customer input in errors and no traceback in API response.
                print(f"Model process request failed: {type(exc).__name__}", file=sys.stderr)
                send({"ok": False, "error": "GPU inference failed"})
    finally:
        stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
