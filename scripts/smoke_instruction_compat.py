"""Live acceptance for model instructions and unchanged legacy embeddings.

Run CAPTURE against the existing deployment before updating it. Then run VERIFY
against the updated deployment with the same private baseline file. The token
comes only from VISION_EMBEDDINGS_API_TOKEN; never commit the baseline file.
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

QWEN = "qwen3-vl-embedding-8b"
WEMM = "wemm-embedding-9b"
LEGACY = "Represent the image or text for retrieval in a visual gallery."
CUSTOM = "Retrieve relevant documents that answer the user's question."
TEXT = "A blue circle beside a green triangle."
DIM = 1024


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the application's Bearer token to a redirected endpoint.
        return None


def inputs():
    image = Image.new("RGB", (32, 32), (27, 101, 177))
    output = io.BytesIO()
    image.save(output, format="PNG")
    b64 = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "qwen_text": {"model": QWEN, "text": TEXT, "dimension": DIM},
        "qwen_image": {"model": QWEN, "image_base64": b64, "dimension": DIM},
        "wemm_text": {"model": WEMM, "text": TEXT, "dimension": DIM},
        "wemm_image": {"model": WEMM, "image_base64": b64, "dimension": DIM},
    }


def post(url, token, path, body, timeout):
    request = Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    with build_opener(RejectRedirects).open(request, timeout=timeout) as response:
        return json.load(response)


def validate(result, model):
    vector = result.get("vector")
    if result.get("model") != model or not isinstance(vector, list) or len(vector) != DIM:
        raise ValueError("Unexpected model or embedding dimension")
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in vector):
        raise ValueError("Non-finite or invalid vector")
    if abs(math.sqrt(sum(x * x for x in vector)) - 1) > 0.001:
        raise ValueError("Embedding is not L2 normalized")
    return vector


def compare(a, b, label):
    cosine = sum(x * y for x, y in zip(a, b, strict=True))
    maximum = max(abs(x - y) for x, y in zip(a, b, strict=True))
    if cosine < 0.9999 or maximum > 0.01:
        raise ValueError(
            f"Vector compatibility failure {label}: cosine={cosine:.6f}, max_abs={maximum:.6f}"
        )
    print(f"VECTOR_PASS {label} cosine={cosine:.6f} max_abs={maximum:.6f}")


def provenance(result, model, effective_instruction=None):
    meta = result.get("embedding_metadata")
    if not isinstance(meta, dict):
        raise ValueError("Missing embedding_metadata")
    expected_version = "qwen3-vl-adapter-v1" if model == QWEN else "wemm-adapter-v1"
    if meta.get("preprocessing_version") != expected_version:
        raise ValueError("Preprocessing version mismatch")
    expected_hash = (
        hashlib.sha256(effective_instruction.encode()).hexdigest()
        if effective_instruction is not None
        else None
    )
    if meta.get("instruction_sha256") != expected_hash:
        raise ValueError("Instruction hash mismatch")
    if len(meta.get("config_sha256", "")) != 64:
        raise ValueError("Missing config hash")
    return meta


def capture(url, token, baseline_path, timeout):
    if baseline_path.exists():
        raise FileExistsError("Baseline file exists; refusing to overwrite")
    records = {}
    for name, request in inputs().items():
        result = post(url, token, "/v1/embeddings", request, timeout)
        records[name] = {
            "vector": validate(result, request["model"]),
            "revision": result.get("revision"),
            "input_sha256": result.get("input_sha256"),
        }
        print(f"CAPTURE_PASS {name}")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(baseline_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"format": 1, "records": records}, stream)
        stream.write("\n")
    print("BASELINE_SAVED (private, mode 0600)")


def verify(url, token, baseline_path, timeout):
    baseline = json.loads(baseline_path.read_text())
    requests = inputs()
    if baseline.get("format") != 1 or set(baseline.get("records", {})) != set(requests):
        raise ValueError("Invalid or incomplete baseline")
    defaults = {}
    for name, body in requests.items():
        result = post(url, token, "/v1/embeddings", body, timeout)
        old = baseline["records"][name]
        compare(old["vector"], validate(result, body["model"]), name)
        if result.get("revision") != old["revision"]:
            raise ValueError("Checkpoint revision changed; reindex may be required")
        if result.get("input_sha256") != old["input_sha256"]:
            raise ValueError("Raw input hash changed")
        provenance(result, body["model"], LEGACY if body["model"] == QWEN else None)
        defaults[name] = result

    for name in ("qwen_text", "qwen_image"):
        body = requests[name]
        explicit = post(
            url, token, "/v1/embeddings", dict(body, instruction=LEGACY), timeout
        )
        compare(defaults[name]["vector"], validate(explicit, QWEN), name + "_explicit")
        if explicit["embedding_metadata"] != defaults[name]["embedding_metadata"]:
            raise ValueError("Explicit legacy metadata differs")
        custom = post(url, token, "/v1/embeddings", dict(body, instruction=CUSTOM), timeout)
        validate(custom, QWEN)
        new_meta = provenance(custom, QWEN, CUSTOM)
        old_meta = defaults[name]["embedding_metadata"]
        if new_meta["config_sha256"] == old_meta["config_sha256"]:
            raise ValueError("Custom instruction did not change config hash")
        if custom["input_sha256"] != defaults[name]["input_sha256"]:
            raise ValueError("Instruction changed raw input hash")
        if CUSTOM in json.dumps(custom):
            raise ValueError("Raw instruction leaked into response")
        print(f"CUSTOM_PASS {name}")

    for name in ("wemm_text", "wemm_image"):
        try:
            post(
                url, token, "/v1/embeddings",
                dict(requests[name], instruction=CUSTOM), timeout,
            )
        except HTTPError as error:
            if error.code != 422:
                raise ValueError("Expected WeMM HTTP 422") from error
        else:
            raise ValueError("WeMM silently accepted unsupported instruction")
        print(f"WEMM_REJECTION_PASS {name}")

    batch = post(
        url, token, "/v1/embeddings/batches",
        {"items": [dict(requests["qwen_text"], instruction=CUSTOM), requests["wemm_text"]]},
        timeout,
    )
    if batch.get("count") != 2 or [
        item.get("model") for item in batch.get("items", [])
    ] != [QWEN, WEMM]:
        raise ValueError("Mixed-model batch route or order failed")
    for item in batch["items"]:
        validate(item, item["model"])
        provenance(item, item["model"], CUSTOM if item["model"] == QWEN else None)
    print("LIVE_ACCEPTANCE_PASS legacy=4 explicit=2 custom=2 rejection=2 batch=1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "verify"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if not args.base_url.startswith("https://"):
        parser.error("Expected an HTTPS base URL")
    if not 1 <= args.timeout <= 3600:
        parser.error("Timeout must be 1 to 3600 seconds")
    token = os.getenv("VISION_EMBEDDINGS_API_TOKEN", "")
    if len(token) < 16:
        parser.error("Set VISION_EMBEDDINGS_API_TOKEN securely in the environment")
    if args.mode == "capture":
        capture(args.base_url, token, args.baseline_path, args.timeout)
    else:
        verify(args.base_url, token, args.baseline_path, args.timeout)


if __name__ == "__main__":
    main()
