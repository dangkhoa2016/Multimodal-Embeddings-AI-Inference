# Source for the standalone notebook code cells.
# %% NOTEBOOK CELL config
#@title Benchmark configuration / Cấu hình benchmark { display-mode: "form" }
import os
from pathlib import Path

RUN_BENCHMARK = False #@param {type:"boolean"}
ALLOW_CREDENTIAL_PROMPTS = False #@param {type:"boolean"}
RUN_COLD = False #@param {type:"boolean"}
RUN_WARM = True #@param {type:"boolean"}
RUN_ALTERNATE = False #@param {type:"boolean"}
RUN_CONCURRENT = False #@param {type:"boolean"}
RUN_MIXED_BATCH = False #@param {type:"boolean"}
CONCURRENT_WORKERS = 2 #@param {type:"integer"}
CONCURRENT_REQUESTS = 4 #@param {type:"integer"}
MIXED_BATCH_REPETITIONS = 3 #@param {type:"integer"}
QWEN_INSTRUCTION_MODE = "default" #@param ["default", "custom"]
QWEN_CUSTOM_INSTRUCTION = "Retrieve images relevant to this description." #@param {type:"string"}
QWEN_LEGACY_INSTRUCTION = "Represent the image or text for retrieval in a visual gallery."
# Deployment-specific starting examples; edit to match your own Modal settings.
# Ví dụ cấu hình ban đầu; hãy sửa theo deployment Modal của bạn.
GATEWAY_SCALEDOWN_SECONDS = 60 #@param {type:"integer"}
GPU_SCALEDOWN_SECONDS = 180 #@param {type:"integer"}
COLD_WINDOW_PADDING_SECONDS = 30 #@param {type:"integer"}
COLD_REPETITIONS = 5 #@param {type:"integer"}
COLD_TARGET_MODEL = "both" #@param ["both", "qwen3-vl-embedding-8b", "wemm-embedding-9b"]
WARM_REPETITIONS = 4 #@param {type:"integer"}
WARMUP_REQUESTS = 1 #@param {type:"integer"}
ALTERNATE_REPETITIONS = 5 #@param {type:"integer"}
GATEWAY_MIN_CONTAINERS = 0 #@param {type:"integer"}
GPU_MIN_CONTAINERS = 0 #@param {type:"integer"}
DIMENSION = 1024
TIMEOUT_SECONDS = 900
MODELS = ("qwen3-vl-embedding-8b", "wemm-embedding-9b")
# SHA256 of published code-cell template, excluding this self-referential assignment.
NOTEBOOK_TEMPLATE_SHA256 = "191d75b7104940c064ff0afde5118f63feffde5bd50b3733a9b06910ed6f3481"
TEXT_SAMPLE = os.environ.get("BENCHMARK_TEXT", "a blue sphere on a light background").strip()
IMAGE_PATH = os.environ.get("BENCHMARK_IMAGE_PATH", "").strip()

def require_config(condition, english, vietnamese):
    if not condition:
        raise ValueError(english + " / " + vietnamese)


def require_integer_range(name, value, low, high):
    require_config(
        type(value) is int and low <= value <= high,
        f"{name} must be an integer from {low} to {high}",
        f"{name} phải là số nguyên từ {low} đến {high}",
    )


require_config(
    1 <= len(TEXT_SAMPLE) <= 8192,
    "BENCHMARK_TEXT (TEXT_SAMPLE) must contain 1–8192 characters",
    "BENCHMARK_TEXT (TEXT_SAMPLE) phải có 1–8192 ký tự",
)
require_config(DIMENSION == 1024, "DIMENSION must remain 1024",
               "DIMENSION phải giữ bằng 1024")
require_integer_range("CONCURRENT_WORKERS", CONCURRENT_WORKERS, 1, 4)
require_integer_range("CONCURRENT_REQUESTS", CONCURRENT_REQUESTS, 2, 16)
require_integer_range("MIXED_BATCH_REPETITIONS", MIXED_BATCH_REPETITIONS, 1, 10)
require_config(
    QWEN_INSTRUCTION_MODE in ("default", "custom"),
    "QWEN_INSTRUCTION_MODE must be default or custom",
    "QWEN_INSTRUCTION_MODE phải là default hoặc custom",
)
QWEN_CUSTOM_INSTRUCTION = os.environ.get("BENCHMARK_QWEN_INSTRUCTION", QWEN_CUSTOM_INSTRUCTION).strip()
if QWEN_INSTRUCTION_MODE == "custom":
    require_config(
        1 <= len(QWEN_CUSTOM_INSTRUCTION) <= 512,
        "BENCHMARK_QWEN_INSTRUCTION must contain 1–512 characters in custom mode",
        "BENCHMARK_QWEN_INSTRUCTION phải có 1–512 ký tự ở chế độ custom",
    )
    require_config(
        all(ord(ch) >= 32 and ord(ch) != 127 for ch in QWEN_CUSTOM_INSTRUCTION),
        "BENCHMARK_QWEN_INSTRUCTION must not contain control characters",
        "BENCHMARK_QWEN_INSTRUCTION không được chứa ký tự điều khiển",
    )
require_integer_range("COLD_REPETITIONS", COLD_REPETITIONS, 1, 20)
require_config(
    COLD_TARGET_MODEL in ("both", *MODELS),
    "COLD_TARGET_MODEL must be both or a supported model ID",
    "COLD_TARGET_MODEL phải là both hoặc ID của model được hỗ trợ",
)
COLD_ACTIVE_MODELS = MODELS if COLD_TARGET_MODEL == "both" else (COLD_TARGET_MODEL,)
require_config(
    type(GATEWAY_SCALEDOWN_SECONDS) is int and GATEWAY_SCALEDOWN_SECONDS >= 0
    and type(GPU_SCALEDOWN_SECONDS) is int and GPU_SCALEDOWN_SECONDS >= 0,
    "GATEWAY_SCALEDOWN_SECONDS and GPU_SCALEDOWN_SECONDS must be nonnegative integers",
    "GATEWAY_SCALEDOWN_SECONDS và GPU_SCALEDOWN_SECONDS phải là số nguyên không âm",
)
require_integer_range("COLD_WINDOW_PADDING_SECONDS", COLD_WINDOW_PADDING_SECONDS, 1, 3600)
COLD_IDLE_WAIT_SECONDS = (
    max(GATEWAY_SCALEDOWN_SECONDS, GPU_SCALEDOWN_SECONDS)
    + COLD_WINDOW_PADDING_SECONDS
)
COLD_TOTAL_IDLE_SECONDS = (
    len(COLD_ACTIVE_MODELS) * COLD_REPETITIONS * COLD_IDLE_WAIT_SECONDS if RUN_COLD else 0
)
require_config(
    COLD_IDLE_WAIT_SECONDS <= 86400,
    "Calculated COLD_IDLE_WAIT_SECONDS must not exceed 86400",
    "COLD_IDLE_WAIT_SECONDS tính được không được vượt 86400",
)
require_integer_range("WARM_REPETITIONS", WARM_REPETITIONS, 2, 1000)
require_config(
    WARM_REPETITIONS % 2 == 0,
    "WARM_REPETITIONS must be even to balance model ordering",
    "WARM_REPETITIONS phải là số chẵn để cân bằng thứ tự model",
)
require_integer_range("WARMUP_REQUESTS", WARMUP_REQUESTS, 1, 50)
require_integer_range("ALTERNATE_REPETITIONS", ALTERNATE_REPETITIONS, 1, 50)
if RUN_BENCHMARK and RUN_COLD:
    require_config(
        GATEWAY_MIN_CONTAINERS == GPU_MIN_CONTAINERS == 0,
        "Cold runs require GATEWAY_MIN_CONTAINERS=0 and GPU_MIN_CONTAINERS=0",
        "Chạy cold yêu cầu GATEWAY_MIN_CONTAINERS=0 và GPU_MIN_CONTAINERS=0",
    )
    require_config(
        COLD_IDLE_WAIT_SECONDS > max(GATEWAY_SCALEDOWN_SECONDS, GPU_SCALEDOWN_SECONDS),
        "COLD_IDLE_WAIT_SECONDS must exceed both scale-down intervals",
        "COLD_IDLE_WAIT_SECONDS phải lớn hơn cả hai khoảng scale-down",
    )
# Calculate an upper bound for approved HTTP POST attempts before any credentials
# or network access. A failed or interrupted phase can dispatch fewer requests.
PLANNED_POSTS = {
    "cold": len(COLD_ACTIVE_MODELS) * COLD_REPETITIONS if RUN_COLD else 0,
    "warmup": 2 * len(MODELS) * 2 * WARMUP_REQUESTS if RUN_WARM else 0,
    "warm_measured": 2 * len(MODELS) * WARM_REPETITIONS if RUN_WARM else 0,
    "alternate_setup": len(MODELS) if RUN_ALTERNATE else 0,
    "alternate_measured": len(MODELS) * ALTERNATE_REPETITIONS if RUN_ALTERNATE else 0,
    "concurrent": len(MODELS) * CONCURRENT_REQUESTS if RUN_CONCURRENT else 0,
    "mixed_batch": MIXED_BATCH_REPETITIONS if RUN_MIXED_BATCH else 0,
}
PLANNED_POST_TOTAL = sum(PLANNED_POSTS.values())
# Each mixed batch is one HTTP POST containing two model-specific embedding items.
PLANNED_EMBEDDING_ITEMS = PLANNED_POST_TOTAL + PLANNED_POSTS["mixed_batch"]
print("Planned workload (preview only) / Khối lượng dự kiến (chỉ xem trước)")
print("Authorization / Cho phép chạy:", "ON / BẬT" if RUN_BENCHMARK else "OFF / TẮT")
for phase, english, vietnamese in (
    ("cold", "Cold image requests", "Lượt đo ảnh cold"),
    ("warmup", "Warm-up requests (excluded from measurements)", "Request làm nóng (không tính vào kết quả đo)"),
    ("warm_measured", "Warm measured requests", "Request đo warm"),
    ("alternate_setup", "Alternating setup requests", "Request chuẩn bị đo luân phiên"),
    ("alternate_measured", "Alternating measured requests", "Request đo luân phiên"),
    ("concurrent", "Concurrent requests (maximum)", "Request đồng thời (tối đa)"),
    ("mixed_batch", "Mixed-batch HTTP requests (two items each)", "HTTP request batch hai model (mỗi request hai item)"),
):
    if PLANNED_POSTS[phase]:
        print(f"  {english} / {vietnamese}: {PLANNED_POSTS[phase]}")
print("Total planned HTTP POST attempts (upper bound) / "
      "Tổng lượt HTTP POST dự kiến (giới hạn trên):", PLANNED_POST_TOTAL)
print("Total planned embedding items / Tổng embedding items dự kiến:",
      PLANNED_EMBEDDING_ITEMS)
if RUN_COLD:
    print("Calculated idle only (excludes API/GPU processing) / "
          "Chỉ tính thời gian chờ cold (không gồm thời gian xử lý API/GPU):",
          COLD_IDLE_WAIT_SECONDS, "seconds per trial / giây mỗi lượt;",
          round(COLD_TOTAL_IDLE_SECONDS / 60, 2), "minutes total / phút tổng cộng")
print("Actual dispatch may stop early; no automatic retry or cost estimate. / "
      "Lượt gửi thực tế có thể dừng sớm; không tự thử lại hay ước tính chi phí.")

if RUN_COLD:
    print("Verify your actual Modal settings; displayed defaults are examples only. / Kiểm tra cấu hình Modal thực tế; các giá trị hiển thị chỉ là ví dụ.")
if not RUN_BENCHMARK:
    print("OFF / ĐANG TẮT: Set RUN_BENCHMARK=True to authorize live benchmark. / "
          "Đặt RUN_BENCHMARK=True để cho phép chạy benchmark thật.")
# %% NOTEBOOK CELL client
#@title API client and preflight (run; no edits) / Kết nối và kiểm tra API (chạy, không sửa) { display-mode: "form" }
import base64
# Client map / Sơ đồ client: validate settings, create an immutable run identity,
# authenticate API requests, verify telemetry, and persist redacted attempt journals.
import hashlib
import json
import math
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from getpass import getpass
from http.client import InvalidURL
from urllib.parse import urlsplit

class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, url):
        return None

opener = urllib.request.build_opener(RejectRedirect)
base_url, token = "", ""
rows, revisions, embedding_configs = [], {}, {}
concurrent_trials, concurrent_summary, batch_trials = [], [], []
sequence = 0
run_failure = None
run_completed_phases = set()
run_started_phases = set()  # Fail-closed on an interrupted or re-executed cell.
run_requested_phases = {
    name for name, enabled in (
        ("cold", RUN_COLD), ("warm", RUN_WARM),
        ("alternate", RUN_ALTERNATE), ("concurrent", RUN_CONCURRENT),
        ("mixed_batch", RUN_MIXED_BATCH),
    ) if RUN_BENCHMARK and enabled
}

def operator_source_commit(name):
    value = os.environ.get(name, "").strip().lower()
    if value and not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(name + " must be a 40-character Git SHA / phải là Git SHA gồm 40 ký tự")
    return {"value": value or None,
            "status": "operator_provided" if value else "unavailable"}

provenance = {
    "notebook_template": {
        "code_sha256": NOTEBOOK_TEMPLATE_SHA256,
        "status": "embedded_template_digest_not_runtime_verified",
    },
    "client_source_commit": operator_source_commit("BENCHMARK_CLIENT_SOURCE_SHA"),
    "server_source_commit": operator_source_commit("BENCHMARK_SERVER_SOURCE_SHA"),
    "note": "Declared commits are not verified; the client commit is never the server commit.",
}
benchmark_config = {
    "models": list(MODELS), "dimension": DIMENSION,
    "timeout_seconds": TIMEOUT_SECONDS,
    "run_cold": RUN_COLD, "run_warm": RUN_WARM,
    "run_alternate": RUN_ALTERNATE,
    "run_concurrent": RUN_CONCURRENT, "run_mixed_batch": RUN_MIXED_BATCH,
    "cold_target_model": COLD_TARGET_MODEL,
    "cold_idle_wait_seconds": COLD_IDLE_WAIT_SECONDS,
    "cold_window_padding_seconds": COLD_WINDOW_PADDING_SECONDS,
    "estimated_total_idle_seconds": COLD_TOTAL_IDLE_SECONDS,
    "gateway_scaledown_seconds_declared": GATEWAY_SCALEDOWN_SECONDS,
    "gpu_scaledown_seconds_declared": GPU_SCALEDOWN_SECONDS,
    "gateway_min_containers_declared": GATEWAY_MIN_CONTAINERS,
    "gpu_min_containers_declared": GPU_MIN_CONTAINERS,
    "cold_attempts_per_model": COLD_REPETITIONS if RUN_COLD else 0,
    "warm_attempts_per_modality_per_model": WARM_REPETITIONS if RUN_WARM else 0,
    "warmup_requests_per_modality_per_model_per_block": WARMUP_REQUESTS if RUN_WARM else 0,
    "warmup_requests_per_modality_per_model": 2 * WARMUP_REQUESTS if RUN_WARM else 0,
    "alternate_attempts_per_model": ALTERNATE_REPETITIONS if RUN_ALTERNATE else 0,
    "qwen_instruction_mode": QWEN_INSTRUCTION_MODE,
    "qwen_instruction_sha256": (
        hashlib.sha256(QWEN_CUSTOM_INSTRUCTION.encode("utf-8")).hexdigest()
        if QWEN_INSTRUCTION_MODE == "custom" else None
    ),
    "concurrent_workers": CONCURRENT_WORKERS if RUN_CONCURRENT else 0,
    "concurrent_requests_per_model": CONCURRENT_REQUESTS if RUN_CONCURRENT else 0,
    "mixed_batch_repetitions": MIXED_BATCH_REPETITIONS if RUN_MIXED_BATCH else 0,
    "worker_id_source": "optional per-worker lifetime ID",
    "retries": 0, "sequential_cold_warm_alternate": True,
}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sample_png(size=128):
    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))
    scanlines = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            row.extend((255 * x // (size-1), 255 * y // (size-1), 120 + (x+y)%2*65))
        scanlines.append(bytes(row))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(scanlines), 9))
            + chunk(b"IEND", b""))

def read_benchmark_image(path):
    """Read a benchmark image without exposing private filesystem paths on failure."""
    if not path:
        return sample_png()
    try:
        return Path(path).expanduser().read_bytes()
    except (OSError, ValueError):
        raise ValueError(
            "Cannot read the selected benchmark image in this runtime; "
            "check the file and permissions. / "
            "Không thể đọc ảnh benchmark đã chọn trong runtime này; "
            "hãy kiểm tra file và quyền truy cập."
        ) from None

image_bytes = read_benchmark_image(IMAGE_PATH)
if not 0 < len(image_bytes) <= 5*1024*1024:
    raise ValueError("Image must be between 1 and 5,242,880 bytes / Ảnh phải có kích thước từ 1 đến 5.242.880 byte")
if not (image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        or image_bytes.startswith(b"\xff\xd8\xff")
        or (image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP")):
    raise ValueError("Input must be PNG/JPEG/WebP / Đầu vào phải là ảnh PNG/JPEG/WebP")
hashes = {"text": hashlib.sha256(TEXT_SAMPLE.encode("utf-8")).hexdigest(),
          "image": hashlib.sha256(image_bytes).hexdigest()}

# A run has immutable, secret-free input/configuration identity.
from uuid import uuid4

# -- Immutable run identity and stale-configuration guards --
def current_run_configuration():
    image_source = read_benchmark_image(IMAGE_PATH)
    return {
        "run_benchmark": RUN_BENCHMARK,
        "run_cold": RUN_COLD, "run_warm": RUN_WARM,
        "run_alternate": RUN_ALTERNATE, "run_concurrent": RUN_CONCURRENT,
        "run_mixed_batch": RUN_MIXED_BATCH,
        "concurrent_workers": CONCURRENT_WORKERS,
        "concurrent_requests": CONCURRENT_REQUESTS,
        "mixed_batch_repetitions": MIXED_BATCH_REPETITIONS,
        "cold_repetitions": COLD_REPETITIONS, "cold_target_model": COLD_TARGET_MODEL,
        "cold_active_models": list(COLD_ACTIVE_MODELS),
        "gateway_scaledown_seconds": GATEWAY_SCALEDOWN_SECONDS,
        "gpu_scaledown_seconds": GPU_SCALEDOWN_SECONDS,
        "cold_window_padding_seconds": COLD_WINDOW_PADDING_SECONDS,
        "cold_idle_wait_seconds": COLD_IDLE_WAIT_SECONDS,
        "cold_total_idle_seconds": COLD_TOTAL_IDLE_SECONDS,
        "gateway_min_containers": GATEWAY_MIN_CONTAINERS,
        "gpu_min_containers": GPU_MIN_CONTAINERS,
        "warm_repetitions": WARM_REPETITIONS, "warmup_requests": WARMUP_REQUESTS,
        "alternate_repetitions": ALTERNATE_REPETITIONS,
        "qwen_instruction_mode": QWEN_INSTRUCTION_MODE,
        "qwen_instruction_sha256": hashlib.sha256(
            (QWEN_CUSTOM_INSTRUCTION if QWEN_INSTRUCTION_MODE == "custom"
             else QWEN_LEGACY_INSTRUCTION).encode("utf-8")
        ).hexdigest(),
        "models": list(MODELS), "dimension": DIMENSION,
        "timeout_seconds": TIMEOUT_SECONDS,
        "text_sha256": hashlib.sha256(TEXT_SAMPLE.encode("utf-8")).hexdigest(),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_source_sha256": hashlib.sha256(image_source).hexdigest(),
    }

run_id = uuid4().hex
run_configuration_snapshot = current_run_configuration()
run_configuration_sha256 = hashlib.sha256(json.dumps(
    run_configuration_snapshot, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
benchmark_config["run_id"] = run_id
benchmark_config["run_configuration_sha256"] = run_configuration_sha256

def assert_run_active(phase):
    """Reject changed settings/inputs and repeated phases before any POST."""
    if run_failure is not None:
        raise RuntimeError("Earlier benchmark failure; start a new run via client / Benchmark trước đã lỗi; hãy bắt đầu lần chạy mới từ cell client")
    try:
        matches = current_run_configuration() == run_configuration_snapshot
    except (OSError, ValueError, TypeError):
        matches = False
    if not matches:
        stop_benchmark(phase, "RunConfigurationChanged")
    if "preflight" not in run_completed_phases:
        stop_benchmark(phase, "PreflightNotCompleted")
    if phase != "report" and phase in run_completed_phases:
        stop_benchmark(phase, "PhaseAlreadyCompleted")
    if phase != "report" and phase in run_started_phases:
        stop_benchmark(phase, "PhaseAlreadyStarted")
    if phase != "report":
        # Register before any sleep, thread dispatch or chargeable POST. A Stop
        # button can interrupt a cell without running a Python except block.
        run_started_phases.add(phase)
        write_benchmark_report({
            "status": "IN_PROGRESS", "phase": phase, "created_utc": utc_now(),
            "provenance": provenance, "config": benchmark_config,
            "summary": {}, "trials": rows,
            "concurrent": {"summary": concurrent_summary, "trials": concurrent_trials},
            "mixed_model_batches": batch_trials,
            "limitations": [
                "Phase started; a partial or interrupted phase cannot be retried in this run.",
                "Re-run API client/preflight to start a fresh run after interruption.",
            ],
        })


def validated_benchmark_token(value):
    if (not isinstance(value, str) or len(value) < 16
            or not all(33 <= ord(ch) <= 126 for ch in value)):
        raise ValueError("Invalid Bearer token format / Định dạng Bearer token không hợp lệ")
    return value


def validated_benchmark_base_url(value):
    value = value.strip().rstrip("/")
    try:
        if any(ord(ch) <= 32 or ord(ch) == 127 for ch in value):
            raise ValueError
        parsed = urlsplit(value)
        parsed.port  # Validate malformed and out-of-range ports before HTTP.
        if (parsed.scheme != "https" or not parsed.hostname
                or parsed.username or parsed.password or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment):
            raise ValueError
    except ValueError:
        raise ValueError(
            "HTTPS base URL only, no /v1, whitespace or embedded credentials. / "
            "Chỉ nhập URL HTTPS gốc hợp lệ, không có /v1, khoảng trắng hoặc credential."
        ) from None
    return value


# -- Authenticated HTTP client and redacted attempt journal --
def http_json(path, body=None):
    safe_token = validated_benchmark_token(token)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Authorization": "Bearer " + safe_token}
    if data is not None:
        headers["Content-Type"] = "application/json"
    try:
        req = urllib.request.Request(validated_benchmark_base_url(base_url) + path, data=data, headers=headers,
                                     method="POST" if body is not None else "GET")
        with urllib.request.build_opener(RejectRedirect).open(req, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError("HTTP %d on %s; no retry / không thử lại" % (error.code, path)) from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("HTTP network/timeout failure on %s; no retry / lỗi mạng hoặc quá thời gian; không thử lại" % path) from None
    except (ValueError, UnicodeError, InvalidURL):
        # Never surface raw urllib header errors with credentials.
        raise RuntimeError("Invalid HTTP request/response encoding; no retry / Mã hóa HTTP request/response không hợp lệ; không thử lại") from None

def write_benchmark_report(payload):
    """Atomically replace a prior report without persisting credentials."""
    output_dir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    output = output_dir / "multimodal_embeddings_performance_summary.json"
    staged = output.with_name(output.name + ".tmp")
    try:
        staged.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        os.replace(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    return output

def save_failed_report(phase, error_type, failed_trial=None, interrupted=False):
    """Keep redacted partial evidence without finalizing aggregate statistics."""
    global run_failure
    run_failure = {"phase": phase, "error_type": error_type}
    if failed_trial is not None:
        failed_trial.update(status="failed", confirmed=False, error_type=error_type)
    partial = {
        "status": "INCOMPLETE_INTERRUPTED" if interrupted else "INCOMPLETE_FAILED",
        "created_utc": utc_now(),
        "failure": {"phase": phase, "error_type": error_type},
        "models": {m: {"revision": revisions.get(m), "dimension": DIMENSION}
                   for m in MODELS},
        "input_sha256": hashes, "embedding_configs": embedding_configs,
        "provenance": provenance, "config": benchmark_config,
        "retries": 0, "sequential": True, "summary": {}, "trials": rows,
        "started_phases": sorted(run_started_phases),
        "concurrent": {"summary": concurrent_summary, "trials": concurrent_trials},
        "mixed_model_batches": batch_trials,
        "limitations": [
            "Incomplete run: no aggregate statistics have been finalized.",
            "Requests already in flight may finish before the concurrent phase stops.",
            "No retry; no endpoint, credentials, raw inputs or embedding vectors saved.",
        ],
    }
    return write_benchmark_report(partial)

def stop_benchmark(phase, error_type, failed_trial=None):
    output = save_failed_report(phase, error_type, failed_trial)
    raise RuntimeError("Benchmark stopped / Benchmark đã dừng ở bước " + phase + " (" + error_type +
                       "); partial report saved / đã lưu báo cáo chưa hoàn tất: " + output.name)

def interrupt_benchmark(phase):
    """Called only on KeyboardInterrupt; never schedule another POST in this run."""
    if run_failure is None:
        save_failed_report(phase, "KeyboardInterrupt", interrupted=True)
    print("INTERRUPTED / ĐÃ NGẮT: Re-run API client/preflight for a NEW run. "
          "Do not re-run this phase; in-flight server work may continue. / "
          "Chạy lại cell client/preflight để bắt đầu lượt MỚI; đừng chạy lại phase "
          "đã ngắt vì tác vụ trên server vẫn có thể tiếp tục.")

if RUN_BENCHMARK:
    write_benchmark_report({
        "status": "IN_PROGRESS", "created_utc": utc_now(),
        "provenance": provenance, "config": benchmark_config,
        "summary": {}, "trials": [], "concurrent": {"summary": [], "trials": []},
        "mixed_model_batches": [],
        "limitations": ["Benchmark started; any prior report has been replaced."],
    })
    try:
        if (not ALLOW_CREDENTIAL_PROMPTS
                and (not os.environ.get("VISION_EMBEDDINGS_BASE_URL", "").strip()
                     or not os.environ.get("VISION_EMBEDDINGS_API_TOKEN", "").strip())):
            raise RuntimeError(
                "Missing API credentials: set VISION_EMBEDDINGS_BASE_URL and "
                "VISION_EMBEDDINGS_API_TOKEN, or enable ALLOW_CREDENTIAL_PROMPTS "
                "for an interactive run. / Thiếu thông tin xác thực API: đặt hai "
                "biến môi trường hoặc cho phép prompt khi chạy tương tác."
            )
        base_url = os.environ.get("VISION_EMBEDDINGS_BASE_URL", "") or getpass(
            "HTTPS endpoint / URL API (hidden): ")
        base_url = validated_benchmark_base_url(base_url)
        token = os.environ.get("VISION_EMBEDDINGS_API_TOKEN", "") or getpass(
            "Application Bearer token / Token ứng dụng (hidden): ")
        validated_benchmark_token(token)
        if http_json("/v1/health").get("backend") != "modal":
            raise RuntimeError("Real Modal endpoint required; mock rejected / Cần endpoint Modal thật; không chấp nhận mock")
        catalog = http_json("/v1/models")
        found = {m["id"]: m for m in catalog.get("models", [])}
        if not set(MODELS).issubset(found):
            raise RuntimeError("Both embedding models must be available in catalog / Danh mục API phải có đủ hai model embedding")
        if QWEN_INSTRUCTION_MODE == "custom" and found[MODELS[0]].get("supports_custom_instruction") is not True:
            raise RuntimeError("API lacks Qwen custom instruction capability / API không hỗ trợ instruction tùy chỉnh của Qwen")
        print("Verified live API/catalog / Đã xác thực API và model.")
        run_completed_phases.add("preflight")
    except KeyboardInterrupt:
        interrupt_benchmark("preflight")
        raise
    except Exception as error:
        save_failed_report("preflight", type(error).__name__)
        raise

# %% NOTEBOOK CELL client-request-helpers
#@title Request and validation helpers (run; no edits) / Hàm hỗ trợ request và kiểm tra (chạy, không sửa) { display-mode: "form" }
# -- Request construction and response validation --
def request_body(model, modality, reqid):
    body = {"model": model, "dimension": DIMENSION, "request_id": reqid}
    if modality == "text":
        body["text"] = TEXT_SAMPLE
    else:
        body["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
    if model == MODELS[0] and QWEN_INSTRUCTION_MODE == "custom":
        body["instruction"] = QWEN_CUSTOM_INSTRUCTION
    return body

def checked_metadata(data, model):
    meta = data.get("embedding_metadata")
    if not isinstance(meta, dict):
        raise RuntimeError("Missing embedding_metadata / Thiếu embedding_metadata")
    pre, ins, config_hash = (meta.get(k) for k in
                             ("preprocessing_version", "instruction_sha256", "config_sha256"))
    def sha(v):
        return isinstance(v, str) and len(v) == 64 and all(x in "0123456789abcdef" for x in v)
    effective = (QWEN_CUSTOM_INSTRUCTION if QWEN_INSTRUCTION_MODE == "custom"
                 else QWEN_LEGACY_INSTRUCTION) if model == MODELS[0] else None
    expected_ins = hashlib.sha256(effective.encode("utf-8")).hexdigest() if effective else None
    if not pre or not isinstance(pre, str) or ins != expected_ins or not sha(config_hash):
        raise RuntimeError("Invalid instruction or configuration metadata / Metadata instruction hoặc cấu hình không hợp lệ")
    canonical = {"dimension": DIMENSION, "instruction_sha256": expected_ins,
                 "model": model, "preprocessing_version": pre,
                 "revision": data.get("revision")}
    expected_config = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if config_hash != expected_config:
        raise RuntimeError("Configuration SHA-256 mismatch / SHA-256 của cấu hình embedding không khớp")
    return dict(meta)

def register_provenance(model, revision, metadata):
    """Require stable checkpoint and encoding configuration across every measurement."""
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("Missing model checkpoint revision / Thiếu revision của checkpoint model")
    if model in revisions and revisions[model] != revision:
        raise RuntimeError("Model revision changed during benchmark / Revision model thay đổi trong khi benchmark")
    if model in embedding_configs and embedding_configs[model] != metadata:
        raise RuntimeError("Embedding configuration changed during benchmark / Cấu hình embedding thay đổi trong khi benchmark")
    revisions[model] = revision
    embedding_configs[model] = dict(metadata)

def checked_gpu_timing(data, expected_batch_size=1):
    """Validate each worker's GPU instrumentation for all benchmark phases."""
    timing = data.get("timing")
    if not isinstance(timing, dict):
        raise RuntimeError("Instrumented GPU timing required / Thiếu dữ liệu đo thời gian GPU bắt buộc")
    state, first = timing.get("model_state"), timing.get("gpu_worker_first_request")
    worker_id = timing.get("gpu_worker_id")
    if worker_id is not None and (not isinstance(worker_id, str)
                                  or len(worker_id) != 32
                                  or any(ch not in "0123456789abcdef" for ch in worker_id)):
        raise RuntimeError("Invalid opaque GPU worker identifier / Mã định danh GPU worker không hợp lệ")
    if (state not in {"initial_load", "model_reload", "warm"}
        or type(first) is not bool or type(timing.get("batch_size")) is not int
        or type(expected_batch_size) is not int or expected_batch_size < 1
        or timing["batch_size"] != expected_batch_size):
        raise RuntimeError("Invalid GPU state, first-request flag or batch size / Trạng thái GPU, cờ request đầu hoặc kích thước batch không hợp lệ")
    def ms(value, name, optional=False):
        if optional and value is None:
            return None
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RuntimeError("Invalid timing field / Trường thời gian không hợp lệ: " + name)
        return round(value, 3)
    return {
        "model_state": state, "gpu_worker_first_request": first,
        "gpu_worker_id": worker_id,
        "api_elapsed_ms": ms(data.get("elapsed_ms"), "elapsed_ms"),
        "gpu_worker_ms": ms(timing.get("gpu_worker_ms"), "gpu_worker_ms"),
        "model_load_ms": ms(timing.get("model_load_ms"), "model_load_ms"),
        "gpu_encode_ms": ms(timing.get("gpu_encode_ms"), "gpu_encode_ms", True),
        "gpu_dispatch_overhead_ms": ms(timing.get("gpu_dispatch_overhead_ms"),
                                       "gpu_dispatch_overhead_ms", True),
    }

# -- One validated sample; no automatic POST retry --
def _sample_checked(model, modality, phase, round_number):
    if model not in MODELS or modality not in ("text", "image"):
        raise ValueError("Invalid model/modality / Model hoặc loại đầu vào không hợp lệ")
    reqid = "benchmark-%06d-%s" % (sequence, hashes[modality][:12])
    body = request_body(model, modality, reqid)
    start = time.perf_counter()
    data = http_json("/v1/embeddings", body)
    client_ms = (time.perf_counter() - start) * 1000
    if (data.get("model") != model or data.get("backend") != "modal"
        or data.get("request_id") != reqid
        or data.get("input_sha256") != hashes[modality]):
        raise RuntimeError("Model/backend/request ID/input digest mismatch / Model, backend, request ID hoặc mã băm đầu vào không khớp")
    metadata = checked_metadata(data, model)
    register_provenance(model, data.get("revision"), metadata)
    declared_dimension = data.get("dimension")
    if type(declared_dimension) is not int or declared_dimension != DIMENSION:
        raise RuntimeError("Incorrect declared embedding dimension / Số chiều embedding khai báo không chính xác")
    vector = data.get("vector")
    if not isinstance(vector, list) or len(vector) != DIMENSION:
        raise RuntimeError("Incorrect embedding dimension / Số chiều vector embedding không chính xác")
    if not all(type(v) in (float, int) and math.isfinite(v) for v in vector):
        raise RuntimeError("Non-finite embedding / Vector embedding chứa giá trị không hữu hạn")
    if abs(math.sqrt(sum(v*v for v in vector)) - 1.0) >= 0.001:
        raise RuntimeError("Embedding not L2-normalized / Vector embedding chưa được chuẩn hóa L2")
    instrument = checked_gpu_timing(data)
    return {
        "phase": phase, "model": model, "modality": modality, "round": round_number,
        "embedding_metadata": metadata, "client_elapsed_ms": round(client_ms, 3),
        **instrument,
    }

def save_sequential_progress(phase):
    phase = {"warmup": "warm", "alternate_setup": "alternate"}.get(phase, phase)
    write_benchmark_report({
        "status": "IN_PROGRESS", "phase": phase, "created_utc": utc_now(),
        "provenance": provenance, "config": benchmark_config,
        "retries": 0, "summary": {}, "trials": rows,
        "started_phases": sorted(run_started_phases),
        "concurrent": {"summary": concurrent_summary, "trials": concurrent_trials},
        "mixed_model_batches": batch_trials,
        "limitations": [
            "An in-progress attempt does not prove server acceptance or completion.",
            "No request bodies, credentials or embedding vectors are saved.",
        ],
    })


def sample(model, modality, phase, round_number, context=None):
    """Persist each attempt before dispatch and update it without duplicate rows."""
    global sequence
    context = dict(context or {})  # Only benchmark-generated timing metadata.
    started_at_utc, started_monotonic = utc_now(), time.perf_counter()
    sequence += 1
    attempt = {
        "sequence": sequence, "phase": phase, "model": model, "modality": modality,
        "round": round_number, "status": "in_progress", "confirmed": False,
        "started_at_utc": started_at_utc,
    }
    attempt.update(context)
    rows.append(attempt)
    try:
        save_sequential_progress(phase)
        result = _sample_checked(model, modality, phase, round_number)
        # Keep the successful-return bookkeeping inside the interruption handler.
        attempt.update(result)
        attempt.update(context)
        attempt.update(finished_at_utc=utc_now(), status="completed", confirmed=False)
        save_sequential_progress(phase)
    except KeyboardInterrupt:
        attempt.update(
            status="interrupted", confirmed=False, error_type="KeyboardInterrupt",
            finished_at_utc=utc_now(),
            client_elapsed_ms=round((time.perf_counter()-started_monotonic)*1000, 3),
        )
        if phase == "cold":
            attempt["cold_classification"] = "interrupted"
        save_failed_report(phase, "KeyboardInterrupt", interrupted=True)
        raise
    except Exception as error:
        attempt.update(
            status="failed", confirmed=False, error_type=type(error).__name__,
            finished_at_utc=utc_now(),
            client_elapsed_ms=round((time.perf_counter()-started_monotonic)*1000, 3),
        )
        if phase == "cold":
            attempt["cold_classification"] = "failed"
        save_failed_report(phase, type(error).__name__)
        raise
    return attempt

# %% NOTEBOOK CELL cold-run
#@title Cold-start trials / Đo cold-start { display-mode: "form" }
if RUN_BENCHMARK and RUN_COLD:
    assert_run_active("cold")
    try:
        for index in range(COLD_REPETITIONS):
            order = (COLD_ACTIVE_MODELS if index % 2 == 0
                     else tuple(reversed(COLD_ACTIVE_MODELS)))
            for model in order:
                print("Cold %d/%d %s: idle %ds, no probes / không probe" %
                      (index+1, COLD_REPETITIONS, model, COLD_IDLE_WAIT_SECONDS))
                idle_started_at_utc = utc_now()
                idle_started_monotonic = time.perf_counter()
                time.sleep(COLD_IDLE_WAIT_SECONDS)
                idle_finished_at_utc = utc_now()
                idle_elapsed_seconds = round(time.perf_counter()-idle_started_monotonic,3)
                result = sample(model, "image", "cold", index+1, context={
                    "idle_started_at_utc": idle_started_at_utc,
                    "idle_finished_at_utc": idle_finished_at_utc,
                    "idle_elapsed_seconds": idle_elapsed_seconds,
                })
                result["confirmed"] = (result["gpu_worker_first_request"] is True
                                       and result["model_state"] == "initial_load")
                result["cold_classification"] = (
                    "confirmed_initial_load" if result["confirmed"]
                    else result["model_state"]
                )
                print("GPU COLD CONFIRMED" if result["confirmed"] else "GPU COLD UNCONFIRMED",
                      "state:", result["model_state"], "worker:",
                      result["gpu_worker_id"] or "unavailable")
    except KeyboardInterrupt:
        interrupt_benchmark("cold")
        raise
if RUN_BENCHMARK and RUN_COLD:
    run_completed_phases.add("cold")
# %% NOTEBOOK CELL warm-run
#@title Warm-up and warm latency / Làm nóng và đo độ trễ warm { display-mode: "form" }
import random
if RUN_BENCHMARK and RUN_WARM:
    assert_run_active("warm")
    try:
        for block in range(2):
            order = MODELS if block == 0 else tuple(reversed(MODELS))
            modalities = ["text"] * (WARM_REPETITIONS // 2) + ["image"] * (WARM_REPETITIONS // 2)
            random.Random(20261107 + block).shuffle(modalities)
            for model in order:
                for modality in ("text", "image"):
                    for _ in range(WARMUP_REQUESTS):
                        sample(model, modality, "warmup", block+1)
                for modality in modalities:
                    result = sample(model, modality, "warm", block+1)
                    result["confirmed"] = (result["model_state"] == "warm"
                                           and result["gpu_worker_first_request"] is False)
                print("Warm block done / Hoàn tất warm:", block+1, model)
    except KeyboardInterrupt:
        interrupt_benchmark("warm")
        raise
if RUN_BENCHMARK and RUN_WARM:
    run_completed_phases.add("warm")
# %% NOTEBOOK CELL alternate-run
#@title Alternating model routes / Đo luân phiên hai model { display-mode: "form" }
if RUN_BENCHMARK and RUN_ALTERNATE:
    assert_run_active("alternate")
    try:
        alternate_workers = {}
        for model_id in MODELS:
            setup = sample(model_id, "text", "alternate_setup", 0)
            alternate_workers[model_id] = setup["gpu_worker_id"]
            if not alternate_workers[model_id]:
                stop_benchmark("alternate_setup", "MissingGPUWorkerID", setup)
        if alternate_workers[MODELS[0]] == alternate_workers[MODELS[1]]:
            stop_benchmark("alternate_setup", "SharedGPUWorkerID", setup)
        for index in range(ALTERNATE_REPETITIONS):
            for target in (MODELS[1], MODELS[0]):
                result = sample(target, "text", "alternate", index + 1)
                result["confirmed"] = (
                    result["model_state"] == "warm"
                    and result["gpu_worker_first_request"] is False
                    and result["gpu_worker_id"] == alternate_workers[target]
                    and alternate_workers[MODELS[0]] != alternate_workers[MODELS[1]]
                )
                if not result["confirmed"]:
                    print("UNCONFIRMED alternating route / Chưa xác nhận:", target, index + 1)
        print("Dedicated GPU workers / Hai GPU worker riêng:", alternate_workers)
    except KeyboardInterrupt:
        interrupt_benchmark("alternate")
        raise
if RUN_BENCHMARK and RUN_ALTERNATE:
    run_completed_phases.add("alternate")
# %% NOTEBOOK CELL concurrent-run
#@title Concurrent request burst / Đo nhóm request đồng thời { display-mode: "form" }
# Concurrent endpoint latency: explicit opt-in; independent of cold/warm phases.
# Concurrent probe map / Sơ đồ thử nghiệm đồng thời: journal each dispatch,
# collect completed responses, and never retry unconfirmed requests.
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from uuid import uuid4
if RUN_BENCHMARK and RUN_CONCURRENT:
    assert_run_active("concurrent")
concurrent_trials, concurrent_summary = [], []
concurrent_outcomes = {}  # Redacted worker outcomes survive an interrupted submit.

def concurrent_probe(model, index):
    reqid = "parallel-" + uuid4().hex
    started = time.perf_counter()
    data = http_json("/v1/embeddings", request_body(model, "text", reqid))
    wall = round((time.perf_counter() - started) * 1000, 3)
    if (data.get("model") != model or data.get("request_id") != reqid
        or data.get("backend") != "modal" or data.get("input_sha256") != hashes["text"]
        or data.get("dimension") != DIMENSION):
        raise RuntimeError("Concurrent request identity mismatch / Định danh request đồng thời không khớp")
    metadata = checked_metadata(data, model)
    vector = data.get("vector")
    if (not isinstance(vector, list) or len(vector) != DIMENSION
        or not all(type(v) in (float, int) and math.isfinite(v) for v in vector)
        or abs(math.sqrt(sum(v*v for v in vector)) - 1.0) >= 0.001):
        raise RuntimeError("Invalid concurrent embedding / Embedding của request đồng thời không hợp lệ")
    # A completed concurrent trial must also pass GPU instrumentation checks.
    # An absent worker ID is acceptable; invalid required timing is not.
    instrument = checked_gpu_timing(data)
    trial = {"model": model, "index": index, "status": "completed",
             "gpu_instrumentation_verified": True,
             "client_elapsed_ms": wall, "model_state": instrument["model_state"],
             "gpu_worker_first_request": instrument["gpu_worker_first_request"],
             "gpu_worker_id": instrument["gpu_worker_id"],
             "config_sha256": metadata["config_sha256"]}
    # Return the small provenance envelope; do not expose the raw vector.
    return trial, data.get("revision"), metadata

def observed_concurrent(model, index):
    """Record every dispatched POST, including redacted failures."""
    began, started = utc_now(), time.perf_counter()
    try:
        trial, revision, metadata = concurrent_probe(model, index)
    except Exception as error:
        trial = {"model": model, "index": index, "status": "failed",
                 "error_type": type(error).__name__}
        revision, metadata = None, None
    trial.update(phase="concurrent", started_at_utc=began,
                 finished_at_utc=utc_now())
    trial.setdefault("client_elapsed_ms", round((time.perf_counter()-started)*1000, 3))
    outcome = (trial, revision, metadata)
    concurrent_outcomes[model, index] = outcome
    return outcome

def save_concurrent_progress():
    # A scheduled attempt does not prove server acceptance or completion.
    write_benchmark_report({
        "status": "IN_PROGRESS", "phase": "concurrent", "created_utc": utc_now(),
        "provenance": provenance, "config": benchmark_config,
        "retries": 0, "summary": {}, "trials": rows,
        "started_phases": sorted(run_started_phases),
        "concurrent": {"summary": concurrent_summary, "trials": concurrent_trials},
        "mixed_model_batches": batch_trials,
        "limitations": [
            "In-progress attempts do not prove server acceptance or completion.",
            "An interrupted phase cannot be retried in this run.",
            "Burst wall time includes notebook dispatch and journal-writing overhead.",
            "No request bodies, credentials or embedding vectors are saved.",
        ],
    })


def concurrent_marker(model, index):
    return next(t for t in concurrent_trials if t["model"] == model and t["index"] == index)


def submit_concurrent(pool, futures, model, index):
    # Persist before dispatch so even forced UI Stop leaves a redacted attempt.
    concurrent_trials.append({
        "phase": "concurrent", "model": model, "index": index,
        "status": "in_progress", "started_at_utc": utc_now(),
    })
    save_concurrent_progress()
    futures[pool.submit(observed_concurrent, model, index)] = index


def collect_concurrent(future, model, index):
    """Only the collector changes shared revision/config state and attempt markers."""
    try:
        trial, revision, metadata = (
            future.result() if future is not None else concurrent_outcomes[model, index]
        )
        if trial["status"] == "completed":
            try:
                register_provenance(model, revision, metadata)
            except Exception as error:
                trial.update(status="failed", error_type=type(error).__name__)
    except Exception as error:
        trial = {"phase": "concurrent", "model": model, "index": index,
                 "status": "failed", "error_type": type(error).__name__,
                 "finished_at_utc": utc_now()}
    concurrent_marker(model, index).update(trial)
    if run_failure is None:
        save_concurrent_progress()
    return trial.get("error_type")


if RUN_BENCHMARK and RUN_CONCURRENT:
    try:
        for model in MODELS:
            started, failure_type = time.perf_counter(), None
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as pool:
                futures, next_index = {}, 0
                try:
                    for _ in range(min(CONCURRENT_WORKERS, CONCURRENT_REQUESTS)):
                        submit_concurrent(pool, futures, model, next_index)
                        next_index += 1
                    while futures:
                        completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in sorted(completed, key=lambda f: futures[f]):
                            index = futures[future]
                            error_type = collect_concurrent(future, model, index)
                            del futures[future]
                            if error_type and not failure_type:
                                failure_type = error_type
                        if failure_type:
                            # No replacement work; preserve already-dispatched results.
                            for future, index in list(futures.items()):
                                collect_concurrent(future, model, index)
                                del futures[future]
                            break
                        for _ in completed:
                            if next_index < CONCURRENT_REQUESTS:
                                submit_concurrent(pool, futures, model, next_index)
                                next_index += 1
                except KeyboardInterrupt:
                    # Save immediately, before waiting for any in-flight HTTP request.
                    save_failed_report("concurrent", "KeyboardInterrupt", interrupted=True)
                    # Cancel every pending task before waiting on any running task.
                    for future, index in list(futures.items()):
                        if future.cancel():
                            concurrent_marker(model, index).update(
                                status="cancelled", finished_at_utc=utc_now())
                    raise  # Executor shutdown waits for already-dispatched requests.
            if failure_type:
                stop_benchmark("concurrent", failure_type)
            wall = round((time.perf_counter()-started)*1000, 3)
            done = sum(t["status"] == "completed" and t["model"] == model
                       for t in concurrent_trials)
            concurrent_summary.append({
                "model": model, "attempts": CONCURRENT_REQUESTS, "completed": done,
                "workers_requested": CONCURRENT_WORKERS, "burst_wall_ms": wall,
                "observed_burst_items_per_second": round(done*1000/wall, 3) if wall else None,
            })
        print("Concurrent burst / Nhóm request đồng thời:", concurrent_summary)
    except KeyboardInterrupt:
        # A submit may be interrupted after dispatch but before its Future is assigned.
        # Worker outcomes remain available after executor shutdown in either case.
        for trial in concurrent_trials:
            key = trial["model"], trial["index"]
            if trial["status"] == "in_progress" and key in concurrent_outcomes:
                collect_concurrent(None, *key)
        save_failed_report("concurrent", "KeyboardInterrupt", interrupted=True)
        interrupt_benchmark("concurrent")
        raise
if RUN_BENCHMARK and RUN_CONCURRENT:
    run_completed_phases.add("concurrent")
# %% NOTEBOOK CELL mixed-batch-run
#@title Mixed-model batches / Đo batch hai model { display-mode: "form" }
# Opt-in two-model batch: one HTTP POST, not a sustained throughput claim.
if RUN_BENCHMARK and RUN_MIXED_BATCH:
    assert_run_active("mixed_batch")
batch_trials = []

def save_mixed_batch_progress():
    # Persist a sanitized attempt marker before each POST, including forced UI Stop.
    write_benchmark_report({
        "status": "IN_PROGRESS", "phase": "mixed_batch", "created_utc": utc_now(),
        "provenance": provenance, "config": benchmark_config,
        "retries": 0, "summary": {}, "trials": rows,
        "started_phases": sorted(run_started_phases),
        "concurrent": {"summary": concurrent_summary, "trials": concurrent_trials},
        "mixed_model_batches": batch_trials,
        "limitations": [
            "Started attempts do not prove server acceptance or completion.",
            "An interrupted phase cannot be retried in this run.",
            "No request bodies, private image content or vectors are saved.",
        ],
    })

if RUN_BENCHMARK and RUN_MIXED_BATCH:
    try:
        for index in range(MIXED_BATCH_REPETITIONS):
            items = [request_body(model, "text", "batch-" + uuid4().hex) for model in MODELS]
            trial = {
                "index": index + 1, "phase": "mixed_batch",
                "status": "in_progress", "confirmed": False,
                "batch_size_requested": 2, "started_at_utc": utc_now(),
            }
            batch_trials.append(trial)
            save_mixed_batch_progress()
            started = time.perf_counter()
            try:
                response = http_json("/v1/embeddings/batches", {"items": items})
                wall = round((time.perf_counter() - started) * 1000, 3)
                results = response.get("items")
                if response.get("count") != 2 or not isinstance(results, list) or len(results) != 2:
                    raise RuntimeError("Mixed batch count mismatch / Số kết quả của mixed batch không khớp")
                workers, configs, observed, gpu_states = {}, {}, [], {}
                # The gateway groups requests by model and sends each group to
                # its own GPU worker; HTTP batch count is not the worker batch size.
                worker_batch_sizes = {
                    model: sum(item["model"] == model for item in items) for model in MODELS
                }
                for request, data in zip(items, results, strict=True):
                    model = request["model"]
                    if (data.get("model") != model or data.get("request_id") != request["request_id"]
                        or data.get("backend") != "modal"
                        or type(data.get("dimension")) is not int or data["dimension"] != DIMENSION
                        or data.get("input_sha256") != hashes["text"]):
                        raise RuntimeError("Mixed batch routing or order mismatch / Định tuyến hoặc thứ tự mixed batch không khớp")
                    meta = checked_metadata(data, model)
                    vector = data.get("vector")
                    if (not isinstance(vector, list) or len(vector) != DIMENSION
                        or not all(type(v) in (float, int) and math.isfinite(v) for v in vector)
                        or abs(math.sqrt(sum(v*v for v in vector)) - 1.0) >= 0.001):
                        raise RuntimeError("Invalid mixed batch embedding / Vector embedding của mixed batch không hợp lệ")
                    instrumentation = checked_gpu_timing(
                        data, expected_batch_size=worker_batch_sizes[model]
                    )
                    workers[model] = instrumentation["gpu_worker_id"]
                    gpu_states[model] = {
                        "model_state": instrumentation["model_state"],
                        "gpu_worker_first_request": instrumentation["gpu_worker_first_request"],
                        "worker_batch_size": worker_batch_sizes[model],
                    }
                    configs[model] = meta["config_sha256"]
                    observed.append((model, data.get("revision"), meta))
                if all(workers.values()) and len(set(workers.values())) != 2:
                    raise RuntimeError("Two models unexpectedly share a GPU worker ID / Hai model bất ngờ dùng cùng mã GPU worker")
                for model, revision, metadata in observed:
                    register_provenance(model, revision, metadata)
                trial.update(status="completed", confirmed=True,
                             batch_size=2, client_wall_ms=wall,
                             gpu_instrumentation_verified=True,
                             gpu_worker_ids=workers, gpu_states=gpu_states,
                             config_sha256=configs, finished_at_utc=utc_now())
                save_mixed_batch_progress()
            except KeyboardInterrupt:
                trial.update(status="interrupted", confirmed=False,
                             error_type="KeyboardInterrupt", finished_at_utc=utc_now())
                raise
            except Exception as error:
                trial.update(status="failed", confirmed=False,
                             error_type=type(error).__name__,
                             finished_at_utc=utc_now())
                stop_benchmark("mixed_batch", type(error).__name__)
        print("Mixed-model batch / Batch hai model:", len(batch_trials))
    except KeyboardInterrupt:
        interrupt_benchmark("mixed_batch")
        raise
if RUN_BENCHMARK and RUN_MIXED_BATCH:
    run_completed_phases.add("mixed_batch")
# %% NOTEBOOK CELL report
#@title Validate and export results / Kiểm tra và xuất báo cáo { display-mode: "form" }
import statistics
# Report map / Sơ đồ báo cáo: validate completed phases and trial counts,
# calculate descriptive latency summaries, and write sanitized JSON/HTML.
# Also support focused offline tests that invoke the report without optional cells.
concurrent_trials = globals().get("concurrent_trials", [])
concurrent_summary = globals().get("concurrent_summary", [])
batch_trials = globals().get("batch_trials", [])
from datetime import datetime, timezone

def percentile(xs, percentage):
    values = sorted(xs)
    if not values:
        return None
    x = (len(values)-1)*percentage/100
    a, b = math.floor(x), math.ceil(x)
    return values[a] + (values[b]-values[a])*(x-a)

def timing_stats(samples, key):
    values = [s[key] for s in samples
              if type(s.get(key)) in (int, float) and math.isfinite(s[key])]
    if not values:
        return None
    return {"n": len(values), "p50_ms": round(percentile(values, 50), 3),
            "p95_ms": round(percentile(values, 95), 3),
            "mean_ms": round(statistics.mean(values), 3)}

if not RUN_BENCHMARK and run_configuration_snapshot["run_benchmark"]:
    stop_benchmark("report", "RunConfigurationChanged")

if RUN_BENCHMARK:
    # A failed preflight has no failed trial row: never erase its JSON artifact.
    if run_failure is not None:
        raise RuntimeError(
            "Earlier benchmark failure; INCOMPLETE_FAILED report preserved / "
            "Lượt benchmark trước thất bại; đã giữ báo cáo INCOMPLETE_FAILED: "
            + run_failure["phase"] + " (" + run_failure["error_type"] + ")"
        )
    assert_run_active("report")
    active_phases = {
        name for name, enabled in (
            ("cold", RUN_COLD), ("warm", RUN_WARM),
            ("alternate", RUN_ALTERNATE), ("concurrent", RUN_CONCURRENT),
            ("mixed_batch", RUN_MIXED_BATCH),
        ) if enabled
    }
    if active_phases != run_requested_phases:
        stop_benchmark("report", "PhaseSelectionChanged")
    missing_phases = ({"preflight"} | run_requested_phases) - run_completed_phases
    if missing_phases:
        stop_benchmark("report", "MissingPhases_" + "_".join(sorted(missing_phases)))
    if (any(r.get("status") == "failed" for r in rows)
        or any(r.get("status") == "failed" for r in concurrent_trials + batch_trials)):
        stop_benchmark("report", "FailedTrialAlreadyRecorded")
    # Verify every phase count. Reused, stale or partial observations cannot
    # silently become a successful measured report.
    from collections import Counter
    observed = Counter(
        (r["phase"], r["model"], r["modality"], r["round"]) for r in rows
    )
    expected = Counter()
    if RUN_COLD:
        for model in COLD_ACTIVE_MODELS:
            for trial in range(1, COLD_REPETITIONS + 1):
                expected["cold", model, "image", trial] += 1
    if RUN_WARM:
        for model in MODELS:
            for block in (1, 2):
                for modality in ("text", "image"):
                    expected["warmup", model, modality, block] += WARMUP_REQUESTS
                    expected["warm", model, modality, block] += WARM_REPETITIONS // 2
    if RUN_ALTERNATE:
        for model in MODELS:
            expected["alternate_setup", model, "text", 0] += 1
            for trial in range(1, ALTERNATE_REPETITIONS + 1):
                expected["alternate", model, "text", trial] += 1
    if observed != expected:
        stop_benchmark("report", "TrialCountMismatch")
    if (len(concurrent_trials) != (len(MODELS) * CONCURRENT_REQUESTS
                                  if RUN_CONCURRENT else 0)
        or len(batch_trials) != (MIXED_BATCH_REPETITIONS if RUN_MIXED_BATCH else 0)
        or len(concurrent_summary) != (len(MODELS) if RUN_CONCURRENT else 0)):
        stop_benchmark("report", "OptionalTrialCountMismatch")
    summary = {}
    for phase, model, modality in sorted({
        (r["phase"], r["model"], r["modality"]) for r in rows
        if r["phase"] in {"cold", "warm", "alternate"}
    }):
        attempts = [r for r in rows
                    if (r["phase"], r["model"], r["modality"])
                    == (phase, model, modality)]
        confirmed = [r for r in attempts if r["confirmed"]]
        state_counts = {}
        for attempt in attempts:
            state = attempt.get("model_state", "failed")
            state_counts[state] = state_counts.get(state, 0) + 1
        summary[" / ".join((phase, model, modality))] = {
            "state_counts": state_counts,
            "attempts": len(attempts), "confirmed": len(confirmed),
            "unconfirmed": len(attempts)-len(confirmed),
            "client": timing_stats(confirmed, "client_elapsed_ms"),
            "api": timing_stats(confirmed, "api_elapsed_ms"),
            "model_load": timing_stats(confirmed, "model_load_ms"),
            "gpu_encode": timing_stats(confirmed, "gpu_encode_ms"),
        }
    report = {
        "status": (
            "LIVE_BENCHMARK_MEASURED" if (
                any(r["confirmed"] for r in rows)
                or any(t["status"] == "completed" for t in concurrent_trials + batch_trials))
            else "NO_CONFIRMED_STATES"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": {m: {"revision": revisions.get(m), "dimension": DIMENSION} for m in MODELS},
        "input_sha256": hashes,
        "embedding_configs": embedding_configs,
        "provenance": provenance,
        "config": benchmark_config,
        "sample_size_context": {
            "default_warm_post_requests": 24,
            "default_warmup_requests": 8,
            "default_measured_requests": 16,
            "default_warm_measurements_per_model_modality": 4,
            "client_latency_n_is_per_group": True,
            "note_en": "p95 is descriptive only; small confirmed samples do not establish production tail latency.",
            "note_vi": "p95 chỉ mô tả mẫu đã xác nhận; ít mẫu không chứng minh độ trễ đuôi production.",
        },
        "summary": summary, "trials": rows,
        "concurrent": {"summary": concurrent_summary, "trials": concurrent_trials},
        "mixed_model_batches": batch_trials,
        "limitations": [
            "GPU cold evidence is not independent proof of CPU gateway cold.",
            "Idle sleep is not proof of scale-down; other requests may interfere.",
            "Alternating warm routing requires present, distinct and stable worker IDs.",
            "Unconfirmed states are excluded from cold/warm/alternate statistics.",
            "Concurrent bursts show observed short-window items/s, not sustained production throughput.",
            "Burst wall time includes notebook dispatch and journal-writing overhead.",
            "Mixed-model batch wall time is one HTTP call, not standalone per-model latency.",
            "Retrieval quality and identity accuracy are not measured.",
        ],
    }
    output = write_benchmark_report(report)
    # Bilingual on-screen summary; the full redacted JSON remains the source of details.
    lines = [
        "| Phase / Giai đoạn | Model | Input / Đầu vào | Confirmed / Xác nhận | Latency n | p50 (ms) | p95 (ms) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, values in summary.items():
        phase, model, modality = label.split(" / ")
        client = values["client"] or {}
        p50 = f'{client["p50_ms"]:.2f}' if "p50_ms" in client else "N/A"
        p95 = f'{client["p95_ms"]:.2f}' if "p95_ms" in client else "N/A"
        lines.append(
            f'| {phase} | {model} | {modality} | '
            f'{values["confirmed"]}/{values["attempts"]} | {client.get("n", 0)} | {p50} | {p95} |'
        )
    if summary:
        sample_note = (
            "**Sample size / Số mẫu:** Latency n is the count of confirmed samples "
            "with finite client latency in each group. Default warm-only: 24 POSTs "
            "= 8 warm-ups + 16 measured (at most 4 per model/modality). "
            "p95 is descriptive, not production tail-latency evidence. / "
            "Latency n là số mẫu xác nhận có thời gian client hợp lệ trong từng nhóm. "
            "Mặc định warm: 24 POST = 8 warm-up + 16 đo (tối đa 4 mẫu mỗi model/loại). "
            "p95 chỉ mô tả mẫu, không phải bằng chứng độ trễ đuôi production."
        )
        rendered = "\n".join(lines)
        try:
            from IPython.display import Markdown, display
            display(Markdown("### Confirmed latency / Độ trễ đã xác nhận\n\n" + rendered + "\n\n" + sample_note))
        except ImportError:
            print(rendered)
            print(sample_note)
    else:
        print("No confirmed timing states / Chưa có mẫu đã xác nhận.")
    print("LIVE REPORT / BÁO CÁO THỰC TẾ")
    for label, values in summary.items():
        c = values["client"]
        print(label, "confirmed", values["confirmed"], "/", values["attempts"],
              "client p50/p95 (ms)", (c["p50_ms"], c["p95_ms"]) if c else "N/A")
    if RUN_CONCURRENT:
        print("Concurrent bursts / Nhóm đồng thời:", concurrent_summary)
    if RUN_MIXED_BATCH:
        print("Mixed-model batch attempts / Số lượt batch:", len(batch_trials))
    print("Saved / Đã lưu:", output.name)
else:
    print("No live benchmark performed / Chưa chạy benchmark thật.")
