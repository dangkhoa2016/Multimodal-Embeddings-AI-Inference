# Source for the standalone notebook code cells.
# %% NOTEBOOK CELL configure-code
#@title Model configuration / Cấu hình model { display-mode: "form" }
# English: Select one model. Colab: use the dropdown; Kaggle: edit the value.
# Tiếng Việt: Chọn một model. Colab: dùng danh sách chọn; Kaggle: sửa giá trị.
MODEL_ID = "qwen3-vl-embedding-8b" #@param ["qwen3-vl-embedding-8b", "wemm-embedding-9b"]
DIMENSION = 1024 #@param {type:"integer"}
TIMEOUT_SECONDS = 900 #@param {type:"integer"}
INSTRUCTION_MODE = "default" #@param ["default", "custom"]
CUSTOM_INSTRUCTION = "Retrieve images that match the text description." #@param {type:"string"}
RUN_API_DEMO = False #@param {type:"boolean"}
# Safe Run All: preview without credentials unless explicitly enabled.

# Invalidate previous configuration when only this editable form is re-run.
# Khi chỉ chạy lại form, cấu hình cũ không còn được phép sử dụng.
configuration_ready = False
# %% NOTEBOOK CELL configure-helpers-code
#@title Configuration helpers (run; no edits) / Hàm hỗ trợ cấu hình (chạy, không sửa) { display-mode: "form" }
# Helper map / Sơ đồ helper: validate configuration and environment overrides.
# Then manage session state, status messages, and bilingual recovery guidance.
# Internal validation, safe environment overrides, and bilingual status rendering.
# Kiểm tra hợp lệ, ưu tiên biến môi trường an toàn và hiển thị trạng thái song ngữ.
# Revoke success from any earlier run before validation or display.
# Thu hồi trạng thái hợp lệ từ lượt chạy trước ngay khi bắt đầu.
configuration_ready = False
import os
import sys
from html import escape as escape_html
from pathlib import Path

QWEN_LEGACY_INSTRUCTION = "Represent the image or text for retrieval in a visual gallery."
BUNDLED_SAMPLE_TEXTS = (
    "a flat bright-green triangle on a cream background",
    "a glossy cobalt-blue 3D sphere on a pale background",
    "a realistic orange gift box tied with a ribbon on a wooden table",
)
# Keep user-edited samples independently of model configuration.
# A rerun of this cell must not discard the previous Customize samples values.
SAMPLE_DEFAULTS = globals().get("SAMPLE_DEFAULTS", {
    "VISION_EMBEDDINGS_SAMPLE_TEXT": BUNDLED_SAMPLE_TEXTS[0],
    "VISION_EMBEDDINGS_SAMPLE_TEXT_2": BUNDLED_SAMPLE_TEXTS[1],
    "VISION_EMBEDDINGS_SAMPLE_TEXT_3": BUNDLED_SAMPLE_TEXTS[2],
    "VISION_EMBEDDINGS_IMAGE_PATH": "",
    "VISION_EMBEDDINGS_IMAGE_PATH_2": "",
    "VISION_EMBEDDINGS_IMAGE_PATH_3": "",
})
DEFAULTS = {
    "VISION_EMBEDDINGS_MAX_IMAGE_BYTES": "5242880",
    "VISION_EMBEDDINGS_REQUIRE_REAL": "true",
}

def message_pair(english, vietnamese):
    """Return complete, separately labeled English and Vietnamese messages."""
    return f"English:\n{english}\n\nTiếng Việt:\n{vietnamese}"


def validate_api_token(value):
    # Never echo the credential in exceptions, including invalid HTTP headers.
    if (not isinstance(value, str) or len(value) < 16
            or not all(33 <= ord(ch) <= 126 for ch in value)):
        raise ValueError(message_pair(
            "Bearer token must be 16+ printable ASCII characters without whitespace.",
            "Bearer token cần ít nhất 16 ký tự ASCII in được, không có khoảng trắng.",
        ))
    return value


def option(name):
    fallback = SAMPLE_DEFAULTS[name] if name in SAMPLE_DEFAULTS else DEFAULTS[name]
    return os.environ.get(name, fallback).strip()

def current_configuration_selection():
    # Local identity only; never print environment values.
    return tuple(str(os.environ.get(key, fallback)).strip() for key, fallback in (
        ("VISION_EMBEDDINGS_MODEL", MODEL_ID),
        ("VISION_EMBEDDINGS_DIMENSION", DIMENSION),
        ("VISION_EMBEDDINGS_REQUEST_TIMEOUT", TIMEOUT_SECONDS),
        ("VISION_EMBEDDINGS_INSTRUCTION_MODE", INSTRUCTION_MODE),
        ("VISION_EMBEDDINGS_CUSTOM_INSTRUCTION", CUSTOM_INSTRUCTION),
        ("VISION_EMBEDDINGS_MAX_IMAGE_BYTES", DEFAULTS["VISION_EMBEDDINGS_MAX_IMAGE_BYTES"]),
        ("VISION_EMBEDDINGS_REQUIRE_REAL", DEFAULTS["VISION_EMBEDDINGS_REQUIRE_REAL"]),
        ("RUN_API_DEMO", RUN_API_DEMO),
    ))


def require_current_configuration():
    if (globals().get("configuration_ready") is not True
            or current_configuration_selection()
            != globals().get("configuration_selection_snapshot")):
        globals()["configuration_ready"] = False
        raise RuntimeError(message_pair(
            "Configuration changed. Rerun Configuration helpers before continuing.",
            "Cấu hình đã đổi. Hãy chạy lại Configuration helpers trước khi tiếp tục.",
        ))


def positive_int(value, name, minimum, maximum):
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(message_pair(
            f"{name} must be between {minimum} and {maximum}",
            f"{name} phải từ {minimum} đến {maximum}",
        ))
    return result

def require_real():
    value = option("VISION_EMBEDDINGS_REQUIRE_REAL").lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(message_pair(
            "VISION_EMBEDDINGS_REQUIRE_REAL must be true or false",
            "VISION_EMBEDDINGS_REQUIRE_REAL phải là true hoặc false",
        ))
    return value in {"true", "1", "yes"}

_next_model_id = os.environ.get("VISION_EMBEDDINGS_MODEL", MODEL_ID).strip()
if _next_model_id not in {"qwen3-vl-embedding-8b", "wemm-embedding-9b"}:
    raise ValueError(message_pair(
        "Select qwen3-vl-embedding-8b or wemm-embedding-9b",
        "Chọn qwen3-vl-embedding-8b hoặc wemm-embedding-9b",
    ))

_next_dimension = positive_int(os.environ.get("VISION_EMBEDDINGS_DIMENSION", str(DIMENSION)),
                         "VISION_EMBEDDINGS_DIMENSION", 64, 4096)
if _next_model_id == "wemm-embedding-9b" and _next_dimension not in {64, 128, 256, 512, 1024, 2048, 4096}:
    raise ValueError(message_pair(
        "WeMM supports 64, 128, 256, 512, 1024, 2048, or 4096",
        "WeMM hỗ trợ 64, 128, 256, 512, 1024, 2048 hoặc 4096",
    ))
_next_instruction_mode = os.environ.get("VISION_EMBEDDINGS_INSTRUCTION_MODE", INSTRUCTION_MODE).strip().lower()
if _next_instruction_mode not in {"default", "custom"}:
    raise ValueError(message_pair("Choose default or custom instruction mode.",
                                  "Chọn chế độ instruction default hoặc custom."))
_next_custom_instruction = os.environ.get("VISION_EMBEDDINGS_CUSTOM_INSTRUCTION", CUSTOM_INSTRUCTION).strip()
if _next_instruction_mode == "custom":
    if _next_model_id != "qwen3-vl-embedding-8b":
        raise ValueError(message_pair(
            "Only Qwen3-VL-Embedding-8B supports custom instruction; use default for WeMM.",
            "Chỉ Qwen3-VL-Embedding-8B hỗ trợ custom instruction; chọn default cho WeMM."))
    if not 1 <= len(_next_custom_instruction) <= 512 or any(ord(ch) < 32 or ord(ch) == 127 for ch in _next_custom_instruction):
        raise ValueError(message_pair("Instruction must be one line of 1–512 characters without controls.",
                                      "Instruction phải gồm 1–512 ký tự trên một dòng, không có ký tự điều khiển."))
_next_timeout = positive_int(os.environ.get("VISION_EMBEDDINGS_REQUEST_TIMEOUT", str(TIMEOUT_SECONDS)),
                       "VISION_EMBEDDINGS_REQUEST_TIMEOUT", 1, 3600)
_next_image_limit = positive_int(option("VISION_EMBEDDINGS_MAX_IMAGE_BYTES"),
                           "VISION_EMBEDDINGS_MAX_IMAGE_BYTES", 1, 5242880)

# Publish the complete validated configuration, never a partially updated one.
(model_id, dimension, instruction_mode, custom_instruction, timeout, image_limit) = (
    _next_model_id, _next_dimension, _next_instruction_mode,
    _next_custom_instruction, _next_timeout, _next_image_limit,
)

def stage_markup(number, title_en, body_en, title_vi, body_vi, working=False):
    """Accessible, self-contained card: English first, Vietnamese underneath."""
    def clean(value):
        return escape_html(str(value), quote=True)

    accent = "#2563eb" if working else "#15803d"
    prefix = "IN PROGRESS" if working else "READY"
    return (
        '<div role="status" aria-live="polite" '
        'style="box-sizing:border-box;width:100%;max-width:960px;'
        'background:#ffffff !important;color:#15283b !important;'
        'border:1px solid #cbd5e1;border-radius:14px;padding:18px 22px;'
        'font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;'
        'font-size:16px;line-height:1.65;">'
        f'<div style="color:{accent} !important;font-weight:750;'
        'font-size:13px;letter-spacing:0.04em;">'
        f'{prefix} · {clean(number)}</div>'
        f'<div lang="en"><div style="font-size:19px;font-weight:750;'
        'color:#10253b !important;margin:5px 0 4px;">'
        f'{clean(title_en)}</div>'
        '<div style="font-size:15px;color:#334155 !important;line-height:1.65;">'
        f'{clean(body_en)}</div></div>'
        '<div style="border-top:1px solid #dbe4ed;margin:14px 0;"></div>'
        f'<div lang="vi"><div style="font-size:19px;font-weight:750;'
        'color:#10253b !important;margin:5px 0 4px;">'
        f'{clean(title_vi)}</div>'
        '<div style="font-size:15px;color:#334155 !important;line-height:1.65;">'
        f'{clean(body_vi)}</div></div>'
        '</div>'
    )


def show_stage(number, title_en, body_en, title_vi, body_vi, working=False, handle=None):
    """Use a rich notebook panel; keep a readable plain-Python fallback."""
    view = stage_markup(number, title_en, body_en, title_vi, body_vi, working)
    try:
        from IPython.display import HTML, display
    except ImportError:
        print(f"English — {title_en}: {body_en}")
        print(f"Tiếng Việt — {title_vi}: {body_vi}")
        return None
    if handle is not None:
        handle.update(HTML(view))
        return None
    if working:
        from IPython.display import DisplayHandle
        progress = DisplayHandle()
        progress.display(HTML(view))
        return progress
    display(HTML(view))
    return None


# Show only effective configuration sources, never secret environment values.
model_source = ("environment override: VISION_EMBEDDINGS_MODEL"
                if "VISION_EMBEDDINGS_MODEL" in os.environ else "form: MODEL_ID")
dimension_source = ("environment override: VISION_EMBEDDINGS_DIMENSION"
                    if "VISION_EMBEDDINGS_DIMENSION" in os.environ else "form: DIMENSION")
model_source_vi = ("biến môi trường: VISION_EMBEDDINGS_MODEL"
                   if "VISION_EMBEDDINGS_MODEL" in os.environ else "biểu mẫu: MODEL_ID")
dimension_source_vi = ("biến môi trường: VISION_EMBEDDINGS_DIMENSION"
                       if "VISION_EMBEDDINGS_DIMENSION" in os.environ else "biểu mẫu: DIMENSION")

platform = ("Kaggle" if Path("/kaggle/working").exists()
            else "Colab" if "google.colab" in sys.modules or Path("/content").exists()
            else "Jupyter")
show_stage(
    "1 / 5",
    "Model configured",
    f"Using {model_id} on {platform}. Each input will become a {dimension:,}-number vector. "
    f"Effective sources — model: {model_source}; dimension: {dimension_source}.",
    "Đã chọn model",
    f"Đang dùng {model_id} trên {platform}. Mỗi đầu vào sẽ được chuyển thành "
    f"một vector gồm {dimension:,} giá trị số. "
    f"Nguồn cấu hình có hiệu lực — model: {model_source_vi}; số chiều: {dimension_source_vi}.",
)

# Only a fully validated and displayed configuration is ready for use.
configuration_selection_snapshot = current_configuration_selection()
configuration_ready = True
# %% NOTEBOOK CELL recovery-code
#@title What should I run next? / Nên chạy bước nào tiếp theo? { display-mode: "form" }
# Read-only recovery guidance: no prompts, HTTP calls or automatic POST retries.
# Chỉ hướng dẫn phục hồi: không nhập secret, gọi HTTP hay tự gửi lại POST.
def next_notebook_step():
    state = globals()
    if state.get("configuration_ready") is not True:
        return "Model configuration → Configuration helpers"
    try:
        if current_configuration_selection() != configuration_selection_snapshot:
            return "Model configuration → Configuration helpers"
    except Exception:
        return "Model configuration → Configuration helpers"
    if not state.get("sample_prepared") or not state.get("sample_selection_snapshot"):
        return "Customize samples → Prepare input samples"
    try:
        if current_sample_selection() != sample_selection_snapshot:
            return "Customize samples → Prepare input samples"
    except Exception:
        return "Customize samples → Prepare input samples"
    import os
    pending_url = os.environ.get("VISION_EMBEDDINGS_BASE_URL", "").strip().rstrip("/")
    pending_token = os.environ.get("VISION_EMBEDDINGS_API_TOKEN", "").strip()
    if ((pending_url and pending_url != state.get("base_url"))
            or (pending_token and pending_token != state.get("api_token"))):
        return "Secure API connection → Check API connection"
    if not state.get("connection_ready"):
        return "Secure API connection → Check API connection"
    try:
        if (state.get("verified_model_id") != model_id
                or state.get("verified_connection_identity") != current_connection_identity()):
            return "Check API connection"
    except Exception:
        return "Secure API connection → Check API connection"
    run = state.get("completed_run_config")
    if (not isinstance(run, dict)
            or run.get("connection_identity") != state.get("verified_connection_identity")):
        return "Generate embeddings → Results"
    import hashlib
    active_hash = (hashlib.sha256(custom_instruction.encode("utf-8")).hexdigest()
                   if instruction_mode == "custom" else None)
    inputs = {
        "text": tuple(hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
                      for text in text_samples),
        "image": tuple(hashlib.sha256(image).hexdigest() for image in image_samples),
    }
    if (run.get("model") != model_id or run.get("dimension") != dimension
            or run.get("instruction_mode") != instruction_mode
            or run.get("custom_instruction_sha256") != active_hash
            or run.get("input_sha256") != inputs
            or run.get("sample_labels") != tuple(sample_labels)
            or run.get("text_sources") != tuple(text_sources)
            or run.get("image_sources") != tuple(image_sources)):
        return "Generate embeddings → Results"
    return "Results"

_next_step = next_notebook_step()
recovery_message = (
    "English: " + _next_step + ". No requests sent. After an interrupted POST, "
    "check the deployment before deliberately starting a new inference run.\n\n"
    "Tiếng Việt: " + _next_step + ". Chưa gửi request. Sau POST bị ngắt, "
    "kiểm tra deployment trước khi chủ động bắt đầu lượt inference mới."
)
print(recovery_message)
# %% NOTEBOOK CELL sample-options-code
#@title Customize samples / Tùy chỉnh dữ liệu mẫu { display-mode: "form" }
# English: Existing environment variables override these values.
# Tiếng Việt: Biến môi trường đã đặt sẽ được ưu tiên hơn các giá trị này.
SAMPLE_TEXT_1 = "a flat bright-green triangle on a cream background" #@param {type:"string"}
SAMPLE_TEXT_2 = "a glossy cobalt-blue 3D sphere on a pale background" #@param {type:"string"}
SAMPLE_TEXT_3 = "a realistic orange gift box tied with a ribbon on a wooden table" #@param {type:"string"}
IMAGE_PATH_1 = "" #@param {type:"string"}
IMAGE_PATH_2 = "" #@param {type:"string"}
IMAGE_PATH_3 = "" #@param {type:"string"}
PREVIEW_CUSTOM_IMAGES = False #@param {type:"boolean"}

sample_prepared = False  # New selection requires successful preparation.
SAMPLE_DEFAULTS.update({
    "VISION_EMBEDDINGS_SAMPLE_TEXT": SAMPLE_TEXT_1,
    "VISION_EMBEDDINGS_SAMPLE_TEXT_2": SAMPLE_TEXT_2,
    "VISION_EMBEDDINGS_SAMPLE_TEXT_3": SAMPLE_TEXT_3,
    "VISION_EMBEDDINGS_IMAGE_PATH": IMAGE_PATH_1,
    "VISION_EMBEDDINGS_IMAGE_PATH_2": IMAGE_PATH_2,
    "VISION_EMBEDDINGS_IMAGE_PATH_3": IMAGE_PATH_3,
})
# %% NOTEBOOK CELL sample-preparation-code
#@title Prepare input samples (run; no edits) / Chuẩn bị dữ liệu mẫu (chạy, không sửa) { display-mode: "form" }
# Helper map / Sơ đồ helper: decode built-in samples, validate optional images,
# calculate input digests, and render privacy-safe previews.
# Helper cell: image Base64 and image validation. Edit inputs only in Customize samples.
# Cell trợ giúp: ảnh Base64 và kiểm tra ảnh. Chỉ sửa đầu vào ở Customize samples.
sample_prepared = False
sample_selection_snapshot = None
import base64
import hashlib
import math


EMBEDDED_WEBP_BASE64 = (
    (
        "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"
        "AAAAAAAAAAA="
    ),
    (
        "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"
    ),
    (
        "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"
    ),
)


SYNTHETIC_IMAGES = tuple(base64.b64decode(x) for x in EMBEDDED_WEBP_BASE64)


def builtin_preview_html():
    """Show bundled reference images without depending on Markdown attachments."""
    names = (
        ("2D green triangle", "Tam giác xanh 2D"),
        ("3D blue sphere", "Hình cầu xanh 3D"),
        ("Realistic orange gift box", "Hộp quà màu cam"),
    )
    cards = []
    for encoded, (english, vietnamese) in zip(EMBEDDED_WEBP_BASE64, names, strict=True):
        cards.append(
            '<figure style="box-sizing:border-box;flex:1 1 210px;min-width:0;'
            'margin:0;border:1px solid #dbe4ed;border-radius:12px;'
            'padding:12px;background:#fff;color:#15283b;">'
            f'<img src="data:image/webp;base64,{encoded}" '
            f'alt="{escape_html(english + " / " + vietnamese, quote=True)}" '
            'style="width:100%;height:210px;object-fit:contain;" />'
            '<figcaption style="margin-top:8px;font:14px/1.45 system-ui,Arial,sans-serif;'
            'color:#15283b;">'
            f'<span lang="en" style="display:block">{escape_html(english)}</span>'
            f'<span lang="vi" style="display:block">{escape_html(vietnamese)}</span>'
            '</figcaption></figure>'
        )
    return (
        '<p role="note" style="max-width:960px;padding:10px 12px;'
        'border:1px solid #b6c6d6;border-radius:8px;'
        'background:#f5f8fc;color:#15283b;font:14px/1.5 system-ui,Arial,sans-serif;">'
        '<div lang="en"><strong>Built-in samples only.</strong> '
        'If you select custom images, those are submitted to the API but not previewed here.</div>'
        '<div lang="vi"><strong>Chỉ ảnh mẫu nhúng sẵn.</strong> '
        'Nếu chọn ảnh riêng, ảnh đó được gửi tới API nhưng không hiển thị ở đây.</div>'
        '</p>'
        '<div role="group" aria-label="Built-in image previews" '
        'style="display:flex;flex-wrap:wrap;gap:12px;max-width:960px;">'
        + "".join(cards) + '</div>'
    )


# Preview is independent of private input validation: a bad path cannot hide it.
# Xem trước ảnh nhúng không phụ thuộc vào kiểm tra đường dẫn ảnh riêng.
try:
    from IPython.display import HTML, display
except ImportError:
    print("English: Bundled previews require IPython.")
    print("Tiếng Việt: Cần IPython để xem ảnh nhúng.")
else:
    display(HTML(builtin_preview_html()))


TEXT_KEYS = ("VISION_EMBEDDINGS_SAMPLE_TEXT", "VISION_EMBEDDINGS_SAMPLE_TEXT_2",
             "VISION_EMBEDDINGS_SAMPLE_TEXT_3")
IMAGE_KEYS = ("VISION_EMBEDDINGS_IMAGE_PATH", "VISION_EMBEDDINGS_IMAGE_PATH_2",
              "VISION_EMBEDDINGS_IMAGE_PATH_3")

SAMPLE_NAMES = ("2D green triangle", "3D glossy blue sphere", "Realistic orange gift box")
LABEL_KEYS = ("VISION_EMBEDDINGS_SAMPLE_LABEL", "VISION_EMBEDDINGS_SAMPLE_LABEL_2",
              "VISION_EMBEDDINGS_SAMPLE_LABEL_3")
def current_sample_selection():
    selected = tuple(option(key) for key in TEXT_KEYS + IMAGE_KEYS)
    labels = tuple(os.environ.get(key, "").strip() for key in LABEL_KEYS)
    digests = []
    for path in selected[len(TEXT_KEYS):]:
        if not path:
            digests.append(None)
            continue
        try:
            digests.append(hashlib.sha256(
                Path(path).expanduser().read_bytes()
            ).hexdigest())
        except (OSError, ValueError):
            raise RuntimeError(message_pair(
                "Custom image is unreadable; prepare samples again.",
                "Ảnh riêng không đọc được; hãy chuẩn bị lại dữ liệu mẫu.",
            )) from None
    return selected, labels, tuple(digests), image_limit, dimension

_prepared_text_samples = [option(key) for key in TEXT_KEYS]
_prepared_text_sources = [
    "custom" if key in os.environ or SAMPLE_DEFAULTS[key] != BUNDLED_SAMPLE_TEXTS[i]
    else "built-in" for i, key in enumerate(TEXT_KEYS)
]
_prepared_image_samples = []
_prepared_image_paths = [option(key) for key in IMAGE_KEYS]
_prepared_image_sources = ["custom" if path else "built-in"
                           for path in _prepared_image_paths]
_prepared_sample_labels = []
for index, key in enumerate(LABEL_KEYS, 1):
    label = os.environ.get(key, "").strip()
    if len(label) > 60 or any(ord(ch) < 32 or ord(ch) == 127 for ch in label):
        raise ValueError(message_pair(
            f"Sample label {index} must be 0–60 characters without controls.",
            f"Nhãn mẫu {index} phải dài 0–60 ký tự và không có ký tự điều khiển.",
        ))
    _prepared_sample_labels.append(label)
_prepared_default_pairs = all(x == "built-in" for x in _prepared_text_sources + _prepared_image_sources)

for index, (key, path) in enumerate(zip(IMAGE_KEYS, _prepared_image_paths, strict=True)):
    try:
        image = Path(path).expanduser().read_bytes() if path else SYNTHETIC_IMAGES[index]
    except OSError:
        raise ValueError(message_pair(
            f"Image {index + 1}: cannot read the custom file. Check that it exists "
            "and is readable inside this Colab/Kaggle runtime.",
            f"Ảnh {index + 1}: không đọc được file tùy chỉnh. Hãy kiểm tra file "
            "tồn tại và có quyền đọc trong runtime Colab/Kaggle.",
        )) from None
    if not image or len(image) > image_limit:
        raise ValueError(message_pair(
            f"{key}: image is empty or exceeds the configured size limit",
            f"{key}: ảnh trống hoặc vượt quá giới hạn dung lượng",
        ))
    is_png = image.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = image.startswith(b"\xff\xd8\xff")
    is_webp = image[:4] == b"RIFF" and image[8:12] == b"WEBP"
    if not (is_png or is_jpeg or is_webp):
        raise ValueError(message_pair(
            f"{key}: image must be PNG, JPEG or WebP",
            f"{key}: ảnh phải là PNG, JPEG hoặc WebP",
        ))
    _prepared_image_samples.append(image)

for index, text in enumerate(_prepared_text_samples, 1):
    if not text or len(text) > 8192:
        raise ValueError(message_pair(
            f"Text sample {index} must contain 1–8192 characters",
            f"Văn bản mẫu {index} phải có 1–8192 ký tự",
        ))

# Publish all six samples atomically, only once validation is complete.
_selection_snapshot = current_sample_selection()
text_samples = _prepared_text_samples
text_sources = _prepared_text_sources
image_samples = _prepared_image_samples
image_paths = _prepared_image_paths
image_sources = _prepared_image_sources
sample_labels = _prepared_sample_labels
default_pairs = _prepared_default_pairs
sample_selection_snapshot = _selection_snapshot
sample_prepared = True

# Print variable names only; never expose overridden text, labels or image paths.
_sample_override_keys = tuple(
    key for key in TEXT_KEYS + IMAGE_KEYS + LABEL_KEYS if key in os.environ
)
_sample_source_en = (
    "Environment overrides: " + ", ".join(_sample_override_keys) + ". "
    "Values (including private paths and labels) are hidden."
    if _sample_override_keys else "No sample environment overrides."
)
_sample_source_vi = (
    "Biến môi trường ghi đè: " + ", ".join(_sample_override_keys) + ". "
    "Không hiển thị giá trị (kể cả đường dẫn và nhãn riêng)."
    if _sample_override_keys else "Không có biến môi trường ghi đè dữ liệu mẫu."
)

show_stage(
    "2 / 5",
    "Six demo inputs are ready",
    "Prepared three text sentences and three separate images for one selected model. "
    + _sample_source_en,
    "Đã chuẩn bị sáu đầu vào",
    "Đã chuẩn bị ba câu văn bản và ba ảnh riêng biệt cho một model đã chọn. "
    + _sample_source_vi,
)
# The gallery above always shows bundled references. Identify the actual
# inference inputs without automatically disclosing private image pixels/paths.
_custom_slots = [str(i + 1) for i, source in enumerate(image_sources)
                 if source == "custom"]
if _custom_slots:
    _slots = ", ".join(_custom_slots)
    print(message_pair(
        "INPUT WARNING: Image slot(s) " + _slots
        + " use CUSTOM files for inference. The gallery above shows BUNDLED "
        "references, NOT these files. Custom pixels are hidden unless PREVIEW_CUSTOM_IMAGES=True; paths stay hidden.",
        "CẢNH BÁO ĐẦU VÀO: Vị trí ảnh " + _slots
        + " dùng file RIÊNG cho inference. Gallery phía trên là ảnh mẫu "
        "NHÚNG SẴN, KHÔNG phải ảnh riêng đã chọn. Ảnh riêng chỉ hiển thị khi PREVIEW_CUSTOM_IMAGES=True; đường dẫn luôn ẩn.",
    ))

if globals().get("PREVIEW_CUSTOM_IMAGES", False) and _custom_slots:
    # Explicit consent permits pixels in outputs; never include private paths/labels.
    # Chủ động bật để hiển thị ảnh trong output; không kèm đường dẫn/nhãn riêng.
    try:
        from IPython.display import HTML, display
    except ImportError:
        print("English: Custom previews require IPython.")
        print("Tiếng Việt: Cần IPython để xem ảnh riêng.")
    else:
        _custom_cards = []
        for _i, (_image, _source) in enumerate(zip(image_samples, image_sources, strict=True), 1):
            if _source != "custom":
                continue
            _mime = ("image/png" if _image.startswith(b"\x89PNG")
                     else "image/jpeg" if _image.startswith(b"\xff\xd8") else "image/webp")
            _encoded = base64.b64encode(_image).decode("ascii")
            _custom_cards.append(
                '<figure><figcaption><span lang="en" style="display:block">Custom image ' + str(_i)
                + '</span><span lang="vi" style="display:block">Ảnh riêng ' + str(_i) + '</span></figcaption>'
                f'<img src="data:{_mime};base64,{_encoded}" alt="Custom image {_i}" '
                'style="max-width:280px;max-height:210px;object-fit:contain" /></figure>'
            )
        display(HTML('<p lang="en">Private image preview — clear outputs before sharing.</p>'
                     '<p lang="vi">Xem ảnh riêng — xóa output trước khi chia sẻ.</p>'
                     + ''.join(_custom_cards)))
# %% NOTEBOOK CELL credentials-input
#@title Secure API connection / Kết nối API bảo mật { display-mode: "form" }
# Invalidate the old session before any prompt or validation can fail.
connection_ready = False
configured_connection_identity = None
verified_model_id = None
verified_connection_identity = None
base_url = None
api_token = None
from getpass import getpass
import hashlib
from urllib.parse import urlsplit


def validated_base_url(value):
    value = value.strip().rstrip("/")
    try:
        if any(ord(ch) <= 32 or ord(ch) == 127 for ch in value):
            raise ValueError
        parsed = urlsplit(value)
        parsed.port  # Validate malformed and out-of-range ports before HTTP.
    except ValueError:
        # URL parser diagnostics may include embedded credentials or private hosts.
        raise ValueError(message_pair(
            "Enter a valid API base URL without credentials or whitespace.",
            "Nhập URL API gốc hợp lệ, không chứa credential hoặc khoảng trắng.",
        )) from None
    local = parsed.hostname in {"127.0.0.1", "localhost"}
    if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
        raise ValueError(message_pair(
            "HTTPS API endpoint required (HTTP allowed only for localhost)",
            "Cần URL HTTPS (HTTP chỉ được chấp nhận cho localhost)",
        ))
    if (not parsed.hostname or parsed.username or parsed.password
        or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError(message_pair(
            "Enter only the API base URL, without /v1, query, or credentials",
            "Chỉ nhập URL API gốc, không thêm /v1, query hoặc credential",
        ))
    return value

preview_only_mode = not (
    bool(globals().get("RUN_API_DEMO", False))
    or (bool(os.environ.get("VISION_EMBEDDINGS_BASE_URL", "").strip())
        and bool(os.environ.get("VISION_EMBEDDINGS_API_TOKEN", "").strip()))
)
if preview_only_mode:
    show_stage("3 / 5", "Preview only — no credentials requested",
        "Bundled images are ready. To connect, enable RUN_API_DEMO in Model configuration and rerun the helpers. No HTTP or GPU request was made.",
        "Chỉ xem trước — không yêu cầu thông tin xác thực",
        "Ảnh mẫu đã sẵn sàng. Để kết nối, bật RUN_API_DEMO trong Model configuration rồi chạy lại helper. Không gửi request HTTP hoặc GPU.")
else:
    _candidate_url = os.environ.get("VISION_EMBEDDINGS_BASE_URL", "").strip()
    if not _candidate_url:
        _candidate_url = getpass(
            "English: API endpoint URL (HTTPS)\n"
            "Tiếng Việt: URL HTTPS của API\n> "
        ).strip()
    _candidate_url = validated_base_url(_candidate_url)

    _candidate_token = os.environ.get("VISION_EMBEDDINGS_API_TOKEN", "").strip()
    if not _candidate_token:
        _candidate_token = getpass(
            "English: API Bearer token (hidden)\n"
            "Tiếng Việt: API Bearer token (nhập ẩn)\n> "
        ).strip()
    _candidate_token = validate_api_token(_candidate_token)
    # Publish only a complete validated session, never a partial URL/token pair.
    _candidate_identity = (
        _candidate_url, hashlib.sha256(_candidate_token.encode("utf-8")).hexdigest()
    )
    base_url, api_token = _candidate_url, _candidate_token
    configured_connection_identity = _candidate_identity
    connection_ready = True
    show_stage(
        "3 / 5",
        "Secure connection details are ready",
        "The API address and access token are available in this session. "
        "Neither is shown in the notebook output.",
        "Thông tin kết nối bảo mật đã sẵn sàng",
        "Địa chỉ API và mã truy cập chỉ được sử dụng trong phiên chạy này. "
        "Notebook không hiển thị hai giá trị đó.",
    )
# %% NOTEBOOK CELL http-verification-code
#@title Check API connection / Kiểm tra kết nối API { display-mode: "form" }
# A form edit requires Configuration helpers before network or report reuse.
if globals().get("configuration_ready") is not True:
    # No helper code has to exist yet: this must work in a fresh runtime.
    raise RuntimeError(
        "English: Run Configuration helpers after editing Model configuration.\n\n"
        "Tiếng Việt: Hãy chạy Configuration helpers sau khi sửa Model configuration."
    )
require_current_configuration()

import hashlib
import json
from http.client import InvalidURL
import urllib.error
import urllib.request


class NoCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # EN: Never forward an API Bearer token through a redirect.
        # VI: Không chuyển tiếp Bearer token qua URL chuyển hướng.
        return None

opener = urllib.request.build_opener(NoCrossHostRedirect)

def current_connection_identity():
    """Reject stale, partial or manually edited connections before any HTTP traffic."""
    error = message_pair(
        "Secure API connection is invalid or changed. Rerun Secure API connection "
        "and Check API connection.",
        "Kết nối API bảo mật không hợp lệ hoặc đã thay đổi. Hãy chạy lại "
        "Secure API connection và Check API connection.",
    )
    if globals().get("connection_ready") is not True:
        raise RuntimeError(error)
    try:
        safe_url = validated_base_url(base_url)
        safe_token = validate_api_token(api_token)
        identity = (safe_url, hashlib.sha256(safe_token.encode("utf-8")).hexdigest())
    except (AttributeError, TypeError, ValueError):
        raise RuntimeError(error) from None
    if (safe_url != base_url
            or identity != globals().get("configured_connection_identity")):
        raise RuntimeError(error)
    return identity


def api_request(path, payload=None):
    require_current_configuration()
    safe_token = validate_api_token(api_token)
    safe_url, _ = current_connection_identity()
    headers = {"Authorization": "Bearer " + safe_token}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    try:
        request = urllib.request.Request(
            safe_url + path,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Never expose response bodies: they may contain private input.
        hints = {
            401: ("Check the API Bearer token.", "Kiểm tra Bearer token API."),
            403: ("Check access permission.", "Kiểm tra quyền truy cập."),
            413: ("Reduce image/request byte size.", "Giảm dung lượng ảnh/request."),
            422: ("Check fields, dimension, image pixels and file format.",
                  "Kiểm tra trường, số chiều, pixel ảnh và định dạng."),
            429: ("Check the Bearer token, authentication lockout and inference limits; "
                  "respect Retry-After.",
                  "Kiểm tra Bearer token, khóa xác thực và giới hạn inference; "
                  "tôn trọng Retry-After."),
            503: ("Check GPU worker/deployment availability.",
                  "Kiểm tra GPU worker và deployment."),
            504: ("Check worker readiness and timeouts.",
                  "Kiểm tra worker và timeout."),
        }
        en, vi = hints.get(exc.code, (
            "Check the API status and request configuration.",
            "Kiểm tra API và cấu hình request.",
        ))
        raise RuntimeError(message_pair(
            f"API request failed: HTTP {exc.code} on {path}. {en}",
            f"Request API thất bại: HTTP {exc.code} tại {path}. {vi}",
        )) from None
    except (TimeoutError, urllib.error.URLError) as exc:
        if (isinstance(exc, TimeoutError)
                or isinstance(getattr(exc, "reason", None), TimeoutError)):
            # GPU processing may continue after client timeout. Never retry POST.
            raise RuntimeError(message_pair(
                "API request timed out. Check the client timeout, deployment and GPU "
                "worker readiness. The server may still process this request; "
                "no automatic POST retry was sent.",
                "Request API quá thời gian chờ. Kiểm tra timeout phía client, "
                "deployment và trạng thái GPU worker. Server có thể vẫn đang xử lý "
                "request này; không tự động gửi lại POST.",
            )) from None
        raise RuntimeError(message_pair(
            f"Cannot reach the configured API endpoint on {path}",
            f"Không thể kết nối đến API đã cấu hình tại {path}",
        )) from None
    except (ValueError, UnicodeError, InvalidURL):
        # Original urllib/header exceptions may embed the API token.
        raise RuntimeError(message_pair(
            "Invalid HTTP request or response encoding; check API configuration.",
            "Request HTTP hoặc mã hóa phản hồi không hợp lệ; kiểm tra cấu hình API.",
        )) from None

verified_model_id = None  # Previous model capability is not valid for new preflight.
verified_connection_identity = None
if globals().get("preview_only_mode", False):
    show_stage("4 / 5", "API check skipped", "Preview mode makes no HTTP requests.",
        "Bỏ qua kiểm tra API", "Chế độ xem trước không gửi request HTTP.")
else:
    verified_model_id = None  # Previous model capability is not valid for new preflight.
    verified_connection_identity = None
    _preflight_identity = current_connection_identity()
    health = api_request("/v1/health")
    if require_real() and health.get("backend") != "modal":
        raise RuntimeError(message_pair(
            "Endpoint is not a real Modal backend. For mock tests only, "
            "set VISION_EMBEDDINGS_REQUIRE_REAL=false.",
            "Endpoint không phải backend Modal thật. Chỉ khi kiểm thử mock, "
            "đặt VISION_EMBEDDINGS_REQUIRE_REAL=false.",
        ))
    catalog = api_request("/v1/models")
    available = {item["id"]: item for item in catalog.get("models", [])}
    selected_model_info = api_request("/v1/models/" + model_id)
    supports_custom_instruction = selected_model_info.get("supports_custom_instruction")
    if type(supports_custom_instruction) is not bool:
        raise RuntimeError(message_pair("Missing model instruction capability.",
                                        "Thiếu thông tin khả năng hỗ trợ instruction."))
    if instruction_mode == "custom" and not supports_custom_instruction:
        raise RuntimeError(message_pair("API does not support custom instruction for this model.",
                                        "API không hỗ trợ custom instruction với model này."))
    if model_id not in available:
        raise RuntimeError(message_pair(
            f"Selected model not in API catalog: {model_id}",
            f"Model đã chọn không có trong danh mục API: {model_id}",
        ))
    # Only a completed preflight on the same connection may authorize inference.
    require_current_configuration()
    if current_connection_identity() != _preflight_identity:
        raise RuntimeError(message_pair(
            "API connection changed during preflight. Rerun Check API connection.",
            "Kết nối API đã thay đổi khi kiểm tra. Hãy chạy lại Check API connection.",
        ))
    verified_connection_identity = _preflight_identity
    verified_model_id = model_id
    show_stage(
        "4 / 5",
        "API connection verified",
        f"The API responded successfully (version {health.get('version')}). "
        f"The selected model {model_id} is listed. GPU inference is checked in the next step.",
        "Đã xác thực kết nối API",
        f"API phản hồi thành công (phiên bản {health.get('version')}). "
        f"Model {model_id} có trong danh sách. GPU sẽ được kiểm tra ở bước tiếp theo.",
    )
# %% NOTEBOOK CELL embedding-code
#@title Generate embeddings / Tạo embedding { display-mode: "form" }
# A form edit requires Configuration helpers before network or report reuse.
if globals().get("configuration_ready") is not True:
    # No helper code has to exist yet: this must work in a fresh runtime.
    raise RuntimeError(
        "English: Run Configuration helpers after editing Model configuration.\n\n"
        "Tiếng Việt: Hãy chạy Configuration helpers sau khi sửa Model configuration."
    )
require_current_configuration()

import time

def check_embedding_metadata(result, model, instruction):
    meta = result.get("embedding_metadata")
    if not isinstance(meta, dict):
        raise RuntimeError("English: Missing embedding_metadata\nTiếng Việt: Thiếu metadata embedding")
    pre = meta.get("preprocessing_version")
    instruction_hash = meta.get("instruction_sha256")
    config_hash = meta.get("config_sha256")
    def is_hash(v):
        return isinstance(v, str) and len(v) == 64 and all(ch in "0123456789abcdef" for ch in v)
    expected_instruction_hash = (
        hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        if model == "qwen3-vl-embedding-8b" else None
    )
    if not isinstance(pre, str) or not pre or instruction_hash != expected_instruction_hash or not is_hash(config_hash):
        raise RuntimeError("English: Invalid embedding metadata\nTiếng Việt: Metadata embedding không hợp lệ")
    config = {
        "dimension": dimension, "instruction_sha256": instruction_hash,
        "model": model, "preprocessing_version": pre, "revision": result.get("revision"),
    }
    expected = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if config_hash != expected:
        raise RuntimeError("English: Configuration SHA-256 mismatch\nTiếng Việt: Mã băm cấu hình không khớp")
    return meta


def require_verified_connection(selected_model):
    require_current_configuration()
    if (globals().get("verified_model_id") != selected_model
            or globals().get("connection_ready") is not True
            or globals().get("verified_connection_identity") is None
            or globals().get("verified_connection_identity")
            != current_connection_identity()):
        raise RuntimeError(message_pair(
            "Model changed or API not checked. Rerun Check API connection.",
            "Model đã đổi hoặc chưa kiểm tra API. Hãy chạy lại Check API connection.",
        ))


# GPU requests need explicit consent; mock tests use the existing local route.
# Chỉ gửi GPU POST khi cho phép; mock cục bộ vẫn dùng endpoint đồng bộ.
USE_ASYNC_JOBS = True #@param {type:"boolean"}
APPROVE_NEW_GPU_JOBS = False #@param {type:"boolean"}
POLL_INTERVAL_SECONDS = 3 #@param {type:"integer"}
MAX_POLL_SECONDS = 900 #@param {type:"integer"}

import os
import tempfile
from pathlib import Path

if not 1 <= int(POLL_INTERVAL_SECONDS) <= 60:
    raise ValueError("English: Polling interval must be 1–60 seconds\nTiếng Việt: Khoảng chờ polling phải từ 1–60 giây")
if not 1 <= int(MAX_POLL_SECONDS) <= 3600:
    raise ValueError("English: Polling window must be 1–3600 seconds\nTiếng Việt: Thời gian polling tối đa phải từ 1–3600 giây")

# This file persists across cell re-runs and some notebook restarts. It holds
# verified vectors, not the raw inputs, the API URL or the Bearer token.
# Do not upload it with public notebook outputs if using private images.
_ASYNC_CACHE_PATH = Path(os.environ.get(
    "VISION_EMBEDDINGS_NOTEBOOK_CACHE", "multimodal_embeddings_api_demo_cache.json"
)).expanduser()

def _read_async_cache():
    if not _ASYNC_CACHE_PATH.exists():
        return {"schema_version": 1, "records": {}}
    if _ASYNC_CACHE_PATH.stat().st_size > 8_000_000:
        raise RuntimeError("English: Cache exceeds 8 MB\nTiếng Việt: Cache vượt 8 MB")
    cache = json.loads(_ASYNC_CACHE_PATH.read_text(encoding="utf-8"))
    if (not isinstance(cache, dict) or cache.get("schema_version") != 1
            or not isinstance(cache.get("records"), dict)):
        raise RuntimeError("English: Invalid async cache; do not overwrite without backing it up\nTiếng Việt: Cache async không hợp lệ; hãy sao lưu trước khi ghi đè")
    return cache

_ASYNC_CACHE = _read_async_cache()

def _save_async_cache():
    _ASYNC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".async-embedding-", suffix=".tmp",
                                    dir=str(_ASYNC_CACHE_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_ASYNC_CACHE, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, _ASYNC_CACHE_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

def _async_identity(body, raw, *, legacy=False):
    # Identity is scoped to the deployment and embedding profile, NEVER to the
    # Bearer token. Rotating a credential must not trigger another GPU job.
    # Store only the hash, never the API address or credential.
    spec = globals().get("selected_model_info", {})
    composite = {
        "deployment_url": current_connection_identity()[0],
        "model": body["model"],
        "revision": spec.get("revision") if isinstance(spec, dict) else None,
        "dimension": body["dimension"],
        "kind": "image" if "image_base64" in body else "text",
        "image_or_text_sha256": hashlib.sha256(raw).hexdigest(),
        "instruction": body.get("instruction"),
    }
    if legacy:
        # Read pre-upgrade checkpoint keys when the original token is unchanged.
        # Never create new jobs using this credential-bound legacy format.
        composite.pop("deployment_url")
        composite["connection"] = current_connection_identity()
    key = hashlib.sha256(
        json.dumps(composite, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return "demo-" + key[:40]

def _explicit_legacy_resume_id(stable_id):
    """Opt-in, GET-only recovery of a saved pre-rotation legacy job."""
    raw_mapping = os.environ.get("VISION_EMBEDDINGS_LEGACY_REQUEST_ID_MAP", "").strip()
    if not raw_mapping:
        return None
    try:
        mapping = json.loads(raw_mapping)
    except (ValueError, TypeError):
        raise RuntimeError(message_pair(
            "Invalid legacy request-ID JSON mapping; do not send a new GPU POST.",
            "Mapping JSON request ID cũ không hợp lệ; không gửi GPU POST mới.",
        )) from None
    if not isinstance(mapping, dict):
        raise RuntimeError("English: Legacy request-ID mapping must be a JSON object\nTiếng Việt: Mapping request ID cũ phải là đối tượng JSON")
    older = mapping.get(stable_id)
    if older is None:
        return None
    # Mapping only selects a record already saved locally. It never invents a
    # new request ID or re-submits an uncertain legacy job.
    if (not isinstance(older, str) or len(older) != 45
            or not older.startswith("demo-")
            or any(char not in "0123456789abcdef" for char in older[5:])):
        raise RuntimeError("English: Invalid legacy request ID\nTiếng Việt: Mã request cũ không hợp lệ")
    record = _ASYNC_CACHE["records"].get(older)
    if not isinstance(record, dict):
        raise RuntimeError(message_pair(
            "Mapped legacy job not found in saved cache; inspect backup before inference.",
            "Không thấy job cũ trong cache; kiểm tra backup trước khi inference.",
        ))
    if not isinstance(record.get("result"), dict):
        job_id = record.get("job_id")
        if (not isinstance(job_id, str) or len(job_id) != 64
                or any(char not in "0123456789abcdef" for char in job_id)):
            raise RuntimeError(message_pair(
                "Legacy job lacks a recoverable job ID. Do not submit a new POST.",
                "Job cũ thiếu job ID để khôi phục. Không gửi POST mới.",
            ))
    return older


def _async_result(body, raw):
    started = time.perf_counter()  # Include submission and polling in client wall time.
    rid = _async_identity(body, raw)
    if rid not in _ASYNC_CACHE["records"]:
        older = _async_identity(body, raw, legacy=True)
        if older in _ASYNC_CACHE["records"]:
            rid = older  # Recover old jobs/vectors without a new GPU POST.
        else:
            restored = _explicit_legacy_resume_id(rid)
            if restored is not None:
                rid = restored  # Explicit saved pre-rotation ID; GET only.
    record = _ASYNC_CACHE["records"].get(rid)
    if record is not None and not isinstance(record, dict):
        raise RuntimeError("English: Invalid async cache record\nTiếng Việt: Bản ghi cache không hợp lệ")
    if record and isinstance(record.get("result"), dict):
        # A cache hit is not a zero-millisecond inference measurement.
        cached = dict(record["result"])
        cached["_client_timing"] = {
            "cache_hit": True, "submission_ms": None,
            "polling_wait_ms": None, "total_client_ms": None,
        }
        return cached, rid, None

    if record and record.get("state") == "FAILED":
        raise RuntimeError(message_pair(
            "This job failed. Review the deployment; never retry with a new ID automatically.",
            "Job đã lỗi. Hãy kiểm tra deployment; không tự tạo ID mới để retry.",
        ))
    submission_ms = None
    response = None
    if record and record.get("state") == "UNCERTAIN":
        # Lost POST ACK: look up the caller's tenant-scoped job on the server.
        # Never create new GPU work, even when APPROVE_NEW_GPU_JOBS=True.
        try:
            response = api_request("/v1/embeddings/jobs/by-request/" + rid)
        except RuntimeError as exc:
            # Older deployments only support the legacy owner-scoped v1 key.
            # Both recovery paths are strictly GET-only, without re-submission.
            if "HTTP 404" not in str(exc):
                raise
            if not record.get("job_id"):
                record["job_id"] = hashlib.sha256(
                    ("multimodal-embeddings-job-v1:" + rid).encode("utf-8")
                ).hexdigest()
                _ASYNC_CACHE["records"][rid] = record
                _save_async_cache()
        else:
            if isinstance(response, dict) and isinstance(response.get("job_id"), str):
                # A rotated per-client key may have a v2 server namespace;
                # trust only the authenticated read-only server lookup.
                record["job_id"] = response["job_id"]
                _ASYNC_CACHE["records"][rid] = record
                _save_async_cache()
    if response is not None or (record and record.get("job_id")):
        pass  # response ready, or poll known job ID; never re-POST uncertain work
    else:
        if not APPROVE_NEW_GPU_JOBS:
            raise RuntimeError(message_pair(
                "No new GPU POSTs approved. Set APPROVE_NEW_GPU_JOBS=True, or restore the saved cache.",
                "Chưa cho phép GPU POST mới. Bật APPROVE_NEW_GPU_JOBS=True hoặc khôi phục cache đã lưu.",
            ))
        # Reserve the stable request_id before the potentially billable POST.
        # An interrupted response is indeterminate, not evidence of failure.
        _ASYNC_CACHE["records"][rid] = {
            "state": "UNCERTAIN", "job_id": None,
        }
        _save_async_cache()
        submit_started = time.perf_counter()
        response = api_request("/v1/embeddings/async", {**body, "request_id": rid})
        submission_ms = (time.perf_counter() - submit_started) * 1000.0
    deadline = time.monotonic() + int(MAX_POLL_SECONDS)
    while True:
        if response is None:
            job_id = _ASYNC_CACHE["records"][rid].get("job_id")
            response = api_request("/v1/embeddings/jobs/" + job_id)
        if not isinstance(response, dict):
            raise RuntimeError("English: Unexpected async job response\nTiếng Việt: Phản hồi job không hợp lệ")
        state = response.get("state")
        server_id = response.get("job_id")
        if not isinstance(server_id, str) or len(server_id) != 64:
            raise RuntimeError("English: Missing async job ID\nTiếng Việt: Thiếu mã job")
        if record and record.get("job_id") and server_id != record["job_id"]:
            raise RuntimeError("English: Job ID changed\nTiếng Việt: Mã job đã thay đổi")
        if state not in ("QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "INDETERMINATE"):
            raise RuntimeError("English: Unknown async job state\nTiếng Việt: Trạng thái job không hợp lệ")
        _ASYNC_CACHE["records"][rid] = {"state": state, "job_id": server_id}
        _save_async_cache()
        if state == "SUCCEEDED":
            result = response.get("result")
            if not isinstance(result, dict) or result.get("request_id") != rid:
                raise RuntimeError("English: Async result/request ID mismatch\nTiếng Việt: Kết quả không khớp request")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result = dict(result)
            result["_client_timing"] = {
                "cache_hit": False,
                "submission_ms": round(submission_ms, 3) if submission_ms is not None else None,
                "polling_wait_ms": round(max(0.0, elapsed_ms - (submission_ms or 0.0)), 3),
                "total_client_ms": round(elapsed_ms, 3),
            }
            return result, rid, elapsed_ms
        if state == "FAILED":
            raise RuntimeError(message_pair(
                "GPU job failed. Inspect deployment logs; no automatic retry was sent.",
                "GPU job lỗi. Hãy kiểm tra log deployment; không có retry tự động.",
            ))
        if state == "INDETERMINATE":
            raise RuntimeError(message_pair(
                "Coordinator heartbeat or dispatch is overdue. GPU work may still be running. "
                "Inspect deployment logs; never submit a new ID automatically.",
                "Job quá hạn heartbeat hoặc dispatch, GPU có thể vẫn đang chạy. "
                "Kiểm tra log deployment; không tự gửi request ID mới.",
            ))
        if time.monotonic() >= deadline:
            raise RuntimeError(message_pair(
                "Job still pending. Rerun Generate embeddings to poll the same job, without new GPU POST.",
                "Job vẫn đang xử lý. Chạy lại Generate embeddings để poll cùng job, không gửi GPU POST mới.",
            ))
        time.sleep(int(POLL_INTERVAL_SECONDS))
        response = None

def _store_verified_async_result(rid, result):
    if rid is None:
        return
    record = _ASYNC_CACHE["records"].get(rid) or {}
    record["state"] = "SUCCEEDED"
    record["result"] = result
    _ASYNC_CACHE["records"][rid] = record
    _save_async_cache()

def checked_embedding(selected_model, *, text=None, image=None):
    require_verified_connection(selected_model)
    body = {"model": selected_model, "dimension": dimension}
    if (text is None) == (image is None):
        raise ValueError(message_pair(
            "Provide exactly one of text or image",
            "Chỉ cung cấp một loại dữ liệu: văn bản hoặc hình ảnh",
        ))
    normalized_text = text.strip() if text is not None else None
    if text is not None and not normalized_text:
        raise ValueError(message_pair(
            "Text must not be empty after trimming surrounding whitespace.",
            "Văn bản không được rỗng sau khi bỏ khoảng trắng đầu/cuối.",
        ))
    raw = normalized_text.encode("utf-8") if text is not None else image
    if text is not None:
        body["text"] = normalized_text
    else:
        body["image_base64"] = base64.b64encode(image).decode("ascii")
    if instruction_mode == "custom":
        if selected_model != "qwen3-vl-embedding-8b" or not supports_custom_instruction:
            raise ValueError("English: Custom instruction unsupported\nTiếng Việt: Không hỗ trợ instruction tùy chỉnh")
        body["instruction"] = custom_instruction
    async_request_id = None
    if USE_ASYNC_JOBS and health.get("backend") == "modal":
        result, async_request_id, wall_ms = _async_result(body, raw)
    else:
        if health.get("backend") != "mock" and not APPROVE_NEW_GPU_JOBS:
            raise RuntimeError(message_pair(
                "Live synchronous GPU POST not approved. Enable APPROVE_NEW_GPU_JOBS.",
                "Chưa cho phép GPU POST đồng bộ. Hãy bật APPROVE_NEW_GPU_JOBS.",
            ))
        started = time.perf_counter()
        result = api_request("/v1/embeddings", body)
        wall_ms = (time.perf_counter() - started) * 1000
    # None denotes a reused vector, not a zero-latency model execution.
    result["_client_elapsed_ms"] = round(wall_ms, 3) if wall_ms is not None else None
    result["_client_cache_hit"] = wall_ms is None and async_request_id is not None
    vector = result.get("vector")
    if (result.get("model") != selected_model
            or type(result.get("dimension")) is not int
            or result["dimension"] != dimension
            or not isinstance(vector, list)
            or len(vector) != dimension):
        raise RuntimeError(message_pair(
            "Unexpected model, response dimension or vector format",
            "Model, số chiều response hoặc định dạng vector không đúng",
        ))
    if not all(type(v) in (int, float) and math.isfinite(v) for v in vector):
        raise RuntimeError(message_pair(
            "Non-finite embedding",
            "Embedding chứa giá trị không hữu hạn",
        ))
    norm = math.sqrt(sum(float(v) ** 2 for v in vector))
    if abs(norm - 1.0) >= 0.001:
        raise RuntimeError(message_pair(
            "Embedding is not L2-normalized",
            "Embedding chưa được chuẩn hóa L2",
        ))
    if result.get("input_sha256") != hashlib.sha256(raw).hexdigest():
        raise RuntimeError(message_pair(
            "Input digest mismatch",
            "Mã băm dữ liệu đầu vào không khớp",
        ))
    effective_instruction = (
        custom_instruction if instruction_mode == "custom" else QWEN_LEGACY_INSTRUCTION
    ) if selected_model == "qwen3-vl-embedding-8b" else None
    check_embedding_metadata(result, selected_model, effective_instruction)
    if require_real() and result.get("backend") != "modal":
        raise RuntimeError(message_pair(
            "Expected a real Modal backend, not mock/local output",
            "Yêu cầu backend Modal thật, không phải kết quả mock/local",
        ))
    _store_verified_async_result(async_request_id, result)
    return result, norm

def cosine(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b, strict=True))

completed_run_config = None

if globals().get("preview_only_mode", False):
    show_stage("5 / 5", "Preview complete; inference skipped",
        "No embedding requests were submitted. The sample gallery is available above.",
        "Đã xem trước; bỏ qua inference",
        "Không gửi request embedding. Thư viện ảnh mẫu nằm phía trên.")
else:
    # No partial or stale sample configuration may trigger an HTTP POST.
    if (not globals().get("sample_prepared", False)
            or globals().get("sample_selection_snapshot") is None
            or any(len(globals().get(name, ())) != 3 for name in (
                "text_samples", "image_samples", "sample_labels",
                "text_sources", "image_sources"))):
        raise RuntimeError(message_pair(
            "Samples are incomplete or preparation failed. Rerun Prepare input samples.",
            "Dữ liệu mẫu chưa đủ hoặc chuẩn bị thất bại. Hãy chạy lại Prepare input samples.",
        ))
    if current_sample_selection() != sample_selection_snapshot:
        sample_prepared = False
        raise RuntimeError(message_pair(
            "Samples or settings changed. Rerun Prepare input samples.",
            "Dữ liệu mẫu hoặc cấu hình đã thay đổi. Hãy chạy lại Prepare input samples.",
        ))

    require_verified_connection(model_id)

    progress_handle = show_stage(
        "5 / 5",
        "Processing six examples",
        "Embedding three texts and three images using one model. "
        "The first request may need to start the GPU.",
        "Đang xử lý sáu ví dụ",
        "Đang embedding ba văn bản và ba ảnh bằng một model. "
        "Request đầu có thể cần khởi động GPU.",
        working=True,
    )

    text_results, image_results, text_norms, image_norms = [], [], [], []
    for index, (text, image) in enumerate(zip(text_samples, image_samples, strict=True), 1):
        text_result, text_norm = checked_embedding(model_id, text=text)
        image_result, image_norm = checked_embedding(model_id, image=image)
        text_results.append(text_result)
        image_results.append(image_result)
        text_norms.append(text_norm)
        image_norms.append(image_norm)

    revisions = {(result.get("repository"), result.get("revision"))
                 for result in text_results + image_results}
    configs = {(r["embedding_metadata"]["preprocessing_version"],
                r["embedding_metadata"]["config_sha256"]) for r in text_results + image_results}
    if len(configs) != 1:
        raise RuntimeError("English: Embedding config changed\nTiếng Việt: Cấu hình embedding thay đổi")
    if len(revisions) != 1 or any(not all(item) for item in revisions):
        raise RuntimeError(message_pair(
            "Model revisions or repositories differ; refusing cosine comparisons",
            "Repository hoặc revision model khác nhau; từ chối so sánh cosine",
        ))

    similarity_matrix = [
        [round(cosine(text["vector"], image["vector"]), 6) for text in text_results]
        for image in image_results
    ]
    summary = [
        {
            "sample": index + 1,
            "model": model_id,
            "revision": text_results[index].get("revision"),
            "backend": text_results[index].get("backend"),
            "dimension": dimension,
            "text_norm": round(text_norms[index], 6),
            "image_norm": round(image_norms[index], 6),
            "cosine_text_image": similarity_matrix[index][index],
            "text_elapsed_ms": text_results[index].get("elapsed_ms"),
            "image_elapsed_ms": image_results[index].get("elapsed_ms"),
        }
        for index in range(3)
    ]
    completed_run_config = {
        "connection_identity": current_connection_identity(),
        "model": model_id,
        "dimension": dimension,
        "instruction_mode": instruction_mode,
        "custom_instruction_sha256": (
            hashlib.sha256(custom_instruction.encode("utf-8")).hexdigest()
            if instruction_mode == "custom" else None
        ),
        "config_sha256": text_results[0]["embedding_metadata"]["config_sha256"],
        "input_sha256": {
            "text": tuple(hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
                          for text in text_samples),
            "image": tuple(hashlib.sha256(image).hexdigest() for image in image_samples),
        },
        "sample_labels": tuple(sample_labels),
        "text_sources": tuple(text_sources),
        "image_sources": tuple(image_sources),
        "default_pairs": default_pairs,
    }

    show_stage(
        "5 / 5",
        "All six examples passed validation",
        "Three text and three image responses passed verification. "
        "The next section shows the 3 × 3 cosine comparison.",
        "Cả sáu ví dụ đều kiểm tra thành công",
        "Ba response văn bản và ba response ảnh đều hợp lệ. "
        "Phần tiếp theo hiển thị ma trận cosine 3 × 3.",
        handle=progress_handle,
    )
# %% NOTEBOOK CELL results-code
#@title Results (run; no edits) / Kết quả (chạy, không sửa) { display-mode: "form" }
# Report map / Sơ đồ báo cáo: verify current vectors and session provenance,
# compute the 3 x 3 cosine matrix, then render accessible bilingual HTML.
# A form edit requires Configuration helpers before network or report reuse.
if globals().get("configuration_ready") is not True:
    # No helper code has to exist yet: this must work in a fresh runtime.
    raise RuntimeError(
        "English: Run Configuration helpers after editing Model configuration.\n\n"
        "Tiếng Việt: Hãy chạy Configuration helpers sau khi sửa Model configuration."
    )
require_current_configuration()

if globals().get("preview_only_mode", False):
    report_html = None
    show_stage("5 / 5", "Preview only — no cosine scores",
        "Samples are ready, but no embeddings were generated. Enable RUN_API_DEMO and approve GPU jobs for measured results.",
        "Chỉ xem trước — chưa có điểm cosine",
        "Đã chuẩn bị mẫu nhưng chưa tạo embedding. Bật RUN_API_DEMO và phê duyệt GPU job mới để có kết quả đo.")
else:
    # Helper cell: stale-state checks, cosine matrix and bilingual HTML/CSS rendering.
    # Cell trợ giúp: kiểm tra dữ liệu cũ, ma trận cosine và HTML/CSS song ngữ.
    # Accessible English/Vietnamese report for three matched examples and nine comparisons.
    # A user can re-run configuration and report cells without re-running inference.
    # Never label stale embeddings with new model or instruction settings.
    # Results may only describe the currently selected, successfully prepared inputs.
    try:
        samples_current = (
            globals().get("sample_prepared") is True
            and globals().get("sample_selection_snapshot") is not None
            and current_sample_selection() == sample_selection_snapshot
        )
    except (NameError, RuntimeError, OSError, ValueError):
        samples_current = False
    if not samples_current:
        raise RuntimeError(message_pair(
            "Samples changed or preparation failed. Run Prepare input samples → "
            "Generate embeddings before Results.",
            "Dữ liệu mẫu đã đổi hoặc chuẩn bị thất bại. Chạy Prepare input samples → "
            "Generate embeddings trước Results.",
        ))

    run_config = globals().get("completed_run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError(message_pair(
            "No completed embedding run. Run the embedding cell before viewing the report.",
            "Chưa có lượt embedding hoàn tất. Hãy chạy cell tạo embedding trước khi xem báo cáo.",
        ))
    # Only the verified connection that produced these vectors may report success.
    # Chỉ kết nối đã xác minh và tạo các vector này mới được báo cáo thành công.
    try:
        active_connection_identity = current_connection_identity()
    except (NameError, RuntimeError):
        active_connection_identity = None
    if (active_connection_identity is None
            or globals().get("verified_model_id") != model_id
            or globals().get("verified_connection_identity") != active_connection_identity
            or run_config.get("connection_identity") != active_connection_identity):
        raise RuntimeError(message_pair(
            "API connection changed or is unverified. Run Secure API connection → "
            "Check API connection → Generate embeddings before Results.",
            "Kết nối API đã thay đổi hoặc chưa được xác minh. Chạy Secure API connection → "
            "Check API connection → Generate embeddings trước Results.",
        ))

    active_custom_hash = (
        hashlib.sha256(custom_instruction.encode("utf-8")).hexdigest()
        if instruction_mode == "custom" else None
    )
    if (run_config["model"] != model_id
            or run_config["dimension"] != dimension
            or run_config["instruction_mode"] != instruction_mode
            or run_config["custom_instruction_sha256"] != active_custom_hash):
        raise RuntimeError(message_pair(
            "Configuration changed since inference. Run the embedding cell again before reporting.",
            "Cấu hình đã thay đổi sau inference. Hãy chạy lại cell tạo embedding trước khi xem báo cáo.",
        ))
    if any(
        (result.get("embedding_metadata") or {}).get("config_sha256")
        != run_config["config_sha256"]
        for result in text_results + image_results
    ):
        raise RuntimeError(message_pair(
            "Embedding results do not match the recorded run configuration.",
            "Kết quả embedding không khớp với cấu hình được ghi nhận khi chạy.",
        ))

    # Re-running only the sample-preparation cell must not relabel old results.
    current_input_hashes = {
        "text": tuple(hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
                      for text in text_samples),
        "image": tuple(hashlib.sha256(image).hexdigest() for image in image_samples),
    }
    if (run_config["input_sha256"] != current_input_hashes
            or run_config["sample_labels"] != tuple(sample_labels)
            or run_config["text_sources"] != tuple(text_sources)
            or run_config["image_sources"] != tuple(image_sources)
            or run_config["default_pairs"] != default_pairs
            or any(result.get("input_sha256") != run_config["input_sha256"]["text"][i]
                   for i, result in enumerate(text_results))
            or any(result.get("input_sha256") != run_config["input_sha256"]["image"][i]
                   for i, result in enumerate(image_results))):
        raise RuntimeError(message_pair(
            "Inputs or labels changed since inference. Run the embedding cell again before reporting.",
            "Đầu vào hoặc nhãn đã thay đổi sau inference. Hãy chạy lại cell tạo embedding trước khi xem báo cáo.",
        ))

if not globals().get("preview_only_mode", False):
    from html import escape as html_escape


    def safe_text(value):
        return html_escape(str("N/A" if value is None else value), quote=True)


    def seconds(milliseconds):
        if (not isinstance(milliseconds, (int, float))
                or not math.isfinite(milliseconds) or milliseconds < 0):
            return "N/A"
        return f"{milliseconds / 1000:,.2f} s"


    LANGUAGES = {
        "en": {
            "heading": "English — Results",
            "live": "DEMO SUCCESSFUL",
            "mock": "SIMULATION COMPLETE",
            "intro": "Six authenticated requests were validated: three texts and three images.",
            "pair": "Matching pair", "sample": "Sample",
            "sources": "Input sources", "built_in": "built-in", "custom": "custom",
            "score_note": "Cosine score, not a percentage or confidence level.",
            "size": "Numbers in each vector",
            "size_note": "Equal dimensions permit comparison within the selected model.",
            "time": "API processing time (text / image)",
            "time_note": "Two separate totals of 3 requests each; excludes network.",
            "text": "Text", "image": "Image",
            "state": "state", "load": "model load", "encode": "GPU encode",
            "dispatch": "dispatch/scheduling", "wait": "client wait",
            "matrix": "All nine text ↔ image cosine scores",
            "matrix_note": (
                "Rows are image inputs; columns are text inputs. The three diagonal "
                "cells compare inputs with the same index. Only unchanged built-in inputs "
                "are intentionally semantically paired. Higher values are not probabilities."
            ),
            "technical": "Technical verification",
            "model": "Selected model",
            "revision": "Checkpoint revision",
            "backend": "Inference backend",
            "preprocessing": "Preprocessing version",
            "instruction": "Instruction mode",
            "config": "Embedding configuration SHA-256",
            "norm": "L2 norms (text / image) for each pair",
            "times": "Per-sample API processing",
            "verified": (
                "Passed: six vectors, dimensions, finite values, L2 normalization, "
                "same checkpoint revision and six input SHA-256 checksums."
            ),
            "limits": (
                "Three small examples do not establish retrieval accuracy, "
                "biometric identity, or model superiority. Never compare scores "
                "from different model vector spaces."
            ),
            "privacy": (
                "No raw vectors, private image contents, input hashes, tokens "
                "or private endpoint URLs are shown."
            ),
            "timing": "Detailed API timing (when available)",
            "unavailable": "API stage timings unavailable.",
            "network": (
                "Client wait includes HTTP/network time. Dispatch time may include "
                "provisioning; this is not an isolated cold-start measurement."
            ),
        },
        "vi": {
            "heading": "Tiếng Việt — Kết quả",
            "live": "CHẠY THỬ THÀNH CÔNG",
            "mock": "MÔ PHỎNG HOÀN TẤT",
            "intro": "Đã kiểm tra sáu request có xác thực: ba văn bản và ba ảnh.",
            "pair": "Cặp tương ứng", "sample": "Mẫu",
            "sources": "Nguồn đầu vào", "built_in": "có sẵn", "custom": "tùy chỉnh",
            "score_note": "Điểm cosine, không phải tỷ lệ phần trăm hay độ tin cậy.",
            "size": "Số giá trị trong mỗi vector",
            "size_note": "Số chiều bằng nhau cho phép so sánh trong cùng model.",
            "time": "Thời gian xử lý API (văn bản / ảnh)",
            "time_note": "Hai tổng riêng, mỗi tổng gồm 3 request; không gồm thời gian mạng.",
            "text": "Văn bản", "image": "Ảnh",
            "state": "trạng thái", "load": "nạp model", "encode": "GPU mã hóa",
            "dispatch": "điều phối/lập lịch", "wait": "client chờ",
            "matrix": "Chín điểm cosine văn bản ↔ ảnh",
            "matrix_note": (
                "Hàng tương ứng ảnh; cột tương ứng văn bản. Ba ô trên đường chéo "
                "so sánh input cùng chỉ số. Chỉ các mẫu có sẵn không thay đổi mới "
                "được ghép cặp ngữ nghĩa có chủ ý. Giá trị không phải xác suất."
            ),
            "technical": "Kiểm tra kỹ thuật",
            "model": "Model đã chọn",
            "revision": "Phiên bản checkpoint",
            "backend": "Nền tảng inference",
            "preprocessing": "Phiên bản preprocessing",
            "instruction": "Chế độ instruction",
            "config": "SHA-256 cấu hình embedding",
            "norm": "Chuẩn L2 (văn bản / ảnh) theo từng cặp",
            "times": "Thời gian xử lý API theo mẫu",
            "verified": (
                "Đạt: sáu vector, số chiều, giá trị hữu hạn, chuẩn L2, "
                "cùng revision checkpoint và sáu mã băm SHA-256 đầu vào."
            ),
            "limits": (
                "Ba mẫu nhỏ không chứng minh chất lượng truy xuất, "
                "danh tính sinh trắc học hay model nào tốt hơn. "
                "Không so điểm của hai không gian vector model khác nhau."
            ),
            "privacy": (
                "Không hiển thị vector thô, nội dung ảnh riêng, mã băm input, "
                "token hoặc URL endpoint riêng."
            ),
            "timing": "Chi tiết thời gian API (nếu có)",
            "unavailable": "API không cung cấp thời gian từng giai đoạn.",
            "network": (
                "Thời gian client chờ bao gồm HTTP/mạng. Thời gian điều phối "
                "có thể gồm khởi tạo tài nguyên; không phải phép đo cold-start độc lập."
            ),
        },
    }

    SURFACE = "background:#ffffff !important;color:#15283b !important;"
    MUTED = "color:#334155 !important;"


    # -- Report presentation primitives (bilingual, accessible HTML) --
    def metric_card(title, value, note):
        return (
            '<div style="box-sizing:border-box;flex:1 1 200px;min-width:0;'
            'background:#f8fafc !important;color:#15283b !important;'
            'border:1px solid #cbd5e1;border-radius:12px;padding:16px 18px;">'
            f'<div style="font-size:15px;font-weight:650;{MUTED}">'
            f'{safe_text(title)}</div>'
            '<div style="font-size:27px;line-height:1.3;font-weight:800;'
            'color:#10253b !important;margin:10px 0;overflow-wrap:anywhere;">'
            f'{safe_text(value)}</div>'
            f'<div style="font-size:14px;line-height:1.6;{MUTED}">'
            f'{safe_text(note)}</div></div>'
        )


    def detail_row(title, value):
        return (
            '<div style="display:flex;flex-wrap:wrap;gap:8px 18px;'
            'border-bottom:1px solid #dbe4ed;background:#ffffff !important;'
            'color:#15283b !important;padding:12px 14px;line-height:1.5;">'
            f'<div style="flex:1 1 195px;font-size:15px;{MUTED}">'
            f'{safe_text(title)}</div>'
            '<div style="flex:2 1 250px;font-size:15px;font-weight:650;'
            'color:#10253b !important;min-width:0;overflow-wrap:anywhere;'
            'word-break:break-word;">'
            f'{safe_text(value)}</div></div>'
        )


if not globals().get("preview_only_mode", False):
    # -- Semantic similarity matrix and labels --
    def matrix_html(language):
        t = LANGUAGES[language]
        caption = ("Text–image cosine similarity matrix" if language == "en"
                   else "Ma trận tương đồng cosine giữa văn bản và ảnh")
        cells = [
            '<div style="max-width:660px;overflow-x:auto;margin:14px 0;">',
            '<table style="width:100%;border-collapse:separate;border-spacing:4px;'
            'background:#ffffff !important;color:#15283b !important;">',
            f'<caption style="text-align:left;font-size:14px;font-weight:650;'
            f'color:#15283b !important;padding:6px 0;">{safe_text(caption)}</caption>',
            '<thead><tr>',
            '<th scope="col" style="background:#e2e8f0 !important;color:#10253b !important;'
            'padding:10px 4px;text-align:center;font-size:13px;">'
            f'{safe_text(t["image"] + " / " + t["text"])}</th>',
        ]
        for index in range(1, 4):
            cells.append(
                '<th scope="col" style="background:#e2e8f0 !important;'
                'color:#10253b !important;padding:10px 4px;text-align:center;'
                f'font-size:13px;">{safe_text(t["text"])} {index}</th>'
            )
        cells.append('</tr></thead><tbody>')
        for row_index, row in enumerate(similarity_matrix):
            cells.append('<tr>')
            cells.append(
                '<th scope="row" style="background:#e2e8f0 !important;'
                'color:#10253b !important;padding:10px 4px;text-align:center;'
                f'font-size:13px;">{safe_text(t["image"])} {row_index + 1}</th>'
            )
            for col_index, value in enumerate(row):
                bg = "#dbeafe" if row_index == col_index else "#f8fafc"
                cells.append(
                    f'<td style="background:{bg} !important;color:#10253b !important;'
                    'border:1px solid #cbd5e1;padding:10px 4px;text-align:center;'
                    f'font-size:15px;font-weight:700;">{safe_text(f"{value:.4f}")}</td>'
                )
            cells.append('</tr>')
        cells.append('</tbody></table></div>')
        return "".join(cells)


    def perf_row(title, response, language):
        timing = response.get("timing")
        if not isinstance(timing, dict):
            return detail_row(title, LANGUAGES[language]["unavailable"])
        state = timing.get("model_state", "not_available")
        t = LANGUAGES[language]
        return detail_row(
            title,
            f'{t["state"]}={state} | {t["load"]} {seconds(timing.get("model_load_ms"))} | '
            f'{t["encode"]} {seconds(timing.get("gpu_encode_ms"))} | '
            f'{t["dispatch"]} {seconds(timing.get("gpu_dispatch_overhead_ms"))} | '
            f'{t["wait"]} ' + (
                (("cached" if language == "en" else "từ cache") if response.get("_client_cache_hit") else
                 seconds(response.get("_client_elapsed_ms")))
            ),
        )


if not globals().get("preview_only_mode", False):
    # -- Report assembly and provenance display --
    def report_section(language):
        t = LANGUAGES[language]
        mock = summary[0]["backend"] != "modal"
        total_text = sum(float(v.get("elapsed_ms") or 0) for v in text_results)
        total_image = sum(float(v.get("elapsed_ms") or 0) for v in image_results)
        pair_title = t["pair"] if run_config["default_pairs"] else t["sample"]
        metrics = "".join(
            metric_card(
                f'{pair_title} {index}' + (
                    f' — {run_config["sample_labels"][index - 1]}' if run_config["sample_labels"][index - 1] else ""
                ),
                f'{item["cosine_text_image"]:.4f}',
                t["score_note"],
            )
            for index, item in enumerate(summary, 1)
        )
        metrics += metric_card(t["size"], f'{summary[0]["dimension"]:,}', t["size_note"])
        metrics += metric_card(
            t["time"],
            f'{t["text"]}: {seconds(total_text)} / {t["image"]}: {seconds(total_image)}',
            t["time_note"],
        )
        norms = " · ".join(
            f'{pair["text_norm"]:.4f} / {pair["image_norm"]:.4f}'
            for pair in summary
        )
        sources = " · ".join(
            f'{t["sample"]} {index}'
            + (f' ({run_config["sample_labels"][index - 1]})' if run_config["sample_labels"][index - 1] else "")
            + f': {t["text"]}={t["built_in"] if run_config["text_sources"][index - 1] == "built-in" else t["custom"]}'
            + f', {t["image"]}={t["built_in"] if run_config["image_sources"][index - 1] == "built-in" else t["custom"]}'
            for index in range(1, 4)
        )
        details = "".join((
            detail_row(t["sources"], sources),
            detail_row(t["model"], summary[0]["model"]),
            detail_row(t["revision"], summary[0]["revision"]),
            detail_row(t["backend"], summary[0]["backend"]),
            detail_row(t["preprocessing"], (text_results[0].get("embedding_metadata") or {}).get("preprocessing_version")),
            detail_row(t["instruction"], run_config["instruction_mode"]),
            detail_row(t["config"], run_config["config_sha256"]),
            detail_row(t["norm"], norms),
        ))
        times = "".join(
            detail_row(
                f'{pair_title} {index}',
                f'{t["text"]} {seconds(pair["text_elapsed_ms"])} / '
                f'{t["image"]} {seconds(pair["image_elapsed_ms"])}',
            )
            for index, pair in enumerate(summary, 1)
        )
        timing_details = "".join(
            perf_row(f'{t["text"]} {index}', text, language)
            + perf_row(f'{t["image"]} {index}', image, language)
            for index, (text, image) in enumerate(
                zip(text_results, image_results, strict=True), 1
            )
        )
        return (
            f'<section lang="{language}" style="{SURFACE}font-size:16px;line-height:1.6;">'
            f'<h2 style="font-size:23px;font-weight:800;margin:0 0 10px;'
            f'color:#10253b !important;">{safe_text(t["heading"])}</h2>'
            '<div style="color:#166534 !important;font-size:15px;font-weight:800;'
            f'margin:0 0 10px;">✓ {safe_text(t["mock"] if mock else t["live"])}</div>'
            f'<p style="font-size:16px;{MUTED}">{safe_text(t["intro"])}</p>'
            f'<div style="display:flex;flex-wrap:wrap;gap:12px;">{metrics}</div>'
            f'<h3 style="font-size:19px;margin:24px 0 8px;color:#10253b !important;">'
            f'{safe_text(t["matrix"])}</h3>'
            + matrix_html(language)
            + f'<p style="font-size:14px;{MUTED}">{safe_text(t["matrix_note"])}</p>'
            f'<h3 style="font-size:19px;margin:24px 0 8px;color:#10253b !important;">'
            f'{safe_text(t["technical"])}</h3>'
            + details
            + f'<h3 style="font-size:19px;margin:24px 0 8px;color:#10253b !important;">'
            f'{safe_text(t["times"])}</h3>'
            + times
            + f'<p style="font-size:14px;{MUTED}">{safe_text(t["verified"])}</p>'
            + f'<p style="font-size:14px;{MUTED}">{safe_text(t["limits"])}</p>'
            + f'<p style="font-size:14px;{MUTED}">{safe_text(t["privacy"])}</p>'
            + f'<h3 style="font-size:19px;margin:24px 0 8px;color:#10253b !important;">'
            f'{safe_text(t["timing"])}</h3>'
            + timing_details
            + f'<p style="font-size:14px;{MUTED}">{safe_text(t["network"])}</p>'
            + "</section>"
        )


    _custom_result_slots = ", ".join(
        str(i + 1) for i, source in enumerate(run_config["image_sources"])
        if source == "custom"
    )
    _custom_input_report_notice = (
        '<aside role="note" style="border:2px solid #d97706;'
        'background:#fffbeb;color:#78350f;padding:12px 14px;'
        'border-radius:8px;margin-bottom:18px;">'
        '<div lang="en"><strong>Actual inference inputs</strong>'
        '<p>Custom image slots: '
        + safe_text(_custom_result_slots)
        + '. The built-in preview gallery is NOT the submitted image. '
        'Private image content and paths are not included in this report.</p></div>'
        '<div lang="vi"><strong>Ảnh thực sự dùng cho inference</strong>'
        '<p>Vị trí ảnh riêng: ' + safe_text(_custom_result_slots)
        + '. Gallery ảnh mẫu KHÔNG phải ảnh đã gửi. Báo cáo không hiển thị'
        ' nội dung hoặc đường dẫn ảnh riêng.</p></div></aside>'
        if _custom_result_slots else ""
    )

    report_html = (
        f'<div role="region" aria-label="Multimodal embedding demo results" '
        f'style="box-sizing:border-box;width:100%;max-width:960px;{SURFACE}'
        'font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;'
        'padding:24px 26px;border:1px solid #cbd5e1;border-radius:14px;'
        'font-size:16px;line-height:1.6;">'
        '<div style="font-size:14px;color:#475569 !important;'
        'letter-spacing:0.07em;font-weight:800;margin-bottom:20px;">'
        'MULTIMODAL · EMBEDDING DEMO</div>'
        + _custom_input_report_notice
        + report_section("en")
        + '<div style="border-top:2px solid #cbd5e1;margin:28px 0;"></div>'
        + report_section("vi")
        + "</div>"
    )

    try:
        from IPython.display import HTML, display
    except ImportError:
        for language in ("en", "vi"):
            t = LANGUAGES[language]
            print(f'{t["heading"]}: {t["intro"]}')
            for index, pair in enumerate(summary, 1):
                label = t["pair"] if run_config["default_pairs"] else t["sample"]
                print(f'{label} {index}: {pair["cosine_text_image"]:.4f}')
            print(t["matrix"], similarity_matrix)
            print(t["verified"], t["limits"], t["privacy"])
    else:
        display(HTML(report_html))
