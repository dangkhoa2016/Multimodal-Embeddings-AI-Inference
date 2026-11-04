# CompactFaces AI Inference

**REST API embedding đa phương thức sử dụng Qwen3-VL-Embedding-8B và Tencent WeMM-Embedding-9B, triển khai trên một NVIDIA L40S của Modal.** Dự án phục vụ [CompactFaces Synthetic Face Dataset](https://github.com/dangkhoa2016/CompactFaces-Synthetic-Face-Pipeline) và ứng dụng đánh giá Rails 8 dự kiến phát triển.

**Ngôn ngữ:** [English](README.md) | Tiếng Việt

> **Trạng thái v0.2.1:** Đã deploy trên Modal L40S; Qwen3-VL-Embedding-8B và WeMM-Embedding-9B **đều PASS inference ảnh thật**, trả vector 1.024 chiều hữu hạn và chuẩn hóa L2. Gọi HTTPS POST embedding thật, xác thực Gateway (401/200), Ruff và kiểm thử offline đều PASS. Đây là bằng chứng API/model chạy được, **không phải kết quả đánh giá độ chính xác nhận diện khuôn mặt**. Không dùng mock vector để báo cáo AI.

**Bằng chứng nghiệm thu trực tiếp:** [Kết quả REST/GPU ngày 04/11/2026](docs/VERIFICATION.vi.md).

## Chức năng chính

- Cung cấp các endpoint có Bearer token: `GET /v1/health`, `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/embeddings`, `POST /v1/embeddings/batches`.
- Nhận ảnh dạng Base64 hoặc văn bản UTF-8; trả embedding đã chuẩn hóa L2, SHA256 input, model ID, revision và thời gian xử lý.
- Một FastAPI gateway CPU và **một Modal GPU worker `SharedL40SGpu` duy nhất**, cấu hình `gpu="L40S"`, `max_containers=1`. Worker chỉ nạp **một model vào VRAM ở mỗi thời điểm** bằng các subprocess riêng.
- Lưu hai checkpoint trên **một Modal Volume** tên `compactfaces-ai-weights`.
- Tài liệu song ngữ, script setup, tạo secret, tải model, deploy và smoke tests, CI GitHub Actions.

Đây là hệ thống **nghiên cứu tìm kiếm ảnh bằng embedding**, không phải hệ thống xác thực sinh trắc học đã được công nhận. Nhãn danh tính tổng hợp CompactFaces chưa chứng minh năm ảnh của cùng identity luôn nhất quán về sinh trắc.

## Hai model chính thức

| API model ID | Checkpoint Hugging Face | Kích thước | Chiều vector |
|---|---|---:|---|
| `qwen3-vl-embedding-8b` | [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 8B | 64–4096 |
| `wemm-embedding-9b` | [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) | 9B | 64, 128, 256, 512, 1024, 2048, 4096 |

Tencent **không có checkpoint WeMM-Embedding-8B chính thức** trong danh mục hiện tại, nên chọn bản gần nhất lớn hơn là 9B. Cả hai mặc định đầu ra **1024 chiều**. Không so sánh trực tiếp vector của hai model với nhau vì chúng thuộc các không gian embedding riêng biệt.

## Kiến trúc

```text
Rails 8 (ứng dụng dự kiến)
    | HTTPS + Bearer token
    v
FastAPI Gateway (CPU)
    |
    v
SharedL40SGpu (1 L40S, 48 GB, max_containers=1)
    |
    +-- Qwen 8B subprocess (Python environment riêng)
    |
    +-- WeMM 9B subprocess (Python environment riêng)
        Chỉ MỘT subprocess model chạy tại một thời điểm
    |
    v
Modal Volume: compactfaces-ai-weights
```

Khi chuyển model, worker dừng tiến trình cũ và tạo tiến trình mới. Cách này giúp tránh phải giữ 8B + 9B trong VRAM cùng lúc, nhưng sẽ có thêm thời gian load model. Việc GPU thật có đủ VRAM, thời gian khởi động và latency vẫn cần nghiệm thu trên Modal.

## Điều kiện cần

- Máy tính có Python **3.11+**, Git, Bash (Windows nên dùng WSL).
- Tài khoản Modal có quyền dùng GPU L40S và đủ ngân sách/quota.
- Kết nối đến Hugging Face để tải checkpoint một lần.
- Máy đáng tin cậy để giữ API token; **không cần GPU cá nhân** cho bước setup và CI.

## Bước A — chạy mock API và kiểm thử offline

```bash
bash scripts/setup.sh
export CF_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
CF_BACKEND=mock .venv/bin/uvicorn compactfaces_ai_inference.api:local_app \
  --factory --host 127.0.0.1 --port 8000
```

Ở terminal thứ hai (dùng chung token):

```bash
curl -H "Authorization: Bearer $CF_API_TOKEN" http://127.0.0.1:8000/v1/models
python scripts/smoke_api.py --base-url http://127.0.0.1:8000 \
  --model qwen3-vl-embedding-8b --text "synthetic portrait" --dimension 1024
```

**MOCK là số giả lập có thể tái tạo, không phải embedding AI thật.** Không dùng để báo cáo độ chính xác.

## Bước B — triển khai Modal L40S

### 1. Cài Modal CLI, đăng nhập

```bash
source .venv/bin/activate
python -m pip install -e '.[modal]'
modal setup
```

### 2. Tải hai model vào cùng Volume (chạy bằng CPU)

```bash
modal run modal_models.py --selection both
modal volume ls compactfaces-ai-weights /checkpoints
```

Lệnh tải sẽ in ra **commit SHA Hugging Face thực tế** của Qwen và WeMM. Hãy giữ hai SHA này để tạo secret. Source Git không chứa weights; cần đủ dung lượng Volume và băng thông khi tải.

### 3. Tạo Modal Secret và API token

Thay hai giá trị bên dưới bằng SHA thực (40 ký tự) từ bước 2:

```bash
python scripts/create_modal_secret.py \
  --qwen8-revision <SHA_40_KY_TU_CUA_QWEN> \
  --wemm9-revision <SHA_40_KY_TU_CUA_WEMM>
```

Script tự tạo `CF_API_TOKEN` ngẫu nhiên, lưu cục bộ ở `~/.config/compactfaces-ai-inference/modal-secret.env` với quyền **0600**, rồi đăng ký Modal Secret tên `compactfaces-api` với ba biến:

- `CF_API_TOKEN` — Bearer token của ứng dụng.
- `CF_QWEN8_REVISION` — commit SHA bất biến của Qwen.
- `CF_WEMM9_REVISION` — commit SHA bất biến của WeMM.

Script không in token lên màn hình. Không commit/chia sẻ file chứa token. Khi cần cập nhật secret và giữ token cũ, chạy lại với `--force`.

### 4. Deploy REST API và GPU worker

```bash
modal deploy modal_app.py
```

Modal sẽ hiển thị HTTPS endpoint thực tế. Sao chép chính xác URL đó. Việc deploy thành công **không đồng nghĩa** đã test load model thật, vì worker sẽ nạp khi có request đầu tiên.

### 5. Test hai model thật

Nạp token cục bộ (không cần in token ra màn hình):

```bash
export CF_API_TOKEN="$(python3 - <<'PY'
from pathlib import Path
p = Path.home()/'.config/compactfaces-ai-inference/modal-secret.env'
print(next(x.partition('=')[2] for x in p.read_text().splitlines() if x.startswith('CF_API_TOKEN=')))
PY
)"
```

Sau đó gửi ảnh PNG/JPEG mà bạn có quyền sử dụng:

```bash
python scripts/smoke_both.py \
  --base-url 'https://URL-THUC-TE-DUOC-MODAL-CUNG-CAP.modal.run' \
  --image /duong/dan/anh_duoc_phep.png --dimension 1024
```

Script sẽ chạy hai model lần lượt, kiểm tra model ID, SHA256 của ảnh, kích thước vector, L2 norm và `backend=modal`. Kết quả thật chỉ được xác nhận sau khi hoàn thành lệnh trên với dữ liệu thật. Cần đo thêm peak VRAM, cold/warm latency và tình huống đổi model qua lại.

### 6. Kết nối ứng dụng Rails 8

Rails chỉ cần lưu **API URL + `CF_API_TOKEN`** trong Rails Credentials/secret manager và gọi HTTPS server-side. Không nhúng token vào JavaScript hay HTML. Xem [hướng dẫn Rails](docs/RAILS_INTEGRATION.vi.md).

## API ví dụ

```bash
curl -X POST 'https://URL-THUC-TE.modal.run/v1/embeddings' \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"wemm-embedding-9b","text":"a smiling face","dimension":1024}'
```

Ảnh gửi bằng trường `image_base64` (Base64 thuần, không thêm `data:image/...`). Kết quả chứa `vector`, `dimension`, `model`, `repository`, `revision`, `backend`, `input_sha256`, `elapsed_ms`. Xem [API Contract](docs/API_CONTRACT.md). Bản này dùng **Bearer token tầng ứng dụng**, chưa bật Modal Proxy Authentication để giữ Rails đơn giản với một token.

## Vận hành, chi phí và giới hạn

- GPU L40S 48 GB, worker tối đa một container, scale về 0 khi idle. **Không** có nghĩa là tổng chi phí GPU bị giới hạn theo tháng; cần thiết lập cảnh báo ngân sách Modal.
- Đổi model có thể khiến thời gian xử lý tăng đáng kể do phải unload/load. Hãy gom batch theo model, thay vì đổi sau mỗi ảnh.
- API hiện xử lý đồng bộ; mặc định tối đa tám item/batch. Với 935 ảnh CompactFaces, Rails/Solid Queue nên chia thành nhiều job nhỏ có retry và resume.
- Không huấn luyện model. Không tuyên bố độ chính xác Face Recognition khi chưa đánh giá bộ dữ liệu và giao thức benchmark độc lập.
- CPU API và Volume cũng có thể phát sinh chi phí riêng ngoài GPU.

Xem [hướng dẫn công bố lên GitHub](docs/PUBLISH_GITHUB.vi.md) để tạo repository từ xa.

Tài liệu khác: [Shared Volume](docs/SHARED_VOLUME.md), [khắc phục lỗi](docs/TROUBLESHOOTING.vi.md), [bảo mật](SECURITY.md), [giấy phép](LICENSE-NOTES.md), [Contributing](CONTRIBUTING.md), [Changelog](CHANGELOG.md).

**Phạm vi license:** mã nguồn/tài liệu do dự án sở hữu dùng MIT. Model weights và thư viện upstream có giấy phép riêng. Dữ liệu CompactFaces công bố riêng với phạm vi CC BY-NC 4.0; MIT cho code không đồng nghĩa có quyền thương mại đối với ảnh.

**Tác giả:** Đăng Khoa · Liên hệ: `i.am@dangkhoa.dev`.

Xem thêm: [Vận hành Modal và nghiệm thu](docs/OPERATIONS.vi.md).
