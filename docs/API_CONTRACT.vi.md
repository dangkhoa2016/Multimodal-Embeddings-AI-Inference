# Hợp đồng API

🌐 Ngôn ngữ / Language: [English](API_CONTRACT.md) | **Tiếng Việt**

Sử dụng endpoint được cấp cho **deployment của chính bạn**; repository không cung cấp inference endpoint công khai dùng chung.

Mọi endpoint `/v1/*` yêu cầu `Authorization: Bearer <VISION_EMBEDDINGS_API_TOKEN>`. Sử dụng TLS khi không chạy trên localhost và giữ token phía server.

## Các endpoint chỉ đọc

- `GET /v1/health`: trạng thái gateway và version, **không** khẳng định model GPU sẵn sàng.
- `GET /v1/models`: các mã model, số chiều được hỗ trợ và revision checkpoint đã cố định.
- `GET /v1/models/{model_id}`: metadata của một model hoặc HTTP 404.

## POST /v1/embeddings

Chỉ gửi một trong hai trường `image_base64` (PNG/JPEG/WebP) hoặc `text` (UTF-8). Bắt buộc có `model`; có thể thêm `dimension` và `request_id` (tối đa 128 ký tự). Đối với text, schema loại bỏ khoảng trắng ở đầu/cuối trước inference; input_sha256 được tính từ byte UTF-8 sau khi strip, không phải chuỗi gốc. Ví dụ, hai dấu cách ở hai đầu hello world không làm đổi hash. Với ảnh, SHA256 được tính từ byte ảnh giải mã Base64, không áp dụng quy tắc text.

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "a landscape photograph",
  "dimension": 1024,
  "request_id": "request-example-001"
}
```

Phản hồi minh họa (vector được rút gọn **không** phải kết quả model thật):

```json
{
  "model": "qwen3-vl-embedding-8b",
  "repository": "Qwen/Qwen3-VL-Embedding-8B",
  "revision": "<PINNED_MODEL_REVISION>",
  "backend": "modal",
  "dimension": 1024,
  "vector": [0.001, -0.002],
  "input_sha256": "<SHA256_OF_INPUT>",
  "request_id": "request-example-001",
  "elapsed_ms": 123.0,
  "timing": {"model_state": "warm", "gpu_worker_first_request": false, "model_load_ms": 0.0, "gpu_encode_ms": 76.0, "gpu_worker_ms": 80.0, "gpu_dispatch_overhead_ms": 3.0, "batch_size": 1},
  "embedding_metadata": {"preprocessing_version": "qwen3-vl-adapter-v1", "instruction_sha256": "<HASH>", "config_sha256": "<HASH>"},
  "normalized": true
}
```

Vector phản hồi thật có đúng `dimension` phần tử hữu hạn và chuẩn L2 xấp xỉ 1. `elapsed_ms` thể hiện thời gian xử lý inference của request hoặc nhóm batch, **không** phải độ trễ mạng từ đầu đến cuối.

## Instruction Qwen tùy chọn và metadata cấu hình vector

Chỉ với **Qwen3-VL-Embedding-8B**, trường `instruction` tùy chọn mô tả tác vụ trên một dòng, dài 1–512 ký tự. Giá trị rỗng, chứa ký tự điều khiển hoặc quá dài trả HTTP 422. Danh sách model khai báo `supports_custom_instruction`.

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "How can I reset my password?",
  "dimension": 1024,
  "instruction": "Retrieve help articles that answer the user's question."
}
```

Khi không truyền instruction, dịch vụ giữ nguyên instruction tìm kiếm thư viện ảnh trước đây của Qwen và vector tương ứng. WeMM không nhận instruction tùy chỉnh; nếu truyền sẽ nhận HTTP 422.

Đối tượng `embedding_metadata` bổ sung trong response gồm `preprocessing_version`, `instruction_sha256` (null đối với WeMM) và `config_sha256`. Response không lặp lại instruction gốc. Trường `input_sha256` cũ chỉ băm dữ liệu đầu vào, nên không thể dùng một mình làm khóa cache vector.

Phân tách chỉ mục theo model, revision, số chiều và phiên bản tiền xử lý. Dùng cặp `(input_sha256, config_sha256)` để cache từng embedding. Instruction của query và document có thể khác nhau có chủ đích; hai mã băm instruction không bắt buộc giống nhau để so cosine. Cần re-embed các mục chỉ mục bị ảnh hưởng nếu cấu hình mã hóa thay đổi.

## Chi tiết thời gian hiệu năng

Phản hồi inference đã xác thực vẫn giữ elapsed_ms (thời gian chờ inference phía gateway), đồng thời bổ sung timing khi Modal GPU worker hỗ trợ đo lường. Client cũ có thể bỏ qua timing; backend mock/local trả về null.

- model_state: initial_load (lần nạp model đầu quan sát trên worker), model_reload (khởi động lại tiến trình đã dừng) hoặc warm (tái sử dụng tiến trình). gpu_worker_first_request đánh dấu request đầu tiên quan sát được trong GPU container đó. gpu_worker_id là mã ngẫu nhiên gồm 32 ký tự hex để nhận diện vòng đời của GPU worker; không phải hostname/container ID hoặc bằng chứng autoscaler. Deployment cũ có thể không trả trường này.
- model_load_ms: thời gian chạy subprocess, khởi tạo model/checkpoint và thay model. Giá trị bằng 0 khi warm. gpu_encode_ms: thời gian tiền xử lý và tính vector theo batch trong subprocess.
- gpu_worker_ms: thời gian thực thi Modal GPU method; gpu_dispatch_overhead_ms: thời gian khứ hồi từ gateway đến worker trừ thời gian worker, bao gồm điều phối, truyền dữ liệu và có thể cả khởi tạo container. Đây **không phải cold start thuần túy**.
- batch_size cho biết số input dùng chung các phép đo. Các thời gian dùng để chẩn đoán, không phải cam kết độ trễ; muốn đo tổng thời gian client phải sử dụng bộ đếm riêng trên Kaggle, Colab hoặc Rails.

Tất cả giá trị thời gian tính bằng mili giây. Request đầu được quan sát có thể đã warm nếu dịch vụ đang chạy; các client đồng thời có thể ảnh hưởng trạng thái và thời gian.

## Mã model và số chiều

- [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B): `qwen3-vl-embedding-8b`, số chiều nguyên từ 64 đến 4096.
- [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B): `wemm-embedding-9b`, số chiều 64, 128, 256, 512, 1024, 2048 hoặc 4096.

Bỏ qua `dimension` hoặc gửi `null` sẽ chọn mặc định 1024 chiều. Gửi `0` hoặc số chiều không được hỗ trợ sẽ nhận HTTP 422.

Vector từ các mã model, revision, số chiều và cấu hình tiền xử lý khác nhau không thể sử dụng thay thế cho nhau.

## POST /v1/embeddings/batches

```json
{"items": [
  {"model":"qwen3-vl-embedding-8b", "text":"one", "dimension":1024},
  {"model":"wemm-embedding-9b", "text":"two", "dimension":1024}
]}
```

Trả về `{"count": 2, "items": [...]}` theo thứ tự đầu vào. Gateway gom request theo model và gửi các nhóm model độc lập tới các GPU worker riêng. Giới hạn batch mặc định là **8** (schema tối đa **32**).

## Lỗi và giới hạn đầu vào

- HTTP 401: thiếu hoặc sai Bearer token.
- HTTP 404: không tìm thấy mã model ở route metadata.
- HTTP 413: request body hoặc ảnh vượt giới hạn cấu hình.
- HTTP 422: Base64/ảnh, số chiều, model, batch size hoặc schema không hợp lệ.
- HTTP 503: worker inference không sẵn sàng; client không nhận chi tiết lỗi nội bộ.

Giới hạn mặc định: **5 MiB** mỗi ảnh, **12 triệu** pixel sau giải mã, **8** phần tử mỗi batch và **8192** ký tự văn bản. Ảnh phải là JPEG, PNG hoặc WebP; API không tải ảnh từ URL bên ngoài.

## Tương thích

Phiên bản đường dẫn vẫn là `/v1`. Cần tạo lại chỉ mục khi revision model, quy trình tiền xử lý hoặc số chiều thay đổi; cache embedding bằng cặp `(input_sha256, embedding_metadata.config_sha256)`.
