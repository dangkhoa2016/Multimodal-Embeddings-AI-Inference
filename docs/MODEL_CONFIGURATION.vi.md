# Cấu hình tùy chỉnh theo từng model

🌐 Ngôn ngữ / Language: [English](MODEL_CONFIGURATION.md) | **Tiếng Việt**

API embedding hỗ trợ object JSON `model_config` tùy chọn theo từng model và từng request. Khi không truyền, dịch vụ giữ nguyên mặc định (1024 chiều, instruction Qwen như trước). Các trường `dimension` và `instruction` ở cấp request cũ vẫn tương thích.

Gọi `GET /v1/models/{model_id}` có xác thực, đọc `model_config_schema.properties` để lấy tên tham số, kiểu dữ liệu, giới hạn và giá trị mặc định. Client có thể tự tạo form mà không cần hard-code.

## Danh sách tùy chọn

| Tùy chọn | Qwen/Qwen3-VL-Embedding-8B | tencent/WeMM-Embedding-9B |
| --- | --- | --- |
| `dimension` | Số nguyên 64–4096 | 64, 128, 256, 512, 1024, 2048, 4096 |
| `instruction` | Văn bản một dòng, 1–512 ký tự | Không hỗ trợ |
| `normalize` | Boolean, mặc định true | Boolean, mặc định true |
| `max_length` | Giới hạn token 64–32768, mặc định 8192 | Adapter hiện tại chưa hỗ trợ |
| `min_pixels`, `max_pixels` | Giới hạn pixel khi xử lý ảnh | Giới hạn pixel khi xử lý ảnh |

Các tùy chọn pixel **chỉ áp dụng cho ảnh**. Giới hạn token/pixel cao có thể làm tăng VRAM và chi phí; request hợp lệ không có nghĩa GPU chắc chắn xử lý được. API không chấp nhận tham số tùy ý của model loader, dtype, đường dẫn file, checkpoint hay hạ tầng GPU.

### Request đơn

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "How can I reset my password?",
  "model_config": {
    "dimension": 2048,
    "instruction": "Retrieve helpful support articles.",
    "max_length": 4096,
    "normalize": true
  }
}
```

### Batch có cấu hình riêng

```json
{
  "items": [
    {"model": "qwen3-vl-embedding-8b", "text": "a landscape", "model_config": {"dimension": 2048, "instruction": "Retrieve similar photographs."}},
    {"model": "wemm-embedding-9b", "text": "a landscape", "model_config": {"dimension": 1024, "normalize": true}}
  ]
}
```

Cấu hình cùng giá trị ở hai vị trí được chấp nhận; khác giá trị, sai kiểu, option không hỗ trợ hoặc dimension sai sẽ nhận HTTP 422 **trước khi gửi GPU**. Cả endpoint đồng bộ và async đều nhận `model_config`; cơ chế idempotency theo `request_id` gắn với cấu hình có hiệu lực.

Nếu `normalize=false`, vector vẫn hữu hạn nhưng **không bảo đảm** có chuẩn L2 bằng 1; response ghi `normalized=false`. Mã `embedding_metadata.config_sha256` phản ánh các tùy chọn khác mặc định. Cần cache bằng `(input_sha256, config_sha256)` và tách chỉ mục theo cấu hình mã hóa.

Demo/Benchmark Notebook độc lập tiếp tục dùng cấu hình mặc định nhằm bảo đảm tái lập kết quả. Client HTTP có xác thực có thể gửi tùy chỉnh nâng cao; notebook không tự cho phép GPU job tính phí. Đường đi option đã được kiểm thử CPU/mock và CI, việc công bố số liệu hiệu năng hoặc chất lượng cần có phép đo GPU thực tế và dữ liệu nguồn tương ứng.
