# Kết nối Rails 8 — chỉ gọi server-side

Rails là ứng dụng chính quản lý dataset, benchmark, job và vector PostgreSQL; **không** tải PyTorch/CUDA hay model weights.

## Cấu hình

Dùng `bin/rails credentials:edit` để lưu:

```yaml
compactfaces_inference:
  base_url: "https://URL-THUC-TE.modal.run"
  token: "TOKEN-BI-MAT-KHONG-DUOC-COMMIT"
```

Token lấy từ file local được bảo vệ `~/.config/compactfaces-ai-inference/modal-secret.env`, không đưa vào JavaScript/frontend. Mẫu Ruby `Net::HTTP` đầy đủ ở [RAILS_INTEGRATION.md](RAILS_INTEGRATION.md).

Trình tự: Rails tải ảnh hợp lệ → mã hóa Base64 → POST `/v1/embeddings` với `Authorization: Bearer TOKEN` → nhận vector + SHA256 + revision → lưu pgvector → đánh giá similarity bằng Rails.

## Thiết kế database/job đề xuất

- Bảng: `inference_providers`, `embedding_models`, `face_images`, `face_embeddings`, `benchmark_runs`, `benchmark_items`.
- Khóa duy nhất theo SHA256 ảnh + model + revision + dimension + preprocessing version để retry không sinh bản ghi trùng.
- Rails Solid Queue chia 935 ảnh thành các batch nhỏ; lưu checkpoint để resume; retry theo exponential backoff khi Modal cold start hoặc timeout.
- Không trộn vector của Qwen và WeMM; hai model có không gian embedding độc lập.
- Kiểm tra HTTPS và giới hạn host của API URL do admin cấu hình để tránh Rails-side SSRF.
- Tuyệt đối không ghi log Bearer token, ảnh Base64 hay vector nhạy cảm.

**Chú ý:** `GET /v1/health` chỉ xác nhận FastAPI gateway đang phản hồi; không chứng minh GPU hay model đã sẵn sàng. Việc `rank-1/top-5` có ý nghĩa đòi hỏi đánh giá bộ CompactFaces và tập test thích hợp, không được suy ra chỉ từ metadata identity.
