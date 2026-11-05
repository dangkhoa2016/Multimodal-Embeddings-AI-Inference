# Khắc phục lỗi và kiểm tra nghiệm thu

🌐 Ngôn ngữ / Language: [English](TROUBLESHOOTING.md) | **Tiếng Việt**

## Lỗi thường gặp

| Triệu chứng | Nguyên nhân có thể | Hướng xử lý |
| --- | --- | --- |
| HTTP `401` | Thiếu hoặc sai Bearer token | Kiểm tra `VISION_EMBEDDINGS_API_TOKEN` phía server mà không in giá trị token |
| HTTP `413`/`422` | Đầu vào quá lớn hoặc schema không hợp lệ | Kiểm tra Base64, giới hạn ảnh, mã model, số chiều và batch size |
| HTTP `503` | Worker inference không sẵn sàng | Xem log được bảo vệ, revision checkpoint và bộ nhớ GPU khả dụng |
| Lỗi nạp model | Dependency hoặc checkpoint không khớp | Dựng lại môi trường đã pin và xác minh checkpoint |
| Latency cao | GPU cold start hoặc định tuyến model | Gom batch theo model và đo cold/warm request riêng |
| GPU hết bộ nhớ | Model activations hoặc đầu vào quá lớn | Giảm batch/kích thước ảnh; chỉ nạp một tiến trình model |
| Số chiều vector bất thường | Revision model hoặc giả định của client sai | Từ chối kết quả không khớp; không tự đệm hoặc cắt vector |

## Danh sách kiểm tra nghiệm thu

1. Chạy kiểm thử offline, lint và xác minh SHA256 của source.
2. Tải và xác minh hai checkpoint đầy đủ: `Qwen/Qwen3-VL-Embedding-8B` và `tencent/WeMM-Embedding-9B`.
3. Xác nhận request API không có token bị từ chối và request catalog có token hợp lệ thành công.
4. Xác nhận hai model trả về vector 1.024 chiều hữu hạn, chuẩn hóa trên cùng ảnh được phép sử dụng.
5. Xác nhận khả năng định tuyến model, phản hồi lỗi đã lọc và theo dõi bộ nhớ GPU khi tải tăng.
6. Đánh giá chất lượng tìm kiếm độc lập trước khi tuyên bố độ chính xác hoặc tính phù hợp với nhận diện sinh trắc.

Xem quy trình smoke test chức năng trong [Nghiệm thu](VERIFICATION.vi.md).
