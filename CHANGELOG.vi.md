# Lịch sử phát hành

🌐 Ngôn ngữ / Language: [English](CHANGELOG.md) | **Tiếng Việt**

## 1.0.0 — phiên bản mã nguồn đầu tiên

Phiên bản mã nguồn đầu tiên của Multimodal Embeddings AI Inference.

- Cung cấp API embedding ảnh và văn bản có xác thực, dùng Qwen3-VL-Embedding-8B và Tencent WeMM-Embedding-9B.
- Phục vụ mỗi checkpoint trên một GPU worker NVIDIA L40S chuyên biệt của Modal, với Model Volume riêng.
- Bao gồm hướng dẫn cài đặt, quản lý checkpoint, triển khai, bảo mật, ví dụ HTTP client không phụ thuộc framework và tài liệu song ngữ.
- Cung cấp quy trình smoke test HTTPS có xác thực, có thể tái lập cho cả hai model, bao gồm kiểm tra checkpoint revision và chuẩn hóa vector.
- Cung cấp CI Python 3.11/3.12, kiểm thử hợp đồng API, lint và manifest SHA256 của source.
- Bao gồm Demo Notebook song ngữ với sáu request và ảnh mẫu xem trước không cần token, cùng notebook benchmark hiệu năng riêng chỉ chạy khi chủ động bật.
- Hỗ trợ đo warm latency cân bằng, cold GPU tùy chọn, định tuyến worker luân phiên, concurrency giới hạn và batch hai model. Không công bố thống kê benchmark đo được tại đây.

Để kiểm tra deployment của mình, xem [Kiểm thử và nghiệm thu](docs/VERIFICATION.vi.md). Phiên bản mã nguồn này không tuyên bố đã đo độ chính xác nhận diện khuôn mặt hoặc hiệu năng cấp production.
