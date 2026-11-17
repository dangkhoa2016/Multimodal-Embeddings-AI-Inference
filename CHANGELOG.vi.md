# Thông tin phát hành

🌐 Ngôn ngữ / Language: [English](CHANGELOG.md) | **Tiếng Việt**

## 1.0.0 — Bản phát hành đầu tiên

**Multimodal Embeddings AI Inference 1.0.0** cung cấp dịch vụ tạo embedding ảnh và văn bản, cùng công cụ triển khai, ví dụ tích hợp và notebook song ngữ cho ứng dụng tìm kiếm đa phương thức.

### API và mô hình

- Hỗ trợ [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B), với không gian embedding độc lập cho từng model.
- Cung cấp API HTTPS có xác thực cho ảnh, văn bản, request đơn và batch; hỗ trợ instruction Qwen tùy chỉnh và kích thước vector theo khả năng của từng model.
- Trả về metadata để ứng dụng kiểm tra model, cấu hình embedding và thông tin xử lý.

### Bảo mật và xử lý bất đồng bộ

- Hỗ trợ GPU job bất đồng bộ với `request_id`, polling trạng thái, khôi phục kết quả chỉ đọc và các biện pháp hạn chế gửi trùng GPU request.
- Có thể cấu hình API key riêng từng client, giới hạn request, kiểm soát concurrency và Redis/Valkey chung cho nhiều gateway.
- Cung cấp công cụ quản lý key offline, metrics chỉ dành cho người vận hành và biện pháp bảo vệ khi gặp credential không hợp lệ, lỗi backend hoặc job quá hạn.

### Notebook và tài liệu tích hợp

- **Demo Notebook EN/VI:** xem trước ba cặp ảnh và văn bản mẫu mà không cần token; khi chủ động bật inference, hiển thị ma trận cosine 3 × 3 từ kết quả thực tế.
- **Benchmark Notebook EN/VI:** đo warm latency, cold-start tùy chọn, định tuyến luân phiên, concurrency có giới hạn và batch hỗn hợp; xuất báo cáo JSON theo deployment và cấu hình sử dụng.
- Notebook đóng vai trò client chạy trên CPU; các lệnh gọi inference có thể phát sinh chi phí phải được bật chủ động.
- Bao gồm ví dụ HTTP client và hướng dẫn tích hợp, triển khai, bảo mật, xử lý sự cố bằng tiếng Anh và tiếng Việt.

### Triển khai và kiểm chứng

- Hỗ trợ triển khai Modal với GPU worker NVIDIA L40S riêng cho từng model và Model Volume lưu checkpoint.
- Bao gồm chế độ mock cục bộ không cần GPU, kiểm thử tự động, kiểm tra bảo mật offline và manifest SHA-256 cho mã nguồn và tài sản phát hành.

**Phạm vi số liệu benchmark:** Kết quả GPU đính kèm phản ánh môi trường và cấu hình đo cụ thể; không bảo đảm SLA production hoặc chất lượng embedding trên dữ liệu của người dùng.

**Bắt đầu:** Xem [hướng dẫn vận hành](docs/OPERATIONS.vi.md), [hướng dẫn notebook](docs/NOTEBOOK_GUIDE.vi.md) và [ghi chú giấy phép](LICENSE-NOTES.vi.md).
