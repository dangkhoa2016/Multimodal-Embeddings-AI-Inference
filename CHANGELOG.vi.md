# Lịch sử phát hành

🌐 Ngôn ngữ / Language: [English](CHANGELOG.md) | **Tiếng Việt**

## 1.0.1

Bản vá trình bày song ngữ, tính minh bạch của release và tài liệu dành cho cộng đồng.

- Tách tiếng Anh và tiếng Việt trong các output console, nhãn ảnh và bảng kết quả notebook, không thay đổi logic inference hoặc tiêu chí xác nhận benchmark.
- Bổ sung regression tests kiểm tra cấu trúc output song ngữ và bảo toàn các tùy chọn GPU an toàn mặc định.
- Cải thiện điều hướng README, làm rõ thông số ví dụ và báo cáo benchmark chỉ mang tính minh họa.
- Bổ sung công cụ tạo checksum riêng cho các release assets tải về, không thay thế manifest nguồn của repository.
- Giữ nguyên các kết quả benchmark GPU lịch sử; bản vá này không tuyên bố có phép đo GPU mới.

## 1.0.0

Embedding ảnh và văn bản, công cụ triển khai cùng notebook để xây dựng ứng dụng đa phương thức.

- Cung cấp API embedding ảnh và văn bản có xác thực, dùng Qwen3-VL-Embedding-8B và Tencent WeMM-Embedding-9B.
- Phục vụ mỗi checkpoint trên một GPU worker NVIDIA L40S chuyên biệt của Modal, với Model Volume riêng.
- Bao gồm hướng dẫn cài đặt, quản lý checkpoint, triển khai, bảo mật, ví dụ HTTP client không phụ thuộc framework và tài liệu song ngữ.
- Hỗ trợ request đơn và batch, instruction Qwen tùy chỉnh cùng metadata phản hồi để theo dõi model và cấu hình mã hóa.
- Bổ sung GPU job async có chống gửi trùng, CPU coordinator, polling kết quả và notebook demo song ngữ cho phép tiếp tục mà không lặp GPU POST.
- Bổ sung heartbeat, cảnh báo job quá hạn không sửa trạng thái thật, giữ ổn định request ID khi đổi token, tra cứu chỉ đọc job theo namespace client và phân biệt latency/cache hit, không tự retry GPU.
- Hỗ trợ API key riêng từng client, durable outstanding-job admission và Redis/Valkey distributed rate-limit, sync concurrency tùy chọn.
- Thêm công cụ quản lý key offline, owner-only metrics không lộ dữ liệu, fail-closed khi backend lỗi và kiểm thử audit bảo mật.
- Giới hạn thời gian chờ inference trên Modal bằng deadline có yêu cầu hủy, giữ health phản hồi khi có tải và từ chối credential sai định dạng an toàn.
- Chặn peer đang bị khóa trước khi kiểm tra credential và tách biệt xác thực/giới hạn body giữa các API app được mount.
- Lược bỏ thông tin nhạy cảm trong lỗi kết nối notebook và lưu lượt benchmark tuần tự trước khi gửi.
- Bao gồm Demo Notebook song ngữ với sáu request và ảnh mẫu xem trước không cần token, cùng notebook benchmark hiệu năng riêng chỉ chạy khi chủ động bật.
- Hỗ trợ đo warm latency cân bằng, cold GPU tùy chọn, định tuyến worker luân phiên, concurrency giới hạn và batch hai model. Notebook xuất thống kê đo được theo deployment và cấu hình của người vận hành.

Xem [Vận hành](docs/OPERATIONS.vi.md) để triển khai dịch vụ và [hướng dẫn notebook](docs/NOTEBOOK_GUIDE.vi.md) để chạy các ví dụ.
