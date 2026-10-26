# Chính sách bảo mật

🌐 Ngôn ngữ / Language: [English](SECURITY.md) | **Tiếng Việt**

Vui lòng báo cáo lỗ hổng riêng tư tới **i.am@dangkhoa.dev**, không đăng credentials hoặc dữ liệu riêng tư lên issue công khai.

## Xác thực và xử lý dữ liệu

- Hai endpoint `GET /health` và `GET /v1/models` được công khai, không cần Bearer token. Tất cả endpoint `/v1/*` còn lại yêu cầu xác thực Bearer (`VISION_EMBEDDINGS_API_TOKEN` hoặc API key client được cấp quyền); `GET /v1/ops/metrics` chỉ dành cho owner. Lưu token trong secret manager của deployment và không nhúng vào JavaScript phía trình duyệt.
- API nhận ảnh JPEG, PNG hoặc WebP dạng Base64, hoặc văn bản. API không tải dữ liệu từ URL đầu vào tùy ý.
- Giới hạn dung lượng và pixel giúp giảm lạm dụng. Gateway có kiểm soát request theo từng process, API key riêng theo client được lưu bằng digest SHA-256, cách ly async jobs, durable outstanding-job permits và Redis/Valkey rate/concurrency phân tán **tùy chọn** (HTTP 429). Đây **không phải** trần chi phí; cần Modal Workspace budget và spend limit. Xem [bảo mật kỹ thuật](docs/SECURITY.vi.md) và [vận hành](docs/OPERATIONS.vi.md).
- Byte ảnh tải lên được xử lý trong bộ nhớ. Không ghi log request body, ảnh gốc, token, header xác thực hoặc embedding nhạy cảm.
- Các endpoint tài liệu API tương tác (`/docs`, `/redoc` và `/openapi.json`) đã bị tắt trong source triển khai.

Deployment Modal dùng Secret riêng cho gateway (chứa Bearer token API) và GPU worker (chỉ chứa revision checkpoint công khai đã ghim). Không gắn credentials của gateway vào model worker. Khi nâng cấp deployment đang chạy, cần tạo Secret worker trước khi deploy lại; xem [Vận hành](docs/OPERATIONS.vi.md).

## Bảo vệ vận hành

- Không đưa model weights, dữ liệu đầu vào riêng tư hoặc credentials của deployment vào source repository. Ứng dụng production phải giữ API token ở backend; notebook cá nhân sử dụng runtime riêng, prompt ẩn và xóa output trước khi chia sẻ.
- Bật TLS, kiểm soát truy cập và giới hạn request/ngân sách trên nền tảng trước khi mở dịch vụ cho người dùng không tin cậy.
- Rà soát mã model upstream, dependencies và giấy phép trước khi redeploy hoặc phân phối artifact.
- Smoke test inference trên GPU thành công **không** chứng minh độ chính xác xác thực danh tính sinh trắc. Trường hợp sử dụng này phải được đánh giá độc lập, có sự đồng ý và rà soát quyền riêng tư.

Dự án cung cấp API inference phục vụ nghiên cứu, không phải chứng nhận bảo mật cho production.
