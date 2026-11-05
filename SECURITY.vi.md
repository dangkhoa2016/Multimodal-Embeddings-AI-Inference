# Chính sách bảo mật

🌐 Ngôn ngữ / Language: [English](SECURITY.md) | **Tiếng Việt**

Vui lòng báo cáo lỗ hổng riêng tư tới **i.am@dangkhoa.dev**, không đăng credentials hoặc dữ liệu riêng tư lên issue công khai.

## Xác thực và xử lý dữ liệu

- Mọi endpoint `/v1/*` yêu cầu Bearer token của ứng dụng (`VISION_EMBEDDINGS_API_TOKEN`). Lưu trong secret manager của deployment và không nhúng vào JavaScript phía trình duyệt.
- API nhận ảnh JPEG, PNG hoặc WebP dạng Base64, hoặc văn bản. API không tải dữ liệu từ URL đầu vào tùy ý.
- Giới hạn dung lượng và pixel giúp giảm lạm dụng tài nguyên. Dịch vụ chưa tích hợp rate limiting, quota theo client hoặc trần chi phí.
- Byte ảnh tải lên được xử lý trong bộ nhớ. Không ghi log request body, ảnh gốc, token, header xác thực hoặc embedding nhạy cảm.
- Các endpoint tài liệu API tương tác (`/docs`, `/redoc` và `/openapi.json`) đã bị tắt trong source triển khai.

## Bảo vệ vận hành

- Không đưa model weights, dữ liệu đầu vào riêng tư hoặc credentials của deployment vào source repository. Ứng dụng production phải giữ API token ở backend; notebook cá nhân sử dụng runtime riêng, prompt ẩn và xóa output trước khi chia sẻ.
- Bật TLS, kiểm soát truy cập và giới hạn request/ngân sách trên nền tảng trước khi mở dịch vụ cho người dùng không tin cậy.
- Rà soát mã model upstream, dependencies và giấy phép trước khi redeploy hoặc phân phối artifact.
- Smoke test inference trên GPU thành công **không** chứng minh độ chính xác xác thực danh tính sinh trắc. Trường hợp sử dụng này phải được đánh giá độc lập, có sự đồng ý và rà soát quyền riêng tư.

Dự án cung cấp API inference phục vụ nghiên cứu, không phải chứng nhận bảo mật cho production.
