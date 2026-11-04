# Khắc phục lỗi và nghiệm thu Modal

| Lỗi | Cách xử lý |
|---|---|
| `401` | Kiểm tra Bearer token của Rails có khớp Secret `CF_API_TOKEN`; không in token vào log |
| `503` khi gọi GPU | Xem log riêng tư Modal: thiếu checkpoint, sai dependency hoặc lỗi VRAM |
| `Missing or mismatched checkpoint` | Kiểm tra SHA đã pin và marker Volume; tải lại model tương ứng |
| Qwen import error | Kiểm tra commit source Qwen, phiên bản Transformers và packages trong env Qwen |
| WeMM import error | Xác nhận `transformers==5.2.0` trong env WeMM độc lập |
| Request quá chậm | Có thể cold start hoặc phải đổi model; gom các ảnh thành batch theo model |
| CUDA OOM | Giảm batch/độ phân giải, không nạp đồng thời hai model, đo peak VRAM thật |
| `413/422` | Dữ liệu ảnh, kích thước, Base64 hoặc dimension không hợp lệ |

## Các điều kiện PASS cần có

1. `pytest -q` PASS trên CPU, không có secret hoặc weights trong Git.
2. Hai checkpoint được tải vào một Modal Volume và có commit SHA xác minh.
3. API nhận Bearer token, trả `/v1/models`, từ chối token không hợp lệ.
4. Qwen 8B trả vector 1024 chiều thực sự từ ảnh PNG hợp lệ.
5. WeMM 9B trả vector 1024 chiều thực sự từ cùng ảnh đó.
6. Thử Qwen → WeMM → Qwen trên một L40S và ghi VRAM peak + thời gian load.
7. Ảnh lỗi bị từ chối; lỗi nội bộ không lộ token, stack trace, đường dẫn nhạy cảm.
8. Chỉ công bố kết quả benchmark sau khi có đánh giá dữ liệu và giao thức test độc lập.

**Bản source hiện mới có bằng chứng kiểm thử offline, chưa nghiệm thu GPU.**
