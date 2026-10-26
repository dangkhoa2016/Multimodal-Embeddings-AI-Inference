# Khắc phục lỗi

🌐 Ngôn ngữ / Language: [English](TROUBLESHOOTING.md) | **Tiếng Việt**

## Chẩn đoán multi-client và async admission

Nếu client A nhận HTTP 404 khi xem job của client B, đây là cơ chế cách ly tenant có chủ đích, không chứng minh job đã mất. Khi xoay API key phải giữ client ID cũ; xóa toàn bộ digest sẽ thu hồi quyền poll. HTTP 429 với `Async outstanding job capacity reached` nghĩa là hết durable permit toàn cục/riêng client, không phải giới hạn 4 request inflight của gateway. HTTP 503 với `Job registry temporarily unavailable` hoặc `Job admission temporarily unavailable` phải xem là **bất định**: kiểm tra lại bằng request/job ID gốc sau khi khôi phục và đối soát log vận hành; không tạo ID khác hay tự động gửi lại. Permit bị kẹt chỉ nên phục hồi bằng đối soát của người vận hành, không xóa chỉ vì heartbeat đã cũ. Marker `job:start:v1:` và `permit:release:` là cơ chế bảo vệ once-only có chủ đích. Không bao giờ xóa toàn bộ Modal Dict để giải phóng slot vì Dict còn chứa idempotency và embedding vectors. Dùng `scripts/manage_client_keys.py` và `scripts/client_registry_bundle.py` để xoay key offline, kiểm tra và áp dụng Secret bằng thao tác thủ công. Khi thay gateway Secret trên Modal, phải bảo toàn `VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON`; helper tạo Secret cơ bản không tự bảo toàn registry bên ngoài.

## Chẩn đoán shared Redis/Valkey admission

Khi bật distributed rate-limit, POST inference có thể trả **HTTP 429** với
`Distributed embedding budget exceeded`,
`Distributed synchronous concurrency limit reached` hoặc
`Gateway request body capacity reached`. Tuân thủ `Retry-After`;
không tự động gửi lại sync inference hay đổi ID của async job đã nhận
để vượt quota. HTTP **503** với
`Distributed inference admission unavailable` nghĩa là Redis không kết
nối được hoặc ACK không chắc chắn. Kiểm tra TLS, ACL, failover, namespace,
không chuyển về bộ đếm process-local để lách giới hạn. GET health vẫn
có thể trả 200 khi Redis lỗi.

Hết hạn sync lease không khẳng định GPU đã ngừng. Reset Redis, đổi namespace
hoặc xóa rate keys có thể làm quota được tính lại từ đầu: phải đối chiếu
Redis, Modal logs và Workspace budget/spend limit. Mọi gateway phải dùng cùng
Redis database và namespace; không cho phép multi-gateway khi shared
admission đang tắt. Xem [Vận hành](OPERATIONS.vi.md#rate-limit-phân-tán-và-bảo-vệ-gateway-khi-quá-tải).

## Lỗi thường gặp

| Triệu chứng | Nguyên nhân có thể | Hướng xử lý |
| --- | --- | --- |
| Không tìm thấy Secret/Volume/App | Các bước tải model, tạo Secret và deploy dùng khác Workspace/Environment, hoặc thiếu tài nguyên | Kiểm tra `modal profile list`, `modal environment list`, `MODAL_ENVIRONMENT`, sau đó xem `modal app list`, `modal volume list`, `modal secret list` trong cùng Environment; không đổi context giữa chừng |
| Tải checkpoint thất bại | Quyền truy cập model, mạng, thời gian CPU hoặc storage quota không đủ | Xem log hàm `modal_models.py` tương ứng; đối chiếu quyền truy cập Hugging Face, SHA cố định và dung lượng Volume trước khi thử lại |
| Không cấp phát được L40S/GPU | Thanh toán, quota L40S hoặc GPU capacity trong Workspace | Kiểm tra Billing và GPU quota của Modal; deploy thành công không có nghĩa đã giữ sẵn GPU |
| HTTP `401` | Thiếu hoặc sai Bearer token | Kiểm tra `VISION_EMBEDDINGS_API_TOKEN` phía server mà không in giá trị token |
| HTTP `413`/`422` | Đầu vào quá lớn hoặc schema không hợp lệ | Kiểm tra Base64, giới hạn ảnh, mã model, số chiều và batch size |
| HTTP `503` | Worker inference không sẵn sàng | Xem log được bảo vệ, revision checkpoint và bộ nhớ GPU khả dụng |
| HTTP `429` | Khóa xác thực tạm thời, slot inference đang bận hoặc hết số embedding/phút | Đọc `detail`; nếu khóa xác thực, kiểm tra token và chờ hết `Retry-After`, nếu không thì giảm request chồng nhau |
| HTTP `504` | Thời gian chờ cấp GPU hoặc xử lý vượt thời hạn | Kiểm tra hàng đợi/tài nguyên và log worker trên Modal trước khi quyết định tạo request mới tính phí; không tự động retry POST bị timeout |
| Lỗi nạp model | Dependency hoặc checkpoint không khớp | Dựng lại môi trường đã pin và xác minh checkpoint |
| Latency cao | GPU cold start hoặc định tuyến model | Gom batch theo model và đo cold/warm request riêng |
| GPU hết bộ nhớ | Model activations hoặc đầu vào quá lớn | Giảm batch/kích thước ảnh; chỉ nạp một tiến trình model |
| Số chiều vector bất thường | Revision model hoặc giả định của client sai | Từ chối kết quả không khớp; không tự đệm hoặc cắt vector |

## Xử lý sự cố khi cài đặt lần đầu

1. Bắt đầu từ [Vận hành](OPERATIONS.vi.md#triển-khai-lần-đầu-điều-kiện-và-trình-tự), kiểm tra giới hạn chi phí và Workspace/Environment đã chọn.
2. Phải có **hai** kết quả tải checkpoint `READY` và **SHA trùng khớp** với hai Secret. Không trộn nhánh `main` có thể thay đổi với revision đã ghim.
3. Theo [Kiểm tra deployment](VERIFICATION.vi.md#kiểm-tra-tài-nguyên-và-gateway-trước-khi-chạy-gpu) để xác minh tài nguyên và CPU health có xác thực **trước** GPU inference.
4. Nếu GPU smoke trực tiếp PASS nhưng HTTPS thất bại, kiểm tra gateway URL, Bearer token, quota và log deployment; gọi trực tiếp Modal SDK không đi qua HTTPS gateway.
5. Nếu HTTPS đồng bộ lần đầu trả `504`, cần điều tra request gốc đã được nhận hay chưa trước khi cho phép request mới tính phí. **Không tự retry POST bị timeout**; nên dùng API async với request ID ổn định cho cold start.

Không đưa Bearer token, embedding đầy đủ hoặc ảnh riêng vào issue/log.

## Sau khi triển khai

1. Chạy kiểm thử offline, lint và xác minh SHA256 của source.
2. Tải và xác minh hai checkpoint đầy đủ: `Qwen/Qwen3-VL-Embedding-8B` và `tencent/WeMM-Embedding-9B`.
3. Xác nhận request API không có token bị từ chối và request catalog có token hợp lệ thành công.
4. Xác nhận hai model trả về vector 1.024 chiều hữu hạn, chuẩn hóa trên cùng ảnh được phép sử dụng.
5. Xác nhận khả năng định tuyến model, phản hồi lỗi đã lọc và theo dõi bộ nhớ GPU khi tải tăng.
6. Đánh giá chất lượng tìm kiếm bằng ví dụ đại diện cho ứng dụng của bạn.

Xem quy trình smoke test chức năng trong [Nghiệm thu](VERIFICATION.vi.md).
