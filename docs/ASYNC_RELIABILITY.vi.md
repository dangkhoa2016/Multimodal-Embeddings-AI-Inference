# Độ tin cậy của inference bất đồng bộ và cách khôi phục

🌐 Ngôn ngữ / Language: [English](ASYNC_RELIABILITY.md) | **Tiếng Việt**

Tài liệu này hướng dẫn vận hành API embedding bất đồng bộ khi công việc trên GPU
Modal có thể kéo dài hơn thời gian chờ HTTP. Hãy đọc cùng
[hợp đồng API](API_CONTRACT.vi.md), [hướng dẫn vận hành](OPERATIONS.vi.md)
và cấu hình giới hạn chi tiêu tại Modal Workspace.

## Định danh request và trạng thái GPU

Mỗi request async mới được chấp nhận sẽ gắn `request_id` với model, dữ liệu,
checkpoint revision và cấu hình embedding. Theo dõi bằng `job_id`; nếu mất
phản hồi của POST, dùng
`GET /v1/embeddings/jobs/by-request/{request_id}` đã xác thực.
Những lệnh GET này không khởi chạy GPU.

`QUEUED` và `RUNNING` không chứng minh công việc đã thành công.
`INDETERMINATE` cho biết timeout, heartbeat hoặc phản hồi từ GPU chưa đủ
để xác định kết quả. Xác nhận đã tiếp nhận lệnh hủy **không** chứng minh
GPU đã dừng. Không tự tạo request ID thay thế hoặc gửi POST lại. Bản ghi
job hết hạn cũng không chứng minh chưa từng có công việc GPU được gửi.

## Cơ chế giữ và nhả permit

Coordinator sử dụng hai nhóm permit logic: một nhóm toàn cục và một nhóm
cho từng client đã xác thực. Mỗi slot có các **claim theo thế hệ** nối tiếp
nhau. Khi kết quả cuối đã được lưu bền vững, chủ sở hữu ghi release receipt
bất biến, gắn đúng claim và acquisition ID. Chỉ khi receipt khớp claim
trước đó, thế hệ tiếp theo mới được nhận. Không có thao tác tự động xóa
khóa Modal Dict có thể tái sử dụng.

Cách tổ chức chỉ thêm bản ghi này tránh lỗi đua giữa `get()` và `pop()`:
một tác vụ release bị trễ không thể xóa claim của request khác sau khi
khóa hết hạn hoặc được cấp lại. Cursor chỉ giúp tăng tốc tra cứu; nếu thiếu
claim trước, receipt không khớp, mất xác nhận lưu dữ liệu hoặc còn permit
kiểu cũ, dịch vụ sẽ từ chối cấp thêm để bảo toàn an toàn. Mỗi phần tử Dict
có thể hết hạn sau bảy ngày không hoạt động nên cơ chế này không phải trần
chi phí vô thời hạn.

Khi cursor gợi ý bị ghi lùi, quá trình phục hồi dùng tìm kiếm tăng theo cấp số nhân và tìm kiếm nhị phân trên claim/release receipt bất biến, thay vì quét lần lượt toàn bộ generations. Receipt của thế hệ trước bị thiếu hoặc hết hạn **không** chứng minh slot trống: hệ thống từ chối cấp permit để bảo đảm an toàn. Checkpoint hợp lệ đã ghi trước đó vẫn có thể giúp bỏ qua lịch sử cũ. Log gateway chỉ ghi tổng số lượt đọc store và thời gian reserve (store_reads, elapsed_ms), không ghi token, payload, vector hay định danh client. Giới hạn số lượt đọc giả định chuỗi release còn nguyên vẹn; không phải cam kết khi record Modal Dict bị thiếu/hết hạn hoặc provider chậm.

## Chuyển đổi admission phiên bản: bắt buộc drain

**Không hỗ trợ admission writer v1/v2 cùng hoạt động.** Gateway v1 chỉ biết
khóa permit chưa có version như `permit:global:0` và `permit:client:*`;
v2 dùng claim thế hệ bất biến `:v2:`. Kiểm tra khóa legacy ở v2 chỉ bảo vệ
một chiều: writer v1 vẫn có thể nhận job khi v2 đã giữ permit.
Giới hạn Redis/Valkey không thay thế được cơ chế loại trừ durable admission.

Bắt buộc nâng cấp theo **maintenance window và drain**: (1) ngừng nhận job
async mới ở bản cũ, kể cả request HTTP đang chờ admission; (2) xác nhận tất
cả gateway/coordinator cũ đã dừng hoàn toàn, không còn khả năng ghi permit
legacy; (3) đối soát từng job/permit legacy với GET chỉ đọc và Modal logs,
giữ trạng thái occupied khi chưa rõ kết quả; (4) chỉ bật writer v2 sau đó,
đồng thời xác minh không có writer cũ được khởi động lại. Rolling deployment
có writer cũ/mới chồng lấn là không an toàn. Tuyệt đối không xóa hàng loạt
Modal Dict hoặc permit chưa được đối soát. Nếu chưa thể drain hết writer cũ,
**không bật v2** trên cùng store.

## Đối soát dành cho người vận hành

1. Xác định job gốc qua endpoint chỉ đọc và
   [công cụ xem job](../scripts/inspect_async_job.py); không đổi request ID,
   model hoặc revision.
2. Đối chiếu log gateway, CPU coordinator và GPU worker trong Modal. Chỉ
   điều chỉnh capacity sau khi xác minh công việc từ xa thật sự đã dừng;
   heartbeat quá hạn hoặc ACK hủy không đủ làm bằng chứng.
3. Lưu vector đã hoàn tất vào kho lưu trữ của ứng dụng. Khi chưa xác định
   kết quả, tiếp tục coi job và permit liên quan đang được giữ.
4. Với bản ghi permit kiểu cũ chưa có thế hệ, cần rà soát tất cả job còn
   tồn tại trước khi thay thế/loại bỏ. Dịch vụ cố ý không tự xóa khóa cũ.
   Không xóa toàn bộ Dict như một thao tác xử lý thường quy.

Khi chạy nhiều HTTP gateway, cấu hình thêm Redis/Valkey để chia sẻ hạn mức
rate-limit và concurrency đồng bộ; đây là cơ chế khác với permit async.
Hãy thiết lập ngân sách và giới hạn chi tiêu của Workspace vì admission và
rate-limit không bảo đảm giới hạn tổng chi phí nhà cung cấp.

## Khôi phục từ notebook

Demo Notebook có thể lưu journal kiểm soát phiên ở nơi riêng tư.
Với `RUN_API_DEMO=False`, notebook chỉ preview kể cả khi có credential.
`RUN_API_DEMO=True` cho phép gửi GPU job có thể phát sinh phí, nhưng
job chưa rõ kết quả chỉ được đối soát bằng GET, không tự gửi lại POST. Nếu phản hồi của endpoint khôi phục chỉ đọc không hợp lệ, notebook
dừng và giữ nguyên journal; không tự chuyển sang gửi POST. Cần giữ kín
journal vì có thể chứa các vector embedding đã hoàn thành.
