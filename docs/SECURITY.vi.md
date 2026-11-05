# Bảo mật và giám sát

🌐 Ngôn ngữ / Language: [English](SECURITY.md) | **Tiếng Việt**

Tài liệu mô tả cơ chế trong source phiên bản **1.0.0**. Đây **không phải**
chứng nhận deployment Modal thực, cấu hình Redis, hạ tầng bên ngoài, việc
lưu API key hay kết quả inference của model.

## Phạm vi các cơ chế bảo mật

Dự án cung cấp cơ chế bảo vệ ở cấp API cho owner và các client độc lập
(tùy chọn). Đây **không phải** hệ thống xác thực người dùng cuối, quản lý
tổ chức hoặc phân quyền multi-tenant doanh nghiệp hoàn chỉnh. API token
theo client và quyền sở hữu async job không đồng nghĩa cách ly tenant
toàn diện, kiểm soát thanh toán hay chứng nhận an toàn thông tin.

Người vận hành dịch vụ dùng chung/doanh nghiệp phải tự đánh giá ranh giới
tenant đầu cuối, quyền người dùng, quản lý secret, chống lạm dụng,
audit, xử lý sự cố và trần chi phí từ nhà cung cấp. Xem
[phạm vi triển khai](OPERATIONS.vi.md#phạm-vi-triển-khai-và-sử-dụng-trong-doanh-nghiệp)
và checklist dành cho người vận hành bên dưới.

## Ranh giới tin cậy

- Mọi owned API route `/v1/*` ngoại trừ `GET /v1/models` công khai đều kiểm tra Bearer token trước khi đọc body.
  Owner token được quản lý riêng với các API key theo client được lưu bằng
  digest SHA-256; client ID ổn định dùng để cách ly quyền xem async jobs.
- `GET /v1/ops/metrics` **chỉ dành cho owner**. Client có key hợp lệ
  nhưng không phải owner nhận HTTP 404; người chưa xác thực nhận HTTP 401
  (hoặc lockout). Không giao owner token cho client không tin cậy.
- Redis/Valkey là shared ledger tùy chọn cho rate/concurrency giữa nhiều
  CPU gateway. Credential Redis chỉ thuộc gateway Secret; GPU worker chỉ
  nhận model revisions, không nhận API token hay Redis URL.
- Modal Dict lưu metadata async job và vector đã tính. Redis rate buckets
  và Dict permits **không** phải trần chi phí tuyệt đối; phải cấu hình
  Workspace budget/spend limit ở Modal.

## Metrics bảo vệ quyền riêng tư

Endpoint metrics owner-only chỉ có **thống kê giới hạn số nhãn**: tổng
response theo nhóm route/HTTP status, lượt trạng thái async job được
quan sát, một số lý do từ chối tại middleware và thời gian sống gateway.
Không có raw URL, IP, request ID, tenant ID, Bearer token/digest, prompt,
ảnh, vector hoặc stack trace. Response có `Cache-Control: no-store`.

Phạm vi metrics là `gateway_process_only`: reset khi container restart,
**không** tổng hợp qua mọi gateway Redis hoặc detached coordinator.
`async_job_state_observations` tăng khi người dùng submit/poll job;
**không phải** số GPU job hiện đang chạy. Muốn giám sát toàn hệ thống
cần thêm log/metrics provider đã hạn chế truy cập và che dữ liệu nhạy
cảm. Không quét toàn bộ Modal Dict khi mỗi lượt gọi metrics.

HTTP 422 từ Pydantic loại bỏ `input`/`ctx` mang dữ liệu do người dùng
gửi; lỗi ảnh không đưa nội dung exception của Pillow ra ngoài. Lỗi store
hoặc provider chỉ ghi thông báo cố định, không chứa exception, ảnh hay
credential. Cần cấu hình reverse proxy không ghi raw HTTP body.

## Security checks offline (chỉ CPU)

```bash
python scripts/security_audit.py
python -m pytest -q tests/test_security_monitoring.py
```

Script kiểm tra các bất biến bảo mật tĩnh và từ chối tên file tracked
có nguy cơ chứa Secret. Regression tests kiểm tra metrics owner-only,
ẩn token, HTTP 422 an toàn, lỗi backend và cardinality hữu hạn. CI chạy
không dùng Modal credentials, mạng hoặc GPU. Đây là biện pháp bổ sung,
**không** thay thế CVE scan, pentest, live Redis fault injection, ứng
phó sự cố hay audit Cloud/Modal network và phân quyền.

## Checklist cho người vận hành production

1. Giữ owner token trong secret manager chỉ chủ sở hữu truy cập; quản lý
   key client riêng. Dùng công cụ offline tại [Vận hành](OPERATIONS.vi.md#api-key-riêng-theo-client-và-giới-hạn-async-jobs-bền-vững-tùy-chọn)
   để xoay/thu hồi key theo client ID.
2. Bắt buộc TLS và least privilege cho Redis/Valkey; bảo vệ Modal
   Secrets, Volumes, service URL và logs. Không làm mất registry hoặc
   pinned checkpoint revisions khi thay Secret.
3. Theo dõi HTTP 401/429/503, async `INDETERMINATE`, lỗi admission,
   permit/lease bất định, Redis reset, worker failure và chi phí thực
   trong Modal. Metrics mỗi gateway không thể chứng minh tổng job toàn cục.
4. Khi POST/dispatch/timeout chưa rõ kết quả, dùng async ID ổn định và
   GET có xác thực; không tự retry GPU hoặc xóa permit chỉ vì heartbeat
   quá thời hạn.
5. Kiểm tra quy trình khôi phục và thay credential trên deployment thật
   **trước khi** tăng CPU gateways. CPU test PASS không chứng minh live
   deployment đã được chứng nhận bảo mật.
