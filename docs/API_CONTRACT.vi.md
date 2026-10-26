# Hợp đồng API

🌐 Ngôn ngữ / Language: [English](API_CONTRACT.md) | **Tiếng Việt**

Sử dụng endpoint được cấp cho **deployment của chính bạn**; repository không cung cấp inference endpoint công khai dùng chung.

Mọi endpoint `/v1/*` yêu cầu `Authorization: Bearer <API_TOKEN>`. Mặc định là token của chủ deployment (`VISION_EMBEDDINGS_API_TOKEN`); người vận hành có thể cấu hình thêm các mã băm SHA-256 của API key riêng từng client. Dùng TLS ngoài localhost và không để lộ token trong notebook công khai.

## Các endpoint chỉ đọc

- `GET /v1/ops/metrics`: **chỉ owner Bearer token**; các counters theo
  gateway process gồm HTTP statuses, lượt quan sát async jobs, uptime và
  một số lý do bị middleware từ chối. Client khác nhận 404; chưa xác thực
  nhận 401/lockout. Không chứa token, raw URL, request ID, client ID, ảnh,
  văn bản hoặc vector. `async_job_state_observations` **không phải** số
  GPU job live toàn hệ thống; counters reset khi restart. Response có
  `Cache-Control: no-store`. Xem [Bảo mật](SECURITY.vi.md).

- `GET /v1/health`: trạng thái gateway và version, **không** khẳng định model GPU sẵn sàng.
- `GET /v1/models`: các mã model, số chiều được hỗ trợ và revision checkpoint đã cố định.
- `GET /v1/models/{model_id}`: metadata của một model hoặc HTTP 404.

## POST /v1/embeddings

Chỉ gửi một trong hai trường `image_base64` (PNG/JPEG/WebP) hoặc `text` (UTF-8). Bắt buộc có `model`; có thể thêm `dimension` và `request_id` (tối đa 128 ký tự). Đối với text, schema loại bỏ khoảng trắng ở đầu/cuối trước inference; input_sha256 được tính từ byte UTF-8 sau khi strip, không phải chuỗi gốc. Ví dụ, hai dấu cách ở hai đầu hello world không làm đổi hash. Với ảnh, SHA256 được tính từ byte ảnh giải mã Base64, không áp dụng quy tắc text.

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "a landscape photograph",
  "dimension": 1024,
  "request_id": "request-example-001"
}
```

Phản hồi minh họa (vector được rút gọn **không** phải kết quả model thật):

```json
{
  "model": "qwen3-vl-embedding-8b",
  "repository": "Qwen/Qwen3-VL-Embedding-8B",
  "revision": "<PINNED_MODEL_REVISION>",
  "backend": "modal",
  "dimension": 1024,
  "vector": [0.001, -0.002],
  "input_sha256": "<SHA256_OF_INPUT>",
  "request_id": "request-example-001",
  "elapsed_ms": 123.0,
  "timing": {"model_state": "warm", "gpu_worker_first_request": false, "model_load_ms": 0.0, "gpu_encode_ms": 76.0, "gpu_worker_ms": 80.0, "gpu_dispatch_overhead_ms": 3.0, "batch_size": 1},
  "embedding_metadata": {"preprocessing_version": "qwen3-vl-adapter-v1", "instruction_sha256": "<HASH>", "config_sha256": "<HASH>"},
  "normalized": true
}
```

Vector phản hồi thật có đúng `dimension` phần tử hữu hạn và chuẩn L2 xấp xỉ 1. `elapsed_ms` thể hiện thời gian xử lý inference của request hoặc nhóm batch, **không** phải độ trễ mạng từ đầu đến cuối.

## Instruction Qwen tùy chọn và metadata cấu hình vector

Chỉ với **Qwen3-VL-Embedding-8B**, trường `instruction` tùy chọn mô tả tác vụ trên một dòng, dài 1–512 ký tự. Giá trị rỗng, chứa ký tự điều khiển hoặc quá dài trả HTTP 422. Danh sách model khai báo `supports_custom_instruction`.

```json
{
  "model": "qwen3-vl-embedding-8b",
  "text": "How can I reset my password?",
  "dimension": 1024,
  "instruction": "Retrieve help articles that answer the user's question."
}
```

Khi không truyền instruction, dịch vụ giữ nguyên instruction tìm kiếm thư viện ảnh trước đây của Qwen và vector tương ứng. WeMM không nhận instruction tùy chỉnh; nếu truyền sẽ nhận HTTP 422.

Đối tượng `embedding_metadata` bổ sung trong response gồm `preprocessing_version`, `instruction_sha256` (null đối với WeMM) và `config_sha256`. Response không lặp lại instruction gốc. Trường `input_sha256` cũ chỉ băm dữ liệu đầu vào, nên không thể dùng một mình làm khóa cache vector.

Phân tách chỉ mục theo model, revision, số chiều và phiên bản tiền xử lý. Dùng cặp `(input_sha256, config_sha256)` để cache từng embedding. Instruction của query và document có thể khác nhau có chủ đích; hai mã băm instruction không bắt buộc giống nhau để so cosine. Cần re-embed các mục chỉ mục bị ảnh hưởng nếu cấu hình mã hóa thay đổi.

## Chi tiết thời gian hiệu năng

Phản hồi inference đã xác thực vẫn giữ elapsed_ms (thời gian chờ inference phía gateway), đồng thời bổ sung timing khi Modal GPU worker hỗ trợ đo lường. Client cũ có thể bỏ qua timing; backend mock/local trả về null.

- model_state: initial_load (lần nạp model đầu quan sát trên worker), model_reload (khởi động lại tiến trình đã dừng) hoặc warm (tái sử dụng tiến trình). gpu_worker_first_request đánh dấu request đầu tiên quan sát được trong GPU container đó. gpu_worker_id là mã ngẫu nhiên gồm 32 ký tự hex để nhận diện vòng đời của GPU worker; không phải hostname/container ID hoặc bằng chứng autoscaler. Deployment cũ có thể không trả trường này.
- model_load_ms: thời gian chạy subprocess, khởi tạo model/checkpoint và thay model. Giá trị bằng 0 khi warm. gpu_encode_ms: thời gian tiền xử lý và tính vector theo batch trong subprocess.
- gpu_worker_ms: thời gian thực thi Modal GPU method; gpu_dispatch_overhead_ms: thời gian khứ hồi từ gateway đến worker trừ thời gian worker, bao gồm điều phối, truyền dữ liệu và có thể cả khởi tạo container. Đây **không phải cold start thuần túy**.
- batch_size cho biết số input dùng chung các phép đo. Các thời gian dùng để chẩn đoán, không phải cam kết độ trễ; muốn đo tổng thời gian client phải sử dụng bộ đếm riêng trên Kaggle, Colab hoặc Rails.

Tất cả giá trị thời gian tính bằng mili giây. Request đầu được quan sát có thể đã warm nếu dịch vụ đang chạy; các client đồng thời có thể ảnh hưởng trạng thái và thời gian.

## Mã model và số chiều

- [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B): `qwen3-vl-embedding-8b`, số chiều nguyên từ 64 đến 4096.
- [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B): `wemm-embedding-9b`, số chiều 64, 128, 256, 512, 1024, 2048 hoặc 4096.

Bỏ qua `dimension` hoặc gửi `null` sẽ chọn mặc định 1024 chiều. Gửi `0` hoặc số chiều không được hỗ trợ sẽ nhận HTTP 422.

Vector từ các mã model, revision, số chiều và cấu hình tiền xử lý khác nhau không thể sử dụng thay thế cho nhau.

## GPU job bất đồng bộ (cold-start và khôi phục)

Sử dụng **POST /v1/embeddings/async** khi Modal phải chờ cấp phát GPU hoặc tải
model lâu hơn deadline HTTP đồng bộ. Endpoint nhận một embedding và bắt buộc
có `request_id` ổn định (8–128 ký tự ASCII gồm chữ, số, `_`, `-`, `.`,
hoặc `:`; ký tự đầu là chữ/số).

- HTTP **202** trả `job_id`, `state=QUEUED` hoặc `RUNNING`, và header
  `Retry-After: 3`: **chưa phải** kết quả embedding.
- Poll **GET /v1/embeddings/jobs/{job_id}** với Bearer token hiện còn hợp lệ;
  sau khi xoay token, dùng token mới và giữ nguyên job ID.
- `SUCCEEDED` trả HTTP **200** và `result` có cấu trúc như API đồng bộ.
  `FAILED` là trạng thái cuối; **không tự retry**.
- Khi mất ACK POST async, ưu tiên **GET
  /v1/embeddings/jobs/by-request/{request_id}** với key của cùng client.
  Server tự suy ra job ID thuộc namespace client mà **không gọi GPU**.
  HTTP 404 có thể là job chưa được tạo **hoặc** bản ghi đã hết hạn.
  Notebook ưu tiên GET này; với server cũ sẽ dự phòng GET legacy owner.
  Gửi lại cùng ID/profile chỉ được chống trùng khi bản ghi còn tồn tại;
  record hết hạn có thể gây thêm GPU work tính phí. **Không tự POST lại**
  sau khi mất ACK. Dùng ID cũ với input/model/dimension/instruction khác
  trả **409**.
- Submission mới giữ một slot ngắn hạn ở gateway đến khi có ACK dispatch
  và giữ các durable permit toàn cục/theo client cho đến khi có kết quả cuối
  chắc chắn. Nếu bật Redis/Valkey shared rate budget thì chỉ job mới bị tính
  embedding unit; polling hoặc gửi lại đúng async ID đã nhận không bị trừ
  thêm quota phân tán. Đây không phải giới hạn chi phí tuyệt đối.
- Job gắn với revision model và profile mã hóa đã được nhận. Coordinator có
  profile khác sẽ báo job lỗi trước khi gửi công việc tới GPU.
- CPU coordinator tách biệt có thể chờ GPU L40S trong lúc client polling,
  nhưng có giới hạn thực thi 3.600 giây, không bảo đảm luôn thành công.
- Coordinator cập nhật heartbeat khoảng 30 giây/lần. Job `RUNNING` mất
  heartbeat trên 180 giây hoặc `QUEUED` quá 24 giờ sẽ được báo
  `INDETERMINATE` (HTTP **200**) kèm cảnh báo. Đây chỉ là **trạng thái
  chẩn đoán khi đọc**, không chứng minh GPU đã ngừng. Bản ghi idempotency
  vẫn được giữ và có thể nhận kết quả thành công sau đó bằng cùng job ID.
  Không tự tạo request ID mới.

Modal Dict (mặc định `multimodal-embeddings-async-jobs`; có thể đổi bằng
`VISION_EMBEDDINGS_JOB_DICT_NAME`) lưu trạng thái và vector thành công,
không lưu ảnh gốc hoặc token. Bản ghi hết hạn sau bảy ngày không truy cập;
hãy chuyển embedding hoàn tất vào kho vector bền vững của ứng dụng.
Giữ kín job ID/vector nếu dữ liệu nhạy cảm. Ngắt kết nối client không tự hủy
GPU job đã được gửi. `INDETERMINATE` có thể chuyển trở lại kết quả
thật khi coordinator tiếp tục hoặc hoàn tất. Kiểm tra log Modal/job store
trước khi quyết định xử lý thủ công; không có cơ chế retry, dispatch lại
hoặc xóa tự động.

Endpoint đồng bộ giữ nguyên, **không có tính idempotency**: `request_id` chỉ
dùng để đối chiếu. Endpoint async là tùy chọn, nhận một embedding/request.

### Xác minh checkpoint đã nạp

Với inference trên Modal, service yêu cầu model ID, repository và SHA checkpoint
40 ký tự do GPU worker trả về khớp revision đã ghim trước khi nhận vector.
Subprocess chốt identity từ marker snapshot đã kiểm tra khi nạp model; phản hồi
warm giữ nguyên identity đó. Worker kiểm tra revision mong đợi trước khi encode.
Thiếu identity, revision chưa ghim hoặc không khớp sẽ làm request thất bại; job
async chuyển thành `FAILED` và không có kết quả. Worker cũ thiếu giao thức này
bị từ chối. Đây là kiểm tra metadata snapshot, không phải chứng thực độc lập
nội dung các file trọng số.

## POST /v1/embeddings/batches

```json
{"items": [
  {"model":"qwen3-vl-embedding-8b", "text":"one", "dimension":1024},
  {"model":"wemm-embedding-9b", "text":"two", "dimension":1024}
]}
```

Trả về `{"count": 2, "items": [...]}` theo thứ tự đầu vào. Gateway gom request theo model và gửi các nhóm model độc lập tới các GPU worker riêng. Giới hạn batch mặc định là **8** (schema tối đa **32**).

## Rate-limit phân tán (tùy chọn)

Deployment một gateway mặc định dùng quota cục bộ. Nếu bật Redis/Valkey
shared limiter, tất cả gateway dùng thời gian Redis server để trừ quota
**embedding-unit** toàn cục và từng client theo cách atomic. Một batch
bốn embedding tính bốn units; async route chỉ trừ khi có job mới, không
trừ khi GET polling hoặc đọc lại cùng job đã tồn tại. Request GPU đồng bộ
cũng sử dụng concurrency lease chung có thời hạn. Hết rate/concurrency/
body capacity trả **429** kèm `Retry-After`; Redis lỗi trả **503**,
không fallback về quota process-local yếu hơn. Token bucket nạp đều
trong 60 giây và cho phép burst tới capacity. Request đồng bộ vẫn
không có idempotency dù có trường `request_id`.

## Lỗi và giới hạn đầu vào

- HTTP 401: thiếu hoặc sai Bearer token.
- HTTP 404: không tìm thấy mã model ở route metadata.
- HTTP 413: request body hoặc ảnh vượt giới hạn cấu hình.
- HTTP 422: Base64/ảnh, số chiều, model, batch size hoặc schema không hợp lệ.
- HTTP 429: khóa xác thực tạm thời hoặc vượt giới hạn số embedding/phút hay request inference đồng thời trên gateway; `Retry-After` thông báo thời gian chờ. Trường `detail` phân biệt khóa xác thực (`Too many failed authentication attempts`) với giới hạn inference.
- HTTP 503: worker inference không sẵn sàng; client không nhận chi tiết lỗi nội bộ.
- HTTP 504: hết thời hạn inference trên Modal, tính cả thời gian chờ cấp GPU; API yêu cầu hủy công việc từ xa và trả `Retry-After`.

API token gồm **16–4089 ký tự ASCII hiển thị được**. Mặc định, ba lần xác thực sai trong năm phút sẽ khóa peer ASGI trong 15 phút; mọi credential từ peer đó đều bị chặn đến khi hết hạn. Xem [khóa xác thực](OPERATIONS.vi.md#giới-hạn-xác-thực-bearer-token-thất-bại-và-khóa-tạm-thời).

Giới hạn mặc định: **5 MiB** mỗi ảnh, **12 triệu** pixel sau giải mã, **8** phần tử mỗi batch và **8192** ký tự văn bản. Ảnh phải là JPEG, PNG hoặc WebP; API không tải ảnh từ URL bên ngoài.

Mặc định 120 embedding/phút và 4 request inference đồng thời trên mỗi gateway container; batch có N phần tử tính N đơn vị. Bộ đếm lưu trong RAM, không phải quota phân tán hay trần chi phí cứng.

Với backend Modal, thời hạn inference mặc định là **60 giây**, cấu hình qua `VISION_EMBEDDINGS_INFERENCE_TIMEOUT_SECONDS` (> 0, tối đa 120). Khi timeout, API giải phóng slot và yêu cầu hủy các lời gọi GPU của request đó; bước dọn dẹp chỉ chờ thêm tối đa 5 giây. Request ID dùng để đối chiếu, không phải khóa idempotency. Không tự động gửi lại POST sau timeout.

## Tương thích

Phiên bản đường dẫn vẫn là `/v1`. Cần tạo lại chỉ mục khi revision model, quy trình tiền xử lý hoặc số chiều thay đổi; cache embedding bằng cặp `(input_sha256, embedding_metadata.config_sha256)`.
