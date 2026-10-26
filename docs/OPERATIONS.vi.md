# Triển khai và vận hành Modal

🌐 Ngôn ngữ / Language: [English](OPERATIONS.md) | **Tiếng Việt**

Các bước dưới đây hướng dẫn triển khai và vận hành **API Multimodal Embeddings AI Inference** trên Modal.

### Điều hướng nhanh cho người vận hành

| Công việc | Bắt đầu tại | Tác động GPU/Secret |
| --- | --- | --- |
| Triển khai lần đầu | [Checklist cài đặt](#triển-khai-lần-đầu-điều-kiện-và-trình-tự) | Download/build có thể tính phí; GPU tests cần duyệt riêng |
| Tải checkpoint | [Tải và ghim model](#tải-và-cố-định-checkpoint) | CPU/storage từ xa |
| Tạo owner token và revision | [Xác thực](#cấu-hình-xác-thực) | Chỉ đổi Secret khi operator tự chạy helper |
| Cấp/xoay API key theo client | [Per-client keys](#api-key-riêng-theo-client-và-giới-hạn-async-jobs-bền-vững-tùy-chọn) | Stage offline trước; cập nhật Secret thủ công |
| Mở rộng CPU gateway | [Rate-limit phân tán](#rate-limit-phân-tán-và-bảo-vệ-gateway-khi-quá-tải) | Cần Redis/Valkey và deploy có kiểm soát |
| Xác minh deployment | [Kiểm tra](#kiểm-tra-deployment) | GPU tests riêng cần xác nhận chi phí |
| Điều tra async job bất định | [Đối soát thủ công](#đối-soát-thủ-công-các-job-async-quá-hạn) | Ưu tiên GET; không tự gửi GPU job lại |

[README tiếng Việt](../README.vi.md) mô tả cách thử API mock; tài liệu này dành cho **người vận hành**. [API contract](API_CONTRACT.vi.md) và [hướng dẫn client](CLIENT_INTEGRATION.vi.md) dành cho ứng dụng gọi endpoint đã triển khai.

### Tham chiếu cấu hình

`.env.example` chỉ là mẫu khởi chạy mock local, không phải cấu hình
production đầy đủ. Khi sử dụng nhiều client hoặc nhiều gateway, hãy tham khảo
[mục cấp API key](#api-key-riêng-theo-client-và-giới-hạn-async-jobs-bền-vững-tùy-chọn)
và [rate-limit phân tán](#rate-limit-phân-tán-và-bảo-vệ-gateway-khi-quá-tải).
Giá trị mặc định của `Settings.from_env()`:

| Biến | Mặc định |
| --- | --- |
| `VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON` | `{}` |
| `VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL` | `8` |
| `VISION_EMBEDDINGS_MAX_ASYNC_JOBS_PER_CLIENT` | `2` |
| `VISION_EMBEDDINGS_MAX_CLIENT_EMBEDDINGS_PER_MINUTE` | `60` |
| `VISION_EMBEDDINGS_MAX_BODY_REQUESTS` | `8` |
| `VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT` | Tự động khi không đặt/để trống; `.env.example` đặt `0` |
| `VISION_EMBEDDINGS_RATE_REDIS_URL` | Chưa đặt |
| `VISION_EMBEDDINGS_RATE_REDIS_NAMESPACE` | `multimodal-embeddings-v1` |
| `VISION_EMBEDDINGS_MAX_DISTRIBUTED_SYNC_INFLIGHT` | `4` |
| `VISION_EMBEDDINGS_MAX_DISTRIBUTED_CLIENT_SYNC_INFLIGHT` | `2` |
| `VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS` | `1` (đặt lúc deploy) |

## Bảo mật và giám sát hoạt động

Dùng `GET /v1/ops/metrics` với **owner token** để xem tổng responses
theo HTTP status (đặc biệt 401/429/503), uptime và các lượt quan sát
trạng thái async jobs trong từng gateway. Counters reset khi restart,
không quét Redis/Modal Dict và **không** phản ánh số GPU job đang chạy
trên toàn hệ thống. Đối soát Modal logs, job store và spending records
để xác nhận chi phí và jobs thực tế. Giữ metrics response riêng tư.

```bash
python scripts/security_audit.py
python -m pytest -q tests/test_security_monitoring.py
```

Audit CPU kiểm tra các bất biến bảo mật tĩnh; regression tests kiểm
tra owner-only, token redaction, request errors an toàn và log không
chứa exception nhạy cảm. Xem [tài liệu Bảo mật](SECURITY.vi.md).
Vẫn cần kiểm tra Redis provider, mạng, dependency và live Modal riêng.

## Cấu trúc hệ thống

- FastAPI gateway chạy CPU nhận request HTTPS có xác thực Bearer token.
- Hai GPU worker độc lập, mỗi worker sử dụng một NVIDIA L40S với `max_containers=1` và chỉ nạp model được phân công; không chuyển model trong cùng worker.
- Hai checkpoint model nằm trong hai Modal Volume riêng, tách biệt với VRAM của GPU.
- Chế độ async dùng CPU coordinator tách rời và Modal Dict lưu trạng thái job/vector hoàn tất để chống gửi trùng. Coordinator cập nhật heartbeat mỗi 30 giây; ngắt kết nối client không tự hủy GPU job.

## Triển khai lần đầu: điều kiện và trình tự

**Ai có thể làm theo?** Người có quyền đọc repository này, Python 3.11–3.13,
Git, Bash (Linux/macOS hoặc WSL trên Windows), và tài khoản Modal được phép
tạo App, Secret, Volume. Workspace phải có khả năng dùng storage, CPU để tải
model và **tối đa hai NVIDIA L40S worker** (mỗi model một GPU). Máy cá nhân
không cần GPU. Modal CPU tải checkpoint từ Hugging Face, vì vậy cần quyền
truy cập model công khai và kết nối mạng phù hợp.

**Trước khi phát sinh phí**, kiểm tra thanh toán, khả năng dùng GPU và cả
[Workspace usage budget lẫn spend limit](https://modal.com/docs/guide/budgets).
Tải model/build tiêu tốn CPU và dung lượng lưu trữ; các bước inference và
smoke GPU có thể tính phí. Source/CI không xác nhận được quota hay ngân sách
trên tài khoản của người dùng.

Đối với cài đặt mới, làm theo đúng thứ tự:

1. [Cài đặt môi trường cục bộ](#cài-đặt), chạy `modal setup`.
2. [Chọn cùng Workspace và Environment](#chọn-workspace-và-environment-nhất-quán)
   cho **mọi** thao tác tải model, tạo Secret, deploy và nghiệm thu.
3. [Tải hai checkpoint đã ghim revision](#tải-và-cố-định-checkpoint);
   đối chiếu cả hai dòng `READY` và SHA.
4. [Tạo hai Secret độc lập](#cấu-hình-xác-thực) với đúng các revision SHA đó.
5. [Triển khai](#triển-khai), ghi nhận URL HTTPS do tài khoản Modal của bạn cấp.
6. [Kiểm tra deployment](#kiểm-tra-deployment): kiểm tra metadata trước;
   chỉ chạy GPU và HTTPS inference khi đã chủ động chấp nhận chi phí.

Lệnh deploy thành công chỉ chứng minh App đã được đăng ký, **không chứng minh**
model lớn đã nạp xong hoặc API inference HTTPS đã đạt nghiệm thu.
Xem [Kiểm tra deployment](VERIFICATION.vi.md) và
[Khắc phục lỗi](TROUBLESHOOTING.vi.md).

## Xử lý job async bị quá hạn

CPU coordinator có giới hạn chạy 3.600 giây. Khi job `RUNNING` không cập
nhật heartbeat quá 180 giây, hoặc `QUEUED` quá 24 giờ, API status trả
`INDETERMINATE` (HTTP 200) để cảnh báo. Đây **không phải** tự báo thất bại,
hủy hoặc retry: job ID vẫn được giữ và có thể tiếp tục nhận kết quả về
sau. Kiểm tra log Modal, job Dict và hoạt động GPU trước khi thao tác thủ
công. Không tự tạo request ID mới chỉ vì client timeout; điều đó có thể
phát sinh GPU inference trùng. Kho trạng thái job không thay thế vector
database bền vững; cần lưu kết quả đã kiểm tra vào ứng dụng.

Xoay vòng Bearer token không đổi khóa job trên backend, và không đổi
request ID mới do notebook tạo nếu deployment/model/cấu hình/dữ liệu
giữ nguyên. Không cần tạo lại embedding chỉ vì thay token.

## Đối soát thủ công các job async quá hạn

Đây là **quy trình cho người vận hành**, không phải retry tự động và cũng
không khẳng định Modal đang chạy đúng commit nguồn mới nhất:

1. Ghi lại job ID, trạng thái, thời gian tạo/cập nhật và revision deployment;
   không ghi token, ảnh gốc hay vector đầy đủ vào log.
2. Gửi **GET** đã xác thực tới status của đúng job cũ; kiểm tra Modal Dict,
   log CPU coordinator và function tương ứng. Chỉ xem log GPU worker để xác
   định công việc gốc đã được chấp nhận, còn hoạt động hoặc hoàn tất;
   tuyệt đối không gửi inference POST để kiểm tra.
3. Khi có kết quả cuối, xác minh model, revision, dimension và digest đầu
   vào trước khi lưu vào database bền vững của ứng dụng.
4. Nếu mất ACK dispatch hoặc trạng thái vẫn chưa rõ, giữ nguyên khóa
   idempotency để người vận hành kiểm tra tiếp. Không tự xóa/gửi lại job,
   tạo request ID khác hay coi heartbeat hết hạn là GPU đã dừng.
5. Theo tài liệu, bản ghi Modal Dict có thể hết hạn sau khoảng 7 ngày
   không hoạt động. Lưu kết quả đã xác minh vào nơi bền vững. Khi cả
   cache lẫn bản ghi server đều mất, việc tạo GPU inference mới cần
   phê duyệt riêng vì có thể phát sinh phí.

Để kiểm tra chỉ đọc, người vận hành nạp `VISION_EMBEDDINGS_API_TOKEN` qua
trình quản lý secret (không đưa token vào shell history), sau đó chạy:

```bash
python scripts/inspect_async_job.py --base-url https://YOUR_DEPLOYMENT.modal.run \
  --request-id demo-YOUR_STABLE_REQUEST_ID
# Hoặc dùng --job-id với mã job 64 ký tự đã biết.
```

Công cụ chỉ thực hiện **một GET có xác thực**, không POST và không in vector,
token hay nội dung ảnh. HTTP 404 không chứng minh GPU đã ngừng; hãy đối chiếu
log Modal và phiên bản backend trước khi quyết định thao tác thủ công.

Ở mã nguồn, CPU coordinator chờ heartbeat kết thúc **trước khi** ghi
`SUCCEEDED` hay `FAILED`, tránh ghi đè từ snapshot `RUNNING` cũ. Đây
không phải giải pháp cho hard-kill, GPU remote bất định hoặc deployment chưa
đồng bộ source. Hiện chưa triển khai endpoint quản trị đối soát hay tự dispatch lại.

## Tên định danh và quyền sở hữu tài nguyên

- Biến môi trường sử dụng tiền tố `VISION_EMBEDDINGS_` (ví dụ `VISION_EMBEDDINGS_API_TOKEN`).
- Python import package: `multimodal_embeddings_ai_inference`.
- Modal App: `multimodal-embeddings-ai-inference`.
- Modal Secret: `multimodal-embeddings-api`.
- Các Modal Volume: `multimodal-embeddings-qwen-weights` và `multimodal-embeddings-wemm-weights`.

Cấu hình `VISION_EMBEDDINGS_QWEN_VOLUME_NAME / VISION_EMBEDDINGS_WEMM_VOLUME_NAME` nhất quán cho cả quá trình tải checkpoint và inference trên GPU. Chạy `scripts/create_modal_secret.py` để tạo hai Secret riêng: xác thực gateway và revision cho GPU worker.

## Cài đặt

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m pip install -e '.[modal]'
modal setup
```

Các lệnh này chạy trên máy cục bộ mà không cần GPU; Modal thực hiện việc tải model và inference từ xa.

## Chọn Workspace và Environment nhất quán

Sau `modal setup`, xem Modal profile/Workspace đang được dùng và danh sách
Environment. Chọn **một Environment** trước khi tạo tài nguyên.
Biến `MODAL_ENVIRONMENT` được các lệnh Modal CLI kế thừa, kể cả các lệnh
`modal secret create` do helper xác thực gọi:

```bash
modal profile list
modal environment list
export MODAL_ENVIRONMENT="YOUR_ENVIRONMENT"
modal config show --redact
```

Thay `YOUR_ENVIRONMENT` bằng tên Environment có thật trong danh sách.
Profile đang hoạt động chọn Workspace; không đổi profile hoặc
`MODAL_ENVIRONMENT` giữa chừng. App, Secret và Volume được tìm theo
Environment; tải checkpoint ở nơi này nhưng deploy ở nơi khác sẽ dẫn đến
thiếu tài nguyên. Xem [tài liệu Modal Environments](https://modal.com/docs/guide/environments).

## Tải và cố định checkpoint

```bash
modal run modal_models.py --selection both
```

Lưu revision bất biến cho [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B). Các revision đã được xác minh là:

- `Qwen/Qwen3-VL-Embedding-8B`: `2c4565515e0f265c6511776e7193b22c0968ddc7`.
- `tencent/WeMM-Embedding-9B`: `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`.

Các SHA xác định snapshot model upstream công khai, **không phải access token**. Không tự ý thay bằng nhánh `main` có thể thay đổi khi cần tái lập kết quả.

Hai hàm tải model chỉ in `READY <repository> @ <SHA-40-ký-tự> -> <volume>`
khi đã tải snapshot và commit Volume thành công. Đối chiếu **cả hai**
dòng và sử dụng **chính xác các SHA đó** khi tạo Secret; không đoán theo
nhánh `main`. Bước này **không** khởi động GPU L40S và chưa chứng minh
model có thể được nạp. Nếu tải lỗi, kiểm tra log Modal, dung lượng storage
và quyền truy cập model trước khi đổi revision hoặc thử lại.

## Tái lập môi trường dependency GPU

CPU gateway và hai GPU worker Qwen/WeMM sử dụng môi trường Python độc lập. Dependency của worker được khóa trong [bộ Qwen](../constraints/gpu-qwen.txt) và [bộ WeMM](../constraints/gpu-wemm.txt); các yêu cầu trực tiếp để chỉnh sửa nằm trong file `.in` tương ứng. Hai model cố ý sử dụng các phiên bản Transformers và Hugging Face Hub khác nhau.

Khi thực sự cần cập nhật bộ khóa, chạy các lệnh `uv` sau trên máy đáng tin cậy, xem xét diff, rồi kiểm tra inference GPU trên môi trường mới để gắn đúng cấu hình với báo cáo hiệu năng:

```bash
uv pip compile constraints/gpu-qwen.in --python-version 3.11 --python-platform x86_64-manylinux_2_28 --no-header --no-annotate -o constraints/gpu-qwen.txt
uv pip compile constraints/gpu-wemm.in --python-version 3.11 --python-platform x86_64-manylinux_2_28 --no-header --no-annotate -o constraints/gpu-wemm.txt
```

Các bộ khóa dành cho **Python 3.11 / Linux x86_64**. Resolve dependency không phải nghiệm thu GPU và không bảo đảm nội dung wheel giống nhau giữa các package index. Sau khi triển khai, lưu danh sách gói đã cài (`pip freeze`), môi trường CUDA/GPU, source SHA và revision model cùng kết quả kiểm tra GPU để tái lập môi trường khi xử lý sự cố. Không công khai API token hoặc dữ liệu riêng tư.

## Cấu hình xác thực

```bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482
```

Ví dụ ghim các revision checkpoint cụ thể. Hãy đối chiếu revision của checkpoint đã tải và thay các SHA trong ví dụ nếu khác trước khi tạo Secret.

Helper tạo Bearer token API, đăng ký hai Modal Secret riêng và lưu các giá trị tương ứng trên **máy chạy lệnh**, với quyền chỉ chủ sở hữu được đọc. Thông báo hoàn tất chỉ hiển thị đường dẫn file, không hiển thị token. Dùng file này khi cấu hình các client; không in token, commit file hoặc dán vào code notebook công khai.

**Phân quyền Secret tối thiểu:** Gateway sử dụng `VISION_EMBEDDINGS_MODAL_SECRET_NAME` (`multimodal-embeddings-api`), chứa Bearer token và revision model. Hai GPU worker dùng `VISION_EMBEDDINGS_MODAL_MODEL_SECRET_NAME` (`multimodal-embeddings-model-config`), chỉ chứa hai revision SHA, **không chứa Bearer token**. Hai tên Secret phải khác nhau. Helper tạo Secret worker trước, nhưng các lệnh Modal CLI không có transaction chung.

**Chuyển đổi deployment hiện có:** Trước khi deploy lại `modal_app.py`, chạy helper với đúng revision SHA đã ghim và `--force` (không kèm `--rotate-token`). Thao tác này giữ token đã lưu, tạo Secret worker mới và cập nhật gateway. Nghiệm thu cả hai model sau khi deploy. Chỉ rotate token khi có thể cập nhật các client.

Để chạy smoke test trên chính máy đó, nạp file mà không hiển thị nội dung:

```bash
TOKEN_FILE="/duong/dan/duoc/helper/hien/thi"
set -a
source "$TOKEN_FILE"
set +a
```

Với Colab/Kaggle hoặc máy khác, chỉ chuyển API token qua secret manager đáng tin cậy hoặc ô nhập ẩn. Không chuyển file cấu hình chứa revision và token lên notebook công khai.

Nạp file cũng xuất hai biến `VISION_EMBEDDINGS_QWEN8_REVISION` và `VISION_EMBEDDINGS_WEMM9_REVISION`. Bước [kiểm tra model qua HTTPS](VERIFICATION.vi.md) yêu cầu các SHA checkpoint mong đợi này; cũng có thể truyền qua `--qwen-revision` và `--wemm-revision`. Script từ chối redirect và revision không khớp trước khi báo PASS cho model tương ứng.

## API key riêng theo client và giới hạn async jobs bền vững (tùy chọn)

Bearer token cũ tiếp tục dùng được cho client `owner`. Với các ứng dụng/khách hàng độc lập, cấu hình biến môi trường **chỉ ở gateway** `VISION_EMBEDDINGS_CLIENT_KEY_HASHES_JSON` dưới dạng JSON ánh xạ client ID ổn định (2–64 ký tự chữ/số/gạch dưới/gạch ngang) tới danh sách 1–4 mã băm SHA-256 viết thường. Ví dụ: `{"client_a":["<64-HEX-SHA256>"]}`. Tạo token ngẫu nhiên mạnh riêng cho mỗi client, tính SHA-256 tại máy quản trị, chỉ gửi **token gốc** cho đúng client qua secret manager và giữ mã băm trong cấu hình gateway. Không đưa API token hoặc client registry vào Secret của GPU worker. Registry lỗi, digest trùng hoặc dùng client ID dành riêng `owner` sẽ khiến gateway từ chối khởi động.

**Quản lý API key offline (không dùng mạng/GPU):** Hai công cụ dòng lệnh chỉ xử lý trên máy quản trị. `scripts/manage_client_keys.py` duy trì registry JSON chỉ chứa digest SHA-256, quyền 0600; mỗi token gốc mới được lưu vào một file 0600 riêng và không bao giờ bị ghi đè hoặc in ra terminal. Client ID không thay đổi khi xoay key. Thư mục chứa credential bắt buộc có quyền 0700.

```bash
# Helper cấu hình trước đó đã in đường dẫn file owner credential.
OWNER_CREDENTIAL_FILE="/duong/dan/do/helper/hien/thi"
KEY_DIR="$(dirname "$OWNER_CREDENTIAL_FILE")"
# Lệnh minh họa: người quản trị chủ động chạy trên máy tin cậy.
python scripts/manage_client_keys.py create --client-id client_a \
  --registry "$KEY_DIR/client-keys.json" --token-out "$KEY_DIR/client-a.key"
python scripts/manage_client_keys.py rotate --client-id client_a \
  --registry "$KEY_DIR/client-keys.json" --token-out "$KEY_DIR/client-a-new.key"
python scripts/manage_client_keys.py inspect --registry "$KEY_DIR/client-keys.json"
# Sau khi đã cấp key mới và áp dụng cấu hình, thu hồi digest cũ:
python scripts/manage_client_keys.py revoke-key --client-id client_a \
  --registry "$KEY_DIR/client-keys.json" --digest '<OLD_KEY_SHA256_HEX>'
# Thu hồi tất cả: revoke-client --client-id client_a --confirm-revoke-all
```

`scripts/client_registry_bundle.py` đọc file credential owner-only do `scripts/create_modal_secret.py` lưu (chứa Bearer token chủ sở hữu và hai revision ghim), hợp nhất registry client hiện có, bảo toàn các biến owner `VISION_EMBEDDINGS_*` khác, rồi tạo **file dotenv 0600 mới** nằm ngoài Git repository. Công cụ không gọi Modal, không ghi đè Secret đang chạy và không tiết lộ credential. Nếu thiếu registry và file owner cũng không chứa registry, lệnh sẽ từ chối trừ khi người vận hành chủ động truyền `--confirm-empty-registry`; không âm thầm coi thiếu registry là quyết định thu hồi khách hàng.

```bash
python scripts/client_registry_bundle.py --owner-file "$OWNER_CREDENTIAL_FILE" \
  --registry "$KEY_DIR/client-keys.json" --output "$KEY_DIR/gateway-next.env"
# Tự kiểm tra file riêng tư: token cũ, hai model revision và toàn bộ client.
# Chỉ sau khi operator đồng ý mới cập nhật Modal Secret hiện có:
# modal secret create --from-dotenv "$KEY_DIR/gateway-next.env" \
#   --force multimodal-embeddings-api
```

Helper `scripts/create_modal_secret.py --force` cơ bản **không tự bảo toàn** registry đã thêm từ bên ngoài; không chạy lại sau khi có client nếu chưa khôi phục cấu hình đầy đủ. Thay đổi Modal Secret không áp dụng ngay cho container đang chạy: người vận hành cần tự deploy/khởi động lại khi thích hợp và kiểm tra credential mới theo đường chỉ đọc trước khi chạy GPU. **Không dùng Bash `source` với dotenv chứa JSON** vì shell có thể làm mất dấu nháy; chỉ truyền file cho công cụ đọc dotenv. Không commit owner file, staged dotenv hoặc raw token.

Xoay key: thêm digest mới vào cùng client ID, cấp token mới qua kênh riêng, sau đó bỏ digest cũ và cập nhật Secret. Thu hồi key: xóa các digest tương ứng. Job đã nhận gắn với **client ID ổn định**, không gắn với token cụ thể, nên đổi key không gây tạo lại embedding. Client đã bị thu hồi không truy cập được job. Record cũ chưa có `client_id` chỉ dành cho `owner`.

Giới hạn async mặc định: `VISION_EMBEDDINGS_MAX_ASYNC_JOBS_GLOBAL=8` và `VISION_EMBEDDINGS_MAX_ASYNC_JOBS_PER_CLIENT=2` dùng Modal Dict để đặt chỗ cho các job chưa hoàn tất; `VISION_EMBEDDINGS_MAX_CLIENT_EMBEDDINGS_PER_MINUTE=60` giới hạn embedding units mỗi phút cho client khác owner ở mức process-local (owner vẫn theo giới hạn toàn cục cũ 120). Các giới hạn global per-process và sync inflight vẫn còn hiệu lực. HTTP 429 kèm `Retry-After` nghĩa là hết năng lực/hạn mức; không đổi request ID để lách quota. Slot chỉ được giải phóng sau khi ghi trạng thái cuối thành công. Claim một-lần của coordinator ngăn hai detached coordinators gọi GPU trùng cùng job; marker giải phóng riêng cho từng slot ngăn hai release cạnh tranh xóa nhầm slot vừa cấp lại. Lỗi trong lúc claim hoặc release có thể làm kẹt permit và phải đối soát thủ công.

**Mô hình lỗi:** Modal Dict chỉ có conditional put atomic theo từng key, không có transaction đa key giữa global và client. Nếu crash giữa các bước, permit có thể bị giữ lại gây từ chối an toàn thay vì tích lũy GPU jobs tính phí. Job `QUEUED`/`RUNNING` bất định **không được tự giải phóng, xóa, gửi lại hay retry**; cần đối chiếu trạng thái, Modal logs và chi phí trước khi phục hồi thủ công. Theo tài liệu Modal hiện hành, entry Dict hết hạn sau 7 ngày không sử dụng nên đây không phải trần chi phí vĩnh viễn. Bật budget và spend limit ở Workspace. Gateway và coordinator phải dùng cùng Modal Dict trong một Workspace/Environment; giới hạn local theo process vẫn hoạt động bổ sung; triển khai nhiều gateway còn bắt buộc bật Redis/Valkey shared limiter ở mục dưới.

## Rate-limit phân tán và bảo vệ gateway khi quá tải

**Tùy chọn đối với deployment một gateway hiện có; bắt buộc khi chạy nhiều
gateway.** Mặc định CPU gateway vẫn chỉ có `max_containers=1`. Muốn mở rộng
tối đa tám gateway, hãy chuẩn bị dịch vụ Redis/Valkey có TLS, xác thực, giới
hạn mạng và lưu dữ liệu bền vững. Cấu hình các biến sau **chỉ tại gateway**,
không đưa vào coordinator hay hai GPU worker:

- `VISION_EMBEDDINGS_RATE_REDIS_URL`: URI `rediss://...` riêng tư đặt trong
  gateway Secret cùng owner Bearer token và hai revision đã ghim. Kết nối
  từ xa bắt buộc TLS; `redis://` chỉ dùng được với loopback local.
- `VISION_EMBEDDINGS_DISTRIBUTED_RATE_LIMIT=1`: bật limiter dùng chung
  tường minh (yêu cầu Redis URL hợp lệ). Nếu **không đặt hoặc để trống**,
  `Settings.from_env()` tự bật **chỉ khi** có Redis URL. Đặt `0` sẽ tắt
  dù đã có URL; cấu hình nhiều gateway khi đó bị từ chối (fail-closed).
  **File `.env.example` chủ động đặt `0`**: chỉ thêm Redis URL vào bản
  sao của file này sẽ không kích hoạt limiter. Với nhiều gateway, hãy
  đổi thành `1` (khuyến nghị, tường minh), hoặc bỏ/để trống cờ để tự bật.
- `VISION_EMBEDDINGS_RATE_REDIS_NAMESPACE`: namespace chung (mặc định
  `multimodal-embeddings-v1`). Mọi gateway phải dùng cùng Redis database
  và namespace; đổi namespace sẽ đặt lại lịch sử quota.
- `VISION_EMBEDDINGS_GATEWAY_MAX_CONTAINERS=2`: thiết lập ở **shell dùng
  để deploy** trước khi chạy `modal_app.py`, không chỉ trong runtime Secret.
  Gateway image truyền giá trị không nhạy cảm này cho container. Nếu tăng
  trên một mà thiếu shared rate-limit, ứng dụng từ chối khởi động.
- `VISION_EMBEDDINGS_MAX_EMBEDDINGS_PER_MINUTE=120` và
  `VISION_EMBEDDINGS_MAX_CLIENT_EMBEDDINGS_PER_MINUTE=60`: quota toàn
  cục/theo client tính theo **số embedding** thay vì số HTTP requests.
- `VISION_EMBEDDINGS_MAX_DISTRIBUTED_SYNC_INFLIGHT=4` và
  `VISION_EMBEDDINGS_MAX_DISTRIBUTED_CLIENT_SYNC_INFLIGHT=2`: giới hạn
  HTTP GPU synchronous inflight chung cho tất cả gateway.
- `VISION_EMBEDDINGS_MAX_BODY_REQUESTS=8`: giới hạn số request POST
  đã xác thực có thể đọc/giải mã body đồng thời ở mỗi gateway. Các
  request GET health/status vẫn có thể phục vụ.

**Thuật toán:** Một Redis Lua `EVAL` dùng thời gian server để kiểm tra
**atomic** hai token buckets toàn cục/client và leases sync concurrency.
Mỗi request hợp lệ trừ đúng `batch_size` embedding units trên cả hai
bucket; request bị chặn không trừ quota. Token bucket cho phép burst tới
mức capacity rồi nạp đều trong 60 giây, **không phải** sliding-window
chính xác từng giây. Key chỉ chứa hash của client ID, có TTL và dùng chung
Redis Cluster hash-tag để Lua thao tác cùng shard; không lưu token, ảnh,
vector. Lease sync chỉ giải phóng sớm sau khi có kết quả chắc chắn; khi
timeout/mất kết nối/lỗi GPU bất định, lease được giữ đến lúc hết deadline
inference cộng 60 giây. Hết hạn lease **không** chứng minh GPU đã dừng.
Async jobs tiếp tục dùng durable outstanding permits cũ, đồng thời async
job **mới** vẫn trừ shared rate budget.

**Khi có sự cố:** Redis không kết nối được, dữ liệu trả về sai hoặc mất ACK
sẽ trả **HTTP 503** và tuyệt đối không fallback về quota process-local.
Hết ngân sách/concurrency trả **429** có `Retry-After`; quá nhiều body POST
đồng thời cũng trả 429. Endpoint GET health/metadata vẫn dùng được khi Bearer
token hợp lệ. Các quota process-local cũ vẫn giữ như tầng bảo vệ bổ sung.

**Yêu cầu vận hành:** Redis/Valkey cần hỗ trợ `EVAL`, `TIME`,
`HGET/HSET`, `EXPIRE`, `ZADD/ZCARD/ZREM/ZREMRANGEBYSCORE`.
Bật persistence, backup và giám sát reset/failover. Nếu Redis bị reset, lịch
sử rate và concurrency có thể bị mất; đây **không** phải hard cap về chi
phí. Phải bật budget/spend limit tại Modal Workspace. Cập nhật Redis URL
bằng cách tạo private staged gateway Secret như trên; không in URI, không
commit vào Git, notebook hoặc Secret GPU worker. Hiện nghiệm thu bằng
CPU tests với Redis emulator dùng chung; độ trễ và HA/failover thực tế
vẫn cần kiểm thử riêng trước khi tăng số gateway.

## Giới hạn xác thực Bearer token thất bại và khóa tạm thời

Các route API được bảo vệ kiểm tra xác thực trước khi đọc request body hoặc chuyển công việc sang GPU, kể cả khi app được mount hoặc có prefix. Theo mỗi peer ASGI, hai lần thiếu hoặc sai token đầu tiên trả HTTP 401; lần sai thứ ba trong năm phút trả HTTP 429 và chặn mọi lượt xác thực tiếp theo trong 15 phút. Peer đang bị khóa bị từ chối trước khi so sánh token, kể cả request có token đúng. `Retry-After` thông báo số giây còn lại; request trong thời gian khóa không kéo dài thời hạn đó. Xác thực thành công trước khi bị khóa sẽ xóa các lần sai trước đó. Client chung địa chỉ NAT hoặc proxy cùng chịu thời gian khóa.

Có thể cấu hình trong môi trường gateway với các giá trị mặc định: VISION_EMBEDDINGS_AUTH_FAILURE_LIMIT=3, VISION_EMBEDDINGS_AUTH_FAILURE_WINDOW_SECONDS=300, VISION_EMBEDDINGS_AUTH_LOCKOUT_SECONDS=900, VISION_EMBEDDINGS_AUTH_MAX_TRACKED_CLIENTS=4096.

**Ranh giới tin cậy của proxy:** Ứng dụng chủ động không đọc header X-Forwarded-For hoặc Forwarded do client cung cấp để quyết định quyền truy cập. Ứng dụng dùng địa chỉ peer do ASGI cung cấp, địa chỉ này chỉ nên được điều chỉnh bởi proxy đã xác minh tin cậy. Nếu Modal hay proxy khác đưa nhiều client về chung một địa chỉ peer, các client sẽ dùng chung bộ đếm và thời gian khóa; chưa thể khẳng định đã xác định đúng IP gốc nếu chưa kiểm chứng đường đi của request.

**Phạm vi:** Bộ giới hạn an toàn với nhiều thread và giới hạn bộ nhớ nhưng nằm riêng trong RAM của từng CPU gateway process; bộ đếm và khóa được xóa khi container khởi động lại hoặc peer bị loại khỏi bộ nhớ theo giới hạn số entry. Modal gateway hiện chỉ cho phép một container, nhưng nếu mở rộng ngang cần bộ đếm nguyên tử dùng chung hoặc rate limiter tại edge/WAF đáng tin cậy. Kẻ tấn công đổi IP liên tục vẫn có thể vượt giới hạn theo peer, do đó nên có lớp edge khi cần. Sử dụng token ngẫu nhiên dài và thay ngay khi bị lộ. Proxy authentication của Modal là lớp bảo vệ bổ sung tùy chọn, nhưng cần thay đổi cấu hình client.

## Chủ động thay API token

Chỉ dùng `--force` sẽ ghi đè Modal Secret nhưng **tái sử dụng token đã lưu**; `--rotate-token --force` sinh token mới, cập nhật Secret rồi thay file cục bộ nếu Modal CLI thành công. Không cần xóa file token cũ để xoay vòng.

~~~bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482 \
  --force --rotate-token
~~~

Trước khi rotate, hãy chuẩn bị cập nhật client. Tùy cách Modal triển khai/giữ container đang chạy, cần restart hoặc redeploy API để token mới được nạp; kiểm tra token mới thành công và token cũ bị từ chối (HTTP 401) sau khi triển khai xong. Nếu Modal CLI lỗi, helper giữ nguyên file token cục bộ. Nếu mất file cục bộ, --force tự sinh token mới.


## Triển khai

```bash
modal deploy modal_app.py
```

Sử dụng deployment URL do chính tài khoản Modal của bạn trả về. Cấu hình URL trong client gọi API hoặc secret manager phía server; không hardcode URL dịch vụ đang hoạt động vào repository.

## Kiểm tra deployment

Trước tiên xác nhận App và tài nguyên trong Environment đã chọn mà **không**
tạo GPU inference:

```bash
modal app list
modal volume list
modal secret list
```

Các lệnh tiếp theo là **hai phép kiểm tra inference riêng biệt có thể tính phí**,
không phải kiểm tra setup chỉ đọc. Dùng ảnh được phép xử lý và chủ động chấp
nhận phí GPU:

```bash
# Gọi trực tiếp hai GPU worker qua Modal SDK (không kiểm tra xác thực HTTPS).
python scripts/smoke_modal_gpu.py --image /path/to/authorized-image.png

# Nạp file token/revision riêng như hướng dẫn phía trên trước khi chạy.
# Nghiệm thu end-to-end HTTPS:
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png --dimension 1024
```

Thay URL bằng **HTTPS base URL của App bạn vừa deploy**, không dùng endpoint
của tác giả. Gọi GPU trực tiếp chứng minh model hoạt động nhưng bỏ qua xác
thực Bearer ở gateway. Kiểm tra HTTPS xác thực toàn bộ đường truyền, hai
model, SHA revision đã ghim và vector chuẩn hóa. Nên chạy ngay sau phép
kiểm tra worker khi model còn có thể đang warm: API đồng bộ có **deadline
server mặc định 60 giây**, nên cold-start ban đầu có thể trả HTTP 504.
Timeout **không cho phép tự động gửi lại**; kiểm tra log Modal và cân nhắc
[API async](API_CONTRACT.vi.md#gpu-job-bất-đồng-bộ-cold-start-và-khôi-phục)
với request ID ổn định, có phê duyệt riêng nếu phát sinh GPU job mới.
Xem [Kiểm tra deployment](VERIFICATION.vi.md) để biết tiêu chí PASS và xử lý.

## Kiểm soát truy cập và chi phí

**Trước khi mở hosted API cho người dùng không tin cậy**, chủ deployment phải thiết lập cả **Workspace budget** (trần tổng mức sử dụng hằng tháng, trước khi trừ credit) và **Workspace spend limit** (trần tiền thực trả sau credit) trong **Usage & Billing** của Modal. Đây là hai giới hạn **được Modal thực thi**, khác với cảnh báo thanh toán. Khi đạt spend limit, workload còn được credit phù hợp vẫn có thể tiếp tục đến khi đạt Workspace budget. Chỉ Owner/Manager được quyền chỉnh cấu hình; mã nguồn và CI không thể kiểm chứng giá trị trong tài khoản. Xem [tài liệu Modal Budgets chính thức](https://modal.com/docs/guide/budgets). Với gói Team/Enterprise, có thể bổ sung Environment compute budget nhưng không thay thế giới hạn cấp Workspace.

Deployment mẫu chỉ cho phép **một CPU gateway container** và tối đa **một GPU worker container cho mỗi model**, với `min_containers=0`. Gateway giới hạn mặc định **120 embedding items/phút** và **4 request đồng thời** (HTTP 429 kèm `Retry-After`); mỗi item của batch tính một đơn vị. Bộ đếm trong RAM đặt lại khi restart nên đây **không phải quota bền vững toàn cục, quota theo client hay trần tiền**. Deployment/môi trường chạy chồng nhau vẫn có thể phát sinh chi phí.

CPU gateway nhận **16 HTTP input đồng thời** trong container đó, dành chỗ cho health check và phản hồi quota ngay khi các lời gọi GPU đang chờ. API đồng bộ chỉ nhận tối đa 4 request inference theo mặc định; mỗi GPU worker xử lý từng input một. Khi có mười request đồng bộ chồng nhau và bốn slot đang bận, sáu request còn lại nhận HTTP 429 mà không vào hàng đợi GPU. Nếu tăng giới hạn inference, phải giữ dưới 16 để dành dung lượng cho request điều khiển; cấu hình không hợp lệ bị từ chối khi khởi động. Xem [input concurrency của Modal](https://modal.com/docs/guide/concurrent-inputs).

`VISION_EMBEDDINGS_INFERENCE_TIMEOUT_SECONDS` mặc định 60 (tối đa 120), tính cả thời gian chờ cấp GPU và xử lý. Hết hạn sẽ trả HTTP 504 kèm `Retry-After`, giải phóng slot và yêu cầu hủy các input từ xa thuộc request. Mỗi worker xử lý một input tại một thời điểm nên có thể dừng container bị ảnh hưởng mà không ngắt input đồng thời khác. Nếu một nhóm trong batch lỗi, công việc còn dở ở nhóm còn lại cũng được hủy. Bước dọn dẹp có giới hạn có thể làm phản hồi chậm thêm tối đa 5 giây.



Nếu phục vụ nhiều client, cần **rate limiting và quota theo từng client ở ingress/proxy có xác thực** trước gateway; không cung cấp Bearer token ứng dụng cho caller không tin cậy. Theo dõi chi phí và chuẩn bị cách tắt dịch vụ (`modal app stop multimodal-embeddings-ai-inference`). Trước khi chia sẻ dịch vụ với người khác, xác minh Workspace budget, spend limit và các giới hạn ingress trong tài khoản Modal của bạn.

## Job async lưu trạng thái và thời gian chờ GPU

Với client mới, dùng [endpoint async](API_CONTRACT.vi.md#gpu-job-bất-đồng-bộ-cold-start-và-khôi-phục)
thay vì POST nhiều lần vào API đồng bộ có deadline 60 giây.
CPU function `run_embedding_job` tách biệt chờ GPU với `timeout=3600`;
hai L40S GPU worker vẫn `min_containers=0` để tránh phí luôn bật.
Modal Dict mặc định `multimodal-embeddings-async-jobs` lưu trạng thái,
vector hoàn thành; có thể thay tên bằng `VISION_EMBEDDINGS_JOB_DICT_NAME`.

Submission async mới dùng một gateway request slot ngắn hạn khi chờ ACK
dispatch, đồng thời đặt **durable global/per-client outstanding-job permits**
đến khi có trạng thái cuối chắc chắn. Hai loại này độc lập với sync inflight.
Coordinator nhận Secret revision model, đối chiếu encoding profile trước
khi gửi GPU. Khi job còn bất định, permit có thể bị giữ đến khi đối soát
thủ công; không xóa permit chỉ vì heartbeat quá hạn.

Không xóa Dict trước khi chuyển kết quả vào kho dữ liệu bền vững.
Bản ghi hết hạn sau bảy ngày không truy cập. Khi không chắc dispatch hoặc ACK
phía client, dùng **GET /v1/embeddings/jobs/by-request/{request_id}** có
xác thực để tìm job ID theo namespace client, hoặc GET job ID đã lưu.
Kiểm tra deployment, không POST lại chỉ vì mất ACK: request giống hệt vẫn có thể tạo công việc tính phí mới khi bản ghi
đã mất hoặc hết hạn. Cần điều tra và chủ động quyết định trước khi gửi công
việc mới trong trường hợp này. Không bảo đảm GPU luôn có sẵn
hoặc tự retry job lỗi. GPU có thể tiếp tục tính phí khi client đã ngắt.
Mock server CPU chưa cấu hình Modal job coordinator; dùng API đồng bộ
để thử mock offline.

## Cập nhật revision checkpoint và worker

Triển khai gateway, coordinator tách rời và cả hai lớp GPU worker từ cùng mã
nguồn sau khi đặt Secret revision đã ghim và tải snapshot tương ứng. Thay các
container worker cũ trong lần triển khai đó; đổi Secret không đổi nhãn cho trọng
số đã có trong bộ nhớ. Không thay nội dung Volume model khi worker đang nạp
model từ đó.

Supervisor đối chiếu revision mong đợi của request với identity được chốt khi
subprocess nạp model trước khi encode; service kiểm tra model/repository/SHA trả
về trước khi nhận vector. Worker warm còn SHA cũ hoặc worker cũ thiếu identity
sẽ bị từ chối. Cần triển khai lại toàn bộ app thay vì chỉ cập nhật coordinator.
ID async đã thất bại vẫn ở trạng thái kết thúc: kiểm tra nguyên nhân trước khi
chủ động cho phép một job mới có thể phát sinh phí.

## Log, độ tin cậy và chi phí

```bash
modal app list
modal app logs multimodal-embeddings-ai-inference
```

Coi log của nhà cung cấp là dữ liệu vận hành riêng tư. Không ghi token, ảnh tải lên, embedding thô hoặc metadata có thể định danh đầu vào.
Request có thể phát sinh độ trễ cold-start và thời gian điều phối GPU. Cần tính chi phí GPU, CPU, lưu trữ và mạng; cấu hình giới hạn ngân sách, tần suất trước khi mở truy cập cho người dùng bên ngoài.

Các trường API được mô tả trong [hợp đồng API](API_CONTRACT.vi.md); quy trình kiểm thử được mô tả trong [Nghiệm thu](VERIFICATION.vi.md).
