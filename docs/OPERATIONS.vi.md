# Triển khai và vận hành Modal

🌐 Ngôn ngữ / Language: [English](OPERATIONS.md) | **Tiếng Việt**

Các bước dưới đây hướng dẫn triển khai và vận hành **API Multimodal Embeddings AI Inference** trên Modal.

## Cấu trúc hệ thống

- FastAPI gateway chạy CPU nhận request HTTPS có xác thực Bearer token.
- Hai GPU worker độc lập, mỗi worker sử dụng một NVIDIA L40S với `max_containers=1` và chỉ nạp model được phân công; không chuyển model trong cùng worker.
- Hai checkpoint model nằm trong hai Modal Volume riêng, tách biệt với VRAM của GPU.

## Tên định danh và quyền sở hữu tài nguyên

- Biến môi trường sử dụng tiền tố `VISION_EMBEDDINGS_` (ví dụ `VISION_EMBEDDINGS_API_TOKEN`).
- Python import package: `multimodal_embeddings_ai_inference`.
- Modal App: `multimodal-embeddings-ai-inference`.
- Modal Secret: `multimodal-embeddings-api`.
- Các Modal Volume: `multimodal-embeddings-qwen-weights` và `multimodal-embeddings-wemm-weights`.

Cấu hình `VISION_EMBEDDINGS_QWEN_VOLUME_NAME / VISION_EMBEDDINGS_WEMM_VOLUME_NAME` nhất quán cho cả quá trình tải checkpoint và inference trên GPU. Chạy `scripts/create_modal_secret.py` để tạo Secret xác thực API.

## Cài đặt

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m pip install -e '.[modal]'
modal setup
```

Các lệnh này chạy trên máy cục bộ mà không cần GPU; Modal thực hiện việc tải model và inference từ xa.

## Tải và cố định checkpoint

```bash
modal run modal_models.py --selection both
```

Lưu revision bất biến cho [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B). Các revision đã được xác minh là:

- `Qwen/Qwen3-VL-Embedding-8B`: `2c4565515e0f265c6511776e7193b22c0968ddc7`.
- `tencent/WeMM-Embedding-9B`: `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`.

Các SHA xác định snapshot model upstream công khai, **không phải access token**. Không tự ý thay bằng nhánh `main` có thể thay đổi khi cần tái lập kết quả.

## Cấu hình xác thực

```bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482
```

Ví dụ ghim các revision checkpoint cụ thể. Hãy đối chiếu revision của checkpoint đã tải và thay các SHA trong ví dụ nếu khác trước khi tạo Secret.

Helper tạo Bearer token API, đăng ký Modal Secret và lưu các giá trị tương ứng trên **máy chạy lệnh**, với quyền chỉ chủ sở hữu được đọc. Thông báo hoàn tất chỉ hiển thị đường dẫn file, không hiển thị token. Dùng file này khi cấu hình các client; không in token, commit file hoặc dán vào code notebook công khai.

Để chạy smoke test trên chính máy đó, nạp file mà không hiển thị nội dung:

```bash
TOKEN_FILE="/duong/dan/duoc/helper/hien/thi"
set -a
source "$TOKEN_FILE"
set +a
```

Với Colab/Kaggle hoặc máy khác, chỉ chuyển API token qua secret manager đáng tin cậy hoặc ô nhập ẩn. Không chuyển file cấu hình chứa revision và token lên notebook công khai.

Nạp file cũng xuất hai biến `VISION_EMBEDDINGS_QWEN8_REVISION` và `VISION_EMBEDDINGS_WEMM9_REVISION`. Smoke test HTTPS trong README yêu cầu các SHA checkpoint mong đợi này; cũng có thể truyền qua `--qwen-revision` và `--wemm-revision`. Script từ chối redirect và revision không khớp trước khi báo PASS cho model tương ứng.

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

## Kiểm thử nghiệm thu

```bash
.venv/bin/ruff check src modal_app.py modal_models.py scripts tests
.venv/bin/python -m pytest -q
python scripts/smoke_modal_gpu.py --image /path/to/authorized-image.png
```

Smoke test GPU thật phải trả về vector hữu hạn, chuẩn hóa L2 cho cả hai mã model và chuyển qua lại thành công. Đường HTTPS cũng phải được kiểm tra bằng request có xác thực tới `/v1/embeddings`.

## Kiểm soát truy cập và chi phí

Nếu deployment phục vụ nhiều client, hãy cấu hình **rate limiting và quota theo từng client tại ingress/proxy có xác thực**. Chỉ cấp quyền cho người gọi tin cậy, theo dõi mức sử dụng và cảnh báo thanh toán, đồng thời giới hạn GPU scaling/concurrency theo ngân sách. Chuẩn bị cách tắt deployment kịp thời. Cảnh báo ngân sách **không** phải trần chi phí cứng. Dịch vụ **không** thực thi quota toàn cục hoặc theo client, vì vậy không phát chung Bearer token cho client không tin cậy.

## Log, độ tin cậy và chi phí

```bash
modal app list
modal app logs multimodal-embeddings-ai-inference
```

Coi log của nhà cung cấp là dữ liệu vận hành riêng tư. Không ghi token, ảnh tải lên, embedding thô hoặc metadata có thể định danh đầu vào.
Request có thể phát sinh độ trễ cold-start và thời gian điều phối GPU. Cần tính chi phí GPU, CPU, lưu trữ và mạng; cấu hình giới hạn ngân sách, tần suất trước khi mở truy cập cho người dùng bên ngoài.

Các trường API được mô tả trong [hợp đồng API](API_CONTRACT.vi.md); quy trình kiểm thử được mô tả trong [Nghiệm thu](VERIFICATION.vi.md).
