# Kiểm tra deployment của bạn

🌐 Ngôn ngữ / Language: [English](VERIFICATION.md) | **Tiếng Việt**

Dùng các bước này sau khi triển khai hoặc cập nhật Multimodal Embeddings AI Inference API. Chúng giúp kiểm tra xác thực, phản hồi model và kết nối notebook trước khi tích hợp ứng dụng.

## Model và thông tin kết nối

Dịch vụ cung cấp [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) với mã `qwen3-vl-embedding-8b` và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) với mã `wemm-embedding-9b`. Mỗi model có không gian vector riêng.

Chuẩn bị URL HTTPS, API Bearer token và checkpoint revision dự kiến của deployment. Nếu chưa triển khai, bắt đầu với [Vận hành](OPERATIONS.vi.md). Tài liệu đó cũng hướng dẫn nạp token và revision đã lưu vào shell.

## Kiểm tra tài nguyên và gateway trước khi chạy GPU

Trong **cùng Modal Workspace và Environment** đã tải checkpoint, tạo Secret
và deploy, kiểm tra tên tài nguyên mà không tạo inference:

```bash
modal app list
modal volume list
modal secret list
```

Cần thấy App, hai Volume model và hai Secret riêng cho API/revision model.
Chỉ thấy tài nguyên tồn tại **không có nghĩa** model đã nạp được. Xem
[triển khai lần đầu](OPERATIONS.vi.md#triển-khai-lần-đầu-điều-kiện-và-trình-tự)
để chọn profile và Environment.

Nạp credential riêng trên máy do `scripts/create_modal_secret.py` tạo,
như trong [Vận hành](OPERATIONS.vi.md#cấu-hình-xác-thực).
Không ghi nội dung file vào log hoặc output notebook chia sẻ.
Với URL thực tế do `modal deploy` trả về, kiểm tra CPU gateway mà
không yêu cầu GPU embedding:

```bash
curl --fail-with-body --silent --show-error \
  'https://YOUR_DEPLOYMENT.modal.run/health'
```

Phản hồi health công khai chỉ xác minh CPU gateway, không chứng minh GPU
đã sẵn sàng. Danh sách model cũng công khai, chỉ đọc. Inference, async jobs,
thông tin riêng từng model và vận hành vẫn cần Bearer token hợp lệ. Thử token
sai liên tục ở endpoint riêng tư có thể khiến client bị tạm khóa.

## Kiểm tra phản hồi model

**Các lệnh sau chạy inference thật trên L40S và có thể phát sinh phí.**
Chọn ảnh mà bạn được phép xử lý. Hai phép kiểm tra phục vụ mục tiêu khác nhau:

```bash
# 1. Gọi trực tiếp GPU worker Qwen và WeMM qua Modal SDK.
#    Không đi qua HTTPS gateway và không kiểm tra Bearer.
python scripts/smoke_modal_gpu.py --image /path/to/authorized-image.png

# 2. Gọi qua HTTPS có xác thực, mặc định async cho cold start.
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png \
  --dimension 1024 --mode async --model both

# Chỉ GET để khôi phục job WeMM đã gửi (không POST mới).
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png \
  --dimension 1024 --model wemm-embedding-9b --resume
```

Kiểm tra GPU trực tiếp phải in `REAL_PASS` cho **cả hai** model, xác thực
vector hữu hạn, L2-normalized và worker ID riêng biệt. Kiểm tra HTTPS phải
in `"status": "PASS"` và `"revision_verified": true` cho **cả hai**
model. Script kiểm tra đúng model, `backend=modal`, revision SHA đã ghim,
digest đầu vào, 1.024 chiều và chuẩn L2. Nạp
`VISION_EMBEDDINGS_QWEN8_REVISION` / `VISION_EMBEDDINGS_WEMM9_REVISION`
từ file riêng, hoặc truyền `--qwen-revision` / `--wemm-revision`.

Phép kiểm tra HTTPS mặc định dùng **async** để xử lý thời gian chờ cấp GPU
và nạp model có thể vượt quá deadline của gateway đồng bộ. Script kiểm tra
`GET /health` và `GET /v1/models` công khai trước khi inference có xác thực;
**hai GET này không làm ấm GPU hoặc chứng minh model đã sẵn sàng**.
Có thể chọn riêng `--model qwen3-vl-embedding-8b` hoặc
`--model wemm-embedding-9b` thay vì chạy cả hai. Script in `request_id`
ổn định **trước** một POST async mới, sau đó chỉ polling bằng GET và tách
thời gian thực tế phía client khỏi timing API/GPU (nếu có).

`--resume` **chỉ GET**, không tạo job mới. Hãy giữ nguyên URL, ảnh,
dimension, model, revision và client xác thực để dựng lại ID tự động,
hoặc truyền lại cùng `--request-id` đã dùng khi gửi job. Nếu chọn
`--model both`, custom ID sẽ được thêm hậu tố riêng cho từng model.
Mất ACK, polling timeout hoặc trạng thái `INDETERMINATE` **không cho phép
gửi lại POST**; cần xem trạng thái job và Modal logs. HTTP 404 cũng có
thể là bản ghi đã hết hạn. Một lần chạy mới không có `--resume` có thể
tạo inference tính phí khi không còn bản ghi; job store không lưu vĩnh viễn.
Chế độ sync cũ vẫn có qua `--mode sync`, nhưng deadline gateway mặc định
**60 giây** bao gồm GPU scheduling; không tự retry HTTP 504. Xem
[API async](API_CONTRACT.vi.md#gpu-job-bất-đồng-bộ-cold-start-và-khôi-phục)
và [hướng dẫn xử lý sự cố](TROUBLESHOOTING.vi.md).

Hai phép smoke không chứng minh retrieval accuracy, SLA hay phiên bản Git
đang chạy thật trên Modal.

## Kiểm tra kết nối notebook

Mở [demo API](../notebooks/multimodal_embeddings_api_demo.ipynb), chuẩn bị dữ liệu mẫu, kết nối deployment rồi chạy **Generate embeddings → Results**. Bạn sẽ thấy sáu phản hồi đã xác minh và báo cáo cosine song ngữ. Làm theo [hướng dẫn notebook](NOTEBOOK_GUIDE.vi.md) khi thay đầu vào hoặc kết nối lại.

Nếu dùng cả Colab và Kaggle, kiểm tra từng nền tảng bạn muốn hỗ trợ. Có thể dùng [notebook hiệu năng](../notebooks/multimodal_embeddings_benchmark.ipynb) để đo latency; mặc định là `RUN_BENCHMARK=False` và chỉ gọi API khi được bật.

## Khám phá API cục bộ

Để thử luồng HTTP trước khi triển khai:

```bash
bash scripts/setup.sh --quick
bash scripts/quickstart_mock.sh
```

Mock cục bộ tự khởi động và tắt. Khi thành công, kết quả có `"pass": true` và `"backend": "mock"`. Chế độ này dùng vector mô phỏng; kết nối dịch vụ đã triển khai để kiểm tra embedding do model tạo ra.

Kiểm thử phát triển, đồng bộ source notebook và bảo trì checksum được trình bày trong [Hướng dẫn đóng góp](../CONTRIBUTING.vi.md).

## Chia sẻ thông tin hữu ích

Với mỗi phiên sử dụng deployment, ghi kết quả quan sát GPU hoặc giao diện notebook cùng môi trường và phiên bản đã dùng khi báo lỗi hoặc chia sẻ kết quả hiệu năng. [Bảng kiểm tra deployment](ACCEPTANCE_TEMPLATE.vi.md) là mẫu tùy chọn giúp sắp xếp các thông tin này.

Nêu model bị ảnh hưởng, bước tái hiện, kết quả mong đợi và thông báo lỗi đã loại thông tin nhạy cảm. Giữ token, URL riêng, ảnh đầu vào và vector embedding thô ngoài báo cáo công khai. Xem [Hỗ trợ](../.github/SUPPORT.md) để chọn kênh báo cáo.

Kiểm tra chức năng bao phủ deployment và cấu hình bạn đã chạy. Đánh giá chất lượng retrieval riêng bằng ví dụ đại diện cho ứng dụng của bạn.
