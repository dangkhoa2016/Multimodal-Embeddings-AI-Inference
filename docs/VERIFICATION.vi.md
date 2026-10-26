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
  -H "Authorization: Bearer $VISION_EMBEDDINGS_API_TOKEN" \
  'https://YOUR_DEPLOYMENT.modal.run/v1/health'
```

Gateway trả phản hồi health có xác thực chỉ chứng minh đường API CPU hoạt
động, chưa kiểm tra model. Request thiếu/sai token phải bị từ chối; không
liên tục thử token sai vì gateway có thể tạm khóa client.

## Kiểm tra phản hồi model

**Các lệnh sau chạy inference thật trên L40S và có thể phát sinh phí.**
Chọn ảnh mà bạn được phép xử lý. Hai phép kiểm tra phục vụ mục tiêu khác nhau:

```bash
# 1. Gọi trực tiếp GPU worker Qwen và WeMM qua Modal SDK.
#    Không đi qua HTTPS gateway và không kiểm tra Bearer.
python scripts/smoke_modal_gpu.py --image /path/to/authorized-image.png

# 2. Gọi qua HTTPS gateway có xác thực tới cả hai model.
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png --dimension 1024
```

Kiểm tra GPU trực tiếp phải in `REAL_PASS` cho **cả hai** model, xác thực
vector hữu hạn, L2-normalized và worker ID riêng biệt. Kiểm tra HTTPS phải
in `"status": "PASS"` và `"revision_verified": true` cho **cả hai**
model. Script kiểm tra đúng model, `backend=modal`, revision SHA đã ghim,
digest đầu vào, 1.024 chiều và chuẩn L2. Nạp
`VISION_EMBEDDINGS_QWEN8_REVISION` / `VISION_EMBEDDINGS_WEMM9_REVISION`
từ file riêng, hoặc truyền `--qwen-revision` / `--wemm-revision`.

Nên kiểm tra HTTPS ngay sau phép gọi GPU trực tiếp khi container còn có
thể đang warm. Endpoint đồng bộ có **deadline server mặc định 60 giây**;
cold-start lần đầu có thể trả HTTP 504 khi chờ cấp GPU/nạp model quá lâu.
POST bị timeout vẫn có thể đã kích hoạt công việc tính phí.
**Không tự động gửi lại**: kiểm tra log Modal và
[hướng dẫn xử lý sự cố](TROUBLESHOOTING.vi.md); với cold-start mới,
ưu tiên [API async dùng request ID ổn định](API_CONTRACT.vi.md#gpu-job-bất-đồng-bộ-cold-start-và-khôi-phục)
và chỉ cho phép GPU job mới khi chủ động chấp nhận phí.
Các phép smoke không chứng minh độ chính xác retrieval, SLA hay Modal đang
chạy đúng Git commit mới nhất.

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
