# Kiểm thử và nghiệm thu

🌐 Ngôn ngữ / Language: [English](VERIFICATION.md) | **Tiếng Việt**

Tài liệu này hướng dẫn kiểm tra một deployment của Multimodal Embeddings AI Inference API. Hãy thực hiện các bước trên môi trường do bạn quản lý. Các lệnh mẫu và kết quả mong đợi là **quy trình xác minh**, không phải báo cáo hiệu năng hoặc chứng nhận cho một deployment cụ thể.

## Model embedding được hỗ trợ

Dịch vụ hỗ trợ hai model tạo vector embedding từ ảnh và văn bản, mỗi model có không gian vector riêng:

| Tên đầy đủ trên Hugging Face | Mã model trong request API | Mục đích |
| --- | --- | --- |
| [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | `qwen3-vl-embedding-8b` | Tạo embedding từ ảnh hoặc văn bản để tìm kiếm ngữ nghĩa và tìm kiếm đa phương thức |
| [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) | `wemm-embedding-9b` | Tạo embedding từ ảnh hoặc văn bản để tìm ảnh tương đồng và tìm ảnh bằng văn bản |

“Qwen” và “WeMM” chỉ là tên gọi ngắn trong tài liệu; **mã model** mới là giá trị chính xác cần truyền vào trường `model` trong request API. Không trộn vector do hai model tạo ra vào cùng một chỉ mục.

## Kiểm thử CPU cục bộ

Từ thư mục gốc repository, chạy setup **khi cài đặt lần đầu** hoặc khi cần cập nhật môi trường Python và dependency:

```bash
bash scripts/setup.sh
```

Script setup tạo hoặc sử dụng lại `.venv`, cài đặt/cập nhật dependency CPU, rồi chạy Ruff và pytest. Thông thường có thể chạy lại, nhưng việc này có thể truy cập kho package, tốn thời gian và cập nhật dependency trong giới hạn cấu hình; **không cần chạy setup trước mỗi lần kiểm tra mock**.

Sau khi setup thành công, có thể chạy kiểm thử mock nhiều lần khi cần:

```bash
bash scripts/quickstart_mock.sh
python scripts/audit_release.py
python scripts/update_sha256_manifest.py --check
sha256sum -c SHA256SUMS.txt
```

Mỗi lần chạy quickstart tự tạo Bearer token tạm, chọn cổng localhost còn trống, khởi động API mock, xác minh phản hồi rồi dừng server. Log tạm được xóa khi thành công; nếu gặp lỗi, log được giữ lại và đường dẫn được in ra để kiểm tra. Script **không** tải checkpoint của hai model hoặc dùng GPU. Mock PASS chỉ xác minh contract API cục bộ, không đánh giá chất lượng embedding thực tế.

| Kiểm tra | Kết quả mong đợi và cách xác minh |
| --- | --- |
| Kiểm thử CPU (pytest, Ruff) | Chỉ ghi **PASS** khi các lệnh hoàn tất thành công. Cần kiểm tra [CI workflow](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/actions/workflows/ci.yml) gắn đúng source commit. |

Nếu thiếu bằng chứng chạy trực tiếp, hãy ghi trạng thái GPU hoặc giao diện notebook tương ứng là **chưa xác minh độc lập**.

## Xác minh inference đã triển khai

Sử dụng HTTPS endpoint do bạn vận hành hoặc được phép truy cập. Cấu hình Bearer token ứng dụng và checkpoint revision dự kiến theo [Vận hành](OPERATIONS.vi.md#cấu-hình-xác-thực). Chạy smoke test với ảnh được phép xử lý:

```bash
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png --dimension 1024
```

Thiết lập `VISION_EMBEDDINGS_QWEN8_REVISION` và `VISION_EMBEDDINGS_WEMM9_REVISION`, hoặc dùng hai tham số tương ứng `--qwen-revision` và `--wemm-revision`.

Với mỗi model, cần xác nhận:

- Request không có xác thực hợp lệ bị từ chối; request được cấp quyền thành công.
- Model ID và checkpoint revision khớp cấu hình dự kiến (`revision_verified: true`).
- Vector chứa số hữu hạn, có đúng số chiều yêu cầu và được chuẩn hóa L2.
- Đầu vào văn bản và ảnh hợp lệ được định tuyến tới đúng model; lỗi không tiết lộ dữ liệu nhạy cảm.

Mock PASS không chứng minh GPU thực tế hoạt động hoặc chất lượng tìm kiếm.

## Xác minh notebook trên nền tảng lưu trữ

[Notebook demo API](../notebooks/multimodal_embeddings_api_demo.ipynb) và [notebook hiệu năng](../notebooks/multimodal_embeddings_benchmark.ipynb) gọi API HTTPS có sẵn, không tự triển khai GPU.

Khi sử dụng Kaggle, Colab hoặc Jupyter, hãy kiểm tra ô cấu hình chỉnh sửa được, ảnh xem trước, biểu mẫu nhập token được bảo vệ, bước xác minh capability của model, dữ liệu phản hồi và hiển thị báo cáo. Kiểm tra khả năng phục hồi sau cấu hình sai hoặc request bị ngắt mà không tự gửi lại POST có thể đã hoàn tất.

Notebook benchmark mặc định `RUN_BENCHMARK=False`. Chỉ bật các phép thử được cho phép, tính đến chi phí GPU và ghi rõ từng loại phép đo.

## Truy vết mã nguồn nghiệm thu

Đối với mỗi phiên xác minh thực tế, ghi lại source commit của ứng dụng, checkpoint revision, thời gian UTC, môi trường, trường hợp kiểm thử và kết quả đã lọc dữ liệu nhạy cảm. Source của client và server được triển khai có thể khác nhau; cần ghi riêng khi phù hợp.

```bash
git rev-parse HEAD
sha256sum -c SHA256SUMS.txt
python scripts/record_acceptance.py --run-checks --output /tmp/acceptance-session.json
```

Công cụ ghi nhận nghiệm thu chỉ thu thập thông tin source và kiểm thử CPU; **không** gọi inference endpoint hoặc tự xác minh GPU hay giao diện notebook. Bằng chứng CI phải gắn với đúng commit thay vì chỉ dựa vào badge chung. Có thể dùng [mẫu báo cáo nghiệm thu](ACCEPTANCE_TEMPLATE.vi.md) khi cần.

Không công bố token, URL dịch vụ riêng, ảnh riêng tư, dữ liệu đầu vào gốc, vector embedding thô hoặc thông tin định danh worker.

## Phạm vi và giới hạn

Kiểm thử chức năng chỉ xác minh hành vi của cấu hình đã chạy. Chúng không chứng minh độ chính xác retrieval, mức độ tin cậy nhận dạng sinh trắc, độ sẵn sàng production, throughput duy trì, độ trễ p95 hay chi phí mỗi request. Cần đánh giá retrieval riêng bằng dữ liệu đại diện được phép sử dụng và tách vector của các model hoặc cấu hình khác nhau.

Xem [Vận hành](OPERATIONS.vi.md) và [Bảo mật](../SECURITY.vi.md).
