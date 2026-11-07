
# Multimodal Embeddings AI Inference

**API tạo vector embedding cho ảnh và văn bản**, sử dụng [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B). Được thiết kế để tự triển khai với FastAPI gateway và hai GPU worker NVIDIA L40S độc lập trên Modal.

[![CI](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/actions/workflows/ci.yml) ![Version](https://img.shields.io/badge/version-1.0.0-2563EB) [![License: MIT](https://img.shields.io/badge/license-MIT-16A34A)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-inference-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![Modal L40S](https://img.shields.io/badge/Modal-NVIDIA%20L40S-111827)](https://modal.com/) [![Image and text](https://img.shields.io/badge/input-image%20%2B%20text-0F766E)](docs/API_CONTRACT.vi.md)

🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

**Chọn cách bắt đầu**

- **Thử cục bộ:** Chạy [quickstart mock](#bắt-đầu-nhanh-kiểm-thử-hợp-đồng-api-cục-bộ), không cần GPU hoặc tài khoản Modal.
- **Gọi API đã có:** Dùng [notebook CPU](#notebook-tương-tác-cho-kaggle-và-google-colab) hoặc [HTTP client tối giản](docs/CLIENT_INTEGRATION.vi.md), với URL HTTPS và Bearer token ứng dụng (mã truy cập API) của bạn.
- **Tự host model:** Xem [hướng dẫn triển khai Modal](#triển-khai-inference-thật-trên-modal) để tải checkpoint và phục vụ cả hai model.

## Tổng quan

Ứng dụng viết bằng **Python, Node.js, C#/.NET, Java, Go hoặc ngôn ngữ khác** đều có thể gọi API để tạo embedding ảnh/văn bản bằng HTTP và JSON tiêu chuẩn.

- **Đầu vào:** ảnh PNG/JPEG/WebP mã hóa Base64 hoặc văn bản UTF-8.
- **Đầu ra:** vector số đã chuẩn hóa, mã model, revision checkpoint, SHA256 đầu vào, số chiều và thời gian xử lý.
- **Phục vụ inference:** FastAPI gateway chạy CPU và hai GPU worker NVIDIA L40S chuyên biệt, mỗi worker chỉ nạp một model.
- **Xác thực:** Bearer token ở tầng ứng dụng. Ứng dụng production nên giữ token ở backend; notebook Colab/Kaggle cá nhân dùng runtime riêng và ô nhập ẩn, không lưu token trong cell công khai.

## Gọi API đã triển khai

Dự án cung cấp mã nguồn để triển khai dịch vụ AI inference qua HTTPS trên Modal. Sau khi triển khai, bạn có thể gọi API từ Python, Ruby on Rails, Node.js hoặc bất kỳ HTTP client nào bằng HTTPS endpoint và Bearer token ứng dụng (`VISION_EMBEDDINGS_API_TOKEN`) của deployment. Ví dụ sau sử dụng mã API của **tencent/WeMM-Embedding-9B**:

```bash
curl -X POST 'https://YOUR_DEPLOYMENT.modal.run/v1/embeddings' \
  -H "Authorization: Bearer $VISION_EMBEDDINGS_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"wemm-embedding-9b","text":"a mountain landscape","dimension":1024}'
```

Với đầu vào là ảnh, gửi `image_base64` thay cho `text`. Xem [hợp đồng API](docs/API_CONTRACT.vi.md) để biết các trường phản hồi, giới hạn và lỗi.

## Ví dụ ứng dụng

- **Tìm ảnh bằng văn bản:** tra cứu kho ảnh đã lập chỉ mục bằng vector văn bản và hình ảnh từ cùng model và revision.
- **Tìm ảnh tương đồng:** tìm các ảnh liên quan trong bộ sưu tập được phép sử dụng.
- **Tích hợp ứng dụng:** gọi API HTTPS có xác thực từ backend và lưu vector vào chỉ mục tìm kiếm riêng.

Đây là các trường hợp sử dụng tiềm năng, không phải kết quả đo độ chính xác. Cần đánh giá chất lượng retrieval trên dữ liệu đại diện cho từng tác vụ.

## Model được hỗ trợ

| Tên đầy đủ trên Hugging Face | Mã API | Tham số | Số chiều hỗ trợ |
| --- | --- | ---: | --- |
| [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | `qwen3-vl-embedding-8b` | 8B | 64–4096 |
| [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) | `wemm-embedding-9b` | 9B | 64, 128, 256, 512, 1024, 2048, 4096 |

Cả hai model mặc định trả về **1.024 chiều**. Không gian vector của chúng độc lập: không trộn embedding giữa các model, revision hoặc quy tắc tiền xử lý khác nhau.

## Kiến trúc

```text
Ứng dụng HTTPS ---> FastAPI Gateway (CPU / Bearer)
                           |
                    định tuyến model
                       /          \
           Qwen worker             WeMM worker
             (L40S)                   (L40S)
               |                        |
           Volume Qwen              Volume WeMM
               |                        |
   Qwen3-VL-Embedding-8B      Tencent WeMM-Embedding-9B
```

Mỗi GPU worker chỉ nạp model được phân công; hai worker scale-to-zero độc lập.

## Yêu cầu

- **Thử mock cục bộ:** Python 3.11+, Git, curl và Bash (WSL nếu dùng Windows). Không cần GPU, tài khoản Modal hoặc API token.
- **Notebook gọi API đã có:** Python 3 CPU runtime trên Kaggle/Colab, kết nối Internet, HTTPS endpoint đã triển khai và application Bearer token. Không cần Modal CLI hoặc model weights.
- **Tự triển khai GPU inference:** Các công cụ cục bộ ở trên, tài khoản Modal có quota L40S và lưu trữ, quyền truy cập repository model chính thức trên Hugging Face cùng nơi quản lý secret an toàn.

## Bắt đầu nhanh: kiểm thử hợp đồng API cục bộ

Trong Bash terminal (bao gồm Google Colab hoặc Kaggle), chạy:

```bash
git clone https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference.git
cd Multimodal-Embeddings-AI-Inference
bash scripts/setup.sh
bash scripts/quickstart_mock.sh
```

**Lệnh setup làm gì:** `scripts/setup.sh` tạo môi trường Python ảo, cài các dependency phát triển, biên dịch kiểm tra mã Python, chạy Ruff và toàn bộ pytest hiện có. Đây là bước thiết lập cho contributor/phát triển nên có thể sinh nhiều output; lệnh **không** khởi chạy inference hay sử dụng GPU. Sau đó, `scripts/quickstart_mock.sh` mới là phép thử hợp đồng API mock ngắn gọn.

Nếu đã clone repository, bỏ qua `git clone` và `cd`; chạy hai lệnh cuối từ thư mục gốc repository. Script kiểm thử tự tạo một token tạm, khởi động mock API, xác minh embedding response có xác thực rồi dừng server. Không cần terminal thứ hai hoặc cấu hình token thủ công.

Kiểm thử thành công trả về `"pass": true` và `"backend": "mock"`. Đây **không phải inference thật của Qwen3-VL-Embedding-8B/Tencent WeMM-Embedding-9B**; muốn thử endpoint Modal đã triển khai cần URL HTTPS và API Bearer token riêng của deployment đó.

Setup và CI dùng bộ phiên bản CPU trong [constraints/cpu.txt](constraints/cpu.txt). Xem [Đóng góp](CONTRIBUTING.vi.md) để tái tạo môi trường; dependency GPU được quản lý riêng.

## Notebook tương tác cho Kaggle và Google Colab

[Notebook API song ngữ Anh–Việt](notebooks/multimodal_embeddings_api_demo.ipynb) chạy trên CPU, không quản lý tài nguyên Modal.

- **Cần có:** HTTPS endpoint và Bearer token của deployment do bạn triển khai hoặc được cấp quyền truy cập. Có thể xem trước ảnh mẫu **không cần URL API hoặc token**.
- **Thực hiện:** Chọn Qwen3-VL-Embedding-8B (mặc định) hoặc Tencent WeMM-Embedding-9B, chạy **Model configuration** và cell thu gọn **Configuration helpers**, tùy chỉnh ba cặp text/ảnh trong **Customize samples** (nếu cần), xem trước ảnh mẫu rồi nhập thông tin xác thực trong **Kết nối API**.
- **Kết quả:** Sáu request được xác minh và ma trận cosine 3 × 3 có nhãn. Để tích hợp một request không cần helper notebook, xem [HTTP client Python tối giản](docs/CLIENT_INTEGRATION.vi.md#python-311-thư-viện-chuẩn). Xem [hướng dẫn tiếng Việt](docs/NOTEBOOK_GUIDE.vi.md) hoặc [bản tiếng Anh](docs/NOTEBOOK_GUIDE.md).

**Dành cho người đóng góp:** Các code cell của notebook có [mã Python để review](notebooks/source/) được đối chiếu bằng `python scripts/sync_notebooks.py --check`. Notebook công bố vẫn chạy độc lập. Bản demo có bảng xem trước bố cục kết quả để trống, không sử dụng điểm tương đồng giả.

**Mở notebook API:** [Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_api_demo.ipynb) · [Notebook trên GitHub](notebooks/multimodal_embeddings_api_demo.ipynb). Người dùng Kaggle có thể import file notebook.

## Benchmark hiệu năng inference

[Notebook benchmark](notebooks/multimodal_embeddings_benchmark.ipynb) so sánh hành vi phục vụ API của Qwen3-VL-Embedding-8B và Tencent WeMM-Embedding-9B.

- **Đo:** Độ trễ warm cân bằng, cold-idle khi chủ động bật và định tuyến giữa hai GPU worker; không đo chất lượng tìm kiếm hay độ chính xác nhận diện.
- **Cần:** Deployment inference HTTPS của chính bạn có xác thực; không cần dataset hay GPU cục bộ.
- **Mặc định:** `RUN_BENCHMARK=False` không gửi request. Bật lên với cấu hình an toàn sẽ gửi **24 POST request** (8 warm-up + 16 lần đo text/ảnh; tối đa 4 mẫu mỗi model/loại dữ liệu, chỉ phù hợp smoke check chứ không đại diện p95 ổn định) mà không chờ cold. Xem [giao thức benchmark](benchmarks/README.vi.md) để biết cấu hình và giới hạn.

**Chạy lại và nút Stop:** Không thể tiếp tục phase đã ngắt trong cùng lượt đo. Bắt đầu lượt đo mới tại **Cấu hình → API client/preflight**. Xem [phục hồi benchmark và giới hạn](benchmarks/README.vi.md).

**Mở notebook benchmark:** [Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_benchmark.ipynb) · [Notebook trên GitHub](notebooks/multimodal_embeddings_benchmark.ipynb).

## Triển khai inference thật trên Modal

Cần tài khoản Modal riêng, quota L40S/lưu trữ và quyền truy cập hai repository model. Tải checkpoint và chạy inference có thể phát sinh chi phí.

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m pip install -e '.[modal]'
modal setup
modal run modal_models.py --selection both
```

Trước khi triển khai, làm theo [Cấu hình xác thực](docs/OPERATIONS.vi.md#cấu-hình-xác-thực) để tạo Secret ứng dụng với các revision bất biến vừa tải. Phần đó cũng hướng dẫn nạp file token chỉ chủ sở hữu được đọc mà không hiển thị nội dung.

```bash
modal deploy modal_app.py
```

Dùng URL HTTPS của deployment riêng. Sau khi nạp token và các revision đã ghim từ file chỉ chủ sở hữu được đọc theo hướng dẫn trên, kiểm tra cả hai model bằng ảnh được phép xử lý:

```bash
python scripts/smoke_both.py \
  --base-url 'https://YOUR_DEPLOYMENT.modal.run' \
  --image /path/to/authorized-image.png --dimension 1024
```

Script đọc SHA checkpoint mong đợi từ `VISION_EMBEDDINGS_QWEN8_REVISION` và `VISION_EMBEDDINGS_WEMM9_REVISION`; hoặc truyền qua `--qwen-revision` và `--wemm-revision`. Thiếu revision hoặc revision không khớp sẽ dừng kiểm tra. Script từ chối HTTP redirect.

Thành công là mỗi model có một bản ghi `PASS` với `revision_verified: true` và vector đã chuẩn hóa. Xem [Vận hành](docs/OPERATIONS.vi.md) để xoay vòng, thu hồi token, redeploy và cấp quyền cho client; xem [Nghiệm thu](docs/VERIFICATION.vi.md#truy-vết-mã-nguồn-nghiệm-thu) để lưu bằng chứng có thể truy vết.

## Các endpoint REST

| Phương thức | Endpoint | Mục đích |
| --- | --- | --- |
| `GET` | `/v1/health` | Chỉ kiểm tra gateway còn phản hồi |
| `GET` | `/v1/models` | Danh sách model và revision |
| `GET` | `/v1/models/{model_id}` | Thông số của một model |
| `POST` | `/v1/embeddings` | Mã hóa một ảnh hoặc văn bản thành vector |
| `POST` | `/v1/embeddings/batches` | Tạo embedding theo batch có giới hạn |

### Ví dụ response rút gọn

JSON minh họa đã rút gọn, không phải phản hồi đo thực tế. Các giá trị vector và độ trễ là ví dụ; vector thật có đủ 1.024 phần tử.

```json
{
  "model": "wemm-embedding-9b",
  "revision": "<checkpoint-revision>",
  "dimension": 1024,
  "vector": [0.0123, -0.0456, 0.0789],
  "normalized": true,
  "input_sha256": "<computed SHA-256 of the input>",
  "elapsed_ms": 152.4,
  "embedding_metadata": {
    "config_sha256": "<embedding configuration SHA-256>"
  }
}
```

Xem [schema API đầy đủ](docs/API_CONTRACT.vi.md).

## Instruction embedding theo từng tác vụ

**Qwen3-VL-Embedding-8B** nhận instruction tùy chọn cho tìm kiếm tài liệu ngữ nghĩa, tìm ảnh và RAG đa phương thức. Khi không gửi trường này, API sử dụng instruction mặc định của Qwen: “Represent the image or text for retrieval in a visual gallery.” Muốn tái lập vector cần giữ nguyên instruction, revision model và quy tắc tiền xử lý.
**Tencent WeMM-Embedding-9B** không nhận trường `instruction` tùy chỉnh trong API này; request như vậy nhận HTTP 422. Response có metadata mã băm phục vụ truy vết cấu hình nhưng không trả instruction gốc. Xem [hợp đồng API](docs/API_CONTRACT.vi.md).

## Tích hợp, quyền riêng tư và giới hạn

- Xem [ví dụ tích hợp Python, Node.js và C#/.NET](docs/CLIENT_INTEGRATION.vi.md) để gọi API bằng HTTP client.
- Ứng dụng production phải giữ token ở backend. Với notebook Kaggle/Colab cá nhân, nhập qua prompt ẩn trong runtime riêng và xóa output trước khi chia sẻ. Không ghi log ảnh gốc, Base64, embedding hoặc header xác thực.
- Dịch vụ chưa tích hợp rate limiting hoặc giới hạn chi phí. Hãy giới hạn client và cấu hình bảo vệ trên nền tảng trước khi mở rộng truy cập.
- Điểm cosine **không phải xác suất hay mức độ tin cậy đã hiệu chuẩn**. Cần kiểm tra chất lượng retrieval bằng dữ liệu gán nhãn đại diện.

## Tài liệu

> **Kiểm thử và xác minh:** Repository cung cấp bộ kiểm thử tự động cho API, công cụ kiểm tra inference và hướng dẫn xác minh sau triển khai. Xem [Tài liệu xác minh](docs/VERIFICATION.vi.md) để biết thêm chi tiết.

- [Vận hành](docs/OPERATIONS.vi.md) · [Nghiệm thu](docs/VERIFICATION.vi.md) · [Khắc phục lỗi](docs/TROUBLESHOOTING.vi.md) · [HTTP clients](docs/CLIENT_INTEGRATION.vi.md)
- [Hợp đồng API](docs/API_CONTRACT.vi.md) · [Model Volumes](docs/MODEL_VOLUMES.vi.md) · [Bảo mật](SECURITY.vi.md)
- [Đóng góp](CONTRIBUTING.vi.md) · [Lịch sử phát hành](CHANGELOG.vi.md) · [Ghi chú giấy phép](LICENSE-NOTES.vi.md)
- [Hướng dẫn notebook](docs/NOTEBOOK_GUIDE.vi.md) · [Hỗ trợ](.github/SUPPORT.md) · [Quy tắc ứng xử](.github/CODE_OF_CONDUCT.md) · [Gửi issue](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/issues/new/choose)

## Giấy phép và tác giả

Mã nguồn và tài liệu do dự án sở hữu được cấp phép MIT; model weights và dependencies bên thứ ba tuân theo điều khoản riêng. Repository không phân phối model weights hoặc dataset của người dùng.

**Tác giả duy trì:** Đăng Khoa · `i.am@dangkhoa.dev`.
