
# Hướng dẫn notebook — Google Colab và Kaggle

🌐 Ngôn ngữ / Language: [English](NOTEBOOK_GUIDE.md) | **Tiếng Việt**

[Notebook Multimodal Embeddings AI Inference song ngữ Anh–Việt](../notebooks/multimodal_embeddings_api_demo.ipynb) là **HTTP client**, chạy trên CPU, không cần model weights, CUDA binary hay Modal CLI. Inference server được triển khai và quản lý **riêng biệt**.

API hỗ trợ [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B). **Mặc định chọn Qwen3-VL-Embedding-8B; mỗi lượt chạy notebook chỉ sử dụng một model.**

## Mở notebook

[Mở trực tiếp trên Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_api_demo.ipynb) hoặc [xem file notebook](../notebooks/multimodal_embeddings_api_demo.ipynb). Với Kaggle, import file notebook vào một notebook Python 3 CPU mới.

**Điều kiện inference:** Cần URL HTTPS của API đã triển khai và Bearer token ứng dụng.

## Thông tin kết nối bắt buộc

| Biến môi trường | Bắt buộc? | Mô tả |
| --- | --- | --- |
| `VISION_EMBEDDINGS_BASE_URL` | Có | HTTPS endpoint đã deploy, ví dụ `https://YOUR_DEPLOYMENT.modal.run` (không thêm `/v1`) |
| `VISION_EMBEDDINGS_API_TOKEN` | Có | Bearer token của ứng dụng, do người vận hành API cung cấp |

Có thể xem trước ba ảnh nhúng sẵn trước khi cung cấp hai giá trị này. Sau đó, nhập chúng bằng ô ẩn trong mục **API connection**. Nếu đã đặt biến môi trường trong runtime, notebook sẽ ưu tiên dùng giá trị đó. Notebook không hỏi `MODAL_TOKEN_ID` hoặc `MODAL_TOKEN_SECRET`: chúng dành cho quản lý tài nguyên Modal, không phải gọi API này.

**Bảo trì mã notebook:** Các code cell được đồng bộ từ [mã Python dễ review](../notebooks/source/) bằng `python scripts/sync_notebooks.py --check` (hoặc `--write` sau khi cố ý chỉnh source). Notebook công bố vẫn tự chứa: Kaggle và Colab không cần clone repository hoặc tải các file source này.

## Cách chạy

1. Mở notebook bằng **Google Colab** hoặc **Kaggle Notebooks** với runtime Python 3 **CPU**; không cần clone source để chỉ gọi API.
2. Trong mục **Model configuration**, chọn `MODEL_ID` (mặc định **Qwen3-VL-Embedding-8B**), `DIMENSION` và `TIMEOUT_SECONDS`. Colab có dropdown; trên Kaggle, sửa trực tiếp giá trị trong form cấu hình. Tiếp theo chạy cell thu gọn **Configuration helpers**. Biến môi trường tương ứng sẽ được ưu tiên.
3. Tùy chọn sửa **Customize samples** để thay ba câu text và ba đường dẫn ảnh trong cell cấu hình riêng. Sau đó chạy **Prepare input samples** để xem ba ảnh trong **Built-in image previews**: 512 × 512 px (tranh 2D, hình 3D và cảnh phong cách ảnh thật); bước này không cần credentials và không gọi API.
4. Trong mục **API connection**, nhập URL HTTPS đã triển khai và API Bearer token bằng ô ẩn nếu chưa đặt biến môi trường. Chạy kiểm tra xác thực/model catalog, rồi mới chạy ba ví dụ văn bản và ba ví dụ ảnh. Notebook gửi **đúng sáu inference request tuần tự** cho **một model đã chọn**.
5. Xem báo cáo HTML rõ ràng theo thứ tự Anh trước, Việt sau: ma trận cosine 3 × 3 (không phải phần trăm), số chiều, thời gian sáu request, chuẩn L2 và revision. Các bước trước cũng có thẻ trạng thái dễ hiểu. Màu sắc tương phản cao giúp đọc được trên nền tối của Kaggle.
6. Xóa toàn bộ output và khởi động lại runtime trước khi chia sẻ notebook.

## Cell được sửa và cell hỗ trợ

Chỉ sửa **Model configuration**, **Customize samples** và nhập thông tin trong prompt ẩn **Secure API connection**. Chạy các cell thu gọn **Configuration helpers**, **Prepare input samples** và **Results** mà không sửa validation, dữ liệu Base64 hay HTML/CSS. Phần tích hợp HTTP nằm tại **Check API connection** và **Generate embeddings**. Code hỗ trợ được nhúng sẵn nên notebook tiếp tục chạy độc lập trên Kaggle và Colab, không cần cài thêm package.

## Thay dữ liệu của bạn trong notebook

Sau khi chạy **Model configuration** và **Configuration helpers**, mở **Customize samples / Tùy chỉnh dữ liệu mẫu**. Sửa `SAMPLE_TEXT_1` đến `SAMPLE_TEXT_3` và (nếu muốn) `IMAGE_PATH_1` đến `IMAGE_PATH_3`. Đường dẫn trống sẽ dùng ảnh WebP nhúng sẵn. Đường dẫn có giá trị phải là file PNG/JPEG/WebP trong runtime hiện tại mà bạn có quyền xử lý. Ví dụ: Colab `/content/my-image.png` sau khi file đã được tải lên, hoặc Kaggle `/kaggle/input/my-dataset/my-image.png` khi dataset đã được attach. Đây là mẫu đường dẫn, không phải các file có sẵn mặc định.

**Thay đổi kết nối:** Theo bảng phục hồi của notebook. Kết nối lại thất bại sẽ vô hiệu hóa preflight trước đó và chặn inference. Results cũng từ chối kết nối chưa xác minh hoặc vector được tạo từ endpoint/token khác. Sau khi đổi thông tin kết nối, chạy **Secure API connection → Check API connection → Generate embeddings → Results**.

**Thứ tự chạy lại:** Dùng **What should I run next?** ngay dưới **Configuration helpers** để xác định bước phục hồi, hoặc theo bảng song ngữ **Sau khi sửa hoặc ngắt thao tác** ở đầu. Cell phục hồi không gửi request và không tự gửi lại POST. Khi đổi model, chạy lại **Configuration helpers**, chuẩn bị mẫu và thực hiện **Check API connection** mới trước inference. Các trường mẫu đã sửa vẫn được giữ nhưng không được dùng lại capability check của model cũ.

**Nguồn cấu hình có hiệu lực:** Demo cho biết model và số chiều vector lấy từ form hay biến môi trường ghi đè. Sau khi chuẩn bị dữ liệu mẫu, notebook chỉ hiển thị **tên biến môi trường** đang ghi đè văn bản, đường dẫn ảnh hoặc nhãn. Những thông báo này không hiển thị nội dung riêng, giá trị nhãn, đường dẫn ảnh hay credential. Muốn dùng giá trị trong form, bỏ biến môi trường tương ứng rồi chạy lại các cell liên quan.

**Ưu tiên:** biến môi trường `VISION_EMBEDDINGS_SAMPLE_TEXT`, `VISION_EMBEDDINGS_IMAGE_PATH` và các biến có hậu tố `_2` / `_3` ghi đè giá trị sửa trong cell. Muốn dùng trường hiển thị, bỏ biến môi trường trùng tên. Khi đổi model, dữ liệu mẫu vẫn được giữ; bảng phục hồi liệt kê các bước cần thiết. Notebook sẽ từ chối hiển thị kết quả cũ như thể được tạo từ input mới. Preview luôn chỉ hiện ảnh nhúng sẵn, không tự hiển thị ảnh riêng.

Ảnh mẫu nhúng được xem trước trước khi kiểm tra file tùy chỉnh. Với ảnh tùy chỉnh thiếu hoặc không đọc được, notebook báo số thứ tự ảnh và hướng dẫn song ngữ mà không lộ đường dẫn. Kiểm tra file trong runtime hiện tại, sửa đường dẫn rồi chạy lại chuẩn bị mẫu trước khi inference.

## Các biến môi trường tùy chọn

| Biến môi trường | Mặc định | Mục đích |
| --- | --- | --- |
| `VISION_EMBEDDINGS_MODEL` | `qwen3-vl-embedding-8b` | Một model: `qwen3-vl-embedding-8b` hoặc `wemm-embedding-9b` |
| `VISION_EMBEDDINGS_DIMENSION` | `1024` | Số chiều vector đầu ra |
| `VISION_EMBEDDINGS_REQUEST_TIMEOUT` | `900` | Timeout HTTP tính bằng giây |
| `VISION_EMBEDDINGS_INSTRUCTION_MODE` | `default` | Không gửi trường `instruction`; sử dụng mặc định của API. Chế độ `custom` chỉ dành cho Qwen3-VL-Embedding-8B |
| `VISION_EMBEDDINGS_CUSTOM_INSTRUCTION` | Instruction truy xuất ví dụ | Chỉ dùng ở chế độ custom (một dòng, 1–512 ký tự) |
| `VISION_EMBEDDINGS_SAMPLE_TEXT` | `a flat bright-green triangle on a cream background` | Văn bản 1 |
| `VISION_EMBEDDINGS_SAMPLE_TEXT_2` | `a glossy cobalt-blue 3D sphere on a pale background` | Văn bản 2 |
| `VISION_EMBEDDINGS_SAMPLE_TEXT_3` | `a realistic orange gift box tied with a ribbon on a wooden table` | Văn bản 3 |
| `VISION_EMBEDDINGS_IMAGE_PATH` | trống | Ảnh 1: file được phép sử dụng hoặc tam giác xanh 2D |
| `VISION_EMBEDDINGS_IMAGE_PATH_2` | trống | Ảnh 2: file được phép sử dụng hoặc cầu xanh 3D |
| `VISION_EMBEDDINGS_IMAGE_PATH_3` | trống | Ảnh 3: file được phép sử dụng hoặc hộp quà cam ảnh thật |
| `VISION_EMBEDDINGS_MAX_IMAGE_BYTES` | `5242880` | Giới hạn ảnh phía notebook (tối đa 5 MiB) |
| `VISION_EMBEDDINGS_REQUIRE_REAL` | `true` | Mặc định từ chối backend mock/local |

API công bố khả năng instruction của model qua endpoint chi tiết. Notebook bắt buộc kiểm tra SHA-256 đầu vào, phiên bản preprocessing, hash instruction và SHA-256 cấu hình. Với text, hash được kiểm tra từ byte UTF-8 sau khi strip khoảng trắng đầu/cuối, đúng với schema server; ảnh vẫn băm byte nguyên gốc. Sáu ví dụ dùng cùng cấu hình instruction; ứng dụng retrieval thực tế có thể dùng instruction khác cho truy vấn và tài liệu. Notebook từ chối custom instruction cho Tencent WeMM-Embedding-9B trước khi gửi request inference. Khoảng trắng ở đầu/cuối custom instruction của Qwen được loại bỏ trước khi kiểm tra và tính SHA-256; instruction rỗng sau chuẩn hóa bị từ chối.

Qwen3-VL-Embedding-8B hỗ trợ các giá trị chiều nguyên 64–4096. Tencent WeMM-Embedding-9B hỗ trợ 64, 128, 256, 512, 1024, 2048 và 4096. Cell cấu hình sẽ từ chối model/dimension không hợp lệ trước khi gửi request.

**Mẹo cho Kaggle/Jupyter:** Notebook hoạt động độc lập, không cần clone repository, Kaggle Input dataset hoặc tải model weights vào runtime. Chỉ sửa các cell cấu hình và dữ liệu mẫu ngắn. Một số trình xem vẫn hiển thị helper dù metadata Jupyter đánh dấu ẩn; hãy thu gọn nếu giao diện hỗ trợ, rồi chạy đúng thứ tự. Không dán token vào code cell sẽ lưu.

**Minh họa bố cục, không phải số đo:** Demo hiển thị ma trận cosine 3 × 3, ba hàng văn bản và ba cột ảnh. Mỗi ô là `cosine(text_i, image_j)`, **không** phải độ tin cậy danh tính hay kết quả benchmark. Giá trị phụ thuộc vào model và đầu vào.

## Notebook benchmark hiệu năng

Demo API gửi **đúng sáu request tuần tự cho một model được chọn** (ba text, ba ảnh); không có chế độ chuyển đổi luân phiên giữa hai model hoặc benchmark. Để thử GPU cold lặp lại (thời gian chờ tính từ cấu hình deployment có thể sửa), độ trễ warm cân bằng cho text/ảnh và warmup độc lập cho hai GPU worker theo cả hai chiều gọi, dùng [notebook benchmark riêng](../notebooks/multimodal_embeddings_benchmark.ipynb) và [giao thức benchmark](../benchmarks/README.vi.md). Benchmark chỉ chạy khi chủ động bật, mặc định không đo cold. [Mở benchmark trên Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_benchmark.ipynb).

## Kết quả thành công sẽ hiển thị gì?

Notebook hiển thị báo cáo song ngữ gồm **chế độ instruction và SHA-256 cấu hình embedding**, **model đã chọn**, **ba điểm cosine cho cặp tương ứng**, **ma trận 3 × 3 văn bản/ảnh**, **số chiều vector**, **revision checkpoint**, **chuẩn L2** và **thời gian sáu request API**. Các giá trị do deployment thật tính tại runtime; tài liệu này không đưa điểm số hoặc thời gian GPU giả lập.

## Phân biệt API và quản lý hạ tầng

Notebook này **không cài Modal CLI, quản lý Modal App/Secret/Volume, tải checkpoint hoặc deploy inference server**. Người cần tự triển khai server hãy xem [Vận hành](OPERATIONS.vi.md); notebook chỉ là client có xác thực của endpoint có sẵn.

## Quyền riêng tư, chi phí và giới hạn

- Chỉ xử lý ảnh được phép sử dụng. Notebook nhúng sẵn ba ảnh WebP 512 × 512 px do AI tạo, không cần tải từ Internet: tam giác xanh 2D, cầu xanh bóng 3D và hộp quà cam phong cách ảnh thật. Khi không nhập đường dẫn riêng, notebook gửi chính ba ảnh nhúng này; đây không phải dữ liệu benchmark tìm kiếm. Notebook không tự hiển thị ảnh riêng do người dùng cung cấp.
- API token và endpoint không được in hoặc lưu vào file. Ô nhập ẩn không làm bộ nhớ runtime trở nên riêng tư: chỉ chạy code đáng tin và restart sau khi dùng.
- Không chia sẻ raw embedding, ảnh riêng tư, URL riêng hoặc authorization header. Người xem notebook không được mặc nhiên thừa hưởng quyền truy cập API.
- Ngay cả sáu request nhỏ cũng có thể khởi động container L40S tính phí; thời gian API hiển thị theo giây có thể bao gồm cold start hoặc khởi động GPU worker độc lập, không phải độ trễ mạng toàn phần.
- `GET /v1/health` chỉ chứng minh gateway phản hồi, không chứng minh GPU sẵn sàng. Cấu hình mặc định từ chối mock khi chạy demo thực tế.
- Báo cáo giải thích điểm cosine trong khoảng xấp xỉ −1 đến +1 và chuẩn L2 gần 1. **Điểm cosine không phải phần trăm hay độ tin cậy**, cũng không chứng minh danh tính sinh trắc học hoặc chất lượng tìm kiếm.
- Kiểm thử mock chỉ xác nhận logic notebook; chạy trên Kaggle/Colab và inference GPU Modal thực tế cần nghiệm thu riêng.

Xem [Hợp đồng API](API_CONTRACT.vi.md), [Vận hành](OPERATIONS.vi.md), và [Nghiệm thu](VERIFICATION.vi.md).

**Đầu vào tùy chỉnh:** Biến VISION_EMBEDDINGS_SAMPLE_LABEL (_2, _3) cho phép đặt nhãn ba mẫu. Khi thay văn bản hoặc ảnh, báo cáo hiển thị trạng thái có sẵn/tùy chỉnh cho từng mẫu, không đưa đường dẫn hay preview ảnh riêng. Đường chéo cosine chỉ so các input cùng chỉ số, không mặc định là ghép cặp đúng nghĩa. Lỗi HTTP 401, 413, 422 và 503 có hướng dẫn song ngữ nhưng không hiển thị response body.
