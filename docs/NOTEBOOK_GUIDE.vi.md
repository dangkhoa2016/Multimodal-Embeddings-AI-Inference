# Sử dụng notebook trên Google Colab và Kaggle

🌐 Ngôn ngữ / Language: [English](NOTEBOOK_GUIDE.md) | **Tiếng Việt**

Khám phá embedding ảnh và văn bản với [notebook API song ngữ Anh–Việt](../notebooks/multimodal_embeddings_api_demo.ipynb). Chọn [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) hoặc [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B), chạy các ví dụ có sẵn rồi thay bằng dữ liệu của bạn.

## Bạn cần chuẩn bị

- Runtime Python 3 **CPU** trên Colab hoặc Kaggle có Internet. Notebook đã có code client và ảnh mẫu; bạn có thể mở trực tiếp mà không cần clone repository hoặc cài model weights.
- Để inference, cần **URL HTTPS gốc và Bearer token ứng dụng** của API đã triển khai. Bạn có thể dùng [deployment Modal của mình](OPERATIONS.vi.md) hoặc dịch vụ bạn được cấp quyền truy cập.
- Để xem trước ảnh mẫu, chỉ cần chạy các bước chuẩn bị dữ liệu bên dưới. Thông tin xác thực chỉ cần khi kết nối API.

## Mở notebook

[Mở trên Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_api_demo.ipynb) hoặc [tải notebook](../notebooks/multimodal_embeddings_api_demo.ipynb). Trên Kaggle, import file đó vào một Python notebook và bật Internet.

**Kaggle Save & Run All không tương tác:** Nếu bật `RUN_API_DEMO=True`, hãy chuẩn bị URL/token qua secret hoặc biến môi trường an toàn trước khi chạy. Nếu thiếu credential, notebook dùng prompt ẩn và cần phiên chạy tương tác; không ghi token vào source notebook.

**Run All an toàn:** `RUN_API_DEMO=False` luôn chỉ preview, không chờ nhập, gửi HTTP/GPU hoặc tạo điểm cosine giả, **kể cả khi đã có credential môi trường**. Bật `RUN_API_DEMO=True` để gọi API và GPU thực tế có thể phát sinh phí; nhập credential qua prompt ẩn hoặc dùng biến môi trường. Không cần công tắc phê duyệt khác.

## Chạy demo

1. **Chọn model.** Trong **Model configuration**, chọn `MODEL_ID`, `DIMENSION` và `TIMEOUT_SECONDS`. Chạy **Model configuration → Configuration helpers**. Mặc định là Qwen3-VL-Embedding-8B với 1.024 chiều. Colab có form cấu hình; trên Kaggle, sửa giá trị trong code cell.
2. **Chuẩn bị ví dụ.** Chạy **Customize samples → Prepare input samples**. Mặc định gồm ba câu mô tả ghép với tam giác xanh, hình cầu xanh và hộp quà cam. Bước này hiển thị ảnh mẫu mà không gọi API.
3. **Kết nối API.** Trong **API connection**, chạy **Secure API connection → Check API connection**. Nhập URL HTTPS gốc không có `/v1` và API token qua prompt ẩn.
4. **Tạo embedding và so sánh.** Với `RUN_API_DEMO=True`, chạy **Generate embeddings → Results** cho ba văn bản và ba ảnh. Modal tự dùng async job có thể tiếp tục, không cần bật phê duyệt khác. Có thể phát sinh tối đa sáu GPU POST mới có tính phí khi chưa có cache/job dùng lại; kết quả hoặc job đã lưu có thể giảm về không.

Bạn có thể dừng sau bước chuẩn bị dữ liệu để xem preview, hoặc tiếp tục inference khi đã có thông tin kết nối. Request API có thể khởi động GPU worker tính phí trong tài khoản triển khai.

## Dùng dữ liệu của bạn

Sửa `SAMPLE_TEXT_1` đến `SAMPLE_TEXT_3` trong **Customize samples**. Để dùng ảnh riêng, đặt `IMAGE_PATH_1` đến `IMAGE_PATH_3` thành đường dẫn file PNG, JPEG hoặc WebP có trong runtime; để trống nếu muốn giữ ảnh nhúng sẵn.

Ví dụ, tải file lên Colab rồi dùng `/content/my-image.png`, hoặc attach dataset trên Kaggle và dùng `/kaggle/input/my-dataset/my-image.png`. Đây là đường dẫn ví dụ; hãy chọn file bạn được phép gửi tới API.

Gallery ảnh mẫu chỉ để tham khảo, **không chứng minh đó là ảnh đã gửi inference**. Sau bước chuẩn bị file riêng, notebook cảnh báo song ngữ và liệt kê vị trí ảnh riêng; phần Results cũng cảnh báo lại. Để bảo vệ quyền riêng tư, nội dung và đường dẫn ảnh riêng không tự hiển thị. Nếu file thiếu hoặc không đọc được, notebook báo mẫu bị ảnh hưởng.

Để xem trước ảnh riêng đã chọn, chủ động bật `PREVIEW_CUSTOM_IMAGES=True` trong
**Customize samples**, rồi chạy lại **Prepare input samples**. Ảnh sẽ xuất hiện
trong output, đường dẫn vẫn được ẩn. Xóa output trước khi chia sẻ notebook.

## Đọc kết quả

Ma trận 3 × 3 có **hàng là ảnh, cột là văn bản**. Ô `(i, j)` là `cosine(image_i, text_j)`. Đường chéo so sánh các mẫu cùng chỉ số; khi thay đầu vào, bạn quyết định các cặp đó có mô tả cùng chủ thể hay không.

Điểm cosine biểu thị độ tương đồng vector, không phải phần trăm tin cậy. Báo cáo còn có model, revision checkpoint, số chiều vector, chuẩn L2, chế độ instruction và thời gian xử lý request. Thời gian API khác với toàn bộ thời gian đi qua mạng. Dùng notebook hiệu năng bên dưới để thực hiện các phép đo latency có cấu trúc.

## Thay đổi dữ liệu hoặc tiếp tục sau khi ngắt

Cell **What should I run next?** ở cuối notebook chạy sau Results và tóm tắt trạng thái mới nhất. Ở chế độ preview, cell xác nhận lượt chạy không gửi HTTP/GPU; cảnh báo khôi phục job chỉ xuất hiện khi cache riêng có công việc chưa hoàn tất. Chạy lại riêng cell hướng dẫn không gửi request. Bảng chạy lại ở đầu notebook hướng dẫn khi đổi dữ liệu mẫu, model hoặc thông tin kết nối.

Nếu request inference bị timeout hoặc ngắt, kiểm tra deployment trước khi bắt đầu lượt mới: GPU có thể vẫn đang xử lý. Notebook không tự động gửi lại POST. Với lỗi kết nối, xem [Khắc phục lỗi](TROUBLESHOOTING.vi.md).

## Tiếp tục từng embedding mà không tạo GPU job trùng

Demo Notebook dùng async job có thể tiếp tục trên Modal. `RUN_API_DEMO=True`
là lựa chọn chủ động duy nhất để gọi API/GPU thực tế. Vector đã xác minh
và job ID được lưu sau mỗi kết quả; **Run All** lần sau dùng lại cache hoặc
GET job cũ trước, không tự POST trùng.

File `multimodal_embeddings_api_demo_cache.json` lưu checkpoint giữa các
lần chạy cell. File chứa vector, request ID, job ID; **không lưu** ảnh gốc,
địa chỉ API hay Bearer token. Vector vẫn có thể nhạy cảm: không công khai
file này kèm output notebook. Có thể dùng `VISION_EMBEDDINGS_NOTEBOOK_CACHE`
để chọn đường dẫn riêng, hoặc khôi phục file từ backup khi runtime restart.
Nếu file mất, chạy lại **cùng đầu vào trên cùng deployment** sẽ tạo lại
request ID idempotent, kể cả khi đã đổi Bearer token. Server chỉ trả lại kết quả cũ **khi bản ghi job backend vẫn còn**. Theo tài liệu, Modal Dict có thể hết hạn sau khoảng 7 ngày không hoạt động; nếu cả checkpoint và bản ghi backend đều mất, POST cùng request ID vẫn có thể kích hoạt GPU mới và phát sinh phí. Kiểm tra job cũ trước khi chủ động bắt đầu công việc mới. URL deployment, revision
model, dimension, instruction và digest dữ liệu vẫn thuộc khóa này.
Notebook có thể đọc checkpoint kiểu cũ gắn với token của phiên trước
nếu thông tin xác thực đó không đổi khi nâng cấp. Không tự tạo ID mới
để retry timeout.

**Mất ACK của POST async:** Với checkpoint `UNCERTAIN`, notebook ưu tiên
**GET chỉ đọc** tới `/v1/embeddings/jobs/by-request/{request_id}`. API
tự tính job key cho client đã xác thực, kể cả sau khi xoay API key riêng.
Nếu server cũ chưa có endpoint (HTTP 404), notebook dự phòng bằng khóa
legacy của owner và vẫn **chỉ GET**. Không gửi GPU POST lần hai từ bản ghi chưa rõ kết quả. Nếu record không tồn tại/hết hạn thì phải đối soát,
không tự retry. Xem [hướng dẫn đối soát](OPERATIONS.vi.md#đối-soát-thủ-công-các-job-async-quá-hạn).

**Khôi phục cache kiểu cũ sau khi xoay token (chỉ đọc):** Giữ file cache riêng của phiên trước. Nếu job dùng ID cũ phụ thuộc credential, đặt `VISION_EMBEDDINGS_LEGACY_REQUEST_ID_MAP` thành JSON object ánh xạ request ID ổn định *hiện tại* `demo-...` tới khóa `demo-...` *cũ* trong cache. Lấy hai ID từ notebook đã cấu hình và cache riêng của bạn; không điền token hay dữ liệu riêng vào mapping. Cache cũ phải có kết quả hoàn tất hoặc job ID dài 64 ký tự. Notebook chỉ tiếp tục bằng **GET**, không gửi công việc mới; nếu record không tìm thấy hoặc không rõ job ID thì dừng an toàn, không POST mới. Vector khôi phục vẫn được đối chiếu digest đầu vào, model, revision và dimension. Nếu backend đã hết hạn và cache không còn vector hoàn tất thì mapping không thể phục hồi kết quả. Không đoán ID hoặc tìm cách khôi phục bằng token cũ.

Job `QUEUED` hoặc `RUNNING` được polling tối đa `MAX_POLL_SECONDS`
(mặc định 900), có thể chạy lại cell để tiếp tục. Job `FAILED` không tự retry. `INDETERMINATE` báo dispatch hoặc
heartbeat CPU coordinator quá hạn, **không chứng minh GPU đã dừng**;
cần kiểm tra log Modal và không tự gửi request ID mới. Phần Results
phân biệt vector từ cache (không đo thời gian client mới) với job
mới/tiếp tục; thời gian client gồm gửi POST, mạng và chờ polling,
không phải GPU-only latency. Notebook dành cho backend Modal có async;
nếu deployment cũ chưa hỗ trợ, hãy cập nhật backend trước khi chạy notebook
thay vì bỏ cơ chế chống gửi trùng.

**Xác minh sau khi triển khai lại:** Vector cache cũ vẫn có thể hợp lệ khi URL và revision không đổi, nhưng **không chứng minh GPU của deployment mới đã chạy**. Để xác minh GPU trên deployment mới, chủ động thay một văn bản hoặc ảnh mẫu được phép sử dụng, xác nhận kết quả tương ứng không phải cache và đối chiếu lượt gọi GPU trên Modal. Không xóa journal của job chưa rõ kết quả/đang chạy để ép gửi POST mới.

## Cấu hình model

Qwen3-VL-Embedding-8B chấp nhận số chiều nguyên từ 64–4096. Tencent WeMM-Embedding-9B chấp nhận 64, 128, 256, 512, 1024, 2048 hoặc 4096.

Dùng instruction mặc định cho dữ liệu demo, hoặc chọn custom instruction cho Qwen3-VL-Embedding-8B để mô tả tác vụ retrieval của bạn. WeMM dùng chế độ mặc định. Xem [tài liệu API](API_CONTRACT.vi.md#instruction-qwen-tùy-chọn-và-metadata-cấu-hình-vector) để biết chi tiết về instruction và metadata embedding.

## Biến môi trường tùy chọn

Bạn có thể cung cấp thông tin kết nối qua `VISION_EMBEDDINGS_BASE_URL` và `VISION_EMBEDDINGS_API_TOKEN` thay cho prompt. Biến môi trường đã đặt được ưu tiên hơn các trường trong form. Bỏ giá trị ghi đè nếu muốn dùng cấu hình nhập trong form.

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

Các biến tùy chọn `VISION_EMBEDDINGS_SAMPLE_LABEL`, `VISION_EMBEDDINGS_SAMPLE_LABEL_2` và `VISION_EMBEDDINGS_SAMPLE_LABEL_3` đặt tên hiển thị cho ví dụ. Notebook cho biết cấu hình nào dùng biến môi trường, đồng thời giữ credential và đường dẫn riêng ngoài các thông báo trạng thái.

## Đo hiệu năng phục vụ API

Dùng [notebook hiệu năng](../notebooks/multimodal_embeddings_benchmark.ipynb) để so sánh latency warm, cold-start tùy chọn, định tuyến luân phiên giữa model và các đợt request đồng thời có giới hạn. Notebook chạy trên CPU, gọi API của bạn và xuất báo cáo JSON. Đọc [hướng dẫn benchmark](../benchmarks/README.vi.md) trước khi bật phép đo; mặc định `RUN_BENCHMARK=False` không gửi request API.

## Chia sẻ và nhận hỗ trợ

Giữ runtime riêng tư, nhập credential qua prompt ẩn, xóa output và khởi động lại runtime trước khi chia sẻ. Sử dụng dữ liệu bạn được phép xử lý.

Với câu hỏi hoặc lỗi có thể tái hiện, xem [Hỗ trợ](../.github/SUPPORT.md). Dự án hoan nghênh ví dụ, bản dịch và cải tiến notebook qua [Hướng dẫn đóng góp](../CONTRIBUTING.vi.md).
