# Benchmark hiệu năng inference

🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Thư mục này mô tả [notebook benchmark hiệu năng](../notebooks/multimodal_embeddings_benchmark.ipynb) của Multimodal Embeddings API có xác thực. Notebook Python 3 chạy **CPU** gọi deployment HTTPS đã có, phục vụ [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) trên hai GPU NVIDIA L40S độc lập. Không cần GPU cục bộ, CUDA, tải checkpoint hay Modal CLI.

**Các model được đo:** Qwen3-VL-Embedding-8B (dòng Qwen3, khoảng 8 tỷ tham số) và Tencent WeMM-Embedding-9B (Tencent, khoảng 9 tỷ tham số) chuyển văn bản và hình ảnh thành vector số. Notebook này đo thời gian phục vụ API, không đánh giá chất lượng vector hay độ chính xác nhận diện khuôn mặt.

Xem [kết quả benchmark GPU](results/README.vi.md) về warm latency, concurrency giới hạn, batch hai model và kiểm tra quá tải với mười request, kèm báo cáo JSON gốc. Đây là kết quả đo deployment tại thời điểm đó.

## Chạy trên Kaggle hoặc Google Colab

1. Dùng notebook Python 3 CPU có Internet: [mở trên Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_benchmark.ipynb) hoặc import [notebook benchmark](../notebooks/multimodal_embeddings_benchmark.ipynb) vào Kaggle.
2. **Nhập các giá trị thật của deployment**: gateway/GPU scaledown và `min_containers` trong notebook. Các giá trị điền sẵn **gateway 60 giây / GPU 180 giây / min-containers 0** chỉ là ví dụ có thể chỉnh từ một deployment, **không phải mặc định chung của Modal**. Tránh request cạnh tranh trong lúc chờ cold. Timing API thông thường không tự xác minh được autoscaler.
3. Mặc định an toàn là `RUN_BENCHMARK=False`, `BENCHMARK_MODE="warm"`, `WARM_REPETITIONS=4` và `WARMUP_REQUESTS=1`: chỉ bật `RUN_BENCHMARK` sẽ gửi tối đa 24 POST request warm-up/đo lường (cả text và ảnh), không chờ cold. Muốn đo cold, chọn `BENCHMARK_MODE="cold"` rồi bật `RUN_BENCHMARK=True`. Dropdown còn có `alternating`, `concurrent`, `mixed_batch` và `full`, cho phép chạy cả năm trong một phiên. Cung cấp `VISION_EMBEDDINGS_BASE_URL` (HTTPS base URL không có `/v1`) và `VISION_EMBEDDINGS_API_TOKEN` (Bearer token ứng dụng), hoặc đặt đủ hai tên trong Kaggle Secrets. Nếu thiếu một hay cả hai giá trị, notebook tự mở ô nhập ẩn cho từng giá trị còn thiếu khi `RUN_BENCHMARK=True`. Mặc định **không** gửi request.
4. Chạy toàn bộ cell và xem thống kê notebook cùng file `multimodal_embeddings_performance_summary.json`. Báo cáo không lưu credential, vector thô, URL hoặc nội dung input.

**Chọn chế độ benchmark:** Form chỉ có một dropdown `BENCHMARK_MODE`. Chọn `full` sẽ chạy tuần tự cả năm phase trong một phiên và xuất một JSON report; các lựa chọn khác chỉ chạy một phase. Các cờ kỹ thuật được suy ra tự động.

| Chế độ | Nội dung đo | Giới hạn POST theo cấu hình ví dụ |
| --- | --- | ---: |
| `warm` (mặc định) | Độ trễ text/ảnh sau warm-up | 24 (8 warm-up + 16 lượt đo) |
| `cold` | Khởi động bằng job bất đồng bộ, theo dõi bằng GET | 10 (5/model; không tính GET) |
| `alternating` | Độ trễ điều phối khi luân phiên Qwen/WeMM | 12 (2 lượt chuẩn bị + 10 lượt đo) |
| `concurrent` | Chuẩn bị GPU async rồi đo sync đồng thời | 10 (2 chuẩn bị + 8 đo) |
| `mixed_batch` | Chuẩn bị GPU async trước mỗi batch hai model | 9 POST / 12 embedding items |
| `full` | Cold → warm → alternating → concurrent → mixed_batch | **65 POST / 68 embedding items** |

Các số trên là **giới hạn request theo cấu hình ví dụ**, không phải ước tính chi phí/tốc độ. Full còn có **35 phút chờ cold idle tính toán** theo scale-down ví dụ, *chưa kể* thời gian xử lý API/GPU; có thể phát sinh phí. Kiểm tra hai giá trị `min_containers=0` và scale-down thực tế. Full dừng khi một phase lỗi và giữ báo cáo dở; không chạy lại hoặc tự retry POST có thể đã gửi. Chế độ được ghi vào JSON report. Muốn đổi chế độ, bắt đầu lượt đo và preflight mới; không chạy lại phase đã hoàn tất.

Cell Cấu hình hiển thị trước số HTTP POST tối đa dự kiến theo từng phase,
tách riêng lượt warm-up và lượt đo, số embedding items trong mixed batch cùng
thời gian chờ cold tính từ cấu hình. Phần xem trước song ngữ này không gửi
request; lượt gửi thực tế có thể ít hơn nếu gặp lỗi. Không ước tính thời gian
xử lý GPU hay chi phí tiền.

**Giao thức cold bất đồng bộ:** Mỗi lượt cold lưu `async_request_id` ngẫu nhiên trước khi gửi **một** `POST /v1/embeddings/async`, sau đó dùng `GET /v1/embeddings/jobs/{job_id}` để theo dõi đến `SUCCEEDED`, `FAILED`, `INDETERMINATE` hoặc hết hạn. Có thể chỉnh `COLD_JOB_TIMEOUT_SECONDS` (mặc định 1800) và `COLD_POLL_INTERVAL_SECONDS` (mặc định 15). Với cấu hình mặc định, mỗi lượt có tối đa 120 GET, **ngoài** 65 POST theo kế hoạch full. Khi mất xác nhận POST, có thể tra cứu an toàn qua GET `/v1/embeddings/jobs/by-request/{request_id}`; không tự gửi lại POST. Job registry phải hoạt động.

**Warm-up / readiness bất đồng bộ:** Các lượt warm-up text/ảnh, chuẩn bị alternating, trước concurrent và trước từng mixed batch đều gửi mỗi lượt một `POST /v1/embeddings/async` được ghi journal trước dispatch, sau đó dùng GET job polling có giới hạn. Dùng cùng `COLD_JOB_TIMEOUT_SECONDS` / `COLD_POLL_INTERVAL_SECONDS` để tránh thêm cấu hình phức tạp. Chỉ các request đo (warm/alternating/concurrent/mixed batch) dùng API sync. Mỗi phép đo phải có `model_state=warm`, cờ first-request false và worker ID khớp với lần chuẩn bị; nếu không hoặc bằng chứng quá cũ theo scale-down đã khai báo, notebook dừng fail-closed, không tự gửi lại POST hay gọi keep-alive vô hạn. Gateway `/health` không chứng minh GPU warm. Các cấu hình scale-down do người chạy khai báo, không phải bằng chứng autoscaler thực tế. Mặc định full: 65 POST, trong đó 10 cold, 8 warm-up, 16 warm measured, 2 alternate setup, 10 alternate measured, 2 concurrent setup, 8 concurrent measured, 6 mixed setup và 3 batch POST. GET poll không nằm trong POST cap.

**Diễn giải timing:** Cold `client_elapsed_ms` gồm gửi job, hàng đợi/cấp phát, nạp model, GPU, GET polling và truyền mạng. Báo cáo ghi `async_ack_elapsed_ms`, `async_after_ack_ms`; thời gian nạp model dựa trên GPU telemetry đã xác minh. **API job không cung cấp riêng thời gian chờ cấp GPU** (`gpu_allocation_wait_ms=null`). Các nhãn `http_504_unconfirmed_cause`, `async_indeterminate`, `async_job_wait_timeout_unconfirmed` cần Modal logs để xác định nguyên nhân thực. Full sẽ dừng an toàn khi cold lỗi, không thử lại GPU POST.

**Cách kết nối:** Notebook ưu tiên biến môi trường, sau đó Kaggle Secrets, cuối cùng tự mở prompt nhập ẩn cho **từng giá trị còn thiếu**. Khi `RUN_BENCHMARK=False`, không đọc secret hoặc hỏi nhập. Với Kaggle Save & Run All không tương tác, cần cấp quyền cho cả hai secret từ trước vì runtime có thể không hỗ trợ prompt. Nếu nhập credential thất bại, notebook không gửi POST benchmark và không ghi đè báo cáo cũ. Không ghi Bearer token trực tiếp vào source hoặc output notebook.

**Thứ tự chạy lần đầu:** Đặt `RUN_BENCHMARK=True` trong Cấu hình **trước** cell API client/preflight, rồi chạy **Cấu hình benchmark → API client/preflight → Hàm hỗ trợ request và kiểm tra → các phase đã bật → Báo cáo** theo thứ tự. Mặc định lưu `False` không gửi POST. Nếu đã chạy client khi cờ còn tắt, phải chạy lại Cấu hình và client/preflight sau khi bật cờ.

**Nhật ký lượt đo tuần tự:** Mỗi lượt cold, warm-up, warm hoặc luân phiên được lưu là `in_progress` trước khi gửi và cập nhật tại chỗ sau phản hồi. Khi bắt được Stop, lượt thử vẫn được giữ kể cả lúc ghi nhận phản hồi; dừng cưỡng bức có thể để lại marker đang chờ. Marker đang chờ không chứng minh server đã tiếp nhận và không có POST nào được gửi lại.

**Nhật ký lượt thử mixed batch:** Mỗi lượt mixed-model batch được ghi vào JSON đã lược bỏ dữ liệu nhạy cảm với trạng thái `in_progress` trước khi POST. KeyboardInterrupt được bắt sẽ đổi thành `interrupted` và giữ nguyên các lượt hoàn tất trước đó; nếu giao diện dừng cưỡng bức, lượt cuối có thể vẫn ở `in_progress` mà không chứng minh server đã tiếp nhận. Hai trạng thái này không phải phép đo được xác nhận và POST không được tự động gửi lại.

**Nhật ký lượt thử concurrent:** Mỗi request được lên lịch được ghi với trạng thái `in_progress` trước khi gửi. Khi bắt được KeyboardInterrupt, notebook lập tức lưu báo cáo `INCOMPLETE_INTERRUPTED`, hủy công việc chưa bắt đầu và giữ các phản hồi đã xác minh từ request đang chạy mà không lên lịch request thay thế. Nếu giao diện dừng cưỡng bức, lượt thử có thể vẫn ở `in_progress`; trạng thái này không chứng minh server đã tiếp nhận hoặc hoàn tất. Thời gian toàn nhóm bao gồm cả chi phí điều phối và ghi nhật ký phía notebook.

**An toàn khi chạy lại:** Mỗi lần chạy API client/preflight tạo mã lượt đo mới và SHA-256 của cấu hình đo cùng mã băm đầu vào. Chạy lại bất kỳ phase đã bắt đầu, kể cả phase bị ngắt, hoặc đổi cấu hình/đầu vào mà chưa chạy lại preflight sẽ bị chặn trước POST tiếp theo. KeyboardInterrupt bắt được sẽ lưu INCOMPLETE_INTERRUPTED; trường hợp giao diện buộc dừng có thể còn IN_PROGRESS cho đến khi thử chạy lại. Request đã gửi có thể vẫn chạy trên server. Muốn bắt đầu mới, hãy chạy theo thứ tự: Cấu hình benchmark → API client/preflight → Hàm hỗ trợ request và kiểm tra → các phase đã bật → Báo cáo. Báo cáo cuối kiểm tra số lượt đo dự kiến; không cộng dồn dữ liệu cũ hoặc trùng lặp. Mã lượt đo và mã băm không chứng minh được commit mã nguồn trên server.

**Provenance:** Có thể khai báo BENCHMARK_CLIENT_SOURCE_SHA và BENCHMARK_SERVER_SOURCE_SHA (commit Git 40 ký tự) khi biết chính xác; các giá trị được gắn nhãn do operator cung cấp, không xác minh qua API. SHA256 template notebook nhúng sẵn được test tại repository nhưng không tự xác thực lại sau khi chỉnh notebook. JSON ghi cấu hình phép đo đã lọc và nguồn mã trong cả báo cáo thành công và thất bại. Không lưu token, URL cá nhân, ảnh hoặc instruction gốc.

### Thu thập bằng chứng để tái hiện phép đo

Với lượt đo mới, hãy ghi **Git commit phía client** của checkout dùng để xuất notebook, **commit phía server** do người vận hành đối soát, revision checkpoint và giấy phép model, checksum của dependency lockfile, phiên bản Python/CUDA, cấu hình Modal worker liên quan, thời điểm UTC và SHA-256 bộ dữ liệu đầu vào. Đối chiếu hash file của deployment với snapshot mã nguồn lưu trữ. Commit do người chạy tự nhập chỉ là thông tin khai báo, **không phải** xác thực độc lập từ server; cần phân biệt rõ trong báo cáo. Các JSON kết quả tháng 11/2026 được giữ nguyên và không thể bổ sung hồi tố một commit mã nguồn đã được xác minh.

## Giao thức cân bằng

| Trường hợp | Qwen3-VL-Embedding-8B | Tencent WeMM-Embedding-9B | Điều kiện chấp nhận |
| --- | ---: | ---: | --- |
| GPU cold | Số lượt và thời gian chờ tính từ cấu hình | Cùng số lượt và thời gian chờ | `gpu_worker_first_request=true` và `model_state=initial_load` |
| Warm-up | Số request cho mỗi modality × 2 modality × 2 khối | Cùng số lượt | Không tính |
| Warm text | Số lượt bằng nhau, chỉnh được | Cùng số lượt | `model_state=warm` |
| Warm image | Số lượt bằng nhau, chỉnh được | Cùng số lượt | `model_state=warm` |
| Gọi luân phiên | Số lượt chỉnh được | Cùng số lượt | `model_state=warm` |

Hai model nhận cùng văn bản, byte PNG, số chiều vector (1.024), một input/request. Cold đảo thứ tự model theo vòng và **không probe hai function** trong khoảng idle được tính từ cấu hình. Warm chạy hai khối đảo chiều (Qwen3-VL-Embedding-8B→Tencent WeMM-Embedding-9B, Tencent WeMM-Embedding-9B→Qwen3-VL-Embedding-8B), warm-up cả text và ảnh trước phần đo modality ngẫu nhiên. Mặc định mỗi model có hai text và hai image warm-up, rồi tám request đo. JSON đếm đủ hai khối. Thử gọi luân phiên đo routing trên hai worker độc lập. POST đã đo không retry; lượt lỗi hoặc không xác nhận không được tính nhầm thành warm retry.

**Quan trọng:** Thời gian chờ tính từ cấu hình có thể tạo điều kiện Modal scale down, nhưng sleep không chứng minh điều đó. Notebook tính `COLD_IDLE_WAIT_SECONDS = max(GATEWAY_SCALEDOWN_SECONDS, GPU_SCALEDOWN_SECONDS) + COLD_WINDOW_PADDING_SECONDS`. Ví dụ gateway 50 giây, GPU 120 giây, đệm 30 giây thì chờ 150 giây/lượt. GPU worker trả cờ request đầu; **không thể xác nhận riêng** trạng thái cold của API gateway từ API hiện tại. Client timing gồm mạng HTTP, xử lý gateway, điều phối GPU, nạp model và inference. Không tách riêng thành phần khởi động gateway/cấp phát GPU. Notebook hiển thị **ước tính tổng thời gian chờ idle** từ số lần thử, số model và thời gian chờ/lượt; thời gian inference cộng thêm và GPU có thể phát sinh chi phí.

## Phép đo concurrent và mixed-model batch tùy chọn

Các phép đo cold, warm và gọi luân phiên vẫn **tuần tự**. Cold không phải chế độ mặc định; chỉ chọn `BENCHMARK_MODE="cold"` sau khi xác minh thời gian idle và chi phí của deployment. Hai kịch bản khác có trong dropdown: chọn `BENCHMARK_MODE="concurrent"` để thử nhóm HTTP request đồng thời, hoặc `BENCHMARK_MODE="mixed_batch"` để gửi mỗi model một item qua `/v1/embeddings/batches`. Chỉ chạy kịch bản đã chọn khi `RUN_BENCHMARK=True`. Tối đa bốn client worker và 16 request mỗi model; nhiều nhất mười lần gửi mixed batch. Mỗi response được kiểm tra `embedding_metadata`, `instruction_sha256` và `config_sha256`. `QWEN_INSTRUCTION_MODE="default"` không gửi instruction; chỉ Qwen3-VL-Embedding-8B hỗ trợ `custom`. Tencent WeMM-Embedding-9B không nhận instruction.

Gọi luân phiên warm cần `gpu_worker_id` của hai model khác nhau và ổn định. Concurrent cần GPU timing hợp lệ và worker ID khớp với lượt chuẩn bị; nếu bằng chứng không xác nhận warm, cả phase sẽ dừng. Chỉ request hoàn tất sau khi xác minh timing mới được tính vào số item/giây của **nhóm request ngắn**; wall-clock latency phía client được ghi riêng; batch ghi thời gian một HTTP call và số item thực tế. Đây không phải chứng cứ về thông lượng ổn định hay giới hạn GPU autoscaling.

Dù chỉ chạy concurrent hoặc mixed-model batch, báo cáo vẫn ghi revision và cấu hình embedding đã xác minh cho từng model; nếu cấu hình thay đổi giữa các response, lượt đo được đánh dấu lỗi.

## Kết quả và giới hạn

JSON ghi toàn bộ lượt thử, timestamp UTC, thời gian idle thực đo, trạng thái GPU, mã worker ngẫu nhiên tùy chọn (cần API phiên bản hỗ trợ) và trạng thái xác nhận/chưa xác nhận, revision model, checksum mẫu và p50/p95/mean mô tả chỉ cho mẫu xác nhận. Các phép đo cold/warm/luân phiên đánh giá **latency endpoint tuần tự**. Nhóm concurrent và mixed batch tùy chọn chỉ đo API với tải giới hạn, **không chứng minh throughput duy trì**. Các phép đo không đánh giá chất lượng ngữ nghĩa, face verification, accuracy retrieval, xếp hạng cross-modal, GPU utilization, SLA deployment hay khả năng sẵn sàng production. Không trộn embedding của Qwen3-VL-Embedding-8B và Tencent WeMM-Embedding-9B vào cùng không gian vector. Dùng dataset benchmark riêng phù hợp để đánh giá retrieval ở ứng dụng downstream. Nếu request HTTP, kiểm tra metadata hoặc định tuyến lỗi, notebook ghi đè báo cáo cũ bằng báo cáo JSON từng phần `INCOMPLETE_FAILED` đã loại dữ liệu nhạy cảm rồi dừng, không lên lịch các pha tiếp theo và không tính thống kê tổng hợp cuối cùng. Khi concurrent thất bại, chỉ các request đã được gửi (tối đa số worker đang chạy) có thể tiếp tục hoàn thành và được ghi vào báo cáo; không gửi các request còn lại. Không retry bất kỳ POST nào. Trước lần gọi API đầu tiên, notebook ghi `IN_PROGRESS` thay cho báo cáo cũ.

**Lưu ý số mẫu:** Với cấu hình warm mặc định, 24 POST gồm 8 warm-up và 16 lần đo, tối đa 4 mẫu cho mỗi model/loại đầu vào. Bảng hiển thị cột `Latency n` và báo cáo JSON chứa `sample_size_context`. p95 trên ít mẫu chỉ có tính mô tả, không đại diện độ trễ đuôi production.

Chạy notebook với deployment được phép truy cập để tạo báo cáo độ trễ đo được, số lượng mẫu và thông tin truy vết cấu hình của lượt chạy.

## Đánh giá retrieval văn bản–hình ảnh offline

Sử dụng [công cụ đánh giá offline](../scripts/evaluate_retrieval.py) để tính
**hit@1, recall@K và mean reciprocal rank (MRR)** từ embedding đã lưu:

```bash
python scripts/evaluate_retrieval.py --input saved-embeddings.json --top-k 3
```

JSON khai báo `model_id`, `revision` được ghim và `config_sha256` ở cấp
dataset **và trong từng bản ghi ảnh/truy vấn**. Ảnh có `id` và `vector`;
truy vấn có `id`, `vector` và nhãn `relevant_images`.
Công cụ từ chối trộn model/revision/cấu hình, vector không hợp lệ,
sai chiều và nhãn không tồn tại. Không gửi request API, không chạy GPU.
Chỉ số chỉ đại diện cho dataset có nhãn, không chứng minh độ chính xác
ngữ nghĩa tổng quát của model.
