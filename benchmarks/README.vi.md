# Benchmark hiệu năng inference

🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Thư mục này mô tả [notebook benchmark hiệu năng](../notebooks/multimodal_embeddings_benchmark.ipynb) của Multimodal Embeddings API có xác thực. Notebook Python 3 chạy **CPU** gọi deployment HTTPS đã có, phục vụ [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) trên hai GPU NVIDIA L40S độc lập. Không cần GPU cục bộ, CUDA, tải checkpoint hay Modal CLI.

**Các model được đo:** Qwen3-VL-Embedding-8B (dòng Qwen3, khoảng 8 tỷ tham số) và Tencent WeMM-Embedding-9B (Tencent, khoảng 9 tỷ tham số) chuyển văn bản và hình ảnh thành vector số. Notebook này đo thời gian phục vụ API, không đánh giá chất lượng vector hay độ chính xác nhận diện khuôn mặt.

## Chạy trên Kaggle hoặc Google Colab

1. Dùng notebook Python 3 CPU có Internet: [mở trên Google Colab](https://colab.research.google.com/github/dangkhoa2016/Multimodal-Embeddings-AI-Inference/blob/main/notebooks/multimodal_embeddings_benchmark.ipynb) hoặc import [notebook benchmark](../notebooks/multimodal_embeddings_benchmark.ipynb) vào Kaggle.
2. **Nhập các giá trị thật của deployment**: gateway/GPU scaledown và `min_containers` trong notebook. Các giá trị điền sẵn **gateway 60 giây / GPU 180 giây / min-containers 0** chỉ là ví dụ có thể chỉnh từ một deployment, **không phải mặc định chung của Modal**. Tránh request cạnh tranh trong lúc chờ cold. Timing API thông thường không tự xác minh được autoscaler.
3. Mặc định an toàn là `RUN_BENCHMARK=False`, `RUN_COLD=False`, `RUN_WARM=True`, `WARM_REPETITIONS=4` và `WARMUP_REQUESTS=1`: chỉ bật `RUN_BENCHMARK` sẽ gửi tổng cộng 24 POST request warm-up/đo lường (cả text và ảnh) và không chờ cold. Muốn đo riêng cold, phải chủ động đặt `RUN_COLD=True`, `RUN_WARM=False` và `RUN_ALTERNATE=False`, rồi `RUN_BENCHMARK = True`. Cung cấp `VISION_EMBEDDINGS_BASE_URL` (HTTPS base URL không có `/v1`) và `VISION_EMBEDDINGS_API_TOKEN` (Bearer token ứng dụng), hoặc nhập qua prompt ẩn. Mặc định **không** gửi request.
4. Chạy toàn bộ cell và xem thống kê notebook cùng file `multimodal_embeddings_performance_summary.json`. Báo cáo không lưu credential, vector thô, URL hoặc nội dung input.

**Thứ tự chạy lần đầu:** Đặt `RUN_BENCHMARK=True` trong Cấu hình **trước** cell API client/preflight, rồi chạy các phase đã bật và Báo cáo. Mặc định lưu `False` không gửi POST. Nếu đã chạy client khi cờ còn tắt, phải chạy lại Cấu hình và client/preflight sau khi bật cờ.

**Nhật ký lượt thử mixed batch:** Mỗi lượt mixed-model batch được ghi vào JSON đã lược bỏ dữ liệu nhạy cảm với trạng thái `in_progress` trước khi POST. KeyboardInterrupt được bắt sẽ đổi thành `interrupted` và giữ nguyên các lượt hoàn tất trước đó; nếu giao diện dừng cưỡng bức, lượt cuối có thể vẫn ở `in_progress` mà không chứng minh server đã tiếp nhận. Hai trạng thái này không phải phép đo được xác nhận và POST không được tự động gửi lại.

**Nhật ký lượt thử concurrent:** Mỗi request được lên lịch được ghi với trạng thái `in_progress` trước khi gửi. Khi bắt được KeyboardInterrupt, notebook lập tức lưu báo cáo `INCOMPLETE_INTERRUPTED`, hủy công việc chưa bắt đầu và giữ các phản hồi đã xác minh từ request đang chạy mà không lên lịch request thay thế. Nếu giao diện dừng cưỡng bức, lượt thử có thể vẫn ở `in_progress`; trạng thái này không chứng minh server đã tiếp nhận hoặc hoàn tất. Thời gian toàn nhóm bao gồm cả chi phí điều phối và ghi nhật ký phía notebook.

**An toàn khi chạy lại:** Mỗi lần chạy API client/preflight tạo mã lượt đo mới và SHA-256 của cấu hình đo cùng mã băm đầu vào. Chạy lại bất kỳ phase đã bắt đầu, kể cả phase bị ngắt, hoặc đổi cấu hình/đầu vào mà chưa chạy lại preflight sẽ bị chặn trước POST tiếp theo. KeyboardInterrupt bắt được sẽ lưu INCOMPLETE_INTERRUPTED; trường hợp giao diện buộc dừng có thể còn IN_PROGRESS cho đến khi thử chạy lại. Request đã gửi có thể vẫn chạy trên server. Muốn bắt đầu mới, hãy chạy theo thứ tự: Cấu hình → API client/preflight → các phase đã bật → Báo cáo. Báo cáo cuối kiểm tra số lượt đo dự kiến; không cộng dồn dữ liệu cũ hoặc trùng lặp. Mã lượt đo và mã băm không chứng minh được commit mã nguồn trên server.

**Provenance:** Có thể khai báo BENCHMARK_CLIENT_SOURCE_SHA và BENCHMARK_SERVER_SOURCE_SHA (commit Git 40 ký tự) khi biết chính xác; các giá trị được gắn nhãn do operator cung cấp, không xác minh qua API. SHA256 template notebook nhúng sẵn được test tại repository nhưng không tự xác thực lại sau khi chỉnh notebook. JSON ghi cấu hình phép đo đã lọc và nguồn mã trong cả báo cáo thành công và thất bại. Không lưu token, URL cá nhân, ảnh hoặc instruction gốc.

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

Các phép đo cold, warm và gọi luân phiên vẫn **tuần tự**. Mặc định không đo cold; chỉ bật sau khi xác minh thời gian idle và chi phí của deployment. Hai phép đo mới **tắt mặc định**: bật `RUN_CONCURRENT=True` để thử nhóm HTTP request đồng thời, hoặc `RUN_MIXED_BATCH=True` để gửi mỗi model một item qua `/v1/embeddings/batches`. Chỉ có hiệu lực khi `RUN_BENCHMARK=True`. Tối đa bốn client worker và 16 request mỗi model; nhiều nhất mười lần gửi mixed batch. Mỗi response được kiểm tra `embedding_metadata`, `instruction_sha256` và `config_sha256`. `QWEN_INSTRUCTION_MODE="default"` không gửi instruction; chỉ Qwen3-VL-Embedding-8B hỗ trợ `custom`. Tencent WeMM-Embedding-9B không nhận instruction.

Gọi luân phiên warm cần `gpu_worker_id` của hai model khác nhau và ổn định. Concurrent yêu cầu GPU timing hợp lệ (trạng thái model, cờ request đầu, batch size và các trường thời gian không âm), nhưng worker ID vẫn tùy chọn. Chỉ request hoàn tất sau khi xác minh timing mới được tính vào số item/giây của **nhóm request ngắn**; wall-clock latency phía client được ghi riêng; batch ghi thời gian một HTTP call và số item thực tế. Đây không phải chứng cứ về thông lượng ổn định hay giới hạn GPU autoscaling.

Dù chỉ chạy concurrent hoặc mixed-model batch, báo cáo vẫn ghi revision và cấu hình embedding đã xác minh cho từng model; nếu cấu hình thay đổi giữa các response, lượt đo được đánh dấu lỗi.

## Kết quả và giới hạn

JSON ghi toàn bộ lượt thử, timestamp UTC, thời gian idle thực đo, trạng thái GPU, mã worker ngẫu nhiên tùy chọn (cần API phiên bản hỗ trợ) và trạng thái xác nhận/chưa xác nhận, revision model, checksum mẫu và p50/p95/mean mô tả chỉ cho mẫu xác nhận. Các phép đo cold/warm/luân phiên đánh giá **latency endpoint tuần tự**. Nhóm concurrent và mixed batch tùy chọn chỉ đo API với tải giới hạn, **không chứng minh throughput duy trì**. Các phép đo không đánh giá chất lượng ngữ nghĩa, face verification, accuracy retrieval, xếp hạng cross-modal, GPU utilization, SLA deployment hay khả năng sẵn sàng production. Không trộn embedding của Qwen3-VL-Embedding-8B và Tencent WeMM-Embedding-9B vào cùng không gian vector. Dùng dataset benchmark riêng phù hợp để đánh giá retrieval ở ứng dụng downstream. Nếu request HTTP, kiểm tra metadata hoặc định tuyến lỗi, notebook ghi đè báo cáo cũ bằng báo cáo JSON từng phần `INCOMPLETE_FAILED` đã loại dữ liệu nhạy cảm rồi dừng, không lên lịch các pha tiếp theo và không tính thống kê tổng hợp cuối cùng. Khi concurrent thất bại, chỉ các request đã được gửi (tối đa số worker đang chạy) có thể tiếp tục hoàn thành và được ghi vào báo cáo; không gửi các request còn lại. Không retry bất kỳ POST nào. Trước lần gọi API đầu tiên, notebook ghi `IN_PROGRESS` thay cho báo cáo cũ.

**Lưu ý số mẫu:** Với cấu hình warm mặc định, 24 POST gồm 8 warm-up và 16 lần đo, tối đa 4 mẫu cho mỗi model/loại đầu vào. Bảng hiển thị cột `Latency n` và báo cáo JSON chứa `sample_size_context`. p95 trên ít mẫu chỉ có tính mô tả, không đại diện độ trễ đuôi production.

Chưa có số liệu hiệu năng thực tế được công bố từ notebook này.
