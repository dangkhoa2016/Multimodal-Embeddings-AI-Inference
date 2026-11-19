# Kết quả benchmark API — Tổng quan các phiên đo

🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Đây là trang tổng quan các kết quả benchmark API đã công bố. **Full Benchmark mới nhất của phiên bản `v1.0.0` được đo ngày 19/11/2026 theo giờ Việt Nam.** Các bảng chi tiết phía dưới vẫn là số liệu lịch sử ngày 08/11/2026, không phải số liệu của lần đo mới nhất.

## Các phiên benchmark đã công bố

| Ngày đo | Phạm vi | Báo cáo và dữ liệu |
| --- | --- | --- |
| **19/11/2026 (UTC+7; 18/11 UTC)** — mới nhất | Full Benchmark 5 giai đoạn; **65 HTTP POST thành công / 68 embedding items** | [Báo cáo tiếng Việt](full-2026-11-19.vi.md) · [English](full-2026-11-19.md) · [JSON](full-2026-11-19.json) · [Provenance](full-2026-11-19.provenance.json) |
| **17/11/2026 (UTC)** — lịch sử | Full Benchmark độc lập; **65 HTTP POST / 68 embedding items** | [Báo cáo tiếng Việt](full-2026-11-17.vi.md) · [English](full-2026-11-17.md) · [JSON](full-2026-11-17.json) · [Provenance](full-2026-11-17.provenance.json) |
| **08/11/2026 (UTC)** — lịch sử | 35 POST serving / 38 embedding items; kiểm tra quá tải 10 POST riêng; ví dụ tìm ảnh bằng văn bản riêng | [Serving](serving.json) · [Overload](overload.json) · [Image search](image-search.json) · [Chi tiết](#phép-đo-lịch-sử-08112026-utc) |

**Provenance ngày 17/11:** URL Modal trực tiếp của người vận hành được che có chủ đích trong provenance JSON công bố; bản gốc chưa chỉnh sửa được lưu riêng. Số liệu đo và kết quả tổng hợp không thay đổi.

**Diễn giải:** Ngày đo không nhất thiết là ngày đưa dữ liệu vào Git. Các phiên đo độc lập, không cộng dồn lượt POST. Kết quả chỉ mô tả tải thử nghiệm cụ thể; không phải đánh giá retrieval accuracy hoặc cam kết SLA. Dữ liệu ngày 19/11 chưa chứng thực độc lập commit nguồn phía client/server.

## Phép đo lịch sử — 08/11/2026 (UTC)

Báo cáo trình bày kết quả đo hiệu năng API và thử nghiệm tìm ảnh bằng văn bản với `Qwen/Qwen3-VL-Embedding-8B` và `tencent/WeMM-Embedding-9B`. Notebook sử dụng client Linux CPU (Python 3.13.18) gọi API embeddings qua HTTPS.

**Cấu hình thử nghiệm:** Hai GPU worker NVIDIA L40S trên Modal (mỗi model một worker), CPU gateway hỗ trợ 16 đầu vào HTTP đồng thời, giới hạn bốn yêu cầu inference và thời hạn xử lý 60 giây.

**Cách hiểu kết quả:** Độ trễ client bao gồm thời gian truyền tải và điều phối; thời gian mã hóa GPU do worker báo cáo. Kết quả mô tả các bài thử cụ thể, không phải cam kết hiệu năng production hoặc đánh giá chất lượng model.

**Dữ liệu và kiểm tra:** Các file JSON lưu số liệu và metadata liên quan. Chạy `python scripts/audit_benchmark_evidence.py` để xem lại dữ liệu offline, không sử dụng GPU hoặc gửi yêu cầu mạng.

**Phân biệt ngày đo và ngày đưa vào Git:** Các phép đo serving, overload và tìm ảnh được thực hiện vào **08/11/2026 theo UTC** (sáng 09/11/2026 theo giờ Việt Nam, UTC+7). Ngày đưa file vào Git là ngày công bố dữ liệu, không phải thời điểm chạy phép đo. Riêng `image-search.json` được thêm vào Git ngày **17/11/2026 (UTC+7)** để công bố kết quả đã thu thập trước đó.

## Độ trễ warm

[Báo cáo phục vụ](serving.json) gồm **35 POST request / 38 embedding item**: 8 warm-up, 16 request warm được đo, 8 request đồng thời và 3 batch hai model. Embedding có 1.024 chiều. Đầu vào là văn bản công khai và ảnh PNG gradient RGB 128 × 128 được tạo xác định trong notebook.

| Model | Đầu vào | Số mẫu | Client p50 (ms) | Client p95 (ms) | GPU encode p50 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-Embedding-8B | Văn bản | 4 | 2680.75 | 2697.13 | 25.77 |
| Qwen3-VL-Embedding-8B | Ảnh | 4 | 3535.12 | 3887.78 | 40.34 |
| WeMM-Embedding-9B | Văn bản | 4 | 2674.98 | 2720.28 | 128.90 |
| WeMM-Embedding-9B | Ảnh | 4 | 3459.50 | 3483.03 | 148.19 |

Thời gian client gồm truyền tải, xử lý tại gateway, điều phối từ xa và inference. Thời gian GPU encode là thành phần do worker báo cáo. Mỗi p95 chỉ mô tả bốn quan sát trong lần chạy này.

## Request đồng thời và batch hai model

Mỗi model nhận một đợt riêng gồm bốn request với hai worker client:

| Model | Hoàn tất / đã thử | Worker client | Thời gian đợt (ms) | Client p50 (ms) | Client p95 (ms) | Item/giây quan sát |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-Embedding-8B | 4/4 | 2 | 5636.87 | 2703.76 | 2948.13 | 0.71 |
| WeMM-Embedding-9B | 4/4 | 2 | 5566.91 | 2696.45 | 2832.68 | 0.72 |

Ba batch hai model chứa một item cho mỗi model trong từng batch:

| Hoàn tất / đã thử | Item hợp lệ | Client p50 (ms) | Client p95 (ms) | Trung bình (ms) | Item/giây quan sát |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3/3 | 6 | 2902.64 | 2918.23 | 2903.40 | 0.69 |

Thời gian mỗi đợt gồm điều phối và ghi nhật ký phía client. Độ trễ mixed batch tính cho cả HTTP batch; item/giây bằng số item hợp lệ chia tổng thời gian các batch. Các phép đo mô tả những lần chạy ngắn với mức đồng thời và đầu vào đã chọn. Xem [hướng dẫn benchmark](../README.vi.md) để biết quy tắc tính toán và phân loại.

## Kiểm tra quá tải với mười request

[Báo cáo quá tải](overload.json) riêng ghi nhận mười POST văn bản đồng thời, đầu ra 64 chiều và không tự động gửi lại POST:

| HTTP status | Số request | Khoảng thời gian client (giây) |
| --- | ---: | ---: |
| 200 | 4 | 19.391–28.215 |
| 429 | 6 | 1.377–1.791 |

Health trả HTTP 200 trong **1,108 giây** khi đợt request đang chạy. Call graph của nền tảng cho thấy bốn lời gọi GPU thành công và không có lời gọi GPU cho sáu request bị từ chối. Các request được nhận có hai lần nạp model ban đầu. Phép kiểm tra này xác nhận giới hạn nhận request và khả năng phản hồi của health; test hồi quy tự động kiểm tra riêng việc hết deadline và hủy lời gọi từ xa.

## Tìm ảnh bằng văn bản

[Báo cáo tìm ảnh](image-search.json) dùng ba ảnh WebP đi kèm, Qwen3-VL-Embedding-8B và 1.024 chiều. Ba POST ảnh tạo một chỉ mục tái sử dụng; hai POST văn bản truy vấn chỉ mục đó:

| Truy vấn | Ảnh đứng đầu | Cosine |
| --- | --- | ---: |
| an orange gift box with a ribbon | gift.webp | 0.7600 |
| a blue sphere | sphere.webp | 0.6648 |

Đây là ví dụ truy xuất nhỏ được lưu từ lần chạy trước. Hãy đánh giá chất lượng xếp hạng trên dữ liệu đại diện, dùng cùng revision model và cấu hình mã hóa. [API Demo Notebook](../../notebooks/multimodal_embeddings_api_demo.ipynb) có ba ảnh mẫu này; [hướng dẫn benchmark](../README.vi.md) giải thích cách đo deployment của bạn.

## Dữ liệu nguồn và giới hạn truy xuất

Các bảng serving, overload và tìm ảnh phía trên là **dữ liệu ngày 08/11/2026 (UTC)**: [serving.json](serving.json), [overload.json](overload.json), [image-search.json](image-search.json). Snapshot mã nguồn deployment lịch sử chưa được xác minh độc lập.

Các Full Benchmark ngày [17/11/2026](full-2026-11-17.vi.md) và [19/11/2026](full-2026-11-19.vi.md) được lưu thành những phiên riêng biệt. Không trộn kết quả hoặc provenance giữa các phiên. [JSON ngày 19/11](full-2026-11-19.json) không chứa chứng thực độc lập commit nguồn phía client/server. Các số liệu không bảo đảm SLA production hay chất lượng embedding.
