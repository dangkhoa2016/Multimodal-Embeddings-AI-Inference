# Kết quả đo GPU lưu trữ — 2026-11-07

🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Các kết quả này được ghi từ deployment Modal thật vào **2026-11-07 UTC**, dùng Qwen/Qwen3-VL-Embedding-8B và tencent/WeMM-Embedding-9B. Các code cell của notebook hiệu năng chạy trên client CPU Linux với Python 3.13.18, gọi API HTTPS từ xa.

Deployment dùng một worker L40S cho mỗi model, một gateway CPU nhận tối đa 16 input HTTP đồng thời, giới hạn bốn request inference và deadline inference 60 giây. Revision checkpoint và digest nội dung source được lưu trong các file JSON. Phần cứng GPU và danh tính source là cấu hình do người vận hành cung cấp; timing trong phản hồi không tự xác thực được chúng.

Lần chạy này không ghi lại mã commit Git. JSON phục vụ và quá tải có lưu digest SHA-256 của các file mã nguồn runtime để đối chiếu với phiên bản được đo. Các báo cáo không chứa API credential, URL deployment hoặc vector embedding thô.

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
