# Bảng kiểm chứng deployment

🌐 Ngôn ngữ / Language: [English](ACCEPTANCE_TEMPLATE.md) | **Tiếng Việt**

Biểu mẫu này dùng để ghi kết quả của từng deployment do người vận hành kiểm tra, không phải bảng trạng thái phát hành của repository. Sao chép ra ngoài mã nguồn và điền PASS/FAIL/N/A cùng quan sát cho phiên GPU, Colab và Kaggle. Xem [Hướng dẫn xác minh](VERIFICATION.vi.md) để biết quy trình.

## Định danh phiên chạy

| Trường | Giá trị quan sát |
| --- | --- |
| Bắt đầu/kết thúc UTC | — |
| Commit/tree client và checkout sạch | — |
| Source commit của server đã deploy | Ghi SHA server riêng, không suy ra từ client |
| Bằng chứng source server | Khai báo của người vận hành hoặc bản ghi deployment; nêu rõ loại |
| URL CI run cụ thể | — |
| Môi trường | Colab/Kaggle/CPU client; phiên bản Python; phần cứng GPU nếu có |
| Revision checkpoint Qwen | — |
| Revision checkpoint WeMM | — |
| Phase được bật và số request | — |

Dùng `scripts/record_acceptance.py` để xuất định danh mã nguồn, môi trường Python và kết quả kiểm tra CPU tùy chọn; ghi quan sát GPU/giao diện theo từng deployment tại đây. Luôn ghi đúng revision mã nguồn được dùng cho mỗi lượt chạy.

## Các trường hợp quan sát

| Trường hợp | Trạng thái | Quan sát đã lược dữ liệu nhạy cảm |
| --- | --- | --- |
| GPU: text/ảnh có xác thực cho Qwen | — | — |
| GPU: text/ảnh có xác thực cho WeMM | — | — |
| GPU: đúng revision, vector hữu hạn/chuẩn hóa, số chiều | — | — |
| GPU: worker độc lập và định tuyến luân phiên | — | — |
| Hosting: Workspace budget và spend limit của Modal | — | Ghi cấu hình giới hạn chi phí; không lưu thông tin thanh toán/token |
| Hosting: quota theo client tại ingress và kiểm soát token | — | Ghi chính sách quota và kiểm soát token của deployment |
| UI: dropdown model/trường có thể sửa | — | — |
| UI: preview ảnh nhúng trước thông tin xác thực | — | — |
| UI: prompt kết nối ẩn | — | — |
| UI: sáu request và ma trận/báo cáo song ngữ | — | — |
| UI: nền sáng/tối và khả năng truy cập bảng | — | — |
| UI: đổi input/model, hướng dẫn phục hồi cục bộ | — | — |
| UI: thiếu ảnh tùy chỉnh và phục hồi | — | — |
| UI: benchmark tắt không gọi API | — | — |
| UI: Stop trong phase warm, chặn chạy lại | — | — |

Ghi kết quả Colab và Kaggle theo từng phiên. Lượt chỉ đo warm không đánh giá cold-start; ghi rõ phạm vi quan sát. Không tự gửi lại POST đã dispatch.

## Bằng chứng và ghi chú

Đính kèm quan sát hoặc ảnh chụp đã lược dữ liệu nhạy cảm theo từng trường hợp, kèm runtime và định danh source. Không đưa token, URL dịch vụ riêng, dữ liệu đầu vào, mã băm input riêng hoặc vector thô. Ghi rõ phạm vi, kết quả và lỗi quan sát được. Bảng này mô tả một phiên deployment cụ thể, không phải chứng nhận độ chính xác retrieval hoặc năng lực production.
