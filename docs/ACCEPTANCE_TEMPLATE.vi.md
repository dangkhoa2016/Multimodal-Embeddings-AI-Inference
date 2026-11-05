# Mẫu báo cáo nghiệm thu

🌐 Ngôn ngữ / Language: [English](ACCEPTANCE_TEMPLATE.md) | **Tiếng Việt**

Sao chép mẫu ra ngoài repository cho từng phiên GPU hoặc notebook thực tế. Mọi trường hợp bắt đầu ở **NOT_RUN**; chỉ đổi thành PASS/FAIL sau khi quan sát. Tách bằng chứng CPU, GPU và giao diện. Quy trình chính nằm trong [Nghiệm thu](VERIFICATION.vi.md).

## Định danh phiên chạy

| Trường | Giá trị quan sát |
| --- | --- |
| Bắt đầu/kết thúc UTC | Chưa ghi nhận |
| Commit/tree client và checkout sạch | Chưa ghi nhận |
| Source commit của server đã deploy | Chưa biết; không thay bằng SHA client |
| Bằng chứng source server | Khai báo của người vận hành hoặc bản ghi deployment; nêu rõ loại |
| URL CI run cụ thể | Chưa ghi nhận |
| Môi trường | Colab/Kaggle/CPU client; phiên bản Python; phần cứng GPU nếu có |
| Revision checkpoint Qwen | Chưa ghi nhận |
| Revision checkpoint WeMM | Chưa ghi nhận |
| Phase được bật và số request | Chưa ghi nhận |

Tạo JSON đi kèm bằng `scripts/record_acceptance.py` để ghi source và phiên bản package CPU; công cụ không thử GPU hoặc giao diện. Ghi lại đúng source SHA đã kiểm thử; không thay thế bằng SHA của phiên bản mã nguồn khác.

## Các trường hợp quan sát

| Trường hợp | Trạng thái | Quan sát đã lược dữ liệu nhạy cảm |
| --- | --- | --- |
| GPU: text/ảnh có xác thực cho Qwen | NOT_RUN | Chưa có bằng chứng |
| GPU: text/ảnh có xác thực cho WeMM | NOT_RUN | Chưa có bằng chứng |
| GPU: đúng revision, vector hữu hạn/chuẩn hóa, số chiều | NOT_RUN | Chưa có bằng chứng |
| GPU: worker độc lập và định tuyến luân phiên | NOT_RUN | Chưa có bằng chứng |
| UI: dropdown model/trường có thể sửa | NOT_RUN | Chưa có bằng chứng |
| UI: preview ảnh nhúng trước thông tin xác thực | NOT_RUN | Chưa có bằng chứng |
| UI: prompt kết nối ẩn | NOT_RUN | Chưa có bằng chứng |
| UI: sáu request và ma trận/báo cáo song ngữ | NOT_RUN | Chưa có bằng chứng |
| UI: nền sáng/tối và khả năng truy cập bảng | NOT_RUN | Chưa có bằng chứng |
| UI: đổi input/model, hướng dẫn phục hồi cục bộ | NOT_RUN | Chưa có bằng chứng |
| UI: thiếu ảnh tùy chỉnh và phục hồi | NOT_RUN | Chưa có bằng chứng |
| UI: benchmark tắt không gọi API | NOT_RUN | Chưa có bằng chứng |
| UI: Stop trong phase warm, chặn chạy lại | NOT_RUN | Chưa có bằng chứng |

Thử Colab và Kaggle trong các phiên riêng. Lượt chỉ đo warm không xác nhận cold-start; kiểm tra accessibility còn thiếu phải giữ NOT_RUN. Không tự động gửi lại POST đã dispatch.

## Bằng chứng và khoảng trống còn lại

Đính kèm quan sát hoặc ảnh chụp đã lược dữ liệu nhạy cảm theo từng trường hợp, kèm runtime và định danh source. Không đưa token, URL dịch vụ riêng, dữ liệu đầu vào, mã băm input riêng hoặc vector thô. Nêu rõ trường hợp lỗi hoặc chưa thử. Mẫu này không phải bằng chứng chạy thành công và không chứng minh độ chính xác retrieval hoặc năng lực production.
