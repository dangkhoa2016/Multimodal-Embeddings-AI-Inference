# Giấy phép và thông báo bên thứ ba

🌐 Ngôn ngữ / Language: [English](LICENSE-NOTES.md) | **Tiếng Việt**

Mã nguồn và tài liệu gốc của dự án được cấp phép **MIT**. Xem [LICENSE](LICENSE).

Các checkpoint model và mã inference upstream thuộc bên thứ ba và **không được phân phối** trong repository này:

- [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) — model card chính thức ghi **Apache-2.0**; cần kiểm tra điều khoản hiện hành tại repository model.
- [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) — Tencent phát hành model parameters và mã nguồn theo **Apache-2.0**; thành phần bên thứ ba vẫn tuân theo giấy phép riêng.
- [Mã nguồn Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) — được tải riêng lúc build. Dependencies có thể có điều khoản bổ sung.

Với mỗi deployment, cần kiểm tra model repository, model card, dependencies và các thông báo áp dụng. Repository có ba ảnh WebP tổng hợp nhúng trong Demo Notebook, nhưng không chứa pretrained weights, bộ sưu tập ảnh hay dataset bên thứ ba. Giấy phép MIT cho source **không chuyển đổi giấy phép** của checkpoint hoặc mã upstream.

API tạo embedding ảnh và văn bản cho thử nghiệm tìm kiếm tổng quát. Đây **không phải** dịch vụ xác thực sinh trắc học đã được chứng nhận.
