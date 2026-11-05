# Đóng góp

🌐 Ngôn ngữ / Language: [English](CONTRIBUTING.md) | **Tiếng Việt**

Dự án hoan nghênh các đóng góp cho API inference embedding đa phương thức.

1. Mở issue mô tả lỗi hoặc đề xuất cải tiến. Không tải lên ảnh riêng tư, token hoặc model weights.
2. Làm việc trên nhánh riêng, phạm vi thay đổi rõ ràng. Bổ sung kiểm thử offline khi thay đổi hành vi.
3. Đồng bộ tài liệu tiếng Anh và tiếng Việt với mọi thay đổi hướng đến người dùng.
4. Chạy `bash scripts/setup.sh` để tạo môi trường `.venv`. Script không tự kích hoạt môi trường trong shell hiện tại. Chạy `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check src modal_app.py modal_models.py scripts tests` và `sha256sum -c SHA256SUMS.txt`.
5. Sau khi sửa file, chỉ stage những file thêm/sửa/xóa cần thiết bằng `git add <paths>` (và `git add -u` khi có file bị xóa). Chạy `.venv/bin/python scripts/update_sha256_manifest.py`, stage `SHA256SUMS.txt`, rồi kiểm tra bằng `.venv/bin/python scripts/update_sha256_manifest.py --check` và `sha256sum -c SHA256SUMS.txt`. Phải stage file mới trước khi tạo manifest vì công cụ dựa trên Git index giống CI.
6. Dùng Conventional Commit subject ngắn gọn, nội dung commit body trình bày bằng bullet.

CI không cần GPU credentials. Thay đổi inference model thật cần có báo cáo smoke test GPU đã loại thông tin nhạy cảm và nêu rõ revision model upstream. Không đưa ảnh gốc độ phân giải đầy đủ hoặc credentials vào báo cáo.

## Tái tạo môi trường kiểm tra CPU

Setup và CI cài `.[dev]` bằng `constraints/cpu.txt`. Bộ ghim bao phủ dependency runtime/test CPU, có NumPy riêng cho Python 3.11. Bộ này không ghim Modal CLI, image GPU hoặc checkpoint model; các phần đó có yêu cầu triển khai riêng. Metadata package giữ khoảng phiên bản được hỗ trợ.

```bash
python -m pip install -c constraints/cpu.txt -e '.[dev]'
```

Chủ động cập nhật phiên bản: resolve trong môi trường Python 3.11 và 3.12 mới, chạy toàn bộ test và Ruff trên mỗi môi trường, chạy quickstart mock cùng kiểm tra tài liệu/manifest rồi lưu báo cáo nghiệm thu. Cài build tool và pip riêng; constraints không phải kho wheel hoặc lockfile GPU. Xem [Nghiệm thu](docs/VERIFICATION.vi.md) để lưu bằng chứng và checklist nền tảng notebook.
