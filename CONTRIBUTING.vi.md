# Đóng góp

🌐 Ngôn ngữ / Language: [English](CONTRIBUTING.md) | **Tiếng Việt**

Dự án hoan nghênh đóng góp từ người tích hợp API, sử dụng notebook, cải thiện bản dịch đến phát triển code inference. Bạn có thể đóng góp tài liệu, ví dụ hoặc thay đổi được kiểm thử trên CPU mà không cần có GPU.

## Bạn có thể đóng góp gì?

- Báo lỗi có thể tái hiện hoặc mô tả nhu cầu sử dụng qua [mẫu issue](https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference/issues/new/choose) phù hợp.
- Cải thiện tài liệu Anh–Việt và hướng dẫn notebook.
- Bổ sung ví dụ client, tích hợp ứng dụng hoặc ý tưởng hỗ trợ model.
- Sửa code, cải thiện kiểm tra dữ liệu hoặc thêm test cho thay đổi hành vi.

Với câu hỏi sử dụng, xem [Hỗ trợ](.github/SUPPORT.md). Báo cáo lỗ hổng riêng theo [Bảo mật](SECURITY.vi.md). Không đưa token, ảnh riêng tư hoặc model weights vào issue và pull request.

## Pull request đầu tiên

1. Với thay đổi lớn, mở issue trước để trao đổi phạm vi và khả năng tương thích API.
2. Fork repository và làm việc trên một nhánh tập trung cho thay đổi đó.
3. Thực hiện thay đổi, đồng bộ tài liệu người dùng Anh–Việt và bổ sung kiểm thử offline cho hành vi bị ảnh hưởng.
4. Chạy các bước kiểm tra phù hợp bên dưới, rồi mở pull request giải thích thay đổi đối với người dùng và cách bạn kiểm chứng.

Dùng tiêu đề Conventional Commit ngắn gọn và phần mô tả nêu thay đổi cùng lý do. Việc tham gia cộng đồng tuân theo [Quy tắc ứng xử](.github/CODE_OF_CONDUCT.md).

## Môi trường phát triển

Từ repository đã clone:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

**Lệnh setup làm gì:** Tạo hoặc sử dụng lại `.venv`, cài dependency phát triển CPU theo `constraints/cpu.txt`, rồi chạy Ruff và pytest. Công cụ triển khai GPU và checkpoint model được cài riêng; xem [Vận hành](docs/OPERATIONS.vi.md).

## Chỉnh sửa code notebook

Code cell được bảo trì trong `notebooks/source/` rồi nhúng vào file `.ipynb` công bố. Sửa source Python, sau đó chạy:

```bash
python scripts/sync_notebooks.py --write
```

Với notebook benchmark, lệnh này cũng cập nhật `NOTEBOOK_TEMPLATE_SHA256` trong source Python và code cell công bố. `--check` xác minh đồng bộ và mã băm mà không sửa hai file.

Sửa phần Markdown giải thích trực tiếp trong notebook. Đồng bộ hai ngôn ngữ và giữ hình preview README nhất quán với ảnh mẫu. Notebook kết quả cần tiếp tục chạy độc lập trên Colab và Kaggle.

## Chuẩn bị và kiểm tra thay đổi

Stage đúng các file thêm, sửa và xóa trước khi cập nhật checksum manifest. Trình tạo manifest lấy đường dẫn từ Git index, vì vậy cần stage cả file mới lẫn đường dẫn đã xóa trước. Manifest không chứa README.md và README.vi.md, vì hai file này được thêm trong commit README cuối cùng và đã được Git bảo vệ tính toàn vẹn.

```bash
python scripts/update_sha256_manifest.py
git add SHA256SUMS.txt
python scripts/update_sha256_manifest.py --check
```

Chạy các bước kiểm tra trước khi gửi đóng góp:

```bash
python -m pytest -q
bash scripts/quickstart_mock.sh
ruff check src modal_app.py modal_models.py scripts tests
ruff check notebooks/source --select E4,E7,E9,F63,F7,F823 --ignore E402
python scripts/sync_notebooks.py --check
python scripts/audit_release.py
sha256sum -c SHA256SUMS.txt
```

CI chạy test CPU trên Python 3.11–3.13 và kiểm tra HTTP client với Node.js 24. Các bước này không yêu cầu GPU.

## Báo cáo thay đổi inference và notebook

Với thay đổi inference thật, đính kèm báo cáo kiểm thử đã loại thông tin nhạy cảm, nêu revision model, phiên bản dependency và các trường hợp đã chạy. Với notebook, ghi nền tảng sử dụng và những phần đã kiểm tra: đầu vào, preview, xác thực, kết quả, phục hồi sau khi ngắt. Có thể dùng [bảng kiểm tra deployment](docs/ACCEPTANCE_TEMPLATE.vi.md) để ghi nhận quan sát.

Để thu thập định danh mã nguồn và kiểm thử CPU cho báo cáo:

```bash
git rev-parse HEAD
python scripts/record_acceptance.py --run-checks --output /tmp/acceptance-session.json
```

Công cụ này thu thập thông tin source và CPU; GPU và giao diện notebook cần được chạy kiểm tra riêng. Ghi riêng revision client và server đã triển khai nếu khác nhau. Chỉ chia sẻ kết quả đã loại thông tin nhạy cảm.

## Cập nhật dependency

Phiên bản runtime và test CPU được ghim trong `constraints/cpu.txt`; dependency GPU có bộ khóa Qwen và WeMM riêng. Khi cập nhật bản ghim CPU, resolve trong môi trường Python 3.11, 3.12 và 3.13 mới, chạy toàn bộ test, Ruff trên mỗi môi trường và thử quickstart mock. Giữ các khoảng phiên bản được hỗ trợ trong `pyproject.toml`. Xem [Vận hành](docs/OPERATIONS.vi.md#tái-lập-môi-trường-dependency-gpu) để cập nhật dependency GPU.
