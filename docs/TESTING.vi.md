# Chiến lược kiểm thử và xác minh

🌐 Ngôn ngữ / Language: [English](TESTING.md) | **Tiếng Việt**

Dự án phân biệt **kiểm thử tính đúng đắn bằng CPU** với **inference GPU thật
chỉ khi người vận hành chủ động cho phép**. Test PASS không chứng minh chất
lượng truy xuất của hai model.

## Nhóm kiểm thử CPU

| Lệnh | Phạm vi |
| --- | --- |
| `PYTHONPATH=src python -m pytest -q` | Toàn bộ bộ test CPU |
| `PYTHONPATH=src python -m pytest -q -m core` | API, vector, encoder, retrieval và routing |
| `PYTHONPATH=src python -m pytest -q -m security` | Xác thực, quota, async job và admission |
| `PYTHONPATH=src python -m pytest -q -m notebook` | Notebook demo và benchmark |
| `PYTHONPATH=src python -m pytest -q -m publication` | Release, evidence và tài liệu |
| `PYTHONPATH=src python -m pytest -p pytest_cov -q --cov=multimodal_embeddings_ai_inference --cov-report=term-missing` | Coverage mã nguồn |

Cài dependency: `python -m pip install -e '.[dev]'`.
CI kiểm thử CPU trên Python 3.11, 3.12, 3.13. Nếu file evidence được ghi
vào Git trước thời gian đo cuối cùng, CI phải thất bại. Checksum trùng nhau
không đủ chứng minh mã nguồn thực sự chạy trên GPU deployment.

## Đánh giá retrieval từ embedding thật đã lưu

Chuẩn bị riêng hai file JSON cùng workload có ground truth, chứa model_id,
revision đã pin, config_sha256, vector ảnh, vector truy vấn và nhãn
`relevant_images`. Không trộn vector của Qwen và WeMM cùng không gian.

    python scripts/evaluate_retrieval.py       --input qwen-saved-embeddings.json       --input wemm-saved-embeddings.json       --top-k 5 --require-both-models

Script tính Recall@K, Hit@1, MRR từ dữ liệu đã lưu; không gọi GPU và không
in vector gốc. Unit test dùng vector tổng hợp chỉ để kiểm tra công thức.
Evidence tìm ảnh sẵn có mới gồm hai truy vấn Qwen; chưa có tập đánh giá WeMM
tương ứng đủ để đưa ra kết luận về chất lượng model.

## Smoke test GPU thật có chủ đích

Chỉ người được phép vận hành deployment chủ động chạy lệnh có chi phí:

    python scripts/smoke_modal_gpu.py       --image /path/to/authorized-image.png       --text "an orange gift box"       --qwen-revision "$VISION_EMBEDDINGS_QWEN8_REVISION"       --wemm-revision "$VISION_EMBEDDINGS_WEMM9_REVISION"

Lệnh kiểm tra identity, checkpoint revision, số chiều, chuẩn hóa vector và
ID worker độc lập. **CI không tự chạy GPU.** Smoke PASS không chứng minh
chất lượng retrieval hoặc độ tin cậy production.

## Xác minh file Release công khai

Tải đủ năm asset của phiên bản v1.0.0 vào thư mục rỗng, sau đó chạy:

    python scripts/verify_release_assets.py       --directory /path/to/downloaded-assets       --git-ref v1.0.0 --require-supplementary

Kiểm tra này đối chiếu file tải xuống với Git tag và SHA-256 manifest.
CI PASS không thay thế việc kiểm tra asset đã công bố.
