# Vận hành Modal và nghiệm thu bản phát hành

Tài liệu vận hành **CompactFaces AI Inference** trên một máy mới. Không chứa khóa bí mật.

## Kiến trúc

- Một FastAPI Gateway qua HTTPS, xác thực Bearer token tại ứng dụng.
- Một Modal GPU worker L40S, `max_containers=1`, `min_containers=0`.
- Hai checkpoint cố định revision trong cùng Volume `compactfaces-ai-weights`.
- Chỉ một tiến trình model chiếm VRAM tại một thời điểm. Khi chuyển model phải nạp lại checkpoint.

## Thiết lập

```bash
bash scripts/setup.sh
.venv/bin/python -m pip install -e '.[modal]'
modal setup
```

Nếu Python thiếu `ensurepip`, script tự dùng `uv` khi có sẵn. Không cần GPU trên máy cài đặt.

## Tải checkpoint vào Volume

```bash
modal run modal_models.py --selection both
modal volume ls compactfaces-ai-weights /checkpoints
```

Cố định revision, không tự ý dùng `main` khi tái hiện benchmark:

- Qwen3-VL-Embedding-8B: `2c4565515e0f265c6511776e7193b22c0968ddc7`
- Tencent WeMM-Embedding-9B: `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`

## Tạo Secret

```bash
python scripts/create_modal_secret.py \
  --qwen8-revision 2c4565515e0f265c6511776e7193b22c0968ddc7 \
  --wemm9-revision 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482
```

Token được lưu cục bộ tại `~/.config/compactfaces-ai-inference/modal-secret.env` với quyền chỉ chủ sở hữu đọc/ghi. Không in token ra terminal, không commit lên Git, không chia sẻ file này. Khi cần, thực hiện luân chuyển token trước khi đưa ra môi trường có người dùng khác.

## Triển khai

```bash
modal deploy modal_app.py
```

Lấy HTTPS URL thật từ lệnh `modal deploy modal_app.py` và cấu hình riêng trong môi trường triển khai.
Không commit URL triển khai, định danh tài khoản hoặc thông tin cấu hình riêng vào repository.

URL thay đổi theo workspace của Modal. Rails phải cho cấu hình URL/token, không hardcode.

## Nghiệm thu

```bash
.venv/bin/ruff check src modal_app.py modal_models.py scripts tests
.venv/bin/python -m pytest -q
python scripts/smoke_modal_gpu.py --image /path/to/authorized_image.png --switch-back
```

Smoke test gọi trực tiếp GPU class đã triển khai, dùng ảnh tổng hợp và xác minh embedding 1.024 chiều, các phần tử hữu hạn, chuẩn L2 gần 1. Đối với kiểm thử qua REST API, xem `scripts/smoke_both.py` và README.

Chỉ công bố một model đã nghiệm thu khi chính model đó PASS inference thật trên GPU. Mock test hoặc deploy thành công không tương đương nghiệm thu.

## Logs và chi phí

```bash
modal app logs compactfaces-ai-inference --since 30m --tail 100
modal app list
modal volume ls compactfaces-ai-weights /checkpoints
```

Worker có khả năng scale về 0; chuyển model sẽ mất thời gian nạp lại. Cần tính tổng chi phí GPU, CPU, RAM, Volume và mạng. Traceback được ghi vào log riêng của Modal, còn API công khai chỉ trả lỗi tổng quát.

## Nguyên tắc công bố

Chỉ commit source, ví dụ cấu hình, tài liệu và tests. Không commit model weights, ảnh dataset, `.env`, token, cache hoặc ZIP. Trước khi phát hành, cần xác minh CI và inference thật của cả hai model.
