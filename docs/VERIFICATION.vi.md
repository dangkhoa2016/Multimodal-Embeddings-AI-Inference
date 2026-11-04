# Nghiệm thu inference — 04/11/2026

Tài liệu tổng hợp bằng chứng vận hành CompactFaces AI Inference trong thử nghiệm có kiểm soát. **Đây không phải kết quả đo độ chính xác nhận diện danh tính khuôn mặt.**

## Môi trường và dữ liệu tái lập

- Phần cứng: một NVIDIA L40S, Volume chung cho hai checkpoint và cơ chế chuyển model tuần tự.
- Dữ liệu: một ảnh tổng hợp CompactFaces có quyền sử dụng, chuyển thành PNG RGB 400 × 400. SHA256 ảnh: `f7f65de628690f58ff40770f0a1f689aacb7b77618ebc149f35c6db8e63283a7`. Không đưa ảnh vào source repository.
- Qwen: `Qwen/Qwen3-VL-Embedding-8B`, revision `2c4565515e0f265c6511776e7193b22c0968ddc7`.
- Tencent: `tencent/WeMM-Embedding-9B`, revision `00c52839de57a6d4fd5b78cf5522ccf0ac8ea482`.
- Mã adapter upstream Qwen: `393e2978d27852b0d0230d6994f37f9c15bed73c`.

## Kết quả đã quan sát

| Kiểm thử | Kết quả |
| --- | --- |
| Gọi trực tiếp GPU Qwen → WeMM → Qwen | PASS; vector 1024 chiều hữu hạn, chuẩn hóa L2 |
| HTTPS `POST /v1/embeddings` có Bearer cho cả hai model | PASS; `backend=modal`, đúng revision và SHA256 input |
| Hợp đồng REST có/không có token | 9/9 PASS; HTTP 200, 401 và 404 như kỳ vọng |
| Offline tests và lint | PASS; một cảnh báo Starlette không chặn |

Hai mẫu `elapsed_ms` do API báo là **531,685 ms** (Qwen) và **11.160,632 ms** (WeMM). Chúng chịu tác động của việc load/chuyển model, **không** phải thời gian tổng của kết nối mạng, p95, throughput hoặc cam kết SLA.

## Phạm vi kết luận

Hệ thống chạy model thật thành công, nhưng chưa có đánh giá độ chính xác đối sánh khuôn mặt, negative pairs, recall, retrieval trên toàn dataset, concurrency dài hạn, VRAM peak hay chi phí mỗi request. Cần tách riêng embedding/index của từng model và revision. Nhãn identity tổng hợp không đủ để chứng minh tính nhất quán sinh trắc. Xem [Operations](OPERATIONS.vi.md) và [Security](../SECURITY.md).
