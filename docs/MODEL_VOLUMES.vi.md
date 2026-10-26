# Hai Model Volume riêng và vòng đời GPU

🌐 Ngôn ngữ / Language: [English](MODEL_VOLUMES.md) | **Tiếng Việt**

Deployment gán một GPU worker NVIDIA L40S và một Modal Volume lưu trữ bền vững cho từng model. Worker này không tải hoặc dừng model của worker kia.

```text
FastAPI gateway (CPU)
├── QwenL40SGpu: 1 × L40S
│   └── multimodal-embeddings-qwen-weights
│       └── checkpoints/Qwen/Qwen3-VL-Embedding-8B/
└── WeMML40SGpu: 1 × L40S
    └── multimodal-embeddings-wemm-weights
        └── checkpoints/tencent/WeMM-Embedding-9B/
```

## Workspace, Environment và quyền sở hữu checkpoint

Hai Volume được `modal_models.py` tạo trong **Modal Workspace và
Environment đang hoạt động**. Khi deploy, `modal_app.py` tìm lại đúng
hai tên đó với `create_if_missing=False`. Nếu chọn khác Environment hoặc
đặt sai `VISION_EMBEDDINGS_QWEN_VOLUME_NAME` /
`VISION_EMBEDDINGS_WEMM_VOLUME_NAME`, ứng dụng báo thiếu tài nguyên
thay vì tự dùng lại checkpoint. Kiểm tra `modal profile list`,
`modal environment list` và `MODAL_ENVIRONMENT` trước cả tải model
lẫn deploy; xem [Vận hành](OPERATIONS.vi.md#chọn-workspace-và-environment-nhất-quán).

Khi hàm tải model in `READY <repository> @ <sha> -> <volume>`, snapshot
đã tải xong và Volume đã được commit. Dùng chính xác các SHA đó khi tạo
hai Secret bằng [scripts/create_modal_secret.py](OPERATIONS.vi.md#cấu-hình-xác-thực).
Volume chỉ là **ổ đĩa lưu trữ bền vững**, không giữ sẵn GPU hay chứng minh
model đã nạp và inference thành công; xem các bước riêng trong
[Kiểm tra deployment](VERIFICATION.vi.md).

Với cài đặt mới, chạy `modal run modal_models.py --selection both` để tải [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) vào các Volume riêng. Nếu đã có checkpoint, nên sao chép và kiểm tra thay vì tải lại.

Đặt tên Volume bằng `VISION_EMBEDDINGS_QWEN_VOLUME_NAME` và `VISION_EMBEDDINGS_WEMM_VOLUME_NAME`. Mỗi GPU class mặc định có `min_containers=0`, `max_containers=1` và `scaledown_window=180`. Hai class có thể chạy đồng thời, sử dụng tối đa hai GPU L40S khi cả hai đang hoạt động.

Volume giữ model weights trên đĩa bền vững; **không chiếm sẵn VRAM của GPU**. Cold start vẫn xảy ra sau khi scale-to-zero, nhưng gọi luân phiên hai model khi worker còn warm không gây chuyển model. Phải lưu metadata model và revision checkpoint riêng cho từng worker.

Cố định revision bằng `VISION_EMBEDDINGS_QWEN8_REVISION` và `VISION_EMBEDDINGS_WEMM9_REVISION`; không trộn vector giữa các model hoặc revision khác nhau vào một chỉ mục.

Trước khi dừng deployment cũ, xác minh nội dung của cả hai Volume và kiểm thử hai GPU worker độc lập qua HTTPS API có xác thực.
