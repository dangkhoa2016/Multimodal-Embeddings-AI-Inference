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

Với cài đặt mới, chạy `modal run modal_models.py --selection both` để tải [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) và [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) vào các Volume riêng. Nếu đã có checkpoint, nên sao chép và kiểm tra thay vì tải lại.

Đặt tên Volume bằng `VISION_EMBEDDINGS_QWEN_VOLUME_NAME` và `VISION_EMBEDDINGS_WEMM_VOLUME_NAME`. Mỗi GPU class mặc định có `min_containers=0`, `max_containers=1` và `scaledown_window=180`. Hai class có thể chạy đồng thời, sử dụng tối đa hai GPU L40S khi cả hai đang hoạt động.

Volume giữ model weights trên đĩa bền vững; **không chiếm sẵn VRAM của GPU**. Cold start vẫn xảy ra sau khi scale-to-zero, nhưng gọi luân phiên hai model khi worker còn warm không gây chuyển model. Phải lưu metadata model và revision checkpoint riêng cho từng worker.

Cố định revision bằng `VISION_EMBEDDINGS_QWEN8_REVISION` và `VISION_EMBEDDINGS_WEMM9_REVISION`; không trộn vector giữa các model hoặc revision khác nhau vào một chỉ mục.

Trước khi dừng deployment cũ, xác minh nội dung của cả hai Volume và kiểm thử hai GPU worker độc lập qua HTTPS API có xác thực.
