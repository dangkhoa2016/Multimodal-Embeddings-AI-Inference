# Công bố mã nguồn trên GitHub

Chỉ công bố source, tests, tài liệu và metadata model phục vụ tái lập kết quả. Không đưa weights, ảnh dataset, token hoặc URL triển khai riêng vào repository. Giấy phép model và dataset tách biệt với giấy phép MIT của source.

## Kiểm tra trước khi công bố

1. Xác nhận `git status --short` sạch; rà soát tên/email tác giả và người commit bằng `git log --format=fuller`.
2. Chạy `bash scripts/setup.sh`, `sha256sum -c SHA256SUMS.txt` và kiểm tra file nhạy cảm trong CI.
3. Rà soát **mọi commit có thể truy cập**, không chỉ source hiện tại: `git log --all --name-status`; thực hiện secret scanning trên toàn lịch sử.
4. Kiểm tra tài khoản GitHub đã xác thực và repository đích không có lịch sử không liên quan.
5. Tạo repository GitHub trống, không tạo sẵn README, license hay .gitignore.

## Push lịch sử đã duyệt

Trong thư mục source, thay tên tài khoản và repository bằng giá trị thực:

```bash
git remote -v
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Không force-push đè lên lịch sử remote đang tồn tại, và không đưa nhánh backup chứa lịch sử khác lên GitHub. Sau khi công bố, bật GitHub Actions, secret scanning và branch protection. CI chỉ chạy với source, không cần Modal secret hay GPU.

Để tái lập kết quả inference, công bố revision SHA của model, commit upstream và phương pháp kiểm thử; không công bố credentials hoặc URL triển khai riêng. Xem [kết quả nghiệm thu](VERIFICATION.vi.md) và [hướng dẫn vận hành](OPERATIONS.vi.md).
