## Summary / Tóm tắt

Describe the problem and the change. / Mô tả vấn đề và thay đổi.

## Changes / Nội dung thay đổi

- Describe the key change / Mô tả thay đổi chính

## Validation / Kiểm thử

- [ ] Local tests: `pytest -q`
- [ ] Lint: `ruff check src modal_app.py modal_models.py scripts tests`
- [ ] Bilingual documentation: `python scripts/audit_release.py`
- [ ] Source manifest: `sha256sum -c --strict SHA256SUMS.txt`
- [ ] Both EN/VI documents updated when relevant / Đã cập nhật cả EN/VI khi cần

## Security and reproducibility / Bảo mật và khả năng tái lập

- [ ] No keys, tokens, private endpoints, user images, model weights or personal data included
- [ ] Real model revisions and dependency changes documented if applicable
- [ ] Model accuracy or real-GPU claims supported by sanitized evidence, if made

## Related issues / Issue liên quan

Closes #
