# License, provenance and dataset boundaries

- This repository's own code and documentation: **MIT**, see [LICENSE](LICENSE).
- [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B): third-party model and weights, published under Apache-2.0; subject to its model card and upstream dependencies.
- [Tencent WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B): third-party model and weights, published under Apache-2.0; third-party components keep their own licenses.
- [Qwen3-VL-Embedding upstream inference code](https://github.com/QwenLM/Qwen3-VL-Embedding) and other libraries: license and version must be reviewed separately. This repository fetches upstream code at Modal image build time, does not relicense it and pins its upstream Git commit (393e2978d27852b0d0230d6994f37f9c15bed73c); license and third-party notices still require independent review before redistribution.
- [CompactFaces Synthetic Face Dataset](https://github.com/dangkhoa2016/CompactFaces-Synthetic-Face-Pipeline): images/derived previews have distinct **CC BY-NC 4.0** licensing only to the extent the publisher has rights; code MIT does **not** grant commercial rights to images. No dataset images, private inputs or model weights are distributed in this API repo.
- This API is a **research/education prototype**. It does not establish biometric uniqueness, synthetic-to-real transfer, visual identity consistency, or suitability for authentication.
