# License and third-party notices

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](LICENSE-NOTES.vi.md)

The project's original code and documentation are licensed under **MIT**. See [LICENSE](LICENSE).

The model checkpoints and their upstream inference implementations are third-party works and are **not distributed** by this repository:

- [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) — the official model card lists **Apache-2.0**; check the model repository for current terms.
- [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) — Tencent publishes the model parameters and code under **Apache-2.0**; third-party components retain their own licenses.
- [Qwen3-VL-Embedding source](https://github.com/QwenLM/Qwen3-VL-Embedding) — source is fetched separately at build time. Its dependencies may have additional terms.

For each deployment, review the model repository, model card, dependencies and applicable notices. The repository includes three synthetic WebP sample images embedded in the Demo Notebook, but no pretrained weights, third-party image collections or third-party datasets. The MIT license on this project's own code **does not relicense** upstream checkpoints or code.

The API provides image and text embeddings for general retrieval experiments. It is **not** a certified biometric authentication service.
