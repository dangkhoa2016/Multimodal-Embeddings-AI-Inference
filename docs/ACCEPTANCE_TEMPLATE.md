# Deployment verification worksheet

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ACCEPTANCE_TEMPLATE.vi.md)

This worksheet records the results of an operator's own deployment session, not the release status of this repository. Copy it outside the source checkout and fill in PASS/FAIL/N/A with observations for GPU, Colab and Kaggle. See the [Verification guide](VERIFICATION.md) for procedures.

## Session identity

| Field | Observed value |
| --- | --- |
| UTC start/end | — |
| Client commit/tree and clean checkout | — |
| Deployed server source commit | Record the server SHA separately from the client |
| Server source evidence | Operator declaration or deployment record; state which |
| Exact CI run URL | — |
| Environment | Colab/Kaggle/CPU client; Python version; GPU hardware if applicable |
| Qwen checkpoint revision | — |
| WeMM checkpoint revision | — |
| Enabled phases and request counts | — |

Use `scripts/record_acceptance.py` to export the source identity, Python environment and optional CPU checks; record deployment-specific GPU/UI observations here. Always identify the source revision used for each run.

## Observed cases

| Case | Status | Sanitized observation |
| --- | --- | --- |
| GPU: authenticated text/image for Qwen | — | — |
| GPU: authenticated text/image for WeMM | — | — |
| GPU: expected revision, finite normalized vectors, dimensions | — | — |
| GPU: independent workers and alternating routing | — | — |
| Hosting: Modal Workspace usage budget and net spend limit | — | Record spend limits; exclude billing details and tokens |
| Hosting: per-client ingress quotas and restricted token access | — | Record ingress quotas and token-access policy |
| UI: model dropdown/editable fields | — | — |
| UI: built-in previews before credentials | — | — |
| UI: hidden connection prompts | — | — |
| UI: six requests and bilingual matrix/report | — | — |
| UI: light/dark mode and table accessibility | — | — |
| UI: changed inputs/model, local recovery hints | — | — |
| UI: missing custom image and recovery | — | — |
| UI: benchmark disabled makes no calls | — | — |
| UI: Stop during warm phase, rerun blocked | — | — |

Record Colab and Kaggle runs separately. Warm-only measurements do not cover cold-start; document each observation. Never retry a dispatched POST automatically.

## Evidence and notes

Attach redacted observations or screenshots by case, including runtime and source identity. Exclude tokens, private service URLs, input data, hashes of private inputs and raw vectors. Record the scope, outcomes and observed errors. This worksheet describes one deployment session; it is not an accuracy or production-capacity certification.
