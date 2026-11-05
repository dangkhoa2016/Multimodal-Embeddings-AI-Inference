# Acceptance record template

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ACCEPTANCE_TEMPLATE.vi.md)

Copy this template outside the repository for each real GPU or hosted notebook session. All cases start as **NOT_RUN**; change a case to PASS/FAIL only after observing it. Keep CPU, GPU and UI evidence separate. The source of truth for procedures is [Verification](VERIFICATION.md).

## Session identity

| Field | Observed value |
| --- | --- |
| UTC start/end | Not recorded |
| Client commit/tree and clean checkout | Not recorded |
| Deployed server source commit | Unknown; do not substitute the client SHA |
| Server source evidence | Operator declaration or deployment record; state which |
| Exact CI run URL | Not recorded |
| Environment | Colab/Kaggle/CPU client; Python version; GPU hardware if applicable |
| Qwen checkpoint revision | Not recorded |
| WeMM checkpoint revision | Not recorded |
| Enabled phases and request counts | Not recorded |

For source and CPU package versions, create the companion JSON using `scripts/record_acceptance.py`; it does not test GPU or hosted UI. Record the source SHA actually tested; a different revision must not be substituted for the executed code.

## Observed cases

| Case | Status | Sanitized observation |
| --- | --- | --- |
| GPU: authenticated text/image for Qwen | NOT_RUN | No evidence yet |
| GPU: authenticated text/image for WeMM | NOT_RUN | No evidence yet |
| GPU: expected revision, finite normalized vectors, dimensions | NOT_RUN | No evidence yet |
| GPU: independent workers and alternating routing | NOT_RUN | No evidence yet |
| UI: model dropdown/editable fields | NOT_RUN | No evidence yet |
| UI: built-in previews before credentials | NOT_RUN | No evidence yet |
| UI: hidden connection prompts | NOT_RUN | No evidence yet |
| UI: six requests and bilingual matrix/report | NOT_RUN | No evidence yet |
| UI: light/dark mode and table accessibility | NOT_RUN | No evidence yet |
| UI: changed inputs/model, local recovery hints | NOT_RUN | No evidence yet |
| UI: missing custom image and recovery | NOT_RUN | No evidence yet |
| UI: benchmark disabled makes no calls | NOT_RUN | No evidence yet |
| UI: Stop during warm phase, rerun blocked | NOT_RUN | No evidence yet |

Test Colab and Kaggle in separate sessions. A warm-only run cannot confirm cold-start behavior; missing accessibility checks stay NOT_RUN. Do not retry a dispatched POST automatically.

## Evidence and remaining gaps

Attach redacted observations or screenshots by case, including runtime and source identity. Exclude tokens, private service URLs, input data, hashes of private inputs and raw vectors. State failed or untested cases explicitly. This template is not evidence of a successful run and does not establish retrieval accuracy or production capacity.
