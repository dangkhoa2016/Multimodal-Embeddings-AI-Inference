# Security Policy

Report vulnerabilities privately by email to `i.am@dangkhoa.dev` rather than filing a public issue. Do not send real faces, access tokens or private datasets in reports.

## Security boundaries

- All `/v1/*` routes require a secret `CF_API_TOKEN`; never commit the token. This is *application-level* Bearer authentication. Optional Modal Proxy Authentication would be a separate layer requiring integration changes.
- The API accepts image bytes encoded as Base64; it does not fetch arbitrary URLs. Images are validated as JPEG/PNG/WebP and bound by size and pixel count.
- Uploaded images are processed in memory and are not intentionally persisted by the gateway. Server log policy must not include raw images, Base64 input or embeddings.
- The GPU worker only reads pre-downloaded, revision-matched checkpoints from the shared Volume. Upstream model code and weights have their own supply-chain risk; review upstream license and commit before deployment.
- Rails must keep the token server-side and use TLS; never send it to a browser client. Rate limiting and quota/budget controls are expected at the upstream gateway or deployment edge before opening a public demo.
- Synthetic portraits are used for research. This is not an identity verification or access-control system; do not apply it to real people without legal/privacy review and explicit consent.

## Production readiness

The release includes mock contract tests, not certification of GPU behavior or biometric accuracy. Confirm live L40S loading, request isolation, memory utilization, cost limits and upstream license compliance before public traffic.
