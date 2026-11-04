# Rails 8 Integration — server-side only

Rails should be a control-plane and evaluation application: imports CompactFaces images, manages jobs and stores/searches embeddings. **It must not load Python models** or directly access Modal Volume/GPU.

## Credentials

Use `bin/rails credentials:edit` (or your organization secret manager):

```yaml
compactfaces_inference:
  base_url: "https://ACTUAL-URL-FROM-MODAL.modal.run"
  token: "REDACTED-KEEP-SECRET"
```

Retrieve the token from the owner-only `~/.config/compactfaces-ai-inference/modal-secret.env` (never place it in committed files). Credentials should only be accessed in Rails on the server side. Example service:

```ruby
# app/services/compactfaces/inference_client.rb
require "net/http"
require "json"

module Compactfaces
  class InferenceClient
    def initialize
      config = Rails.application.credentials.fetch(:compactfaces_inference)
      @base_url = config.fetch(:base_url)
      @token = config.fetch(:token)
      raise "HTTPS required" unless URI(@base_url).scheme == "https"
    end

    def embed_image(image_bytes:, model:, dimension: 1024, request_id: nil)
      body = {
        model: model,
        image_base64: Base64.strict_encode64(image_bytes),
        dimension: dimension,
        request_id: request_id
      }
      post_json("/v1/embeddings", body)
    end

    private

    def post_json(path, payload)
      uri = URI.join(@base_url.end_with?("/") ? @base_url : @base_url + "/", path.delete_prefix("/"))
      req = Net::HTTP::Post.new(uri)
      req["Authorization"] = "Bearer #{@token}"
      req["Content-Type"] = "application/json"
      req.body = JSON.generate(payload)
      response = Net::HTTP.start(uri.host, uri.port, use_ssl: true,
                                 open_timeout: 20, read_timeout: 900) { |http| http.request(req) }
      raise "Inference service HTTP #{response.code}" unless response.is_a?(Net::HTTPSuccess)
      JSON.parse(response.body)
    end
  end
end
```

This is an integration example, not a claim of an already-tested Rails project. Do not log the request body, Bearer token, Base64 image or raw embeddings. Avoid allowing arbitrary URLs from untrusted users; the base URL should be admin-only and validated against HTTPS/approved host policy to prevent Rails-side SSRF.

## Rails data and job guidance

- Tables: `inference_providers`, `embedding_models`, `face_images`, `face_embeddings`, `benchmark_runs`, `benchmark_items`.
- Index uniqueness: `(image_sha256, model_id, revision, dimension, preprocessing_version)` and store request/job status for retries.
- Use Solid Queue jobs with backoff and bounded concurrency. Keep short-lived jobs and checkpoint every batch; Modal cold starts can dominate latency.
- Store vectors in `pgvector` or `halfvec` according to their dimensions and index limits. Keep Qwen and WeMM in **separate vector spaces**; never search mixed vectors as if comparable.
- Do not assume a different angle/expression means same biometric identity solely from dataset metadata. Evaluate separately with manually checked samples.
- Introduce request idempotency at the Rails persistence layer: retrying an API request must not duplicate its stored embedding. The inference API itself does not offer persistent idempotent job tracking.

## Integration contract

Read model IDs from `GET /v1/models`; POST images to `/v1/embeddings`. Store returned `input_sha256`, `model`, `revision`, `dimension`, `normalized`, `elapsed_ms` and vector. Different revisions or preprocessing parameters invalidate prior caches. A live `GET /v1/health` confirms only gateway liveness, not GPU availability.
