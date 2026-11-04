# HTTP clients: Python, Node.js and C#/.NET

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CLIENT_INTEGRATION.vi.md)

The examples below call the API using HTTP and JSON from Python, Node.js and C#/.NET.
Any server-side application that can send authenticated HTTPS requests can call it.

## Common configuration

```bash
export VISION_EMBEDDINGS_BASE_URL="https://YOUR_DEPLOYMENT.modal.run"
export VISION_EMBEDDINGS_API_TOKEN="YOUR_SERVER_SIDE_BEARER_TOKEN"
```

Keep the token in your server's secret manager. Do not expose it in browser code or commit it to Git.
For local mock testing, use `http://127.0.0.1:8000` and its locally generated token.

## Python 3.11+ (standard library)

Save the following as `client.py` and run `python client.py`.

```python
import json
import os
from urllib.request import HTTPRedirectHandler, Request, build_opener

class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise RuntimeError("Redirect blocked for authenticated inference requests.")

url = os.environ["VISION_EMBEDDINGS_BASE_URL"].rstrip("/") + "/v1/embeddings"
token = os.environ["VISION_EMBEDDINGS_API_TOKEN"]
payload = {
    "model": "qwen3-vl-embedding-8b",
    "text": "a mountain landscape",
    "dimension": 1024,
}
request = Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    },
    method="POST",
)
with build_opener(RejectRedirects()).open(request, timeout=900) as response:
    result = json.load(response)
print(result["model"], len(result["vector"]))
```

### Minimal text–image cosine comparison (Python)

For the shortest complete multimodal example, first run the Python text request above. Append the following to the same `client.py` file to send **one authorized image** to the **same model**, then compare its vector with the text vector. Supply an actual local PNG/JPEG/WebP path. This is an API usage demonstration, not retrieval-quality validation. Both responses must have the same nonempty checkpoint revision and valid `embedding_metadata.config_sha256`; missing metadata prevents comparison.

```python
import base64
from pathlib import Path

image_bytes = Path("YOUR_AUTHORIZED_IMAGE.webp").read_bytes()
image_request = Request(
    url,
    data=json.dumps({
        "model": payload["model"],
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "dimension": payload["dimension"],
    }).encode("utf-8"),
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    method="POST",
)
with build_opener(RejectRedirects()).open(image_request, timeout=900) as response:
    image_result = json.load(response)
def embedding_identity(item):
    metadata = item.get("embedding_metadata")
    revision = item.get("revision")
    config_sha = metadata.get("config_sha256") if isinstance(metadata, dict) else None
    if not isinstance(revision, str) or not revision:
        raise ValueError("Embedding revision is required")
    if not isinstance(config_sha, str) or len(config_sha) != 64:
        raise ValueError("Embedding configuration SHA-256 is required")
    if any(ch not in "0123456789abcdef" for ch in config_sha):
        raise ValueError("Invalid embedding configuration SHA-256")
    return item["model"], item["dimension"], revision, config_sha, len(item["vector"])

if embedding_identity(result) != embedding_identity(image_result):
    raise ValueError("Text and image embedding configurations differ")
cosine = sum(float(a) * float(b) for a, b in
             zip(result["vector"], image_result["vector"], strict=True))
print("Text–image cosine:", round(cosine, 4))
```

Keep the token server-side; raw images are sent to your selected API. The full notebook adds response-digest, model-revision and L2 checks, bilingual feedback and accessible HTML.

## Node.js 18+ (built-in fetch)

Save as `client.mjs` and run `node client.mjs`; no third-party NPM package is needed.

```javascript
const baseUrl = process.env.VISION_EMBEDDINGS_BASE_URL?.replace(/\/$/, "");
const token = process.env.VISION_EMBEDDINGS_API_TOKEN;
if (!baseUrl || !token) throw new Error("Missing API configuration");

const response = await fetch(baseUrl + "/v1/embeddings", {
  redirect: "error", // Do not forward input bodies to redirected endpoints.
  method: "POST",
  headers: {
    Authorization: "Bearer " + token,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "wemm-embedding-9b",
    text: "a mountain landscape",
    dimension: 1024,
  }),
});
if (!response.ok) throw new Error("Inference HTTP " + response.status);
const result = await response.json();
console.log(result.model, result.vector.length);
```

## C# / .NET 6+ (HttpClient)

Create a console project with `dotnet new console`, replace `Program.cs` with the code below and run `dotnet run`.

```csharp
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;

var baseUrl = Environment.GetEnvironmentVariable("VISION_EMBEDDINGS_BASE_URL")
    ?? throw new InvalidOperationException("Missing VISION_EMBEDDINGS_BASE_URL");
var token = Environment.GetEnvironmentVariable("VISION_EMBEDDINGS_API_TOKEN")
    ?? throw new InvalidOperationException("Missing VISION_EMBEDDINGS_API_TOKEN");
var timeoutValue = Environment.GetEnvironmentVariable("VISION_EMBEDDINGS_REQUEST_TIMEOUT");
var timeoutSeconds = string.IsNullOrWhiteSpace(timeoutValue)
    ? 900 : int.Parse(timeoutValue);
if (timeoutSeconds is < 1 or > 3600)
    throw new ArgumentOutOfRangeException(nameof(timeoutSeconds));
using var client = new HttpClient(new HttpClientHandler
{
    AllowAutoRedirect = false // Never resend input data to redirected endpoints.
})
{
    BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/"),
    Timeout = TimeSpan.FromSeconds(timeoutSeconds)
};
client.DefaultRequestHeaders.Authorization =
    new AuthenticationHeaderValue("Bearer", token);
using var response = await client.PostAsJsonAsync(
    "v1/embeddings",
    new {
        model = "qwen3-vl-embedding-8b",
        text = "a mountain landscape",
        dimension = 1024
    });
response.EnsureSuccessStatusCode();
using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
var result = document.RootElement;
Console.WriteLine(result.GetProperty("model").GetString() + " " +
                  result.GetProperty("vector").GetArrayLength());
```

**C# timeout:** VISION_EMBEDDINGS_REQUEST_TIMEOUT is in seconds (example default 900, range 1–3600). Adjust for your deployment cold-start behavior. Do not automatically retry a timed-out POST because the server may still be working and incur more cost.

**Redirect policy:** Python, Node.js and C# explicitly reject redirects for authenticated POST requests. Point your client to the final HTTPS URL; do not silently resend text/image data after HTTP 3xx.

## Models and image input

- `qwen3-vl-embedding-8b` is [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B).
- `wemm-embedding-9b` is [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B).

To embed an image instead of text, replace the `text` field with `image_base64` containing Base64-encoded JPEG/PNG/WebP bytes (without a data-URI prefix).
The response contains `model`, `repository`, `revision`, `dimension`, `vector`, `input_sha256` and `elapsed_ms`.
Keep each model and checkpoint revision in a separate vector index; their embeddings are not interchangeable.

See the [API contract](API_CONTRACT.md) for endpoints, limits and error codes.
