# HTTP client: Python, Node.js và C#/.NET

🌐 Ngôn ngữ / Language: [English](CLIENT_INTEGRATION.md) | **Tiếng Việt**

Các ví dụ dưới đây gọi API bằng HTTP và JSON từ Python, Node.js và C#/.NET.
Bất kỳ ứng dụng phía server nào gửi được request HTTPS có xác thực đều có thể gọi API.

## Cấu hình chung

```bash
export VISION_EMBEDDINGS_BASE_URL="https://YOUR_DEPLOYMENT.modal.run"
export VISION_EMBEDDINGS_API_TOKEN="YOUR_SERVER_SIDE_BEARER_TOKEN"
```

Lưu token trong secret manager của server. Không nhúng token vào mã trình duyệt hoặc commit lên Git.

## Mock server cục bộ

Hoàn tất [quickstart cục bộ](../README.vi.md#thử-api-cục-bộ) để cài dependency CPU. Để thử các ví dụ client mà chưa triển khai model, khởi động mock server và giữ dịch vụ chạy:

```bash
source .venv/bin/activate
MOCK_API_PORT="${MOCK_API_PORT:-8000}"
export VISION_EMBEDDINGS_BASE_URL="http://127.0.0.1:$MOCK_API_PORT"
export VISION_EMBEDDINGS_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
VISION_EMBEDDINGS_BACKEND=mock python -m uvicorn \
  multimodal_embeddings_ai_inference.api:local_app --factory \
  --host 127.0.0.1 --port "$MOCK_API_PORT" &
MOCK_API_PID=$!
```

Chờ thông báo `Application startup complete`, rồi chạy ví dụ Python, Node.js hoặc C# trong **chính terminal này**. Client và server dùng chung token vừa tạo và URL gốc. Mock trả về vector mô phỏng. Nếu cổng 8000 đã được dùng, đặt `MOCK_API_PORT` thành cổng còn trống trước khi chạy khối lệnh.

Dừng mock server sau khi dùng xong:

```bash
kill "$MOCK_API_PID"
```

Script quickstart dùng server tạm riêng và tự tắt khi hoàn tất; script đó không giữ dịch vụ chạy cho các client này.

## Python 3.11+ (thư viện chuẩn)

Lưu đoạn sau thành `client.py` rồi chạy `python client.py`.

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

### So sánh cosine văn bản–ảnh tối giản (Python)

Để xem ví dụ đa phương thức ngắn gọn, chạy request văn bản bằng Python ở trên. Thêm đoạn sau vào cuối cùng file `client.py` để gửi **một ảnh được phép sử dụng** đến **cùng model**, rồi so sánh vector ảnh với vector văn bản. Cần thay đường dẫn bằng file PNG/JPEG/WebP có thật. Đây là minh họa cách gọi API, **không phải** kiểm chứng chất lượng tìm kiếm. Hai phản hồi phải có cùng checkpoint revision và `embedding_metadata.config_sha256` hợp lệ; thiếu metadata cũng phải từ chối so sánh.

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
    raise ValueError("Cấu hình embedding văn bản và ảnh không khớp")
cosine = sum(float(a) * float(b) for a, b in
             zip(result["vector"], image_result["vector"], strict=True))
print("Cosine văn bản–ảnh:", round(cosine, 4))
```

Giữ token ở server; ảnh gốc được gửi tới API đã chọn. Notebook demo đầy đủ bổ sung xác minh mã băm, revision model, chuẩn L2, thông báo song ngữ và báo cáo HTML dễ đọc.

## Node.js 18+ (fetch tích hợp)

Lưu thành `client.mjs` rồi chạy `node client.mjs`; không cần cài thêm package NPM.

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

Tạo console project bằng `dotnet new console`, thay `Program.cs` bằng đoạn sau rồi chạy `dotnet run`.

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

**Timeout C#:** Biến `VISION_EMBEDDINGS_REQUEST_TIMEOUT` tính bằng giây (mặc định ví dụ 900, từ 1 đến 3600). Điều chỉnh theo cold start của deployment. Không tự động retry POST khi timeout vì server có thể vẫn chạy và phát sinh chi phí.

**Chính sách redirect:** Python, Node.js và C# đều chặn redirect cho POST có xác thực. Hãy sử dụng URL HTTPS đích cuối cùng; không tự gửi lại nội dung văn bản/ảnh sau HTTP 3xx.

## Gửi và theo dõi embedding bất đồng bộ

Khi GPU cold-start, nên dùng `POST /v1/embeddings/async`. Đặt
`request_id` ổn định cho từng đầu vào/profile, sau đó lưu `job_id`
trả về. Poll bằng **GET**, không khởi tạo inference mới:

```bash
curl --fail-with-body -sS "$VISION_EMBEDDINGS_BASE_URL/v1/embeddings/async" \
  -H "Authorization: Bearer $VISION_EMBEDDINGS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"wemm-embedding-9b","text":"phong cảnh núi","dimension":1024,"request_id":"public-example-001"}'
# Dùng job_id nhận được:
curl --fail-with-body -sS "$VISION_EMBEDDINGS_BASE_URL/v1/embeddings/jobs/YOUR_JOB_ID" \
  -H "Authorization: Bearer $VISION_EMBEDDINGS_API_TOKEN"
```

HTTP 202 nghĩa là đang xếp hàng/xử lý, **chưa thành công**. Khi `SUCCEEDED`,
đọc `result.vector` và lưu vào kho vector bền vững. Nếu mất ACK POST khi
chưa có job ID, **không gửi lại POST**, kể cả cùng request ID. Hãy đối soát
bằng endpoint GET chỉ đọc:

```bash
curl --fail-with-body -sS \
  "$VISION_EMBEDDINGS_BASE_URL/v1/embeddings/jobs/by-request/public-example-001" \
  -H "Authorization: Bearer $VISION_EMBEDDINGS_API_TOKEN"
```

Server tự tính job ID trong namespace của **client đã xác thực**; đổi API key
vẫn có thể xem job nếu giữ nguyên client ID. Nếu GET trả 404, bản ghi có
thể chưa từng được tạo hoặc đã hết hạn: kiểm tra log deployment và cache
riêng, **không tự POST lại, kể cả cùng request ID**. Deployment cũ chưa
có endpoint này vẫn có thể GET qua job ID đã lưu hoặc khóa legacy của owner
(xem Notebook Guide). Endpoint đồng bộ không có idempotency.
Xem [hợp đồng async](API_CONTRACT.vi.md#gpu-job-bất-đồng-bộ-cold-start-và-khôi-phục).

Không tự tạo request ID mới khi timeout hoặc nhận `INDETERMINATE`:
GPU worker cũ có thể vẫn đang chạy. Xoay vòng Bearer token không làm
mất job ID lưu ở backend.

## Model và đầu vào ảnh

- `qwen3-vl-embedding-8b` tương ứng [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B).
- `wemm-embedding-9b` tương ứng [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B).

Để tạo embedding từ ảnh thay cho văn bản, thay trường `text` bằng `image_base64` chứa byte JPEG/PNG/WebP mã hóa Base64 (không có tiền tố data URI).
Phản hồi chứa `model`, `repository`, `revision`, `dimension`, `vector`, `input_sha256` và `elapsed_ms`.
Lưu từng model và revision checkpoint trong chỉ mục vector riêng vì embedding của chúng không thể sử dụng thay thế cho nhau.

Xem [hợp đồng API](API_CONTRACT.vi.md) để biết endpoint, giới hạn và mã lỗi.
