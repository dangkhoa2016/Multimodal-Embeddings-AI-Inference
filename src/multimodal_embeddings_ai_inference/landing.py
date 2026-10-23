# Inline HTML/CSS template intentionally contains long presentation rules.
# ruff: noqa: E501

"""Static, bilingual public landing page for self-hosted API deployments.

The page does not inspect runtime configuration, invoke GPU workers, expose
credentials, or fetch remote assets. Inference and private metadata routes
remain protected; only GET /health and GET /v1/models are public.
"""

from html import escape
from string import Template

PROJECT_URL = "https://github.com/dangkhoa2016/Multimodal-Embeddings-AI-Inference"

COPY = {
    "en": {
        "switch_other": "Tiếng Việt",
        "nav_health": "API health",
        "nav_catalog": "Model catalog",
        "eyebrow": "SELF-HOSTED · MULTIMODAL API",
        "headline_top": "Image and text embeddings,",
        "headline_bottom": "on your infrastructure.",
        "intro": (
            "Build semantic search, image similarity and multimodal retrieval "
            "with two dedicated embedding models behind an authenticated API."
        ),
        "primary": "Explore on GitHub",
        "secondary": "Read API reference",
        "notice": "Embedding requests require a Bearer token. Public health and model metadata do not confirm GPU readiness.",
        "diagram_label": "ARCHITECTURE",
        "flow_client": "Your client",
        "flow_gateway": "CPU gateway",
        "flow_workers": "On-demand GPU workers",
        "terminal_title": "API OVERVIEW",
        "terminal_detail": "Public metadata · Bearer required for inference",
        "model_heading": "Two models. Separate vector spaces.",
        "model_desc": (
            "Choose the model and embedding settings that fit your application. "
            "Vectors from different models are not interchangeable."
        ),
        "qwen_note": "Image + text embeddings",
        "wemm_note": "Image + text embeddings",
        "steps_heading": "Start building",
        "steps_desc": "Everything you need to deploy and connect your own client.",
        "step1": "Deploy the service",
        "step1_body": "Configure a Modal workspace, pinned checkpoints and private Secrets.",
        "step2": "Connect securely",
        "step2_body": "Use a valid Bearer token for embedding and job endpoints.",
        "step3": "Handle cold starts",
        "step3_body": "Prefer asynchronous jobs for longer GPU startup times.",
        "deployment_link": "Deployment guide",
        "client_link": "Client integration",
        "async_link": "Async API contract",
        "footer_note": "MIT-licensed project code. Model weights and dependencies have separate licenses.",
        "footer_links": "Security and operations",
    },
    "vi": {
        "switch_other": "English",
        "nav_health": "Trạng thái API",
        "nav_catalog": "Danh sách model",
        "eyebrow": "API ĐA PHƯƠNG THỨC · TỰ TRIỂN KHAI",
        "headline_top": "Embedding ảnh và văn bản,",
        "headline_bottom": "trên hạ tầng của bạn.",
        "intro": (
            "Xây dựng tìm kiếm ngữ nghĩa, tìm ảnh tương đồng và truy xuất đa phương thức "
            "với hai model embedding độc lập qua API có xác thực."
        ),
        "primary": "Xem mã nguồn GitHub",
        "secondary": "Đọc tài liệu API",
        "notice": "Yêu cầu embedding cần Bearer token. Health và danh sách model công khai không chứng minh GPU đã sẵn sàng.",
        "diagram_label": "KIẾN TRÚC",
        "flow_client": "Ứng dụng của bạn",
        "flow_gateway": "CPU gateway",
        "flow_workers": "GPU khi có yêu cầu",
        "terminal_title": "TỔNG QUAN API",
        "terminal_detail": "Metadata công khai · inference cần Bearer token",
        "model_heading": "Hai model. Hai không gian vector riêng.",
        "model_desc": (
            "Chọn model và cấu hình embedding phù hợp với ứng dụng. "
            "Không được dùng lẫn vector của các model khác nhau."
        ),
        "qwen_note": "Embedding ảnh + văn bản",
        "wemm_note": "Embedding ảnh + văn bản",
        "steps_heading": "Bắt đầu sử dụng",
        "steps_desc": "Tài liệu để triển khai và kết nối ứng dụng của bạn.",
        "step1": "Triển khai dịch vụ",
        "step1_body": "Cấu hình Modal Workspace, checkpoint cố định và Secret riêng tư.",
        "step2": "Kết nối an toàn",
        "step2_body": "Sử dụng Bearer token hợp lệ với endpoint embedding và job.",
        "step3": "Xử lý cold start",
        "step3_body": "Ưu tiên async jobs khi GPU cần nhiều thời gian khởi động.",
        "deployment_link": "Hướng dẫn triển khai",
        "client_link": "Tích hợp ứng dụng",
        "async_link": "Hợp đồng API async",
        "footer_note": "Mã nguồn dự án theo MIT. Trọng số model và thư viện có giấy phép riêng.",
        "footer_links": "Bảo mật và vận hành",
    },
}

PAGE = Template("""<!doctype html>
<html lang="$lang">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0c1322">
  <meta name="description" content="$intro">
  <title>Multimodal Embeddings AI Inference · API</title>
  <style>
    :root{color-scheme:dark;--bg:#0c1322;--panel:#131f33;--line:#30415b;--ink:#f1f7ff;--muted:#a3b3c9;--mint:#7df3ce;--blue:#88b8ff}
    *{box-sizing:border-box}html{scroll-behavior:smooth}
    body{margin:0;background:radial-gradient(ellipse 65% 36% at 91% -5%,#24456d80,transparent),radial-gradient(ellipse 55% 32% at -20% 32%,#164e5280,transparent),var(--bg);color:var(--ink);font:16px/1.62 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    a{color:inherit;text-decoration:none}a:focus-visible{outline:3px solid var(--mint);outline-offset:4px;border-radius:5px}
    .shell{max-width:1160px;margin:auto;padding:0 30px}
    .top{border-bottom:1px solid #ffffff16}.nav{display:flex;align-items:center;justify-content:space-between;gap:20px;min-height:86px}
    .identity{display:flex;align-items:center;gap:13px;font-size:15px;font-weight:750;line-height:1.3;letter-spacing:.01em}
    .mark{width:39px;height:39px;flex:none;display:grid;place-items:center;border:1px solid #60dabe80;border-radius:12px;background:linear-gradient(140deg,#164a5b,#1c3458);color:var(--mint);font-size:22px;font-weight:800}
    .navlinks{display:flex;align-items:center;gap:26px;font-size:13px;color:#c9d7e8}
    .navlinks a:hover,.footer a:hover{color:var(--mint)}
    .nav-status::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--mint);margin-right:8px;vertical-align:1px;box-shadow:0 0 0 3px #7df3ce1d}
    .lang{border:1px solid #536682;border-radius:999px;padding:8px 13px;white-space:nowrap}
    .hero{display:grid;grid-template-columns:minmax(0,1.12fr) minmax(320px,.88fr);align-items:center;gap:60px;padding:106px 0 95px}
    .eyebrow{margin:0 0 22px;color:var(--mint);font-size:12px;letter-spacing:.17em;font-weight:800}
    h1{font-size:clamp(39px,4.8vw,64px);line-height:1.12;letter-spacing:-.048em;margin:0 0 26px;font-weight:800}
    h1 span{display:block;background:linear-gradient(110deg,var(--mint),#8caaff);background-clip:text;-webkit-background-clip:text;color:transparent}
    .intro{color:#bed0e5;font-size:18px;max-width:610px;margin:0 0 28px}
    .actions{display:flex;gap:12px;flex-wrap:wrap}
    .button{display:inline-flex;align-items:center;justify-content:center;gap:10px;border-radius:12px;padding:13px 18px;font-weight:750;font-size:14px;transition:transform .15s ease}
    .button:hover{transform:translateY(-2px)}.primary{background:var(--mint);color:#082132}.secondary{border:1px solid #677d98;color:#e6f0ff}
    .notice{margin:23px 0 0;max-width:590px;font-size:13px;color:var(--muted);border-left:2px solid #69a6d1;padding-left:13px}
    .visual{padding:26px;border:1px solid #3a5471;border-radius:22px;background:linear-gradient(150deg,#1b3047ee,#101a2b);box-shadow:0 25px 100px #0209186b}
    .visual-kicker{display:flex;justify-content:space-between;font-size:11px;letter-spacing:.14em;font-weight:750;color:#9db3cd;margin-bottom:21px}
    .dots{color:var(--mint);letter-spacing:2px}.path{display:grid;gap:10px}
    .node{border:1px solid #53708a;border-radius:12px;background:#1b3045;padding:13px 16px;display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:14px;font-weight:700}
    .node small{font-size:11px;color:var(--mint);font-weight:650}.arrow{text-align:center;line-height:1;color:#90b2ca}
    .split{display:grid;grid-template-columns:1fr 1fr;gap:9px}.split .node{align-items:flex-start;flex-direction:column;gap:3px;font-size:13px}
    .term{border:1px solid #41516a;background:#0b1626;border-radius:13px;padding:17px;margin-top:19px}
    .term-head{color:#91aac5;letter-spacing:.09em;font-size:10px;font-weight:750;margin-bottom:10px}
    .term code{color:#b3f5df;font:12px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
    .term code a{text-decoration:underline;text-underline-offset:3px}.term code a:hover{color:var(--mint)}
    .term p{font-size:11px;color:#8ea5bc;margin:10px 0 0}
    .section{padding:25px 0 82px}.section h2{font-size:clamp(27px,3vw,38px);letter-spacing:-.025em;line-height:1.18;margin:0 0 12px}
    .section-lead{color:var(--muted);max-width:750px;margin:0 0 27px}
    .cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{border:1px solid var(--line);border-radius:17px;background:linear-gradient(145deg,#17273b,#121e30);padding:27px}
    .tag{font-size:11px;font-weight:750;letter-spacing:.11em;color:var(--mint);margin:0 0 14px}.card h3{font-size:21px;line-height:1.3;margin:0 0 14px;overflow-wrap:anywhere}
    .card code{display:inline-block;font-size:12px;background:#0b1727;border:1px solid #334861;padding:5px 9px;color:#bad5f4;border-radius:7px;overflow-wrap:anywhere;max-width:100%}
    .card .sub{font-size:14px;color:var(--muted);margin:18px 0 0}
    .steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.step{padding:22px 23px;border-left:2px solid #408f9e;background:#132337;border-radius:0 12px 12px 0}
    .num{font:700 12px ui-monospace,monospace;color:var(--mint)}.step h3{font-size:17px;margin:9px 0 8px}.step p{font-size:13px;color:var(--muted);margin:0 0 12px}.step a{font-size:13px;color:var(--mint);font-weight:700}
    .footer{border-top:1px solid #ffffff20;padding:28px 0 38px}.footer .shell{display:flex;justify-content:space-between;gap:22px;align-items:flex-start;color:var(--muted);font-size:12px}.footer p{margin:0;max-width:680px}
    @media(max-width:850px){.hero{grid-template-columns:1fr;gap:36px;padding:72px 0 75px}.visual{max-width:680px}.steps{grid-template-columns:1fr}.navlinks{gap:12px}}
    @media(max-width:590px){.shell{padding:0 18px}.nav{min-height:72px}.identity{font-size:12px}.identity span:last-child{display:none}.mark{width:34px;height:34px}.navlinks{gap:12px;font-size:12px}.navlinks .extra{display:inline}.hero{padding:56px 0 64px}.intro{font-size:16px}.visual{padding:18px}.cards{grid-template-columns:1fr}.footer .shell{display:block}.footer p{margin-bottom:11px}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.button{transition:none}}
  </style>
</head>
<body>
  <header class="top">
    <nav class="shell nav" aria-label="Main navigation">
      <a class="identity" href="/" aria-label="Multimodal Embeddings AI Inference">
        <span class="mark" aria-hidden="true">◇</span><span>Multimodal Embeddings<br>AI Inference</span>
      </a>
      <div class="navlinks">
        <a class="extra nav-status" href="/health">$nav_health</a>
        <a class="extra" href="/v1/models">$nav_catalog</a>
        <a class="lang" href="$other_url" hreflang="$other_lang">$switch_other</a>
      </div>
    </nav>
  </header>
  <main class="shell">
    <section class="hero" aria-labelledby="hero-heading">
      <div>
        <p class="eyebrow">$eyebrow</p>
        <h1 id="hero-heading">$headline_top <span>$headline_bottom</span></h1>
        <p class="intro">$intro</p>
        <div class="actions">
          <a class="button primary" href="$project" target="_blank" rel="noopener noreferrer">$primary <span aria-hidden="true">↗</span></a>
          <a class="button secondary" href="$reference" target="_blank" rel="noopener noreferrer">$secondary <span aria-hidden="true">↗</span></a>
        </div>
        <p class="notice">$notice</p>
      </div>
      <div class="visual" aria-label="$diagram_label">
        <div class="visual-kicker"><span>$diagram_label</span><span class="dots" aria-hidden="true">● ● ●</span></div>
        <div class="path">
          <div class="node"><span>$flow_client</span><small>HTTPS</small></div>
          <div class="arrow" aria-hidden="true">↓</div>
          <div class="node"><span>$flow_gateway</span><small>FastAPI · Auth</small></div>
          <div class="arrow" aria-hidden="true">↓</div>
          <div class="node"><span>$flow_workers</span><small>Modal</small></div>
          <div class="split">
            <div class="node"><span>Qwen3-VL-Embedding-8B</span><small>L40S</small></div>
            <div class="node"><span>WeMM-Embedding-9B</span><small>L40S</small></div>
          </div>
        </div>
        <div class="term">
          <div class="term-head">$terminal_title</div>
          <code><a href="/health">GET /health</a> &nbsp; · &nbsp; <a href="/v1/models">GET /v1/models</a></code>
          <p>$terminal_detail</p>
        </div>
      </div>
    </section>

    <section class="section" id="models" aria-labelledby="models-heading">
      <h2 id="models-heading">$model_heading</h2>
      <p class="section-lead">$model_desc</p>
      <div class="cards">
        <article class="card">
          <p class="tag">MODEL 01 · QWEN</p>
          <h3>Qwen/Qwen3-VL-Embedding-8B</h3>
          <code>qwen3-vl-embedding-8b</code>
          <p class="sub">$qwen_note</p>
        </article>
        <article class="card">
          <p class="tag">MODEL 02 · TENCENT</p>
          <h3>tencent/WeMM-Embedding-9B</h3>
          <code>wemm-embedding-9b</code>
          <p class="sub">$wemm_note</p>
        </article>
      </div>
    </section>

    <section class="section" id="start" aria-labelledby="steps-heading">
      <h2 id="steps-heading">$steps_heading</h2>
      <p class="section-lead">$steps_desc</p>
      <div class="steps">
        <article class="step"><span class="num">01</span><h3>$step1</h3><p>$step1_body</p><a href="$deployment" target="_blank" rel="noopener noreferrer">$deployment_link ↗</a></article>
        <article class="step"><span class="num">02</span><h3>$step2</h3><p>$step2_body</p><a href="$client" target="_blank" rel="noopener noreferrer">$client_link ↗</a></article>
        <article class="step"><span class="num">03</span><h3>$step3</h3><p>$step3_body</p><a href="$reference" target="_blank" rel="noopener noreferrer">$async_link ↗</a></article>
      </div>
    </section>
  </main>
  <footer class="footer">
    <div class="shell">
      <p>© Đăng Khoa · <a href="$project/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">MIT</a> · $footer_note</p>
      <a href="$security" target="_blank" rel="noopener noreferrer">$footer_links ↗</a>
    </div>
  </footer>
</body>
</html>""")


def render_landing(language: str = "en") -> str:
    """Render static public HTML; unsupported query values default to English."""
    locale = "vi" if language == "vi" else "en"
    suffix = ".vi" if locale == "vi" else ""
    values = {
        **COPY[locale],
        "lang": locale,
        "project": PROJECT_URL,
        "reference": f"{PROJECT_URL}/blob/main/docs/API_CONTRACT{suffix}.md",
        "deployment": f"{PROJECT_URL}/blob/main/docs/OPERATIONS{suffix}.md",
        "client": f"{PROJECT_URL}/blob/main/docs/CLIENT_INTEGRATION{suffix}.md",
        "security": f"{PROJECT_URL}/blob/main/docs/SECURITY{suffix}.md",
        "other_url": "/?lang=vi" if locale == "en" else "/?lang=en",
        "other_lang": "vi" if locale == "en" else "en",
    }
    return PAGE.substitute({key: escape(value, quote=True) for key, value in values.items()})
