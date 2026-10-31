"""Verify documented Python examples refuse redirects without network access."""

import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("guide", ["CLIENT_INTEGRATION.md", "CLIENT_INTEGRATION.vi.md"])
def test_python_client_rejects_redirect(monkeypatch, guide):
    class ProbeOpener:
        def __init__(self, handler):
            self.handler = handler

        def open(self, request, timeout):
            assert timeout == 900
            assert request.get_method() == "POST"
            return self.handler.redirect_request(
                request, None, 302, "Found", {}, "https://other.example.invalid/"
            )

    def build_opener(*handlers):
        assert len(handlers) == 1
        assert isinstance(handlers[0], urllib.request.HTTPRedirectHandler)
        return ProbeOpener(handlers[0])

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    monkeypatch.setenv("VISION_EMBEDDINGS_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("VISION_EMBEDDINGS_API_TOKEN", "fake-token-for-offline-testing")
    content = (ROOT / "docs" / guide).read_text(encoding="utf-8")
    fence = chr(96) * 3
    example = content.split(fence + "python\n", 1)[1].split("\n" + fence, 1)[0]
    with pytest.raises(RuntimeError, match="Redirect blocked"):
        exec(compile(example, guide, "exec"), {"__name__": "__main__"})


@pytest.mark.parametrize("guide", ["CLIENT_INTEGRATION.md", "CLIENT_INTEGRATION.vi.md"])
def test_node_example_blocks_cross_origin_307_and_never_resends_body(
    guide, monkeypatch, tmp_path
):
    """Execute the exact documented Node client against two local HTTP servers."""
    import shutil
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if shutil.which("node") is None:
        pytest.skip("Node.js is unavailable on this runner")

    reached = {"source": 0, "destination": 0, "body": b"", "auth": None}

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            reached["destination"] += 1
            reached["body"] = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            reached["auth"] = self.headers.get("Authorization")
            response = b'{"model":"wemm-embedding-9b","vector":[0.0,1.0]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_):
            pass

    sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)

    class Redirector(BaseHTTPRequestHandler):
        def do_POST(self):
            reached["source"] += 1
            self.send_response(307)
            self.send_header(
                "Location", f"http://127.0.0.1:{sink.server_port}/incoming"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    source = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    workers = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (sink, source)
    ]
    for worker in workers:
        worker.start()

    content = (ROOT / "docs" / guide).read_text(encoding="utf-8")
    marker = chr(96) * 3
    example = content.split(marker + "javascript\n", 1)[1].split(
        "\n" + marker, 1
    )[0]
    script = tmp_path / "client.mjs"
    script.write_text(example, encoding="utf-8")
    env = __import__("os").environ.copy()
    env.update(
        VISION_EMBEDDINGS_BASE_URL=f"http://127.0.0.1:{source.server_port}",
        VISION_EMBEDDINGS_API_TOKEN="fake-offline-not-a-secret",
    )
    try:
        result = subprocess.run(
            ["node", str(script)], capture_output=True, text=True,
            env=env, timeout=15, check=False,
        )
    finally:
        source.shutdown()
        sink.shutdown()
        source.server_close()
        sink.server_close()
        for worker in workers:
            worker.join(timeout=2)
    assert reached["source"] == 1
    assert reached["destination"] == 0
    assert reached["body"] == b""
    assert reached["auth"] is None
    assert result.returncode != 0
    assert "redirect" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("guide", ["CLIENT_INTEGRATION.md", "CLIENT_INTEGRATION.vi.md"])
def test_csharp_example_disables_auto_redirect(guide):
    """HttpClient's default follows redirects; both docs must override this."""
    content = (ROOT / "docs" / guide).read_text(encoding="utf-8")
    marker = chr(96) * 3
    code = content.split(marker + "csharp\n", 1)[1].split("\n" + marker, 1)[0]
    assert "new HttpClient(new HttpClientHandler" in code
    assert "AllowAutoRedirect = false" in code
    assert "response.EnsureSuccessStatusCode()" in code
