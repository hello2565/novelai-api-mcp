"""本地 mock embeddings 端点，用于离线端到端验证填充脚本。"""

import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        inputs = payload.get("input", [])
        model = payload.get("model", "")
        data = []
        for index, text in enumerate(inputs):
            # 确定性的 4 维伪向量（真实端点形状校验）
            digest = hashlib.sha256((model + text).encode()).digest()
            data.append({
                "index": index,
                "embedding": [b / 255 for b in digest[:4]],
            })
        body = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1])
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
