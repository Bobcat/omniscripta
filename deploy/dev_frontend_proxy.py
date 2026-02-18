import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError

FRONTEND_DIR = os.getenv("TRANSCRIBE_FRONTEND_DIR", "/home/gunnar/projects/transcribe-dev/static")
API_BASE = os.getenv("TRANSCRIBE_API_BASE", "http://127.0.0.1:8001").rstrip("/")
HOST = os.getenv("TRANSCRIBE_FRONTEND_HOST", "127.0.0.1")
PORT = int(os.getenv("TRANSCRIBE_FRONTEND_PORT", "8010"))


class Handler(SimpleHTTPRequestHandler):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

  def _proxy_api(self):
    body = None
    if self.command in ("POST", "PUT", "PATCH"):
      length = int(self.headers.get("Content-Length", "0") or "0")
      body = self.rfile.read(length)

    req = Request(API_BASE + self.path, data=body, method=self.command)
    for key, value in self.headers.items():
      if key.lower() not in ("host", "connection", "content-length", "transfer-encoding"):
        req.add_header(key, value)

    try:
      with urlopen(req, timeout=600) as resp:
        self.send_response(resp.status)
        for key, value in resp.headers.items():
          if key.lower() not in ("transfer-encoding", "connection"):
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(resp.read())
    except HTTPError as err:
      self.send_response(err.code)
      self.end_headers()
      self.wfile.write(err.read())

  def do_GET(self):
    if self.path.startswith("/api/"):
      return self._proxy_api()
    return super().do_GET()

  do_POST = do_PUT = do_PATCH = do_DELETE = do_GET


def main():
  server = ThreadingHTTPServer((HOST, PORT), Handler)
  server.serve_forever()


if __name__ == "__main__":
  main()
