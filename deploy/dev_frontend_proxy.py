import os
import selectors
import socket
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

FRONTEND_DIR = os.getenv("TRANSCRIBE_FRONTEND_DIR", "/home/gunnar/projects/transcribe-dev/static")
API_BASE = os.getenv("TRANSCRIBE_API_BASE", "http://127.0.0.1:8001").rstrip("/")
HOST = os.getenv("TRANSCRIBE_FRONTEND_HOST", "127.0.0.1")
PORT = int(os.getenv("TRANSCRIBE_FRONTEND_PORT", "8010"))
WS_PROXY_BUFFER_BYTES = int(os.getenv("TRANSCRIBE_FRONTEND_WS_BUFFER_BYTES", "65536"))
WS_PROXY_IDLE_TIMEOUT_S = float(os.getenv("TRANSCRIBE_FRONTEND_WS_IDLE_TIMEOUT_S", "1800"))
LIVE_AUTO_GAIN_CONTROL = os.getenv("TRANSCRIBE_LIVE_AUTO_GAIN_CONTROL", "0").strip().lower() in ("1", "true", "yes", "on")

_api_parts = urlparse(API_BASE)
_api_host = _api_parts.hostname or "127.0.0.1"
_api_port = int(_api_parts.port or (443 if _api_parts.scheme == "https" else 80))
_api_netloc = f"{_api_host}:{_api_port}"


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def _proxy_api_http(self):
        body = None
        if self.command in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length)

        req = Request(API_BASE + self.path, data=body, method=self.command)
        for key, value in self.headers.items():
            if key.lower() not in ("connection", "content-length", "transfer-encoding"):
                req.add_header(key, value)

        try:
            with urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.end_headers()
                payload = resp.read()
                if payload:
                    self.wfile.write(payload)
        except HTTPError as err:
            payload = err.read()
            try:
                self.send_response(err.code)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True
                return
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client disconnected while reading proxied response.
            self.close_connection = True
            return
        except Exception as err:
            payload = f"API proxy error: {type(err).__name__}: {err}\n".encode("utf-8", errors="ignore")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True
                return

    def _is_ws_upgrade(self) -> bool:
        if not self.path.startswith("/api/"):
            return False
        upgrade = str(self.headers.get("Upgrade") or "").strip().lower()
        connection = str(self.headers.get("Connection") or "").strip().lower()
        return upgrade == "websocket" and "upgrade" in connection

    def _proxy_api_websocket(self):
        upstream = None
        selector = None
        # Once upgraded, this connection is no longer HTTP. Avoid parsing ws
        # frames as additional HTTP requests after the proxy loop returns.
        self.close_connection = True
        try:
            upstream = socket.create_connection((_api_host, _api_port), timeout=10.0)
            upstream.settimeout(None)

            request_lines = [f"{self.command} {self.path} HTTP/1.1\r\n"]
            for key, value in self.headers.items():
                request_lines.append(f"{key}: {value}\r\n")
            if "host" not in {k.lower() for k in self.headers.keys()}:
                request_lines.append(f"Host: {_api_netloc}\r\n")
            request_lines.append("\r\n")
            upstream.sendall("".join(request_lines).encode("utf-8"))

            selector = selectors.DefaultSelector()
            selector.register(self.connection, selectors.EVENT_READ, upstream)
            selector.register(upstream, selectors.EVENT_READ, self.connection)

            while True:
                events = selector.select(timeout=WS_PROXY_IDLE_TIMEOUT_S)
                if not events:
                    return
                for key, _ in events:
                    src = key.fileobj
                    dst = key.data
                    try:
                        data = src.recv(WS_PROXY_BUFFER_BYTES)
                    except OSError:
                        data = b""
                    if not data:
                        return
                    try:
                        dst.sendall(data)
                    except OSError:
                        return
        except Exception as err:
            try:
                self.send_error(502, f"WebSocket proxy error: {type(err).__name__}: {err}")
            except Exception:
                pass
        finally:
            if selector is not None:
                try:
                    selector.close()
                except Exception:
                    pass
            if upstream is not None:
                try:
                    upstream.close()
                except Exception:
                    pass

    def do_GET(self):
        if self._is_ws_upgrade():
            return self._proxy_api_websocket()
        if self.path == "/api/config":
            return self._serve_config()
        if self.path.startswith("/api/"):
            return self._proxy_api_http()
        return super().do_GET()

    def _serve_config(self):
        import json
        config = {
            "live_auto_gain_control": LIVE_AUTO_GAIN_CONTROL
        }
        payload = json.dumps(config).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path.startswith("/api/"):
            return self._proxy_api_http()
        return self.send_error(405)

    do_PUT = do_PATCH = do_DELETE = do_POST
    do_OPTIONS = do_POST

    def log_message(self, format: str, *args):
        # Keep default access logging but avoid exceptions on unusual bytes.
        try:
            super().log_message(format, *args)
        except Exception:
            pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
