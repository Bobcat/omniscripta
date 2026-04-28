from __future__ import annotations

import json
import os
import time
from urllib.error import URLError
from urllib.request import urlopen
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_API_STARTED_MONO = time.monotonic()


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_sensitive_key(name: str) -> bool:
    k = str(name or "").strip().lower()
    if not k or k.endswith("_env"):
        return False
    if k in {
        "token",
        "hf_token",
        "api_key",
        "apikey",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "authorization",
        "bearer_token",
    }:
        return True
    return (
        k.endswith("_token")
        or k.endswith("_api_key")
        or k.endswith("_apikey")
        or k.endswith("_password")
        or k.endswith("_secret")
    )


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if _is_sensitive_key(str(key)):
                out[str(key)] = "***REDACTED***"
            else:
                out[str(key)] = _redact_sensitive(child)
        return out
    if isinstance(value, list):
        return [_redact_sensitive(v) for v in value]
    return value


def _file_config_source(*, source_id: str, title: str, path: Path) -> Dict[str, Any]:
    exists = path.exists()
    size_bytes: int | None = None
    mtime_utc: str | None = None
    if exists:
        try:
            stat = path.stat()
            size_bytes = int(stat.st_size)
            mtime_utc = _iso_utc(float(stat.st_mtime))
        except Exception:
            size_bytes = None
            mtime_utc = None

    parse_ok = False
    data: Dict[str, Any] = {}
    error: str | None = None
    if not exists:
        error = "File not found"
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = raw
                parse_ok = True
            else:
                error = "JSON root must be an object"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    return {
        "id": source_id,
        "title": title,
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "mtime_utc": mtime_utc,
        "parse_ok": parse_ok,
        "data": _redact_sensitive(data),
        "error": error,
    }


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(dict(out.get(key) or {}), value)
        else:
            out[key] = value
    return out


def _coerce_live_ui_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    src = data if isinstance(data, dict) else {}
    defaults: Dict[str, Any] = {
        "transcript_presentation_mode": "segment_blocks_diarize_hard_v1",
        "speaker_labels_default_enabled": True,
        "transcript_format_rules": {
            "blockEverySegments": 3,
            "blockMinChars": 220,
            "blockMinWords": 35,
        },
    }
    out = _deep_merge_dict(defaults, src)
    rules = out.get("transcript_format_rules")
    if not isinstance(rules, dict):
        rules = dict(defaults["transcript_format_rules"])
    nrules: Dict[str, Any] = {}
    int_keys = {"blockEverySegments", "blockMinChars", "blockMinWords"}
    for k in defaults["transcript_format_rules"].keys():
        v = rules.get(k, defaults["transcript_format_rules"][k])
        if k in int_keys:
            try:
                nrules[k] = max(0, int(v))
            except Exception:
                nrules[k] = int(defaults["transcript_format_rules"][k])
    return {
        "transcript_presentation_mode": "segment_blocks_diarize_hard_v1",
        "speaker_labels_default_enabled": bool(out.get("speaker_labels_default_enabled", True)),
        "transcript_format_rules": nrules,
    }


def _load_ui_settings() -> Dict[str, Any]:
    config_dir = (_REPO_ROOT / "config").resolve()
    base_path = (config_dir / "ui_settings.json").resolve()
    local_path = (config_dir / "ui_settings.local.json").resolve()
    base_obj = _load_json_object(base_path) if base_path.exists() else {}
    local_obj = _load_json_object(local_path) if local_path.exists() else {}
    merged = _deep_merge_dict(base_obj, local_obj)
    live_obj = merged.get("live")
    return {
        "version": str(merged.get("version") or "ui_settings_v1"),
        "live": _coerce_live_ui_settings(live_obj if isinstance(live_obj, dict) else {}),
    }


def _count_queue_dirs(path: Path) -> int:
    try:
        return sum(1 for p in path.iterdir() if p.is_dir() and not str(p.name).startswith("."))
    except Exception:
        return 0


def _queue_snapshot(queue_root: Any) -> Dict[str, Any]:
    return {
        "name": str(getattr(queue_root, "name", "")),
        "base": str(getattr(queue_root, "base", "")),
        "inbox": int(_count_queue_dirs(Path(getattr(queue_root, "inbox", "")))),
        "running": int(_count_queue_dirs(Path(getattr(queue_root, "running", "")))),
        "done": int(_count_queue_dirs(Path(getattr(queue_root, "done", "")))),
        "error": int(_count_queue_dirs(Path(getattr(queue_root, "error", "")))),
    }


def _api_ops_metrics_snapshot() -> Dict[str, Any]:
    from live.config import LIVE_SESSIONS
    from upload.jobs.roots import UPLOAD_PREP_QUEUE, UPLOAD_WORKER_QUEUE
    from upload.pipeline.coordinator import UPLOAD_BATCH_COORDINATOR

    now_ts = datetime.now(timezone.utc).timestamp()
    live_metrics = dict(LIVE_SESSIONS.metrics_payload() or {})
    live_limits = dict(live_metrics.get("limits") or {})
    live_active_sessions = int(max(0, int(live_metrics.get("active_sessions") or 0)))
    live_ws_connected = int(max(0, int(live_metrics.get("active_ws_connected") or 0)))
    live_max_sessions = int(max(0, int(live_limits.get("max_sessions") or 0)))
    upload_prep = _queue_snapshot(UPLOAD_PREP_QUEUE)
    upload_worker = _queue_snapshot(UPLOAD_WORKER_QUEUE)
    coordinator = dict(UPLOAD_BATCH_COORDINATOR.metrics_snapshot() or {})

    pool_metrics_url = os.getenv("ASR_POOL_OPS_METRICS_URL", "http://127.0.0.1:18090/ops/metrics")
    worker_batch_metrics_url = os.getenv("ASR_WORKER_BATCH_OPS_METRICS_URL", "http://127.0.0.1:18111/ops/metrics")
    dependencies = {
        "asr_pool": _ops_service_probe(url=pool_metrics_url),
        "asr_worker_batch": _ops_service_probe(url=worker_batch_metrics_url),
    }
    dependencies_total = int(len(dependencies))
    dependencies_up = int(sum(1 for v in dependencies.values() if bool(v.get("ok"))))

    health = "ok"
    health_reason = "Healthy"
    if bool(coordinator.get("enabled")) and not bool(coordinator.get("running")):
        health = "error"
        health_reason = "Upload coordinator enabled but not running"
    elif live_max_sessions > 0 and live_active_sessions > live_max_sessions:
        health = "error"
        health_reason = "Live active sessions exceeded configured max_sessions"
    elif dependencies_up < dependencies_total:
        health = "warn"
        health_reason = "One or more downstream services unavailable"

    return {
        "service": "portal-api",
        "version": "ops_v1",
        "now_utc": _iso_utc(now_ts),
        "window_s": 0,
        "health": health,
        "health_reason": health_reason,
        "summary": {
            "uptime_s": int(max(0.0, float(time.monotonic() - _API_STARTED_MONO))),
            "live_active_sessions": int(live_active_sessions),
            "live_max_sessions": int(live_max_sessions),
            "live_ws_connected": int(live_ws_connected),
            "upload_prep_inbox": int(upload_prep.get("inbox") or 0),
            "dependencies_up": int(dependencies_up),
            "dependencies_total": int(dependencies_total),
        },
        "details": {
            "live_sessions": live_metrics,
            "upload_queues": {
                "upload_prep": upload_prep,
                "upload_worker": upload_worker,
            },
            "upload_coordinator": coordinator,
            "dependencies": dependencies,
        },
    }


def _api_ops_html(*, metrics_url: str) -> str:
    safe_metrics_url = str(metrics_url or "").strip() or "./api/metrics"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Portal API Operations</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --card: #fff;
      --text: #161616;
      --muted: #5f6470;
      --line: #d8dde8;
      --ok: #157f3b;
      --warn: #9a6b00;
      --error: #b42318;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif;
    }}
    html, body {{ height: 100%; background: var(--bg); overflow: hidden; }}
    body {{ margin: 0; color: var(--text); }}
    main {{ max-width: 980px; margin: 0 auto; padding: 16px; height: 100%; box-sizing: border-box; display: flex; flex-direction: column; }}
    h1 {{ margin: 0 0 10px 0; font-size: 22px; }}
    #meta {{ margin: 0 0 12px 0; color: var(--muted); font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }}
    .card .k {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .card .v {{ font-size: 18px; font-weight: 700; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 700; color: #fff; }}
    .badge.ok {{ background: var(--ok); }}
    .badge.warn {{ background: var(--warn); }}
    .badge.error {{ background: var(--error); }}
    pre {{ background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 12px; overflow: auto; margin: 0; flex: 1; min-height: 0; }}
    @media (max-width: 940px) {{ .cards {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Portal API Operations</h1>
    <div id="meta">Loading...</div>
    <div class="cards">
      <div class="card"><div class="k">Live Active / Max</div><div id="live_sessions" class="v">-</div></div>
      <div class="card"><div class="k">Live WS Connected</div><div id="live_ws" class="v">-</div></div>
      <div class="card"><div class="k">Upload Prep Inbox</div><div id="prep_inbox" class="v">-</div></div>
      <div class="card"><div class="k">Deps Up</div><div id="deps" class="v">-</div></div>
    </div>
    <pre id="raw">loading...</pre>
  </main>
  <script>
    const byId = (id) => document.getElementById(id);
    async function refresh() {{
      try {{
        const resp = await fetch('{safe_metrics_url}', {{ cache: 'no-store' }});
        const data = await resp.json();
        const s = data.summary || {{}};
        const h = String(data.health || 'warn').toLowerCase();
        const reason = String(data.health_reason || "");
        const reasonText = reason ? (" | reason: " + reason) : "";
        byId('meta').innerHTML = "Health: <span class='badge " + h + "'>" + h + "</span>" + reasonText + " | updated: " + new Date().toLocaleTimeString();
        byId('live_sessions').textContent = `${{s.live_active_sessions ?? '-'}} / ${{s.live_max_sessions ?? '-'}}`;
        byId('live_ws').textContent = `${{s.live_ws_connected ?? '-'}}`;
        byId('prep_inbox').textContent = `${{s.upload_prep_inbox ?? '-'}}`;
        byId('deps').textContent = `${{s.dependencies_up ?? '-'}} / ${{s.dependencies_total ?? '-'}}`;
        byId('raw').textContent = JSON.stringify(data, null, 2);
      }} catch (e) {{
        byId('meta').textContent = `Metrics unavailable: ${{String(e)}}`;
        byId('raw').textContent = `fetch error: ${{String(e)}}`;
      }}
    }}
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>"""


def _ops_launcher_html(*, api_ops_url: str, pool_ops_url: str, worker_batch_ops_url: str) -> str:
    safe_api = str(api_ops_url or "").strip()
    safe_pool = str(pool_ops_url or "").strip()
    safe_worker_batch = str(worker_batch_ops_url or "").strip()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Service Operations</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --card: #ffffff;
      --text: #171717;
      --muted: #5f6470;
      --line: #d9dde5;
      --accent: #0a66c2;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
    p {{ margin: 0 0 14px 0; color: var(--muted); }}
    .meta {{
      margin: 0 0 14px 0;
      font-size: 13px;
      color: var(--muted);
    }}
    .links {{
      display: flex;
      gap: 10px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}
    a.button {{
      display: inline-block;
      text-decoration: none;
      border: 1px solid #e2e6ed;
      border-radius: 999px;
      padding: 4px 9px;
      background: #f3f5f9;
      color: #5d6470;
      font-size: 12px;
      line-height: 1.2;
    }}
    a.button.primary {{
      border-color: #e2e6ed;
      color: #5d6470;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card);
      overflow: hidden;
      min-height: 0;
    }}
    .panel header {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      font-weight: 600;
      font-size: 14px;
    }}
    iframe {{
      width: 100%;
      height: 760px;
      border: 0;
      background: #f6f7f9;
      display: block;
    }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Service Operations</h1>
    <div id="meta" class="meta">Loading launcher metrics...</div>
    <div class="links">
      <a class="button primary" href="{safe_api}" target="_blank" rel="noopener">Open Portal API /ops</a>
      <a class="button primary" href="{safe_pool}" target="_blank" rel="noopener">Open ASR Pool /ops</a>
      <a class="button primary" href="{safe_worker_batch}" target="_blank" rel="noopener">Open ASR Worker Batch /ops</a>
    </div>
    <div class="grid">
      <section class="panel">
        <header>Portal API</header>
        <iframe src="{safe_api}" title="Portal API ops"></iframe>
      </section>
      <section class="panel">
        <header>ASR Pool</header>
        <iframe src="{safe_pool}" title="ASR Pool ops"></iframe>
      </section>
      <section class="panel">
        <header>ASR Worker Batch</header>
        <iframe src="{safe_worker_batch}" title="ASR Worker batch ops"></iframe>
      </section>
    </div>
  </main>
  <script>
    async function refreshLauncherMeta() {{
      const el = document.getElementById("meta");
      try {{
        const res = await fetch("./ops/metrics", {{ cache: "no-store" }});
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        const s = data.summary || {{}};
        const up = Number(s.services_up || 0);
        const total = Number(s.services_total || 0);
        const health = String(data.health || "unknown");
        el.textContent = "Launcher health: " + health + " | services up: " + up + "/" + total + " | refreshed: " + new Date().toLocaleTimeString();
      }} catch (err) {{
        el.textContent = "Launcher metrics unavailable: " + err;
      }}
    }}
    refreshLauncherMeta();
    setInterval(refreshLauncherMeta, 5000);
  </script>
</body>
</html>"""


def _ops_service_probe(*, url: str, timeout_s: float = 1.5) -> Dict[str, Any]:
    u = str(url or "").strip()
    if not u:
        return {"url": "", "ok": False, "status_code": None, "error": "missing_url", "elapsed_ms": None}
    started = datetime.now(timezone.utc).timestamp()
    try:
        with urlopen(u, timeout=max(0.2, float(timeout_s))) as resp:
            elapsed_ms = int((datetime.now(timezone.utc).timestamp() - started) * 1000.0)
            return {
                "url": u,
                "ok": int(getattr(resp, "status", 0) or 0) == 200,
                "status_code": int(getattr(resp, "status", 0) or 0),
                "error": None,
                "elapsed_ms": elapsed_ms,
            }
    except URLError as exc:
        elapsed_ms = int((datetime.now(timezone.utc).timestamp() - started) * 1000.0)
        return {
            "url": u,
            "ok": False,
            "status_code": None,
            "error": f"URLError: {exc.reason}",
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:
        elapsed_ms = int((datetime.now(timezone.utc).timestamp() - started) * 1000.0)
        return {
            "url": u,
            "ok": False,
            "status_code": None,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": elapsed_ms,
        }


@router.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@router.get("/ops", response_class=HTMLResponse)
def ops_launcher() -> HTMLResponse:
    api_ops_url = os.getenv("PORTAL_API_OPS_URL", "./ops/api")
    pool_ops_url = os.getenv("ASR_POOL_OPS_URL", "http://127.0.0.1:18090/ops")
    worker_batch_ops_url = os.getenv("ASR_WORKER_BATCH_OPS_URL", "http://127.0.0.1:18111/ops")
    return HTMLResponse(
        _ops_launcher_html(
            api_ops_url=api_ops_url,
            pool_ops_url=pool_ops_url,
            worker_batch_ops_url=worker_batch_ops_url,
        )
    )


@router.get("/ops/metrics")
def ops_launcher_metrics() -> Dict[str, Any]:
    now_utc = _iso_utc(datetime.now(timezone.utc).timestamp())
    api_metrics = _api_ops_metrics_snapshot()
    pool_metrics_url = os.getenv("ASR_POOL_OPS_METRICS_URL", "http://127.0.0.1:18090/ops/metrics")
    worker_batch_metrics_url = os.getenv("ASR_WORKER_BATCH_OPS_METRICS_URL", "http://127.0.0.1:18111/ops/metrics")
    pool = _ops_service_probe(url=pool_metrics_url)
    worker_batch = _ops_service_probe(url=worker_batch_metrics_url)
    api_service = {
        "url": "./ops/api/metrics",
        "ok": str(api_metrics.get("health") or "").strip().lower() != "error",
        "status_code": 200,
        "error": None,
        "elapsed_ms": 0,
        "health": str(api_metrics.get("health") or "unknown"),
    }
    services = {
        "portal_api": api_service,
        "asr_pool": pool,
        "asr_worker_batch": worker_batch,
    }
    services_total = len(services)
    services_up = sum(1 for v in services.values() if bool(v.get("ok")))
    health = "ok" if services_up == services_total else ("warn" if services_up > 0 else "error")
    return {
        "service": "portal-api-ops-launcher",
        "version": "ops_v1",
        "now_utc": now_utc,
        "window_s": 0,
        "health": health,
        "summary": {
            "services_total": services_total,
            "services_up": services_up,
        },
        "details": services,
    }


@router.get("/ops/api", response_class=HTMLResponse)
def api_ops_page() -> HTMLResponse:
    return HTMLResponse(_api_ops_html(metrics_url="./api/metrics"))


@router.get("/ops/api/metrics")
def api_ops_metrics() -> Dict[str, Any]:
    return _api_ops_metrics_snapshot()


@router.get("/demo/settings")
def get_demo_settings() -> Dict[str, Any]:
    config_dir = (_REPO_ROOT / "config").resolve()
    settings_path = (config_dir / "settings.json").resolve()
    local_path = (config_dir / "local.json").resolve()
    return {
        "generated_at_utc": _iso_utc(datetime.now(timezone.utc).timestamp()),
        "sources": [
            _file_config_source(source_id="settings_json", title="settings.json", path=settings_path),
            _file_config_source(source_id="local_json", title="local.json", path=local_path),
        ],
    }


@router.get("/ui/settings")
def get_ui_settings() -> Dict[str, Any]:
    settings = _load_ui_settings()
    return {
        "generated_at_utc": _iso_utc(datetime.now(timezone.utc).timestamp()),
        "version": str(settings.get("version") or "ui_settings_v1"),
        "settings": {
            "live": dict(settings.get("live") or {}),
        },
    }
