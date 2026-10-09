"""Schutz des oeffentlichen Pfads: Redaktion von Geheimnissen und Pfaden, Security-Header, Body-Limit, Rate-Limit je IP.

Reine ASGI-Middleware (kein BaseHTTPMiddleware: das puffert und stoert SSE). Alle Grenzen kommen aus Umgebungsvariablen und werden
bei jedem Aufruf gelesen (Tests setzen sie per monkeypatch).
"""

from __future__ import annotations

import getpass
import json
import os
import re
import time
from collections import deque
from pathlib import Path

from starlette.requests import Request

SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PROJECT_ID")
SECRET_PATTERNS = [
    re.compile(r"tvly-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
]
KEY_VALUE = re.compile(r"""(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password)(["']?\s*[:=]\s*["']?)[^\s"',;)}\]]{8,}""")
HOME_PATH = re.compile(r"(?:/home|/Users)/[^/\s\"']+")


def _secret_values() -> list[str]:
    vals = [v for k, v in os.environ.items() if v and len(v) >= 8 and any(h in k.upper() for h in SECRET_ENV_HINTS)]
    return sorted(set(vals), key=len, reverse=True)


def redact_text(text: str, repo_root: str | None = None) -> str:
    """Keine Geheimnisse und keine Pfade mit Nutzernamen nach aussen. Repo-Wurzel -> ".", Home -> "~", /home/<x> -> "~"."""
    out = text
    for v in _secret_values():
        out = out.replace(v, "[redacted]")
    for pat in SECRET_PATTERNS:
        out = pat.sub("[redacted]", out)
    out = KEY_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", out)
    if repo_root:
        out = out.replace(repo_root, ".")
    home = str(Path.home())
    if len(home) > 1:
        out = out.replace(home, "~")
    out = HOME_PATH.sub("~", out)
    try:
        user = getpass.getuser()
        if len(user) > 2:
            out = re.sub(rf"(?<=[/\\]){re.escape(user)}(?=[/\\\s\"']|$)", "user", out)   # nur als Pfadteil, nie als Wortbestandteil
    except Exception:   # noqa: BLE001
        pass
    return out


TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".svg", ".txt"}


def redact_tree(root: Path, repo_root: str | None = None) -> int:
    """Alle Textdateien unter root redigieren (Export eines Live-Laufs). Anzahl geaenderter Dateien."""
    changed = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in TEXT_SUFFIXES:
            try:
                old = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            new = redact_text(old, repo_root)
            if new != old:
                p.write_text(new)
                changed += 1
    return changed


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"


def security_headers(csp: str) -> list[tuple[bytes, bytes]]:
    return [
        (b"content-security-policy", csp.encode()),
        (b"x-content-type-options", b"nosniff"),
        (b"referrer-policy", b"no-referrer"),
        (b"x-frame-options", b"DENY"),
        (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
        (b"cross-origin-opener-policy", b"same-origin"),
    ]


HITS: dict[str, deque] = {}   # Rate-Limit-Zaehler je (Methode, IP); Tests leeren ihn


class GuardMiddleware:
    """Je Anfrage: Rate-Limit je IP (nur /api/), Body-Limit, danach Security-Header an jede Antwort.

    `app.state.page_csp` (gesetzt von deploy/serve.py) gilt fuer HTML-Antworten, alles andere bekommt die strenge API-Policy.
    """

    def __init__(self, app, state=None) -> None:
        self.app = app
        self.state = state

    # ---- Rate-Limit: gleitendes Fenster 60 s, je IP; POST strenger als GET ----
    def _limited(self, ip: str, method: str) -> bool:
        now = time.monotonic()
        if len(HITS) > 5000:   # Speicher begrenzen: alte Schluessel raus
            for k in [k for k, d in HITS.items() if not d or now - d[-1] >= 60]:
                del HITS[k]
        key = f"{method == 'POST'}|{ip}"
        d = HITS.setdefault(key, deque())
        while d and now - d[0] > 60:
            d.popleft()
        cap = int(_f("RATE_LIMIT_POST_PER_MIN", 10)) if method == "POST" else int(_f("RATE_LIMIT_PER_MIN", 240))
        if cap > 0 and len(d) >= cap:
            return True
        d.append(now)
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path, method = scope["path"], scope["method"]

        async def reject(status: int, reason: str, message: str, extra=()):
            body = json.dumps({"reason": reason, "message": message, "detail": message}).encode()
            hdrs = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *extra, *security_headers(API_CSP)]
            await send({"type": "http.response.start", "status": status, "headers": hdrs})
            await send({"type": "http.response.body", "body": body})

        if path.startswith("/api/"):
            from .live_gate import client_ip

            ip = client_ip(Request(scope))
            if self._limited(ip, method):
                return await reject(429, "rate_limited", "Too many requests. Slow down and try again in a minute.", [(b"retry-after", b"60")])
        max_body = int(_f("MAX_BODY_BYTES", 16384))
        if method in ("POST", "PUT", "PATCH"):
            cl = dict(scope["headers"]).get(b"content-length")
            if cl and cl.isdigit() and int(cl) > max_body:
                return await reject(413, "too_large", "Request body too large.")
            seen = 0
            original_receive = receive

            async def counting_receive():
                nonlocal seen
                msg = await original_receive()
                if msg["type"] == "http.request":
                    seen += len(msg.get("body", b""))
                    if seen > max_body:
                        return {"type": "http.request", "body": b"", "more_body": False}
                return msg

            receive = counting_receive

        page_csp = getattr(self.state, "page_csp", None) if self.state is not None else None

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {k.lower() for k, _ in headers}
                is_html = any(k.lower() == b"content-type" and v.startswith(b"text/html") for k, v in headers)
                csp = page_csp if (is_html and page_csp) else API_CSP
                headers += [(k, v) for k, v in security_headers(csp) if k not in names]
                if path.startswith("/api/") and b"cache-control" not in names:
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
