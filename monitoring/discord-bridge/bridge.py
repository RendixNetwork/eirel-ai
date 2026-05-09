"""Translate Alertmanager webhook payloads to Discord webhook format.

Listens on POST /default and POST /critical. Routes to two distinct
Discord webhook URLs read from env so each severity tier can land in
its own channel. Stdlib only — no third-party deps in the image.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("discord-bridge")

DESTINATIONS = {
    "default": os.environ.get("ALERTMANAGER_DEFAULT_WEBHOOK_URL", "").strip(),
    "critical": os.environ.get("ALERTMANAGER_CRITICAL_WEBHOOK_URL", "").strip(),
}

# Embed colour per severity. Resolved alerts always render green.
SEVERITY_COLOR = {
    "critical": 0xE03131,
    "warning": 0xF59F00,
    "info": 0x4DABF7,
}
RESOLVED_COLOR = 0x37B24D
DEFAULT_COLOR = 0x868E96

PORT = int(os.environ.get("PORT", "8080"))
DISCORD_TIMEOUT = 10.0


def _format_embed(alert: dict[str, Any]) -> dict[str, Any]:
    labels = alert.get("labels") or {}
    annotations = alert.get("annotations") or {}
    alertname = labels.get("alertname", "alert")
    severity = (labels.get("severity") or "info").lower()
    status = (alert.get("status") or "firing").lower()

    summary = (annotations.get("summary") or "").strip()
    description = (annotations.get("description") or "").strip()
    body = "\n".join(s for s in (summary, description) if s) or "(no detail)"

    color = (
        RESOLVED_COLOR if status == "resolved"
        else SEVERITY_COLOR.get(severity, DEFAULT_COLOR)
    )

    fields = [
        {"name": k, "value": str(v)[:1024], "inline": True}
        for k, v in labels.items()
        if k not in {"alertname", "severity"}
    ][:10]

    title_prefix = "[RESOLVED]" if status == "resolved" else "[FIRING]"
    return {
        "title": f"{title_prefix} {alertname} ({severity})",
        "description": body[:4000],
        "color": color,
        "fields": fields,
    }


def _post_discord(url: str, embeds: list[dict[str, Any]]) -> tuple[int, str]:
    payload = json.dumps({"embeds": embeds}).encode("utf-8")
    # Discord routes webhooks through Cloudflare which 403s the default
    # ``Python-urllib/3.x`` UA (``error code: 1010``), so set an explicit
    # identifier.
    req = urlrequest.Request(
        url, data=payload, method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "eirel-alertmanager-discord-bridge/1.0",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=DISCORD_TIMEOUT) as resp:
            return resp.status, resp.read(2048).decode(errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read(2048).decode(errors="replace")
    except URLError as exc:
        return 0, f"network_error: {exc}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._reply(200, {"status": "ok"})
            return
        self._reply(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        channel = self.path.lstrip("/").split("?", 1)[0]
        url = DESTINATIONS.get(channel)
        if not url:
            self._reply(404, {"error": f"unknown_channel:{channel}"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            self._reply(400, {"error": "empty_body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._reply(400, {"error": f"invalid_json:{exc}"})
            return
        alerts = payload.get("alerts") or []
        if not alerts:
            self._reply(202, {"status": "skipped_empty"})
            return
        embeds = [_format_embed(a) for a in alerts][:10]
        status, body = _post_discord(url, embeds)
        log.info("relayed channel=%s alerts=%d discord_status=%d", channel, len(alerts), status)
        if 200 <= status < 300:
            self._reply(200, {"status": "sent", "discord_status": status, "alerts": len(alerts)})
        else:
            self._reply(502, {"status": "discord_error", "discord_status": status, "body": body[:512]})

    def _reply(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    missing = [k for k, v in DESTINATIONS.items() if not v]
    if missing:
        log.warning("destinations not configured: %s — those channels will 404", missing)
    log.info("listening port=%d destinations=%s", PORT, sorted(k for k, v in DESTINATIONS.items() if v))
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
