"""Translate Alertmanager webhook payloads to Discord webhook format.

Listens on POST /default and POST /critical. Routes to two distinct
Discord webhook URLs read from env so each severity tier can land in
its own channel. Stdlib only — no third-party deps in the image.

Also exposes ``GET /metrics`` in Prometheus text format so the bridge
itself is self-monitored: alertmanager being broken silently is what
burned us 2 days ago, and the same risk applies to this relay if it
ever 5xx's en masse without anyone noticing.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
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


# -- Metrics -----------------------------------------------------------------
#
# Hand-rolled minimal registry — keeps the image stdlib-only. Each metric
# is a dict keyed by a sorted tuple of (label, value) pairs so a single
# metric name can carry multiple label combinations without per-combo
# declarations. Render into Prometheus text format on ``GET /metrics``.

_metrics_lock = threading.Lock()


class _Counter:
    """Monotonic counter with optional labels. Thread-safe via the
    module-level ``_metrics_lock``.
    """

    __slots__ = ("name", "help", "_samples")

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self._samples: dict[tuple[tuple[str, str], ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with _metrics_lock:
            self._samples[key] = self._samples.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with _metrics_lock:
            samples = list(self._samples.items())
        if not samples:
            lines.append(f"{self.name} 0")
            return lines
        for key, value in samples:
            if key:
                lbl = ",".join(f'{k}="{_esc(v)}"' for k, v in key)
                lines.append(f"{self.name}{{{lbl}}} {value:g}")
            else:
                lines.append(f"{self.name} {value:g}")
        return lines


class _Gauge:
    """Set-only gauge (no inc/dec needed for this use case)."""

    __slots__ = ("name", "help", "_samples")

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self._samples: dict[tuple[tuple[str, str], ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with _metrics_lock:
            self._samples[key] = float(value)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with _metrics_lock:
            samples = list(self._samples.items())
        if not samples:
            lines.append(f"{self.name} 0")
            return lines
        for key, value in samples:
            if key:
                lbl = ",".join(f'{k}="{_esc(v)}"' for k, v in key)
                lines.append(f"{self.name}{{{lbl}}} {value:g}")
            else:
                lines.append(f"{self.name} {value:g}")
        return lines


def _esc(value: str) -> str:
    """Escape a label value for Prometheus text format."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# Counters: per-channel webhook flow accounting.
ALERTS_RECEIVED = _Counter(
    "discord_bridge_alerts_received_total",
    "Alert objects extracted from incoming alertmanager POSTs, per channel.",
)
ALERTS_SKIPPED_EMPTY = _Counter(
    "discord_bridge_payloads_skipped_empty_total",
    "Incoming POSTs whose ``alerts`` array was empty, per channel.",
)
RELAY_ATTEMPTS = _Counter(
    "discord_bridge_relay_attempts_total",
    "Discord POST attempts, labelled by channel + outcome "
    "(ok / http_4xx / http_5xx / network_error).",
)
RELAY_LATENCY_SUM = _Counter(
    "discord_bridge_relay_latency_seconds_sum",
    "Sum of Discord POST wall-clock durations, per channel + outcome.",
)
RELAY_LATENCY_COUNT = _Counter(
    "discord_bridge_relay_latency_seconds_count",
    "Count of Discord POST attempts contributing to the latency sum.",
)
DESTINATION_CONFIGURED = _Gauge(
    "discord_bridge_destination_configured",
    "1 if the channel has a webhook URL configured, 0 otherwise.",
)

_REGISTRY: list[Any] = [
    ALERTS_RECEIVED, ALERTS_SKIPPED_EMPTY, RELAY_ATTEMPTS,
    RELAY_LATENCY_SUM, RELAY_LATENCY_COUNT, DESTINATION_CONFIGURED,
]


def _classify_outcome(status: int) -> str:
    if status == 0:
        return "network_error"
    if 200 <= status < 300:
        return "ok"
    if 400 <= status < 500:
        return "http_4xx"
    if 500 <= status < 600:
        return "http_5xx"
    return "other"


# -- Embed rendering ---------------------------------------------------------


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


# -- HTTP handler ------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._reply(200, {"status": "ok"})
            return
        if self.path == "/metrics":
            self._reply_metrics()
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
            ALERTS_SKIPPED_EMPTY.inc(channel=channel)
            self._reply(202, {"status": "skipped_empty"})
            return
        ALERTS_RECEIVED.inc(amount=len(alerts), channel=channel)
        embeds = [_format_embed(a) for a in alerts][:10]
        start = time.monotonic()
        status, body = _post_discord(url, embeds)
        elapsed = time.monotonic() - start
        outcome = _classify_outcome(status)
        RELAY_ATTEMPTS.inc(channel=channel, outcome=outcome)
        RELAY_LATENCY_SUM.inc(amount=elapsed, channel=channel, outcome=outcome)
        RELAY_LATENCY_COUNT.inc(channel=channel, outcome=outcome)
        log.info(
            "relayed channel=%s alerts=%d discord_status=%d outcome=%s elapsed=%.3fs",
            channel, len(alerts), status, outcome, elapsed,
        )
        if outcome == "ok":
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

    def _reply_metrics(self) -> None:
        lines: list[str] = []
        for metric in _REGISTRY:
            lines.extend(metric.render())
        body = ("\n".join(lines) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    missing = [k for k, v in DESTINATIONS.items() if not v]
    if missing:
        log.warning("destinations not configured: %s — those channels will 404", missing)
    # Seed the configured gauge so /metrics has a row for each known
    # channel from the first request onward, before any alert lands.
    for channel, url in DESTINATIONS.items():
        DESTINATION_CONFIGURED.set(1.0 if url else 0.0, channel=channel)
    log.info("listening port=%d destinations=%s", PORT, sorted(k for k, v in DESTINATIONS.items() if v))
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
