"""Runtime SSRF guard for the MCP relay.

Belt-and-suspenders: an integration's ``base_url`` is also checked at
admin-registration time. The runtime check ensures that even if a
disabled integration row gets reactivated with a tampered base_url, or
DNS rebinds an originally-public host to a private IP, the relay
refuses to call.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

__all__ = ["MCPSSRFError", "validate_base_url"]


class MCPSSRFError(ValueError):
    """Raised when an integration's base_url fails SSRF policy."""


_BLOCKED_HOSTS = frozenset({
    "metadata.google.internal",
    "metadata",
    "instance-data",
    "instance-data.ec2.internal",
})


def _is_private(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_base_url(url: str, *, allow_http: bool = False) -> tuple[str, str]:
    """Validate ``url`` against SSRF policy. Returns ``(scheme, host)``.

    Production (``allow_http=False``) requires HTTPS and resolves DNS to
    catch hostnames that point at private IPs (DNS rebinding). Dev /
    test (``allow_http=True``) skips the DNS step so tests can use
    fixture hostnames that don't resolve. The literal-IP check runs in
    both modes — that's the primary defense against
    ``169.254.169.254``-style attacks.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("https",) and not (allow_http and scheme == "http"):
        raise MCPSSRFError(
            f"unsupported scheme {scheme!r}; production allows https only"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise MCPSSRFError("base_url has no hostname")
    if host in _BLOCKED_HOSTS:
        raise MCPSSRFError(f"hostname {host!r} is on the SSRF blocklist")
    parsed_ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        parsed_ip = None
    if parsed_ip is not None:
        if _is_private(parsed_ip):
            raise MCPSSRFError(
                f"IP {host!r} is in a private/reserved range"
            )
        return scheme, host
    if allow_http:
        # Dev / test mode — skip the DNS step. The integration's actual
        # endpoint is mocked in tests via httpx.MockTransport, so DNS
        # would only fail against fixture hostnames that don't exist.
        return scheme, host
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise MCPSSRFError(f"DNS resolution failed for {host!r}: {exc}") from exc
    for entry in infos:
        sockaddr = entry[4]
        if not sockaddr:
            continue
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if _is_private(ip):
            raise MCPSSRFError(
                f"hostname {host!r} resolves to private IP {ip!r}"
            )
    return scheme, host
