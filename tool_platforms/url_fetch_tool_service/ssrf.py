"""SSRF guard for the URL-fetch service.

Blocks loopback, RFC1918 / link-local / unique-local ranges, and known
cloud-metadata endpoints. Resolves DNS up front so a hostname that
points at an internal IP (DNS-rebinding) is rejected before we open
the connection.

Intentionally conservative: deny by default for any private-or-special
range. Operators who need to allow specific internal hosts can extend
``_EXTRA_ALLOWED_HOSTS`` via the ``EIREL_URL_FETCH_ALLOWED_HOSTS`` env
var.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

__all__ = ["UrlFetchSSRFError", "validate_url"]


class UrlFetchSSRFError(ValueError):
    """Raised when a URL fails the SSRF policy."""


# Cloud-metadata hostnames that resolve to private IPs but are also
# attacked by name (some clients short-circuit DNS for them).
_BLOCKED_HOSTS = frozenset({
    "metadata.google.internal",
    "metadata",
    "instance-data",
    "instance-data.ec2.internal",
})


def _extra_allowed_hosts() -> frozenset[str]:
    raw = os.getenv("EIREL_URL_FETCH_ALLOWED_HOSTS", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_url(url: str) -> tuple[str, str]:
    """Validate ``url`` against SSRF policy. Returns ``(scheme, host)``.

    Raises :class:`UrlFetchSSRFError` on policy violation. Resolves
    DNS to verify that hostnames don't point at private IPs; both A
    and AAAA records are checked.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise UrlFetchSSRFError(f"unsupported scheme {scheme!r}; only http/https are allowed")
    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlFetchSSRFError("URL has no hostname")
    if host in _BLOCKED_HOSTS:
        raise UrlFetchSSRFError(f"hostname {host!r} is on the SSRF blocklist")

    # If the host is already an IP literal, validate it directly.
    try:
        ip = ipaddress.ip_address(host)
        if _is_private_ip(ip):
            raise UrlFetchSSRFError(f"IP {host!r} is in a private/reserved range")
        return scheme, host
    except ValueError:
        pass

    if host in _extra_allowed_hosts():
        return scheme, host

    # Resolve DNS — block if any returned address is private.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UrlFetchSSRFError(f"DNS resolution failed for {host!r}: {exc}") from None
    seen_any = False
    for entry in infos:
        addr = entry[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        seen_any = True
        if _is_private_ip(ip):
            raise UrlFetchSSRFError(
                f"hostname {host!r} resolves to private/reserved IP {addr!r}"
            )
    if not seen_any:
        raise UrlFetchSSRFError(f"DNS resolution returned no usable addresses for {host!r}")
    return scheme, host
