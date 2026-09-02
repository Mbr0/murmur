#!/usr/bin/env python3
"""HTTP hardening shared by everything that sends a Murmur credential.

Three modules hand a bearer token or an API key to a remote host —
:mod:`engines.cloud`, :mod:`engines.byok` and
:mod:`services.license_service` — and all three need the same two guarantees,
neither of which ``urllib.request.urlopen`` provides:

* **The credential only ever leaves over TLS.** :func:`require_https_base_url`
  refuses anything but ``https://``. Plain ``http://`` is accepted for the
  loopback host alone, which is what the loopback servers in ``tests/`` bind
  to; nothing else can reach it.
* **A redirect never carries the credential somewhere else.** ``urlopen``
  follows 30x responses and re-sends every header the caller set, so a
  compromised — or merely misconfigured — proxy can bounce a request to a host
  of its choosing and be handed the lease.
  :func:`open_no_cross_host_redirect` runs the request through an opener whose
  redirect handler refuses any hop that changes scheme, host or port, and
  strips the credential headers even from a same-origin hop, where the new path
  may well be a different service.

The refusal is an :class:`~engines.base.EngineError` rather than a silent
"follow it anyway": a proxy that redirects is misbehaving, and the caller must
see that rather than discover it in someone else's access log.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from engines.base import EngineError

#: Headers dropped before any redirect is followed. Compared lower-cased.
CREDENTIAL_HEADERS = frozenset(
    {"authorization", "x-api-key", "proxy-authorization", "cookie"}
)

#: The only hosts allowed to speak plain ``http``: the test servers' loopback.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})

#: Substring every refused redirect carries, so callers can assert on it.
REDIRECT_REFUSED = "redirect refused"

_DEFAULT_PORTS = {"http": 80, "https": 443}


def require_https_base_url(base_url: Any, *, what: str = "base_url") -> str:
    """Return ``base_url`` without its trailing slash, or raise.

    Only ``https://`` is accepted, because every caller is about to attach a
    bearer credential. ``http://`` passes for :data:`LOOPBACK_HOSTS` alone —
    the loopback servers the tests stand up — and for nothing else.

    Raises :class:`ValueError` early, at construction time, so a mistyped
    endpoint is a startup failure rather than a leaked token.
    """
    cleaned = (base_url or "").strip().rstrip("/") if isinstance(base_url, str) else ""
    if not cleaned:
        raise ValueError(f"{what} is required and must be an absolute https:// URL")

    parsed = urllib.parse.urlsplit(cleaned)
    try:
        host = (parsed.hostname or "").lower()
    except ValueError:  # a malformed netloc, e.g. a bad port
        host = ""

    if host:
        if parsed.scheme == "https":
            return cleaned
        if parsed.scheme == "http" and host in LOOPBACK_HOSTS:
            return cleaned

    raise ValueError(
        f"{what} must be an absolute https:// URL "
        f"(plain http:// is allowed for loopback only), got {base_url!r}"
    )


def _origin(url: str) -> tuple[str, str, int | None]:
    """``(scheme, host, port)`` with the scheme's default port filled in."""
    parsed = urllib.parse.urlsplit(url)
    scheme = (parsed.scheme or "").lower()
    try:
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return (scheme, "", None)
    return (scheme, host, port or _DEFAULT_PORTS.get(scheme))


def _strip_credentials(request: urllib.request.Request) -> urllib.request.Request:
    """Remove every credential header from ``request``, in place."""
    for name in list(request.headers) + list(request.unredirected_hdrs):
        if name.lower() in CREDENTIAL_HEADERS:
            request.remove_header(name)
    return request


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuses a redirect off the origin, and de-credentials the rest."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urljoin(req.full_url, newurl)
        if _origin(target) != _origin(req.full_url):
            raise EngineError(
                f"{REDIRECT_REFUSED}: HTTP {code} pointed at a different host"
            )
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is None:
            return None
        return _strip_credentials(new_request)


_opener: urllib.request.OpenerDirector | None = None


def safe_urlopen(request: urllib.request.Request, timeout: float | None = None):
    """``urlopen`` that refuses a cross-origin redirect. Same return value."""
    global _opener
    if _opener is None:
        # A benign race here only builds the opener twice; both are equivalent.
        _opener = urllib.request.build_opener(SafeRedirectHandler())
    return _opener.open(request, timeout=timeout)


def open_no_cross_host_redirect(
    request: urllib.request.Request,
    timeout: float | None = None,
    http_open: Callable | None = None,
) -> Any:
    """One round trip through the hardened opener.

    ``http_open`` is the engines' injection seam. The stdlib ``urlopen`` (and
    an absent opener) is replaced by :func:`safe_urlopen`; anything else is a
    test double and is called verbatim, since it is not an HTTP stack and
    cannot redirect anywhere.
    """
    opener = http_open
    if opener is None or opener is urllib.request.urlopen:
        opener = safe_urlopen
    return opener(request, timeout=timeout)
