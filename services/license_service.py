#!/usr/bin/env python3
"""Boske lease verification, entitlements, device linking and the Pro gate.

**The Pro gate is one function.** Everything in the app that wants to know
whether a paid feature may run calls :func:`is_pro_feature_enabled` with a
name from :data:`PRO_FEATURES`. No window, menu, engine or service reads
:class:`Entitlements`, a :class:`Lease` or a license file directly; the one
writer of the gate is :func:`set_current_entitlements`, which
:class:`LicenseService` keeps in sync. Adding a second entitlement check
anywhere else is a bug, not a shortcut.

Lease token contract (Murmur and Boske must agree; see decision D6)
-------------------------------------------------------------------
A lease is a compact JWS: ``base64url(header).base64url(payload).signature``.

* header — ``{"alg": "EdDSA", "typ": "JWT"}``. Ed25519 only. ``HS256`` and
  ``none`` are rejected, so a stolen payload cannot be re-signed with a
  symmetric key or stripped of its signature.
* payload claims

  ===============  ==========================================================
  ``iss``          issuer, must equal :data:`LEASE_ISSUER`
  ``aud``          audience, must be (or contain) :data:`LEASE_AUDIENCE`
  ``sub``          Boske account id
  ``dev``          device id this lease was issued to; checked against this
                   Mac's own id, so a lease copied off another machine's
                   Keychain grants nothing here
  ``iat``          issued-at, seconds since the epoch
  ``exp``          expiry, seconds since the epoch; required
  ``ent``          ``{"pro": bool, "cloud_voice": bool, "msm_minutes": int}``
  ``plan``         optional plan name, for display only
  ===============  ==========================================================

``msm_minutes`` is the monthly cloud-voice minute allowance. ``ent`` is
mandatory and fully typed: a missing or mistyped flag is an error, never a
silently-false default.

Storage and logging rules
-------------------------
* Lease tokens live in the Keychain through the :class:`SecretStore`
  protocol (Wave 3 supplies the Keychain-backed implementation in
  ``services/keychain.py``). A lease is **never** written to the JSON config
  and never sits next to ``DEFAULT_CONFIG``. The two item names are
  :data:`LEASE_SECRET_NAME` and :data:`DEVICE_ID_SECRET_NAME`; they must match
  the Keychain store's own constants.
* No token ever reaches a log record or an exception message. Failures are
  reported by shape ("lease signature is not valid", "status=401"), never by
  content.

Transport
---------
Every Boske endpoint is reached over ``https`` — the lease travels as a bearer
token — and the URL the user is sent to approve a device must be ``https`` too.
Both are checked at parse time rather than trusted; see
:func:`engines._http.require_https_base_url`.

Release
-------
:data:`BOSKE_PUBLIC_KEY_PEM` ships as a placeholder. ``scripts/release.sh``
must substitute the real Boske Ed25519 public key before signing the bundle;
verification against the placeholder raises rather than passing anything.
``MURMUR_BOSKE_PUBLIC_KEY_PEM`` overrides the embedded key for staging.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from engines._http import require_https_base_url

#: Issuer every genuine lease carries.
LEASE_ISSUER = "boske"

#: Audience every genuine lease carries; also the device-link scope.
LEASE_AUDIENCE = "boske-llm-proxy"

#: Scope Murmur requests when linking a device.
LINK_SCOPE = LEASE_AUDIENCE

#: The only signature algorithm Murmur accepts.
LEASE_ALGORITHM = "EdDSA"

#: OAuth client id Murmur presents to Boske.
CLIENT_ID = "murmur"

#: RFC 8628 grant type for the device-authorization flow.
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

#: Boske endpoints. Paths are provisional until confirmed against Boske (D6).
DEVICE_CODE_PATH = "/v1/device/code"
DEVICE_TOKEN_PATH = "/v1/device/token"
DEVICE_REFRESH_PATH = "/v1/device/refresh"

#: Accepted clock difference between Murmur and Boske, in seconds.
MAX_CLOCK_SKEW_S = 300.0

#: Pro survives this many days past ``exp`` so a failed renewal is not a wall.
DEFAULT_GRACE_DAYS = 7

SECONDS_PER_DAY = 86400.0

#: How long :meth:`LicenseService.current_entitlements` reuses a verification.
ENTITLEMENTS_CACHE_TTL_S = 60.0

#: Refresh once this fraction of the lease lifetime has elapsed.
REFRESH_AT_LIFETIME_FRACTION = 0.8

#: RFC 8628: each ``slow_down`` adds five seconds to the poll interval.
SLOW_DOWN_INCREMENT_S = 5.0

DEFAULT_POLL_INTERVAL_S = 5.0

#: Secret-store key under which the lease token is kept. Must equal the
#: Keychain store's ``ITEM_LEASE``.
LEASE_SECRET_NAME = "boske-lease"

#: Secret-store key under which this Mac's device id is kept. Minted once, on
#: first use, and then read back forever; the Keychain store backs it.
DEVICE_ID_SECRET_NAME = "device-id"

#: Environment override, for staging builds pointed at a test issuer.
PUBLIC_KEY_ENV_VAR = "MURMUR_BOSKE_PUBLIC_KEY_PEM"

#: Deliberately unparseable. ``release.sh`` replaces it with the real key.
PLACEHOLDER_PUBLIC_KEY_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MURMUR_RELEASE_MUST_REPLACE_THIS_PLACEHOLDER\n"
    "-----END PUBLIC KEY-----\n"
)

#: The embedded Boske signing key. Placeholder in source; real in a release.
BOSKE_PUBLIC_KEY_PEM = PLACEHOLDER_PUBLIC_KEY_PEM

_PLACEHOLDER_MARKER = "MURMUR_RELEASE_MUST_REPLACE_THIS_PLACEHOLDER"

_logger = logging.getLogger(__name__)

#: ``(method, url, data, headers) -> (status, json)``. Injected so the whole
#: module is testable without a network and without a real HTTP stack.
HttpTransport = Callable[[str, str, dict, dict], tuple[int, dict]]


class LicenseError(Exception):
    """Base class for every error raised by this module."""


class LeaseError(LicenseError):
    """A lease token is absent, malformed, unsigned, or not addressed to us."""


class LeaseExpired(LeaseError):
    """The lease is authentic but past ``exp``.

    Carries the verified :class:`Lease` so the caller can still evaluate the
    grace window instead of dropping the user to the free tier at midnight.
    """

    def __init__(self, message: str, lease: "Lease"):
        super().__init__(message)
        self.lease = lease


class LinkError(LicenseError):
    """The device-authorization flow failed."""


class LinkExpired(LinkError):
    """The user code expired before anyone approved it."""


class LinkDenied(LinkError):
    """The user declined the link request."""


@dataclass(frozen=True)
class Lease:
    """A verified Boske lease. Holds no token, so it is safe to log."""

    issuer: str
    audience: str
    account_id: str
    device_id: str
    issued_at: float
    expires_at: float
    pro: bool
    cloud_voice: bool
    msm_minutes: int
    plan: str | None = None

    @property
    def lifetime_s(self) -> float:
        return max(0.0, self.expires_at - self.issued_at)

    @property
    def refresh_due_at(self) -> float:
        """When the lease has burned :data:`REFRESH_AT_LIFETIME_FRACTION`."""
        return self.issued_at + REFRESH_AT_LIFETIME_FRACTION * self.lifetime_s

    def is_expired(self, now: float) -> bool:
        """Expired at ``exp`` itself, so cloud stops the moment it lapses."""
        return now >= self.expires_at

    def grace_ends_at(self, grace_days: int = DEFAULT_GRACE_DAYS) -> float:
        return self.expires_at + grace_days * SECONDS_PER_DAY


@dataclass(frozen=True)
class Entitlements:
    """What the current license permits. The only input to the Pro gate."""

    pro: bool
    cloud_voice: bool
    msm_minutes: int
    expires_at: float | None
    in_grace: bool
    source: str  # "lease" | "none"

    @classmethod
    def none(cls) -> "Entitlements":
        """No license at all: free tier."""
        return cls(
            pro=False,
            cloud_voice=False,
            msm_minutes=0,
            expires_at=None,
            in_grace=False,
            source="none",
        )


@dataclass
class LinkSession:
    """An in-flight RFC 8628 device-authorization request.

    Mutable because the server drives the cadence: ``slow_down`` widens
    ``interval_s`` and every poll pushes ``next_poll_at`` forward.
    ``device_code`` is a bearer-grade secret and is kept out of ``repr``.
    """

    device_code: str = field(repr=False)
    user_code: str
    verification_url: str
    interval_s: float
    expires_at: float
    next_poll_at: float = 0.0


# --------------------------------------------------------------------------
# Lease verification
# --------------------------------------------------------------------------


def resolve_public_key_pem(explicit: str | None = None) -> str:
    """Pick the verification key: explicit, then env override, then embedded.

    Raises :class:`LeaseError` when the result is the source placeholder, so
    an unsigned developer build fails loudly instead of trusting anything.
    """
    pem = explicit or os.environ.get(PUBLIC_KEY_ENV_VAR) or BOSKE_PUBLIC_KEY_PEM
    if not pem or not pem.strip() or _PLACEHOLDER_MARKER in pem:
        raise LeaseError("no Boske public key embedded")
    return pem


def _load_ed25519_public_key(pem: str) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(pem.encode("utf-8"))
    except (ValueError, TypeError) as error:
        raise LeaseError(f"Boske public key is unreadable: {type(error).__name__}") from None
    if not isinstance(key, Ed25519PublicKey):
        raise LeaseError("Boske public key is not Ed25519")
    return key


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment + padding)
    except (binascii.Error, ValueError):
        raise LeaseError("malformed lease token") from None


def _decode_json_segment(segment: str, what: str) -> dict:
    try:
        value = json.loads(_b64url_decode(segment))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise LeaseError(f"lease {what} is not valid JSON") from None
    if not isinstance(value, dict):
        raise LeaseError(f"lease {what} is not an object")
    return value


def _require_str(claims: dict, name: str) -> str:
    value = claims.get(name)
    if not isinstance(value, str) or not value:
        raise LeaseError(f"lease claim {name!r} is missing or not a string")
    return value


def _require_number(claims: dict, name: str) -> float:
    value = claims.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeaseError(f"lease claim {name!r} is missing or not a number")
    return float(value)


def _entitlement_claims(claims: dict) -> tuple[bool, bool, int, str | None]:
    ent = claims.get("ent")
    if not isinstance(ent, dict):
        raise LeaseError("lease claim 'ent' is missing or not an object")
    pro = ent.get("pro")
    cloud_voice = ent.get("cloud_voice")
    msm_minutes = ent.get("msm_minutes")
    if not isinstance(pro, bool) or not isinstance(cloud_voice, bool):
        raise LeaseError("lease entitlement flags are missing or not booleans")
    if isinstance(msm_minutes, bool) or not isinstance(msm_minutes, int):
        raise LeaseError("lease entitlement 'msm_minutes' is missing or not an integer")
    plan = claims.get("plan")
    if plan is not None and not isinstance(plan, str):
        raise LeaseError("lease claim 'plan' is not a string")
    return pro, cloud_voice, msm_minutes, plan


def verify_lease(
    token: str,
    *,
    public_key_pem: str | None = None,
    now: float | None = None,
    device_id: str | None = None,
) -> Lease:
    """Verify a lease token offline and return the claims it carries.

    Raises :class:`LeaseError` for anything inauthentic or not addressed to
    this app, and :class:`LeaseExpired` (which subclasses it, and carries the
    otherwise-valid lease) when only ``exp`` has passed.

    ``device_id`` is this Mac's own id. When given, the ``dev`` claim must
    match it: a genuine, unexpired lease lifted from another machine's
    Keychain is signed by Boske and would otherwise verify perfectly here.
    The check outranks expiry, because a lease for someone else's device must
    not even reach the grace window. Omitting ``device_id`` skips the check,
    for callers that have no store to read an id from;
    :class:`LicenseService` always supplies one.
    """
    key = _load_ed25519_public_key(resolve_public_key_pem(public_key_pem))
    now = time.time() if now is None else now

    if not isinstance(token, str) or token.count(".") != 2:
        raise LeaseError("malformed lease token")
    header_b64, payload_b64, signature_b64 = token.split(".")
    if not header_b64 or not payload_b64 or not signature_b64:
        raise LeaseError("malformed lease token")

    header = _decode_json_segment(header_b64, "header")
    if header.get("alg") != LEASE_ALGORITHM:
        raise LeaseError(f"lease algorithm is not {LEASE_ALGORITHM}")

    try:
        key.verify(_b64url_decode(signature_b64), f"{header_b64}.{payload_b64}".encode("ascii"))
    except InvalidSignature:
        raise LeaseError("lease signature is not valid") from None

    claims = _decode_json_segment(payload_b64, "payload")

    issuer = _require_str(claims, "iss")
    if issuer != LEASE_ISSUER:
        raise LeaseError(f"lease issuer is not {LEASE_ISSUER!r}")

    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if LEASE_AUDIENCE not in audiences:
        raise LeaseError(f"lease audience is not {LEASE_AUDIENCE!r}")

    account_id = _require_str(claims, "sub")
    claimed_device = _require_str(claims, "dev")
    if device_id is not None and claimed_device != device_id:
        raise LeaseError("lease issued to another device")
    issued_at = _require_number(claims, "iat")
    if issued_at > now + MAX_CLOCK_SKEW_S:
        raise LeaseError("lease was issued in the future")
    if "exp" not in claims:
        raise LeaseError("lease has no 'exp' claim")
    expires_at = _require_number(claims, "exp")
    pro, cloud_voice, msm_minutes, plan = _entitlement_claims(claims)

    lease = Lease(
        issuer=issuer,
        audience=LEASE_AUDIENCE,
        account_id=account_id,
        device_id=claimed_device,
        issued_at=issued_at,
        expires_at=expires_at,
        pro=pro,
        cloud_voice=cloud_voice,
        msm_minutes=msm_minutes,
        plan=plan,
    )
    if lease.is_expired(now):
        raise LeaseExpired("lease has expired", lease)
    return lease


def entitlements_from_lease(
    lease: Lease,
    now: float,
    grace_days: int = DEFAULT_GRACE_DAYS,
) -> Entitlements:
    """Turn a verified lease into entitlements, applying the grace window.

    Pro survives ``grace_days`` past ``exp`` (flagged ``in_grace`` so the UI
    can nag); cloud voice and its minute allowance stop at ``exp`` itself,
    because every cloud minute costs us money.
    """
    expired = lease.is_expired(now)
    within_grace = now < lease.grace_ends_at(grace_days)
    pro = lease.pro and within_grace
    return Entitlements(
        pro=pro,
        cloud_voice=lease.cloud_voice and not expired,
        msm_minutes=lease.msm_minutes if not expired else 0,
        expires_at=lease.expires_at,
        in_grace=pro and expired,
        source="lease",
    )


# --------------------------------------------------------------------------
# Device linking (RFC 8628 device authorization grant)
# --------------------------------------------------------------------------


def _endpoint(base_url: str, path: str) -> str:
    """A Boske endpoint, refusing any base URL that is not https."""
    return f"{require_https_base_url(base_url, what='Boske base_url')}{path}"


def _require_https_link(url: str) -> str:
    """The URL the user is about to open in a browser, or :class:`LinkError`.

    The response naming it is unauthenticated at this point in the flow, and
    the user is about to approve a device on whatever page it opens. Anything
    but ``https`` — a plain-http hop, a ``javascript:`` URL, a scheme-relative
    one — is refused rather than handed to the browser.
    """
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise LinkError("device link verification URL is not https")
    return url


def _payload_str(payload: dict, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise LinkError(f"device link response has no {name!r}")
    return value


def _payload_number(payload: dict, name: str, default: float | None = None) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LinkError(f"device link response has no numeric {name!r}")
    return float(value)


def start_device_link(
    http: HttpTransport,
    base_url: str,
    *,
    now: float | None = None,
    device_id: str | None = None,
) -> LinkSession:
    """Ask Boske for a user code, mirroring the Boske desktop app's flow.

    ``device_id`` is sent so the lease Boske issues is bound to this Mac and
    :func:`verify_lease` can check it later.
    """
    now = time.time() if now is None else now
    data: dict[str, Any] = {"client_id": CLIENT_ID, "scope": LINK_SCOPE}
    if device_id:
        data["device_id"] = device_id
    status, payload = http(
        "POST",
        _endpoint(base_url, DEVICE_CODE_PATH),
        data,
        {"Accept": "application/json"},
    )
    if status != 200 or not isinstance(payload, dict):
        raise LinkError(f"device authorization request failed (status={status})")

    verification_url = (
        payload.get("verification_uri_complete")
        or payload.get("verification_uri")
        or payload.get("verification_url")
    )
    if not isinstance(verification_url, str) or not verification_url:
        raise LinkError("device link response has no 'verification_uri'")
    verification_url = _require_https_link(verification_url)

    interval_s = _payload_number(payload, "interval", DEFAULT_POLL_INTERVAL_S)
    return LinkSession(
        device_code=_payload_str(payload, "device_code"),
        user_code=_payload_str(payload, "user_code"),
        verification_url=verification_url,
        interval_s=interval_s,
        expires_at=now + _payload_number(payload, "expires_in"),
        next_poll_at=now,
    )


def poll_device_link(
    http: HttpTransport,
    base_url: str,
    session: LinkSession,
    now: float,
) -> str | None:
    """Poll once. Returns the lease token, or ``None`` while still pending.

    Polls no faster than the interval Boske asked for: a call made too early
    returns ``None`` without touching the network, and ``slow_down`` widens
    the interval per RFC 8628 rather than being retried immediately.
    """
    if now >= session.expires_at:
        raise LinkExpired("device authorization request expired")
    if now < session.next_poll_at:
        return None
    session.next_poll_at = now + session.interval_s

    status, payload = http(
        "POST",
        _endpoint(base_url, DEVICE_TOKEN_PATH),
        {
            "client_id": CLIENT_ID,
            "device_code": session.device_code,
            "grant_type": DEVICE_GRANT_TYPE,
        },
        {"Accept": "application/json"},
    )
    if not isinstance(payload, dict):
        raise LinkError(f"device token response is not an object (status={status})")
    if status == 200:
        return _payload_str(payload, "access_token")

    error = payload.get("error")
    if error == "authorization_pending":
        return None
    if error == "slow_down":
        session.interval_s += SLOW_DOWN_INCREMENT_S
        session.next_poll_at = now + session.interval_s
        return None
    if error == "expired_token":
        raise LinkExpired("device authorization request expired")
    if error == "access_denied":
        raise LinkDenied("device link was declined")
    raise LinkError(
        f"device authorization failed (status={status}, error={str(error)[:64]!r})"
    )


# --------------------------------------------------------------------------
# Secret storage and the license service
# --------------------------------------------------------------------------


class SecretStore(Protocol):
    """Where lease tokens live. Wave 3's Keychain store implements this."""

    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None: ...

    def delete(self, name: str) -> None: ...


def device_id_from_store(store: SecretStore) -> str:
    """This Mac's device id, minted on first use and then read back.

    The id is a plain UUID kept next to the lease under
    :data:`DEVICE_ID_SECRET_NAME`. It identifies the machine to Boske at link
    time and is what :func:`verify_lease` compares the ``dev`` claim against,
    so it must survive restarts — which is exactly what the Keychain-backed
    store gives it.
    """
    existing = store.get(DEVICE_ID_SECRET_NAME)
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    minted = str(uuid.uuid4())
    store.set(DEVICE_ID_SECRET_NAME, minted)
    return minted


class InMemorySecretStore:
    """Process-local store for tests. Never used for a real lease on disk."""

    def __init__(self, initial: dict[str, str] | None = None):
        self._values: dict[str, str] = dict(initial or {})

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def delete(self, name: str) -> None:
        self._values.pop(name, None)


class LicenseService:
    """Owns the stored lease and keeps the Pro gate in sync with it.

    Every entitlement answer comes from re-verifying the stored token, so a
    tampered Keychain entry cannot grant Pro. The result is cached for
    :data:`ENTITLEMENTS_CACHE_TTL_S` because verification runs on the UI path.
    """

    def __init__(
        self,
        secret_store: SecretStore,
        http: HttpTransport,
        base_url: str,
        clock: Callable[[], float] = time.time,
        public_key_pem: str | None = None,
        logger: Any = None,
        grace_days: int = DEFAULT_GRACE_DAYS,
        device_id_provider: Callable[[], str] | None = None,
    ):
        self._secret_store = secret_store
        self._http = http
        # The lease is sent to this host as a bearer token, so an endpoint that
        # is not https is refused before the service exists.
        self._base_url = require_https_base_url(base_url, what="Boske base_url")
        self._clock = clock
        self._device_id_provider = device_id_provider or (
            lambda: device_id_from_store(secret_store)
        )
        self._public_key_pem = public_key_pem
        self._logger = logger if logger is not None else _logger
        self._grace_days = grace_days
        self._lock = threading.RLock()
        self._cached: Entitlements | None = None
        self._cached_at = 0.0
        self._link_session: LinkSession | None = None

    # -- identity ----------------------------------------------------------

    def device_id(self) -> str:
        """This Mac's device id; minted and stored on the first call."""
        return str(self._device_id_provider())

    # -- entitlements ------------------------------------------------------

    def current_entitlements(self) -> Entitlements:
        """Entitlements for the stored lease, and publish them to the gate."""
        with self._lock:
            now = self._clock()
            if self._cached is None or now - self._cached_at >= ENTITLEMENTS_CACHE_TTL_S:
                self._cached = self._compute_entitlements(now)
                self._cached_at = now
            entitlements = self._cached
        set_current_entitlements(entitlements)
        return entitlements

    def _compute_entitlements(self, now: float) -> Entitlements:
        token = self._secret_store.get(LEASE_SECRET_NAME)
        if not token:
            return Entitlements.none()
        lease = self._verified_lease(token)
        if lease is None:
            return Entitlements.none()
        return entitlements_from_lease(lease, now, self._grace_days)

    def _verified_lease(self, token: str) -> Lease | None:
        """Verify a stored token. Expired-but-authentic still yields a lease.

        A lease issued to another device is rejected here like any other
        inauthentic one: it grants nothing, and only the shape is logged.
        """
        try:
            return verify_lease(
                token,
                public_key_pem=self._public_key_pem,
                now=self._clock(),
                device_id=self.device_id(),
            )
        except LeaseExpired as expired:
            return expired.lease
        except LeaseError as error:
            self._logger.warning("Stored lease rejected: %s", error)
            return None

    def store_lease(self, token: str) -> Entitlements:
        """Verify, then keep the token in the secret store. Never in JSON.

        An authentic but expired lease is still stored, so the grace window
        survives a restart. An inauthentic one — including one issued to
        another device — raises and is not stored.
        """
        now = self._clock()
        try:
            lease = verify_lease(
                token,
                public_key_pem=self._public_key_pem,
                now=now,
                device_id=self.device_id(),
            )
        except LeaseExpired as expired:
            lease = expired.lease
        with self._lock:
            self._secret_store.set(LEASE_SECRET_NAME, token)
            entitlements = entitlements_from_lease(lease, now, self._grace_days)
            self._cached = entitlements
            self._cached_at = now
        set_current_entitlements(entitlements)
        return entitlements

    def sign_out(self) -> None:
        """Forget the lease and drop straight back to the free tier."""
        with self._lock:
            self._secret_store.delete(LEASE_SECRET_NAME)
            self._link_session = None
            self._cached = Entitlements.none()
            self._cached_at = self._clock()
        set_current_entitlements(Entitlements.none())

    # -- background refresh ------------------------------------------------

    def refresh_if_needed(self) -> bool:
        """Renew the lease once 80% of its lifetime has passed.

        Returns whether a new lease was stored. Every failure keeps the old
        lease and logs the shape of the failure, never the token.
        """
        token = self._secret_store.get(LEASE_SECRET_NAME)
        if not token:
            return False
        lease = self._verified_lease(token)
        if lease is None:
            return False
        if self._clock() < lease.refresh_due_at:
            return False

        try:
            status, payload = self._http(
                "POST",
                _endpoint(self._base_url, DEVICE_REFRESH_PATH),
                {"client_id": CLIENT_ID},
                {"Authorization": f"Bearer {token}", "Accept": "application/json"},
            )
        except Exception as error:  # transport failures must not lose the lease
            self._logger.warning("Lease refresh failed: %s", type(error).__name__)
            return False
        if status != 200 or not isinstance(payload, dict):
            self._logger.warning("Lease refresh failed (status=%s)", status)
            return False
        new_token = payload.get("access_token")
        if not isinstance(new_token, str) or not new_token:
            self._logger.warning("Lease refresh returned no access_token")
            return False
        try:
            self.store_lease(new_token)
        except LeaseError as error:
            self._logger.warning("Refreshed lease rejected: %s", error)
            return False
        return True

    # -- device linking ----------------------------------------------------

    def begin_link(self) -> LinkSession:
        session = start_device_link(
            self._http,
            self._base_url,
            now=self._clock(),
            device_id=self.device_id(),
        )
        with self._lock:
            self._link_session = session
        return session

    def poll_link(self) -> Entitlements | None:
        """``None`` while the user has not approved yet, entitlements once linked."""
        with self._lock:
            session = self._link_session
        if session is None:
            raise LinkError("no device link in progress")
        token = poll_device_link(self._http, self._base_url, session, self._clock())
        if token is None:
            return None
        with self._lock:
            self._link_session = None
        return self.store_lease(token)


# --------------------------------------------------------------------------
# The Pro gate — the only place the app asks whether a feature may run
# --------------------------------------------------------------------------

#: Every gated feature. ``is_pro_feature_enabled`` rejects anything else.
PRO_FEATURES = (
    "cleanup",
    "modes",
    "context",
    "vocabulary_beyond_free",
    "snippets",
    "coding_mode",
    "cloud_voice",
)

#: Features needing the cloud-voice entitlement rather than plain Pro.
_CLOUD_FEATURES = frozenset({"cloud_voice"})

_current_entitlements: Entitlements = Entitlements.none()


def set_current_entitlements(entitlements: Entitlements) -> None:
    """Publish entitlements to the gate. :class:`LicenseService` calls this."""
    assert isinstance(entitlements, Entitlements), "entitlements must be Entitlements"
    global _current_entitlements
    _current_entitlements = entitlements


def get_current_entitlements() -> Entitlements:
    """Read-only view for display (plan name, expiry) — never for gating."""
    return _current_entitlements


def is_pro_feature_enabled(feature: str) -> bool:
    """The one Pro gate. Nothing else in the app may check entitlements."""
    assert feature in PRO_FEATURES, f"unknown Pro feature: {feature!r}"
    entitlements = _current_entitlements
    if feature in _CLOUD_FEATURES:
        return entitlements.cloud_voice
    return entitlements.pro
