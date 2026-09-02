#!/usr/bin/env python3
"""Which engine this dictation runs on, and what the user is told about it.

One pure function, :func:`route_engine`, and no state: it is handed the config
mode, the entitlements, the usage counters and the length of the clip, and it
returns a :class:`Route` — an engine id, an optional one-line notice, and a
short machine-readable ``reason`` for the log. It reads no config file, opens
no socket and shows no alert; the caller does all three. That is what makes the
whole routing table testable as a table.

**Cloud is a mode, not an engine choice.** ``cloud_mode`` is orthogonal to the
speech engine the user picked in Settings: ``local_engine_id`` is whichever of
``whispercpp``/``voxtral_mlx`` they configured, and it is the answer whenever
the cloud route is unavailable. So a user who switched their local engine still
gets *their* engine back when the allowance runs out, not a default.

The table
---------

===================================  ===================  =======================
condition                            engine               notice
===================================  ===================  =======================
``own_key`` and a key is stored      ``byok``             —
``own_key`` and no key               ``local_engine_id``  :data:`NOTICE_ADD_KEY`
``off``                              ``local_engine_id``  —
``murmur_cloud``, no lease or no
entitlement and no trial left        ``local_engine_id``  :data:`NOTICE_SIGN_IN`
``murmur_cloud``, clip over
:data:`MAX_CLIP_SECONDS`             ``local_engine_id``  :data:`NOTICE_CLIP_TOO_LONG`
``murmur_cloud``, past the soft
limit                                ``local_engine_id``  :data:`ALLOWANCE_MESSAGE`,
                                                          once per period
``murmur_cloud``, otherwise          ``cloud``            —
===================================  ===================  =======================

Order matters, and it is the order above. Entitlement comes first because a
user who has not signed in was never offered the cloud, so telling them their
hour-long recording is too long would answer a question they did not ask.

The allowance notice is the one the folder rules single out: cloud → local is
the only fallback the app is allowed to take on its own, and it must say so
**once**. :func:`route_engine` only asks whether the notice is still pending
(``usage.fallback_notice_pending``); marking it shown is the caller's job,
after it has actually displayed it, so a route computed and discarded does not
burn the one notice the user gets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from cleanup.vocabulary import FREE_TERM_LIMIT
from engines.cloud import ALLOWANCE_MESSAGE, CloudAllowanceExhausted, CloudAuthError

#: Values of the ``cloud_mode`` config key.
CLOUD_MODE_OFF = "off"
CLOUD_MODE_MURMUR = "murmur_cloud"
CLOUD_MODE_OWN_KEY = "own_key"
CLOUD_MODES = (CLOUD_MODE_OFF, CLOUD_MODE_MURMUR, CLOUD_MODE_OWN_KEY)

#: Engine ids this module can return besides ``local_engine_id``.
ENGINE_CLOUD = "cloud"
ENGINE_BYOK = "byok"

#: The engines that send audio off this Mac. One table rather than an
#: ``in (ENGINE_CLOUD, ENGINE_BYOK)`` written out at each of the three places
#: that asks: the usage meter, the history origin and what the pipeline tells
#: the user is happening. A hosted engine missing from one of those would be
#: counted, recorded and described as local, which is the one mistake this app
#: must not make.
REMOTE_ENGINE_IDS: tuple[str, ...] = (ENGINE_CLOUD, ENGINE_BYOK)

#: Longest clip Murmur Cloud accepts, in seconds. Mirrors
#: :data:`engines.cloud.MAX_MINUTES`; a longer recording is transcribed here.
MAX_CLIP_SECONDS = 3600.0

NOTICE_ADD_KEY = "Add your API key in Settings › Account"
NOTICE_SIGN_IN = "Sign in with your Boske ID to use Murmur Cloud"
NOTICE_CLIP_TOO_LONG = (
    "Recordings over 60 minutes are transcribed on this Mac"
)


@dataclass(frozen=True)
class Route:
    """Where one dictation goes, and what to say about it.

    ``notice`` is user-facing text or None. ``reason`` is for the log and for
    tests: short, stable, and never shown to anyone.
    """

    engine_id: str
    notice: str | None
    reason: str


def _entitled_to_cloud(entitlements: Any, usage: Any) -> bool:
    """Whether this account may reach the proxy at all right now.

    Either a paid ``cloud_voice`` entitlement, or the free trial with seconds
    still on it. Both arrive on a lease: decision D6 has no anonymous cloud
    endpoint, so the trial requires a Boske sign-in like everything else.
    """
    if entitlements is not None and getattr(entitlements, "cloud_voice", False):
        return True
    return usage.trial_remaining_seconds() > 0


def route_engine(
    *,
    cloud_mode: str,
    local_engine_id: str,
    entitlements: Any,
    has_lease: bool,
    usage: Any,
    key_present: bool,
    clip_seconds: float | None,
) -> Route:
    """Resolve one dictation to an engine, a notice and a reason.

    ``entitlements`` may be None (no license service, or none loaded yet),
    which reads as "nothing entitled". ``clip_seconds`` may be None when the
    length is not known yet; an unknown length never blocks the cloud, because
    the engine enforces its own cap anyway.

    Pure: nothing here records usage, marks a notice shown or touches config.
    """
    assert local_engine_id, "local_engine_id is required"
    assert usage is not None, "usage is required"

    mode = str(cloud_mode or CLOUD_MODE_OFF).strip() or CLOUD_MODE_OFF

    if mode == CLOUD_MODE_OWN_KEY:
        if key_present:
            return Route(ENGINE_BYOK, None, "own key")
        return Route(local_engine_id, NOTICE_ADD_KEY, "own key missing")

    if mode != CLOUD_MODE_MURMUR:
        # "off", and anything unrecognised: an unknown mode must not be a way
        # to reach a metered engine.
        return Route(local_engine_id, None, "cloud off")

    if not has_lease:
        return Route(local_engine_id, NOTICE_SIGN_IN, "no lease")
    if not _entitled_to_cloud(entitlements, usage):
        return Route(local_engine_id, NOTICE_SIGN_IN, "not entitled")

    if clip_seconds is not None and float(clip_seconds) > MAX_CLIP_SECONDS:
        return Route(local_engine_id, NOTICE_CLIP_TOO_LONG, "clip too long")

    if usage.should_switch_to_local(None):
        # The one self-taken fallback, and the one notice that goes with it.
        notice = ALLOWANCE_MESSAGE if usage.fallback_notice_pending else None
        return Route(local_engine_id, notice, "allowance soft limit")

    return Route(ENGINE_CLOUD, None, "cloud")


def after_cloud_failure(exc: BaseException, *, local_engine_id: str) -> Route:
    """Where to re-run a clip the cloud just refused.

    The two failures the proxy contract names get the wording the user needs:
    a spent allowance says so (once — the caller still checks
    ``fallback_notice_pending`` before showing it), a rejected lease asks them
    to sign in again. Anything else falls back silently, because the user asked
    for a transcript and a transient proxy error is not their problem.
    """
    assert local_engine_id, "local_engine_id is required"
    if isinstance(exc, CloudAllowanceExhausted):
        return Route(local_engine_id, ALLOWANCE_MESSAGE, "allowance exhausted")
    if isinstance(exc, CloudAuthError):
        return Route(local_engine_id, NOTICE_SIGN_IN, "lease rejected")
    return Route(local_engine_id, None, "cloud failed")


def effective_vocabulary_terms(
    terms: Iterable[str] | None,
    pro_gate: Callable[[str], bool],
) -> tuple[str, ...]:
    """The vocabulary terms this install may actually use.

    The free tier keeps the first :data:`~cleanup.vocabulary.FREE_TERM_LIMIT`
    terms and drops the rest for this pass only — nothing is deleted from the
    user's list, so subscribing brings every term straight back. Truncating
    here rather than in the editor is deliberate: the list is the user's data,
    the limit is a licensing state, and the two must not be conflated on disk.

    ``pro_gate`` is :func:`services.license_service.is_pro_feature_enabled`,
    injected so this stays pure.
    """
    assert callable(pro_gate), "pro_gate must be callable"
    items: Sequence[str] = tuple(terms or ())
    if pro_gate("vocabulary_beyond_free"):
        return tuple(items)
    return tuple(items[:FREE_TERM_LIMIT])


__all__ = [
    "ALLOWANCE_MESSAGE",
    "CLOUD_MODES",
    "CLOUD_MODE_MURMUR",
    "CLOUD_MODE_OFF",
    "CLOUD_MODE_OWN_KEY",
    "ENGINE_BYOK",
    "ENGINE_CLOUD",
    "MAX_CLIP_SECONDS",
    "NOTICE_ADD_KEY",
    "NOTICE_CLIP_TOO_LONG",
    "NOTICE_SIGN_IN",
    "REMOTE_ENGINE_IDS",
    "Route",
    "after_cloud_failure",
    "effective_vocabulary_terms",
    "route_engine",
]
