#!/usr/bin/env python3
"""Monthly transcription counters and the Murmur Cloud soft limit.

Pure logic over the app's config dict: no AppKit, no network, no files. The
service is handed a ``config_store`` — any object with ``load() -> dict`` and
``save(dict)``, which the app satisfies by wrapping
:meth:`services.persistence_service.PersistenceService.load_config` and
``update_config`` (see :class:`app.services.UsageConfigStore`, which narrows the
save to this service's own ten keys) — and a ``clock`` returning a
:class:`~datetime.datetime`.

Config keys (merge :data:`USAGE_DEFAULTS` into the app's ``DEFAULT_CONFIG``):

=================================  ==================================================
``usage_month``                    ``"YYYY-MM"`` the counters below belong to
``usage_cloud_seconds``            audio sent to Murmur Cloud this month
``usage_cloud_words``              words it returned this month
``usage_local_seconds``            audio transcribed on this Mac this month
``usage_local_words``              words the local engine returned this month
``cloud_fallback_notice_shown``    the "switched to local" notice was shown this month
``cloud_trial_seconds_used``       free-tier one-time 60-minute trial, never reset
``usage_allowance_minutes``        allowance the proxy last reported, in minutes
``usage_remote_minutes_used``      minutes the proxy said were spent, at that moment
``usage_allowance_fetched_at``     ISO-8601 stamp of that reading
=================================  ==================================================

Rollover is automatic: the first call in a new month resets the four counters
and the notice flag — the allowance resets with the billing period, so the
notice is shown once per period rather than once per install. The one-time
trial deliberately survives every rollover.

Own-key (BYOK) transcription is accepted by :meth:`UsageService.record` and
counted nowhere: it is billed by the user's own provider and must never push
the Murmur Cloud counters towards the soft limit.

**Never fall back on a guess.** Switching to the local engine is a visible
product decision, so it is taken only from an allowance the proxy actually
reported and that is younger than :data:`ALLOWANCE_TTL_S`. No allowance, a
zero one, or a stale reading all mean "we do not know", and the answer is then
always False; :meth:`UsageService.allowance_is_stale` is how the app knows to
refresh off the dictation path rather than blocking a recording on an HTTP
round trip.

**Every write is serialised.** Reading the config, changing a counter and
saving it back is three steps, and two dictations can finish at once; an
:class:`~threading.RLock` around each read-modify-write keeps the pair from
losing one of them.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module dependency-free
    from engines.cloud import Usage

_logger = logging.getLogger(__name__)

#: Config keys this service owns.
KEY_MONTH = "usage_month"
KEY_CLOUD_SECONDS = "usage_cloud_seconds"
KEY_CLOUD_WORDS = "usage_cloud_words"
KEY_LOCAL_SECONDS = "usage_local_seconds"
KEY_LOCAL_WORDS = "usage_local_words"
KEY_NOTICE_SHOWN = "cloud_fallback_notice_shown"
KEY_TRIAL_SECONDS = "cloud_trial_seconds_used"
KEY_ALLOWANCE_MINUTES = "usage_allowance_minutes"
KEY_REMOTE_MINUTES_USED = "usage_remote_minutes_used"
KEY_ALLOWANCE_FETCHED_AT = "usage_allowance_fetched_at"

#: Defaults for the keys above; merge into the app's ``DEFAULT_CONFIG``.
USAGE_DEFAULTS: dict[str, Any] = {
    KEY_MONTH: None,
    KEY_CLOUD_SECONDS: 0.0,
    KEY_CLOUD_WORDS: 0,
    KEY_LOCAL_SECONDS: 0.0,
    KEY_LOCAL_WORDS: 0,
    KEY_NOTICE_SHOWN: False,
    KEY_TRIAL_SECONDS: 0.0,
    KEY_ALLOWANCE_MINUTES: None,
    KEY_REMOTE_MINUTES_USED: 0.0,
    KEY_ALLOWANCE_FETCHED_AT: None,
}

#: Keys cleared when the month turns over. The trial is not one of them.
_MONTHLY_KEYS = (
    KEY_CLOUD_SECONDS,
    KEY_CLOUD_WORDS,
    KEY_LOCAL_SECONDS,
    KEY_LOCAL_WORDS,
    KEY_NOTICE_SHOWN,
)

#: Where a transcription came from. ``byok`` is accepted but not metered.
ORIGINS = ("local", "cloud", "byok")

#: Share of the allowance at which Murmur switches to the local engine.
SOFT_LIMIT = 0.95

#: Free-tier one-time cloud trial, in seconds.
TRIAL_TOTAL_SECONDS = 3600.0

#: How long a fetched allowance may be used before it must be re-read. Long
#: enough that no dictation waits on the network, short enough that the switch
#: to local lands in the same session the allowance ran out.
ALLOWANCE_TTL_S = 900

_MONTH_FORMAT = "%Y-%m"
_PERIOD_LABEL_FORMAT = "%B %Y"


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a stored number; a corrupt value reads as the default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a stored integer; a corrupt value reads as the default."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _cached_allowance(config: dict[str, Any]) -> float | None:
    """The cached allowance in minutes, or None when there isn't a usable one.

    Absent, corrupt and zero all collapse to None on purpose: each of them
    means "we do not know what this account is allowed".
    """
    raw = config.get(KEY_ALLOWANCE_MINUTES)
    if raw is None:
        return None
    allowance = _as_float(raw, default=0.0)
    return allowance if allowance > 0 else None


@dataclass(frozen=True)
class UsageSummary:
    """Snapshot of the current month, for Settings and the menu."""

    month: str
    cloud_seconds: float
    cloud_words: int
    local_seconds: float
    local_words: int
    trial_seconds_used: float
    fallback_notice_pending: bool
    #: The allowance last reported by the proxy, in whole minutes, or None
    #: when none has ever been fetched. None means "unknown", not "zero".
    allowance_minutes: int | None = None

    @property
    def cloud_minutes(self) -> float:
        return self.cloud_seconds / 60.0

    @property
    def local_minutes(self) -> float:
        return self.local_seconds / 60.0

    @property
    def period_label(self) -> str:
        """The billing period in words, e.g. ``"September 2026"``.

        Falls back to the stored month verbatim rather than raising: a corrupt
        config must not take a Settings pane down.
        """
        try:
            return datetime.strptime(self.month, _MONTH_FORMAT).strftime(_PERIOD_LABEL_FORMAT)
        except (TypeError, ValueError):
            return str(self.month)


class UsageService:
    """Counts what was transcribed where, and says when to fall back to local."""

    def __init__(
        self,
        config_store: Any,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        assert config_store is not None, "config_store is required"
        assert hasattr(config_store, "load") and hasattr(config_store, "save"), (
            "config_store must expose load() -> dict and save(dict)"
        )
        assert callable(clock), "clock must be callable"
        self._store = config_store
        self._clock = clock
        # Serialises every load-modify-save below. Reentrant because the
        # public methods call one another (``should_switch_to_local`` reads a
        # summary, ``summary`` may persist a rollover).
        self._lock = threading.RLock()

    # -- stored state ------------------------------------------------------

    def _current_month(self) -> str:
        return self._clock().strftime(_MONTH_FORMAT)

    def _read(self) -> tuple[dict[str, Any], bool]:
        """Load the config, rolling the month over. Returns ``(config, changed)``."""
        config = dict(self._store.load() or {})
        month = self._current_month()
        if config.get(KEY_MONTH) == month:
            return config, False
        for key in _MONTHLY_KEYS:
            config[key] = USAGE_DEFAULTS[key]
        config[KEY_MONTH] = month
        config.setdefault(KEY_TRIAL_SECONDS, USAGE_DEFAULTS[KEY_TRIAL_SECONDS])
        return config, True

    def _read_fresh(self) -> dict[str, Any]:
        """Read for a query, persisting a rollover the query itself caused."""
        with self._lock:
            config, changed = self._read()
            if changed:
                self._store.save(config)
            return config

    # -- recording ---------------------------------------------------------

    def record(self, origin: str, seconds: float, words: int) -> None:
        """Add one finished transcription to this month's counters.

        ``origin`` is ``"local"``, ``"cloud"`` or ``"byok"``; own-key usage is
        not metered, so it only triggers the month rollover.

        Read-modify-save under the lock: two dictations finishing at the same
        moment must add up, not overwrite each other.
        """
        assert origin in ORIGINS, f"unknown origin {origin!r}; expected one of {ORIGINS}"
        assert seconds >= 0, "seconds must not be negative"
        assert words >= 0, "words must not be negative"

        with self._lock:
            config, _changed = self._read()
            if origin == "cloud":
                config[KEY_CLOUD_SECONDS] = (
                    _as_float(config.get(KEY_CLOUD_SECONDS)) + float(seconds)
                )
                config[KEY_CLOUD_WORDS] = _as_int(config.get(KEY_CLOUD_WORDS)) + int(words)
            elif origin == "local":
                config[KEY_LOCAL_SECONDS] = (
                    _as_float(config.get(KEY_LOCAL_SECONDS)) + float(seconds)
                )
                config[KEY_LOCAL_WORDS] = _as_int(config.get(KEY_LOCAL_WORDS)) + int(words)
            self._store.save(config)

    def summary(self) -> UsageSummary:
        """This month's counters, rolled over if the month has turned."""
        with self._lock:
            config = self._read_fresh()
            allowance = _cached_allowance(config)
            return UsageSummary(
                month=str(config.get(KEY_MONTH) or self._current_month()),
                cloud_seconds=_as_float(config.get(KEY_CLOUD_SECONDS)),
                cloud_words=_as_int(config.get(KEY_CLOUD_WORDS)),
                local_seconds=_as_float(config.get(KEY_LOCAL_SECONDS)),
                local_words=_as_int(config.get(KEY_LOCAL_WORDS)),
                trial_seconds_used=_as_float(config.get(KEY_TRIAL_SECONDS)),
                fallback_notice_pending=not bool(config.get(KEY_NOTICE_SHOWN)),
                allowance_minutes=int(round(allowance)) if allowance else None,
            )

    # -- the one deliberate product fallback -------------------------------

    @property
    def known_allowance_minutes(self) -> float | None:
        """Allowance the proxy last reported, or None if it never has.

        Persisted, so it survives a restart; :meth:`allowance_is_stale` says
        whether it is recent enough to act on.
        """
        return _cached_allowance(self._read_fresh())

    def refresh_allowance(self, usage: "Usage") -> None:
        """Cache a fresh reading from ``GET /v1/voice/usage``.

        An allowance of zero or less is *unknown*, not spent, so it is not
        cached at all: writing it would be indistinguishable from "your plan
        gives you nothing" and would move the user onto the local engine.
        """
        assert usage is not None, "usage is required"
        allowance = float(getattr(usage, "minutes_allowance", 0.0) or 0.0)
        if allowance <= 0:
            _logger.info(
                "Cloud usage came back without an allowance; keeping the cloud engine"
            )
            return
        with self._lock:
            config, _changed = self._read()
            config[KEY_ALLOWANCE_MINUTES] = allowance
            config[KEY_REMOTE_MINUTES_USED] = float(
                getattr(usage, "minutes_used", 0.0) or 0.0
            )
            config[KEY_ALLOWANCE_FETCHED_AT] = self._clock().isoformat()
            self._store.save(config)

    def allowance_is_stale(self, allowance_ttl_s: float = ALLOWANCE_TTL_S) -> bool:
        """True when the cached allowance is missing or older than the TTL.

        The wiring polls this off the dictation path: a recording must never
        wait on an HTTP round trip to find out how many minutes are left.
        """
        assert allowance_ttl_s > 0, "allowance_ttl_s must be positive"
        config = self._read_fresh()
        if _cached_allowance(config) is None:
            return True
        age = self._age_seconds(config.get(KEY_ALLOWANCE_FETCHED_AT))
        return age is None or age < 0.0 or age > allowance_ttl_s

    def _age_seconds(self, stamp: Any) -> float | None:
        """Seconds since ``stamp``, or None when it cannot be read as a time."""
        if not isinstance(stamp, str) or not stamp.strip():
            return None
        try:
            when = datetime.fromisoformat(stamp.strip())
        except ValueError:
            return None
        now = self._clock()
        if (when.tzinfo is None) != (now.tzinfo is None):
            when, now = when.replace(tzinfo=None), now.replace(tzinfo=None)
        return (now - when).total_seconds()

    def should_switch_to_local(
        self,
        remote: "Usage | None" = None,
        *,
        soft_limit: float = SOFT_LIMIT,
        allowance_ttl_s: float = ALLOWANCE_TTL_S,
    ) -> bool:
        """True when Murmur Cloud is at or past ``soft_limit`` of the allowance.

        A ``remote`` reading is authoritative and is cached on the way through.
        Without one the cached reading is used, but only while it is younger
        than ``allowance_ttl_s``; local cloud minutes recorded since that
        reading still count, so the limit is not missed between refreshes.

        **False is the answer to every unknown**: no allowance in the reading,
        nothing cached, or a cache too old to trust. Falling back to the local
        engine is a visible change the user did not ask for, and it must never
        rest on a guess.
        """
        assert 0 < soft_limit <= 1, "soft_limit must be within (0, 1]"

        with self._lock:
            if remote is not None:
                self.refresh_allowance(remote)
                fraction = remote.fraction_used
                if fraction is None:
                    _logger.info(
                        "Cloud allowance is unknown (allowance=%s); staying on the "
                        "cloud engine rather than guessing",
                        getattr(remote, "minutes_allowance", None),
                    )
                    return False
                return fraction >= soft_limit

            if self.allowance_is_stale(allowance_ttl_s):
                return False
            config = self._read_fresh()
            allowance = _cached_allowance(config)
            if allowance is None:  # pragma: no cover - allowance_is_stale covers it
                return False
            # The proxy's own number, or the minutes we have counted locally
            # since it was taken, whichever is further along.
            used = max(
                _as_float(config.get(KEY_REMOTE_MINUTES_USED)),
                _as_float(config.get(KEY_CLOUD_SECONDS)) / 60.0,
            )
            return used / allowance >= soft_limit

    @property
    def fallback_notice_pending(self) -> bool:
        """True until the "switched to local" notice was shown this month."""
        return not bool(self._read_fresh().get(KEY_NOTICE_SHOWN))

    def mark_fallback_notice_shown(self) -> None:
        """Remember that the user has been told. Idempotent."""
        with self._lock:
            config, _changed = self._read()
            config[KEY_NOTICE_SHOWN] = True
            self._store.save(config)

    # -- free-tier trial ---------------------------------------------------

    def trial_remaining_seconds(self, trial_total: float = TRIAL_TOTAL_SECONDS) -> float:
        """Seconds left of the one-time cloud trial. Never negative."""
        assert trial_total >= 0, "trial_total must not be negative"
        used = _as_float(self._read_fresh().get(KEY_TRIAL_SECONDS))
        return max(0.0, trial_total - used)

    def consume_trial(
        self,
        seconds: float,
        trial_total: float = TRIAL_TOTAL_SECONDS,
    ) -> float:
        """Spend ``seconds`` of the trial; return what is left afterwards.

        The stored total is clamped at ``trial_total`` so a long clip cannot
        run the counter away past the trial it is spending.
        """
        assert seconds >= 0, "seconds must not be negative"
        assert trial_total >= 0, "trial_total must not be negative"
        with self._lock:
            config, _changed = self._read()
            used = min(trial_total, _as_float(config.get(KEY_TRIAL_SECONDS)) + float(seconds))
            config[KEY_TRIAL_SECONDS] = used
            self._store.save(config)
        return max(0.0, trial_total - used)
