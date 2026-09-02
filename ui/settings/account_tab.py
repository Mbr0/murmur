#!/usr/bin/env python3
"""Settings → Account: licence, Boske ID sign-in, own keys, version, updates.

Three halves, as in the other tabs:

* :class:`AccountTabModel` — plain Python. The Pro status line, the
  device-linking state machine, the own-key (BYOK) operations and
  :meth:`AccountTabModel.apply`, which names the one config key this tab owns.
* :class:`AccountTab` — the AppKit rendering. Imports Cocoa inside
  :meth:`AccountTab.build`, never at module scope.
* The keychain rules the tab exists to keep, below.

**No secret is ever written to the config file.** The Boske lease is handled by
``services/license_service.py`` (Wave 4); the two own-key API keys are written
straight to the login keychain through :class:`~services.keychain.KeychainStore`
and are never read back into the UI. The "key stored" indicator asks
``has(item)``, so a key that is stored is never echoed into a field, a label, a
log line or the config.

Sign-in is device linking (decision D6), not key pasting: ``begin_link()``
returns a user code and a verification URL, and the model polls on an injected
scheduler until the poller reports success, the code expires, or the user
cancels. Murmur Cloud credentials never pass through the keyboard.

Everything the model touches is injected — the licence provider, the two link
callables, sign-out, the secret store, the scheduler and the clock — so the
tests drive the whole flow without a network, a keychain or a run loop. The
Wave 4 ``LicenseService`` fits without adapters:
``license_provider=service.current_entitlements``,
``link_starter=service.begin_link``, ``link_poller=service.poll_link``,
``sign_out=service.sign_out``.

Config keys owned here:

- ``update_channel``: ``"stable" | "beta"``, default ``"stable"``. The only key
  this tab writes.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable

from services.keychain import BYOK_ITEMS, KeychainError
from ui.settings import register_tab
from ui.settings.base import TAB_ACCOUNT, TabContext

logger = logging.getLogger(__name__)

#: Config key for the update channel, and its two values.
CONFIG_UPDATE_CHANNEL = "update_channel"
CHANNEL_STABLE = "stable"
CHANNEL_BETA = "beta"
UPDATE_CHANNELS: tuple[str, ...] = (CHANNEL_STABLE, CHANNEL_BETA)
DEFAULT_UPDATE_CHANNEL = CHANNEL_STABLE

CHANNEL_LABELS: dict[str, str] = {
    CHANNEL_STABLE: "Stable",
    CHANNEL_BETA: "Beta (early builds, more bugs)",
}

#: Own-key providers, in display order, and the keychain item each one uses.
BYOK_PROVIDERS: tuple[str, ...] = ("mistral", "openai")
BYOK_PROVIDER_LABELS: dict[str, str] = {"mistral": "Mistral", "openai": "OpenAI"}

#: Names of the providers this tab reads out of :attr:`TabContext.services`.
SERVICE_LICENSE = "license"
SERVICE_KEYCHAIN = "keychain"
SERVICE_SCHEDULER = "scheduler"

#: Device-linking states.
LINK_IDLE = "idle"
LINK_PENDING = "pending"
LINK_SUCCESS = "success"
LINK_EXPIRED = "expired"
LINK_CANCELLED = "cancelled"
LINK_DENIED = "denied"
LINK_FAILED = "failed"

LINK_STATES: tuple[str, ...] = (
    LINK_IDLE,
    LINK_PENDING,
    LINK_SUCCESS,
    LINK_EXPIRED,
    LINK_CANCELLED,
    LINK_DENIED,
    LINK_FAILED,
)

#: Fallback poll interval when the link session does not name one.
DEFAULT_POLL_INTERVAL_S = 5.0

STATUS_FREE = "Free"
STATUS_PRO = "Pro"
STATUS_PRO_UNTIL = "Pro until {date}"
STATUS_PRO_GRACE = "Pro (grace, renew by {date})"
STATUS_PRO_GRACE_NO_DATE = "Pro (grace)"
CLOUD_VOICE_INCLUDED = "Cloud voice included"
CLOUD_VOICE_MINUTES = "Cloud voice included — {minutes} minutes a month"
STATUS_UNAVAILABLE = "Licence status unavailable"

KEY_STORED = "Key stored"
KEY_NOT_STORED = "No key stored"
KEY_UNAVAILABLE = "Keychain unavailable"

#: What one look-up can say about a provider's key, without reading it.
KEY_STATE_STORED = "stored"
KEY_STATE_ABSENT = "absent"
KEY_STATE_UNAVAILABLE = "unavailable"

KEY_STATE_LABELS: dict[str, str] = {
    KEY_STATE_STORED: KEY_STORED,
    KEY_STATE_ABSENT: KEY_NOT_STORED,
    KEY_STATE_UNAVAILABLE: KEY_UNAVAILABLE,
}

LINK_INSTRUCTION = "Enter {code} at {url}"
LINK_SIGNED_IN = "Signed in with your Boske ID"
LINK_EXPIRED_TEXT = "That code expired. Start again when you are ready."
LINK_CANCELLED_TEXT = "Sign-in cancelled."
LINK_DENIED_TEXT = "Sign-in was declined"
LINK_FAILED_TEXT = "Sign-in could not be completed. Check your connection and try again."

INTERNAL_BUILD_MARKER = "internal build"

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def format_expiry(value: Any) -> str:
    """A licence date as "3 April 2027". Empty string when there is no date.

    Accepts what a lease can carry: a ``datetime``, a ``date``, an ISO-8601
    string (with or without a trailing ``Z``), or a POSIX timestamp. Anything
    it cannot read becomes "", and the caller drops the date from its line
    rather than printing something wrong about the user's licence.
    """
    if value is None or value == "":
        return ""
    moment: date | None = None
    if isinstance(value, datetime):
        moment = value.date()
    elif isinstance(value, date):
        moment = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            moment = datetime.fromtimestamp(float(value)).date()
        except (OverflowError, OSError, ValueError):
            return ""
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text).date()
        except ValueError:
            try:
                moment = date.fromisoformat(value.strip()[:10])
            except ValueError:
                return ""
    if moment is None:
        return ""
    return f"{moment.day} {_MONTHS[moment.month - 1]} {moment.year}"


def normalised_channel(value: Any) -> str:
    """One of :data:`UPDATE_CHANNELS`; anything else means the default.

    A config file edited by hand must not stop Settings from opening, so an
    unknown channel is read as "stable" rather than asserted on.
    """
    text = str(value or "").strip().lower()
    return text if text in UPDATE_CHANNELS else DEFAULT_UPDATE_CHANNEL


def as_timestamp(value: Any) -> float | None:
    """A deadline as a POSIX timestamp, or None when it cannot be read.

    Unreadable means "no local deadline": the model then waits for the poller
    to say the code is dead rather than inventing an expiry of its own.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).timestamp()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _default_now() -> float:
    """POSIX time. A function so tests can hand the model a frozen clock."""
    import time

    return time.time()


def _poll_error_state(error: Exception) -> str:
    """LINK_EXPIRED or LINK_DENIED for the poller's two named exceptions.

    Classified by class name rather than ``isinstance``, so this module needs
    no import of ``services/license_service.py`` (Wave 4): the real poller
    raises ``LinkExpired`` and ``LinkDenied``, both subclasses of ``LinkError``
    and, like every exception, of ``Exception``. Anything else is LINK_FAILED.
    """
    name = type(error).__name__
    if name == "LinkExpired":
        return LINK_EXPIRED
    if name == "LinkDenied":
        return LINK_DENIED
    return LINK_FAILED


def _start_timer(delay_s: float, callback: Callable[[], None]) -> Any:
    """Default scheduler: a daemon ``threading.Timer`` that fires once."""
    import threading

    timer = threading.Timer(delay_s, callback)
    timer.daemon = True
    timer.start()
    return timer


class AccountTabModel:
    """The Account tab as plain data and one small state machine.

    Every dependency is a callable so the tests need neither a network nor a
    keychain:

    ``license_provider()``
        returns the current ``Entitlements`` (``pro``, ``cloud_voice``,
        ``msm_minutes``, ``expires_at``, ``in_grace``, ``source``) or None.
    ``link_starter()``
        starts device linking and returns a ``LinkSession`` (``user_code``,
        ``verification_url``, ``interval_s``, ``expires_at``).
    ``link_poller()``
        returns True once the code has been claimed.
    ``sign_out()``
        forgets the lease. The own keys are deliberately left alone.
    ``keychain``
        a ``SecretStore``: ``get``/``set``/``delete``, optionally ``has``.
    ``scheduler(delay_s, callback)``
        schedules one poll and returns something with ``cancel()``.
    ``now()``
        POSIX time, for the expiry check.
    """

    def __init__(
        self,
        config: dict,
        *,
        license_provider: Callable[[], Any],
        link_starter: Callable[[], Any],
        link_poller: Callable[[], bool],
        sign_out: Callable[[], None],
        keychain: Any,
        version: str,
        build_info: dict,
        scheduler: Callable[[float, Callable[[], None]], Any] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        assert config is not None, "config is required"
        assert callable(license_provider), "license_provider must be callable"
        assert callable(link_starter), "link_starter must be callable"
        assert callable(link_poller), "link_poller must be callable"
        assert callable(sign_out), "sign_out must be callable"
        assert keychain is not None, "a secret store is required"
        assert version, "version is required"

        self._config = config
        self._license_provider = license_provider
        self._link_starter = link_starter
        self._link_poller = link_poller
        self._sign_out = sign_out
        self.keychain = keychain
        self.version = str(version)
        self.build_info = dict(build_info or {})
        self._scheduler = scheduler or _start_timer
        self._now = now or _default_now

        self.entitlements: Any = None
        self.license_available = True
        self.link_state = LINK_IDLE
        self.session: Any = None
        self._timer: Any = None

        self.update_channel = normalised_channel(config.get(CONFIG_UPDATE_CHANNEL))
        self._original_channel = self.update_channel

        self.refresh_entitlements()

    # -- licence status --------------------------------------------------

    def refresh_entitlements(self) -> Any:
        """Re-read the licence. A provider that fails leaves the tab usable."""
        try:
            self.entitlements = self._license_provider()
            self.license_available = True
        except Exception as error:  # noqa: BLE001 - any client failure, never fatal here
            self.entitlements = None
            self.license_available = False
            logger.warning("Licence status could not be read: %s", error)
        return self.entitlements

    @property
    def is_pro(self) -> bool:
        return bool(getattr(self.entitlements, "pro", False))

    @property
    def has_cloud_voice(self) -> bool:
        return bool(getattr(self.entitlements, "cloud_voice", False))

    @property
    def is_signed_in(self) -> bool:
        """Whether a lease is present at all, Pro or not."""
        return self.entitlements is not None and bool(getattr(self.entitlements, "source", ""))

    def status_line(self) -> str:
        """The one line that says what this Mac is entitled to."""
        if not self.license_available:
            return STATUS_UNAVAILABLE
        if not self.is_pro:
            return STATUS_FREE
        when = format_expiry(getattr(self.entitlements, "expires_at", None))
        if getattr(self.entitlements, "in_grace", False):
            return STATUS_PRO_GRACE.format(date=when) if when else STATUS_PRO_GRACE_NO_DATE
        return STATUS_PRO_UNTIL.format(date=when) if when else STATUS_PRO

    def detail_line(self) -> str:
        """The cloud-voice line under the status, or "" when not entitled."""
        if not self.license_available or not self.has_cloud_voice:
            return ""
        minutes = getattr(self.entitlements, "msm_minutes", None)
        if isinstance(minutes, (int, float)) and not isinstance(minutes, bool) and minutes > 0:
            return CLOUD_VOICE_MINUTES.format(minutes=int(minutes))
        return CLOUD_VOICE_INCLUDED

    # -- device linking --------------------------------------------------

    @property
    def user_code(self) -> str:
        return str(getattr(self.session, "user_code", "") or "")

    @property
    def verification_url(self) -> str:
        return str(getattr(self.session, "verification_url", "") or "")

    @property
    def is_linking(self) -> bool:
        return self.link_state == LINK_PENDING

    def link_line(self) -> str:
        """What to show under the sign-in button for the current state."""
        if self.link_state == LINK_PENDING:
            return LINK_INSTRUCTION.format(code=self.user_code, url=self.verification_url)
        if self.link_state == LINK_SUCCESS:
            return LINK_SIGNED_IN
        if self.link_state == LINK_EXPIRED:
            return LINK_EXPIRED_TEXT
        if self.link_state == LINK_CANCELLED:
            return LINK_CANCELLED_TEXT
        if self.link_state == LINK_DENIED:
            return LINK_DENIED_TEXT
        if self.link_state == LINK_FAILED:
            return LINK_FAILED_TEXT
        return ""

    def begin_link(self) -> Any:
        """Ask for a user code and start polling. Returns the link session.

        Starting again while a code is live restarts the flow: the old timer is
        cancelled first, so two codes are never polled at once.
        """
        self._cancel_timer()
        try:
            session = self._link_starter()
        except Exception as error:  # noqa: BLE001 - any client failure shows as a failed link
            self.session = None
            self.link_state = LINK_FAILED
            logger.warning("Could not start Boske ID sign-in: %s", error)
            return None
        assert session is not None, "link_starter returned no session"
        self.session = session
        self.link_state = LINK_PENDING
        self._schedule_poll()
        return session

    def poll_link(self) -> str:
        """One poll. Returns the state afterwards.

        Called by the scheduler and, in tests, directly. Polling when no code
        is live is a no-op, so a timer that fires after cancel changes nothing.
        The poller returns the ``Entitlements`` object once the code is
        claimed, or ``None`` while waiting — success is judged by ``is not
        None``, not truthiness, so an entitlements object that happens to be
        falsy is still a success.
        """
        if self.link_state != LINK_PENDING:
            return self.link_state
        try:
            result = self._link_poller()
        except Exception as error:  # noqa: BLE001 - a poll failure ends the attempt
            self._cancel_timer()
            self.link_state = _poll_error_state(error)
            if self.link_state != LINK_FAILED:
                self.session = None
            logger.warning("Boske ID sign-in could not be polled: %s", error)
            return self.link_state

        if result is not None:
            self._cancel_timer()
            self.link_state = LINK_SUCCESS
            self.session = None
            self.refresh_entitlements()
            return self.link_state

        if self._deadline_passed():
            self._cancel_timer()
            self.link_state = LINK_EXPIRED
            self.session = None
            return self.link_state

        self._schedule_poll()
        return self.link_state

    def cancel_link(self) -> str:
        """Stop waiting for the code. Does nothing when nothing is pending."""
        if self.link_state != LINK_PENDING:
            return self.link_state
        self._cancel_timer()
        self.session = None
        self.link_state = LINK_CANCELLED
        return self.link_state

    def sign_out(self) -> None:
        """Forget the lease and go back to Free. Own keys are left in place."""
        self._cancel_timer()
        self.session = None
        self.link_state = LINK_IDLE
        self._sign_out()
        self.refresh_entitlements()

    def poll_interval_s(self) -> float:
        """How long to wait before the next poll."""
        interval = getattr(self.session, "interval_s", None)
        if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval > 0:
            return float(interval)
        return DEFAULT_POLL_INTERVAL_S

    def _deadline_passed(self) -> bool:
        deadline = as_timestamp(getattr(self.session, "expires_at", None))
        return deadline is not None and self._now() >= deadline

    def _schedule_poll(self) -> None:
        self._timer = self._scheduler(self.poll_interval_s(), self.poll_link)

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is None:
            return
        cancel = getattr(timer, "cancel", None)
        if cancel is not None:
            cancel()

    # -- own keys (BYOK) -------------------------------------------------

    @staticmethod
    def item_for(provider: str) -> str:
        """The keychain item name for a provider."""
        assert provider in BYOK_ITEMS, (
            f"Unknown provider {provider!r}; expected one of {', '.join(BYOK_PROVIDERS)}"
        )
        return BYOK_ITEMS[provider]

    def key_state(self, provider: str) -> str:
        """One look-up: stored, absent, or a Keychain that will not answer.

        A locked keychain, a cancelled prompt, or an unsigned bundle without
        the keychain entitlement all raise out of ``has``. Asking is part of
        building the tab, so raising here means Settings never opens at all —
        hence the refusal is caught, reported as a third state, and logged as
        an OSStatus. The secret is never touched on any path.
        """
        item = self.item_for(provider)
        try:
            has = getattr(self.keychain, "has", None)
            if callable(has):
                stored = bool(has(item))
            else:
                # A store with no ``has`` still must not leak: the value is
                # compared against None and dropped, never returned or logged.
                stored = self.keychain.get(item) is not None
        except KeychainError as error:
            # KeychainUnavailable is a KeychainError, so both land here. The
            # error carries an OSStatus and an item name, never the key.
            logger.warning(
                "The Keychain could not be read for %s (OSStatus %s)", provider, error.status
            )
            return KEY_STATE_UNAVAILABLE
        return KEY_STATE_STORED if stored else KEY_STATE_ABSENT

    def key_stored(self, provider: str) -> bool:
        """Whether a key exists, asked without reading the key itself.

        A Keychain that cannot answer reads as "no key" here; ask
        :meth:`key_available` to tell that apart from an empty slot.
        """
        return self.key_state(provider) == KEY_STATE_STORED

    def key_available(self, provider: str) -> bool:
        """Whether the Keychain answered at all — the gate on Save and Remove.

        Offering buttons that cannot work is worse than showing none: both
        would fail in the same way the indicator has already explained.
        """
        return self.key_state(provider) != KEY_STATE_UNAVAILABLE

    def key_indicator(self, provider: str) -> str:
        """"Key stored", "No key stored", or "Keychain unavailable". Never the key."""
        return KEY_STATE_LABELS[self.key_state(provider)]

    def save_key(self, provider: str, value: str) -> None:
        """Write an own key to the keychain. Never to the config file."""
        item = self.item_for(provider)
        assert isinstance(value, str), "an API key must be a string"
        secret = value.strip()
        if not secret:
            raise ValueError(f"no key to store for {BYOK_PROVIDER_LABELS[provider]}")
        self.keychain.set(item, secret)
        logger.info("Stored an own key for %s", provider)  # the provider, never the key

    def remove_key(self, provider: str) -> None:
        """Delete an own key from the keychain."""
        self.keychain.delete(self.item_for(provider))
        logger.info("Removed the own key for %s", provider)

    # -- version and update channel --------------------------------------

    def version_line(self) -> str:
        """"Murmur 1.0.0", plus the internal-build marker when unsigned."""
        line = f"Murmur {self.version}"
        if self.build_info.get("signed") is False:
            return f"{line} · {INTERNAL_BUILD_MARKER}"
        return line

    def channel_labels(self) -> tuple[str, ...]:
        return tuple(CHANNEL_LABELS[channel] for channel in UPDATE_CHANNELS)

    def channel_index(self) -> int:
        return UPDATE_CHANNELS.index(self.update_channel)

    def set_update_channel(self, channel: str) -> None:
        assert channel in UPDATE_CHANNELS, (
            f"Unknown update channel {channel!r}; expected one of {', '.join(UPDATE_CHANNELS)}"
        )
        self.update_channel = channel

    def set_channel_index(self, index: int) -> None:
        assert 0 <= index < len(UPDATE_CHANNELS), f"row {index} is out of range"
        self.set_update_channel(UPDATE_CHANNELS[index])

    def apply(self) -> dict:
        """The config keys that changed. Only ever ``update_channel``."""
        if self.update_channel == self._original_channel:
            return {}
        return {CONFIG_UPDATE_CHANNEL: self.update_channel}

    def mark_saved(self) -> None:
        """Called once :meth:`apply`'s dict has been persisted."""
        self._original_channel = self.update_channel

    def close(self) -> None:
        """Stop polling and drop the code. The window calls this on the way out.

        A pending sign-in is cancelled rather than merely unscheduled: the tab
        is gone, so a code nobody can read must not keep a timer alive polling
        Boske into a window that no longer exists. Safe to call twice.
        """
        self.cancel_link()
        self._cancel_timer()


# -- AppKit ------------------------------------------------------------------

SECTION_LICENCE = "Your licence"
SECTION_OWN_KEY = "Own key"
SECTION_ABOUT = "Version and updates"

SIGN_IN_BUTTON = "Sign in with Boske ID"
SIGN_OUT_BUTTON = "Sign out"
OPEN_BUTTON = "Open"
CANCEL_BUTTON = "Cancel"
SAVE_KEY_BUTTON = "Save"
REMOVE_KEY_BUTTON = "Remove"
CHECK_UPDATES_BUTTON = "Check for Updates…"
CHANNEL_LABEL = "Update channel"

OWN_KEY_HINT = (
    "Your own API key is stored in the macOS Keychain, never in Murmur's "
    "settings file, and is only used when Cloud is set to Own key."
)
LICENCE_HINT = (
    "Signing in links this Mac to your Boske ID. No key to paste, and nothing "
    "is sent until you turn Cloud on."
)

NO_KEY_TITLE = "No key to save"
NO_KEY_BODY = "Type your {provider} API key into the field first."
KEY_SAVED_TITLE = "Key stored"
KEY_SAVED_BODY = "Your {provider} key is in the macOS Keychain."
KEYCHAIN_FAILED_TITLE = "The Keychain refused that"
KEYCHAIN_FAILED_BODY = "Murmur could not reach the macOS Keychain.\n\n{detail}"
NO_LICENCE_TITLE = "Sign-in is not available yet"
NO_LICENCE_BODY = "This build has no licence service wired up."

#: ``TabContext.services`` keys for the version strings.
SERVICE_VERSION = "version"
SERVICE_BUILD_INFO = "build_info"

UNKNOWN_VERSION = "unknown"


def _resolved_theme(theme: Any) -> Any:
    """The context's theme, or ``ui_theme`` imported on demand."""
    if theme is not None:
        return theme
    import ui_theme

    return ui_theme


def _license_callables(service: Any) -> dict[str, Callable]:
    """The four licence callables, or inert stands-in when there is no service.

    Wave 4 owns ``services/license_service.py``; until it is wired up (and in
    any build without it) the tab still has to open and still has to say
    "Free" rather than raise.
    """
    if service is None:
        return {
            "license_provider": lambda: None,
            "link_starter": _no_license_service,
            "link_poller": lambda: False,
            "sign_out": lambda: None,
        }
    return {
        "license_provider": service.current_entitlements,
        "link_starter": service.begin_link,
        "link_poller": service.poll_link,
        "sign_out": service.sign_out,
    }


def _no_license_service() -> Any:
    raise RuntimeError("no licence service is wired up in this build")


def _make_secure_field(theme: Any, width: int = 260) -> Any:
    """An ``NSSecureTextField``: the key is dots on screen and never read back."""
    from Cocoa import NSMakeRect, NSSecureTextField

    field = NSSecureTextField.alloc().initWithFrame_(NSMakeRect(0, 0, width, 24))
    field.setPlaceholderString_("API key")
    field.setAppearance_(theme.control_appearance())
    return field


@register_tab
class AccountTab:
    """The AppKit rendering of :class:`AccountTabModel`.

    Like the other tabs there is no Save button: the update channel is written
    the moment it changes, and the two key buttons act immediately. The key
    field is cleared as soon as the key is in the Keychain, so a stored secret
    is never left sitting on screen.
    """

    identifier = TAB_ACCOUNT
    title = "Account"

    def __init__(self) -> None:
        self.context: TabContext | None = None
        self.model: AccountTabModel | None = None
        self._view = None
        self._status_label = None
        self._detail_label = None
        self._link_label = None
        self._sign_in_button = None
        self._sign_out_button = None
        self._open_button = None
        self._cancel_button = None
        self._channel_popup = None
        self._key_fields: dict[str, Any] = {}
        self._key_indicators: dict[str, Any] = {}
        self._key_buttons: dict[str, tuple[Any, ...]] = {}
        self._dispatch: Callable[[Callable[[], None]], None] | None = None

    # -- building --------------------------------------------------------

    def build(self, context: TabContext) -> Any:
        """Lay the tab out and return its content view."""
        assert context is not None, "context is required"
        from Cocoa import NSMakeRect, NSView

        from ui.download_sheet import main_thread_dispatcher
        from ui.settings.base import (
            CONTENT_MARGIN,
            make_button,
            make_hint,
            make_label,
            make_popup,
            make_section_title,
            stack_horizontal,
            stack_vertical,
        )

        self.context = context
        self._dispatch = main_thread_dispatcher()
        theme = _resolved_theme(context.theme)
        self.model = self._make_model(context)

        self._status_label = make_label(self.model.status_line(), theme, size=13, bold=True)
        self._detail_label = make_label(self.model.detail_line() or " ", theme)
        self._link_label = make_label(self.model.link_line() or " ", theme)
        self._sign_in_button = make_button(SIGN_IN_BUTTON, theme, self._sign_in_clicked, primary=True)
        self._sign_out_button = make_button(SIGN_OUT_BUTTON, theme, self._sign_out_clicked, width=120)
        self._open_button = make_button(OPEN_BUTTON, theme, self._open_clicked, width=90)
        self._cancel_button = make_button(CANCEL_BUTTON, theme, self._cancel_clicked, width=90)

        rows: list[Any] = [
            make_section_title(SECTION_LICENCE, theme),
            self._status_label,
            self._detail_label,
            stack_horizontal([self._sign_in_button, self._sign_out_button]),
            self._link_label,
            stack_horizontal([self._open_button, self._cancel_button]),
            make_hint(LICENCE_HINT, theme),
            make_section_title(SECTION_OWN_KEY, theme),
        ]
        for provider in BYOK_PROVIDERS:
            rows.extend(self._key_rows(provider, theme))
        rows.append(make_hint(OWN_KEY_HINT, theme))

        self._channel_popup = make_popup(
            list(self.model.channel_labels()),
            self.model.channel_index(),
            theme,
            self._channel_changed,
        )
        rows.extend(
            [
                make_section_title(SECTION_ABOUT, theme),
                make_label(self.model.version_line(), theme),
                stack_horizontal([make_label(CHANNEL_LABEL, theme), self._channel_popup]),
                make_button(CHECK_UPDATES_BUTTON, theme, self._check_updates_clicked),
            ]
        )

        container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 480, 560))
        stack = stack_vertical(rows)
        container.addSubview_(stack)
        stack.leadingAnchor().constraintEqualToAnchor_constant_(
            container.leadingAnchor(), CONTENT_MARGIN
        ).setActive_(True)
        stack.topAnchor().constraintEqualToAnchor_constant_(
            container.topAnchor(), CONTENT_MARGIN
        ).setActive_(True)
        self._view = container
        self.refresh()
        return container

    def _make_model(self, context: TabContext) -> AccountTabModel:
        """Build the model from whatever the window injected."""
        from services.keychain import KeychainStore
        from services.update_service import read_build_info

        build_info = context.service(SERVICE_BUILD_INFO)
        if build_info is None:
            build_info = read_build_info()
        version = context.service(SERVICE_VERSION) or build_info.get("version") or UNKNOWN_VERSION
        return AccountTabModel(
            context.config,
            keychain=context.service(SERVICE_KEYCHAIN) or KeychainStore(),
            version=str(version),
            build_info=build_info,
            scheduler=context.service(SERVICE_SCHEDULER) or self._schedule,
            **_license_callables(context.service(SERVICE_LICENSE)),
        )

    def _key_rows(self, provider: str, theme: Any) -> list[Any]:
        """Label, secure field, Save, Remove and the stored/not-stored line."""
        from ui.settings.base import make_button, make_label, stack_horizontal

        field = _make_secure_field(theme)
        indicator = make_label(self.model.key_indicator(provider), theme)
        save = make_button(SAVE_KEY_BUTTON, theme, self._save_key_action(provider), width=80)
        remove = make_button(
            REMOVE_KEY_BUTTON, theme, self._remove_key_action(provider), width=90
        )
        self._key_fields[provider] = field
        self._key_indicators[provider] = indicator
        self._key_buttons[provider] = (save, remove)
        return [
            make_label(BYOK_PROVIDER_LABELS[provider], theme, bold=True),
            stack_horizontal([field, save, remove]),
            indicator,
        ]

    # -- refreshing ------------------------------------------------------

    def refresh(self) -> None:
        """Re-read the model into the controls. Safe to call from any thread."""
        if self.model is None or self._status_label is None:
            return
        model = self.model
        self._status_label.setStringValue_(model.status_line())
        self._detail_label.setStringValue_(model.detail_line() or " ")
        self._link_label.setStringValue_(model.link_line() or " ")
        self._sign_in_button.setEnabled_(not model.is_linking)
        self._sign_out_button.setEnabled_(model.is_signed_in)
        self._open_button.setEnabled_(model.is_linking and bool(model.verification_url))
        self._cancel_button.setEnabled_(model.is_linking)
        for provider, indicator in self._key_indicators.items():
            state = model.key_state(provider)
            indicator.setStringValue_(KEY_STATE_LABELS[state])
            usable = state != KEY_STATE_UNAVAILABLE
            for button in self._key_buttons.get(provider, ()):
                button.setEnabled_(usable)
            field = self._key_fields.get(provider)
            if field is not None:
                field.setEnabled_(usable)
        if self._channel_popup is not None:
            self._channel_popup.selectItemAtIndex_(model.channel_index())

    def _refresh_later(self) -> None:
        """Refresh from a timer thread, on the main thread."""
        if self._dispatch is None:
            self.refresh()
            return
        self._dispatch(self.refresh)

    def _schedule(self, delay_s: float, callback: Callable[[], None]) -> Any:
        """Default scheduler: poll off the main thread, then redraw on it."""

        def run() -> None:
            callback()
            self._refresh_later()

        return _start_timer(delay_s, run)

    # -- actions ---------------------------------------------------------

    def _sign_in_clicked(self, _sender) -> None:
        import ui_alerts

        if self.context.service(SERVICE_LICENSE) is None:
            ui_alerts.show_alert(NO_LICENCE_TITLE, NO_LICENCE_BODY)
            return
        self.model.begin_link()
        self.refresh()

    def _sign_out_clicked(self, _sender) -> None:
        self.model.sign_out()
        self.refresh()

    def _cancel_clicked(self, _sender) -> None:
        self.model.cancel_link()
        self.refresh()

    def _open_clicked(self, _sender) -> None:
        """Open the verification page in the browser."""
        url = self.model.verification_url
        if not url:
            return
        from Cocoa import NSURL, NSWorkspace

        NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url))

    def _save_key_action(self, provider: str) -> Callable[[Any], None]:
        def action(_sender) -> None:
            self._save_key(provider)

        return action

    def _remove_key_action(self, provider: str) -> Callable[[Any], None]:
        def action(_sender) -> None:
            self._remove_key(provider)

        return action

    def _save_key(self, provider: str) -> None:
        import ui_alerts

        from services.keychain import KeychainError

        field = self._key_fields[provider]
        label = BYOK_PROVIDER_LABELS[provider]
        try:
            self.model.save_key(provider, str(field.stringValue() or ""))
        except ValueError:
            ui_alerts.show_alert(NO_KEY_TITLE, NO_KEY_BODY.format(provider=label))
            return
        except KeychainError as error:
            # ``error`` carries an OSStatus and an item name, never the key.
            logger.error("Could not store the %s key: %s", provider, error)
            ui_alerts.show_alert(
                KEYCHAIN_FAILED_TITLE, KEYCHAIN_FAILED_BODY.format(detail=error)
            )
            return
        field.setStringValue_("")  # never leave a secret on screen
        self.refresh()
        ui_alerts.show_alert(KEY_SAVED_TITLE, KEY_SAVED_BODY.format(provider=label))

    def _remove_key(self, provider: str) -> None:
        import ui_alerts

        from services.keychain import KeychainError

        try:
            self.model.remove_key(provider)
        except KeychainError as error:
            logger.error("Could not remove the %s key: %s", provider, error)
            ui_alerts.show_alert(
                KEYCHAIN_FAILED_TITLE, KEYCHAIN_FAILED_BODY.format(detail=error)
            )
            return
        self._key_fields[provider].setStringValue_("")
        self.refresh()

    def _channel_changed(self, sender) -> None:
        self.model.set_channel_index(int(sender.indexOfSelectedItem()))
        changed = self.model.apply()
        if changed:
            self.context.save(changed)
            self.model.mark_saved()

    def _check_updates_clicked(self, _sender) -> None:
        """Ask the running app to check. ``check_updates`` is a rumps callback."""
        self.context.app_call("check_updates", None)

    # -- closing ---------------------------------------------------------

    def close(self) -> None:
        """Stop the device-link poll when the window goes away.

        Without this the model's timer re-arms itself forever, polling Boske
        and refreshing controls belonging to a window that has closed.
        """
        if self.model is not None:
            self.model.close()


__all__ = [
    "BYOK_PROVIDERS",
    "BYOK_PROVIDER_LABELS",
    "CHANNEL_BETA",
    "CHANNEL_LABELS",
    "CHANNEL_STABLE",
    "CONFIG_UPDATE_CHANNEL",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_UPDATE_CHANNEL",
    "KEY_NOT_STORED",
    "KEY_STATE_ABSENT",
    "KEY_STATE_LABELS",
    "KEY_STATE_STORED",
    "KEY_STATE_UNAVAILABLE",
    "KEY_STORED",
    "KEY_UNAVAILABLE",
    "LINK_CANCELLED",
    "LINK_DENIED",
    "LINK_EXPIRED",
    "LINK_FAILED",
    "LINK_IDLE",
    "LINK_PENDING",
    "LINK_STATES",
    "LINK_SUCCESS",
    "SERVICE_BUILD_INFO",
    "SERVICE_KEYCHAIN",
    "SERVICE_LICENSE",
    "SERVICE_SCHEDULER",
    "SERVICE_VERSION",
    "STATUS_FREE",
    "UPDATE_CHANNELS",
    "AccountTab",
    "AccountTabModel",
    "as_timestamp",
    "format_expiry",
    "normalised_channel",
]