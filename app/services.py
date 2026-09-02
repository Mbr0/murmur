"""Composing the licence, the usage counters and the Keychain, once, at startup.

The half of :class:`~app.lifecycle.MurmurApp` that owns the Wave 4 services:
what they are built from, how the lease is renewed in the background, and the
one dict the Settings tabs are allowed to reach the app through.

No AppKit at import. The one alert this module can raise goes through
:mod:`ui.alerts`, which loads its own.
"""

import threading
import time

from engines.factory import cloud_base_url
from services.keychain import KeychainStore
from services.license_service import (
    InMemorySecretStore,
    LicenseService,
    is_pro_feature_enabled,
)
from services.persistence_service import DEFAULT_CONFIG
from services.update_service import read_build_info
from services.usage_service import USAGE_DEFAULTS, UsageService
from ui import alerts as ui_alerts

from app.config import APP_NAME, APP_VERSION, AUDIO_DIR, logger
from app.decisions import (
    ENTITLEMENT_POLL_INTERVAL_S,
    LaunchAtLoginUnavailable,
    apply_launch_at_login,
    boske_http_transport,
    launch_at_login_enabled,
    next_refresh_delay,
    pinned_cloud_config,
    publish_entitlements,
    should_refresh_entitlements,
)


class UsageConfigStore:
    """The ``load()``/``save()`` pair :class:`UsageService` counts through.

    ``load`` is the whole config, because the service reads its own ten keys out
    of it. ``save`` writes **only** those ten back, through
    :meth:`~services.persistence_service.PersistenceService.update_config`: the
    service hands back the config it loaded, and saving that snapshot verbatim
    would revert every key Settings or the engine wrote while a dictation was in
    flight. This adapter is the difference between counting minutes and losing a
    model switch.
    """

    #: The keys this store is allowed to write. Exactly the service's own.
    KEYS: tuple[str, ...] = tuple(USAGE_DEFAULTS)

    def __init__(self, persistence) -> None:
        assert persistence is not None, "persistence is required"
        self._persistence = persistence

    def load(self) -> dict:
        return self._persistence.load_config(dict(DEFAULT_CONFIG))

    def save(self, config: dict) -> None:
        assert config is not None, "config is required"
        self._persistence.update_config(
            {key: config[key] for key in self.KEYS if key in config}
        )


class ServicesMixin:
    """The licence, the counters and the hosted clients, mixed into the app.

    Split out of ``MurmurApp`` in Wave 5. Every method still reads and writes
    ``self``; what changed is which file it is read in.
    """

    def _keychain(self):
        """The secret store for the Account tab, or ``None`` when unreachable.

        Resolved once. The Security binding is what can be missing — off macOS,
        or in a stripped build — and asking again on every window open would
        only repeat the same failure and the same log line.

        Every failure is caught, not only :class:`KeychainUnavailable`. The
        ctypes backend loads a library and reads symbols out of it, so a broken
        one raises ``OSError``, ``ValueError`` or a plain ``KeychainError`` just
        as readily — and anything that escapes here reaches
        :meth:`open_settings_window_safely` as "Could not open Settings", which
        is a whole window lost over one optional feature. The type is logged so
        the log still says which failure it was.
        """
        if self._keychain_probed:
            return self._keychain_store
        self._keychain_probed = True
        try:
            store = KeychainStore()
            store.backend  # resolve now, so an unavailable keychain is known here
        except Exception as error:  # noqa: BLE001 - the backend raises widely
            logger.warning(
                "The keychain is unavailable (%s); own keys cannot be stored: %s",
                type(error).__name__,
                error,
            )
            self._keychain_store = None
        else:
            self._keychain_store = store
        return self._keychain_store

    # -- Wave 4: licence, usage and the hosted clients --------------------

    def _build_services(self):
        """Compose the licence and usage services once, at startup.

        Both are cheap to build and neither touches the network here: the
        licence service only reads a stored lease when it is asked, and the
        usage service only reads config. The proxy origin is read once and kept
        on ``self`` so the licence service, the cloud engine and cloud cleanup
        all speak to the same host — a config edited afterwards changes the
        engines on the next dictation, but never splits the lease from the host
        it was issued for mid-session.

        Nothing here may raise: a mistyped ``cloud_base_url`` or a locked
        Keychain must cost the cloud features, not the menu bar.
        """
        config = self.runtime_config()
        self.cloud_base_url = cloud_base_url(config)
        #: True when the lease could not be stored — see
        #: :meth:`_build_license_service`. Read by the menu and the Account tab.
        self.secret_store_is_volatile = False
        #: Said once a session, not once a dictation. See :meth:`_hosted_config`.
        self._base_url_drift_logged = False
        self.usage = UsageService(config_store=UsageConfigStore(self.persistence))
        self.license_service = self._build_license_service(self.cloud_base_url)
        #: The hosted engine in use, with the config it was built from.
        self._remote_engine = None
        self._remote_engine_key = None
        self._remote_engine_lock = threading.Lock()
        #: The cloud cleanup client, cached the same way and for the same reason.
        self._cloud_cleanup_client = None
        self._cloud_cleanup_base_url = None
        #: Read and written only by the entitlement thread.
        self._entitlements_refreshed_at = None
        #: Consecutive failed renewals, and when the next one may be tried.
        self._entitlement_failures = 0
        self._entitlements_retry_at = None
        #: Notice kinds already shown this session. See :meth:`_announce_route`.
        self._session_notices = set()
        self.account_item = None
        # Published here, synchronously, before anything asks the gate: reading
        # the stored lease is a Keychain read and a signature check, no network,
        # and every decision taken during startup — the cleanup pre-warm most of
        # all — would otherwise race the background thread and see the free tier
        # on a paying Mac.
        publish_entitlements(self.license_service)

    def _build_license_service(self, base_url):
        """The licence service, or ``None`` when this build cannot have one.

        The secret store is the Keychain when it is reachable and an in-memory
        one when it is not: a lease that cannot be persisted is worth nothing
        after a quit, but it lets the Account tab open and the app run rather
        than raising out of ``__init__``.

        That substitution is recorded in ``secret_store_is_volatile`` and said
        out loud in both places the account is shown. Signing in, seeing "Pro",
        quitting and being Free again is the kind of thing a user reports as a
        billing failure; a Mac that cannot keep the lease has to say so before
        the sign-in, not after the relaunch.
        """
        secret_store = self._keychain()
        self.secret_store_is_volatile = secret_store is None
        if secret_store is None:
            logger.warning(
                "No Keychain: the licence lease is kept in memory only and the "
                "sign-in will not survive a quit."
            )
            secret_store = InMemorySecretStore()
        try:
            return LicenseService(secret_store, boske_http_transport, base_url)
        except Exception as error:  # noqa: BLE001 - a bad base URL raises ValueError
            logger.error(
                "The licence service could not be built (%s); Pro and Murmur "
                "Cloud are unavailable this session: %s",
                type(error).__name__,
                error,
            )
            return None

    def _start_entitlement_refresh(self):
        """Publish entitlements now, then keep them fresh for the session."""
        threading.Thread(target=self._entitlement_worker, daemon=True).start()

    def _entitlement_worker(self):
        """Renew the lease and republish entitlements, at launch and every 6 h.

        A daemon thread that sleeps in short steps rather than for six hours, so
        a sign-in on the Account tab reaches the menu within the minute. Every
        failure is logged and slept off; the gate keeps whatever it last had,
        which for a failed first pass is the free tier.
        """
        while True:
            try:
                self._refresh_entitlements_once()
            except Exception as error:  # noqa: BLE001 - a daemon must not die
                logger.warning(
                    "The entitlement refresh failed: %s", type(error).__name__
                )
            time.sleep(ENTITLEMENT_POLL_INTERVAL_S)

    def _refresh_entitlements_once(self):
        """One pass of the loop above: renew when due, then publish and redraw.

        The clock is stamped on **success** only. Stamping it before the call
        meant a renewal that failed — no network at launch is the ordinary case —
        counted as done and was not tried again for six hours, so a paying Mac
        spent the afternoon on the free tier. A failure now backs off instead:
        five minutes, doubling to an hour. See :func:`next_refresh_delay`.
        """
        if self._entitlement_refresh_due():
            if self.license_service is None:
                self._entitlement_refresh_succeeded()
            else:
                try:
                    self.license_service.refresh_if_needed()
                except Exception as error:  # noqa: BLE001 - transport raises widely
                    self._entitlement_refresh_failed(error)
                else:
                    self._entitlement_refresh_succeeded()
        entitlements = publish_entitlements(self.license_service)
        self._refresh_account_menu(entitlements)

    def _entitlement_refresh_due(self):
        """Whether to attempt a renewal now: the interval, or a due retry."""
        now = time.time()
        retry_at = self._entitlements_retry_at
        if retry_at is not None:
            # A retry is always sooner than the interval, so it is the answer
            # whenever one is outstanding.
            return now >= retry_at
        return should_refresh_entitlements(
            last_refresh_at=self._entitlements_refreshed_at, now=now
        )

    def _entitlement_refresh_succeeded(self):
        """Stamp the clock and forget the backoff."""
        self._entitlements_refreshed_at = time.time()
        self._entitlement_failures = 0
        self._entitlements_retry_at = None

    def _entitlement_refresh_failed(self, error):
        """Leave the clock alone and schedule the next attempt."""
        self._entitlement_failures += 1
        delay = next_refresh_delay(self._entitlement_failures)
        self._entitlements_retry_at = time.time() + delay
        logger.warning(
            "Lease refresh failed (%s); retrying in %d s", type(error).__name__, int(delay)
        )

    def _hosted_config(self, config):
        """``config`` with the proxy origin pinned to the one built at launch.

        The only door to a hosted client, so every one of them — the lease, the
        audio, the cleanup text — reaches the host the licence service was
        built for. A ``cloud_base_url`` edited mid-session is honoured at the
        next launch and said once here, because saying it per dictation would
        bury it.
        """
        pinned = pinned_cloud_config(config, self.cloud_base_url)
        if pinned is not config and not getattr(self, "_base_url_drift_logged", False):
            self._base_url_drift_logged = True
            logger.warning(
                "cloud_base_url has changed since launch; this session keeps "
                "using the origin it signed in against. Restart Murmur to use "
                "the new one."
            )
        return pinned

    def _settings_services(self):
        """Everything the Settings tabs are allowed to reach the app through.

        One dict, documented in :mod:`ui.settings.window`. Every key is listed
        even when its value is ``None``: a tab asking for a provider that could
        not be built — no licence service on a build with an unusable proxy
        origin — is a supported state, and spelling it out here is what keeps
        that a visible decision rather than a missing key nobody notices.

        ``scheduler`` is deliberately ``None``. The Account tab's own default
        polls off the main thread *and* redraws on it; anything handed in here
        would replace both halves and leave the sign-in line stale.

        ``usage`` is the *callable* the Engine tab wants — it asks for a summary
        on every refresh — and ``license`` is the service itself, whose four
        methods the Account tab binds. ``pro_gate`` is
        :func:`~services.license_service.is_pro_feature_enabled` unadorned: it
        reads the published entitlements out of memory, so the tabs may ask it
        once per gated control without touching the disk.

        ``secret_store_volatile`` is the Keychain being unreachable. The Account
        tab is where a sign-in is offered, so it is where the app has to admit
        that the sign-in will not outlive the process.
        """
        return {
            "usage": self.usage.summary if self.usage is not None else None,
            "license": self.license_service,
            "pro_gate": is_pro_feature_enabled,
            "keychain": self._keychain(),
            "secret_store_volatile": bool(
                getattr(self, "secret_store_is_volatile", False)
            ),
            "scheduler": None,
            "version": APP_VERSION,
            "build_info": read_build_info(),
            "persistence": self.persistence,
            "audio_dir": AUDIO_DIR,
        }

    def set_launch_at_login(self, enabled: bool) -> bool:
        """Register or unregister Murmur as a login item. Returns the new state.

        Shadowed with ``None`` in :meth:`__init__` where ``SMAppService`` is not
        reachable, so Settings never offers the switch it could not honour; a
        service that disappears between then and now raises
        :class:`LaunchAtLoginUnavailable`. A refusal from the framework itself —
        the user has the item switched off in System Settings, say — is told to
        the user rather than raised into the click handler that got us here.
        """
        assert isinstance(enabled, bool), f"expected a bool, got {enabled!r}"
        service = self._login_item_service
        if service is None:
            raise LaunchAtLoginUnavailable(
                "ServiceManagement is not available in this build"
            )
        try:
            state = apply_launch_at_login(service, enabled)
        except LaunchAtLoginUnavailable:
            raise
        except Exception as error:  # noqa: BLE001 - the framework raises widely
            logger.error("Could not change the login item: %s", error)
            ui_alerts.show_alert(
                APP_NAME,
                "Murmur could not change whether it starts at login. "
                "You can set it in System Settings › General › Login Items.",
            )
            return launch_at_login_enabled(service)
        logger.info("Launch at login %s", "on" if state else "off")
        return state
