"""Tests for the Settings "Account" tab state (Wave 3, E3e).

Everything here is the pure model: no AppKit, no network, no keychain. The
licence service, the two link callables, sign-out, the secret store, the
scheduler and the clock are all injected, so the whole device-linking flow runs
in-process and deterministically.

The invariant these tests exist to hold: an own key goes to the secret store
and nowhere else — not into the config dict, not into any line the tab shows.
"""

import unittest
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from services.keychain import ITEM_BYOK_MISTRAL, ITEM_BYOK_OPENAI, InMemorySecretStore
from ui.settings.base import TAB_ACCOUNT
from ui.settings.account_tab import (
    CHANNEL_BETA,
    CHANNEL_STABLE,
    CONFIG_UPDATE_CHANNEL,
    DEFAULT_POLL_INTERVAL_S,
    KEY_NOT_STORED,
    KEY_STORED,
    LINK_CANCELLED,
    LINK_EXPIRED,
    LINK_FAILED,
    LINK_IDLE,
    LINK_PENDING,
    LINK_SUCCESS,
    STATUS_FREE,
    STATUS_UNAVAILABLE,
    AccountTab,
    AccountTabModel,
    as_timestamp,
    format_expiry,
    normalised_channel,
)

SECRET = "sk-live-account-tab-should-not-echo-me"
VERSION = "1.2.3"


@dataclass
class FakeEntitlements:
    """Shaped like Wave 4's ``Entitlements``."""

    pro: bool = False
    cloud_voice: bool = False
    msm_minutes: int = 0
    expires_at: Any = None
    in_grace: bool = False
    source: str = ""


@dataclass
class FakeLinkSession:
    """Shaped like Wave 4's ``LinkSession``."""

    user_code: str = "WDJB-MJHT"
    verification_url: str = "https://boske.app/link"
    interval_s: float = 5.0
    expires_at: Any = 1_000.0


class FakeTimer:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeScheduler:
    """Records what would have been scheduled; ``fire()`` runs it."""

    def __init__(self) -> None:
        self.pending: list[tuple[float, Any]] = []
        self.timers: list[FakeTimer] = []

    def __call__(self, delay_s: float, callback) -> FakeTimer:
        timer = FakeTimer()
        self.pending.append((delay_s, callback))
        self.timers.append(timer)
        return timer

    def fire(self) -> None:
        assert self.pending, "nothing was scheduled"
        _delay, callback = self.pending.pop()
        callback()

    @property
    def delays(self) -> list[float]:
        return [delay for delay, _callback in self.pending]

    @property
    def cancelled(self) -> int:
        return sum(1 for timer in self.timers if timer.cancelled)


class ModelHarness:
    """One model plus the fakes behind it, so tests read as one line of setup."""

    def __init__(self, config=None, entitlements=None, **overrides):
        self.config = dict(config or {})
        self.entitlements = entitlements
        self.session = FakeLinkSession()
        self.linked = False
        self.sign_outs = 0
        self.poll_calls = 0
        self.start_error: Exception | None = None
        self.poll_error: Exception | None = None
        self.license_error: Exception | None = None
        self.keychain = InMemorySecretStore()
        self.scheduler = FakeScheduler()
        self.clock = 0.0
        self.model = AccountTabModel(
            self.config,
            license_provider=self._license,
            link_starter=self._start,
            link_poller=self._poll,
            sign_out=self._sign_out,
            keychain=self.keychain,
            version=overrides.pop("version", VERSION),
            build_info=overrides.pop("build_info", {}),
            scheduler=self.scheduler,
            now=lambda: self.clock,
            **overrides,
        )

    def _license(self):
        if self.license_error is not None:
            raise self.license_error
        return self.entitlements

    def _start(self):
        if self.start_error is not None:
            raise self.start_error
        return self.session

    def _poll(self) -> bool:
        self.poll_calls += 1
        if self.poll_error is not None:
            raise self.poll_error
        return self.linked

    def _sign_out(self) -> None:
        self.sign_outs += 1
        self.entitlements = None


class StatusLineTest(unittest.TestCase):
    def test_no_entitlements_is_free(self):
        self.assertEqual(STATUS_FREE, ModelHarness().model.status_line())

    def test_entitlements_without_pro_is_free(self):
        harness = ModelHarness(entitlements=FakeEntitlements(pro=False, source="lease"))
        self.assertEqual(STATUS_FREE, harness.model.status_line())

    def test_pro_names_the_expiry_date(self):
        harness = ModelHarness(
            entitlements=FakeEntitlements(pro=True, expires_at="2027-04-03T10:00:00", source="lease")
        )
        self.assertEqual("Pro until 3 April 2027", harness.model.status_line())

    def test_pro_without_a_date_says_pro(self):
        harness = ModelHarness(entitlements=FakeEntitlements(pro=True, source="lease"))
        self.assertEqual("Pro", harness.model.status_line())

    def test_grace_says_renew_by(self):
        harness = ModelHarness(
            entitlements=FakeEntitlements(
                pro=True, in_grace=True, expires_at=date(2027, 4, 3), source="lease"
            )
        )
        self.assertEqual("Pro (grace, renew by 3 April 2027)", harness.model.status_line())

    def test_grace_without_a_date(self):
        harness = ModelHarness(
            entitlements=FakeEntitlements(pro=True, in_grace=True, source="lease")
        )
        self.assertEqual("Pro (grace)", harness.model.status_line())

    def test_licence_provider_failure_is_reported_not_raised(self):
        harness = ModelHarness()
        harness.license_error = RuntimeError("proxy unreachable")
        with self.assertLogs("ui.settings.account_tab", level="WARNING"):
            harness.model.refresh_entitlements()
        self.assertEqual(STATUS_UNAVAILABLE, harness.model.status_line())
        self.assertFalse(harness.model.license_available)

    def test_cloud_voice_detail_lines(self):
        plain = ModelHarness(entitlements=FakeEntitlements(pro=True, cloud_voice=True))
        self.assertEqual("Cloud voice included", plain.model.detail_line())

        metered = ModelHarness(
            entitlements=FakeEntitlements(pro=True, cloud_voice=True, msm_minutes=300)
        )
        self.assertEqual("Cloud voice included — 300 minutes a month", metered.model.detail_line())

        self.assertEqual("", ModelHarness().model.detail_line())

    def test_signed_in_follows_the_lease_source(self):
        self.assertFalse(ModelHarness().model.is_signed_in)
        harness = ModelHarness(entitlements=FakeEntitlements(source="lease"))
        self.assertTrue(harness.model.is_signed_in)


class ExpiryFormattingTest(unittest.TestCase):
    def test_accepts_the_shapes_a_lease_can_carry(self):
        self.assertEqual("3 April 2027", format_expiry(datetime(2027, 4, 3, 9, 30)))
        self.assertEqual("3 April 2027", format_expiry(date(2027, 4, 3)))
        self.assertEqual("3 April 2027", format_expiry("2027-04-03"))
        self.assertEqual("3 April 2027", format_expiry("2027-04-03T09:30:00Z"))
        self.assertEqual("3 April 2027", format_expiry(datetime(2027, 4, 3).timestamp()))

    def test_unreadable_dates_become_empty(self):
        for value in (None, "", "soon", object(), True):
            with self.subTest(value=value):
                self.assertEqual("", format_expiry(value))

    def test_as_timestamp_is_none_when_there_is_no_deadline(self):
        self.assertIsNone(as_timestamp(None))
        self.assertIsNone(as_timestamp("whenever"))
        self.assertEqual(1_000.0, as_timestamp(1_000))


class LinkFlowTest(unittest.TestCase):
    def setUp(self):
        self.harness = ModelHarness()
        self.model = self.harness.model

    def test_begin_shows_the_code_and_schedules_the_first_poll(self):
        session = self.model.begin_link()
        self.assertIs(self.harness.session, session)
        self.assertEqual(LINK_PENDING, self.model.link_state)
        self.assertEqual("WDJB-MJHT", self.model.user_code)
        self.assertEqual("https://boske.app/link", self.model.verification_url)
        self.assertEqual([5.0], self.harness.scheduler.delays)
        self.assertIn("WDJB-MJHT", self.model.link_line())
        self.assertIn("https://boske.app/link", self.model.link_line())

    def test_a_poll_that_finds_nothing_keeps_waiting(self):
        self.model.begin_link()
        self.harness.scheduler.fire()
        self.assertEqual(LINK_PENDING, self.model.link_state)
        self.assertEqual(1, self.harness.poll_calls)
        self.assertEqual([5.0], self.harness.scheduler.delays)  # rescheduled

    def test_success_stops_polling_and_re_reads_the_licence(self):
        self.model.begin_link()
        self.harness.linked = True
        self.harness.entitlements = FakeEntitlements(pro=True, source="lease")
        self.harness.scheduler.fire()
        self.assertEqual(LINK_SUCCESS, self.model.link_state)
        self.assertEqual("Pro", self.model.status_line())
        self.assertEqual("Signed in with your Boske ID", self.model.link_line())
        self.assertEqual([], self.harness.scheduler.delays)
        self.assertEqual(1, self.harness.scheduler.cancelled)

    def test_expiry_ends_the_attempt(self):
        self.model.begin_link()
        self.harness.clock = 2_000.0  # past the session's expires_at
        self.harness.scheduler.fire()
        self.assertEqual(LINK_EXPIRED, self.model.link_state)
        self.assertEqual("", self.model.user_code)
        self.assertEqual([], self.harness.scheduler.delays)
        self.assertIn("expired", self.model.link_line())

    def test_a_session_without_a_deadline_never_expires_locally(self):
        self.harness.session = FakeLinkSession(expires_at=None)
        self.model.begin_link()
        self.harness.clock = 10_000_000.0
        self.harness.scheduler.fire()
        self.assertEqual(LINK_PENDING, self.model.link_state)

    def test_cancel_stops_the_timer_and_later_polls_do_nothing(self):
        self.model.begin_link()
        self.assertEqual(LINK_CANCELLED, self.model.cancel_link())
        self.assertEqual(1, self.harness.scheduler.cancelled)
        self.assertEqual("Sign-in cancelled.", self.model.link_line())

        self.harness.scheduler.fire()  # a timer that fired anyway
        self.assertEqual(LINK_CANCELLED, self.model.link_state)
        self.assertEqual(0, self.harness.poll_calls)

    def test_cancel_when_nothing_is_pending_changes_nothing(self):
        self.assertEqual(LINK_IDLE, self.model.cancel_link())

    def test_a_failing_starter_shows_a_failed_link(self):
        self.harness.start_error = RuntimeError("no network")
        with self.assertLogs("ui.settings.account_tab", level="WARNING"):
            self.assertIsNone(self.model.begin_link())
        self.assertEqual(LINK_FAILED, self.model.link_state)
        self.assertEqual([], self.harness.scheduler.delays)

    def test_a_failing_poller_ends_the_attempt(self):
        self.model.begin_link()
        self.harness.poll_error = RuntimeError("gateway timeout")
        with self.assertLogs("ui.settings.account_tab", level="WARNING"):
            self.harness.scheduler.fire()
        self.assertEqual(LINK_FAILED, self.model.link_state)
        self.assertEqual(1, self.harness.scheduler.cancelled)

    def test_restarting_cancels_the_previous_code(self):
        self.model.begin_link()
        self.model.begin_link()
        self.assertEqual(1, self.harness.scheduler.cancelled)
        self.assertEqual(LINK_PENDING, self.model.link_state)

    def test_interval_falls_back_when_the_session_omits_it(self):
        self.harness.session = FakeLinkSession(interval_s=0)
        self.model.begin_link()
        self.assertEqual([DEFAULT_POLL_INTERVAL_S], self.harness.scheduler.delays)

    def test_sign_out_forgets_the_lease_and_stops_polling(self):
        harness = ModelHarness(entitlements=FakeEntitlements(pro=True, source="lease"))
        harness.model.begin_link()
        harness.model.sign_out()
        self.assertEqual(1, harness.sign_outs)
        self.assertEqual(STATUS_FREE, harness.model.status_line())
        self.assertEqual(LINK_IDLE, harness.model.link_state)
        self.assertEqual(1, harness.scheduler.cancelled)

    def test_close_cancels_a_pending_poll(self):
        self.model.begin_link()
        self.model.close()
        self.assertEqual(1, self.harness.scheduler.cancelled)


class OwnKeyTest(unittest.TestCase):
    def setUp(self):
        self.harness = ModelHarness()
        self.model = self.harness.model

    def test_indicator_reports_presence_only(self):
        self.assertEqual(KEY_NOT_STORED, self.model.key_indicator("mistral"))
        self.model.save_key("mistral", SECRET)
        self.assertEqual(KEY_STORED, self.model.key_indicator("mistral"))
        self.assertNotIn(SECRET, self.model.key_indicator("mistral"))

    def test_keys_land_on_the_documented_keychain_items(self):
        self.model.save_key("mistral", SECRET)
        self.model.save_key("openai", "sk-openai-other")
        self.assertEqual(SECRET, self.harness.keychain.get(ITEM_BYOK_MISTRAL))
        self.assertEqual("sk-openai-other", self.harness.keychain.get(ITEM_BYOK_OPENAI))

    def test_no_secret_ever_reaches_the_config(self):
        self.model.save_key("openai", SECRET)
        self.assertNotIn(SECRET, repr(self.harness.config))
        self.assertEqual({}, self.model.apply())
        for line in (self.model.status_line(), self.model.detail_line(), self.model.link_line()):
            self.assertNotIn(SECRET, line)

    def test_whitespace_is_trimmed_and_an_empty_key_is_refused(self):
        self.model.save_key("mistral", f"  {SECRET}\n")
        self.assertEqual(SECRET, self.harness.keychain.get(ITEM_BYOK_MISTRAL))
        with self.assertRaises(ValueError):
            self.model.save_key("openai", "   ")
        self.assertFalse(self.model.key_stored("openai"))

    def test_remove_deletes_the_item(self):
        self.model.save_key("mistral", SECRET)
        self.model.remove_key("mistral")
        self.assertFalse(self.model.key_stored("mistral"))
        self.model.remove_key("mistral")  # idempotent

    def test_unknown_provider_is_a_programming_error(self):
        with self.assertRaises(AssertionError):
            self.model.save_key("anthropic", SECRET)

    def test_works_with_a_store_that_has_only_get_set_delete(self):
        """Wave 4's ``SecretStore`` protocol is the minimum this tab needs."""

        class MinimalStore:
            def __init__(self):
                self.values = {}

            def get(self, name):
                return self.values.get(name)

            def set(self, name, value):
                self.values[name] = value

            def delete(self, name):
                self.values.pop(name, None)

        harness = ModelHarness()
        harness.model.keychain = MinimalStore()
        self.assertFalse(harness.model.key_stored("mistral"))
        harness.model.save_key("mistral", SECRET)
        self.assertTrue(harness.model.key_stored("mistral"))
        self.assertEqual(KEY_STORED, harness.model.key_indicator("mistral"))

    def test_storing_a_key_logs_the_provider_and_never_the_key(self):
        with self.assertLogs("ui.settings.account_tab", level="INFO") as captured:
            self.model.save_key("mistral", SECRET)
            self.model.remove_key("mistral")
        written = " ".join(captured.output)
        self.assertNotIn(SECRET, written)
        self.assertIn("mistral", written)

    def test_sign_out_leaves_the_own_keys_alone(self):
        self.model.save_key("mistral", SECRET)
        self.model.sign_out()
        self.assertTrue(self.model.key_stored("mistral"))


class VersionAndChannelTest(unittest.TestCase):
    def test_version_line_marks_an_internal_build(self):
        plain = ModelHarness(build_info={"signed": True})
        self.assertEqual("Murmur 1.2.3", plain.model.version_line())

        internal = ModelHarness(build_info={"signed": False})
        self.assertEqual("Murmur 1.2.3 · internal build", internal.model.version_line())

        source_run = ModelHarness(build_info={})
        self.assertEqual("Murmur 1.2.3", source_run.model.version_line())

    def test_channel_defaults_to_stable(self):
        model = ModelHarness().model
        self.assertEqual(CHANNEL_STABLE, model.update_channel)
        self.assertEqual(0, model.channel_index())
        self.assertEqual({}, model.apply())

    def test_an_unknown_channel_on_disk_reads_as_stable(self):
        model = ModelHarness(config={CONFIG_UPDATE_CHANNEL: "nightly"}).model
        self.assertEqual(CHANNEL_STABLE, model.update_channel)
        self.assertEqual(CHANNEL_STABLE, normalised_channel(None))
        self.assertEqual(CHANNEL_BETA, normalised_channel(" BETA "))

    def test_apply_reports_only_the_changed_channel(self):
        harness = ModelHarness(config={CONFIG_UPDATE_CHANNEL: CHANNEL_STABLE, "language": "fr"})
        model = harness.model
        model.set_update_channel(CHANNEL_BETA)
        self.assertEqual({CONFIG_UPDATE_CHANNEL: CHANNEL_BETA}, model.apply())
        model.mark_saved()
        self.assertEqual({}, model.apply())
        model.set_update_channel(CHANNEL_STABLE)
        self.assertEqual({CONFIG_UPDATE_CHANNEL: CHANNEL_STABLE}, model.apply())

    def test_channel_index_round_trips(self):
        model = ModelHarness().model
        model.set_channel_index(1)
        self.assertEqual(CHANNEL_BETA, model.update_channel)
        self.assertEqual(1, model.channel_index())
        with self.assertRaises(AssertionError):
            model.set_channel_index(7)
        with self.assertRaises(AssertionError):
            model.set_update_channel("nightly")

    def test_a_beta_channel_already_on_disk_is_kept(self):
        model = ModelHarness(config={CONFIG_UPDATE_CHANNEL: CHANNEL_BETA}).model
        self.assertEqual(CHANNEL_BETA, model.update_channel)
        self.assertEqual({}, model.apply())


class TabRegistrationTest(unittest.TestCase):
    def test_the_tab_registers_itself_under_the_account_identifier(self):
        import ui.settings

        self.assertEqual(TAB_ACCOUNT, AccountTab.identifier)
        self.assertEqual("Account", AccountTab.title)
        self.assertIn(AccountTab, ui.settings.TABS)

    def test_the_model_refuses_to_build_without_its_dependencies(self):
        with self.assertRaises(AssertionError):
            AccountTabModel(
                {},
                license_provider=lambda: None,
                link_starter=lambda: None,
                link_poller=lambda: False,
                sign_out=lambda: None,
                keychain=None,
                version=VERSION,
                build_info={},
            )


if __name__ == "__main__":
    unittest.main()
