"""Tests for the monthly usage counters and the cloud soft limit.

Pure logic: a dict-backed fake config store and an injected clock. No AppKit,
no network, no files.
"""

import threading
import time
import unittest
from datetime import datetime, timedelta

from engines.cloud import Usage
from services.usage_service import (
    ALLOWANCE_TTL_S,
    KEY_ALLOWANCE_FETCHED_AT,
    KEY_ALLOWANCE_MINUTES,
    KEY_CLOUD_SECONDS,
    KEY_CLOUD_WORDS,
    KEY_LOCAL_SECONDS,
    KEY_LOCAL_WORDS,
    KEY_MONTH,
    KEY_NOTICE_SHOWN,
    KEY_REMOTE_MINUTES_USED,
    KEY_TRIAL_SECONDS,
    SOFT_LIMIT,
    TRIAL_TOTAL_SECONDS,
    USAGE_DEFAULTS,
    UsageService,
    UsageSummary,
)


class FakeStore:
    """Stands in for the app's persistence: ``load() -> dict`` and ``save(dict)``."""

    def __init__(self, initial=None):
        self.data = dict(initial or {})
        self.saves = 0

    def load(self):
        return dict(self.data)

    def save(self, config):
        assert isinstance(config, dict)
        self.saves += 1
        self.data = dict(config)


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, moment=datetime(2026, 9, 2, 10, 0, 0)):
        self.moment = moment

    def __call__(self):
        return self.moment


def make_service(initial=None, moment=None):
    store = FakeStore(initial)
    clock = FakeClock(moment) if moment else FakeClock()
    return UsageService(store, clock), store, clock


class RecordingTests(unittest.TestCase):
    def test_first_record_stamps_the_current_month(self):
        service, store, _ = make_service()

        service.record("cloud", seconds=30.0, words=90)

        self.assertEqual(store.data[KEY_MONTH], "2026-09")
        self.assertEqual(store.data[KEY_CLOUD_SECONDS], 30.0)
        self.assertEqual(store.data[KEY_CLOUD_WORDS], 90)

    def test_counters_accumulate_per_origin(self):
        service, store, _ = make_service()

        service.record("cloud", 30.0, 90)
        service.record("cloud", 10.5, 20)
        service.record("local", 5.0, 12)

        self.assertEqual(store.data[KEY_CLOUD_SECONDS], 40.5)
        self.assertEqual(store.data[KEY_CLOUD_WORDS], 110)
        self.assertEqual(store.data[KEY_LOCAL_SECONDS], 5.0)
        self.assertEqual(store.data[KEY_LOCAL_WORDS], 12)

    def test_byok_is_not_metered(self):
        # Own-key usage is billed by the user's own provider (E4c), and must
        # not push the Murmur Cloud counters towards the soft limit.
        service, store, _ = make_service()

        service.record("byok", 600.0, 1800)

        self.assertEqual(store.data[KEY_CLOUD_SECONDS], 0.0)
        self.assertEqual(store.data[KEY_LOCAL_SECONDS], 0.0)

    def test_an_unknown_origin_is_a_programming_error(self):
        service, _, _ = make_service()
        with self.assertRaises(AssertionError):
            service.record("azure", 1.0, 1)

    def test_negative_amounts_are_rejected(self):
        service, _, _ = make_service()
        with self.assertRaises(AssertionError):
            service.record("cloud", -1.0, 1)
        with self.assertRaises(AssertionError):
            service.record("cloud", 1.0, -1)

    def test_unrelated_config_keys_survive_a_write(self):
        service, store, _ = make_service({"hotkey_keycode": 49, "privacy_mode": True})

        service.record("local", 1.0, 2)

        self.assertEqual(store.data["hotkey_keycode"], 49)
        self.assertTrue(store.data["privacy_mode"])

    def test_defaults_cover_every_key_the_service_writes(self):
        service, store, _ = make_service()
        service.record("cloud", 1.0, 1)
        for key in store.data:
            self.assertIn(key, USAGE_DEFAULTS, key)


class RolloverTests(unittest.TestCase):
    def test_a_new_month_resets_the_counters_and_the_notice(self):
        service, store, clock = make_service()
        service.record("cloud", 120.0, 300)
        service.mark_fallback_notice_shown()

        clock.moment = datetime(2026, 10, 1, 0, 5, 0)
        service.record("cloud", 10.0, 30)

        self.assertEqual(store.data[KEY_MONTH], "2026-10")
        self.assertEqual(store.data[KEY_CLOUD_SECONDS], 10.0)
        self.assertEqual(store.data[KEY_CLOUD_WORDS], 30)
        # The allowance resets with the month, so the notice may be shown again.
        self.assertTrue(service.fallback_notice_pending)

    def test_the_one_time_trial_survives_the_rollover(self):
        service, store, clock = make_service()
        service.consume_trial(600.0)

        clock.moment = datetime(2026, 12, 1)
        service.record("local", 1.0, 1)

        self.assertEqual(store.data[KEY_TRIAL_SECONDS], 600.0)
        self.assertEqual(service.trial_remaining_seconds(), TRIAL_TOTAL_SECONDS - 600.0)

    def test_reading_a_summary_in_a_new_month_persists_the_reset(self):
        service, store, clock = make_service()
        service.record("cloud", 120.0, 300)

        clock.moment = datetime(2026, 10, 4)
        summary = service.summary()

        self.assertEqual(summary.month, "2026-10")
        self.assertEqual(summary.cloud_seconds, 0.0)
        self.assertEqual(store.data[KEY_MONTH], "2026-10")

    def test_summary_in_the_same_month_writes_nothing(self):
        service, store, _ = make_service()
        service.record("cloud", 60.0, 100)
        writes = store.saves

        service.summary()

        self.assertEqual(store.saves, writes)


class SummaryTests(unittest.TestCase):
    def test_summary_reports_what_was_recorded(self):
        service, _, _ = make_service()
        service.record("cloud", 90.0, 240)
        service.record("local", 30.0, 60)

        summary = service.summary()

        self.assertIsInstance(summary, UsageSummary)
        self.assertEqual(summary.month, "2026-09")
        self.assertEqual(summary.cloud_seconds, 90.0)
        self.assertEqual(summary.cloud_words, 240)
        self.assertEqual(summary.local_seconds, 30.0)
        self.assertEqual(summary.local_words, 60)
        self.assertEqual(summary.cloud_minutes, 1.5)
        self.assertEqual(summary.local_minutes, 0.5)
        self.assertEqual(summary.trial_seconds_used, 0.0)
        self.assertTrue(summary.fallback_notice_pending)

    def test_a_corrupt_stored_value_reads_as_zero_rather_than_crashing(self):
        service, _, _ = make_service({KEY_MONTH: "2026-09", KEY_CLOUD_SECONDS: "oops"})
        self.assertEqual(service.summary().cloud_seconds, 0.0)


class SoftLimitTests(unittest.TestCase):
    @staticmethod
    def remote(used, allowance=100.0):
        return Usage(
            minutes_used=used, minutes_allowance=allowance, words=0, period_end=None
        )

    def test_below_the_soft_limit_stays_on_cloud(self):
        service, _, _ = make_service()
        self.assertFalse(service.should_switch_to_local(self.remote(94.9)))

    def test_at_the_soft_limit_switches_to_local(self):
        service, _, _ = make_service()
        self.assertTrue(service.should_switch_to_local(self.remote(95.0)))

    def test_at_the_allowance_switches_to_local(self):
        service, _, _ = make_service()
        self.assertTrue(service.should_switch_to_local(self.remote(100.0)))
        self.assertTrue(service.should_switch_to_local(self.remote(140.0)))

    def test_the_default_soft_limit_is_95_percent(self):
        self.assertEqual(SOFT_LIMIT, 0.95)

    def test_the_soft_limit_is_configurable(self):
        service, _, _ = make_service()
        self.assertTrue(service.should_switch_to_local(self.remote(80.0), soft_limit=0.8))
        self.assertFalse(service.should_switch_to_local(self.remote(80.0), soft_limit=0.9))

    def test_an_unknown_allowance_never_forces_the_local_engine(self):
        # A body that omitted ``minutes_allowance`` used to read as zero and so
        # as "exhausted"; an unknown allowance must never move the user.
        service, _, _ = make_service()
        self.assertFalse(service.should_switch_to_local(self.remote(0.0, allowance=0.0)))
        self.assertFalse(service.should_switch_to_local(self.remote(50.0, allowance=-1.0)))

    def test_an_unknown_allowance_is_reported_at_info_rather_than_silently(self):
        service, _, _ = make_service()
        with self.assertLogs("services.usage_service", level="INFO") as captured:
            service.should_switch_to_local(self.remote(0.0, allowance=0.0))
        self.assertTrue(any("allowance" in line for line in captured.output))

    def test_without_a_remote_or_a_known_allowance_nothing_is_assumed(self):
        service, _, _ = make_service()
        service.record("cloud", 100_000.0, 1)
        self.assertFalse(service.should_switch_to_local(None))

    def test_without_a_remote_the_last_known_allowance_is_used(self):
        service, _, _ = make_service()
        service.should_switch_to_local(self.remote(10.0, allowance=100.0))
        self.assertEqual(service.known_allowance_minutes, 100.0)

        service.record("cloud", 94.0 * 60, 1)
        self.assertFalse(service.should_switch_to_local(None))

        service.record("cloud", 1.0 * 60, 1)
        self.assertTrue(service.should_switch_to_local(None))

    def test_a_zero_allowance_is_not_remembered_as_known(self):
        service, _, _ = make_service()
        service.should_switch_to_local(self.remote(0.0, allowance=0.0))
        self.assertIsNone(service.known_allowance_minutes)


class AllowanceCacheTests(unittest.TestCase):
    """The allowance survives a restart, and expires so it cannot mislead."""

    @staticmethod
    def remote(used, allowance=100.0):
        return Usage(
            minutes_used=used, minutes_allowance=allowance, words=0, period_end=None
        )

    def test_refresh_allowance_persists_the_three_values(self):
        service, store, clock = make_service()

        service.refresh_allowance(self.remote(42.0, allowance=300.0))

        self.assertEqual(store.data[KEY_ALLOWANCE_MINUTES], 300.0)
        self.assertEqual(store.data[KEY_REMOTE_MINUTES_USED], 42.0)
        self.assertEqual(store.data[KEY_ALLOWANCE_FETCHED_AT], clock().isoformat())

    def test_an_unknown_allowance_is_not_cached(self):
        service, store, _ = make_service()
        service.refresh_allowance(self.remote(1.0, allowance=0.0))
        self.assertIsNone(store.data.get(KEY_ALLOWANCE_MINUTES))
        self.assertTrue(service.allowance_is_stale())

    def test_a_fresh_cache_decides_without_a_remote_reading(self):
        service, _store, _clock = make_service()
        service.refresh_allowance(self.remote(96.0, allowance=100.0))
        self.assertFalse(service.allowance_is_stale())
        self.assertTrue(service.should_switch_to_local())

    def test_a_cache_that_survives_a_restart_is_still_used(self):
        service, store, clock = make_service()
        service.refresh_allowance(self.remote(96.0, allowance=100.0))

        restarted = UsageService(store, clock)

        self.assertEqual(restarted.known_allowance_minutes, 100.0)
        self.assertTrue(restarted.should_switch_to_local())

    def test_a_cache_older_than_the_ttl_is_stale_and_never_falls_back(self):
        service, _store, clock = make_service()
        service.refresh_allowance(self.remote(99.0, allowance=100.0))

        clock.moment = clock.moment + timedelta(seconds=ALLOWANCE_TTL_S + 1)

        self.assertTrue(service.allowance_is_stale())
        self.assertFalse(service.should_switch_to_local())

    def test_at_the_ttl_the_cache_is_still_fresh(self):
        service, _store, clock = make_service()
        service.refresh_allowance(self.remote(99.0, allowance=100.0))
        clock.moment = clock.moment + timedelta(seconds=ALLOWANCE_TTL_S)
        self.assertFalse(service.allowance_is_stale())
        self.assertTrue(service.should_switch_to_local())

    def test_nothing_cached_is_stale_and_never_falls_back(self):
        service, _, _ = make_service()
        self.assertTrue(service.allowance_is_stale())
        self.assertFalse(service.should_switch_to_local())

    def test_a_corrupt_timestamp_reads_as_stale(self):
        service, store, _ = make_service()
        service.refresh_allowance(self.remote(99.0, allowance=100.0))
        store.data[KEY_ALLOWANCE_FETCHED_AT] = "last tuesday"
        self.assertTrue(service.allowance_is_stale())
        self.assertFalse(service.should_switch_to_local())

    def test_the_default_ttl_is_fifteen_minutes(self):
        self.assertEqual(ALLOWANCE_TTL_S, 900)

    def test_local_minutes_beyond_the_cached_remote_reading_still_count(self):
        service, _, _ = make_service()
        service.refresh_allowance(self.remote(10.0, allowance=100.0))
        service.record("cloud", 96.0 * 60, 10)
        self.assertTrue(service.should_switch_to_local())

    def test_passing_a_remote_reading_refreshes_the_cache(self):
        service, store, _ = make_service()
        service.should_switch_to_local(self.remote(12.0, allowance=250.0))
        self.assertEqual(store.data[KEY_ALLOWANCE_MINUTES], 250.0)
        self.assertFalse(service.allowance_is_stale())


class SummaryDisplayTests(unittest.TestCase):
    """What the Engine tab reads off the summary."""

    def test_the_period_label_is_the_month_in_words(self):
        service, _, _ = make_service()
        self.assertEqual(service.summary().period_label, "September 2026")

    def test_the_period_label_follows_the_rollover(self):
        service, _, clock = make_service()
        clock.moment = datetime(2027, 1, 8)
        self.assertEqual(service.summary().period_label, "January 2027")

    def test_a_corrupt_month_falls_back_to_the_raw_value(self):
        summary = UsageSummary(
            month="nonsense",
            cloud_seconds=0.0,
            cloud_words=0,
            local_seconds=0.0,
            local_words=0,
            trial_seconds_used=0.0,
            fallback_notice_pending=True,
        )
        self.assertEqual(summary.period_label, "nonsense")

    def test_the_allowance_is_none_until_one_has_been_fetched(self):
        service, _, _ = make_service()
        self.assertIsNone(service.summary().allowance_minutes)

    def test_the_summary_reports_the_cached_allowance_in_whole_minutes(self):
        service, _, _ = make_service()
        service.refresh_allowance(
            Usage(minutes_used=1.0, minutes_allowance=300.0, words=0, period_end=None)
        )
        self.assertEqual(service.summary().allowance_minutes, 300)


class ConcurrencyTests(unittest.TestCase):
    """``record`` is load-modify-save, and dictations can finish together."""

    class SlowStore:
        """A store whose read and write are far enough apart to lose a write."""

        def __init__(self):
            self.data = {}

        def load(self):
            snapshot = dict(self.data)
            time.sleep(0.001)
            return snapshot

        def save(self, config):
            self.data = dict(config)

    def test_two_threads_recording_at_once_lose_nothing(self):
        store = self.SlowStore()
        service = UsageService(store, FakeClock())
        errors = []

        def worker():
            try:
                for _ in range(20):
                    service.record("cloud", 1.0, 2)
            except Exception as error:  # pragma: no cover - surfaced by the assert
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(store.data[KEY_CLOUD_SECONDS], 40.0)
        self.assertEqual(store.data[KEY_CLOUD_WORDS], 80)

    def test_the_trial_counter_survives_concurrent_spending(self):
        store = self.SlowStore()
        service = UsageService(store, FakeClock())

        def worker():
            for _ in range(10):
                service.consume_trial(1.0)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(store.data[KEY_TRIAL_SECONDS], 20.0)


class NoticeTests(unittest.TestCase):
    def test_the_notice_is_pending_until_it_is_marked_shown(self):
        service, store, _ = make_service()
        self.assertTrue(service.fallback_notice_pending)

        service.mark_fallback_notice_shown()

        self.assertFalse(service.fallback_notice_pending)
        self.assertTrue(store.data[KEY_NOTICE_SHOWN])

    def test_marking_twice_is_harmless(self):
        service, _, _ = make_service()
        service.mark_fallback_notice_shown()
        service.mark_fallback_notice_shown()
        self.assertFalse(service.fallback_notice_pending)

    def test_a_stored_flag_is_honoured(self):
        service, _, _ = make_service({KEY_MONTH: "2026-09", KEY_NOTICE_SHOWN: True})
        self.assertFalse(service.fallback_notice_pending)


class TrialTests(unittest.TestCase):
    def test_a_fresh_install_has_the_whole_hour(self):
        service, _, _ = make_service()
        self.assertEqual(service.trial_remaining_seconds(), 3600.0)
        self.assertEqual(TRIAL_TOTAL_SECONDS, 3600.0)

    def test_consuming_reduces_what_is_left(self):
        service, store, _ = make_service()

        remaining = service.consume_trial(900.0)

        self.assertEqual(remaining, 2700.0)
        self.assertEqual(store.data[KEY_TRIAL_SECONDS], 900.0)
        self.assertEqual(service.trial_remaining_seconds(), 2700.0)

    def test_the_trial_never_goes_negative_or_past_its_total(self):
        service, store, _ = make_service()

        self.assertEqual(service.consume_trial(5000.0), 0.0)

        self.assertEqual(store.data[KEY_TRIAL_SECONDS], TRIAL_TOTAL_SECONDS)
        self.assertEqual(service.trial_remaining_seconds(), 0.0)

    def test_a_custom_total_is_honoured(self):
        service, _, _ = make_service()
        service.consume_trial(60.0, trial_total=120.0)
        self.assertEqual(service.trial_remaining_seconds(trial_total=120.0), 60.0)

    def test_consuming_a_negative_amount_is_a_programming_error(self):
        service, _, _ = make_service()
        with self.assertRaises(AssertionError):
            service.consume_trial(-1.0)


if __name__ == "__main__":
    unittest.main()
