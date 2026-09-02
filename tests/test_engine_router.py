"""Tests for services/engine_router.py — the routing table, as a table.

The router is pure, so every case here is data: a mode, an entitlement, a usage
state and a clip length in, an engine id and a notice out. Nothing is mocked
because there is nothing to mock.
"""

import unittest

from engines.cloud import (
    ALLOWANCE_MESSAGE,
    CloudAllowanceExhausted,
    CloudAuthError,
)
from engines.base import EngineError
from cleanup.vocabulary import FREE_TERM_LIMIT
from services.engine_router import (
    CLOUD_MODE_MURMUR,
    CLOUD_MODE_OFF,
    CLOUD_MODE_OWN_KEY,
    ENGINE_BYOK,
    ENGINE_CLOUD,
    MAX_CLIP_SECONDS,
    NOTICE_ADD_KEY,
    NOTICE_CLIP_TOO_LONG,
    NOTICE_SIGN_IN,
    REMOTE_ENGINE_IDS,
    Route,
    after_cloud_failure,
    effective_vocabulary_terms,
    route_engine,
)
from services.license_service import Entitlements

LOCAL = "whispercpp"


def entitlements(*, cloud_voice=False, trial_minutes=0):
    return Entitlements(
        pro=cloud_voice,
        cloud_voice=cloud_voice,
        msm_minutes=120 if cloud_voice else 0,
        expires_at=None,
        in_grace=False,
        source="lease",
        trial_minutes=trial_minutes,
    )


class FakeUsage:
    """The three things the router asks a :class:`UsageService`."""

    def __init__(self, *, trial_left=0.0, over_limit=False, notice_pending=True):
        self.trial_left = trial_left
        self.over_limit = over_limit
        self.notice_pending = notice_pending
        self.switch_calls = []

    def trial_remaining_seconds(self):
        return self.trial_left

    def should_switch_to_local(self, remote=None):
        self.switch_calls.append(remote)
        return self.over_limit

    @property
    def fallback_notice_pending(self):
        return self.notice_pending


def route(**overrides):
    kwargs = {
        "cloud_mode": CLOUD_MODE_MURMUR,
        "local_engine_id": LOCAL,
        "entitlements": entitlements(cloud_voice=True),
        "has_lease": True,
        "usage": FakeUsage(),
        "key_present": False,
        "clip_seconds": 30.0,
    }
    kwargs.update(overrides)
    return route_engine(**kwargs)


class OwnKeyMode(unittest.TestCase):
    def test_own_key_with_a_key_goes_to_byok(self):
        result = route(cloud_mode=CLOUD_MODE_OWN_KEY, key_present=True)
        self.assertEqual(result, Route(ENGINE_BYOK, None, "own key"))

    def test_own_key_without_a_key_falls_back_and_says_where_to_add_it(self):
        result = route(cloud_mode=CLOUD_MODE_OWN_KEY, key_present=False)
        self.assertEqual(result, Route(LOCAL, NOTICE_ADD_KEY, "own key missing"))

    def test_own_key_ignores_entitlements_leases_and_the_allowance(self):
        # The user's own provider bills them; nothing about Murmur Cloud applies.
        result = route(
            cloud_mode=CLOUD_MODE_OWN_KEY,
            key_present=True,
            entitlements=None,
            has_lease=False,
            usage=FakeUsage(over_limit=True),
            clip_seconds=10 * MAX_CLIP_SECONDS,
        )
        self.assertEqual(result.engine_id, ENGINE_BYOK)
        self.assertIsNone(result.notice)


class OffMode(unittest.TestCase):
    def test_off_is_local_and_silent(self):
        result = route(cloud_mode=CLOUD_MODE_OFF)
        self.assertEqual(result, Route(LOCAL, None, "cloud off"))

    def test_an_unknown_or_empty_mode_never_reaches_a_metered_engine(self):
        for mode in ("", None, "murmur-cloud", "MURMUR_CLOUD", "premium"):
            with self.subTest(mode=mode):
                self.assertEqual(route(cloud_mode=mode).engine_id, LOCAL)

    def test_the_configured_local_engine_is_the_fallback_not_a_default(self):
        result = route(cloud_mode=CLOUD_MODE_OFF, local_engine_id="voxtral_mlx")
        self.assertEqual(result.engine_id, "voxtral_mlx")


class MurmurCloudMode(unittest.TestCase):
    def test_entitled_under_the_limit_goes_to_the_cloud(self):
        result = route()
        self.assertEqual(result, Route(ENGINE_CLOUD, None, "cloud"))

    def test_a_free_account_with_trial_left_reaches_the_cloud(self):
        result = route(
            entitlements=entitlements(cloud_voice=False, trial_minutes=60),
            usage=FakeUsage(trial_left=1800.0),
        )
        self.assertEqual(result.engine_id, ENGINE_CLOUD)

    def test_no_lease_asks_the_user_to_sign_in(self):
        result = route(has_lease=False, usage=FakeUsage(trial_left=1800.0))
        self.assertEqual(result, Route(LOCAL, NOTICE_SIGN_IN, "no lease"))

    def test_a_lease_without_entitlement_or_trial_asks_to_sign_in(self):
        result = route(entitlements=entitlements(cloud_voice=False), usage=FakeUsage())
        self.assertEqual(result, Route(LOCAL, NOTICE_SIGN_IN, "not entitled"))

    def test_no_entitlements_at_all_reads_as_not_entitled(self):
        result = route(entitlements=None, usage=FakeUsage())
        self.assertEqual(result.engine_id, LOCAL)
        self.assertEqual(result.notice, NOTICE_SIGN_IN)

    def test_a_clip_over_sixty_minutes_stays_on_this_mac(self):
        result = route(clip_seconds=MAX_CLIP_SECONDS + 1)
        self.assertEqual(result, Route(LOCAL, NOTICE_CLIP_TOO_LONG, "clip too long"))

    def test_a_clip_of_exactly_sixty_minutes_still_goes_to_the_cloud(self):
        self.assertEqual(route(clip_seconds=MAX_CLIP_SECONDS).engine_id, ENGINE_CLOUD)

    def test_an_unknown_clip_length_does_not_block_the_cloud(self):
        self.assertEqual(route(clip_seconds=None).engine_id, ENGINE_CLOUD)

    def test_the_soft_limit_falls_back_with_the_allowance_notice(self):
        usage = FakeUsage(over_limit=True, notice_pending=True)
        result = route(usage=usage)
        self.assertEqual(result, Route(LOCAL, ALLOWANCE_MESSAGE, "allowance soft limit"))
        self.assertEqual(usage.switch_calls, [None], "the router must not fetch usage itself")

    def test_the_allowance_notice_is_not_repeated_once_shown(self):
        result = route(usage=FakeUsage(over_limit=True, notice_pending=False))
        self.assertEqual(result, Route(LOCAL, None, "allowance soft limit"))

    def test_routing_never_marks_the_notice_shown(self):
        usage = FakeUsage(over_limit=True, notice_pending=True)
        route(usage=usage)
        route(usage=usage)
        self.assertTrue(usage.fallback_notice_pending, "the caller marks it, not the router")

    def test_entitlement_is_checked_before_clip_length(self):
        # Someone who never signed in is told to sign in, not that their
        # recording is too long for a service they cannot reach.
        result = route(has_lease=False, clip_seconds=MAX_CLIP_SECONDS + 1)
        self.assertEqual(result.notice, NOTICE_SIGN_IN)

    def test_clip_length_is_checked_before_the_allowance(self):
        result = route(
            clip_seconds=MAX_CLIP_SECONDS + 1,
            usage=FakeUsage(over_limit=True),
        )
        self.assertEqual(result.notice, NOTICE_CLIP_TOO_LONG)


class RoutingTable(unittest.TestCase):
    """The whole table in one place, so a reordering shows up as one failure."""

    CASES = (
        ("own key, key stored", CLOUD_MODE_OWN_KEY, True, True, True, False, 10.0,
         ENGINE_BYOK, None),
        ("own key, no key", CLOUD_MODE_OWN_KEY, True, True, False, False, 10.0,
         LOCAL, NOTICE_ADD_KEY),
        ("off", CLOUD_MODE_OFF, True, True, True, False, 10.0, LOCAL, None),
        ("cloud, entitled", CLOUD_MODE_MURMUR, True, True, False, False, 10.0,
         ENGINE_CLOUD, None),
        ("cloud, no lease", CLOUD_MODE_MURMUR, True, False, False, False, 10.0,
         LOCAL, NOTICE_SIGN_IN),
        ("cloud, not entitled", CLOUD_MODE_MURMUR, False, True, False, False, 10.0,
         LOCAL, NOTICE_SIGN_IN),
        ("cloud, long clip", CLOUD_MODE_MURMUR, True, True, False, False, 3601.0,
         LOCAL, NOTICE_CLIP_TOO_LONG),
        ("cloud, over limit", CLOUD_MODE_MURMUR, True, True, False, True, 10.0,
         LOCAL, ALLOWANCE_MESSAGE),
    )

    def test_table(self):
        for (
            name, mode, cloud_voice, has_lease, key_present, over_limit,
            clip_seconds, engine_id, notice,
        ) in self.CASES:
            with self.subTest(case=name):
                result = route_engine(
                    cloud_mode=mode,
                    local_engine_id=LOCAL,
                    entitlements=entitlements(cloud_voice=cloud_voice),
                    has_lease=has_lease,
                    usage=FakeUsage(over_limit=over_limit),
                    key_present=key_present,
                    clip_seconds=clip_seconds,
                )
                self.assertEqual(result.engine_id, engine_id)
                self.assertEqual(result.notice, notice)


class AfterCloudFailure(unittest.TestCase):
    def test_an_exhausted_allowance_falls_back_with_the_allowance_wording(self):
        result = after_cloud_failure(
            CloudAllowanceExhausted(ALLOWANCE_MESSAGE), local_engine_id=LOCAL
        )
        self.assertEqual(result, Route(LOCAL, ALLOWANCE_MESSAGE, "allowance exhausted"))

    def test_a_rejected_lease_asks_the_user_to_sign_in_again(self):
        result = after_cloud_failure(CloudAuthError("401"), local_engine_id=LOCAL)
        self.assertEqual(result, Route(LOCAL, NOTICE_SIGN_IN, "lease rejected"))

    def test_any_other_failure_falls_back_silently(self):
        result = after_cloud_failure(EngineError("HTTP 503"), local_engine_id=LOCAL)
        self.assertEqual(result, Route(LOCAL, None, "cloud failed"))

    def test_the_fallback_honours_the_configured_local_engine(self):
        result = after_cloud_failure(CloudAuthError("401"), local_engine_id="voxtral_mlx")
        self.assertEqual(result.engine_id, "voxtral_mlx")


class VocabularyGate(unittest.TestCase):
    TERMS = tuple(f"term{index}" for index in range(FREE_TERM_LIMIT + 5))

    def test_pro_keeps_every_term(self):
        self.assertEqual(effective_vocabulary_terms(self.TERMS, lambda _f: True), self.TERMS)

    def test_free_keeps_the_first_twenty(self):
        kept = effective_vocabulary_terms(self.TERMS, lambda _f: False)
        self.assertEqual(kept, self.TERMS[:FREE_TERM_LIMIT])

    def test_a_short_list_is_untouched_on_the_free_tier(self):
        short = ("alpha", "beta")
        self.assertEqual(effective_vocabulary_terms(short, lambda _f: False), short)

    def test_it_asks_the_gate_for_the_right_feature(self):
        asked = []

        def gate(feature):
            asked.append(feature)
            return True

        effective_vocabulary_terms(self.TERMS, gate)
        self.assertEqual(asked, ["vocabulary_beyond_free"])

    def test_none_and_empty_are_an_empty_tuple(self):
        for terms in (None, (), []):
            with self.subTest(terms=terms):
                self.assertEqual(effective_vocabulary_terms(terms, lambda _f: False), ())

    def test_the_result_is_always_a_tuple(self):
        self.assertIsInstance(effective_vocabulary_terms(["a"], lambda _f: True), tuple)


class RemoteEngineTableTests(unittest.TestCase):
    """Which engine ids send audio off this Mac, in one place.

    The pipeline asked this three separate times, each as its own
    ``in (ENGINE_CLOUD, ENGINE_BYOK)``. A fourth hosted engine would have had to
    be found in all three, and the one that was missed would have quietly
    counted as local — in the usage meter, in the history origin, and in what
    the Privacy tab tells the user leaves their Mac.
    """

    def test_the_hosted_engines_are_the_two_that_leave_the_mac(self):
        self.assertEqual(REMOTE_ENGINE_IDS, (ENGINE_CLOUD, ENGINE_BYOK))

    def test_every_engine_the_router_can_choose_is_local_or_in_the_table(self):
        for mode, engine_id in (
            (CLOUD_MODE_OFF, "whispercpp"),
            (CLOUD_MODE_MURMUR, ENGINE_CLOUD),
            (CLOUD_MODE_OWN_KEY, ENGINE_BYOK),
        ):
            with self.subTest(mode=mode):
                self.assertIn(engine_id, ("whispercpp", *REMOTE_ENGINE_IDS))


if __name__ == "__main__":
    unittest.main()
