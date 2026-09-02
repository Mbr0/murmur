"""Tests for lease verification, entitlements, device linking and the Pro gate.

Test vectors are signed here with a throwaway Ed25519 key, so nothing in the
repository depends on the real Boske signing key.
"""

import base64
import json
import logging
import os
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from services.license_service import (
    DEFAULT_GRACE_DAYS,
    DEVICE_CODE_PATH,
    DEVICE_ID_SECRET_NAME,
    DEVICE_REFRESH_PATH,
    DEVICE_TOKEN_PATH,
    ENTITLEMENTS_CACHE_TTL_S,
    LEASE_AUDIENCE,
    LEASE_ISSUER,
    LEASE_SECRET_NAME,
    PLACEHOLDER_PUBLIC_KEY_PEM,
    PRO_FEATURES,
    PUBLIC_KEY_ENV_VAR,
    SECONDS_PER_DAY,
    Entitlements,
    InMemorySecretStore,
    LeaseError,
    LeaseExpired,
    LicenseService,
    LinkDenied,
    LinkError,
    LinkExpired,
    LinkSession,
    device_id_from_store,
    entitlements_from_lease,
    get_current_entitlements,
    is_pro_feature_enabled,
    poll_device_link,
    set_current_entitlements,
    start_device_link,
    verify_lease,
)

NOW = 1_800_000_000.0
HOUR = 3600.0
BASE_URL = "https://boske.test"

#: The device id every test lease is issued to; see ``claims()``.
THIS_DEVICE = "dev_abc"


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def encode_segment(payload: dict) -> str:
    return b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def public_pem(private_key: Ed25519PrivateKey) -> str:
    return (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )


def claims(**overrides) -> dict:
    payload = {
        "iss": LEASE_ISSUER,
        "aud": LEASE_AUDIENCE,
        "sub": "acct_123",
        "dev": "dev_abc",
        "iat": NOW - HOUR,
        "exp": NOW + 30 * SECONDS_PER_DAY,
        "ent": {"pro": True, "cloud_voice": True, "msm_minutes": 120},
        "plan": "pro-monthly",
    }
    payload.update(overrides)
    return payload


def make_token(private_key, payload=None, header=None) -> str:
    """Minimal local JWS encoder, so the tests do not depend on a JWT library."""
    header_b64 = encode_segment(header or {"alg": "EdDSA", "typ": "JWT"})
    payload_b64 = encode_segment(claims() if payload is None else payload)
    signature = private_key.sign(f"{header_b64}.{payload_b64}".encode("ascii"))
    return f"{header_b64}.{payload_b64}.{b64url(signature)}"


class FakeHttp:
    """Injectable transport: replays queued ``(status, json)`` pairs."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, data, headers):
        self.calls.append({"method": method, "url": url, "data": data, "headers": headers})
        if not self.responses:
            raise AssertionError(f"unexpected HTTP call to {url}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class LicenseTestCase(unittest.TestCase):
    """Common key material, and a clean Pro gate around every test."""

    def setUp(self):
        self.private_key = Ed25519PrivateKey.generate()
        self.public_pem = public_pem(self.private_key)
        self.other_key = Ed25519PrivateKey.generate()
        set_current_entitlements(Entitlements.none())
        self.addCleanup(set_current_entitlements, Entitlements.none())

    def token(self, payload=None, header=None, key=None) -> str:
        return make_token(key or self.private_key, payload, header)

    def verify(self, token, now=NOW):
        return verify_lease(token, public_key_pem=self.public_pem, now=now)


class VerifyLeaseTests(LicenseTestCase):
    def test_valid_token_yields_claims(self):
        lease = self.verify(self.token())
        self.assertEqual(lease.account_id, "acct_123")
        self.assertEqual(lease.device_id, "dev_abc")
        self.assertEqual(lease.audience, LEASE_AUDIENCE)
        self.assertEqual(lease.plan, "pro-monthly")
        self.assertTrue(lease.pro)
        self.assertTrue(lease.cloud_voice)
        self.assertEqual(lease.msm_minutes, 120)
        self.assertFalse(lease.is_expired(NOW))

    def test_audience_may_be_a_list(self):
        lease = self.verify(self.token(claims(aud=["other", LEASE_AUDIENCE])))
        self.assertEqual(lease.audience, LEASE_AUDIENCE)

    def test_tampered_payload_is_rejected(self):
        header_b64, payload_b64, signature_b64 = self.token().split(".")
        forged = encode_segment(claims(sub="acct_evil"))
        with self.assertRaises(LeaseError) as caught:
            self.verify(f"{header_b64}.{forged}.{signature_b64}")
        self.assertIn("signature", str(caught.exception))

    def test_token_signed_by_another_key_is_rejected(self):
        with self.assertRaises(LeaseError):
            self.verify(self.token(key=self.other_key))

    def test_wrong_audience_is_rejected(self):
        with self.assertRaises(LeaseError) as caught:
            self.verify(self.token(claims(aud="some-other-service")))
        self.assertIn("audience", str(caught.exception))

    def test_wrong_issuer_is_rejected(self):
        with self.assertRaises(LeaseError) as caught:
            self.verify(self.token(claims(iss="not-boske")))
        self.assertIn("issuer", str(caught.exception))

    def test_hs256_algorithm_is_rejected(self):
        header_b64 = encode_segment({"alg": "HS256", "typ": "JWT"})
        payload_b64 = encode_segment(claims())
        forged = f"{header_b64}.{payload_b64}.{b64url(b'mac')}"
        with self.assertRaises(LeaseError) as caught:
            self.verify(forged)
        self.assertIn("algorithm", str(caught.exception))

    def test_alg_none_with_empty_signature_is_rejected(self):
        header_b64 = encode_segment({"alg": "none"})
        payload_b64 = encode_segment(claims())
        with self.assertRaises(LeaseError):
            self.verify(f"{header_b64}.{payload_b64}.")

    def test_malformed_tokens_are_rejected(self):
        for bad in ("", "not-a-token", "a.b", "a.b.c.d", "@@@.@@@.@@@"):
            with self.subTest(token=bad):
                with self.assertRaises(LeaseError):
                    self.verify(bad)

    def test_missing_exp_is_rejected(self):
        payload = claims()
        del payload["exp"]
        with self.assertRaises(LeaseError) as caught:
            self.verify(self.token(payload))
        self.assertIn("exp", str(caught.exception))

    def test_expired_token_raises_lease_expired_carrying_the_lease(self):
        expiry = NOW - SECONDS_PER_DAY
        token = self.token(claims(iat=NOW - 40 * SECONDS_PER_DAY, exp=expiry))
        with self.assertRaises(LeaseExpired) as caught:
            self.verify(token)
        self.assertIsInstance(caught.exception, LeaseError)
        self.assertEqual(caught.exception.lease.expires_at, expiry)
        self.assertTrue(caught.exception.lease.pro)

    def test_expiry_boundary_counts_as_expired(self):
        with self.assertRaises(LeaseExpired):
            self.verify(self.token(claims(exp=NOW)))

    def test_iat_beyond_the_skew_window_is_rejected(self):
        with self.assertRaises(LeaseError) as caught:
            self.verify(self.token(claims(iat=NOW + 600)))
        self.assertIn("future", str(caught.exception))

    def test_iat_inside_the_skew_window_is_accepted(self):
        self.verify(self.token(claims(iat=NOW + 60)))

    def test_malformed_entitlements_are_rejected(self):
        for ent in (None, {}, {"pro": "yes", "cloud_voice": True, "msm_minutes": 1},
                    {"pro": True, "cloud_voice": True, "msm_minutes": "120"}):
            with self.subTest(ent=ent):
                with self.assertRaises(LeaseError):
                    self.verify(self.token(claims(ent=ent)))

    def test_placeholder_key_never_verifies(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LeaseError) as caught:
                verify_lease(self.token(), now=NOW)
            self.assertEqual(str(caught.exception), "no Boske public key embedded")
            with self.assertRaises(LeaseError):
                verify_lease(self.token(), public_key_pem=PLACEHOLDER_PUBLIC_KEY_PEM, now=NOW)

    def test_environment_override_supplies_the_key(self):
        with patch.dict(os.environ, {PUBLIC_KEY_ENV_VAR: self.public_pem}, clear=True):
            lease = verify_lease(self.token(), now=NOW)
        self.assertEqual(lease.account_id, "acct_123")

    def test_a_lease_for_another_device_is_refused_when_a_device_id_is_given(self):
        with self.assertRaises(LeaseError) as caught:
            verify_lease(
                self.token(), public_key_pem=self.public_pem, now=NOW, device_id="dev_other"
            )
        self.assertIn("another device", str(caught.exception))

    def test_a_lease_for_this_device_is_accepted(self):
        lease = verify_lease(
            self.token(), public_key_pem=self.public_pem, now=NOW, device_id=THIS_DEVICE
        )
        self.assertEqual(lease.device_id, THIS_DEVICE)

    def test_the_device_claim_is_only_checked_when_a_device_id_is_supplied(self):
        # Verification without a device id stays possible for tooling that has
        # no store to read one from; the service always supplies one.
        self.assertEqual(self.verify(self.token()).device_id, THIS_DEVICE)

    def test_the_device_check_outranks_expiry(self):
        stale = self.token(claims(iat=NOW - 40 * SECONDS_PER_DAY, exp=NOW - SECONDS_PER_DAY))
        with self.assertRaises(LeaseError) as caught:
            verify_lease(stale, public_key_pem=self.public_pem, now=NOW, device_id="dev_other")
        self.assertNotIsInstance(caught.exception, LeaseExpired)

    def test_the_lease_secret_name_matches_the_keychain_item(self):
        # Wave 3's KeychainStore keeps the lease under ITEM_LEASE = "boske-lease".
        self.assertEqual(LEASE_SECRET_NAME, "boske-lease")
        self.assertEqual(DEVICE_ID_SECRET_NAME, "device-id")

    def test_non_ed25519_key_is_refused(self):
        with self.assertRaises(LeaseError) as caught:
            verify_lease(self.token(), public_key_pem="-----BEGIN PUBLIC KEY-----\nx\n", now=NOW)
        self.assertIn("unreadable", str(caught.exception))


class EntitlementsTests(LicenseTestCase):
    def lease_expiring_now(self):
        """A lease whose ``exp`` is exactly NOW, verified an hour earlier."""
        token = self.token(claims(iat=NOW - 30 * SECONDS_PER_DAY, exp=NOW))
        with self.assertRaises(LeaseExpired) as caught:
            self.verify(token)
        return caught.exception.lease

    def test_no_license_is_the_free_tier(self):
        none = Entitlements.none()
        self.assertFalse(none.pro)
        self.assertFalse(none.cloud_voice)
        self.assertEqual(none.msm_minutes, 0)
        self.assertIsNone(none.expires_at)
        self.assertFalse(none.in_grace)
        self.assertEqual(none.source, "none")

    def test_live_lease_grants_everything(self):
        lease = self.verify(self.token())
        entitlements = entitlements_from_lease(lease, NOW)
        self.assertTrue(entitlements.pro)
        self.assertTrue(entitlements.cloud_voice)
        self.assertEqual(entitlements.msm_minutes, 120)
        self.assertFalse(entitlements.in_grace)
        self.assertEqual(entitlements.source, "lease")

    def test_cloud_voice_stops_the_moment_the_lease_lapses(self):
        lease = self.lease_expiring_now()
        entitlements = entitlements_from_lease(lease, NOW)
        self.assertFalse(entitlements.cloud_voice)
        self.assertEqual(entitlements.msm_minutes, 0)
        self.assertTrue(entitlements.pro)
        self.assertTrue(entitlements.in_grace)

    def test_day_three_of_grace_keeps_pro(self):
        lease = self.lease_expiring_now()
        entitlements = entitlements_from_lease(lease, NOW + 3 * SECONDS_PER_DAY)
        self.assertTrue(entitlements.pro)
        self.assertTrue(entitlements.in_grace)
        self.assertFalse(entitlements.cloud_voice)
        self.assertEqual(entitlements.expires_at, NOW)

    def test_day_eight_ends_grace(self):
        lease = self.lease_expiring_now()
        entitlements = entitlements_from_lease(lease, NOW + 8 * SECONDS_PER_DAY)
        self.assertFalse(entitlements.pro)
        self.assertFalse(entitlements.in_grace)
        self.assertFalse(entitlements.cloud_voice)

    def test_grace_window_length_is_configurable(self):
        lease = self.lease_expiring_now()
        self.assertEqual(DEFAULT_GRACE_DAYS, 7)
        day_three = entitlements_from_lease(lease, NOW + 3 * SECONDS_PER_DAY, grace_days=1)
        self.assertFalse(day_three.pro)

    def test_a_free_lease_never_grants_pro(self):
        payload = claims(ent={"pro": False, "cloud_voice": False, "msm_minutes": 0})
        entitlements = entitlements_from_lease(self.verify(self.token(payload)), NOW)
        self.assertFalse(entitlements.pro)
        self.assertFalse(entitlements.in_grace)


class TrialMinutesTests(LicenseTestCase):
    """``ent.trial_minutes``: the free cloud trial rides on a lease (D6)."""

    def free_account_claims(self, minutes=60, **overrides):
        ent = {"pro": False, "cloud_voice": False, "msm_minutes": 0}
        if minutes is not None:
            ent["trial_minutes"] = minutes
        return claims(ent=ent, plan=None, **overrides)

    def test_a_free_account_lease_carries_sixty_trial_minutes(self):
        lease = self.verify(self.token(self.free_account_claims()))
        self.assertEqual(lease.trial_minutes, 60)
        self.assertFalse(lease.cloud_voice)
        self.assertFalse(lease.pro)

    def test_the_claim_is_optional_and_absent_means_zero(self):
        lease = self.verify(self.token(self.free_account_claims(minutes=None)))
        self.assertEqual(lease.trial_minutes, 0)

    def test_a_paid_lease_carries_no_trial(self):
        self.assertEqual(self.verify(self.token()).trial_minutes, 0)

    def test_a_malformed_trial_claim_is_rejected(self):
        for bad in ("60", 60.5, True, -1, [60], {"minutes": 60}):
            with self.subTest(trial_minutes=bad):
                payload = claims(
                    ent={
                        "pro": False,
                        "cloud_voice": False,
                        "msm_minutes": 0,
                        "trial_minutes": bad,
                    }
                )
                with self.assertRaises(LeaseError) as caught:
                    self.verify(self.token(payload))
                self.assertIn("trial_minutes", str(caught.exception))

    def test_zero_is_a_legal_trial(self):
        lease = self.verify(self.token(self.free_account_claims(minutes=0)))
        self.assertEqual(lease.trial_minutes, 0)

    def test_entitlements_carry_the_trial(self):
        lease = self.verify(self.token(self.free_account_claims()))
        entitlements = entitlements_from_lease(lease, NOW)
        self.assertEqual(entitlements.trial_minutes, 60)
        self.assertFalse(entitlements.cloud_voice)

    def test_the_trial_stops_with_the_lease_like_every_other_cloud_minute(self):
        payload = self.free_account_claims(
            iat=NOW - 30 * SECONDS_PER_DAY, exp=NOW - SECONDS_PER_DAY
        )
        with self.assertRaises(LeaseExpired) as caught:
            self.verify(self.token(payload))
        entitlements = entitlements_from_lease(caught.exception.lease, NOW)
        self.assertEqual(entitlements.trial_minutes, 0)

    def test_no_lease_means_no_trial_at_all(self):
        # There is no anonymous cloud endpoint: signing in is what starts the
        # trial, so the free tier without a lease carries nothing.
        self.assertEqual(Entitlements.none().trial_minutes, 0)


class DeviceLinkTests(LicenseTestCase):
    def code_response(self, **overrides):
        payload = {
            "device_code": "dc_secret",
            "user_code": "WXYZ-1234",
            "verification_uri": "https://boske.test/link",
            "interval": 5,
            "expires_in": 600,
        }
        payload.update(overrides)
        return (200, payload)

    def start(self, http):
        return start_device_link(http, BASE_URL, now=NOW)

    def test_start_posts_the_documented_request(self):
        http = FakeHttp(self.code_response())
        session = self.start(http)
        call = http.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], f"{BASE_URL}{DEVICE_CODE_PATH}")
        self.assertEqual(call["data"], {"client_id": "murmur", "scope": LEASE_AUDIENCE})
        self.assertEqual(session.user_code, "WXYZ-1234")
        self.assertEqual(session.verification_url, "https://boske.test/link")
        self.assertEqual(session.interval_s, 5.0)
        self.assertEqual(session.expires_at, NOW + 600)

    def test_start_prefers_the_complete_verification_uri(self):
        http = FakeHttp(self.code_response(verification_uri_complete="https://b.test/link?c=1"))
        self.assertEqual(self.start(http).verification_url, "https://b.test/link?c=1")

    def test_start_raises_on_a_bad_status(self):
        with self.assertRaises(LinkError):
            self.start(FakeHttp((500, {})))

    def test_start_raises_when_fields_are_missing(self):
        with self.assertRaises(LinkError):
            self.start(FakeHttp((200, {"user_code": "X"})))

    def test_a_verification_url_that_is_not_https_is_refused(self):
        # The user is about to open this in a browser and approve a device;
        # anything but TLS is a phishing hop waiting to happen.
        for bad in (
            "http://boske.test/link",
            "javascript:alert(1)",
            "//boske.test/link",
            "boske.test/link",
        ):
            with self.subTest(url=bad):
                with self.assertRaises(LinkError) as caught:
                    self.start(FakeHttp(self.code_response(verification_uri=bad)))
                self.assertIn("https", str(caught.exception))

    def test_a_non_https_complete_url_is_refused_even_when_the_plain_one_is_safe(self):
        response = self.code_response(verification_uri_complete="http://boske.test/link?c=1")
        with self.assertRaises(LinkError):
            self.start(FakeHttp(response))

    def test_start_refuses_a_base_url_that_is_not_https(self):
        with self.assertRaises(ValueError):
            start_device_link(FakeHttp(self.code_response()), "http://boske.test", now=NOW)

    def test_poll_refuses_a_base_url_that_is_not_https(self):
        http = FakeHttp(self.code_response())
        session = self.start(http)
        with self.assertRaises(ValueError):
            poll_device_link(http, "http://boske.test", session, NOW)

    def test_the_device_id_is_sent_with_the_code_request_when_known(self):
        http = FakeHttp(self.code_response())
        start_device_link(http, BASE_URL, now=NOW, device_id=THIS_DEVICE)
        self.assertEqual(http.calls[0]["data"]["device_id"], THIS_DEVICE)

    def test_happy_path_returns_the_lease_token(self):
        token = self.token()
        http = FakeHttp(self.code_response(), (200, {"access_token": token}))
        session = self.start(http)
        self.assertEqual(poll_device_link(http, BASE_URL, session, NOW), token)
        call = http.calls[1]
        self.assertEqual(call["url"], f"{BASE_URL}{DEVICE_TOKEN_PATH}")
        self.assertEqual(call["data"]["device_code"], "dc_secret")
        self.assertEqual(
            call["data"]["grant_type"], "urn:ietf:params:oauth:grant-type:device_code"
        )

    def test_authorization_pending_returns_none(self):
        http = FakeHttp(self.code_response(), (400, {"error": "authorization_pending"}))
        session = self.start(http)
        self.assertIsNone(poll_device_link(http, BASE_URL, session, NOW))

    def test_interval_is_respected_without_touching_the_network(self):
        http = FakeHttp(self.code_response(), (400, {"error": "authorization_pending"}))
        session = self.start(http)
        poll_device_link(http, BASE_URL, session, NOW)
        self.assertIsNone(poll_device_link(http, BASE_URL, session, NOW + 1))
        self.assertEqual(len(http.calls), 2)

    def test_slow_down_widens_the_interval(self):
        http = FakeHttp(self.code_response(), (400, {"error": "slow_down"}))
        session = self.start(http)
        self.assertIsNone(poll_device_link(http, BASE_URL, session, NOW))
        self.assertEqual(session.interval_s, 10.0)
        self.assertEqual(session.next_poll_at, NOW + 10.0)

    def test_expired_token_raises_link_expired(self):
        http = FakeHttp(self.code_response(), (400, {"error": "expired_token"}))
        session = self.start(http)
        with self.assertRaises(LinkExpired):
            poll_device_link(http, BASE_URL, session, NOW)

    def test_a_lapsed_session_expires_before_any_request(self):
        http = FakeHttp(self.code_response())
        session = self.start(http)
        with self.assertRaises(LinkExpired):
            poll_device_link(http, BASE_URL, session, NOW + 601)
        self.assertEqual(len(http.calls), 1)

    def test_access_denied_raises_link_denied(self):
        http = FakeHttp(self.code_response(), (400, {"error": "access_denied"}))
        session = self.start(http)
        with self.assertRaises(LinkDenied):
            poll_device_link(http, BASE_URL, session, NOW)

    def test_unknown_error_raises_link_error(self):
        http = FakeHttp(self.code_response(), (400, {"error": "invalid_client"}))
        session = self.start(http)
        with self.assertRaises(LinkError) as caught:
            poll_device_link(http, BASE_URL, session, NOW)
        self.assertIn("invalid_client", str(caught.exception))

    def test_device_code_stays_out_of_the_session_repr(self):
        session = self.start(FakeHttp(self.code_response()))
        self.assertNotIn("dc_secret", repr(session))
        self.assertIn("WXYZ-1234", repr(session))


class FakeClock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class LicenseServiceTests(LicenseTestCase):
    def build(self, *, token=None, http=None, clock=None, device_id=THIS_DEVICE):
        initial = {DEVICE_ID_SECRET_NAME: device_id} if device_id else {}
        if token:
            initial[LEASE_SECRET_NAME] = token
        store = InMemorySecretStore(initial)
        clock = clock or FakeClock()
        service = LicenseService(
            store,
            http if http is not None else FakeHttp(),
            BASE_URL,
            clock=clock,
            public_key_pem=self.public_pem,
        )
        return service, store, clock

    def test_the_base_url_must_be_https(self):
        for bad in ("http://boske.test", "boske.test", "ftp://boske.test", ""):
            with self.subTest(base_url=bad):
                with self.assertRaises(ValueError):
                    LicenseService(
                        InMemorySecretStore(),
                        FakeHttp(),
                        bad,
                        clock=FakeClock(),
                        public_key_pem=self.public_pem,
                    )

    def test_a_lease_issued_to_another_device_is_refused(self):
        service, _store, _clock = self.build(device_id="dev_other")
        with self.assertRaises(LeaseError) as caught:
            service.store_lease(self.token())
        self.assertIn("another device", str(caught.exception))

    def test_a_stored_lease_for_another_device_grants_nothing(self):
        service, _store, _clock = self.build(token=self.token(), device_id="dev_other")
        with self.assertLogs("services.license_service", level="WARNING"):
            self.assertEqual(service.current_entitlements(), Entitlements.none())

    def test_a_lease_issued_to_this_device_is_accepted(self):
        service, _store, _clock = self.build()
        self.assertTrue(service.store_lease(self.token()).pro)

    def test_the_device_id_is_minted_once_and_then_reused(self):
        store = InMemorySecretStore()
        first = device_id_from_store(store)
        self.assertTrue(first)
        self.assertEqual(store.get(DEVICE_ID_SECRET_NAME), first)
        self.assertEqual(device_id_from_store(store), first)

    def test_the_service_mints_a_device_id_when_the_store_has_none(self):
        store = InMemorySecretStore()
        service = LicenseService(
            store, FakeHttp(), BASE_URL, clock=FakeClock(), public_key_pem=self.public_pem
        )
        minted = service.device_id()
        self.assertTrue(minted)
        self.assertEqual(store.get(DEVICE_ID_SECRET_NAME), minted)
        self.assertEqual(service.device_id(), minted)

    def test_an_injected_device_id_provider_wins(self):
        store = InMemorySecretStore()
        service = LicenseService(
            store,
            FakeHttp(),
            BASE_URL,
            clock=FakeClock(),
            public_key_pem=self.public_pem,
            device_id_provider=lambda: THIS_DEVICE,
        )
        self.assertEqual(service.device_id(), THIS_DEVICE)
        self.assertTrue(service.store_lease(self.token()).pro)
        self.assertIsNone(store.get(DEVICE_ID_SECRET_NAME))

    def test_begin_link_tells_boske_which_device_is_asking(self):
        code = (200, {
            "device_code": "dc_secret",
            "user_code": "WXYZ-1234",
            "verification_uri": "https://boske.test/link",
            "interval": 0,
            "expires_in": 600,
        })
        http = FakeHttp(code)
        service, _store, _clock = self.build(http=http)
        service.begin_link()
        self.assertEqual(http.calls[0]["data"]["device_id"], THIS_DEVICE)

    def test_no_stored_lease_is_the_free_tier(self):
        service, _store, _clock = self.build()
        self.assertEqual(service.current_entitlements(), Entitlements.none())
        self.assertFalse(is_pro_feature_enabled("cleanup"))

    def test_store_lease_verifies_persists_and_opens_the_gate(self):
        service, store, _clock = self.build()
        entitlements = service.store_lease(self.token())
        self.assertTrue(entitlements.pro)
        self.assertEqual(store.get(LEASE_SECRET_NAME), self.token())
        self.assertTrue(is_pro_feature_enabled("cleanup"))
        self.assertTrue(is_pro_feature_enabled("cloud_voice"))

    def test_store_lease_refuses_a_tampered_token(self):
        service, store, _clock = self.build()
        header, _payload, signature = self.token().split(".")
        tampered = f"{header}.{encode_segment(claims(sub='acct_evil'))}.{signature}"
        with self.assertRaises(LeaseError):
            service.store_lease(tampered)
        self.assertIsNone(store.get(LEASE_SECRET_NAME))
        self.assertFalse(is_pro_feature_enabled("cleanup"))

    def test_store_lease_keeps_an_expired_lease_so_grace_survives_restart(self):
        service, store, _clock = self.build()
        expired = self.token(claims(iat=NOW - 30 * SECONDS_PER_DAY, exp=NOW - SECONDS_PER_DAY))
        entitlements = service.store_lease(expired)
        self.assertTrue(entitlements.pro)
        self.assertTrue(entitlements.in_grace)
        self.assertFalse(entitlements.cloud_voice)
        self.assertEqual(store.get(LEASE_SECRET_NAME), expired)

    def test_a_stored_lease_is_reverified_and_a_forged_one_is_ignored(self):
        forged = make_token(self.other_key)
        service, _store, _clock = self.build(token=forged)
        with self.assertLogs("services.license_service", level="WARNING"):
            self.assertEqual(service.current_entitlements(), Entitlements.none())

    def test_entitlements_are_cached_for_the_ttl_then_recomputed(self):
        service, store, clock = self.build(token=self.token())
        self.assertTrue(service.current_entitlements().pro)
        store.delete(LEASE_SECRET_NAME)
        clock.advance(ENTITLEMENTS_CACHE_TTL_S - 1)
        self.assertTrue(service.current_entitlements().pro)
        clock.advance(2)
        self.assertFalse(service.current_entitlements().pro)

    def test_sign_out_clears_the_store_and_the_gate(self):
        service, store, _clock = self.build(token=self.token())
        self.assertTrue(service.current_entitlements().pro)
        service.sign_out()
        self.assertIsNone(store.get(LEASE_SECRET_NAME))
        self.assertEqual(service.current_entitlements(), Entitlements.none())
        self.assertFalse(is_pro_feature_enabled("cleanup"))

    def test_refresh_waits_until_eighty_percent_of_the_lifetime(self):
        lease_token = self.token(claims(iat=NOW, exp=NOW + 100))
        http = FakeHttp((200, {"access_token": self.token()}))
        clock = FakeClock(NOW + 79)
        service, store, _clock = self.build(token=lease_token, http=http, clock=clock)
        self.assertFalse(service.refresh_if_needed())
        self.assertEqual(http.calls, [])
        self.assertEqual(store.get(LEASE_SECRET_NAME), lease_token)

    def test_refresh_at_eighty_percent_swaps_the_lease(self):
        lease_token = self.token(claims(iat=NOW, exp=NOW + 100))
        renewed = self.token(claims(iat=NOW + 80, exp=NOW + 200))
        http = FakeHttp((200, {"access_token": renewed}))
        clock = FakeClock(NOW + 80)
        service, store, _clock = self.build(token=lease_token, http=http, clock=clock)
        self.assertTrue(service.refresh_if_needed())
        self.assertEqual(store.get(LEASE_SECRET_NAME), renewed)
        call = http.calls[0]
        self.assertEqual(call["url"], f"{BASE_URL}{DEVICE_REFRESH_PATH}")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {lease_token}")

    def test_a_failed_refresh_keeps_the_old_lease(self):
        lease_token = self.token(claims(iat=NOW, exp=NOW + 100))
        for response in ((500, {}), (200, {}), RuntimeError("connection reset")):
            with self.subTest(response=response):
                http = FakeHttp(response)
                service, store, _clock = self.build(
                    token=lease_token, http=http, clock=FakeClock(NOW + 90)
                )
                with self.assertLogs("services.license_service", level="WARNING"):
                    self.assertFalse(service.refresh_if_needed())
                self.assertEqual(store.get(LEASE_SECRET_NAME), lease_token)
                self.assertTrue(service.current_entitlements().pro)

    def test_a_refresh_returning_a_forged_lease_is_rejected(self):
        lease_token = self.token(claims(iat=NOW, exp=NOW + 100))
        http = FakeHttp((200, {"access_token": make_token(self.other_key)}))
        service, store, _clock = self.build(
            token=lease_token, http=http, clock=FakeClock(NOW + 90)
        )
        with self.assertLogs("services.license_service", level="WARNING"):
            self.assertFalse(service.refresh_if_needed())
        self.assertEqual(store.get(LEASE_SECRET_NAME), lease_token)

    def test_refresh_without_a_lease_does_nothing(self):
        http = FakeHttp((200, {"access_token": self.token()}))
        service, _store, _clock = self.build(http=http)
        self.assertFalse(service.refresh_if_needed())
        self.assertEqual(http.calls, [])

    def test_link_wrappers_pend_then_grant(self):
        code = (200, {
            "device_code": "dc_secret",
            "user_code": "WXYZ-1234",
            "verification_uri": "https://boske.test/link",
            "interval": 0,
            "expires_in": 600,
        })
        http = FakeHttp(
            code,
            (400, {"error": "authorization_pending"}),
            (200, {"access_token": self.token()}),
        )
        service, store, _clock = self.build(http=http)
        session = service.begin_link()
        self.assertIsInstance(session, LinkSession)
        self.assertEqual(session.user_code, "WXYZ-1234")
        self.assertIsNone(service.poll_link())
        entitlements = service.poll_link()
        self.assertTrue(entitlements.pro)
        self.assertEqual(store.get(LEASE_SECRET_NAME), self.token())
        self.assertTrue(is_pro_feature_enabled("modes"))

    def test_poll_link_without_begin_link_is_an_error(self):
        service, _store, _clock = self.build()
        with self.assertRaises(LinkError):
            service.poll_link()

    # -- current_lease_token: what the cloud engine is handed --------------

    def test_current_lease_token_returns_the_verified_token(self):
        service, _store, _clock = self.build(token=self.token())
        self.assertEqual(service.current_lease_token(), self.token())

    def test_current_lease_token_is_none_without_a_lease(self):
        service, _store, _clock = self.build()
        self.assertIsNone(service.current_lease_token())

    def test_current_lease_token_refuses_a_forged_token(self):
        service, _store, _clock = self.build(token=make_token(self.other_key))
        with self.assertLogs("services.license_service", level="WARNING"):
            self.assertIsNone(service.current_lease_token())

    def test_current_lease_token_refuses_an_expired_lease_even_in_grace(self):
        # Grace is a Pro concession, not a cloud one: sending an expired lease
        # to the proxy can only come back 401.
        expired = self.token(claims(iat=NOW - 30 * SECONDS_PER_DAY, exp=NOW - SECONDS_PER_DAY))
        service, _store, _clock = self.build(token=expired)
        self.assertTrue(service.current_entitlements().in_grace)
        with self.assertLogs("services.license_service", level="WARNING"):
            self.assertIsNone(service.current_lease_token())

    def test_current_lease_token_refuses_another_devices_lease(self):
        service, _store, _clock = self.build(token=self.token(), device_id="dev_other")
        with self.assertLogs("services.license_service", level="WARNING"):
            self.assertIsNone(service.current_lease_token())

    def test_current_lease_token_follows_a_sign_out(self):
        service, _store, _clock = self.build(token=self.token())
        self.assertIsNotNone(service.current_lease_token())
        service.sign_out()
        self.assertIsNone(service.current_lease_token())


def entitlements(pro, cloud_voice, in_grace=False):
    return Entitlements(
        pro=pro,
        cloud_voice=cloud_voice,
        msm_minutes=120 if cloud_voice else 0,
        expires_at=NOW + 100,
        in_grace=in_grace,
        source="lease",
    )


class ProGateTests(LicenseTestCase):
    def test_feature_list_matches_the_plan(self):
        self.assertEqual(
            PRO_FEATURES,
            (
                "cleanup",
                "modes",
                "context",
                "vocabulary_beyond_free",
                "snippets",
                "coding_mode",
                "cloud_voice",
            ),
        )

    def test_gate_table_for_every_feature_and_state(self):
        states = {
            "none": (Entitlements.none(), False, False),
            "pro-only": (entitlements(True, False), True, False),
            "pro+cloud": (entitlements(True, True), True, True),
            "grace": (entitlements(True, False, in_grace=True), True, False),
        }
        for name, (state, pro_allowed, cloud_allowed) in states.items():
            set_current_entitlements(state)
            for feature in PRO_FEATURES:
                expected = cloud_allowed if feature == "cloud_voice" else pro_allowed
                with self.subTest(state=name, feature=feature):
                    self.assertIs(is_pro_feature_enabled(feature), expected)

    def test_cloud_voice_does_not_ride_on_the_pro_flag(self):
        set_current_entitlements(entitlements(False, True))
        self.assertTrue(is_pro_feature_enabled("cloud_voice"))
        self.assertFalse(is_pro_feature_enabled("cleanup"))

    def test_unknown_feature_names_are_rejected(self):
        for feature in ("", "pro", "vocabulary", "Cleanup"):
            with self.subTest(feature=feature):
                with self.assertRaises(AssertionError):
                    is_pro_feature_enabled(feature)

    def test_published_entitlements_are_readable_for_display(self):
        state = entitlements(True, True)
        set_current_entitlements(state)
        self.assertEqual(get_current_entitlements(), state)

    def test_setting_a_non_entitlements_value_is_rejected(self):
        with self.assertRaises(AssertionError):
            set_current_entitlements({"pro": True})


class TokenLeakTests(LicenseTestCase):
    """The token is a bearer credential: it must not reach logs or messages."""

    def test_no_token_reaches_an_exception_message_or_a_log_record(self):
        valid = self.token()
        header, _payload, signature = valid.split(".")
        tampered = f"{header}.{encode_segment(claims(sub='acct_evil'))}.{signature}"
        forged = make_token(self.other_key)
        expired = self.token(claims(iat=NOW - 30 * SECONDS_PER_DAY, exp=NOW - HOUR))
        due = self.token(claims(iat=NOW - 100, exp=NOW + 10))

        haystacks = []
        for bad in (tampered, forged, "not-a-token", self.token(claims(aud="elsewhere"))):
            with self.assertRaises(LeaseError) as caught:
                self.verify(bad)
            haystacks.extend([str(caught.exception), repr(caught.exception)])
        with self.assertRaises(LeaseExpired) as caught:
            self.verify(expired)
        haystacks.extend(
            [str(caught.exception), repr(caught.exception), repr(caught.exception.lease)]
        )

        with self.assertLogs("services.license_service", level="WARNING") as captured:
            rejecting = LicenseService(
                InMemorySecretStore(
                    {LEASE_SECRET_NAME: tampered, DEVICE_ID_SECRET_NAME: THIS_DEVICE}
                ),
                FakeHttp(),
                BASE_URL,
                clock=FakeClock(),
                public_key_pem=self.public_pem,
            )
            rejecting.current_entitlements()
            refreshing = LicenseService(
                InMemorySecretStore(
                    {LEASE_SECRET_NAME: due, DEVICE_ID_SECRET_NAME: THIS_DEVICE}
                ),
                FakeHttp((401, {"error": "invalid_token"})),
                BASE_URL,
                clock=FakeClock(),
                public_key_pem=self.public_pem,
            )
            self.assertFalse(refreshing.refresh_if_needed())
        haystacks.extend(record.getMessage() for record in captured.records)
        haystacks.extend(captured.output)

        needles = []
        for token in (valid, tampered, forged, expired, due):
            needles.append(token)
            needles.extend(segment for segment in token.split(".") if len(segment) >= 16)
        for haystack in haystacks:
            for needle in needles:
                self.assertNotIn(needle, haystack)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
