"""Tests for engines/factory.py: config plus services in, an Engine out.

Every engine is replaced by a fake through ``engines.register_engine``, so
nothing here needs MLX, a whisper-server binary or a network: the point under
test is which constructor arguments the factory computes, not what the engines
do with them.
"""

import unittest
from pathlib import Path

from engines import register_engine, unregister_engine
from engines.factory import (
    CONFIG_CLOUD_BASE_URL,
    DEFAULT_CLOUD_BASE_URL,
    build_engine,
    byok_item_name,
    cloud_base_url,
)

ENGINE_IDS = ("whispercpp", "voxtral_mlx", "cloud", "byok")


class FakeEngine:
    """Records the kwargs it was built with. Not an ``Engine``; nothing loads it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeModelStore:
    def __init__(self, root="/models"):
        self.root = Path(root)
        self.asked = []

    def engine_model_path(self, model_id):
        self.asked.append(model_id)
        return self.root / model_id / "weights.bin"


class FakeLicenseService:
    def __init__(self, token="lease-token"):
        self.token = token
        self.calls = 0

    def current_lease_token(self):
        self.calls += 1
        return self.token


class FakeKeychain:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.asked = []

    def get(self, name):
        self.asked.append(name)
        return self.values.get(name)


class FactoryTestCase(unittest.TestCase):
    """Registers a fake for every engine id and cleans up afterwards."""

    def setUp(self):
        for engine_id in ENGINE_IDS:
            register_engine(engine_id, FakeEngine)
            self.addCleanup(unregister_engine, engine_id)
        self.store = FakeModelStore()
        self.license = FakeLicenseService()
        self.keychain = FakeKeychain({"byok-mistral": "sk-mistral"})

    def build(self, engine_id, config=None, **kwargs):
        return build_engine(
            engine_id,
            config=dict(config or {}),
            model_store=kwargs.pop("model_store", self.store),
            license_service=kwargs.pop("license_service", self.license),
            keychain=kwargs.pop("keychain", self.keychain),
            **kwargs,
        )


class LocalEngines(FactoryTestCase):
    def test_whispercpp_gets_the_model_path_from_the_store(self):
        engine = self.build("whispercpp", {"model_id": "ggml-turbo"})
        self.assertEqual(self.store.asked, ["ggml-turbo"])
        self.assertEqual(engine.kwargs, {"model_path": Path("/models/ggml-turbo/weights.bin")})

    def test_voxtral_gets_the_model_path_too(self):
        engine = self.build("voxtral_mlx", {"model_id": "voxtral-4bit"})
        self.assertEqual(engine.kwargs["model_path"], Path("/models/voxtral-4bit/weights.bin"))

    def test_a_local_engine_without_a_model_id_raises(self):
        for config in ({}, {"model_id": None}, {"model_id": "  "}):
            with self.subTest(config=config), self.assertRaises(ValueError) as caught:
                self.build("whispercpp", config)
            self.assertIn("model_id", str(caught.exception))

    def test_a_local_engine_is_not_handed_the_callers_transport(self):
        # whisper.cpp talks to a local child process; a cloud transport there
        # would be a category error, so http_open is not forwarded.
        engine = self.build("whispercpp", {"model_id": "m"}, http_open=object())
        self.assertNotIn("http_open", engine.kwargs)


class CloudEngine(FactoryTestCase):
    def test_cloud_uses_the_default_base_url_and_a_lease_callable(self):
        engine = self.build("cloud")
        self.assertEqual(engine.kwargs["base_url"], DEFAULT_CLOUD_BASE_URL)
        self.assertEqual(self.license.calls, 0, "the lease must not be read at build time")
        self.assertEqual(engine.kwargs["lease_provider"](), "lease-token")
        self.assertEqual(self.license.calls, 1)

    def test_cloud_honours_a_configured_base_url(self):
        engine = self.build("cloud", {CONFIG_CLOUD_BASE_URL: "https://proxy.example "})
        self.assertEqual(engine.kwargs["base_url"], "https://proxy.example")

    def test_cloud_lease_provider_is_re_read_every_call(self):
        engine = self.build("cloud")
        provider = engine.kwargs["lease_provider"]
        self.assertEqual(provider(), "lease-token")
        self.license.token = None
        self.assertIsNone(provider(), "a sign-out must take effect without a rebuild")

    def test_cloud_forwards_http_open(self):
        transport = object()
        engine = self.build("cloud", http_open=transport)
        self.assertIs(engine.kwargs["http_open"], transport)

    def test_cloud_without_a_license_service_raises(self):
        with self.assertRaises(ValueError) as caught:
            self.build("cloud", license_service=None)
        self.assertIn("license_service", str(caught.exception))


class ByokEngine(FactoryTestCase):
    def test_byok_reads_the_provider_key_from_the_keychain(self):
        engine = self.build("byok", {"byok_provider": "mistral"})
        self.assertEqual(engine.kwargs["provider"], "mistral")
        self.assertEqual(self.keychain.asked, [], "the key must not be read at build time")
        self.assertEqual(engine.kwargs["key_provider"](), "sk-mistral")
        self.assertEqual(self.keychain.asked, ["byok-mistral"])

    def test_byok_item_name_matches_the_keychain_constants(self):
        self.assertEqual(byok_item_name("mistral"), "byok-mistral")
        self.assertEqual(byok_item_name(" OpenAI "), "byok-openai")

    def test_byok_passes_an_optional_model_and_omits_a_blank_one(self):
        engine = self.build("byok", {"byok_provider": "openai", "byok_model": "whisper-1"})
        self.assertEqual(engine.kwargs["model"], "whisper-1")
        blank = self.build("byok", {"byok_provider": "openai", "byok_model": "  "})
        self.assertNotIn("model", blank.kwargs, "a blank model must leave the engine default")

    def test_byok_without_a_provider_raises(self):
        with self.assertRaises(ValueError) as caught:
            self.build("byok", {})
        self.assertIn("byok_provider", str(caught.exception))

    def test_byok_without_a_keychain_raises(self):
        with self.assertRaises(ValueError) as caught:
            self.build("byok", {"byok_provider": "mistral"}, keychain=None)
        self.assertIn("keychain", str(caught.exception))

    def test_a_missing_key_reads_as_none_rather_than_raising(self):
        engine = self.build("byok", {"byok_provider": "openai"})
        self.assertIsNone(engine.kwargs["key_provider"]())


class UnknownEngine(FactoryTestCase):
    def test_an_unknown_id_raises_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self.build("telepathy", {})
        self.assertIn("telepathy", str(caught.exception))


class CloudBaseUrl(unittest.TestCase):
    def test_default_is_https_and_marked_for_confirmation(self):
        self.assertTrue(DEFAULT_CLOUD_BASE_URL.startswith("https://"))

    def test_absent_blank_and_wrongly_typed_values_fall_back(self):
        for config in (None, {}, {CONFIG_CLOUD_BASE_URL: ""}, {CONFIG_CLOUD_BASE_URL: 7}):
            with self.subTest(config=config):
                self.assertEqual(cloud_base_url(config), DEFAULT_CLOUD_BASE_URL)


if __name__ == "__main__":
    unittest.main()
