import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from murmur import (
    check_for_update_message,
    clear_mic_device_selection,
    engine_is_ready,
    fetch_latest_release_tag,
    is_newer_version,
    normalize_release_tag,
    parse_latest_release_tag,
    resolve_mic_device,
    resolve_mic_device_index,
    should_apply_ready_on_reset,
    should_reject_toggle,
    should_reject_upload,
    skip_audio_user_message,
)


class AppStateGuardTests(unittest.TestCase):
    def test_should_reject_toggle_while_loading_or_processing(self):
        self.assertTrue(
            should_reject_toggle(loading=True, is_processing=False, model_ready=True)
        )
        self.assertTrue(
            should_reject_toggle(loading=False, is_processing=True, model_ready=True)
        )
        self.assertFalse(
            should_reject_toggle(loading=False, is_processing=False, model_ready=True)
        )

    def test_should_reject_toggle_when_model_unavailable(self):
        self.assertTrue(
            should_reject_toggle(loading=False, is_processing=False, model_ready=False)
        )

    def test_should_reject_upload_while_busy(self):
        self.assertTrue(
            should_reject_upload(
                loading=True, is_recording=False, is_processing=False, model_ready=True
            )
        )
        self.assertTrue(
            should_reject_upload(
                loading=False, is_recording=True, is_processing=False, model_ready=True
            )
        )
        self.assertTrue(
            should_reject_upload(
                loading=False, is_recording=False, is_processing=True, model_ready=True
            )
        )
        self.assertFalse(
            should_reject_upload(
                loading=False, is_recording=False, is_processing=False, model_ready=True
            )
        )

    def test_should_reject_upload_when_model_unavailable(self):
        self.assertTrue(
            should_reject_upload(
                loading=False, is_recording=False, is_processing=False, model_ready=False
            )
        )

    def test_should_not_force_ready_while_recording(self):
        self.assertFalse(should_apply_ready_on_reset(is_recording=True))
        self.assertTrue(should_apply_ready_on_reset(is_recording=False))


class EngineReadinessTests(unittest.TestCase):
    """The engine is built inside load_model(), so it is None until that runs."""

    def test_engine_is_not_ready_before_it_is_constructed(self):
        self.assertFalse(engine_is_ready(None))

    def test_engine_is_not_ready_until_it_is_loaded(self):
        self.assertFalse(engine_is_ready(SimpleNamespace(is_loaded=False)))

    def test_engine_is_ready_once_loaded(self):
        self.assertTrue(engine_is_ready(SimpleNamespace(is_loaded=True)))


class MicDeviceConfigTests(unittest.TestCase):
    def test_resolve_mic_unset_uses_system_default(self):
        self.assertIsNone(resolve_mic_device_index(None, {0, 1, 2}))
        self.assertIsNone(resolve_mic_device(None, None, {0: "Built-in"}))

    def test_resolve_mic_valid_index(self):
        self.assertEqual(resolve_mic_device_index(1, {0, 1, 2}), 1)

    def test_resolve_mic_invalid_index_fails_fast(self):
        with self.assertRaises(ValueError):
            resolve_mic_device_index(9, {0, 1, 2})

    def test_resolve_mic_rejects_bool_and_bad_types(self):
        with self.assertRaises(ValueError):
            resolve_mic_device_index(True, {0, 1})
        with self.assertRaises(ValueError):
            resolve_mic_device_index("1", {0, 1})

    def test_resolve_mic_rejects_wrong_device_at_same_index(self):
        devices = {0: "Built-in Mic", 1: "Other Mic"}
        with self.assertRaises(ValueError):
            resolve_mic_device(1, "USB Mic", devices)

    def test_resolve_mic_index_and_name_match(self):
        devices = {0: "Built-in Mic", 1: "USB Mic"}
        self.assertEqual(resolve_mic_device(1, "USB Mic", devices), 1)

    def test_resolve_mic_index_drift_resolves_by_name(self):
        devices = {0: "USB Mic", 2: "Built-in Mic"}
        self.assertEqual(resolve_mic_device(1, "USB Mic", devices), 0)
        # Saved index points at a different device; must not keep index 1.
        devices_swapped = {0: "USB Mic", 1: "Built-in Mic"}
        self.assertEqual(resolve_mic_device(1, "USB Mic", devices_swapped), 0)

    def test_resolve_mic_without_name_uses_index(self):
        devices = {0: "Built-in Mic", 1: "USB Mic"}
        self.assertEqual(resolve_mic_device(1, None, devices), 1)

    def test_clear_mic_device_selection(self):
        cleared = clear_mic_device_selection(
            {"mic_device_index": 2, "mic_device_name": "USB Mic", "model": "base"}
        )
        self.assertIsNone(cleared["mic_device_index"])
        self.assertIsNone(cleared["mic_device_name"])
        self.assertEqual(cleared["model"], "base")


class SkipAudioMessageTests(unittest.TestCase):
    def test_short_vs_quiet_messages(self):
        self.assertIn("short", skip_audio_user_message(0.4, 0.5).lower())
        self.assertIn("quiet", skip_audio_user_message(2.0, 0.001).lower())


class UpdateCheckTests(unittest.TestCase):
    def test_normalize_release_tag(self):
        self.assertEqual(normalize_release_tag("v1.2.3"), "1.2.3")
        self.assertEqual(normalize_release_tag("1.2.3"), "1.2.3")

    def test_is_newer_version(self):
        self.assertTrue(is_newer_version("1.1.0", "1.0.0"))
        self.assertTrue(is_newer_version("v2.0.0", "1.9.9"))
        self.assertFalse(is_newer_version("1.0.0", "1.0.0"))
        self.assertFalse(is_newer_version("1.0.0", "1.0.1"))

    def test_parse_latest_release_tag(self):
        self.assertEqual(
            parse_latest_release_tag({"tag_name": "v1.2.0"}),
            "v1.2.0",
        )
        with self.assertRaises(ValueError):
            parse_latest_release_tag({})

    def test_check_for_update_message_states(self):
        up_to_date = check_for_update_message(
            current_version="1.0.0", latest_tag="v1.0.0", error=None
        )
        self.assertIn("latest", up_to_date.lower())
        self.assertIn("1.0.0", up_to_date)

        available = check_for_update_message(
            current_version="1.0.0", latest_tag="v1.1.0", error=None
        )
        self.assertIn("1.1.0", available)
        self.assertIn("1.0.0", available)

        offline = check_for_update_message(
            current_version="1.0.0", latest_tag=None, error="offline"
        )
        self.assertIn("network", offline.lower())

    def test_check_for_update_message_bad_tag_is_error_not_crash(self):
        msg = check_for_update_message(
            current_version="1.0.0", latest_tag="not-a-version", error=None
        )
        self.assertIn("network", msg.lower())

    def test_check_for_update_message_local_ahead(self):
        msg = check_for_update_message(
            current_version="2.0.0", latest_tag="v1.0.0", error=None
        )
        self.assertIn("ahead", msg.lower())
        self.assertIn("2.0.0", msg)
        self.assertIn("1.0.0", msg)
        self.assertNotIn("you're on the latest", msg.lower())

    def test_fetch_latest_release_tag_reads_tag_only(self):
        payload = json.dumps({"tag_name": "v1.2.3", "body": "notes"}).encode("utf-8")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return payload

        with patch("urllib.request.urlopen", return_value=FakeResponse()) as mocked:
            tag = fetch_latest_release_tag(timeout=1.0)
            self.assertEqual(tag, "v1.2.3")
            request = mocked.call_args.args[0]
            self.assertIn("api.github.com/repos/Mbr0/murmur/releases/latest", request.full_url)
            self.assertEqual(mocked.call_args.kwargs.get("timeout"), 1.0)


if __name__ == "__main__":
    unittest.main()
