import unittest
from unittest.mock import MagicMock, patch

from AppKit import NSEventModifierFlagOption

from services.hotkey_service import (
    DEFAULT_HOTKEY,
    HotkeyBinding,
    SPACE_KEYCODE,
    binding_from_ns_flags,
    binding_has_modifier,
    binding_matches_ns_event,
    capture_label_for_binding,
    format_hotkey,
    format_hotkey_diagnostics,
    hotkey_from_config,
    hotkey_registration_active,
    hotkey_to_config,
    parse_codesign_details,
    permission_status_message,
    register_global_hotkey,
    unregister_global_hotkey,
)


class HotkeyServiceTests(unittest.TestCase):
    def test_default_hotkey_is_option_space(self):
        self.assertEqual(DEFAULT_HOTKEY, HotkeyBinding(keycode=49, option=True))
        self.assertEqual(format_hotkey(DEFAULT_HOTKEY), "⌥ Space")

    def test_hotkey_config_roundtrip(self):
        binding = HotkeyBinding(
            keycode=0,
            command=True,
            shift=True,
        )
        restored = hotkey_from_config(hotkey_to_config(binding))
        self.assertEqual(restored, binding)
        self.assertEqual(format_hotkey(binding), "⇧ ⌘ A")

    def test_hotkey_from_config_uses_defaults_when_missing(self):
        self.assertEqual(hotkey_from_config({}), DEFAULT_HOTKEY)

    def test_binding_matches_ns_event_requires_exact_modifiers(self):
        from AppKit import NSEventModifierFlagShift

        binding = HotkeyBinding(keycode=49, option=True)
        flags = NSEventModifierFlagOption
        self.assertTrue(binding_matches_ns_event(binding, 49, flags))
        self.assertFalse(binding_matches_ns_event(binding, 49, flags | NSEventModifierFlagShift))
        self.assertFalse(binding_matches_ns_event(binding, 36, flags))

    def test_binding_matches_ns_event_with_multiple_modifiers(self):
        binding = HotkeyBinding(keycode=14, command=True, control=True, fn=True)
        from AppKit import (
            NSEventModifierFlagCommand,
            NSEventModifierFlagControl,
            NSEventModifierFlagFunction,
        )

        flags = (
            NSEventModifierFlagCommand
            | NSEventModifierFlagControl
            | NSEventModifierFlagFunction
        )
        self.assertTrue(binding_matches_ns_event(binding, 14, flags))
        self.assertEqual(format_hotkey(binding), "⌃ ⌘ fn E")

    def test_binding_matches_ns_event_tracks_option_space(self):
        binding = DEFAULT_HOTKEY
        self.assertTrue(
            binding_matches_ns_event(
                binding,
                SPACE_KEYCODE,
                NSEventModifierFlagOption,
            )
        )
        self.assertTrue(
            binding_matches_ns_event(
                binding,
                SPACE_KEYCODE,
                0,
                tracked_modifiers=NSEventModifierFlagOption,
            )
        )

    def test_binding_from_ns_flags(self):
        binding = binding_from_ns_flags(49, NSEventModifierFlagOption)
        self.assertEqual(binding, DEFAULT_HOTKEY)
        self.assertTrue(binding_has_modifier(binding))

    def test_capture_label_for_space(self):
        self.assertEqual(
            capture_label_for_binding(DEFAULT_HOTKEY, characters=" "),
            "Space",
        )

    def test_parse_codesign_details(self):
        output = (
            "Executable=/Applications/Murmur.app/Contents/MacOS/Murmur\n"
            "Identifier=com.canopystudio.murmur\n"
            "Signature=adhoc\n"
            "TeamIdentifier=not set\n"
            "CDHash=5498fb9320df9cfec448844a23dd6c423ce0c790\n"
        )
        details = parse_codesign_details(output)
        self.assertEqual(details["Signature"], "adhoc")
        self.assertEqual(details["CDHash"], "5498fb9320df9cfec448844a23dd6c423ce0c790")
        self.assertEqual(details["Identifier"], "com.canopystudio.murmur")

    def test_register_global_hotkey_requires_accessibility(self):
        with patch.dict("sys.modules", {"quickmachotkey": None}):
            with patch("services.hotkey_service.hotkey_permissions_ok", return_value=False):
                with self.assertRaises(RuntimeError):
                    register_global_hotkey(DEFAULT_HOTKEY, lambda: None, lambda _e: None, None)

    def test_register_global_hotkey_carbon_success(self):
        registration = register_global_hotkey(DEFAULT_HOTKEY, lambda: None, lambda _e: None, None)
        self.assertIsNotNone(registration.unregister_fn)
        self.assertTrue(hotkey_registration_active(registration))
        unregister_global_hotkey(registration)

    def test_register_global_hotkey_with_fn_skips_carbon(self):
        """Carbon cannot encode fn; fn bindings must use the NSEvent path."""
        binding = HotkeyBinding(keycode=SPACE_KEYCODE, option=True, fn=True)
        mock_monitor = object()
        mock_ns = MagicMock()
        mock_ns.addGlobalMonitorForEventsMatchingMask_handler_.return_value = mock_monitor
        mock_ns.addLocalMonitorForEventsMatchingMask_handler_.return_value = mock_monitor
        logger = MagicMock()

        with patch("services.hotkey_service.hotkey_permissions_ok", return_value=True):
            with patch("services.hotkey_service.NSEvent", mock_ns):
                registration = register_global_hotkey(
                    binding, lambda: None, lambda _e: None, logger
                )
                try:
                    self.assertIsNone(registration.unregister_fn)
                    self.assertIsNotNone(registration.global_monitor)
                    self.assertTrue(
                        mock_ns.addGlobalMonitorForEventsMatchingMask_handler_.called
                    )
                    self.assertTrue(hotkey_registration_active(registration))
                finally:
                    unregister_global_hotkey(registration)

    def test_permission_status_message_mentions_adhoc_bundle(self):
        diagnostics = {
            "bundled": True,
            "executable": "/Applications/Murmur.app/Contents/MacOS/Murmur",
            "bundle_id": "com.canopystudio.murmur",
            "signature": "adhoc",
            "team_identifier": "not set",
            "cdhash": "abc123",
            "ax_trusted": False,
            "global_monitor_ok": False,
            "shortcut_effective": False,
        }
        message = permission_status_message(diagnostics=diagnostics)
        self.assertIn("Ad-hoc builds", message)
        self.assertIn("CDHash: abc123", format_hotkey_diagnostics(diagnostics))


if __name__ == "__main__":
    unittest.main()
