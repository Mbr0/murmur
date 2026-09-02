import unittest
from unittest.mock import MagicMock, patch

from AppKit import NSEventModifierFlagOption

from services.hotkey_service import (
    ACTION_START,
    ACTION_STOP,
    DEFAULT_HOTKEY,
    DEFAULT_HOTKEY_MODE,
    HOLD_THRESHOLD_S,
    HOTKEY_MODES,
    HOTKEY_MODE_AUTO,
    HOTKEY_MODE_HOLD,
    HOTKEY_MODE_TOGGLE,
    HotkeyBinding,
    PressController,
    SPACE_KEYCODE,
    binding_from_ns_flags,
    binding_has_modifier,
    binding_matches_ns_event,
    binding_matches_ns_key_up,
    capture_label_for_binding,
    carbon_supports_key_up,
    format_hotkey,
    format_hotkey_diagnostics,
    hotkey_from_config,
    hotkey_mode_from_config,
    hotkey_registration_active,
    hotkey_to_config,
    parse_codesign_details,
    permission_status_message,
    register_global_hotkey,
    unregister_global_hotkey,
)

DOWN = "down"
UP = "up"


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


class HotkeyModeConfigTests(unittest.TestCase):
    def test_modes_are_toggle_hold_auto(self):
        self.assertEqual(HOTKEY_MODES, ("toggle", "hold", "auto"))
        self.assertEqual(DEFAULT_HOTKEY_MODE, HOTKEY_MODE_AUTO)
        self.assertEqual(HOLD_THRESHOLD_S, 0.3)

    def test_missing_key_falls_back_to_auto(self):
        self.assertEqual(hotkey_mode_from_config({}), HOTKEY_MODE_AUTO)
        self.assertEqual(hotkey_mode_from_config({"hotkey_keycode": 49}), HOTKEY_MODE_AUTO)

    def test_each_valid_mode_round_trips(self):
        for mode in HOTKEY_MODES:
            with self.subTest(mode=mode):
                self.assertEqual(hotkey_mode_from_config({"hotkey_mode": mode}), mode)

    def test_garbage_mode_fails_fast(self):
        for value in ("Toggle", "push", "", None, 0, 1, True, ["hold"]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    hotkey_mode_from_config({"hotkey_mode": value})


class PressControllerTableTests(unittest.TestCase):
    """Table-driven coverage of every mode, the hold boundary and stray events."""

    CASES = (
        (
            "toggle: key-down alternates and key-up is ignored",
            HOTKEY_MODE_TOGGLE,
            ((DOWN, 0.0), (UP, 0.05), (DOWN, 1.0), (UP, 1.05)),
            (ACTION_START, None, ACTION_STOP, None),
        ),
        (
            "toggle: a long press still only toggles once",
            HOTKEY_MODE_TOGGLE,
            ((DOWN, 0.0), (UP, 9.0)),
            (ACTION_START, None),
        ),
        (
            "toggle: key repeat while held is ignored",
            HOTKEY_MODE_TOGGLE,
            ((DOWN, 0.0), (DOWN, 0.05), (DOWN, 0.1), (UP, 0.2)),
            (ACTION_START, None, None, None),
        ),
        (
            "toggle: key-up without a prior key-down does nothing",
            HOTKEY_MODE_TOGGLE,
            ((UP, 0.0), (DOWN, 1.0)),
            (None, ACTION_START),
        ),
        (
            "hold: key-down starts and key-up stops",
            HOTKEY_MODE_HOLD,
            ((DOWN, 0.0), (UP, 0.05)),
            (ACTION_START, ACTION_STOP),
        ),
        (
            "hold: a press shorter than the threshold still stops on release",
            HOTKEY_MODE_HOLD,
            ((DOWN, 0.0), (UP, 0.001), (DOWN, 1.0), (UP, 5.0)),
            (ACTION_START, ACTION_STOP, ACTION_START, ACTION_STOP),
        ),
        (
            "hold: key repeat while held is ignored",
            HOTKEY_MODE_HOLD,
            ((DOWN, 0.0), (DOWN, 0.1), (DOWN, 0.2), (UP, 0.3)),
            (ACTION_START, None, None, ACTION_STOP),
        ),
        (
            "hold: key-up without a prior key-down does nothing",
            HOTKEY_MODE_HOLD,
            ((UP, 0.0), (UP, 0.1), (DOWN, 1.0), (UP, 1.5)),
            (None, None, ACTION_START, ACTION_STOP),
        ),
        (
            "auto: a long press behaves like hold",
            HOTKEY_MODE_AUTO,
            ((DOWN, 0.0), (UP, 0.301)),
            (ACTION_START, ACTION_STOP),
        ),
        (
            "auto: 301 ms is a hold, so release stops",
            HOTKEY_MODE_AUTO,
            ((DOWN, 10.0), (UP, 10.301)),
            (ACTION_START, ACTION_STOP),
        ),
        (
            "auto: 299 ms latches, and the next key-down stops",
            HOTKEY_MODE_AUTO,
            ((DOWN, 10.0), (UP, 10.299), (DOWN, 20.0), (UP, 20.05)),
            (ACTION_START, None, ACTION_STOP, None),
        ),
        (
            "auto: exactly the threshold counts as a hold",
            HOTKEY_MODE_AUTO,
            ((DOWN, 0.0), (UP, HOLD_THRESHOLD_S)),
            (ACTION_START, ACTION_STOP),
        ),
        (
            "auto: latched recording survives key repeat on the stopping press",
            HOTKEY_MODE_AUTO,
            (
                (DOWN, 0.0),
                (UP, 0.1),
                (DOWN, 5.0),
                (DOWN, 5.05),
                (UP, 5.1),
            ),
            (ACTION_START, None, ACTION_STOP, None, None),
        ),
        (
            "auto: key repeat while holding is ignored",
            HOTKEY_MODE_AUTO,
            ((DOWN, 0.0), (DOWN, 0.1), (DOWN, 0.2), (UP, 0.5)),
            (ACTION_START, None, None, ACTION_STOP),
        ),
        (
            "auto: key-up without a prior key-down does nothing",
            HOTKEY_MODE_AUTO,
            ((UP, 0.0), (DOWN, 1.0), (UP, 1.4)),
            (None, ACTION_START, ACTION_STOP),
        ),
        (
            "auto: two short presses then a long press",
            HOTKEY_MODE_AUTO,
            (
                (DOWN, 0.0),
                (UP, 0.1),
                (DOWN, 1.0),
                (UP, 1.05),
                (DOWN, 2.0),
                (UP, 2.9),
            ),
            (ACTION_START, None, ACTION_STOP, None, ACTION_START, ACTION_STOP),
        ),
    )

    def test_table(self):
        for name, mode, events, expected in self.CASES:
            with self.subTest(name):
                controller = PressController(mode)
                actions = []
                for kind, now in events:
                    if kind == DOWN:
                        actions.append(controller.on_key_down(now))
                    else:
                        actions.append(controller.on_key_up(now))
                self.assertEqual(tuple(actions), expected, name)

    def test_default_mode_is_auto(self):
        self.assertEqual(PressController().mode, HOTKEY_MODE_AUTO)

    def test_invalid_mode_fails_fast(self):
        with self.assertRaises(ValueError):
            PressController("push")
        with self.assertRaises(ValueError):
            PressController(HOTKEY_MODE_AUTO, hold_threshold_s=0)

    def test_custom_threshold_is_honoured(self):
        controller = PressController(HOTKEY_MODE_AUTO, hold_threshold_s=1.0)
        self.assertEqual(controller.on_key_down(0.0), ACTION_START)
        self.assertIsNone(controller.on_key_up(0.5))
        self.assertEqual(controller.on_key_down(2.0), ACTION_STOP)

    def test_is_recording_tracks_actions(self):
        controller = PressController(HOTKEY_MODE_HOLD)
        self.assertFalse(controller.is_recording)
        controller.on_key_down(0.0)
        self.assertTrue(controller.is_recording)
        controller.on_key_up(0.5)
        self.assertFalse(controller.is_recording)

    def test_sync_reconciles_with_a_rejected_start(self):
        controller = PressController(HOTKEY_MODE_TOGGLE)
        self.assertEqual(controller.on_key_down(0.0), ACTION_START)
        controller.sync(False)  # app refused: model still loading
        self.assertFalse(controller.is_recording)
        controller.on_key_up(0.05)
        # Next press starts rather than stopping a recording that never began.
        self.assertEqual(controller.on_key_down(1.0), ACTION_START)

    def test_sync_clears_the_auto_latch_when_recording_ended_elsewhere(self):
        controller = PressController(HOTKEY_MODE_AUTO)
        self.assertEqual(controller.on_key_down(0.0), ACTION_START)
        self.assertIsNone(controller.on_key_up(0.1))
        self.assertTrue(controller.is_latched)
        controller.sync(False)  # stopped from the menu
        self.assertFalse(controller.is_latched)
        self.assertEqual(controller.on_key_down(5.0), ACTION_START)

    def test_reset_clears_all_state(self):
        controller = PressController(HOTKEY_MODE_AUTO)
        controller.on_key_down(0.0)
        controller.reset()
        self.assertFalse(controller.is_recording)
        self.assertFalse(controller.is_latched)
        self.assertEqual(controller.on_key_down(1.0), ACTION_START)


class KeyUpDeliveryTests(unittest.TestCase):
    def test_key_up_matches_on_keycode_alone(self):
        binding = HotkeyBinding(keycode=SPACE_KEYCODE, option=True)
        # Modifiers are usually released first, so the key-up carries no flags.
        self.assertTrue(binding_matches_ns_key_up(binding, SPACE_KEYCODE))
        self.assertFalse(binding_matches_ns_key_up(binding, 36))

    def test_quickmachotkey_cannot_deliver_key_up(self):
        """Documented limitation: the library installs a pressed-only handler."""
        self.assertFalse(carbon_supports_key_up())

    def test_carbon_hold_mode_adds_an_ns_event_key_up_monitor(self):
        mock_ns = MagicMock()
        mock_ns.addGlobalMonitorForEventsMatchingMask_handler_.return_value = object()
        mock_ns.addLocalMonitorForEventsMatchingMask_handler_.return_value = object()
        logger = MagicMock()

        with patch("services.hotkey_service.hotkey_permissions_ok", return_value=True):
            with patch("services.hotkey_service.NSEvent", mock_ns):
                registration = register_global_hotkey(
                    DEFAULT_HOTKEY,
                    lambda: None,
                    lambda _e: None,
                    logger,
                    on_key_up=lambda: None,
                    mode=HOTKEY_MODE_HOLD,
                )
                try:
                    self.assertIsNotNone(registration.unregister_fn)  # Carbon key-down
                    self.assertIsNotNone(registration.global_key_up_monitor)
                    self.assertTrue(logger.warning.called)
                    warnings = " ".join(
                        str(call.args[0]) for call in logger.warning.call_args_list
                    )
                    self.assertIn("key-up", warnings)
                finally:
                    unregister_global_hotkey(registration)

    def test_carbon_toggle_mode_needs_no_key_up_monitor(self):
        mock_ns = MagicMock()
        logger = MagicMock()
        with patch("services.hotkey_service.NSEvent", mock_ns):
            registration = register_global_hotkey(
                DEFAULT_HOTKEY,
                lambda: None,
                lambda _e: None,
                logger,
                mode=HOTKEY_MODE_TOGGLE,
            )
            try:
                self.assertIsNone(registration.global_key_up_monitor)
                self.assertFalse(
                    mock_ns.addGlobalMonitorForEventsMatchingMask_handler_.called
                )
            finally:
                unregister_global_hotkey(registration)

    def test_carbon_hold_mode_warns_when_accessibility_is_missing(self):
        logger = MagicMock()
        with patch("services.hotkey_service.hotkey_permissions_ok", return_value=False):
            registration = register_global_hotkey(
                DEFAULT_HOTKEY,
                lambda: None,
                lambda _e: None,
                logger,
                on_key_up=lambda: None,
                mode=HOTKEY_MODE_AUTO,
            )
            try:
                self.assertIsNone(registration.global_key_up_monitor)
                warnings = " ".join(
                    str(call.args[0]) for call in logger.warning.call_args_list
                )
                self.assertIn("Accessibility", warnings)
            finally:
                unregister_global_hotkey(registration)

    def test_ns_event_path_registers_key_up_monitors(self):
        binding = HotkeyBinding(keycode=SPACE_KEYCODE, option=True, fn=True)
        mock_ns = MagicMock()
        mock_ns.addGlobalMonitorForEventsMatchingMask_handler_.return_value = object()
        mock_ns.addLocalMonitorForEventsMatchingMask_handler_.return_value = object()
        logger = MagicMock()

        with patch("services.hotkey_service.hotkey_permissions_ok", return_value=True):
            with patch("services.hotkey_service.NSEvent", mock_ns):
                registration = register_global_hotkey(
                    binding,
                    lambda: None,
                    lambda _e: None,
                    logger,
                    on_key_up=lambda: None,
                    mode=HOTKEY_MODE_HOLD,
                )
                try:
                    self.assertIsNone(registration.unregister_fn)
                    self.assertIsNotNone(registration.global_monitor)
                    self.assertIsNotNone(registration.global_key_up_monitor)
                    self.assertIsNotNone(registration.local_key_up_monitor)
                finally:
                    unregister_global_hotkey(registration)

    def test_ns_event_key_up_handler_calls_back_on_matching_keycode(self):
        binding = HotkeyBinding(keycode=SPACE_KEYCODE, option=True, fn=True)
        captured = {}
        released = []

        def remember(mask, handler):
            captured[int(mask)] = handler
            return object()

        mock_ns = MagicMock()
        mock_ns.addGlobalMonitorForEventsMatchingMask_handler_.side_effect = remember
        mock_ns.addLocalMonitorForEventsMatchingMask_handler_.return_value = object()

        from AppKit import NSEventMaskKeyUp

        with patch("services.hotkey_service.hotkey_permissions_ok", return_value=True):
            with patch("services.hotkey_service.NSEvent", mock_ns):
                registration = register_global_hotkey(
                    binding,
                    lambda: None,
                    lambda _e: None,
                    MagicMock(),
                    on_key_up=lambda: released.append(True),
                    mode=HOTKEY_MODE_HOLD,
                )
                try:
                    handler = captured[int(NSEventMaskKeyUp)]
                    event = MagicMock()
                    event.keyCode.return_value = SPACE_KEYCODE
                    handler(event)
                    self.assertEqual(len(released), 1)

                    other = MagicMock()
                    other.keyCode.return_value = 36
                    handler(other)
                    self.assertEqual(len(released), 1)
                finally:
                    unregister_global_hotkey(registration)


if __name__ == "__main__":
    unittest.main()
