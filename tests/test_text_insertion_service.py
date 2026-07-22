import unittest
from unittest.mock import MagicMock, call, patch

from services.text_insertion_service import (
    CLIPBOARD_RESTORE_DELAY_S,
    TextInsertionService,
)


class TextInsertionServiceTests(unittest.TestCase):
    def test_restore_delay_constant_is_in_safe_range(self):
        self.assertGreaterEqual(CLIPBOARD_RESTORE_DELAY_S, 0.35)
        self.assertLessEqual(CLIPBOARD_RESTORE_DELAY_S, 0.5)

    @patch("services.text_insertion_service.CGEventPost")
    @patch("services.text_insertion_service.CGEventSetFlags")
    @patch("services.text_insertion_service.CGEventCreateKeyboardEvent")
    @patch("services.text_insertion_service.time.sleep")
    @patch("services.text_insertion_service.pyperclip")
    def test_paste_text_restores_previous_clipboard(
        self,
        mock_pyperclip,
        _mock_sleep,
        mock_create_event,
        _mock_set_flags,
        _mock_post,
    ):
        mock_pyperclip.paste.return_value = "user-clipboard"
        mock_create_event.return_value = MagicMock()
        logger = MagicMock()
        service = TextInsertionService(logger=logger)

        service.paste_text("transcript")

        self.assertEqual(
            mock_pyperclip.method_calls,
            [
                call.paste(),
                call.copy("transcript"),
                call.copy("user-clipboard"),
            ],
        )
        logger.info.assert_called_with("Paste sent via CGEvent")

    @patch("services.text_insertion_service.CGEventPost")
    @patch("services.text_insertion_service.CGEventSetFlags")
    @patch("services.text_insertion_service.CGEventCreateKeyboardEvent")
    @patch("services.text_insertion_service.time.sleep")
    @patch("services.text_insertion_service.pyperclip")
    def test_paste_text_sleeps_restore_delay_after_key_up(
        self,
        mock_pyperclip,
        mock_sleep,
        mock_create_event,
        _mock_set_flags,
        mock_post,
    ):
        mock_pyperclip.paste.return_value = "user-clipboard"
        mock_create_event.return_value = MagicMock()
        logger = MagicMock()
        service = TextInsertionService(logger=logger)

        service.paste_text("transcript")

        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_sleep.call_args_list[-1], call(CLIPBOARD_RESTORE_DELAY_S))

    @patch("services.text_insertion_service.CGEventPost")
    @patch("services.text_insertion_service.CGEventSetFlags")
    @patch("services.text_insertion_service.CGEventCreateKeyboardEvent")
    @patch("services.text_insertion_service.time.sleep")
    @patch("services.text_insertion_service.pyperclip")
    def test_paste_text_restores_clipboard_when_cgevent_post_fails(
        self,
        mock_pyperclip,
        mock_sleep,
        mock_create_event,
        _mock_set_flags,
        mock_post,
    ):
        mock_pyperclip.paste.return_value = "user-clipboard"
        mock_create_event.return_value = MagicMock()
        mock_post.side_effect = RuntimeError("CGEventPost failed")
        logger = MagicMock()
        service = TextInsertionService(logger=logger)

        with self.assertRaises(RuntimeError):
            service.paste_text("transcript")

        self.assertEqual(
            mock_pyperclip.copy.call_args_list,
            [call("transcript"), call("user-clipboard")],
        )
        mock_sleep.assert_any_call(CLIPBOARD_RESTORE_DELAY_S)

    @patch("services.text_insertion_service.CGEventPost")
    @patch("services.text_insertion_service.CGEventSetFlags")
    @patch("services.text_insertion_service.CGEventCreateKeyboardEvent")
    @patch("services.text_insertion_service.time.sleep")
    @patch("services.text_insertion_service.pyperclip")
    def test_paste_text_leaves_transcript_when_restore_fails(
        self,
        mock_pyperclip,
        _mock_sleep,
        mock_create_event,
        _mock_set_flags,
        _mock_post,
    ):
        mock_pyperclip.paste.return_value = "user-clipboard"
        mock_pyperclip.copy.side_effect = [None, RuntimeError("restore failed")]
        mock_create_event.return_value = MagicMock()
        logger = MagicMock()
        service = TextInsertionService(logger=logger)

        service.paste_text("transcript")

        self.assertEqual(
            mock_pyperclip.copy.call_args_list,
            [call("transcript"), call("user-clipboard")],
        )
        logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main()
