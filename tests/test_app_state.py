import unittest

from murmur import (
    should_apply_ready_on_reset,
    should_reject_toggle,
    should_reject_upload,
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


if __name__ == "__main__":
    unittest.main()
