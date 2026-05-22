import unittest

from transcription_filters import is_likely_hallucination, should_skip_audio


class ShouldSkipAudioTests(unittest.TestCase):
    def test_skips_audio_when_duration_is_too_short(self):
        self.assertTrue(should_skip_audio(0.5, 0.2))

    def test_skips_audio_when_audio_is_too_quiet(self):
        self.assertTrue(should_skip_audio(2.0, 0.001))

    def test_accepts_audio_when_duration_and_level_are_valid(self):
        self.assertFalse(should_skip_audio(2.0, 0.1))


class HallucinationFilterTests(unittest.TestCase):
    def test_filters_common_low_signal_hallucination(self):
        self.assertTrue(is_likely_hallucination("thank you"))

    def test_filters_short_text(self):
        self.assertTrue(is_likely_hallucination("ok"))

    def test_keeps_normal_transcription(self):
        self.assertFalse(is_likely_hallucination("Schedule the meeting at 3 PM tomorrow"))


if __name__ == "__main__":
    unittest.main()
