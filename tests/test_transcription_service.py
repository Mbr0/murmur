import unittest
from unittest.mock import Mock

from services.transcription_service import extract_text, transcribe_audio, transcribe_audio_file


class TranscriptionServiceTests(unittest.TestCase):
    def test_extract_text_strips_whitespace(self):
        self.assertEqual(extract_text({"text": "  hello world  "}), "hello world")

    def test_transcribe_audio_uses_local_defaults(self):
        model = Mock()
        model.transcribe.return_value = {"text": "ok"}

        result = transcribe_audio(model, "/tmp/sample.wav")

        self.assertEqual(result["text"], "ok")
        model.transcribe.assert_called_once_with(
            "/tmp/sample.wav",
            fp16=False,
            language=None,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )

    def test_transcribe_audio_file_uses_file_defaults(self):
        model = Mock()
        model.transcribe.return_value = {"text": "file"}

        transcribe_audio_file(model, "/tmp/file.wav")

        model.transcribe.assert_called_once_with("/tmp/file.wav", fp16=False, language=None)


if __name__ == "__main__":
    unittest.main()
