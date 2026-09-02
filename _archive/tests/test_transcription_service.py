import types
import unittest
from unittest.mock import Mock, call

from services.transcription_service import (
    extract_text,
    load_whisper_with_fallback,
    resolve_fp16,
    resolve_whisper_device,
    transcribe_audio,
    transcribe_audio_file,
)


def _fake_torch(*, mps_available: bool):
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available)
        )
    )


class TranscriptionServiceTests(unittest.TestCase):
    def test_extract_text_strips_whitespace(self):
        self.assertEqual(extract_text({"text": "  hello world  "}), "hello world")

    def test_resolve_whisper_device_prefers_mps_when_available(self):
        device = resolve_whisper_device(_fake_torch(mps_available=True))
        self.assertEqual(device, "mps")

    def test_resolve_whisper_device_falls_back_to_cpu(self):
        device = resolve_whisper_device(_fake_torch(mps_available=False))
        self.assertEqual(device, "cpu")

    def test_resolve_fp16_true_only_for_cuda(self):
        self.assertFalse(resolve_fp16("mps"))
        self.assertTrue(resolve_fp16("cuda"))
        self.assertFalse(resolve_fp16("cpu"))

    def test_resolve_fp16_rejects_empty_device(self):
        with self.assertRaises(AssertionError):
            resolve_fp16("")

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

    def test_transcribe_audio_accepts_explicit_fp16(self):
        model = Mock()
        model.transcribe.return_value = {"text": "ok"}

        transcribe_audio(model, "/tmp/sample.wav", fp16=True)

        kwargs = model.transcribe.call_args.kwargs
        self.assertTrue(kwargs["fp16"])

    def test_transcribe_audio_resolves_fp16_false_for_mps_when_omitted(self):
        model = Mock()
        model.transcribe.return_value = {"text": "ok"}

        transcribe_audio(model, "/tmp/sample.wav", device="mps")

        kwargs = model.transcribe.call_args.kwargs
        self.assertFalse(kwargs["fp16"])

    def test_transcribe_audio_resolves_fp16_true_for_cuda_when_omitted(self):
        model = Mock()
        model.transcribe.return_value = {"text": "ok"}

        transcribe_audio(model, "/tmp/sample.wav", device="cuda")

        kwargs = model.transcribe.call_args.kwargs
        self.assertTrue(kwargs["fp16"])

    def test_transcribe_audio_file_uses_file_defaults(self):
        model = Mock()
        model.transcribe.return_value = {"text": "file"}

        transcribe_audio_file(model, "/tmp/file.wav")

        model.transcribe.assert_called_once_with("/tmp/file.wav", fp16=False, language=None)

    def test_transcribe_audio_file_accepts_explicit_fp16(self):
        model = Mock()
        model.transcribe.return_value = {"text": "file"}

        transcribe_audio_file(model, "/tmp/file.wav", fp16=True)

        kwargs = model.transcribe.call_args.kwargs
        self.assertTrue(kwargs["fp16"])

    def test_load_whisper_with_fallback_succeeds_on_mps(self):
        model = object()
        load_fn = Mock(return_value=model)

        loaded, device, fp16 = load_whisper_with_fallback(load_fn, "mps")

        self.assertIs(loaded, model)
        self.assertEqual(device, "mps")
        self.assertFalse(fp16)
        load_fn.assert_called_once_with("mps")

    def test_load_whisper_with_fallback_retries_cpu_when_mps_fails(self):
        model = object()
        load_fn = Mock(side_effect=[RuntimeError("mps boom"), model])

        loaded, device, fp16 = load_whisper_with_fallback(load_fn, "mps")

        self.assertIs(loaded, model)
        self.assertEqual(device, "cpu")
        self.assertFalse(fp16)
        self.assertEqual(load_fn.call_args_list, [call("mps"), call("cpu")])

    def test_load_whisper_with_fallback_surfaces_cpu_failure(self):
        load_fn = Mock(side_effect=RuntimeError("cpu boom"))

        with self.assertRaises(RuntimeError):
            load_whisper_with_fallback(load_fn, "cpu")

        load_fn.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main()
