import types
import unittest
from pathlib import Path
from unittest.mock import patch

from engines import EngineNotLoadedError, EngineUnavailableError, Hints, create_engine
from engines.base import LANGUAGE_AUTO
from engines.whisper_openai import ENGINE_CLASS, WhisperOpenAIEngine

CANNED_RESULT = {
    "text": "  bonjour le monde  ",
    "language": "fr",
    "segments": [
        {"start": 0.0, "end": 1.25, "text": " bonjour "},
        {"start": 1.25, "end": 2.5, "text": " le monde "},
    ],
}


class FakeModel:
    """Records the kwargs Whisper would have been called with."""

    def __init__(self, result=None):
        self.calls = []
        self.result = dict(CANNED_RESULT if result is None else result)

    def transcribe(self, audio_path, **kwargs):
        self.calls.append((audio_path, kwargs))
        return self.result


def fake_whisper_module(model=None):
    """Stand-in for the real ``whisper`` module."""
    model = model or FakeModel()
    calls = []

    def load_model(name, device=None):
        calls.append((name, device))
        return model

    module = types.SimpleNamespace(load_model=load_model)
    module.load_calls = calls
    module.model = model
    return module


def loaded_engine(module=None, model_name="base", device="cpu"):
    module = module or fake_whisper_module()
    engine = WhisperOpenAIEngine(model_name, whisper_module=module)
    with patch("engines.whisper_openai.resolve_whisper_device", return_value=device):
        with patch("engines.whisper_openai._empty_mps_cache"):
            engine.load()
    return engine, module


class LoadTests(unittest.TestCase):
    def test_engine_class_is_exported_for_the_registry(self):
        self.assertIs(ENGINE_CLASS, WhisperOpenAIEngine)

    def test_model_name_is_required(self):
        with self.assertRaises(AssertionError):
            WhisperOpenAIEngine("")

    def test_starts_unloaded(self):
        engine = WhisperOpenAIEngine("base", whisper_module=fake_whisper_module())
        self.assertFalse(engine.is_loaded)

    def test_load_uses_the_injected_module_and_records_device(self):
        engine, module = loaded_engine(model_name="small", device="cpu")
        self.assertTrue(engine.is_loaded)
        self.assertEqual(module.load_calls, [("small", "cpu")])
        self.assertEqual(engine.device, "cpu")
        self.assertFalse(engine.fp16)

    def test_load_falls_back_to_cpu_when_mps_load_fails(self):
        module = fake_whisper_module()
        model = module.model
        attempts = []

        def flaky_load_model(name, device=None):
            attempts.append(device)
            if device == "mps":
                raise RuntimeError("mps boom")
            return model

        module.load_model = flaky_load_model
        engine = WhisperOpenAIEngine("base", whisper_module=module)
        with patch("engines.whisper_openai.resolve_whisper_device", return_value="mps"):
            engine.load()

        self.assertEqual(attempts, ["mps", "cpu"])
        self.assertEqual(engine.device, "cpu")
        self.assertFalse(engine.fp16)

    def test_load_is_idempotent(self):
        engine, module = loaded_engine()
        with patch("engines.whisper_openai.resolve_whisper_device", return_value="cpu"):
            engine.load()
        self.assertEqual(len(module.load_calls), 1)

    def test_load_fn_overrides_the_module(self):
        model = FakeModel()
        engine = WhisperOpenAIEngine("base", load_fn=lambda device: model)
        with patch("engines.whisper_openai.resolve_whisper_device", return_value="cpu"):
            engine.load()
        self.assertTrue(engine.is_loaded)

    def test_missing_whisper_runtime_raises_engine_unavailable(self):
        engine = WhisperOpenAIEngine("base")
        with patch(
            "engines.whisper_openai._import_whisper",
            side_effect=EngineUnavailableError("no whisper"),
        ):
            with self.assertRaises(EngineUnavailableError):
                engine.load()
        self.assertFalse(engine.is_loaded)

    def test_import_whisper_maps_import_error_to_engine_unavailable(self):
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "whisper":
                raise ImportError("No module named 'whisper'")
            return real_import(name, *args, **kwargs)

        engine = WhisperOpenAIEngine("base")
        with patch.object(builtins, "__import__", blocked_import):
            with self.assertRaises(EngineUnavailableError):
                engine.load()


class TranscribeTests(unittest.TestCase):
    def test_transcribe_before_load_raises(self):
        engine = WhisperOpenAIEngine("base", whisper_module=fake_whisper_module())
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(Path("/tmp/clip.wav"))

    def test_transcript_fields_come_from_the_whisper_result(self):
        engine, _ = loaded_engine()
        transcript = engine.transcribe(Path("/tmp/clip.wav"))

        self.assertEqual(transcript.text, "bonjour le monde")
        self.assertEqual(transcript.language, "fr")
        self.assertEqual(transcript.duration_s, 2.5)
        self.assertEqual(transcript.engine_id, "whisper_openai")
        self.assertEqual(len(transcript.segments), 2)
        self.assertEqual(transcript.segments[0].start, 0.0)
        self.assertEqual(transcript.segments[0].end, 1.25)
        self.assertEqual(transcript.segments[0].text, "bonjour")

    def test_transcript_without_segments_has_no_duration(self):
        module = fake_whisper_module(FakeModel({"text": "hi"}))
        engine, _ = loaded_engine(module)
        transcript = engine.transcribe(Path("/tmp/clip.wav"))

        self.assertEqual(transcript.text, "hi")
        self.assertIsNone(transcript.language)
        self.assertIsNone(transcript.duration_s)
        self.assertEqual(transcript.segments, ())

    def test_path_is_passed_to_whisper_as_a_string(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"))
        audio_path, _ = module.model.calls[0]
        self.assertEqual(audio_path, "/tmp/clip.wav")
        self.assertIsInstance(audio_path, str)

    def test_dictation_decoding_defaults_are_preserved(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"))
        _, kwargs = module.model.calls[0]
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertEqual(kwargs["no_speech_threshold"], 0.6)
        self.assertFalse(kwargs["fp16"])

    def test_long_form_conditions_the_decoder_on_previous_text(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"), long_form=True)
        _, kwargs = module.model.calls[0]
        self.assertTrue(kwargs["condition_on_previous_text"])

    def test_long_form_keeps_the_other_decoding_parameters(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"), long_form=True)
        _, kwargs = module.model.calls[0]
        self.assertEqual(kwargs["no_speech_threshold"], 0.6)
        self.assertFalse(kwargs["fp16"])

    def test_language_none_is_passed_through_for_auto_detection(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"), language=None)
        _, kwargs = module.model.calls[0]
        self.assertIsNone(kwargs["language"])

    def test_language_auto_sentinel_becomes_none(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"), language=LANGUAGE_AUTO)
        _, kwargs = module.model.calls[0]
        self.assertIsNone(kwargs["language"])

    def test_explicit_language_reaches_whisper(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"), language="nl")
        _, kwargs = module.model.calls[0]
        self.assertEqual(kwargs["language"], "nl")

    def test_no_initial_prompt_kwarg_without_hints(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"))
        _, kwargs = module.model.calls[0]
        self.assertNotIn("initial_prompt", kwargs)

    def test_empty_hints_do_not_add_an_initial_prompt(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"), hints=Hints())
        _, kwargs = module.model.calls[0]
        self.assertNotIn("initial_prompt", kwargs)

    def test_hint_prompt_maps_to_initial_prompt(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"), hints=Hints(initial_prompt="Murmur notes."))
        _, kwargs = module.model.calls[0]
        self.assertEqual(kwargs["initial_prompt"], "Murmur notes.")

    def test_vocabulary_is_folded_into_the_initial_prompt(self):
        engine, module = loaded_engine()
        engine.transcribe(
            Path("/tmp/clip.wav"),
            hints=Hints(vocabulary=("Murmur", "Boske"), initial_prompt="Notes."),
        )
        _, kwargs = module.model.calls[0]
        self.assertEqual(kwargs["initial_prompt"], "Notes. Murmur, Boske")

    def test_vocabulary_only_still_produces_a_prompt(self):
        engine, module = loaded_engine()
        engine.transcribe(Path("/tmp/clip.wav"), hints=Hints(vocabulary=("Murmur",)))
        _, kwargs = module.model.calls[0]
        self.assertEqual(kwargs["initial_prompt"], "Murmur")

    def test_hints_applied_is_true_when_a_prompt_was_sent(self):
        engine, _ = loaded_engine()
        transcript = engine.transcribe(Path("/tmp/clip.wav"), hints=Hints(vocabulary=("Murmur",)))
        self.assertIs(transcript.hints_applied, True)

    def test_hints_applied_is_unknown_when_no_prompt_was_built(self):
        engine, _ = loaded_engine()
        self.assertIsNone(engine.transcribe(Path("/tmp/clip.wav")).hints_applied)
        self.assertIsNone(engine.transcribe(Path("/tmp/clip.wav"), hints=Hints()).hints_applied)


class InfoAndUnloadTests(unittest.TestCase):
    def test_info_describes_the_adapter(self):
        info = WhisperOpenAIEngine("medium").info()
        self.assertEqual(info.id, "whisper_openai")
        self.assertEqual(info.name, "OpenAI Whisper (PyTorch)")
        self.assertEqual(info.model_id, "medium")
        self.assertEqual(info.size_bytes, 0)
        self.assertEqual(info.languages, (LANGUAGE_AUTO,))
        self.assertFalse(info.supports_streaming)
        self.assertTrue(info.supports_hints)
        self.assertFalse(info.requires_apple_silicon)

    def test_runtime_summary_reports_the_device_and_precision(self):
        engine, _ = loaded_engine(device="cpu")
        self.assertEqual(engine.runtime_summary(), "device=cpu fp16=False")

    def test_unload_drops_the_model_and_is_idempotent(self):
        engine, _ = loaded_engine()
        with patch("engines.whisper_openai._empty_mps_cache") as empty_cache:
            engine.unload()
            engine.unload()
        self.assertFalse(engine.is_loaded)
        self.assertEqual(empty_cache.call_count, 2)

    def test_transcribe_after_unload_raises(self):
        engine, _ = loaded_engine()
        with patch("engines.whisper_openai._empty_mps_cache"):
            engine.unload()
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(Path("/tmp/clip.wav"))

    def test_empty_mps_cache_is_a_no_op_without_torch(self):
        import builtins

        from engines.whisper_openai import _empty_mps_cache

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", blocked_import):
            _empty_mps_cache()

    def test_empty_mps_cache_calls_torch_when_mps_is_available(self):
        calls = []
        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: True)
            ),
            mps=types.SimpleNamespace(empty_cache=lambda: calls.append("emptied")),
        )
        from engines.whisper_openai import _empty_mps_cache

        with patch.dict("sys.modules", {"torch": fake_torch}):
            _empty_mps_cache()
        self.assertEqual(calls, ["emptied"])


class RegistryTests(unittest.TestCase):
    def test_create_engine_builds_the_adapter_with_a_fake_module(self):
        module = fake_whisper_module()
        engine = create_engine("whisper_openai", model_name="base", whisper_module=module)
        self.assertIsInstance(engine, WhisperOpenAIEngine)
        self.assertEqual(engine.info().model_id, "base")
        self.assertFalse(engine.is_loaded)


if __name__ == "__main__":
    unittest.main()
