import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import engines
from engines import (
    ENGINE_IDS,
    LANGUAGE_AUTO,
    Engine,
    EngineCapabilityError,
    EngineError,
    EngineInfo,
    EngineNotLoadedError,
    EngineUnavailableError,
    Hints,
    Partial,
    Segment,
    Transcript,
    create_engine,
    register_engine,
    unregister_engine,
)


class FakeEngine(Engine):
    """Minimal concrete engine used to exercise the abstract base class."""

    supports_hints = True

    def __init__(self, model_id: str = "fake-model") -> None:
        self._model_id = model_id
        self._model: object | None = None
        self.last_long_form: bool | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        self._model = object()

    def unload(self) -> None:
        self._model = None

    def info(self) -> EngineInfo:
        return EngineInfo(
            id="fake",
            name="Fake",
            model_id=self._model_id,
            size_bytes=42,
            languages=(LANGUAGE_AUTO,),
            supports_streaming=self.supports_streaming,
            supports_hints=self.supports_hints,
            requires_apple_silicon=False,
        )

    def _transcribe(self, wav_path, language, hints, long_form) -> Transcript:
        self.last_long_form = long_form
        return Transcript(
            text="hello",
            language=language,
            duration_s=1.5,
            segments=(Segment(start=0.0, end=1.5, text="hello"),),
            engine_id="fake",
        )


class StreamingFakeEngine(FakeEngine):
    supports_streaming = True

    def _stream(self, chunks):
        for index, chunk in enumerate(chunks):
            yield Partial(text=chunk.decode(), is_final=False, start_s=float(index))
        yield Partial(text="done", is_final=True)


class DataclassTests(unittest.TestCase):
    def test_transcript_carries_segments_and_engine_id(self):
        transcript = Transcript(
            text="bonjour",
            language="fr",
            duration_s=2.0,
            segments=(Segment(start=0.0, end=2.0, text="bonjour"),),
            engine_id="whispercpp",
        )
        self.assertEqual(transcript.text, "bonjour")
        self.assertEqual(transcript.language, "fr")
        self.assertEqual(transcript.duration_s, 2.0)
        self.assertEqual(transcript.engine_id, "whispercpp")
        self.assertEqual(transcript.segments[0].end, 2.0)

    def test_transcript_allows_unknown_language_and_duration(self):
        transcript = Transcript(
            text="",
            language=None,
            duration_s=None,
            segments=(),
            engine_id="fake",
        )
        self.assertIsNone(transcript.language)
        self.assertIsNone(transcript.duration_s)
        self.assertEqual(transcript.segments, ())

    def test_transcript_hints_applied_is_unknown_by_default(self):
        transcript = Transcript(
            text="", language=None, duration_s=None, segments=(), engine_id="fake"
        )
        self.assertIsNone(transcript.hints_applied)

    def test_transcript_records_whether_hints_reached_the_decoder(self):
        applied = Transcript(
            text="", language=None, duration_s=None, segments=(), engine_id="fake",
            hints_applied=True,
        )
        ignored = Transcript(
            text="", language=None, duration_s=None, segments=(), engine_id="fake",
            hints_applied=False,
        )
        self.assertIs(applied.hints_applied, True)
        self.assertIs(ignored.hints_applied, False)

    def test_partial_defaults_to_no_timing(self):
        partial = Partial(text="hel", is_final=False)
        self.assertFalse(partial.is_final)
        self.assertIsNone(partial.start_s)
        self.assertIsNone(partial.end_s)

    def test_hints_default_to_empty(self):
        hints = Hints()
        self.assertEqual(hints.vocabulary, ())
        self.assertIsNone(hints.initial_prompt)


class HintsPromptTextTests(unittest.TestCase):
    """``Hints.as_prompt_text`` is the one place hints become a prompt string."""

    def test_prompt_then_vocabulary_joined_by_one_space(self):
        hints = Hints(vocabulary=("Murmur", "Boske"), initial_prompt="Notes.")
        self.assertEqual(hints.as_prompt_text(), "Notes. Murmur, Boske")

    def test_each_half_alone(self):
        self.assertEqual(Hints(vocabulary=("Murmur",)).as_prompt_text(), "Murmur")
        self.assertEqual(Hints(initial_prompt="Notes.").as_prompt_text(), "Notes.")

    def test_surrounding_whitespace_is_stripped(self):
        hints = Hints(vocabulary=("  Murmur  ", "Boske"), initial_prompt="  Notes.  ")
        self.assertEqual(hints.as_prompt_text(), "Notes. Murmur, Boske")

    def test_blank_vocabulary_terms_are_dropped(self):
        hints = Hints(vocabulary=("", "   ", "Murmur"))
        self.assertEqual(hints.as_prompt_text(), "Murmur")

    def test_none_when_nothing_remains(self):
        self.assertIsNone(Hints().as_prompt_text())
        self.assertIsNone(Hints(vocabulary=("", "  "), initial_prompt="   ").as_prompt_text())

    def test_engine_info_fields(self):
        info = EngineInfo(
            id="voxtral_mlx",
            name="Voxtral Mini",
            model_id="voxtral-mini-4b-realtime-4bit",
            size_bytes=1234,
            languages=("en", "fr"),
            supports_streaming=True,
            supports_hints=True,
            requires_apple_silicon=True,
        )
        self.assertEqual(info.languages, ("en", "fr"))
        self.assertTrue(info.requires_apple_silicon)


class EngineContractTests(unittest.TestCase):
    def test_incomplete_subclass_fails_at_instantiation(self):
        class Incomplete(Engine):
            pass

        with self.assertRaises(TypeError):
            Incomplete()

    def test_transcribe_requires_a_loaded_engine(self):
        engine = FakeEngine()
        self.assertFalse(engine.is_loaded)
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(Path("/tmp/clip.wav"))

    def test_transcribe_returns_transcript_when_loaded(self):
        engine = FakeEngine()
        engine.load()
        transcript = engine.transcribe(Path("/tmp/clip.wav"), language="nl", hints=Hints(vocabulary=("Murmur",)))
        self.assertIsInstance(transcript, Transcript)
        self.assertEqual(transcript.text, "hello")
        self.assertEqual(transcript.language, "nl")
        self.assertEqual(transcript.engine_id, "fake")

    def test_unload_makes_the_engine_unusable_again(self):
        engine = FakeEngine()
        engine.load()
        engine.unload()
        self.assertFalse(engine.is_loaded)
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(Path("/tmp/clip.wav"))

    def test_transcribe_defaults_to_dictation_not_long_form(self):
        engine = FakeEngine()
        engine.load()
        engine.transcribe(Path("/tmp/clip.wav"))
        self.assertIs(engine.last_long_form, False)

    def test_long_form_reaches_the_engine_implementation(self):
        engine = FakeEngine()
        engine.load()
        engine.transcribe(Path("/tmp/clip.wav"), long_form=True)
        self.assertIs(engine.last_long_form, True)

    def test_capability_flags_default_to_false(self):
        self.assertFalse(Engine.supports_streaming)
        self.assertFalse(Engine.supports_hints)

    def test_runtime_summary_is_empty_unless_an_engine_overrides_it(self):
        self.assertEqual(FakeEngine().runtime_summary(), "")

    def test_stream_on_non_streaming_engine_raises_capability_error(self):
        engine = FakeEngine()
        engine.load()
        with self.assertRaises(EngineCapabilityError):
            engine.stream([b"chunk"])

    def test_stream_checks_capability_before_load_state(self):
        engine = FakeEngine()
        with self.assertRaises(EngineCapabilityError):
            engine.stream([b"chunk"])

    def test_stream_requires_a_loaded_engine(self):
        engine = StreamingFakeEngine()
        with self.assertRaises(EngineNotLoadedError):
            engine.stream([b"chunk"])

    def test_stream_yields_partials(self):
        engine = StreamingFakeEngine()
        engine.load()
        partials = list(engine.stream([b"one", b"two"]))
        self.assertEqual([p.text for p in partials], ["one", "two", "done"])
        self.assertEqual([p.is_final for p in partials], [False, False, True])

    def test_streaming_engine_without_stream_implementation_fails_loudly(self):
        class BrokenStreamer(FakeEngine):
            supports_streaming = True

        engine = BrokenStreamer()
        engine.load()
        with self.assertRaises(NotImplementedError):
            engine.stream([b"one"])

    def test_info_reports_capabilities(self):
        info = StreamingFakeEngine().info()
        self.assertTrue(info.supports_streaming)
        self.assertTrue(info.supports_hints)
        self.assertEqual(info.languages, (LANGUAGE_AUTO,))


class ExceptionHierarchyTests(unittest.TestCase):
    def test_every_engine_error_shares_a_base(self):
        for error in (EngineNotLoadedError, EngineCapabilityError, EngineUnavailableError):
            self.assertTrue(issubclass(error, EngineError))
        self.assertTrue(issubclass(EngineError, Exception))


class RegistryTests(unittest.TestCase):
    def tearDown(self):
        unregister_engine("whispercpp")
        unregister_engine("test_fake")

    def test_engine_ids(self):
        self.assertEqual(ENGINE_IDS, ("whispercpp", "voxtral_mlx"))

    def test_unknown_engine_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            create_engine("nope")

    def test_registered_factory_is_used_with_kwargs(self):
        register_engine("whispercpp", lambda **kwargs: FakeEngine(**kwargs))
        engine = create_engine("whispercpp", model_id="registered")
        self.assertIsInstance(engine, FakeEngine)
        self.assertEqual(engine.info().model_id, "registered")

    def test_registered_factory_may_use_an_id_outside_engine_ids(self):
        register_engine("test_fake", FakeEngine)
        self.assertIsInstance(create_engine("test_fake"), FakeEngine)

    def test_unregister_is_idempotent(self):
        unregister_engine("test_fake")
        with self.assertRaises(ValueError):
            create_engine("test_fake")

    def test_module_is_imported_lazily_and_engine_class_instantiated(self):
        module = types.ModuleType("engines.voxtral_mlx")
        module.ENGINE_CLASS = FakeEngine
        with patch.dict(sys.modules, {"engines.voxtral_mlx": module}):
            engine = create_engine("voxtral_mlx", model_id="lazy")
        self.assertIsInstance(engine, FakeEngine)
        self.assertEqual(engine.info().model_id, "lazy")

    def test_import_module_is_only_called_when_creating(self):
        module = types.ModuleType("engines.voxtral_mlx")
        module.ENGINE_CLASS = FakeEngine
        with patch("engines.importlib.import_module", return_value=module) as mock_import:
            self.assertEqual(mock_import.call_count, 0)
            create_engine("voxtral_mlx")
            mock_import.assert_called_once_with("engines.voxtral_mlx")

    def test_missing_engine_class_raises_engine_unavailable(self):
        module = types.ModuleType("engines.whispercpp")
        with patch.dict(sys.modules, {"engines.whispercpp": module}):
            with self.assertRaises(EngineUnavailableError):
                create_engine("whispercpp")

    def test_unimportable_module_raises_engine_unavailable(self):
        with patch("engines.importlib.import_module", side_effect=ImportError("mlx missing")):
            with self.assertRaises(EngineUnavailableError):
                create_engine("voxtral_mlx")

    def test_package_exports_the_public_contract(self):
        for name in (
            "Engine",
            "EngineInfo",
            "Transcript",
            "Partial",
            "Segment",
            "Hints",
            "EngineError",
            "EngineNotLoadedError",
            "EngineCapabilityError",
            "EngineUnavailableError",
            "LANGUAGE_AUTO",
            "ENGINE_IDS",
            "create_engine",
            "register_engine",
            "unregister_engine",
        ):
            self.assertIn(name, engines.__all__, name)


if __name__ == "__main__":
    unittest.main()
