import builtins
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from engines import (
    LANGUAGE_AUTO,
    Engine,
    EngineError,
    EngineInfo,
    EngineNotLoadedError,
    EngineUnavailableError,
    Hints,
    Segment,
    Transcript,
    create_engine,
)
from engines import voxtral_mlx
from engines.voxtral_mlx import (
    ENGINE_ID,
    LANGUAGES,
    VoxtralMlxEngine,
    _MlxAudioRuntime,
)
from services.model_profile_service import CHIP_APPLE_SILICON, CHIP_INTEL


def write_wav(path: Path, *, rate: int = 16000, channels: int = 1, width: int = 2, frames: int = 1600) -> Path:
    """Write a silent WAV. Defaults are the only shape the engine accepts."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * (frames * channels * width))
    return path


class FakeSession:
    """Records fed PCM and hands back canned deltas, one per step()."""

    def __init__(self, per_chunk=(), tail=(), never_done: bool = False):
        self.per_chunk = list(per_chunk)
        self.tail = list(tail)
        self.never_done = never_done
        self.fed: list[bytes] = []
        self.closed = False
        self.closes = 0
        self.steps = 0

    @property
    def done(self) -> bool:
        if self.never_done:
            return False
        return self.closed and not self.tail

    def feed_pcm(self, chunk: bytes) -> None:
        assert not self.closed, "fed after close"
        self.fed.append(chunk)

    def close(self) -> None:
        """Idempotent, like the real session: the engine closes it from a finally."""
        self.closes += 1
        self.closed = True

    def step(self) -> list[str]:
        self.steps += 1
        queue = self.tail if self.closed else self.per_chunk
        return [queue.pop(0)] if queue else []


class FakeRuntime:
    """Injectable stand-in for mlx-audio; records every call the engine makes."""

    def __init__(self, result=None, session=None, import_error=None):
        self.result = result if result is not None else SimpleNamespace(text="hello")
        self.session = session
        self.import_error = import_error
        self.model = object()
        self.imports = 0
        self.loaded_paths: list[Path] = []
        self.transcribe_calls: list[dict] = []
        self.session_models: list[object] = []
        self.cache_clears = 0

    def import_backend(self) -> None:
        self.imports += 1
        if self.import_error is not None:
            raise self.import_error

    def load_model(self, model_path):
        self.loaded_paths.append(model_path)
        return self.model

    def transcribe(self, model, wav_path, *, language, hints):
        self.transcribe_calls.append(
            {"model": model, "wav_path": wav_path, "language": language, "hints": hints}
        )
        return self.result

    def create_session(self, model):
        self.session_models.append(model)
        assert self.session is not None, "no session configured on this FakeRuntime"
        return self.session

    def clear_cache(self) -> None:
        self.cache_clears += 1


class VoxtralTestCase(unittest.TestCase):
    """Shared temp model directory and WAV fixture."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.model_path = self.tmp_path / "Voxtral-Mini-4B-Realtime-2602-4bit"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_bytes(b"{}")
        self.wav_path = write_wav(self.tmp_path / "clip.wav")

    def tearDown(self):
        self._tmp.cleanup()

    def loaded_engine(self, runtime=None) -> VoxtralMlxEngine:
        runtime = runtime if runtime is not None else FakeRuntime()
        engine = VoxtralMlxEngine(self.model_path, runtime=runtime)
        with patch("engines.voxtral_mlx.detect_chip", return_value=CHIP_APPLE_SILICON):
            engine.load()
        return engine


class ContractTests(VoxtralTestCase):
    def test_engine_implements_the_base_contract(self):
        engine = VoxtralMlxEngine(self.model_path, runtime=FakeRuntime())
        self.assertIsInstance(engine, Engine)
        self.assertTrue(engine.supports_streaming)
        self.assertFalse(engine.supports_hints)
        self.assertFalse(engine.is_loaded)

    def test_module_exposes_engine_class_for_the_registry(self):
        self.assertIs(voxtral_mlx.ENGINE_CLASS, VoxtralMlxEngine)

    def test_default_runtime_is_the_mlx_audio_wrapper(self):
        engine = VoxtralMlxEngine(self.model_path)
        self.assertIsInstance(engine._runtime, _MlxAudioRuntime)

    def test_created_through_the_registry_with_an_injected_runtime(self):
        runtime = FakeRuntime()
        engine = create_engine("voxtral_mlx", model_path=self.model_path, runtime=runtime)
        self.assertIsInstance(engine, VoxtralMlxEngine)
        self.assertIs(engine._runtime, runtime)
        self.assertEqual(engine.model_path, self.model_path)


class LoadTests(VoxtralTestCase):
    def test_load_imports_the_backend_then_the_model(self):
        runtime = FakeRuntime()
        engine = self.loaded_engine(runtime)
        self.assertTrue(engine.is_loaded)
        self.assertEqual(runtime.imports, 1)
        self.assertEqual(runtime.loaded_paths, [self.model_path])

    def test_load_is_idempotent(self):
        runtime = FakeRuntime()
        engine = self.loaded_engine(runtime)
        with patch("engines.voxtral_mlx.detect_chip", return_value=CHIP_APPLE_SILICON):
            engine.load()
        self.assertEqual(runtime.imports, 1)
        self.assertEqual(len(runtime.loaded_paths), 1)

    def test_load_uses_the_shared_chip_detection(self):
        engine = VoxtralMlxEngine(self.model_path, runtime=FakeRuntime())
        with patch("engines.voxtral_mlx.detect_chip", return_value=CHIP_INTEL) as detect:
            with self.assertRaises(EngineUnavailableError) as ctx:
                engine.load()
        detect.assert_called_once_with()
        self.assertIn("Apple Silicon", str(ctx.exception))
        self.assertFalse(engine.is_loaded)

    def test_load_on_intel_names_the_reported_architecture(self):
        engine = VoxtralMlxEngine(self.model_path, runtime=FakeRuntime())
        # detect_chip() reads platform.machine(), so patching it drives both.
        with patch("engines.voxtral_mlx.platform.machine", return_value="x86_64"):
            with self.assertRaises(EngineUnavailableError) as ctx:
                engine.load()
        self.assertIn("Apple Silicon", str(ctx.exception))
        self.assertIn("x86_64", str(ctx.exception))
        self.assertFalse(engine.is_loaded)

    def test_load_without_a_model_directory_raises_engine_unavailable(self):
        missing = self.tmp_path / "not-downloaded"
        engine = VoxtralMlxEngine(missing, runtime=FakeRuntime())
        with patch("engines.voxtral_mlx.detect_chip", return_value=CHIP_APPLE_SILICON):
            with self.assertRaises(EngineUnavailableError) as ctx:
                engine.load()
        self.assertIn(str(missing), str(ctx.exception))

    def test_load_translates_a_runtime_import_error(self):
        runtime = FakeRuntime(import_error=ImportError("No module named 'mlx'"))
        engine = VoxtralMlxEngine(self.model_path, runtime=runtime)
        with patch("engines.voxtral_mlx.detect_chip", return_value=CHIP_APPLE_SILICON):
            with self.assertRaises(EngineUnavailableError) as ctx:
                engine.load()
        self.assertIn("mlx-audio", str(ctx.exception))
        self.assertFalse(engine.is_loaded)

    def test_default_runtime_load_fails_actionably_when_mlx_is_absent(self):
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name.startswith("mlx"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        engine = VoxtralMlxEngine(self.model_path)
        with patch("engines.voxtral_mlx.detect_chip", return_value=CHIP_APPLE_SILICON):
            with patch("builtins.__import__", side_effect=blocked_import):
                with self.assertRaises(EngineUnavailableError) as ctx:
                    engine.load()
        message = str(ctx.exception)
        self.assertIn("mlx", message)
        self.assertIn("pip install", message)

    def test_unload_drops_the_model_and_clears_the_runtime_cache(self):
        runtime = FakeRuntime()
        engine = self.loaded_engine(runtime)
        engine.unload()
        self.assertFalse(engine.is_loaded)
        self.assertEqual(runtime.cache_clears, 1)
        engine.unload()
        self.assertEqual(runtime.cache_clears, 2)


class TranscribeTests(VoxtralTestCase):
    def test_transcribe_before_load_raises_not_loaded(self):
        engine = VoxtralMlxEngine(self.model_path, runtime=FakeRuntime())
        with self.assertRaises(EngineNotLoadedError):
            engine.transcribe(self.wav_path)

    def test_transcribe_builds_a_transcript(self):
        runtime = FakeRuntime(result=SimpleNamespace(text="  bonjour le monde  "))
        engine = self.loaded_engine(runtime)
        transcript = engine.transcribe(self.wav_path, language="fr")
        self.assertIsInstance(transcript, Transcript)
        self.assertEqual(transcript.text, "bonjour le monde")
        self.assertEqual(transcript.language, "fr")
        self.assertEqual(transcript.engine_id, ENGINE_ID)
        self.assertAlmostEqual(transcript.duration_s, 0.1)
        self.assertEqual(transcript.segments, ())
        self.assertEqual(runtime.transcribe_calls[0]["wav_path"], self.wav_path)
        self.assertIs(runtime.transcribe_calls[0]["model"], runtime.model)

    def test_language_auto_reports_unknown_when_the_runtime_detects_nothing(self):
        engine = self.loaded_engine()
        transcript = engine.transcribe(self.wav_path, language=LANGUAGE_AUTO)
        self.assertIsNone(transcript.language)

    def test_detected_language_from_the_runtime_wins(self):
        runtime = FakeRuntime(result=SimpleNamespace(text="hoi", language="nl"))
        engine = self.loaded_engine(runtime)
        transcript = engine.transcribe(self.wav_path, language=LANGUAGE_AUTO)
        self.assertEqual(transcript.language, "nl")

    def test_segments_are_mapped_when_the_runtime_returns_them(self):
        runtime = FakeRuntime(
            result=SimpleNamespace(
                text="one two",
                segments=[{"start": 0.0, "end": 0.5, "text": "one"}, {"start": 0.5, "end": 1.0, "text": "two"}],
            )
        )
        engine = self.loaded_engine(runtime)
        transcript = engine.transcribe(self.wav_path)
        self.assertEqual(
            transcript.segments,
            (Segment(start=0.0, end=0.5, text="one"), Segment(start=0.5, end=1.0, text="two")),
        )

    def test_unreadable_segments_fail_loudly(self):
        runtime = FakeRuntime(result=SimpleNamespace(text="x", segments=[object()]))
        engine = self.loaded_engine(runtime)
        with self.assertRaises(EngineError):
            engine.transcribe(self.wav_path)

    def test_hints_are_reported_as_not_applied_because_voxtral_cannot_bias(self):
        runtime = FakeRuntime()
        engine = self.loaded_engine(runtime)
        hints = Hints(vocabulary=("Murmur", "Boske"), initial_prompt="Dictation for Canopy Studio.")
        transcript = engine.transcribe(self.wav_path, hints=hints)
        self.assertFalse(engine.supports_hints)
        self.assertIs(transcript.hints_applied, False)

    def test_hints_applied_stays_none_without_anything_to_bias_with(self):
        engine = self.loaded_engine()
        self.assertIsNone(engine.transcribe(self.wav_path).hints_applied)
        self.assertIsNone(engine.transcribe(self.wav_path, hints=Hints()).hints_applied)

    def test_long_form_is_accepted_and_ignored(self):
        runtime = FakeRuntime()
        engine = self.loaded_engine(runtime)
        transcript = engine.transcribe(self.wav_path, long_form=True)
        self.assertEqual(transcript.text, "hello")
        self.assertEqual(len(runtime.transcribe_calls), 1)


class WavFormatTests(VoxtralTestCase):
    def assert_rejects(self, path: Path, *needles: str):
        engine = self.loaded_engine()
        with self.assertRaises(EngineError) as ctx:
            engine.transcribe(path)
        message = str(ctx.exception)
        for needle in needles:
            self.assertIn(needle, message)

    def test_rejects_the_wrong_sample_rate(self):
        path = write_wav(self.tmp_path / "44k.wav", rate=44100)
        self.assert_rejects(path, "44100 Hz", "16000 Hz mono 16-bit")

    def test_rejects_stereo(self):
        path = write_wav(self.tmp_path / "stereo.wav", channels=2)
        self.assert_rejects(path, "2 channel(s)")

    def test_rejects_the_wrong_sample_width(self):
        path = write_wav(self.tmp_path / "8bit.wav", width=1)
        self.assert_rejects(path, "8-bit")

    def test_rejects_a_missing_file(self):
        self.assert_rejects(self.tmp_path / "nope.wav", "file not found")

    def test_rejects_a_file_that_is_not_a_wav(self):
        path = self.tmp_path / "notawav.wav"
        path.write_bytes(b"this is not a RIFF header")
        self.assert_rejects(path, "WAV file")

    def test_format_is_checked_before_the_runtime_is_called(self):
        runtime = FakeRuntime()
        engine = self.loaded_engine(runtime)
        with self.assertRaises(EngineError):
            engine.transcribe(write_wav(self.tmp_path / "48k.wav", rate=48000))
        self.assertEqual(runtime.transcribe_calls, [])


class StreamTests(VoxtralTestCase):
    def test_stream_before_load_raises_not_loaded(self):
        engine = VoxtralMlxEngine(self.model_path, runtime=FakeRuntime(session=FakeSession()))
        with self.assertRaises(EngineNotLoadedError):
            engine.stream([b"\x00\x00"])

    def test_stream_yields_growing_partials_and_a_final(self):
        session = FakeSession(per_chunk=["hel", "lo "], tail=["world"])
        engine = self.loaded_engine(FakeRuntime(session=session))
        partials = list(engine.stream([b"\x00\x00", b"\x01\x00"]))
        self.assertEqual([p.text for p in partials], ["hel", "hello ", "hello world", "hello world"])
        self.assertEqual([p.is_final for p in partials], [False, False, False, True])
        self.assertEqual(session.fed, [b"\x00\x00", b"\x01\x00"])
        self.assertTrue(session.closed)
        self.assertGreaterEqual(session.closes, 1)

    def test_stream_over_no_audio_still_ends_with_a_final(self):
        session = FakeSession()
        engine = self.loaded_engine(FakeRuntime(session=session))
        partials = list(engine.stream([]))
        self.assertEqual(len(partials), 1)
        self.assertTrue(partials[0].is_final)
        self.assertEqual(partials[0].text, "")
        self.assertTrue(session.closed)

    def test_stream_skips_empty_chunks(self):
        session = FakeSession(per_chunk=["hi"])
        engine = self.loaded_engine(FakeRuntime(session=session))
        list(engine.stream([b"", b"\x00\x00"]))
        self.assertEqual(session.fed, [b"\x00\x00"])

    def test_stream_closes_the_session_when_the_consumer_stops_early(self):
        session = FakeSession(per_chunk=["hel", "lo"], tail=["!"])
        engine = self.loaded_engine(FakeRuntime(session=session))
        stream = engine.stream([b"\x00\x00", b"\x01\x00"])
        self.assertEqual(next(stream).text, "hel")
        stream.close()
        self.assertTrue(session.closed)

    def test_stream_closes_the_session_when_a_chunk_is_rejected(self):
        session = FakeSession()
        engine = self.loaded_engine(FakeRuntime(session=session))
        with self.assertRaises(EngineError):
            list(engine.stream([b"\x00"]))
        self.assertTrue(session.closed)

    def test_stream_rejects_a_partial_sample(self):
        engine = self.loaded_engine(FakeRuntime(session=FakeSession()))
        with self.assertRaises(EngineError) as ctx:
            list(engine.stream([b"\x00"]))
        self.assertIn("16-bit PCM", str(ctx.exception))

    def test_stream_rejects_non_bytes_chunks(self):
        engine = self.loaded_engine(FakeRuntime(session=FakeSession()))
        with self.assertRaises(EngineError):
            list(engine.stream(["not bytes"]))

    def test_stream_fails_loudly_when_the_session_never_finishes(self):
        session = FakeSession(never_done=True)
        engine = self.loaded_engine(FakeRuntime(session=session))
        with patch("engines.voxtral_mlx.MAX_DRAIN_STEPS", 3):
            with self.assertRaises(EngineError) as ctx:
                list(engine.stream([b"\x00\x00"]))
        self.assertIn("never finished", str(ctx.exception))


class InfoTests(VoxtralTestCase):
    def test_info_fields(self):
        (self.model_path / "weights.safetensors").write_bytes(b"x" * 1024)
        info = VoxtralMlxEngine(self.model_path, runtime=FakeRuntime()).info()
        self.assertIsInstance(info, EngineInfo)
        self.assertEqual(info.id, "voxtral_mlx")
        self.assertEqual(info.name, "Voxtral Mini 4B Realtime (MLX, 4-bit)")
        self.assertEqual(info.model_id, "Voxtral-Mini-4B-Realtime-2602-4bit")
        self.assertEqual(info.size_bytes, 1024 + 2)  # weights + config.json
        self.assertTrue(info.supports_streaming)
        self.assertFalse(info.supports_hints)
        self.assertTrue(info.requires_apple_silicon)

    def test_info_languages_cover_the_model_card(self):
        info = VoxtralMlxEngine(self.model_path, runtime=FakeRuntime()).info()
        self.assertEqual(info.languages, LANGUAGES)
        self.assertEqual(info.languages[0], LANGUAGE_AUTO)
        for code in ("en", "fr", "de", "nl", "es", "it", "pt", "ar", "hi", "ja", "ko", "ru", "zh"):
            self.assertIn(code, info.languages, code)
        self.assertEqual(len(set(info.languages)), 14)

    def test_info_reports_zero_bytes_before_download(self):
        info = VoxtralMlxEngine(self.tmp_path / "absent", runtime=FakeRuntime()).info()
        self.assertEqual(info.size_bytes, 0)
        self.assertEqual(info.model_id, "absent")

    def test_info_works_without_loading(self):
        engine = VoxtralMlxEngine(self.model_path, runtime=FakeRuntime())
        self.assertFalse(engine.is_loaded)
        self.assertEqual(engine.info().id, ENGINE_ID)


class HintPlumbingTests(unittest.TestCase):
    """The honest-biasing behaviour: what the default runtime does with Hints."""

    def test_the_runtime_has_no_hint_capability_hook_to_guess_with(self):
        self.assertFalse(hasattr(_MlxAudioRuntime(), "supports_hints"))
        self.assertFalse(hasattr(voxtral_mlx, "biasing_parameter"))
        self.assertFalse(hasattr(voxtral_mlx, "BIASING_PARAMETER_NAMES"))

    def test_default_runtime_does_not_smuggle_hints_into_kwargs(self):
        class VoxtralLike:
            def __init__(self):
                self.calls = []

            def generate(self, audio, *, temperature=0.0, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(text="ok")

        model = VoxtralLike()
        _MlxAudioRuntime().transcribe(
            model, Path("/tmp/clip.wav"), language="fr", hints=Hints(initial_prompt="Notes.")
        )
        self.assertEqual(model.calls, [{}])

    def test_clear_cache_is_safe_before_import(self):
        _MlxAudioRuntime().clear_cache()

    def test_load_model_before_import_backend_asserts(self):
        with self.assertRaises(AssertionError):
            _MlxAudioRuntime().load_model(Path("/tmp/model"))


if __name__ == "__main__":
    unittest.main()
