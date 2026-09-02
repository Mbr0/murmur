#!/usr/bin/env python3
"""Bake-off harness for Murmur v2 decision D1 (primary local engine).

Runs one or more engines (whisper.cpp, Voxtral MLX) over the same fixture
clips and writes the markdown table consumed by
``docs/work/active/2026-09-02-murmur-v2/decisions.md`` plus a JSON sidecar
of per-clip results.

Only goes through ``engines.create_engine`` — never imports a concrete
engine module directly, so this script runs even when a given runtime
(MLX, a bundled binary) is absent on the machine; that engine's row becomes
"unavailable: <reason>" instead of aborting the run.

Deviation from the plan: MASTER.md names ``jiwer`` for WER, but
``requirements.txt`` is a hot file shared across Wave 0 agents and this is a
dev-only tool, so WER is implemented here with the stdlib (word-level
Levenshtein over normalised tokens) instead of adding a dependency. See
``word_error_rate`` below; it is unit-tested in ``tests/test_bakeoff.py``.

A ``--engine`` SPEC is a local path, never a Hugging Face repo id: whisper.cpp
takes the ``.bin`` model file, Voxtral takes the model *directory*. Both are
what ``ModelStore.engine_model_path(model_id)`` returns for the corresponding
catalog entry, so use that if you are unsure where a downloaded model landed.

Process isolation: peak RAM comes from ``resource.getrusage``, whose
``ru_maxrss`` is a **process-lifetime high-water mark** and never resets. So a
single process cannot measure two engines independently — engine 2 would
inherit engine 1's ``RUSAGE_CHILDREN`` peak while its own ``RUSAGE_SELF``
delta read ~0, attributing RAM to the wrong engine in the very column D1 is
decided on. When more than one ``--engine`` is given, this script therefore
re-runs itself once per engine (see ``run_bakeoff_isolated``), each child
writing a ``--json-out`` sidecar, and merges the rows into one table. A single
``--engine`` run measures in-process, which is already correct.

Usage (MODELS is where the model store keeps downloads):
    MODELS="$HOME/Library/Application Support/Murmur/models"
    venv/bin/python scripts/tools/bakeoff.py \\
        --engine whispercpp=/path/to/ggml-large-v3-turbo.bin \\
        --engine "voxtral_mlx=$MODELS/voxtral-mini-4b-realtime-4bit" \\
        --fixtures tests/fixtures/audio \\
        --out docs/work/active/2026-09-02-murmur-v2/bakeoff-table.md \\
        --runs 3
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

# Running this file directly (``python scripts/tools/bakeoff.py``) only puts
# scripts/tools/ on sys.path, not the repo root where the engines package
# lives. Add it so `import engines` works regardless of invocation style.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from engines import create_engine  # noqa: E402
from engines.base import EngineUnavailableError  # noqa: E402

#: Languages the bake-off covers, matching the decisions.md table columns.
SUPPORTED_LANGUAGES = ("en", "fr", "nl", "de")

#: Clip duration range (seconds) counted toward the median-latency column.
_LATENCY_DURATION_MIN_S = 8.0
_LATENCY_DURATION_MAX_S = 12.0

_TABLE_HEADER = (
    "| Engine | Model | EN WER | FR WER | NL WER | DE WER "
    "| Median latency (10 s clip) | Peak RAM |"
)
_TABLE_SEPARATOR = (
    "|--------|-------|--------|--------|--------|--------"
    "|----------------------------|----------|"
)


class ManifestError(ValueError):
    """Raised when a fixture manifest is missing required or valid data."""


class ChildRunError(RuntimeError):
    """Raised when a per-engine child process exits non-zero; carries its stderr."""


@dataclass(frozen=True)
class Clip:
    """One fixture clip: its audio file, language, and reference transcript."""

    path: Path
    language: str
    text: str


# ---------------------------------------------------------------------------
# Word error rate
# ---------------------------------------------------------------------------

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace, split into words."""
    lowered = text.lower()
    without_punctuation = _PUNCTUATION_RE.sub("", lowered)
    return without_punctuation.split()


def _levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Word-level edit distance; equals substitutions + deletions + insertions."""
    n, m = len(reference), len(hypothesis)
    previous_row = list(range(m + 1))
    for i in range(1, n + 1):
        current_row = [i] + [0] * m
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                current_row[j] = previous_row[j - 1]
            else:
                current_row[j] = 1 + min(
                    previous_row[j - 1],  # substitution
                    previous_row[j],  # deletion
                    current_row[j - 1],  # insertion
                )
        previous_row = current_row
    return previous_row[m]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = (substitutions + deletions + insertions) / len(reference words).

    Both strings are normalised first (lowercase, punctuation stripped,
    whitespace collapsed), so casing and punctuation differences never count
    as errors. Raises ``ValueError`` when the reference has no words after
    normalisation — WER is undefined against an empty reference.
    """
    reference_tokens = normalize_tokens(reference)
    hypothesis_tokens = normalize_tokens(hypothesis)
    if not reference_tokens:
        raise ValueError("reference text is empty after normalisation")
    distance = _levenshtein_distance(reference_tokens, hypothesis_tokens)
    return distance / len(reference_tokens)


# ---------------------------------------------------------------------------
# Fixture manifest
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> list[Clip]:
    """Load and validate ``manifest.json``; clip paths resolve against its directory.

    Raises ``FileNotFoundError`` when the manifest itself is missing (with a
    pointer to the fixtures README) and ``ManifestError`` for structurally
    invalid entries, e.g. an unsupported ``language``.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "See tests/fixtures/audio/README.md to record clips or run "
            "scripts/tools/make_synthetic_fixtures.sh for a smoke-test set."
        )
    data = json.loads(manifest_path.read_text())
    clips_data = data.get("clips")
    if not isinstance(clips_data, list) or not clips_data:
        raise ManifestError(f"Manifest {manifest_path} has no non-empty 'clips' list")

    fixtures_dir = manifest_path.parent
    clips: list[Clip] = []
    for index, entry in enumerate(clips_data):
        path = entry.get("path") if isinstance(entry, dict) else None
        language = entry.get("language") if isinstance(entry, dict) else None
        text = entry.get("text") if isinstance(entry, dict) else None

        if not path or not isinstance(path, str):
            raise ManifestError(f"Clip #{index} is missing a string 'path'")
        if language not in SUPPORTED_LANGUAGES:
            raise ManifestError(
                f"Clip #{index} ({path}) has unsupported language {language!r}; "
                f"expected one of {SUPPORTED_LANGUAGES}"
            )
        if not text or not isinstance(text, str):
            raise ManifestError(f"Clip #{index} ({path}) is missing reference 'text'")

        clips.append(Clip(path=fixtures_dir / path, language=language, text=text))
    return clips


# ---------------------------------------------------------------------------
# Clip duration and RAM measurement
# ---------------------------------------------------------------------------


def get_wav_duration_s(path: Path) -> float:
    """Duration of a WAV file in seconds, from its frame count and rate."""
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate) if rate else 0.0


def _ru_maxrss_self() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _ru_maxrss_children() -> int:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def measure_peak_ram_gb(baseline_self_bytes: int) -> float:
    """Peak RAM in GB attributable to one engine's create/load/transcribe/unload.

    ``(RUSAGE_SELF.ru_maxrss - baseline)`` catches in-process engines (e.g.
    MLX); ``RUSAGE_CHILDREN.ru_maxrss`` catches engines that shell out to a
    bundled server binary (e.g. whisper.cpp's whisper-server) — both are
    summed since either or both may be in play. macOS reports ``ru_maxrss``
    in bytes (Linux reports KiB); this harness targets macOS only, matching
    the rest of Murmur.

    Both marks are process-lifetime high-water marks that never reset, so this
    is only meaningful for **one** engine per process. Multi-engine runs fan
    out to a child process per engine (``run_bakeoff_isolated``) for exactly
    that reason.
    """
    self_delta = max(0, _ru_maxrss_self() - baseline_self_bytes)
    children_peak = max(0, _ru_maxrss_children())
    return (self_delta + children_peak) / (1024**3)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def mean_wer_by_language(per_clip: list[dict]) -> dict[str, float]:
    """Mean WER per language, only for languages that appear in ``per_clip``."""
    by_language: dict[str, list[float]] = {}
    for result in per_clip:
        by_language.setdefault(result["language"], []).append(result["wer"])
    return {language: statistics.mean(values) for language, values in by_language.items()}


def median_latency_in_duration_bucket(
    per_clip: list[dict],
    min_s: float = _LATENCY_DURATION_MIN_S,
    max_s: float = _LATENCY_DURATION_MAX_S,
) -> float | None:
    """Median latency over clips whose audio duration falls in [min_s, max_s].

    Returns ``None`` when no clip's duration falls in the bucket.
    """
    latencies = [
        result["latency_s"] for result in per_clip if min_s <= result["duration_s"] <= max_s
    ]
    return statistics.median(latencies) if latencies else None


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


def _format_wer(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _format_latency(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def _format_ram(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} GB"


def render_table(results: "OrderedDict[str, dict]") -> str:
    """Render the decisions.md bake-off table from per-engine result rows.

    Each value in ``results`` is either ``{"unavailable": reason}`` or a row
    with ``model``, ``wer`` (dict of language -> mean WER), and optionally
    ``latency_median_s`` / ``peak_ram_gb``.
    """
    lines = [_TABLE_HEADER, _TABLE_SEPARATOR]
    for engine_id, row in results.items():
        if "unavailable" in row:
            lines.append(f"| {engine_id} | unavailable: {row['unavailable']} | | | | | | |")
            continue
        wer = row.get("wer", {})
        lines.append(
            "| {engine} | {model} | {en} | {fr} | {nl} | {de} | {latency} | {ram} |".format(
                engine=engine_id,
                model=row.get("model", ""),
                en=_format_wer(wer.get("en")),
                fr=_format_wer(wer.get("fr")),
                nl=_format_wer(wer.get("nl")),
                de=_format_wer(wer.get("de")),
                latency=_format_latency(row.get("latency_median_s")),
                ram=_format_ram(row.get("peak_ram_gb")),
            )
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def parse_engine_arg(raw: str) -> tuple[str, str]:
    """Parse one repeatable ``--engine id=spec`` value."""
    engine_id, separator, spec = raw.partition("=")
    engine_id = engine_id.strip()
    spec = spec.strip()
    if not separator or not engine_id or not spec:
        raise ValueError(f"--engine expects 'id=spec' (model path or model name), got {raw!r}")
    return engine_id, spec


def run_engine_bakeoff(
    engine_id: str,
    spec: str,
    clips: list[Clip],
    runs: int = 1,
    show_transcripts: bool = False,
) -> dict:
    """Run one engine over every clip and return its result row.

    An ``EngineUnavailableError`` from ``create_engine`` or ``load()`` is
    caught and turned into ``{"unavailable": reason}`` so the caller can
    continue with the next engine; any other exception propagates and
    aborts the whole bake-off, per the harness contract.
    """
    baseline_self_bytes = _ru_maxrss_self()
    try:
        engine = create_engine(engine_id, model_path=spec)
        engine.load()
    except EngineUnavailableError as exc:
        return {"unavailable": str(exc)}

    try:
        per_clip: list[dict] = []
        for clip in clips:
            latencies = []
            transcript_text = ""
            for _ in range(max(1, runs)):
                start = time.perf_counter()
                transcript = engine.transcribe(clip.path, language=clip.language)
                latencies.append(time.perf_counter() - start)
                transcript_text = transcript.text
            result = {
                "path": str(clip.path),
                "language": clip.language,
                "latency_s": statistics.median(latencies),
                "wer": word_error_rate(clip.text, transcript_text),
                "duration_s": get_wav_duration_s(clip.path),
            }
            if show_transcripts:
                result["transcript"] = transcript_text
            per_clip.append(result)
    finally:
        engine.unload()

    return {
        "model": spec,
        "wer": mean_wer_by_language(per_clip),
        "latency_median_s": median_latency_in_duration_bucket(per_clip),
        "peak_ram_gb": measure_peak_ram_gb(baseline_self_bytes),
        "per_clip": per_clip,
    }


def run_bakeoff(
    engine_args: list[str],
    fixtures_dir: Path,
    runs: int = 1,
    show_transcripts: bool = False,
) -> "OrderedDict[str, dict]":
    """Run every requested engine over the fixtures manifest and return results.

    ``results[engine_id]`` includes a ``per_clip`` list (dropped by
    ``render_table``) for the JSON sidecar written by ``main``.
    """
    clips = load_manifest(Path(fixtures_dir) / "manifest.json")
    results: "OrderedDict[str, dict]" = OrderedDict()
    for raw in engine_args:
        engine_id, spec = parse_engine_arg(raw)
        results[engine_id] = run_engine_bakeoff(
            engine_id, spec, clips, runs=runs, show_transcripts=show_transcripts
        )
    return results


# ---------------------------------------------------------------------------
# Process isolation (one child per engine)
# ---------------------------------------------------------------------------


def merge_engine_results(
    per_engine_results: "list[dict] | tuple[dict, ...]",
) -> "OrderedDict[str, dict]":
    """Merge the per-engine result mappings from isolated child runs.

    Each input is one child's ``--json-out`` payload (``{engine_id: row}``).
    Insertion order across the inputs is preserved so the combined table lists
    engines in the order they were requested on the command line, and every row
    keeps the RAM figure measured inside its own process.
    """
    merged: "OrderedDict[str, dict]" = OrderedDict()
    for results in per_engine_results:
        for engine_id, row in results.items():
            merged[engine_id] = row
    return merged


def build_child_command(
    engine_arg: str,
    fixtures_dir: Path | str,
    runs: int,
    json_out: Path | str,
    out: Path | str | None = None,
    show_transcripts: bool = False,
) -> list[str]:
    """argv for a child run measuring exactly one engine.

    ``engine_arg`` is the raw ``id=spec`` string as typed by the user; passing
    a single ``--engine`` keeps the child on the in-process path so it never
    fans out again.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--engine",
        engine_arg,
        "--fixtures",
        str(fixtures_dir),
        "--runs",
        str(runs),
        "--json-out",
        str(json_out),
    ]
    if out is not None:
        command += ["--out", str(out)]
    if show_transcripts:
        command.append("--show-transcripts")
    return command


def run_bakeoff_isolated(
    engine_args: list[str],
    fixtures_dir: Path | str,
    runs: int = 1,
    show_transcripts: bool = False,
) -> "OrderedDict[str, dict]":
    """Run each engine in its own child process and merge the per-engine rows.

    Needed because ``ru_maxrss`` is a process-lifetime high-water mark: measured
    in one process, the second engine's "Peak RAM" would be the first engine's.
    Fails fast — the first child to exit non-zero raises ``ChildRunError``
    carrying its stderr, and no later engine is started.
    """
    parsed = [parse_engine_arg(raw) for raw in engine_args]  # validate before spawning
    per_engine_results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="bakeoff-") as tmp:
        tmp_dir = Path(tmp)
        for index, (raw, (engine_id, _spec)) in enumerate(zip(engine_args, parsed)):
            json_out = tmp_dir / f"{index}-{engine_id}.json"
            command = build_child_command(
                raw,
                fixtures_dir,
                runs,
                json_out,
                out=tmp_dir / f"{index}-{engine_id}.md",
                show_transcripts=show_transcripts,
            )
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise ChildRunError(
                    f"bake-off child for {engine_id!r} exited "
                    f"{completed.returncode}:\n{completed.stderr}"
                )
            per_engine_results.append(json.loads(json_out.read_text()))
    return merge_engine_results(per_engine_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Murmur v2 engine bake-off harness (decision D1).",
    )
    parser.add_argument(
        "--engine",
        action="append",
        default=[],
        metavar="ID=SPEC",
        help=(
            "Repeatable. ID is an engines.ENGINE_IDS value "
            "(whispercpp, voxtral_mlx); SPEC is a local model "
            "path (whispercpp: the .bin file; voxtral_mlx: the model "
            "directory, not a Hugging Face repo id). "
            "e.g. --engine whispercpp=/path/to/ggml-large-v3-turbo.bin. "
            "Passing several runs each engine in its own process so peak RAM "
            "is attributed correctly."
        ),
    )
    parser.add_argument(
        "--fixtures",
        default="tests/fixtures/audio",
        help="Directory containing manifest.json (default: tests/fixtures/audio)",
    )
    parser.add_argument(
        "--out",
        default="bakeoff-table.md",
        help="Markdown table output path; a .json sidecar is written alongside it",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help=(
            "Write the full per-engine result rows (model, WER, latency, peak "
            "RAM, per-clip detail) to this path. Used by the per-engine child "
            "processes of a multi-engine run; also useful on its own."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Transcription repeats per clip; latency is the median across runs",
    )
    parser.add_argument(
        "--show-transcripts",
        action="store_true",
        help="Print and record transcript text (off by default; never logged otherwise)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.engine:
        parser.error("at least one --engine id=spec is required")

    if len(args.engine) > 1:
        # ru_maxrss is a process-lifetime high-water mark, so one engine per
        # process; see run_bakeoff_isolated.
        try:
            results = run_bakeoff_isolated(
                args.engine, args.fixtures, runs=args.runs, show_transcripts=args.show_transcripts
            )
        except ChildRunError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        results = run_bakeoff(
            args.engine, args.fixtures, runs=args.runs, show_transcripts=args.show_transcripts
        )

    if args.show_transcripts:
        for engine_id, row in results.items():
            for clip_result in row.get("per_clip", []):
                print(f"[{engine_id}] {clip_result['path']}: {clip_result.get('transcript', '')}")

    table = render_table(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)

    sidecar = {engine_id: row.get("per_clip", []) for engine_id, row in results.items()}
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")

    if args.json_out:
        json_out_path = Path(args.json_out)
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(json.dumps(results, indent=2) + "\n")

    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
