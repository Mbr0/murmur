"""Every decision the app takes, as a function that takes no ``self``.

Moved out of ``murmur.py`` in Wave 5 with their names, their bodies and their
docstrings unchanged — the test suite imports them by name and asserts the
tables they encode.

The rule that keeps this module worth having: nothing here touches AppKit, the
menu bar, a thread or the disk. The one exception is
:func:`boske_http_transport`, which is a transport by definition and lives here
because it is the licence service's constructor argument and nothing else.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from cleanup.coding_mode import transform_spoken_code
from cleanup.context import resolve_mode
from cleanup.modes import (
    DEFAULT_MODE_ID,
    MODE_IDS,
    MODES,
    TONE_IDS,
    mode_from_config,
    render_system_prompt,
    tone_from_config,
)
from cleanup.snippets import expand_snippets, snippets_from_config
from cleanup.transcription_filters import is_likely_hallucination
from cleanup.vocabulary import Vocabulary, apply_replacements
from engines._http import open_no_cross_host_redirect
from engines.byok import ByokAuthError, ByokRateLimited
from engines.cloud import ALLOWANCE_MESSAGE
from engines.factory import (
    CONFIG_BYOK_MODEL,
    CONFIG_BYOK_PROVIDER,
    CONFIG_CLOUD_BASE_URL,
    byok_item_name,
    cloud_base_url,
)
from engines.model_store import ModelIntegrityError
from services.engine_router import (
    CLOUD_MODE_MURMUR,
    ENGINE_BYOK,
    ENGINE_CLOUD,
    Route,
    effective_vocabulary_terms,
)
from services.hotkey_service import ACTION_START, ACTION_STOP, KEY_UP_MODES
from services.license_service import (
    Entitlements,
    is_pro_feature_enabled,
    set_current_entitlements,
)
from services.persistence_service import (
    DEFAULT_CONFIG,
    ORIGIN_BYOK,
    ORIGIN_CLOUD,
    ORIGIN_LOCAL,
    resolve_cleanup_enabled,
)
from ui.onboarding_window import should_show

from app.config import APP_NAME, logger


def engine_is_ready(engine) -> bool:
    """Whether the engine exists and has finished loading.

    It is built inside :meth:`MurmurApp.load_model`, so it is None until that runs
    and a construction failure is reported through the same UI as a load failure.
    """
    return engine is not None and engine.is_loaded


def should_reject_toggle(*, loading: bool, is_processing: bool, model_ready: bool) -> bool:
    """Whether hotkey/menu toggle must be ignored."""
    return loading or is_processing or not model_ready


def should_toggle_for_press_action(action: str | None, *, is_recording: bool) -> bool:
    """Whether a PressController action needs the recorder toggled.

    The controller decides what the press means; this decides whether the app is
    already in that state. Unknown actions raise instead of being ignored.
    """
    if action is None:
        return False
    if action == ACTION_START:
        return not is_recording
    if action == ACTION_STOP:
        return is_recording
    raise ValueError(f"Unknown press action: {action!r}")


def should_reject_upload(
    *, loading: bool, is_recording: bool, is_processing: bool, model_ready: bool
) -> bool:
    """Whether file upload/transcribe must be ignored."""
    return loading or is_recording or is_processing or not model_ready


def should_apply_ready_on_reset(*, is_recording: bool) -> bool:
    """Whether menu reset may force the ready/idle UI state."""
    return not is_recording


def resolve_mic_device_index(
    saved_index: object, input_device_indices: set[int] | frozenset[int]
) -> int | None:
    """Return persisted mic index, or None for system default. Fail fast if invalid."""
    if saved_index is None:
        return None
    if isinstance(saved_index, bool) or not isinstance(saved_index, int):
        raise ValueError(f"Invalid mic_device_index type: {type(saved_index).__name__}")
    if saved_index not in input_device_indices:
        raise ValueError(f"Microphone device index {saved_index} is not available")
    return saved_index


def resolve_mic_device(
    saved_index: object,
    saved_name: object,
    input_devices: dict[int, str],
) -> int | None:
    """Resolve persisted mic by index+name. Prefer name when index drifted; never accept a mismatched device at the saved index."""
    if saved_name is not None and not isinstance(saved_name, str):
        raise ValueError(f"Invalid mic_device_name type: {type(saved_name).__name__}")
    name = saved_name.strip() if isinstance(saved_name, str) and saved_name.strip() else None

    if saved_index is None and name is None:
        return None
    if saved_index is not None and (
        isinstance(saved_index, bool) or not isinstance(saved_index, int)
    ):
        raise ValueError(f"Invalid mic_device_index type: {type(saved_index).__name__}")

    if saved_index is not None and saved_index in input_devices:
        device_name = input_devices[saved_index]
        if name is None or device_name == name:
            return saved_index
        # Name mismatch at this index — do not accept; try resolve by name below.

    if name is not None:
        matches = [idx for idx, device_name in input_devices.items() if device_name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple microphones named {name!r}")
        raise ValueError(f"Microphone {name!r} is not available")

    raise ValueError(f"Microphone device index {saved_index} is not available")


def mic_selection_changes(device_index: int | None, device_name: str | None) -> dict:
    """The two config keys that record which microphone is in use.

    A *changes* dict for :meth:`~services.persistence_service.PersistenceService.update_config`,
    never a config: ``(None, None)`` clears the selection so a device that has
    gone is not restored on the next launch, and a real pair records the one
    actually in use. Returning the whole config here is what let a microphone
    write revert an engine swap.
    """
    return {"mic_device_index": device_index, "mic_device_name": device_name}


def skip_audio_user_message(duration_seconds: float, max_level: float) -> str:
    """Calm user-facing reason when short/quiet audio is skipped (no transcription text)."""
    if duration_seconds < 1.0:
        return "Recording was too short to transcribe."
    return "Recording was too quiet to transcribe."

#: Config key naming the engine's model; ``None`` until the user or the
#: defaults below fill it in.
CONFIG_ENGINE_ID = "engine_id"
CONFIG_MODEL_ID = "model_id"

#: The pre-Wave-1 config key, a bare openai-whisper size such as ``"medium"``.
#: Its presence is what marks a config as needing the one-off migration.
LEGACY_MODEL_KEY = "model"

#: Menu status while no speech model is on disk.
NO_MODEL_STATUS = "No speech model installed"

#: Menu status while the engine is being swapped.
SWITCHING_STATUS = "Switching engine…"

#: Where :func:`missing_model_action` sends a user with no model.
MISSING_MODEL_ONBOARDING = "onboarding"
MISSING_MODEL_SETTINGS = "settings"

#: Outcomes of :func:`reload_engine_decision`.
RELOAD_START = "start"
RELOAD_UNCHANGED = "unchanged"
RELOAD_BUSY = "busy"
RELOAD_RECORDING = "recording"

#: Why a reload was refused, in the user's words. A refusal is never silent.
RELOAD_REFUSAL_MESSAGES = {
    RELOAD_BUSY: "Murmur is still busy. Choose the model again in a moment.",
    RELOAD_RECORDING: "Stop recording before switching the speech engine.",
}

#: Config key remembering which engines already showed the "hints ignored"
#: notice: ``{engine_id: True}``. Shown once per engine, never per recording.
HINTS_NOTICE_KEY = "hints_notice_shown"


@dataclass(frozen=True)
class EngineSelection:
    """Which engine and model to load, and whether config has to catch up."""

    engine_id: str
    model_id: str
    #: True when config did not name both keys and must be written back.
    needs_persist: bool
    #: True when that write is the one-off migration off ``LEGACY_MODEL_KEY``.
    from_legacy_model_key: bool


def resolve_engine_selection(
    config: dict,
    *,
    default_engine_id: str,
    model_ids_for_engine,
) -> EngineSelection:
    """Resolve the engine and model to load from config, filling in defaults.

    A config that names both keys is honoured as-is. A missing engine falls
    back to this machine's default (chip and RAM, decision D1); a missing model
    falls back to the first catalog model of whichever engine won. Either gap
    means the resolved pair is written back, so the choice is made exactly once
    — including for a legacy config that only carried ``model``.

    An engine with no catalog model at all is a packaging error, not a user
    state, so it raises rather than silently picking another engine.
    """
    assert config is not None, "config is required"
    assert default_engine_id, "default_engine_id is required"

    engine_id = config.get(CONFIG_ENGINE_ID)
    if not isinstance(engine_id, str) or not engine_id:
        engine_id = default_engine_id

    model_id = config.get(CONFIG_MODEL_ID)
    if not isinstance(model_id, str) or not model_id:
        candidates = tuple(model_ids_for_engine(engine_id))
        if not candidates:
            raise ValueError(f"No catalog model for engine {engine_id!r}")
        model_id = candidates[0]

    needs_persist = (
        config.get(CONFIG_ENGINE_ID) != engine_id or config.get(CONFIG_MODEL_ID) != model_id
    )
    return EngineSelection(
        engine_id=engine_id,
        model_id=model_id,
        needs_persist=needs_persist,
        from_legacy_model_key=needs_persist and LEGACY_MODEL_KEY in config,
    )


def missing_model_action(config: dict) -> str:
    """Where to send a user whose chosen model is not downloaded.

    A Mac that never finished the wizard gets the wizard, which can download
    the model in place; anyone else gets Settings, where the same download
    lives. Neither path falls back to another engine behind the user's back.
    """
    assert config is not None, "config is required"
    return MISSING_MODEL_ONBOARDING if should_show(config) else MISSING_MODEL_SETTINGS


def model_unavailable_message(reason: str | None) -> str:
    """Body of the "cannot record/transcribe" notification.

    ``reason`` is the menu status when one explains the block (no model
    installed), and None when the engine simply failed to load.
    """
    if not reason:
        return "Recording is unavailable until the model loads successfully."
    return f"{reason}. Download one from Settings → Speech engine."


def model_status_title(display_name: str | None) -> str:
    """Title of the menu's engine status line."""
    return f"Model: {display_name}" if display_name else NO_MODEL_STATUS


def reload_engine_decision(
    *,
    requested: tuple[str, str],
    active: tuple[str | None, str | None],
    is_reloading: bool,
    is_recording: bool,
    is_processing: bool,
    engine_ready: bool,
    stream_active: bool = False,
) -> str:
    """Whether a requested engine swap may start now.

    Policy: refuse rather than queue. A queued swap would fire minutes later,
    long after the user stopped thinking about it, and a refusal that says so
    is easier to act on than a delayed surprise. Recording and transcription
    both hold the engine, so both block; a second request while one is already
    in flight is refused too.

    ``stream_active`` is the fourth holder and the least obvious one: when the
    batch path gives up waiting for the live decoder it clears both flags and
    finishes the utterance, while the abandoned worker is still inside
    ``engine.stream()``. The app looks idle and is not, so a swap there would
    call ``unload()`` on a model being read.
    """
    assert requested and len(requested) == 2, "requested is (engine_id, model_id)"
    if is_reloading:
        return RELOAD_BUSY
    if is_recording:
        return RELOAD_RECORDING
    if is_processing or stream_active:
        return RELOAD_BUSY
    if engine_ready and tuple(active) == tuple(requested):
        return RELOAD_UNCHANGED
    return RELOAD_START


def should_show_hints_notice(
    config: dict, engine_id: str, *, hints_applied: bool | None, has_terms: bool
) -> bool:
    """Whether to tell the user this engine ignored their vocabulary terms.

    Only when there were terms to ignore, only when the engine said outright
    that it did not use them (``False``, not the ``None`` that means "nothing
    to apply"), and only the first time for that engine.
    """
    assert config is not None, "config is required"
    assert engine_id, "engine_id is required"
    if not has_terms or hints_applied is not False:
        return False
    shown = config.get(HINTS_NOTICE_KEY) or {}
    return not bool(shown.get(engine_id))


def hints_notice_changes(config: dict, engine_id: str) -> dict:
    """The one config key that marks the notice as shown for ``engine_id``.

    A *changes* dict for :meth:`~services.persistence_service.PersistenceService.update_config`,
    carrying the whole ``hints_notice_shown`` map (the other engines' answers
    are kept) and nothing else. ``config`` is only read, for the map it already
    holds.
    """
    assert config is not None, "config is required"
    assert engine_id, "engine_id is required"
    shown = dict(config.get(HINTS_NOTICE_KEY) or {})
    shown[engine_id] = True
    return {HINTS_NOTICE_KEY: shown}


def hints_notice_message(engine_name: str) -> str:
    """The one-time notice itself. Names the engine, never the transcript."""
    assert engine_name, "engine_name is required"
    return f"Vocabulary hints are not supported by {engine_name}"


def push_to_talk_degraded_message(mode: str) -> str | None:
    """What to tell the user when the chosen press mode cannot run as chosen.

    ``hold`` and ``auto`` both need to see the key release, which macOS only
    delivers through an NSEvent monitor, which needs Accessibility. Without it
    Murmur runs the shortcut as ``toggle``. Saying so beats leaving the user
    holding a key that will never stop the recording. None when there is
    nothing to explain.
    """
    if mode not in KEY_UP_MODES:
        return None
    return (
        f"Push-to-talk “{mode}” needs Accessibility to see the key release. "
        "Until it is granted, the shortcut toggles recording on and off instead."
    )


def hotkey_registration_key(binding, mode: str, *, key_up_available: bool = True) -> tuple:
    """What a hotkey registration has to match to be left alone.

    The binding and the press mode, plus the one fact that can make a live
    registration wrong without either of them moving: whether the key-up the
    mode needs is actually being delivered. ``toggle`` never asked for one, so
    its absence is not a degradation.

    Built twice — once for the registration in hand and once for the config —
    and compared by :func:`should_reregister_hotkey`.
    """
    assert binding is not None, "binding is required"
    assert mode, "mode is required"
    satisfied = mode not in KEY_UP_MODES or bool(key_up_available)
    return (binding, mode, satisfied)


def should_reregister_hotkey(active_key, desired_key, *, registered: bool) -> bool:
    """Whether ``reload_hotkey`` must tear the shortcut down and make it again.

    Launch called it twice with the same answer. The 0.3 s startup timer and
    ``applicationDidBecomeActive`` both reach it, so whichever ran second
    unregistered a working Carbon hotkey and registered the identical one — two
    registrations for one shortcut, the "Carbon cannot deliver key-up" warning
    twice in every launch log, and a fresh :class:`PressController` dropped in
    under a key that may already be held down.

    So an unchanged binding over a live registration that is doing its job is a
    no-op. Everything else registers: no registration, one the app has lost
    track of, a binding or mode the user changed, and — the case that matters
    after *Enable Shortcut Permission…* — a live registration that is missing
    the key-up its mode needs.
    """
    if not registered or active_key is None:
        return True
    return active_key != desired_key


#: What quitting does, in the order it has to do it. Every step frees something
#: that outlives the process otherwise: a 2 GB ``llama-server`` child, a Carbon
#: hotkey macOS keeps until the app dies, a floating panel, a loaded model.
#:
#: The order is the whole of it. The live decoders are told to stop *first*,
#: because one of them is inside ``engine.stream()`` and unloading the model
#: under it is a crash in native code rather than an exception. The engine goes
#: after the pill, which is what a half-drawn partial is still writing into.
QUIT_STEPS: tuple[str, ...] = (
    "cancel_stream_workers",
    "stop_cleanup_runtime",
    "close_pill",
    "unload_engine",
    "unregister_hotkey",
)

#: The ceiling on everything a quit waits for, added up. A quit that takes
#: longer than this reads as a hang, and the OS reaps what is left anyway: every
#: worker is a daemon thread and the cleanup child is terminated, not joined.
QUIT_BUDGET_S = 3.0


def quit_time_remaining(elapsed_s: float, *, budget_s: float = QUIT_BUDGET_S) -> float:
    """How long the next quit step may block: what is left of the budget.

    Never negative, so a step that overran cannot hand the next one a wait that
    means "forever" once it is passed to ``Thread.join``.
    """
    return max(0.0, float(budget_s) - float(elapsed_s))


def finalize_transcript(
    raw_text: str,
    vocabulary,
    *,
    detect_hallucination=is_likely_hallucination,
    replace=apply_replacements,
) -> tuple[str, bool]:
    """Return ``(text to paste, was a hallucination)`` for one raw transcript.

    The filter reads the engine's own words, before the user's replacements
    rewrite them. Running it afterwards let a replacement hide a classic
    silence hallucination from the filter — and let one whose output happened
    to look like a hallucination suppress a real transcript.
    """
    assert raw_text is not None, "raw_text is required"
    hallucination = bool(detect_hallucination(raw_text))
    return replace(raw_text, vocabulary), hallucination


def reapply_replacements(text: str, vocabulary, *, replace=apply_replacements) -> str:
    """Run the user's replacements again over cleaned-up text.

    The cleanup pass rewrites sentences, and a rewrite re-cases words: a term
    the user spelled "Murmur" comes back "murmur" the moment the model starts a
    clause with it. The replacements are cheap and idempotent, so the cheapest
    honest fix is to apply them once more on the way to the clipboard.

    Deliberately *not* the hallucination filter. That reads the engine's own
    words (see :func:`finalize_transcript`); re-judging a sentence the model
    wrote would let its phrasing suppress a real transcript.
    """
    assert text is not None, "text is required"
    return replace(text, vocabulary)


def stream_text_for_token(result, token) -> str | None:
    """The live decoder's text, but only when it belongs to ``token``.

    Every utterance takes a number. The worker publishes ``(token, text)`` and
    the collector accepts it only while that number is still the current one.

    Without it: a stream that overran its join timeout was abandoned, kept
    running, and eventually wrote its text into the same slot — which the *next*
    utterance then read and pasted. The user said one thing and got the previous
    sentence. Returns None for a stale token, a stream that failed, or one that
    produced nothing but whitespace.
    """
    if result is None or token is None:
        return None
    result_token, text = result
    if result_token != token:
        return None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None

# ---------------------------------------------------------------------------
# Cleanup pipeline (Wave 2), gated by the licence (Wave 4)
# ---------------------------------------------------------------------------
#
# The Wave 2 placeholder ``pro_enabled(feature, config)`` is gone. Every gate in
# this file is now :func:`services.license_service.is_pro_feature_enabled`,
# called with a feature name and nothing else: the answer comes from the lease
# the licence service published, never from the config file. There is no
# developer override key — a hidden config flag that unlocked Pro was fine while
# the gate was a stand-in and would be a licence bypass now.

#: Menu status while the cleanup server is coming up for the first time.
CLEANUP_PREPARING_STATUS = "Preparing cleanup…"

#: Reason given when the GGUF the cleanup server needs is not on disk.
CLEANUP_MODEL_MISSING_REASON = "the cleanup model is not downloaded"

#: Reason given when the cleanup server would not come up at all.
CLEANUP_START_FAILED_REASON = "the cleanup server could not start"

#: Reason given when it comes up and dies again on the very next request.
CLEANUP_UNSTABLE_REASON = "the cleanup server keeps stopping"

#: Reason given when the server is still loading the model. Not a failure: the
#: start carries on in the background, so the next utterance is cleaned.
CLEANUP_NOT_READY_REASON = "the cleanup model is still loading"

#: Reason given for a request that arrives after the app has begun quitting.
CLEANUP_STOPPING_REASON = "Murmur is shutting the cleanup server down"

#: How long one utterance may wait for the cleanup server's *first* start.
#: The load itself is allowed up to ``LlamaServer.startup_timeout_s`` (60 s, and
#: the client retries once on a dead child, so ~120 s in the worst case) and it
#: keeps running in the background past this — but the user is standing there
#: holding a finished sentence, and eight seconds is already a long time to
#: watch a pill say nothing. Past it the raw text is pasted with a visible
#: notice and the *next* utterance gets the cleaned version.
CLEANUP_FIRST_USE_WAIT_S = 8.0

#: What to do about a cleanup that did not run. See :func:`cleanup_notice_kind`.
CLEANUP_NOTICE_NOTIFY = "notify"
CLEANUP_NOTICE_OFFER = "offer"

#: Menu entry that fetches the cleanup GGUF. It is not a speech engine, so the
#: Settings popup filters it out and this is its only permanent home.
CLEANUP_DOWNLOAD_MENU_TITLE = "Download cleanup model…"

#: Hidden config key: start the cleanup server at launch rather than on the
#: first utterance. Absent from ``DEFAULT_CONFIG`` on purpose — it costs 2 GB of
#: resident memory for a feature the user may not touch this session, so it is
#: opt-out for machines that can clearly afford it and invisible elsewhere.
CLEANUP_PREWARM_KEY = "cleanup_prewarm"
CLEANUP_PREWARM_DEFAULT = True

#: Below this, pre-warming competes with the speech model for RAM and the Mac
#: starts swapping mid-dictation. Matches the cleanup feature's own floor.
CLEANUP_PREWARM_MIN_RAM_GB = 16

#: Why cleanup did not run, when the answer is configuration rather than a
#: failure. These are the user's own settings, so they are logged, never shown.
CLEANUP_OFF_PRO = "Pro is not active"
CLEANUP_OFF_DISABLED = "cleanup is switched off"
CLEANUP_OFF_PASSTHROUGH = "the mode is verbatim dictation"

#: Menu item that toggles ``context_awareness`` rather than naming a mode.
MODE_MENU_AUTOMATIC = "Automatic (by app)"

#: The two language codes ``transform_spoken_code`` has trigger words for.
CODE_TRANSFORM_LANGUAGES = ("en", "fr")


#: Free-tier mode. Everything else the Smart tab offers is a Pro mode.
FREE_MODE_ID = DEFAULT_MODE_ID


def configured_mode_id(config: dict) -> str:
    """The mode the user pinned in Settings, ignoring the front app entirely.

    What a free install gets: :func:`~cleanup.context.resolve_mode` also reads
    the bundle-id table and the per-app overrides, and both of those are the
    ``context`` entitlement. A config naming a mode this build does not have
    falls back to dictation rather than taking the utterance down with it.
    """
    assert config is not None, "config is required"
    try:
        return mode_from_config(config).id
    except Exception as error:  # noqa: BLE001 - an unknown id is user data
        # The type only: the message quotes the mode id, which is user data.
        logger.warning("Ignoring an unreadable cleanup_mode: %s", type(error).__name__)
        return FREE_MODE_ID


def resolve_plan_mode(config: dict, context, *, pro=is_pro_feature_enabled) -> str:
    """The cleanup mode for this utterance, under the Pro gate.

    Two entitlements meet here, and they are separate on purpose:

    * ``context`` buys the *resolution* — the bundle-id table and the per-app
      overrides that pick a mode from what the user is typing into. Without it
      the configured default applies everywhere, which is what a single-mode
      install looks like.
    * ``modes`` buys the *modes themselves*. Free is verbatim dictation, so a
      config still naming Mail (a lapsed plan, or a hand-edited file) lands on
      dictation rather than quietly getting a paid rewrite.
    """
    assert config is not None, "config is required"
    assert callable(pro), "pro must be callable"
    mode_id = resolve_mode(context, config) if pro("context") else configured_mode_id(config)
    if mode_id != FREE_MODE_ID and not pro("modes"):
        return FREE_MODE_ID
    return mode_id


def gated_vocabulary(vocabulary, *, pro=is_pro_feature_enabled):
    """The vocabulary this install may actually use for one utterance.

    Terms past :data:`~cleanup.vocabulary.FREE_TERM_LIMIT` are dropped for this
    pass only — nothing is deleted from the user's list, so a subscription
    brings every term straight back. Replacements are untouched: the gate is
    named for terms, and silently dropping a user's spelling corrections would
    change what their transcript *says*, not how much of it is biased.
    """
    assert vocabulary is not None, "vocabulary is required"
    terms = effective_vocabulary_terms(vocabulary.terms, pro)
    if terms == tuple(vocabulary.terms):
        return vocabulary
    return Vocabulary(terms=terms, replacements=vocabulary.replacements)


def expand_gated_snippets(
    text: str,
    config: dict,
    *,
    pro=is_pro_feature_enabled,
    load=snippets_from_config,
    expand=expand_snippets,
) -> str:
    """Expand the user's spoken snippets, when the plan includes them.

    Runs on the transcript before cleanup, so the model sees the expanded text
    and can punctuate around it. Unreadable snippet data costs the expansion,
    never the transcript.
    """
    assert text is not None, "text is required"
    assert config is not None, "config is required"
    assert callable(pro), "pro must be callable"
    if not pro("snippets"):
        return text
    try:
        snippets = load(config)
    except Exception as error:  # noqa: BLE001 - snippets are user data
        # The type only: the message can carry the snippet trigger or its body.
        logger.warning(
            "Ignoring unreadable snippets in the config: %s", type(error).__name__
        )
        return text
    if not snippets:
        return text
    return expand(text, snippets)

# -- history origin ----------------------------------------------------------

#: Where each engine does its work. Only the exceptions are listed: an engine
#: that is not here runs on this Mac, which is what a speech engine is unless
#: it says otherwise. Keeping the table here — rather than an ``if engine_id ==
#: …`` at each call site — is what lets Wave 4 add a cloud engine by adding one
#: row, and is why nothing else in this file asks which engine is loaded before
#: writing history.
HISTORY_ORIGIN_BY_ENGINE: dict[str, str] = {
    ENGINE_CLOUD: ORIGIN_CLOUD,
    ENGINE_BYOK: ORIGIN_BYOK,
}


def history_origin_for(engine_id: str | None) -> str:
    """Which of ``local | cloud | byok`` an engine's transcriptions come from.

    An unknown or missing engine id reads as local. That is the honest default:
    every engine that ships today decodes on this Mac, and labelling a local
    transcription "cloud" would tell the user their audio left the machine when
    it did not.
    """
    if not engine_id:
        return ORIGIN_LOCAL
    return HISTORY_ORIGIN_BY_ENGINE.get(engine_id, ORIGIN_LOCAL)

# ---------------------------------------------------------------------------
# Engine routing and the licence (Wave 4)
# ---------------------------------------------------------------------------
#
# Three pure functions and one dataclass, so the whole routing table can be
# asserted as a table. The app method that calls them does nothing but gather
# ``self`` into arguments; every decision lives here.

#: Timeout for one Boske licence call. Every one of them is a small JSON
#: round trip, and none of them may hold a background thread open for minutes.
LICENSE_HTTP_TIMEOUT_S = 20.0

#: How often the background thread renews a lease and republishes entitlements.
ENTITLEMENT_REFRESH_INTERVAL_S = 6 * 3600

#: How long that thread sleeps between checks. Short enough that a sign-in in
#: Settings is reflected in the menu without waiting six hours for it.
ENTITLEMENT_POLL_INTERVAL_S = 60.0

#: How long to wait after a renewal that *failed*, and the ceiling that backoff
#: climbs to. Five minutes covers a dropped Wi-Fi or a proxy restart; an hour is
#: short enough that a lease which lapses is renewed well inside its grace week.
ENTITLEMENT_RETRY_BASE_S = 300.0
ENTITLEMENT_RETRY_MAX_S = 3600.0

#: Menu line naming the plan. One of three, and never a number.
ACCOUNT_STATUS_FREE = "Account: Free"
ACCOUNT_STATUS_PRO = "Account: Pro"
ACCOUNT_STATUS_PRO_GRACE = "Account: Pro (grace)"

#: Appended to the account line when the lease lives in memory only. See
#: :meth:`MurmurApp._build_license_service`.
ACCOUNT_STATUS_NOT_SAVED = " (not saved)"

#: Menu item that opens Settings on the Account tab.
SIGN_IN_MENU_TITLE = "Sign in with Boske ID…"


@dataclass(frozen=True)
class RemoteEngineKey:
    """Everything about the config a hosted engine was built from.

    The cloud and own-key engines are built once and kept, because building one
    is cheap but rebuilding it per utterance would re-read config on the
    dictation path. They must still follow a settings change, so the cached
    engine carries the config it was built from and is discarded the moment any
    of it differs. The lease is deliberately absent: it is read through a
    callable at request time, so a sign-out needs no rebuild.
    """

    engine_id: str
    base_url: str
    provider: str | None
    model: str | None


def remote_engine_key(engine_id: str, config: dict) -> RemoteEngineKey:
    """The cache key for the hosted engine ``engine_id`` under ``config``."""
    assert engine_id, "engine_id is required"
    assert config is not None, "config is required"
    provider = config.get(CONFIG_BYOK_PROVIDER)
    model = config.get(CONFIG_BYOK_MODEL)
    return RemoteEngineKey(
        engine_id=engine_id,
        base_url=cloud_base_url(config),
        provider=provider if isinstance(provider, str) and provider.strip() else None,
        model=model if isinstance(model, str) and model.strip() else None,
    )


def pinned_cloud_config(config: dict, base_url: str) -> dict:
    """``config`` as the hosted clients must read it: the proxy origin pinned.

    The proxy origin is read **once**, at launch, and the licence service is
    built against it for the session. Everything else that speaks to the proxy
    used to re-read ``cloud_base_url`` from the live config at request time, so
    an edit to that key mid-session pointed the audio, the transcript and the
    cleanup text at another host while the lease still belonged to the first —
    a redirect, and a lease handed to whoever the config now names.

    So the pinned value wins, and a changed key takes effect at the next
    launch. Returns ``config`` itself when the two already agree, which is the
    ordinary case and lets the caller detect the drift by identity.
    """
    assert config is not None, "config is required"
    assert base_url, "base_url is required"
    if cloud_base_url(config) == base_url:
        return config
    return {**config, CONFIG_CLOUD_BASE_URL: base_url}


#: Display names for the own-key providers, for the one notice that names one.
#: Deliberately not imported from the Settings tab: wording a notification must
#: not make the app depend on a UI module.
BYOK_PROVIDER_NAMES: dict[str, str] = {"mistral": "Mistral", "openai": "OpenAI"}

#: What the user is told when their **own** provider refuses a clip. The cloud
#: has its own wording in :mod:`services.engine_router`; these three are the
#: own-key half, and they name the provider because the key is the user's to fix.
NOTICE_KEY_REJECTED = "Your {provider} key was rejected; check Settings › Account"
NOTICE_KEY_RATE_LIMITED = "{provider} rate limited this request; using the local engine"
NOTICE_KEY_FAILED = "{provider} could not transcribe this; using the local engine"


def byok_provider_name(config: dict) -> str:
    """The own-key provider's display name, for a notice that names it."""
    assert config is not None, "config is required"
    provider = str(config.get(CONFIG_BYOK_PROVIDER) or "").strip().lower()
    if not provider:
        return "Your provider"
    return BYOK_PROVIDER_NAMES.get(provider, provider.title())


def after_byok_failure(exc: BaseException, *, local_engine_id: str, provider: str) -> Route:
    """Where to re-run a clip the user's **own** provider refused.

    Own-key failures used to be sent through
    :func:`~services.engine_router.after_cloud_failure`, which knows only the
    two proxy exceptions: a rejected key matched neither, so it fell back with
    no notice at all and a log line blaming Murmur Cloud. A revoked key then
    downgraded every dictation to the local engine, silently, forever.

    A rejected key is the one failure here the user can act on, so it says
    where to fix it. A rate limit is theirs to wait out, and anything else is
    told plainly rather than blamed on them. The caller shows each of these at
    most once a session — see :meth:`MurmurApp._announce_route`.
    """
    assert local_engine_id, "local_engine_id is required"
    assert provider, "provider is required"
    if isinstance(exc, ByokAuthError):
        return Route(
            local_engine_id, NOTICE_KEY_REJECTED.format(provider=provider), "byok key rejected"
        )
    if isinstance(exc, ByokRateLimited):
        return Route(
            local_engine_id,
            NOTICE_KEY_RATE_LIMITED.format(provider=provider),
            "byok rate limited",
        )
    return Route(local_engine_id, NOTICE_KEY_FAILED.format(provider=provider), "byok failed")


def own_key_present(keychain, config: dict) -> bool:
    """Whether the own-key provider named in ``config`` has a key stored.

    Every failure reads as "no key": an unreachable Keychain must send the
    dictation to the local engine with the "add your key" notice, not raise on
    the recording path.
    """
    assert config is not None, "config is required"
    if keychain is None:
        return False
    provider = config.get(CONFIG_BYOK_PROVIDER) or DEFAULT_CONFIG[CONFIG_BYOK_PROVIDER]
    if not isinstance(provider, str) or not provider.strip():
        return False
    try:
        return bool(keychain.has(byok_item_name(provider)))
    except Exception as error:  # noqa: BLE001 - the Keychain backend raises widely
        logger.warning("Could not ask the Keychain for the own key: %s", type(error).__name__)
        return False


def lease_is_present(license_service) -> bool:
    """Whether a usable lease is stored right now.

    ``current_lease_token`` is already strict — expired, tampered with or issued
    to another Mac all read as None — so this only has to survive the service
    being absent or the Keychain being locked.
    """
    if license_service is None:
        return False
    try:
        return license_service.current_lease_token() is not None
    except Exception as error:  # noqa: BLE001 - the secret store raises widely
        logger.warning("Could not read the stored lease: %s", type(error).__name__)
        return False


def notice_to_show(notice: str | None, *, fallback_pending: bool) -> str | None:
    """The routing notice to actually put on screen, or None.

    The allowance notice is the one the folder rules cap: cloud → local is the
    only fallback Murmur takes on its own, and it says so **once** per billing
    period. Every other notice answers a choice the user just made (own key with
    no key, not signed in) and is shown whenever it applies.
    """
    if not notice:
        return None
    if notice == ALLOWANCE_MESSAGE and not fallback_pending:
        return None
    return notice


def should_consume_trial(entitlements) -> bool:
    """Whether a finished cloud clip spends the free trial rather than the plan.

    An account with ``cloud_voice`` is paying for its minutes and must not have
    its one-time trial drained by them; an account without it reached the proxy
    *because* of the trial, so the trial is what it spends.
    """
    return not bool(getattr(entitlements, "cloud_voice", False))


def should_refresh_allowance(usage, *, cloud_mode: str) -> bool:
    """Whether to re-read ``GET /v1/voice/usage`` after this dictation.

    Only for Murmur Cloud — no other mode meters anything — and only when the
    cached reading is too old to act on. Off the dictation path by construction:
    the answer is used to start a thread once a transcript is already pasted.
    """
    if usage is None:
        return False
    if str(cloud_mode or "").strip() != CLOUD_MODE_MURMUR:
        return False
    try:
        return bool(usage.allowance_is_stale())
    except Exception as error:  # noqa: BLE001 - a corrupt config must not raise here
        logger.warning("Could not check the cloud allowance age: %s", type(error).__name__)
        return False


def should_refresh_entitlements(
    *,
    last_refresh_at: float | None,
    now: float,
    interval_s: float = ENTITLEMENT_REFRESH_INTERVAL_S,
) -> bool:
    """Whether the background thread should renew the lease now.

    True at startup (nothing has been refreshed yet) and every ``interval_s``
    after that. A clock that jumped backwards refreshes rather than waiting the
    difference out: an extra request is cheaper than a lease that never renews.
    """
    assert interval_s > 0, "interval_s must be positive"
    if last_refresh_at is None:
        return True
    elapsed = now - last_refresh_at
    return elapsed < 0 or elapsed >= interval_s


def next_refresh_delay(
    attempt: int,
    *,
    base_s: float = ENTITLEMENT_RETRY_BASE_S,
    max_s: float = ENTITLEMENT_RETRY_MAX_S,
) -> float:
    """How long to wait before retrying a lease renewal that just failed.

    ``attempt`` is 1 for the first failure in a row. The wait doubles from
    :data:`ENTITLEMENT_RETRY_BASE_S` and stops at :data:`ENTITLEMENT_RETRY_MAX_S`:
    a renewal that failed used to stamp the clock anyway and then wait the full
    six hours, which turned a dropped connection at launch into an afternoon on
    the free tier. Backing off rather than retrying every minute is what keeps
    a proxy that is down from being hammered by every running copy of Murmur.
    """
    assert attempt >= 1, "attempt counts from 1"
    assert base_s > 0, "base_s must be positive"
    assert max_s >= base_s, "max_s must not be below base_s"
    return min(base_s * (2 ** (attempt - 1)), max_s)


def account_menu_title(entitlements, *, store_is_volatile: bool = False) -> str:
    """The menu's one-line account status: Free, Pro, or Pro in its grace week.

    ``store_is_volatile`` is the Keychain being unreachable, which means the
    lease behind that line lives in memory and dies with the process. The line
    says so rather than showing a plan the next launch will not have.
    """
    if entitlements is None or not getattr(entitlements, "pro", False):
        title = ACCOUNT_STATUS_FREE
    elif getattr(entitlements, "in_grace", False):
        title = ACCOUNT_STATUS_PRO_GRACE
    else:
        title = ACCOUNT_STATUS_PRO
    return f"{title}{ACCOUNT_STATUS_NOT_SAVED}" if store_is_volatile else title


def publish_entitlements(license_service) -> Any:
    """Re-read the stored lease and publish it to the one Pro gate.

    Returns the entitlements, or None when there is no licence service to ask.
    Never raises: a locked Keychain drops the app to the free tier for this
    pass, which is the safe direction, and says so in the log.
    """
    if license_service is None:
        set_current_entitlements(Entitlements.none())
        return None
    try:
        entitlements = license_service.current_entitlements()
    except Exception as error:  # noqa: BLE001 - verification and storage raise widely
        logger.warning("Could not read the licence: %s", type(error).__name__)
        return None
    # ``current_entitlements`` publishes too; saying it here keeps the contract
    # readable and lets a test double be a plain object with one method.
    set_current_entitlements(entitlements)
    return entitlements


def boske_http_transport(
    method: str,
    url: str,
    data: dict,
    headers: dict,
) -> tuple[int, dict]:
    """The production :data:`services.license_service.HttpTransport`.

    One JSON round trip through the hardened opener, so a 30x can never re-send
    the lease to another host. Returns ``(status, payload)`` with an empty
    payload for a body that is not a JSON object; the licence service treats
    that as a failed call, which is what it is.

    An HTTP error is a status, not an exception: ``poll_device_link`` reads
    ``authorization_pending`` out of a 400 body, and raising here would turn the
    normal case of the device flow into a transport failure.
    """
    assert method, "method is required"
    payload = json.dumps(data or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with open_no_cross_host_redirect(request, LICENSE_HTTP_TIMEOUT_S) as response:
            status = int(response.getcode() or 0)
            body = response.read()
    except urllib.error.HTTPError as error:
        status = int(error.code)
        body = error.read()
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return status, {}
    return status, parsed if isinstance(parsed, dict) else {}

# -- launch at login ---------------------------------------------------------


class LaunchAtLoginUnavailable(RuntimeError):
    """This build cannot register a login item.

    ``ServiceManagement`` is a macOS 13+ framework reached through PyObjC, and
    ``SMAppService`` only works for a real signed bundle. A source run has
    neither, so the setting is offered as unavailable rather than as a switch
    that silently does nothing.
    """


#: ``SMAppServiceStatusEnabled``. Named here so the decision below reads as
#: English and needs no framework import to test.
SM_STATUS_ENABLED = 1


def launch_at_login_enabled(service: Any) -> bool:
    """Whether ``service`` says the login item is registered right now."""
    if service is None:
        return False
    return int(service.status()) == SM_STATUS_ENABLED


def _sm_call(service: Any, name: str) -> None:
    """Call ``register``/``unregister`` across the shapes PyObjC exposes.

    PyObjC bridges ``-registerAndReturnError:`` to ``registerAndReturnError_``,
    returning ``(ok, error)``; some bridge versions also expose the plain
    Swift-style name. Both are accepted, and a ``False`` return is an error to
    raise rather than a value to ignore.
    """
    bridged = getattr(service, f"{name}AndReturnError_", None)
    if bridged is not None:
        ok, error = bridged(None)
        if not ok:
            # An operational refusal, not a missing framework: the user may have
            # switched the item off in System Settings, which they are allowed
            # to do. Told to them, not raised as "unavailable in this build".
            raise RuntimeError(f"could not {name} the login item: {error}")
        return
    plain = getattr(service, name, None)
    if plain is None:
        raise LaunchAtLoginUnavailable(f"SMAppService cannot {name}")
    plain()


def apply_launch_at_login(service: Any, enabled: bool) -> bool:
    """Register or unregister the login item; returns the state afterwards.

    The whole decision, over an injected ``SMAppService`` so it is testable
    without the framework. Asking for the state it is already in does nothing:
    ``register()`` on an already-registered service can put the approval
    prompt back in front of a user who never touched the switch.

    The state is *read back* rather than assumed. ``register`` can succeed and
    still leave the service at ``SMAppServiceStatusRequiresApproval``: the item
    is registered, macOS reports no error, and Murmur will not start at login
    until the user allows it in System Settings. Returning what was asked for
    would put a switch on screen claiming something that is not true yet.
    """
    assert isinstance(enabled, bool), f"expected a bool, got {enabled!r}"
    if service is None:
        raise LaunchAtLoginUnavailable("ServiceManagement is not available in this build")
    if enabled == launch_at_login_enabled(service):
        return enabled
    _sm_call(service, "register" if enabled else "unregister")
    return launch_at_login_enabled(service)


def login_item_service() -> Any | None:
    """``SMAppService.mainAppService()``, or ``None`` when there is none.

    ``ServiceManagement`` is imported here and not at module scope on purpose:
    it does not exist before macOS 13, and an import error at start would take
    the menu bar down with it over a checkbox.
    """
    try:
        from ServiceManagement import SMAppService
    except Exception as error:  # noqa: BLE001 - any import failure means "no"
        logger.info("Launch at login is unavailable: %s", error)
        return None
    main_app_service = getattr(SMAppService, "mainAppService", None)
    if main_app_service is None:
        logger.info("Launch at login is unavailable: SMAppService has no mainAppService")
        return None
    try:
        return main_app_service()
    except Exception as error:  # noqa: BLE001 - an unsigned source run raises here
        logger.info("Launch at login is unavailable: %s", error)
        return None

def language_is_auto(language: str | None) -> bool:
    """Whether the configured language leaves detection to the engine.

    The one place that decides what "auto" means, because two very different
    things depend on the answer and they must never disagree:

    * the cleanup prompt (:func:`prompt_language`) says "the same language as
      the dictation" instead of naming one;
    * the live decode may stand in for the batch pass. Voxtral's ``stream()``
      accepts a language and cannot honour it, so a user who pinned French must
      get the batch result — the pill still showed them the live words while
      they spoke, which is what the pill is for.
    """
    if not language:
        return True
    return str(language).strip().lower() in ("", "auto")


def prompt_language(language: str | None) -> str | None:
    """Language for the cleanup prompt: ``"auto"`` and ``""`` both mean None.

    ``render_system_prompt`` turns None into "the same language as the
    dictation", which is exactly what auto-detect means. Passing the literal
    string "auto" would ask the model to write in a language called "auto".
    """
    if language_is_auto(language):
        return None
    return str(language).strip()


def code_transform_language(language: str | None) -> str:
    """Trigger vocabulary for :func:`transform_spoken_code`; anything unknown is English.

    The transform only ships English and French trigger words and raises on any
    other code. A user dictating code in German must not lose their transcript
    to that, so an unsupported language falls back to the English table, which
    simply matches nothing in their speech.
    """
    stripped = (language or "").strip().lower()
    base = stripped.split("-")[0]
    return base if base in CODE_TRANSFORM_LANGUAGES else "en"

@dataclass(frozen=True)
class CleanupPlan:
    """Whether cleanup runs for this utterance, and under which mode and tone."""

    mode_id: str
    tone_id: str
    enabled: bool
    #: Why it is not running, for the log. Configuration, never an error, so it
    #: is deliberately not shown to the user — only a *failed* pass is.
    reason: str | None = None


def cleanup_plan(config: dict, context, *, pro=is_pro_feature_enabled) -> CleanupPlan:
    """Resolve mode and tone for this utterance and decide whether to clean it.

    Three gates, in the order the plan names them: the Pro entitlement, the
    user's on/off switch, and the mode itself — Dictation is verbatim by
    definition, so it is a skip of the LLM, not a skip of cleanup.

    The mode is resolved under the gate too (:func:`resolve_plan_mode`), so a
    free install lands on dictation and takes the passthrough exit even when
    ``cleanup`` itself is somehow open.
    """
    assert config is not None, "config is required"
    mode_id = resolve_plan_mode(config, context, pro=pro)
    tone_id = tone_from_config(config).id
    if not pro("cleanup"):
        return CleanupPlan(mode_id, tone_id, False, CLEANUP_OFF_PRO)
    if not resolve_cleanup_enabled(config):
        return CleanupPlan(mode_id, tone_id, False, CLEANUP_OFF_DISABLED)
    if MODES[mode_id].is_passthrough:
        return CleanupPlan(mode_id, tone_id, False, CLEANUP_OFF_PASSTHROUGH)
    return CleanupPlan(mode_id, tone_id, True)


@dataclass(frozen=True)
class CleanupOutcome:
    """What the cleanup pass produced. ``text`` is always safe to paste."""

    text: str
    ran: bool
    #: Set only when cleanup was attempted and did not deliver. This is what
    #: becomes the visible "cleanup skipped" notice; None means nothing to say.
    skipped_reason: str | None = None
    elapsed_s: float = 0.0


def cleanup_skipped_message(reason: str) -> str:
    """The visible notice for a cleanup that was attempted and did not deliver."""
    assert reason, "reason is required"
    return f"Cleanup skipped — {reason}. Your text was pasted unchanged."


def cleanup_notice_kind(reason: str | None) -> str | None:
    """What to do about a cleanup that did not run: nothing, a notice, or an offer.

    A missing model is the one failure with a fix the user can act on, so it
    earns the modal; everything else is a notification that says what happened
    and gets out of the way. None means the pass delivered and there is nothing
    to say.
    """
    if not reason:
        return None
    if reason == CLEANUP_MODEL_MISSING_REASON:
        return CLEANUP_NOTICE_OFFER
    return CLEANUP_NOTICE_NOTIFY


def paste_and_settle(text: str, *, type_text, pill=None, offer=None) -> bool:
    """Paste ``text``, tell the pill how it went, and only then run ``offer``.

    The order is the whole point. ``offer`` raises a modal alert, and a modal
    raised *before* the paste takes key focus — so the synthesised ⌘V lands in
    the alert instead of the user's document and the transcript is gone. The
    offer is therefore queued during the cleanup pass and released here, once
    :func:`type_text` has returned.

    Returns True when the text landed.
    """
    assert text is not None, "text is required"
    assert callable(type_text), "type_text must be callable"
    pasted = bool(type_text(text))
    if pill is not None:
        if pasted:
            pill.done(len(text))
        else:
            pill.error("Could not paste")
    if offer is not None:
        offer()
    return pasted


def should_offer_cleanup_download(
    *, declined: bool, downloading: bool, installed: bool
) -> bool:
    """Whether the automatic "download the cleanup model?" alert may be shown.

    Asked at most once a session: a modal on every utterance would be worse than
    no cleanup at all. But a decline is not permanent — that is what
    :data:`CLEANUP_DOWNLOAD_MENU_TITLE` is for. Setting the flag *before* the
    alert (so a "Not now" burned the one chance) is exactly what left the model
    unreachable for the rest of the session.
    """
    return not (declined or downloading or installed)


def cleanup_download_menu_enabled(*, installed: bool, downloading: bool) -> bool:
    """Whether the "Download cleanup model…" entry is clickable.

    Always present, because a user who said "Not now" needs a way back and the
    Settings popup deliberately does not list this model. Dead when there is
    nothing to do: already here, or already coming down.
    """
    return not (installed or downloading)


def should_prewarm_cleanup(
    config: dict, *, pro: bool, cleanup_enabled: bool, installed: bool, ram_gb: int | None
) -> bool:
    """Whether to start the cleanup server at launch instead of on first use.

    Pre-warming trades 2 GB of resident memory, from launch, for the multi-second
    wait the first cleaned utterance would otherwise pay. Worth it only when
    every one of these holds: the feature is licensed and switched on, the model
    is actually on disk, and the Mac has memory to spare. Anything else waits for
    the first utterance, where :data:`CLEANUP_FIRST_USE_WAIT_S` bounds the cost.
    """
    assert config is not None, "config is required"
    if not (pro and cleanup_enabled and installed):
        return False
    if not config.get(CLEANUP_PREWARM_KEY, CLEANUP_PREWARM_DEFAULT):
        return False
    return ram_gb is not None and ram_gb >= CLEANUP_PREWARM_MIN_RAM_GB


def run_cleanup(
    text: str,
    plan: CleanupPlan,
    *,
    cleanup,
    language: str | None = None,
    vocabulary_terms: tuple[str, ...] = (),
    transform_code=transform_spoken_code,
    render=render_system_prompt,
    pro=is_pro_feature_enabled,
) -> CleanupOutcome:
    """Run the LLM pass for one utterance. Never raises on a bad reply.

    Code mode runs the deterministic spoken-punctuation transform *first*: it is
    idempotent and rule-based, so doing it before the model means the model sees
    real code tokens (``--force``) rather than the words for them, and cannot
    "correct" them back into prose. The transform is its own Pro feature
    (``coding_mode``) and is asked for by name here, at the one place it runs.

    ``cleanup`` is the callable that actually talks to the server —
    ``CleanupRuntime.cleanup`` in the app, ``CloudCleanupClient.cleanup`` when
    the dictation went through Murmur Cloud, a fake in the tests. It returns a
    :class:`~cleanup.llama_server.CleanupResult`, whose ``skipped`` flag carries
    the original text: a skip costs the improvement, never the transcript.
    """
    assert text is not None, "text is required"
    assert plan is not None, "plan is required"
    assert callable(cleanup), "cleanup must be callable"
    assert callable(pro), "pro must be callable"
    if not plan.enabled:
        return CleanupOutcome(text=text, ran=False)

    if plan.mode_id == "code" and pro("coding_mode"):
        text = transform_code(text, language=code_transform_language(language))

    system_prompt = render(
        plan.mode_id, plan.tone_id, prompt_language(language), tuple(vocabulary_terms)
    )
    result = cleanup(text, system_prompt)
    if result.skipped:
        return CleanupOutcome(
            text=text,
            ran=False,
            skipped_reason=result.reason or "the cleanup pass did not answer",
            elapsed_s=result.elapsed_s,
        )
    return CleanupOutcome(text=result.text, ran=True, elapsed_s=result.elapsed_s)


def mode_menu_state(config: dict) -> dict[str, bool]:
    """Which "Mode" submenu entries carry a checkmark.

    Every mode id maps to whether it is the configured fallback, and
    :data:`MODE_MENU_AUTOMATIC` to whether the bundle-id table applies. Both can
    be ticked at once, and that is the truth: the table decides per app and the
    ticked mode is what applies everywhere the table says nothing.
    """
    assert config is not None, "config is required"
    active = config.get("cleanup_mode", DEFAULT_CONFIG["cleanup_mode"])
    state = {mode_id: mode_id == active for mode_id in MODE_IDS}
    state[MODE_MENU_AUTOMATIC] = bool(
        config.get("context_awareness", DEFAULT_CONFIG["context_awareness"])
    )
    return state


def tone_menu_state(config: dict) -> dict[str, bool]:
    """Which "Tone" submenu entry carries a checkmark. Exactly one, always."""
    assert config is not None, "config is required"
    active = config.get("cleanup_tone", DEFAULT_CONFIG["cleanup_tone"])
    if active not in TONE_IDS:
        active = DEFAULT_CONFIG["cleanup_tone"]
    return {tone_id: tone_id == active for tone_id in TONE_IDS}


def cleanup_model_missing_message(display_name: str) -> str:
    """Offered once per session when a mode needs the GGUF and it is not here."""
    assert display_name, "display_name is required"
    return (
        f"Cleanup needs {display_name}, which is not downloaded yet.\n\n"
        "Your text was pasted exactly as dictated. Download it now to let "
        "Murmur clean up what you say."
    )


def cleanup_download_status(state) -> str:
    """Menu status while the cleanup model downloads, from the sheet's own state."""
    assert state is not None, "state is required"
    return f"Cleanup model: {state.status_line()}"


def model_integrity_message(display_name: str) -> str:
    """What to say when a model's files no longer match their checksums."""
    assert display_name, "display_name is required"
    return (
        f"{display_name} failed verification: its files do not match the "
        "checksums on record. Delete and re-download the model from "
        "Settings → Speech engine."
    )


def verify_model_before_load(store, model_id: str, verified: set) -> None:
    """Re-hash ``model_id`` before an engine is pointed at it, once per process.

    ``ModelStore.is_installed`` compares file sizes only, so a truncated,
    swapped or tampered model passes it and the engine happily loads whatever
    is on disk. Verification reads every byte, which costs a few seconds per
    gigabyte, so ``verified`` remembers the ids already checked in this process;
    the caller drops an id again after a download or an engine switch.

    A mismatch is re-raised as a plain :class:`RuntimeError` so it reaches the
    user through the same alert as any other failed load.
    """
    assert store is not None, "store is required"
    assert model_id, "model_id is required"
    assert verified is not None, "verified is required"
    if model_id in verified:
        return
    try:
        store.verify(model_id)
    except ModelIntegrityError as error:
        raise RuntimeError(model_integrity_message(model_id)) from error
    verified.add(model_id)


def about_menu_title(version: str, build_info: dict) -> str:
    """The About line: version, plus a warning when the build is not signed.

    ``build_info`` is ``{}`` outside a bundle, so a source run says nothing
    about signing; only a real bundle that reports ``signed: false`` is
    labelled, matching what CI writes into ``build_info.json``.
    """
    assert version, "version is required"
    title = f"{APP_NAME} {version}"
    if build_info.get("signed") is False:
        return f"{title} · internal build"
    return title


def update_available_message(latest_version: str, current_version: str) -> str:
    """Alert body offering an update. Version metadata only."""
    assert latest_version, "latest_version is required"
    assert current_version, "current_version is required"
    return (
        f"Murmur {latest_version} is available (you have {current_version}).\n\n"
        "Murmur will download it, check its signature, replace this copy, and "
        "restart itself."
    )


def should_relaunch_after_install(result) -> bool:
    """Whether this process must start the new bundle before it quits.

    ``install_update`` puts the new app in place but does not run it, because
    only the running app knows how to shut itself down cleanly. So Murmur
    launches the new bundle itself and then quits, and the user never has to
    reopen anything. Skipped only when the installer already relaunched.
    """
    assert result is not None, "result is required"
    return not bool(getattr(result, "relaunched", False))


def update_installed_message(version: str) -> str:
    """Said as the app hands over to the version it just installed."""
    assert version, "version is required"
    return f"Murmur {version} is installed. Restarting now."


def update_relaunch_failed_message(version: str) -> str:
    """Said when the new bundle is in place but would not start."""
    assert version, "version is required"
    return (
        f"Murmur {version} is installed, but it could not be started. "
        "Quit Murmur and open it again."
    )


def download_progress_status(bytes_done: int, bytes_total: int | None) -> str:
    """Menu status while an update downloads. Percent when the size is known."""
    assert bytes_done >= 0, f"bytes_done cannot be negative: {bytes_done}"
    if not bytes_total or bytes_total <= 0:
        return f"Downloading update… {bytes_done // 1_000_000} MB"
    percent = min(100, int(bytes_done * 100 / bytes_total))
    return f"Downloading update… {percent}%"
