#!/usr/bin/env python3
"""State behind the Settings "Speech engine" section and its download sheet.

Everything in this module is plain Python: no AppKit import at module scope,
so the section and the sheet can be unit-tested headlessly and the AppKit code
in ``settings_window.py`` stays a thin rendering of these objects.

Three pieces live here:

* :class:`DownloadSheetState` — what the sheet shows, fed by
  :class:`~engines.model_store.DownloadProgress` ticks.
* :class:`DownloadController` — runs ``ModelStore.download`` off the main
  thread and hops every state change back through an injected dispatcher.
* :class:`EngineSectionModel` — the popup's choices, the selection, and the
  select/download/delete actions, including the config keys they write.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from engines import ENGINE_IDS
from engines.model_store import (
    DownloadCancelled,
    DownloadProgress,
    ModelSpec,
    ModelStoreError,
    human_size,
    models_for_engine,
)
from services.model_profile_service import (
    ENGINE_VOXTRAL_MLX,
    default_engine_for_current_machine,
    detect_chip,
    detect_ram_gb,
    voxtral_eligible,
)

PHASE_IDLE = "idle"
PHASE_DOWNLOADING = "downloading"
PHASE_VERIFYING = "verifying"
PHASE_DONE = "done"
PHASE_FAILED = "failed"
PHASE_CANCELLED = "cancelled"

#: Config keys this section owns. Both are ``None`` until the user chooses.
CONFIG_ENGINE_ID = "engine_id"
CONFIG_MODEL_ID = "model_id"


class DownloadSheetState:
    """Progress of one model download, in the terms the sheet displays.

    ``DownloadProgress`` ticks describe a single *file*; a model can be several
    files, so the state keeps the last tick per file name and sums them. The
    optional ``total_bytes`` seed is the catalog size of the whole model, which
    keeps the progress bar honest before the later files have been opened.
    """

    def __init__(self, model_id: str, total_bytes: int = 0) -> None:
        self.reset(model_id, total_bytes)

    def reset(self, model_id: str, total_bytes: int = 0) -> None:
        """Forget everything and start again on ``model_id``."""
        assert model_id, "model_id is required"
        assert total_bytes >= 0, f"total_bytes cannot be negative: {total_bytes}"
        self.model_id = model_id
        self.phase = PHASE_IDLE
        self.error: str | None = None
        self._expected_total = total_bytes
        self._files: dict[str, tuple[int, int]] = {}

    @property
    def bytes_done(self) -> int:
        """Bytes fetched across every file of the model."""
        return sum(done for done, _ in self._files.values())

    @property
    def bytes_total(self) -> int:
        """Best known size of the whole model."""
        return max(self._expected_total, sum(total for _, total in self._files.values()))

    @property
    def percent(self) -> float:
        """Progress from 0.0 to 100.0; 100.0 once the model is installed."""
        if self.phase == PHASE_DONE:
            return 100.0
        total = self.bytes_total
        if total <= 0:
            return 0.0
        return min(100.0, self.bytes_done * 100.0 / total)

    @property
    def is_active(self) -> bool:
        """True while bytes are moving or the checksums are being read."""
        return self.phase in (PHASE_DOWNLOADING, PHASE_VERIFYING)

    def update(self, progress: DownloadProgress) -> None:
        """Record one progress tick and move the phase to downloading."""
        self._files[progress.file_name] = (progress.bytes_done, progress.bytes_total)
        self.phase = PHASE_DOWNLOADING
        self.error = None

    def mark_verifying(self) -> None:
        """Bytes are in; checksums are being read."""
        self.phase = PHASE_VERIFYING
        self.error = None

    def mark_done(self) -> None:
        """The model is installed and verified."""
        self.phase = PHASE_DONE
        self.error = None

    def mark_failed(self, message: str) -> None:
        """The run stopped on an error. A blank message would hide the cause."""
        assert message, "a failure must carry a message"
        self.phase = PHASE_FAILED
        self.error = message

    def mark_cancelled(self) -> None:
        """The user stopped the download; the partial file is kept for a resume."""
        self.phase = PHASE_CANCELLED
        self.error = None

    def status_line(self) -> str:
        """One line of plain text for the sheet, matching the current phase."""
        if self.phase == PHASE_IDLE:
            return "Ready to download"
        if self.phase == PHASE_DOWNLOADING:
            return f"{human_size(self.bytes_done)} of {human_size(self.bytes_total)}"
        if self.phase == PHASE_VERIFYING:
            return "Verifying…"
        if self.phase == PHASE_DONE:
            return "Installed"
        if self.phase == PHASE_CANCELLED:
            return "Cancelled"
        return f"Failed: {self.error}"


def main_thread_dispatcher() -> Callable[[Callable[[], None]], None]:
    """Return the app's thread→UI hop, matching ``MurmurApp.run_on_main_thread``.

    AppKit is imported inside the closure so this module stays importable in a
    headless test run.
    """

    def dispatch(func: Callable[[], None]) -> None:
        if threading.current_thread() is threading.main_thread():
            func()
            return
        from PyObjCTools import AppHelper

        AppHelper.callAfter(func)

    return dispatch


def _spawn_thread(func: Callable[[], None]):
    """Run ``func`` on a daemon thread. Replaced in tests by a direct call."""
    thread = threading.Thread(target=func, daemon=True, name="murmur-model-download")
    thread.start()
    return thread


class DownloadController:
    """Runs a model download off the main thread and reports it as sheet state.

    ``dispatch`` receives a zero-argument callable and must run it on the UI
    thread; ``on_change`` is called there with the state after every change, so
    a view can redraw without knowing anything about threads. ``spawn`` starts
    the worker and exists so tests can run it inline.
    """

    def __init__(
        self,
        store,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        on_change: Callable[[DownloadSheetState], None] | None = None,
        spawn: Callable[[Callable[[], None]], object] | None = None,
    ) -> None:
        assert store is not None, "store is required"
        self._store = store
        self._dispatch = dispatch if dispatch is not None else main_thread_dispatcher()
        self._on_change = on_change
        self._spawn = spawn if spawn is not None else _spawn_thread
        self.state = DownloadSheetState("(none)")
        self._cancel: threading.Event | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """True between :meth:`start` and the terminal state change."""
        return self._running

    def start(self, model_id: str, total_bytes: int = 0):
        """Begin downloading ``model_id``. Returns whatever ``spawn`` returns."""
        assert model_id, "model_id is required"
        assert not self._running, f"a download of {self.state.model_id} is already running"
        self.state.reset(model_id, total_bytes)
        self._cancel = threading.Event()
        self._running = True
        self._notify()
        cancel = self._cancel
        return self._spawn(lambda: self._run(model_id, cancel))

    def cancel(self) -> None:
        """Ask the running download to stop. Idempotent, safe when idle."""
        if self._cancel is not None:
            self._cancel.set()

    def _run(self, model_id: str, cancel: threading.Event) -> None:
        """Worker body. Every state change is posted through the dispatcher."""
        try:
            self._store.download(model_id, progress=self._on_progress, cancel=cancel)
        except DownloadCancelled:
            self._finish(self.state.mark_cancelled)
            return
        except (ModelStoreError, OSError) as exc:
            self._finish(lambda: self.state.mark_failed(str(exc)))
            return

        self._post(self.state.mark_verifying)
        try:
            self._store.verify(model_id)
        except (ModelStoreError, OSError) as exc:
            self._finish(lambda: self.state.mark_failed(str(exc)))
            return
        self._finish(self.state.mark_done)

    def _on_progress(self, progress: DownloadProgress) -> None:
        """Called on the worker thread by the store; hops to the UI thread."""
        self._post(lambda: self.state.update(progress))

    def _post(self, mutate: Callable[[], None]) -> None:
        def apply() -> None:
            mutate()
            self._notify()

        self._dispatch(apply)

    def _finish(self, mutate: Callable[[], None]) -> None:
        """Apply a terminal state change and release the running flag."""
        self._running = False
        self._post(mutate)

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change(self.state)


@dataclass(frozen=True)
class EngineChoice:
    """One row of the "Speech engine" popup."""

    engine_id: str
    model_id: str
    display_name: str
    size_bytes: int
    license: str
    installed: bool
    recommended: bool

    @property
    def title(self) -> str:
        """Popup title: what it is, how big, whether it is here."""
        state = "Installed" if self.installed else "Not downloaded"
        return f"{self.display_name} · {human_size(self.size_bytes)} · {state}"


class EngineSectionModel:
    """Choices, selection and actions for the Settings "Speech engine" section.

    The section owns two config keys, ``engine_id`` and ``model_id``. Both are
    absent (``None``) until the user picks a model that is actually installed;
    until then the popup merely highlights this machine's default, so a config
    file never claims an engine that cannot run.

    Selecting an installed model is a live change: ``on_engine_change`` fires so
    the app can reload the engine in the background, and only if it accepts are
    the two keys written. Nothing here asks for a restart.

    ``on_engine_change(engine_id, model_id)`` returns None when the swap starts
    (or was already in place) and a user-facing refusal message when it cannot —
    the app is recording, transcribing, or already swapping. ``save_changes``
    receives only the keys this section owns, never a whole config.
    """

    def __init__(
        self,
        config: dict,
        store,
        *,
        chip: str | None = None,
        ram_gb: int | None = None,
        default_engine: str | None = None,
        on_engine_change: Callable[[str, str], str | None] | None = None,
        save_changes: Callable[[dict], None] | None = None,
    ) -> None:
        assert config is not None, "config is required"
        assert store is not None, "store is required"
        self._config = config
        self._store = store
        #: Why the last selection did not take, or None. Read by the view.
        self.refusal: str | None = None
        self._chip = chip if chip is not None else detect_chip()
        self._ram_gb = ram_gb if ram_gb is not None else detect_ram_gb()
        self._default_engine = (
            default_engine
            if default_engine is not None
            else default_engine_for_current_machine()
        )
        self._specs = tuple(
            spec for spec in store.catalog if self._runs_here(spec) and spec.files
        )
        assert self._specs, f"no speech model can run on chip {self._chip!r}"
        self._on_engine_change = on_engine_change
        self._save_changes = save_changes
        self._selected = self._initial_selection()
        self._choices = self._build_choices()

    # -- what the popup shows -------------------------------------------

    @property
    def choices(self) -> tuple[EngineChoice, ...]:
        """Every model this machine can run, in catalog order."""
        return self._choices

    @property
    def selected_model_id(self) -> str:
        """The highlighted row's model id."""
        return self._selected

    @property
    def selected_index(self) -> int:
        """Index of the highlighted row, for ``selectItemAtIndex_``."""
        return [choice.model_id for choice in self._choices].index(self._selected)

    @property
    def selected_choice(self) -> EngineChoice:
        """The highlighted row."""
        return self._choices[self.selected_index]

    @property
    def active_engine_id(self) -> str | None:
        """The engine recorded in config, or None while nothing is chosen."""
        return self._config.get(CONFIG_ENGINE_ID)

    @property
    def active_model_id(self) -> str | None:
        """The model recorded in config, or None while nothing is chosen."""
        return self._config.get(CONFIG_MODEL_ID)

    @property
    def can_download(self) -> bool:
        """True when the highlighted model still has to be fetched."""
        return not self.selected_choice.installed

    @property
    def can_delete(self) -> bool:
        """True when the highlighted model has files to remove."""
        return self.selected_choice.installed

    @property
    def detail_line(self) -> str:
        """The muted line under the popup: size, licence, recommendation."""
        choice = self.selected_choice
        size = human_size(choice.size_bytes)
        parts = [
            f"{size} on this Mac" if choice.installed else f"{size} download",
            choice.license,
        ]
        if choice.recommended:
            parts.append("Recommended for this Mac")
        return " · ".join(parts)

    def spec(self, model_id: str | None = None) -> ModelSpec:
        """Catalog spec for ``model_id``, defaulting to the highlighted row."""
        target = model_id or self._selected
        for spec in self._specs:
            if spec.id == target:
                return spec
        raise AssertionError(f"{target!r} is not offered on this machine")

    # -- actions ---------------------------------------------------------

    def select(self, model_id: str) -> bool:
        """Highlight ``model_id``; activate it when it is installed.

        Returns True when the engine actually changed, so a caller knows
        whether a reload was requested.
        """
        assert any(spec.id == model_id for spec in self._specs), (
            f"{model_id!r} is not offered on this machine"
        )
        self._selected = model_id
        self.refresh()
        if not self.selected_choice.installed:
            return False
        return self._activate(model_id)

    def select_index(self, index: int) -> bool:
        """Highlight the row at ``index``, as the popup reports it."""
        assert 0 <= index < len(self._choices), f"row {index} is out of range"
        return self.select(self._choices[index].model_id)

    def on_download_finished(self, model_id: str) -> bool:
        """Re-read the install state after a download and put the model to work."""
        self.refresh()
        return self.select(model_id)

    def delete(self, model_id: str | None = None) -> str | None:
        """Remove a model's files. Returns a refusal message, or None on success.

        The model Murmur is currently using is never deleted from under the
        running engine, and neither is one that was never downloaded.
        """
        spec = self.spec(model_id)
        if spec.id == self.active_model_id:
            return (
                f"{spec.display_name} is the engine Murmur is using. "
                "Choose another model first, then delete this one."
            )
        if not self._store.is_installed(spec.id):
            return f"{spec.display_name} is not downloaded."
        self._store.delete(spec.id)
        self.refresh()
        return None

    def refresh(self) -> None:
        """Re-read install state from disk and rebuild the popup rows."""
        self._choices = self._build_choices()

    # -- internals -------------------------------------------------------

    def _runs_here(self, spec: ModelSpec) -> bool:
        """Which catalog entries this section may offer.

        The store the app builds also carries the cleanup GGUF
        (``cleanup.llama_server.CLEANUP_MODEL_SPEC``), which is not a speech
        engine at all: offering it here would let a user "select" a chat model
        as their transcriber. So the section lists only specs whose engine is a
        registered speech engine, and Voxtral on top of that is opt-in on
        eligible Apple Silicon (decision D1).
        """
        if spec.engine not in ENGINE_IDS:
            return False
        if spec.engine == ENGINE_VOXTRAL_MLX:
            return voxtral_eligible(self._chip, self._ram_gb)
        return True

    def _recommended_id(self) -> str | None:
        """First model of this machine's default engine, when it is offered."""
        preferred = models_for_engine(self._default_engine, self._specs)
        return preferred[0].id if preferred else None

    def _initial_selection(self) -> str:
        """Config's model when this machine can run it, else the recommendation."""
        configured = self._config.get(CONFIG_MODEL_ID)
        if any(spec.id == configured for spec in self._specs):
            return configured
        return self._recommended_id() or self._specs[0].id

    def _build_choices(self) -> tuple[EngineChoice, ...]:
        recommended = self._recommended_id()
        return tuple(
            EngineChoice(
                engine_id=spec.engine,
                model_id=spec.id,
                display_name=spec.display_name,
                size_bytes=spec.size_bytes,
                license=spec.license,
                installed=self._store.is_installed(spec.id),
                recommended=spec.id == recommended,
            )
            for spec in self._specs
        )

    def _activate(self, model_id: str) -> bool:
        """Ask the app to swap engines, and record the choice only if it agreed.

        The order is the point. Writing the keys first left config naming an
        engine the app had just refused to load, so the app went on running the
        old model while the file — and the next launch — claimed the new one. A
        refusal instead puts the highlight back on the model actually running
        and is reported through :attr:`refusal`.
        """
        spec = self.spec(model_id)
        self.refusal = None
        if (
            self._config.get(CONFIG_ENGINE_ID) == spec.engine
            and self._config.get(CONFIG_MODEL_ID) == spec.id
        ):
            return False
        if self._on_engine_change is not None:
            refusal = self._on_engine_change(spec.engine, spec.id)
            if refusal:
                self.refusal = str(refusal)
                self._selected = self._initial_selection()
                self.refresh()
                return False
        self._config[CONFIG_ENGINE_ID] = spec.engine
        self._config[CONFIG_MODEL_ID] = spec.id
        if self._save_changes is not None:
            self._save_changes(
                {CONFIG_ENGINE_ID: spec.engine, CONFIG_MODEL_ID: spec.id}
            )
        return True


__all__ = [
    "CONFIG_ENGINE_ID",
    "CONFIG_MODEL_ID",
    "PHASE_CANCELLED",
    "PHASE_DONE",
    "PHASE_DOWNLOADING",
    "PHASE_FAILED",
    "PHASE_IDLE",
    "PHASE_VERIFYING",
    "DownloadController",
    "DownloadSheetState",
    "EngineChoice",
    "EngineSectionModel",
    "main_thread_dispatcher",
]
