#!/usr/bin/env python3
"""First-run onboarding wizard: state machine plus its AppKit window.

The module is deliberately AppKit-free at import time. :class:`OnboardingState`
and :func:`should_show` are plain Python and are what the tests drive; PyObjC is
imported lazily inside :class:`OnboardingWindow`, so this module loads on a
machine with no PyObjC at all.

Every side effect the wizard can cause — asking macOS for the microphone,
asking for Accessibility, downloading a model, recording a test sentence — is an
injected callable (see :class:`OnboardingCallbacks`). The wizard itself never
types into another app: the test sentence lands in a field inside the window.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Bumped when the wizard gains a step worth re-showing to existing users.
ONBOARDING_VERSION = 1

CONFIG_KEY_COMPLETED = "onboarding_completed"
CONFIG_KEY_VERSION = "onboarding_version"

STEP_WELCOME = "welcome"
STEP_MICROPHONE = "microphone"
STEP_ACCESSIBILITY = "accessibility"
STEP_ENGINE = "engine"
STEP_TEST = "test"
STEP_DONE = "done"

#: The wizard's steps, in order.
STEPS: tuple[str, ...] = (
    STEP_WELCOME,
    STEP_MICROPHONE,
    STEP_ACCESSIBILITY,
    STEP_ENGINE,
    STEP_TEST,
    STEP_DONE,
)

STATUS_PENDING = "pending"
STATUS_GRANTED = "granted"
STATUS_DENIED = "denied"
STATUS_DOWNLOADING = "downloading"
STATUS_READY = "ready"
STATUS_SKIPPED = "skipped"

STATUSES: tuple[str, ...] = (
    STATUS_PENDING,
    STATUS_GRANTED,
    STATUS_DENIED,
    STATUS_DOWNLOADING,
    STATUS_READY,
    STATUS_SKIPPED,
)

#: Steps that only tell the user something; passing them counts as read.
INFORMATIONAL_STEPS: tuple[str, ...] = (STEP_WELCOME, STEP_DONE)

#: Every user-facing string, in one place, so copy edits never touch logic.
STRINGS: dict[str, dict[str, str]] = {
    "window": {
        "title": "Set up Murmur",
        "step_indicator": "Step {number} of {total}",
        "back": "Back",
        "skip": "Skip",
        "continue": "Continue",
        "finish": "Finish",
        "open_settings": "Open Settings",
    },
    "status": {
        STATUS_PENDING: "Not set up yet",
        STATUS_GRANTED: "Granted",
        STATUS_DENIED: "Not granted",
        STATUS_DOWNLOADING: "Downloading…",
        STATUS_READY: "Ready",
        STATUS_SKIPPED: "Skipped",
    },
    STEP_WELCOME: {
        "title": "Welcome to Murmur",
        "body": (
            "Murmur turns speech into text anywhere on your Mac. Hold the "
            "shortcut, talk, and the words appear where your cursor is.\n\n"
            "Everything stays on this Mac. Audio is transcribed on-device and "
            "nothing is sent to the cloud."
        ),
        "action": "",
    },
    STEP_MICROPHONE: {
        "title": "Microphone",
        "body": (
            "Murmur needs the microphone to hear you. Recording only happens "
            "while you hold the shortcut, and the audio never leaves this Mac."
        ),
        "action": "Allow Microphone",
        "settings": "Open System Settings",
        "denied": (
            "macOS is refusing the microphone. Open System Settings → Privacy "
            "& Security → Microphone and switch Murmur on."
        ),
    },
    STEP_ACCESSIBILITY: {
        "title": "Accessibility",
        "body": (
            "Accessibility lets Murmur do two things: listen for the global "
            "shortcut from any app, and paste the transcription at your "
            "cursor. Without it you can still dictate and copy the text "
            "yourself."
        ),
        "action": "Allow Accessibility",
        "settings": "Open System Settings",
        "denied": (
            "Open System Settings → Privacy & Security → Accessibility, switch "
            "Murmur on, then come back here."
        ),
    },
    STEP_ENGINE: {
        "title": "Speech engine",
        "body": (
            "This Mac runs best with {engine}. The model is downloaded once "
            "and then works offline."
        ),
        "action": "Download",
        "cancel": "Cancel download",
        "skip_note": (
            "You can skip this, but dictation needs a model. Download it later "
            "from Settings → Speech engine."
        ),
    },
    STEP_TEST: {
        "title": "Try it",
        "body": (
            "Press Try it and say a short sentence. The result lands in the "
            "field below — nothing is typed into another app."
        ),
        "action": "Try it",
        "placeholder": "Your test sentence appears here",
        "field_label": "Test transcription",
        "running": "Listening…",
    },
    STEP_DONE: {
        "title": "You're set",
        "body": (
            "Here is where things stand. You can reopen this wizard any time "
            "from the Murmur menu."
        ),
        "action": "",
    },
}


def format_size(size_bytes: int) -> str:
    """Human-readable model size; ``0`` means the size is not on record."""
    if not size_bytes:
        return "unknown size"
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def should_show(config: dict[str, Any]) -> bool:
    """True when this Mac has never finished the wizard, or finished an older one."""
    if not config.get(CONFIG_KEY_COMPLETED, False):
        return True
    version = config.get(CONFIG_KEY_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        return True
    return version < ONBOARDING_VERSION


@dataclass(frozen=True)
class EngineChoice:
    """The engine and model the wizard offers on this machine."""

    engine_id: str
    model_id: str
    display_name: str
    size_bytes: int


# --------------------------------------------------------------------------
# Defaults. Each one imports its dependency lazily so importing this module
# costs nothing and works on a machine without PyObjC or a model catalog.
# --------------------------------------------------------------------------


def default_engine_choice() -> EngineChoice:
    """The engine this machine should default to, with its first catalog model."""
    from engines.model_store import CATALOG
    from services.model_profile_service import default_engine_for_current_machine

    engine_id = default_engine_for_current_machine()
    for spec in CATALOG:
        if spec.engine == engine_id:
            return EngineChoice(
                engine_id=engine_id,
                model_id=spec.id,
                display_name=spec.display_name,
                size_bytes=spec.size_bytes,
            )
    raise LookupError(f"no catalog model for engine {engine_id!r}")


def default_request_microphone() -> bool:
    """Ask macOS for microphone access and wait for the answer."""
    import AVFoundation

    media_type = AVFoundation.AVMediaTypeAudio
    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(media_type)
    if status == 3:  # AVAuthorizationStatusAuthorized
        return True
    if status != 0:  # anything but NotDetermined: only System Settings can change it
        return False

    answer: dict[str, bool] = {}
    settled = threading.Event()

    def completion(granted: bool) -> None:
        answer["granted"] = bool(granted)
        settled.set()

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        media_type, completion
    )
    settled.wait(timeout=60)
    return answer.get("granted", False)


def default_request_accessibility() -> bool:
    """Ask macOS for Accessibility, which the shortcut and paste-at-cursor need."""
    from services.hotkey_service import request_hotkey_permissions

    return bool(request_hotkey_permissions())


def default_open_microphone_settings() -> None:
    """Open System Settings on the Microphone privacy pane."""
    import subprocess

    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"],
        check=False,
    )


def default_open_accessibility_settings() -> None:
    """Open System Settings on the Accessibility privacy pane."""
    from services.hotkey_service import open_privacy_settings

    open_privacy_settings()


def default_download(model_id: str, progress: Any = None, cancel: Any = None) -> Any:
    """Download a catalog model through the shared :class:`ModelStore`."""
    from engines.model_store import ModelStore

    return ModelStore().download(model_id, progress, cancel)


def default_is_installed(model_id: str) -> bool:
    """True when the model is already on disk at its recorded size."""
    from engines.model_store import ModelStore

    return ModelStore().is_installed(model_id)


@dataclass
class OnboardingCallbacks:
    """What the app hands the wizard.

    ``request_microphone``      ``() -> bool``, asks macOS and returns the answer.
    ``request_accessibility``   ``() -> bool``, same for Accessibility.
    ``download``                ``(model_id, progress, cancel) -> Any``, matching
                                :meth:`engines.model_store.ModelStore.download`.
    ``record_and_transcribe``   ``() -> str``, records a short clip and returns the
                                text. Runs off the main thread; the wizard puts the
                                result in its own field, never into another app.
    ``open_settings``           ``() -> None``, opens Murmur's Settings window.
    ``on_finished``             ``(config_updates: dict) -> None``, called once when
                                the wizard closes, with :meth:`OnboardingState.to_config`.
    ``is_installed``            ``(model_id) -> bool``, optional.
    ``open_microphone_settings`` / ``open_accessibility_settings``
                                ``() -> None``, optional System Settings fallbacks.

    Anything left ``None`` falls back to the ``default_*`` function above, except
    ``record_and_transcribe`` and ``on_finished``, which only the app can supply.
    """

    request_microphone: Callable[[], bool] | None = None
    request_accessibility: Callable[[], bool] | None = None
    download: Callable[..., Any] | None = None
    record_and_transcribe: Callable[[], str] | None = None
    open_settings: Callable[[], None] | None = None
    on_finished: Callable[[dict[str, Any]], None] | None = None
    is_installed: Callable[[str], bool] | None = None
    open_microphone_settings: Callable[[], None] | None = None
    open_accessibility_settings: Callable[[], None] | None = None


class OnboardingState:
    """Where the wizard is, what has been granted, and what is left to do.

    Pure Python: no AppKit, no threads of its own. The window drives it and
    reads it back; the tests drive it directly.
    """

    def __init__(
        self,
        *,
        request_microphone: Callable[[], bool] | None = None,
        request_accessibility: Callable[[], bool] | None = None,
        download: Callable[..., Any] | None = None,
        record_and_transcribe: Callable[[], str] | None = None,
        is_installed: Callable[[str], bool] | None = None,
        engine_choice: EngineChoice | None = None,
    ) -> None:
        self._request_microphone = request_microphone
        self._request_accessibility = request_accessibility
        self._download = download
        self._record_and_transcribe = record_and_transcribe
        self._is_installed = is_installed
        self._engine_choice = engine_choice

        self._index = 0
        self._statuses: dict[str, str] = {step: STATUS_PENDING for step in STEPS}
        self._errors: dict[str, str] = {}

        self.test_text = ""
        self.download_bytes_done = 0
        self.download_bytes_total = 0
        self.cancel_event = threading.Event()
        #: Optional observer called on every download tick, on the download thread.
        self.on_download_progress: Callable[[Any], None] | None = None

        if self._model_already_installed():
            self._statuses[STEP_ENGINE] = STATUS_READY

    @classmethod
    def from_callbacks(
        cls,
        callbacks: OnboardingCallbacks,
        *,
        engine_choice: EngineChoice | None = None,
    ) -> "OnboardingState":
        """Build a state wired to ``callbacks``."""
        return cls(
            request_microphone=callbacks.request_microphone,
            request_accessibility=callbacks.request_accessibility,
            download=callbacks.download,
            record_and_transcribe=callbacks.record_and_transcribe,
            is_installed=callbacks.is_installed,
            engine_choice=engine_choice,
        )

    # -- position ---------------------------------------------------------

    @property
    def current_step(self) -> str:
        return STEPS[self._index]

    @property
    def step_number(self) -> int:
        """1-based position, for "Step 2 of 6"."""
        return self._index + 1

    @property
    def step_count(self) -> int:
        return len(STEPS)

    @property
    def can_go_back(self) -> bool:
        return self._index > 0

    @property
    def is_last_step(self) -> bool:
        return self.current_step == STEP_DONE

    def advance(self) -> str:
        """Leave the current step and return the new one; stops on the last step.

        A step still ``pending`` is resolved on the way out: informational steps
        count as read, anything else as skipped — moving past unfinished work is
        a skip, and the summary says so.
        """
        step = self.current_step
        if self._statuses[step] == STATUS_PENDING:
            self._statuses[step] = (
                STATUS_READY if step in INFORMATIONAL_STEPS else STATUS_SKIPPED
            )
        self._index = min(self._index + 1, len(STEPS) - 1)
        return self.current_step

    def skip(self) -> str:
        """Mark the current step skipped and move on."""
        self._statuses[self.current_step] = STATUS_SKIPPED
        self._index = min(self._index + 1, len(STEPS) - 1)
        return self.current_step

    def back(self) -> str:
        """Move back one step, leaving every status alone."""
        self._index = max(self._index - 1, 0)
        return self.current_step

    # -- status -----------------------------------------------------------

    def status(self, step: str) -> str:
        assert step in self._statuses, f"unknown step: {step!r}"
        return self._statuses[step]

    def error(self, step: str) -> str:
        """The last failure message for ``step``, or an empty string."""
        return self._errors.get(step, "")

    def _set_status(self, step: str, status: str) -> None:
        assert step in self._statuses, f"unknown step: {step!r}"
        assert status in STATUSES, f"unknown status: {status!r}"
        self._statuses[step] = status

    @property
    def can_finish(self) -> bool:
        """True once every step before "done" has been resolved one way or another.

        Denied permissions count as resolved: the wizard is skippable and a user
        who says no to Accessibility must still be able to finish.
        """
        return all(
            self._statuses[step] != STATUS_PENDING
            for step in STEPS
            if step != STEP_DONE
        )

    def summary(self) -> list[tuple[str, str]]:
        """``(step title, status label)`` for every step but "done"."""
        return [
            (STRINGS[step]["title"], STRINGS["status"][self._statuses[step]])
            for step in STEPS
            if step != STEP_DONE
        ]

    def to_config(self) -> dict[str, Any]:
        """Config updates to persist when the wizard closes."""
        return {CONFIG_KEY_COMPLETED: True, CONFIG_KEY_VERSION: ONBOARDING_VERSION}

    # -- actions ----------------------------------------------------------

    def request_microphone(self) -> bool:
        """Ask for the microphone and record the answer on the microphone step."""
        return self._probe(STEP_MICROPHONE, self._request_microphone, default_request_microphone)

    def request_accessibility(self) -> bool:
        """Ask for Accessibility and record the answer on the accessibility step."""
        return self._probe(
            STEP_ACCESSIBILITY, self._request_accessibility, default_request_accessibility
        )

    def _probe(
        self,
        step: str,
        injected: Callable[[], bool] | None,
        fallback: Callable[[], bool],
    ) -> bool:
        probe = injected or fallback
        try:
            granted = bool(probe())
        except Exception as error:  # a refused TCC probe must not kill the wizard
            self._errors[step] = str(error)
            self._set_status(step, STATUS_DENIED)
            return False
        self._errors.pop(step, None)
        self._set_status(step, STATUS_GRANTED if granted else STATUS_DENIED)
        return granted

    # -- engine step ------------------------------------------------------

    @property
    def engine_choice(self) -> EngineChoice:
        """The engine and model for this machine, resolved on first use."""
        if self._engine_choice is None:
            self._engine_choice = default_engine_choice()
        return self._engine_choice

    @property
    def engine_summary(self) -> str:
        """One line naming the engine's model and its download size."""
        choice = self.engine_choice
        return f"{choice.display_name} · {format_size(choice.size_bytes)}"

    def _model_already_installed(self) -> bool:
        """True when the engine's model is already on disk, so the step opens ready."""
        probe = self._is_installed or default_is_installed
        return bool(probe(self.engine_choice.model_id))

    @property
    def download_fraction(self) -> float:
        """0.0–1.0 for the file currently transferring, 0.0 when nothing is known."""
        if self.download_bytes_total <= 0:
            return 0.0
        return min(1.0, self.download_bytes_done / self.download_bytes_total)

    def cancel_download(self) -> None:
        """Ask the running download to stop at the next chunk."""
        self.cancel_event.set()

    def start_download(self) -> bool:
        """Download the engine model, blocking until it finishes.

        Callers run this off the main thread. Progress ticks go to
        ``on_download_progress`` on that same thread, exactly as
        :class:`engines.model_store.ModelStore` documents; the window hops to
        the main thread itself.
        """
        downloader = self._download or default_download
        model_id = self.engine_choice.model_id
        self.download_bytes_done = 0
        self.download_bytes_total = 0
        self._errors.pop(STEP_ENGINE, None)
        self._set_status(STEP_ENGINE, STATUS_DOWNLOADING)

        def tick(update: Any) -> None:
            self.download_bytes_done = getattr(update, "bytes_done", 0)
            self.download_bytes_total = getattr(update, "bytes_total", 0)
            observer = self.on_download_progress
            if observer is not None:
                observer(update)

        try:
            downloader(model_id, tick, self.cancel_event)
        except Exception as error:
            self._errors[STEP_ENGINE] = str(error)
            self._set_status(STEP_ENGINE, STATUS_PENDING)
            return False
        self._set_status(STEP_ENGINE, STATUS_READY)
        return True

    # -- test step --------------------------------------------------------

    def run_test(self) -> str:
        """Record a short clip and return the transcription for the wizard's field.

        Never raises: a failure is reported through :meth:`error` so the user can
        skip the step and still finish.
        """
        assert self._record_and_transcribe is not None, (
            "record_and_transcribe is required; the app supplies it at wiring time"
        )
        self._errors.pop(STEP_TEST, None)
        try:
            text = self._record_and_transcribe()
        except Exception as error:
            self._errors[STEP_TEST] = str(error)
            self._set_status(STEP_TEST, STATUS_PENDING)
            return ""
        self.test_text = (text or "").strip()
        self._set_status(STEP_TEST, STATUS_READY if self.test_text else STATUS_PENDING)
        return self.test_text


# --------------------------------------------------------------------------
# AppKit window. Everything below imports PyObjC lazily, on first use.
# --------------------------------------------------------------------------

_APPKIT: Any = None
_BRIDGE_CLASS: Any = None
_ACTIVE_WINDOW: Any = None

TAG_BACK = 1
TAG_SKIP = 2
TAG_CONTINUE = 3
TAG_ACTION = 4
TAG_SYSTEM_SETTINGS = 5
TAG_APP_SETTINGS = 6


def _appkit() -> Any:
    """Import PyObjC and the shared theme once; return them as a namespace."""
    global _APPKIT
    if _APPKIT is not None:
        return _APPKIT

    import types as _types

    import objc
    from Cocoa import (
        NSAccessibilityButtonRole,
        NSApp,
        NSBackingStoreBuffered,
        NSButton,
        NSColor,
        NSFont,
        NSMakeRect,
        NSObject,
        NSProgressIndicator,
        NSTextField,
        NSView,
        NSWindow,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskTitled,
    )
    from PyObjCTools import AppHelper

    import ui_theme

    _APPKIT = _types.SimpleNamespace(
        objc=objc,
        AppHelper=AppHelper,
        ui_theme=ui_theme,
        NSAccessibilityButtonRole=NSAccessibilityButtonRole,
        NSApp=NSApp,
        NSBackingStoreBuffered=NSBackingStoreBuffered,
        NSButton=NSButton,
        NSColor=NSColor,
        NSFont=NSFont,
        NSMakeRect=NSMakeRect,
        NSObject=NSObject,
        NSProgressIndicator=NSProgressIndicator,
        NSTextField=NSTextField,
        NSView=NSView,
        NSWindow=NSWindow,
        NSWindowStyleMaskClosable=NSWindowStyleMaskClosable,
        NSWindowStyleMaskTitled=NSWindowStyleMaskTitled,
    )
    return _APPKIT


def _bridge_class() -> Any:
    """The ObjC object that receives button actions and the window delegate call."""
    global _BRIDGE_CLASS
    if _BRIDGE_CLASS is not None:
        return _BRIDGE_CLASS

    ns = _appkit()
    objc = ns.objc

    class _OnboardingBridge(ns.NSObject):
        def initWithOwner_(self, owner):
            self = objc.super(_OnboardingBridge, self).init()
            if self is None:
                return None
            self.owner = owner
            return self

        def buttonClicked_(self, sender):
            self.owner.handle_action(int(sender.tag()))

        def windowWillClose_(self, notification):
            self.owner.handle_close()

    _BRIDGE_CLASS = _OnboardingBridge
    return _BRIDGE_CLASS


def _label(
    ns: Any,
    parent: Any,
    rect: Any,
    text: str,
    *,
    size: float = 13,
    weight: float = 0.0,
    color: Any = None,
    wrap: bool = False,
    a11y: str | None = None,
) -> Any:
    """A non-editable NSTextField used as a label, themed like the rest of the app."""
    field = ns.NSTextField.alloc().initWithFrame_(rect)
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    field.setFont_(ns.NSFont.systemFontOfSize_weight_(size, weight))
    field.setTextColor_(color if color is not None else ns.ui_theme.primary_text_color())
    if wrap:
        field.setUsesSingleLineMode_(False)
        field.cell().setWraps_(True)
        field.cell().setScrollable_(False)
    if a11y:
        field.setAccessibilityLabel_(a11y)
    parent.addSubview_(field)
    return field


def _button(
    ns: Any,
    parent: Any,
    rect: Any,
    title: str,
    tag: int,
    bridge: Any,
    *,
    primary: bool = False,
    a11y: str | None = None,
    enabled: bool = True,
) -> Any:
    """A themed NSButton wired to the bridge's ``buttonClicked:`` selector."""
    button = ns.NSButton.alloc().initWithFrame_(rect)
    button.setTitle_(title)
    if primary:
        ns.ui_theme.style_primary_button(button)
    else:
        ns.ui_theme.style_dark_button(button)
    button.setTag_(tag)
    button.setEnabled_(enabled)
    button.setAccessibilityLabel_(a11y or title)
    button.setAccessibilityRole_(ns.NSAccessibilityButtonRole)
    button.setTarget_(bridge)
    button.setAction_(ns.objc.selector(bridge.buttonClicked_, signature=b"v@:@"))
    parent.addSubview_(button)
    return button


class OnboardingWindow:
    """The wizard window: fixed size, centred, one step at a time.

    Built in code like :mod:`history_window`, themed through :mod:`ui_theme`,
    every control carrying a VoiceOver label. Long work (download, test
    recording) runs on a background thread and hops back to the main thread the
    way ``murmur.MurmurApp.run_on_main_thread`` does.
    """

    WIDTH = 620
    HEIGHT = 470
    MARGIN = 24
    FOOTER = 56
    HEADER = 88

    def __init__(self, callbacks: OnboardingCallbacks, state: OnboardingState | None = None):
        self.callbacks = callbacks
        self.state = state if state is not None else OnboardingState.from_callbacks(callbacks)
        self._ns = _appkit()
        self._bridge = _bridge_class().alloc().initWithOwner_(self)
        self._finished = False
        self._download_active = False
        self._test_running = False
        self._progress_bar = None
        self._progress_label = None
        self._test_field = None
        self._build_window()
        self.render()

    # -- construction -----------------------------------------------------

    def _build_window(self) -> None:
        ns = self._ns
        rect = ns.NSMakeRect(0, 0, self.WIDTH, self.HEIGHT)
        style = ns.NSWindowStyleMaskTitled | ns.NSWindowStyleMaskClosable
        self.window = ns.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, ns.NSBackingStoreBuffered, False
        )
        self.window.setTitle_(STRINGS["window"]["title"])
        self.window.center()
        self.window.setReleasedWhenClosed_(False)
        self.window.setDelegate_(self._bridge)
        ns.ui_theme.apply_window_theme(self.window)

        content = self.window.contentView()
        inner_width = self.WIDTH - self.MARGIN * 2

        self._indicator_label = _label(
            ns,
            content,
            ns.NSMakeRect(self.MARGIN, self.HEIGHT - 38, inner_width, 16),
            "",
            size=11,
            color=ns.ui_theme.muted_text_color(),
            a11y="Progress through setup",
        )
        self._title_label = _label(
            ns,
            content,
            ns.NSMakeRect(self.MARGIN, self.HEIGHT - 72, inner_width, 28),
            "",
            size=20,
            weight=0.3,
        )
        ns.ui_theme.add_horizontal_rule(
            content, self.MARGIN, self.HEIGHT - self.HEADER, inner_width
        )

        body_height = self.HEIGHT - self.HEADER - self.FOOTER
        self._step_container = ns.NSView.alloc().initWithFrame_(
            ns.NSMakeRect(0, self.FOOTER, self.WIDTH, body_height)
        )
        content.addSubview_(self._step_container)

        footer = ns.ui_theme.add_footer_bar(content, self.WIDTH, self.FOOTER)
        self._back_button = _button(
            ns,
            footer,
            ns.NSMakeRect(self.MARGIN, 14, 84, 28),
            STRINGS["window"]["back"],
            TAG_BACK,
            self._bridge,
            a11y="Go back one step",
        )
        self._continue_button = _button(
            ns,
            footer,
            ns.NSMakeRect(self.WIDTH - self.MARGIN - 116, 14, 116, 28),
            STRINGS["window"]["continue"],
            TAG_CONTINUE,
            self._bridge,
            primary=True,
            a11y="Continue to the next step",
        )
        self._skip_button = _button(
            ns,
            footer,
            ns.NSMakeRect(self.WIDTH - self.MARGIN - 116 - 92, 14, 84, 28),
            STRINGS["window"]["skip"],
            TAG_SKIP,
            self._bridge,
            a11y="Skip this step",
        )

    def show(self) -> None:
        """Bring the wizard to the front."""
        self.window.makeKeyAndOrderFront_(None)
        self._ns.NSApp.activateIgnoringOtherApps_(True)

    # -- rendering --------------------------------------------------------

    def render(self) -> None:
        """Rebuild the step area and the footer for the current step."""
        ns = self._ns
        step = self.state.current_step
        copy = STRINGS[step]

        self._indicator_label.setStringValue_(
            STRINGS["window"]["step_indicator"].format(
                number=self.state.step_number, total=self.state.step_count
            )
        )
        self._title_label.setStringValue_(copy["title"])
        self._title_label.setAccessibilityLabel_(copy["title"])

        for view in list(self._step_container.subviews()):
            view.removeFromSuperview()
        self._progress_bar = None
        self._progress_label = None
        self._test_field = None

        builder = {
            STEP_WELCOME: self._build_plain_step,
            STEP_MICROPHONE: self._build_permission_step,
            STEP_ACCESSIBILITY: self._build_permission_step,
            STEP_ENGINE: self._build_engine_step,
            STEP_TEST: self._build_test_step,
            STEP_DONE: self._build_done_step,
        }[step]
        builder(step)

        self._back_button.setEnabled_(self.state.can_go_back)
        last = self.state.is_last_step
        self._skip_button.setHidden_(last or step == STEP_WELCOME)
        title = STRINGS["window"]["finish"] if last else STRINGS["window"]["continue"]
        self._continue_button.setTitle_(title)
        ns.ui_theme.style_primary_button(self._continue_button)
        self._continue_button.setAccessibilityLabel_(
            "Finish setup" if last else "Continue to the next step"
        )

    def _body(self, step: str, text: str | None = None) -> float:
        """Draw the step's body copy; return the y of the next free row."""
        ns = self._ns
        height = self._step_container.frame().size.height
        body = text if text is not None else STRINGS[step]["body"]
        _label(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, height - 112, self.WIDTH - self.MARGIN * 2, 104),
            body,
            size=13,
            color=ns.ui_theme.muted_text_color(),
            wrap=True,
            a11y=body,
        )
        return height - 150

    def _status_line(self, step: str, y: float) -> float:
        """Draw the step's status and any error under it; return the next free y."""
        ns = self._ns
        status = self.state.status(step)
        granted = status in (STATUS_GRANTED, STATUS_READY)
        _label(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, y, self.WIDTH - self.MARGIN * 2, 18),
            STRINGS["status"][status],
            size=12,
            weight=0.2,
            color=ns.ui_theme.brand_accent_color() if granted else ns.ui_theme.muted_text_color(),
            a11y=f"{STRINGS[step]['title']} status: {STRINGS['status'][status]}",
        )
        return y - 44

    def _note(self, text: str) -> None:
        """A quiet note pinned to the bottom of the step area."""
        ns = self._ns
        _label(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, 16, self.WIDTH - self.MARGIN * 2, 52),
            text,
            size=12,
            color=ns.ui_theme.subtle_text_color(),
            wrap=True,
            a11y=text,
        )

    def _build_plain_step(self, step: str) -> None:
        self._body(step)

    def _build_permission_step(self, step: str) -> None:
        ns = self._ns
        copy = STRINGS[step]
        y = self._body(step)
        y = self._status_line(step, y)
        granted = self.state.status(step) == STATUS_GRANTED
        _button(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, y, 190, 30),
            copy["action"],
            TAG_ACTION,
            self._bridge,
            primary=True,
            enabled=not granted,
            a11y=copy["action"],
        )
        _button(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN + 202, y, 190, 30),
            copy["settings"],
            TAG_SYSTEM_SETTINGS,
            self._bridge,
            a11y=f"{copy['settings']} for {copy['title'].lower()}",
        )
        if self.state.status(step) == STATUS_DENIED:
            hint = self.state.error(step) or copy["denied"]
            self._note(hint)

    def _build_engine_step(self, step: str) -> None:
        ns = self._ns
        copy = STRINGS[step]
        y = self._body(step, copy["body"].format(engine=self.state.engine_summary))
        y = self._status_line(step, y)

        downloading = self._download_active or self.state.status(step) == STATUS_DOWNLOADING
        installed = self.state.status(step) == STATUS_READY
        title = copy["cancel"] if downloading else copy["action"]
        _button(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, y, 190, 30),
            title,
            TAG_ACTION,
            self._bridge,
            primary=not downloading,
            enabled=not installed,
            a11y=f"{title} {self.state.engine_choice.display_name}",
        )

        bar_y = y - 40
        self._progress_bar = ns.NSProgressIndicator.alloc().initWithFrame_(
            ns.NSMakeRect(self.MARGIN, bar_y, self.WIDTH - self.MARGIN * 2, 14)
        )
        self._progress_bar.setStyle_(0)  # NSProgressIndicatorBarStyle
        self._progress_bar.setIndeterminate_(False)
        self._progress_bar.setMinValue_(0.0)
        self._progress_bar.setMaxValue_(1.0)
        self._progress_bar.setDoubleValue_(self.state.download_fraction)
        self._progress_bar.setHidden_(not downloading)
        self._progress_bar.setAccessibilityLabel_("Model download progress")
        self._step_container.addSubview_(self._progress_bar)

        self._progress_label = _label(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, bar_y - 24, self.WIDTH - self.MARGIN * 2, 18),
            self._progress_text(),
            size=12,
            color=ns.ui_theme.muted_text_color(),
        )
        self._note(self.state.error(step) or copy["skip_note"])

    def _build_test_step(self, step: str) -> None:
        ns = self._ns
        copy = STRINGS[step]
        y = self._body(step)
        y = self._status_line(step, y)
        _button(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, y, 190, 30),
            copy["running"] if self._test_running else copy["action"],
            TAG_ACTION,
            self._bridge,
            primary=True,
            enabled=not self._test_running,
            a11y="Record a test sentence",
        )

        field_y = y - 62
        _label(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, field_y + 28, self.WIDTH - self.MARGIN * 2, 16),
            copy["field_label"],
            size=11,
            color=ns.ui_theme.muted_text_color(),
        )
        self._test_field = ns.NSTextField.alloc().initWithFrame_(
            ns.NSMakeRect(self.MARGIN, field_y, self.WIDTH - self.MARGIN * 2, 26)
        )
        self._test_field.setStringValue_(self.state.test_text)
        self._test_field.setPlaceholderString_(copy["placeholder"])
        self._test_field.setEditable_(True)
        self._test_field.setSelectable_(True)
        self._test_field.setFont_(ns.NSFont.systemFontOfSize_(13))
        self._test_field.setAppearance_(ns.ui_theme.control_appearance())
        self._test_field.setAccessibilityLabel_(copy["field_label"])
        self._step_container.addSubview_(self._test_field)
        if self.state.error(step):
            self._note(self.state.error(step))

    def _build_done_step(self, step: str) -> None:
        ns = self._ns
        y = self._body(step)
        for title, label in self.state.summary():
            _label(
                ns,
                self._step_container,
                ns.NSMakeRect(self.MARGIN, y, 260, 18),
                title,
                size=12,
                a11y=f"{title}: {label}",
            )
            _label(
                ns,
                self._step_container,
                ns.NSMakeRect(self.MARGIN + 270, y, 200, 18),
                label,
                size=12,
                color=ns.ui_theme.muted_text_color(),
            )
            y -= 24
        _button(
            ns,
            self._step_container,
            ns.NSMakeRect(self.MARGIN, max(y - 20, 16), 190, 30),
            STRINGS["window"]["open_settings"],
            TAG_APP_SETTINGS,
            self._bridge,
            a11y="Open Murmur Settings",
        )

    # -- actions ----------------------------------------------------------

    def handle_action(self, tag: int) -> None:
        """Every button in the window lands here, tagged."""
        if tag == TAG_BACK:
            self.state.back()
            self.render()
        elif tag == TAG_SKIP:
            self.state.skip()
            self.render()
        elif tag == TAG_CONTINUE:
            if self.state.is_last_step:
                self.finish()
            else:
                self.state.advance()
                self.render()
        elif tag == TAG_ACTION:
            self._step_action()
        elif tag == TAG_SYSTEM_SETTINGS:
            self._open_system_settings()
        elif tag == TAG_APP_SETTINGS:
            if self.callbacks.open_settings is not None:
                self.callbacks.open_settings()

    def _step_action(self) -> None:
        step = self.state.current_step
        if step == STEP_MICROPHONE:
            self._run_off_main(self.state.request_microphone)
        elif step == STEP_ACCESSIBILITY:
            self._run_off_main(self.state.request_accessibility)
        elif step == STEP_ENGINE:
            self._toggle_download()
        elif step == STEP_TEST:
            self._start_test()

    def _open_system_settings(self) -> None:
        if self.state.current_step == STEP_MICROPHONE:
            opener = self.callbacks.open_microphone_settings or default_open_microphone_settings
        else:
            opener = (
                self.callbacks.open_accessibility_settings or default_open_accessibility_settings
            )
        opener()

    # -- background work --------------------------------------------------

    def _on_main(self, func: Callable[[], None]) -> None:
        """Run ``func`` on the main thread, the way murmur.py does."""
        if threading.current_thread() is threading.main_thread():
            func()
        else:
            self._ns.AppHelper.callAfter(func)

    def _run_off_main(self, work: Callable[[], Any]) -> None:
        """Run ``work`` on a daemon thread, then re-render on the main thread."""

        def worker() -> None:
            work()
            self._on_main(self.render)

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_download(self) -> None:
        if self._download_active:
            self.state.cancel_download()
            return
        self.state.cancel_event = threading.Event()
        self.state.on_download_progress = self._download_tick
        self._download_active = True
        self.render()
        threading.Thread(target=self._download_worker, daemon=True).start()

    def _download_worker(self) -> None:
        self.state.start_download()
        self._on_main(self._download_finished)

    def _download_finished(self) -> None:
        self._download_active = False
        self.render()

    def _download_tick(self, update: Any) -> None:
        """Called on the download thread; the UI update hops to the main thread."""
        self._on_main(self._update_progress)

    def _update_progress(self) -> None:
        if self._progress_bar is not None:
            self._progress_bar.setDoubleValue_(self.state.download_fraction)
        if self._progress_label is not None:
            self._progress_label.setStringValue_(self._progress_text())

    def _progress_text(self) -> str:
        if self.state.status(STEP_ENGINE) == STATUS_READY:
            return "Model installed."
        total = self.state.download_bytes_total
        if not total:
            return ""
        return f"{format_size(self.state.download_bytes_done)} of {format_size(total)}"

    def _start_test(self) -> None:
        if self._test_running:
            return
        self._test_running = True
        self.render()

        def worker() -> None:
            self.state.run_test()
            self._on_main(self._test_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _test_finished(self) -> None:
        self._test_running = False
        self.render()

    # -- closing ----------------------------------------------------------

    def finish(self) -> None:
        """Persist the outcome and close the window."""
        self._close(close_window=True)

    def handle_close(self) -> None:
        """The window delegate's ``windowWillClose:``; dismissing counts as done."""
        self._close(close_window=False)

    def _close(self, *, close_window: bool) -> None:
        global _ACTIVE_WINDOW
        if self._finished:
            return
        self._finished = True
        self.state.cancel_download()
        if self.callbacks.on_finished is not None:
            self.callbacks.on_finished(self.state.to_config())
        if _ACTIVE_WINDOW is self:
            _ACTIVE_WINDOW = None
        if close_window:
            self.window.setDelegate_(None)
            self.window.close()


def show_onboarding(app_callbacks: OnboardingCallbacks) -> OnboardingWindow:
    """Show the wizard and return its window.

    ``app_callbacks`` must at least carry ``record_and_transcribe`` and
    ``on_finished``; every other field falls back to the ``default_*`` helpers.
    Calling this again while a wizard is open just brings that one forward, so
    the menu item is safe to click twice.
    """
    global _ACTIVE_WINDOW
    assert app_callbacks is not None, "app_callbacks is required"
    assert app_callbacks.on_finished is not None, (
        "on_finished is required so the app can persist onboarding_completed"
    )
    assert app_callbacks.record_and_transcribe is not None, (
        "record_and_transcribe is required for the test step"
    )

    if _ACTIVE_WINDOW is not None and not _ACTIVE_WINDOW._finished:
        _ACTIVE_WINDOW.show()
        return _ACTIVE_WINDOW

    window = OnboardingWindow(app_callbacks)
    _ACTIVE_WINDOW = window
    window.show()
    return window
