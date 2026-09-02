#!/usr/bin/env python3
"""Settings → Engine: the local model list, the cloud choice, and usage.

Three groups, one model each:

* **Local engine** — every model this Mac can run, one row apiece, with a
  Download button on the ones that are missing and a Delete button on the
  installed ones that are not in use. Wave 1 had a popup, so Delete could only
  ever act on the highlighted model; a row list fixes that.
* **Cloud** — ``off | murmur_cloud | own_key``. Murmur Cloud is only
  selectable when the licence grants ``cloud_voice``; Own key picks a provider
  and points at the Account tab for the key itself, which never touches JSON.
* **Usage this month** — minutes and words, cloud and local, with a progress
  bar when the plan has an allowance and a plain line when it does not.

:class:`EngineTabModel` is all of that as plain Python. :class:`EngineTab` is a
rendering of it, so every rule above is tested without a window server.

Config keys owned here (see ``services/persistence_service.py``):

- ``cloud_mode``: ``off | murmur_cloud | own_key``, ``off`` until chosen.
- ``byok_provider``: ``mistral | openai``, only meaningful under ``own_key``.

``engine_id`` and ``model_id`` are owned by :class:`~ui.download_sheet.EngineSectionModel`
and written live, the moment an installed model is picked — switching models is
a hot swap, not something that waits for the window to close.

No engine id is spelled out in the logic here: the cloud modes reach engine ids
only through :data:`CLOUD_MODE_TO_ENGINE`, and the local rows carry whatever the
catalog says.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from engines.model_store import ModelSpec, ModelStore, human_size
from services.model_profile_service import detect_chip, detect_ram_gb
from ui.download_sheet import (
    PHASE_CANCELLED,
    PHASE_DONE,
    PHASE_FAILED,
    DownloadController,
    EngineSectionModel,
    main_thread_dispatcher,
)
from ui.settings import register_tab
from ui.settings.base import (
    ROW_SPACING,
    TAB_ENGINE,
    TabContext,
    bind_action,
    make_button,
    make_hint,
    make_label,
    make_popup,
    make_section_title,
    stack_horizontal,
    stack_vertical,
)

logger = logging.getLogger(__name__)

# -- Cloud --------------------------------------------------------------------

CONFIG_CLOUD_MODE = "cloud_mode"
CONFIG_BYOK_PROVIDER = "byok_provider"

CLOUD_MODE_OFF = "off"
CLOUD_MODE_MURMUR = "murmur_cloud"
CLOUD_MODE_OWN_KEY = "own_key"

CLOUD_MODES: tuple[str, ...] = (CLOUD_MODE_OFF, CLOUD_MODE_MURMUR, CLOUD_MODE_OWN_KEY)
DEFAULT_CLOUD_MODE = CLOUD_MODE_OFF

#: The only place a cloud engine id is written down. Wave 4 registers
#: ``engines/cloud.py`` and ``engines/byok.py`` under exactly these ids; the UI
#: never branches on them, it looks them up.
CLOUD_MODE_TO_ENGINE: dict[str, str | None] = {
    CLOUD_MODE_OFF: None,
    CLOUD_MODE_MURMUR: "cloud",
    CLOUD_MODE_OWN_KEY: "byok",
}
assert set(CLOUD_MODE_TO_ENGINE) == set(CLOUD_MODES), "every cloud mode needs an engine entry"

CLOUD_MODE_LABELS = {
    CLOUD_MODE_OFF: "Off — everything stays on this Mac",
    CLOUD_MODE_MURMUR: "Murmur Cloud",
    CLOUD_MODE_OWN_KEY: "Own key",
}

CLOUD_MODE_NOTES = {
    CLOUD_MODE_OFF: "No audio and no text leaves this Mac.",
    CLOUD_MODE_MURMUR: "Audio is sent to Murmur Cloud and metered against your plan.",
    CLOUD_MODE_OWN_KEY: "Audio is sent to your own provider account. Not metered by us.",
}

#: Providers a user can bring their own key for.
BYOK_PROVIDERS: tuple[str, ...] = ("mistral", "openai")
BYOK_PROVIDER_LABELS = {"mistral": "Mistral", "openai": "OpenAI"}
DEFAULT_BYOK_PROVIDER = BYOK_PROVIDERS[0]

#: Shown under the provider popup. The key itself lives in the Keychain, put
#: there by the Account tab (E3e) — never in the config file.
BYOK_NOTE = "The {provider} API key is entered on the Account tab and kept in the Keychain."

MURMUR_CLOUD_LOCKED_NOTE = "Sign in with a plan that includes cloud voice to use Murmur Cloud."

NOT_SIGNED_IN = "Not signed in"

USAGE_TITLE = "Usage this month"
USAGE_LABEL_CLOUD = "Murmur Cloud"
USAGE_LABEL_LOCAL = "On this Mac"


# -- Rows ---------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRow:
    """One local model, as the table draws it."""

    engine_id: str
    model_id: str
    display_name: str
    size_bytes: int
    license: str
    installed: bool
    active: bool
    recommended: bool
    selected: bool

    @property
    def size_text(self) -> str:
        return human_size(self.size_bytes)

    @property
    def state_text(self) -> str:
        """Installed, in use, or still to fetch."""
        if self.active:
            return "In use"
        return "Installed" if self.installed else "Not downloaded"

    @property
    def detail(self) -> str:
        """The muted line under the model's name."""
        parts = [self.size_text, self.state_text, self.license]
        if self.recommended:
            parts.append("Recommended for this Mac")
        return " · ".join(parts)

    @property
    def can_download(self) -> bool:
        return not self.installed

    @property
    def can_delete(self) -> bool:
        """Installed and not the one the running engine is holding open."""
        return self.installed and not self.active


@dataclass(frozen=True)
class CloudOption:
    """One radio button of the Cloud group."""

    mode: str
    title: str
    note: str
    enabled: bool
    selected: bool


@dataclass(frozen=True)
class UsageRow:
    """Minutes and words for one origin, over the current period."""

    label: str
    minutes: float
    words: int
    allowance_minutes: int | None

    @property
    def has_progress_bar(self) -> bool:
        """A bar only makes sense against an allowance."""
        return self.allowance_minutes is not None

    @property
    def percent(self) -> float | None:
        """0–100 of the allowance, or None when there is none to fill."""
        if self.allowance_minutes is None:
            return None
        if self.allowance_minutes <= 0:
            return 0.0
        return min(100.0, float(self.minutes) * 100.0 / float(self.allowance_minutes))

    @property
    def text(self) -> str:
        minutes = _format_minutes(self.minutes)
        if self.allowance_minutes is None:
            spent = f"{minutes} min"
        else:
            spent = f"{minutes} of {self.allowance_minutes} min"
        return f"{spent} · {self.words:,} words"


@dataclass(frozen=True)
class UsageBlock:
    """The whole "Usage this month" group."""

    title: str
    period_label: str
    rows: tuple[UsageRow, ...]


def _format_minutes(minutes: float) -> str:
    """``12.0`` → ``"12"``, ``48.5`` → ``"48.5"``. No trailing ``.0`` in the UI."""
    value = round(float(minutes), 1)
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:g}"


def _format_when(value: Any) -> str | None:
    """A date, a datetime or a string, as one short human date."""
    if value is None:
        return None
    strftime = getattr(value, "strftime", None)
    if strftime is not None:
        return strftime("%d %b %Y").lstrip("0")
    text = str(value).strip()
    return text or None


def format_license_line(status: Any | None) -> str:
    """One line describing the signed-in plan, or that nobody is signed in.

    ``status`` is whatever ``services["license"]()`` returns: an object with
    ``pro``, ``cloud_voice``, ``expires_at`` and ``in_grace``. Wave 4 fills it
    in; until then the provider is simply absent and this reads "Not signed in".
    """
    if status is None:
        return NOT_SIGNED_IN
    parts = ["Pro" if getattr(status, "pro", False) else "Free"]
    parts.append(
        "Cloud voice included"
        if getattr(status, "cloud_voice", False)
        else "Cloud voice not included"
    )
    when = _format_when(getattr(status, "expires_at", None))
    if when is not None:
        parts.append(
            f"Grace period until {when}" if getattr(status, "in_grace", False) else f"Renews {when}"
        )
    return " · ".join(parts)


class _CatalogStore:
    """What :class:`EngineSectionModel` needs, built from the tab's callables.

    The tab is handed a catalog and two functions rather than a
    :class:`~engines.model_store.ModelStore`, so a test describes a machine
    with a set of ids and no filesystem. The real tab passes the store's own
    bound methods, so nothing is faked in the app.
    """

    def __init__(
        self,
        catalog: Iterable[ModelSpec],
        installed: Callable[[str], bool],
        delete: Callable[[str], None] | None,
    ) -> None:
        assert catalog is not None, "catalog is required"
        assert callable(installed), "installed must be callable"
        self.catalog = tuple(catalog)
        self._installed = installed
        self._delete = delete

    def is_installed(self, model_id: str) -> bool:
        return bool(self._installed(model_id))

    def delete(self, model_id: str) -> None:
        assert self._delete is not None, (
            f"cannot delete {model_id!r}: EngineTabModel was built without a delete hook"
        )
        self._delete(model_id)


class EngineTabModel:
    """Editing state for Settings → Engine.

    The local half wraps :class:`~ui.download_sheet.EngineSectionModel`, which
    already owns ``engine_id``/``model_id`` and writes them the instant an
    installed model is picked. The cloud half is ordinary tab state: edited
    here, reported by :meth:`apply`, persisted by the window.
    """

    def __init__(
        self,
        config: dict,
        *,
        catalog: Iterable[ModelSpec],
        chip: str,
        ram_gb: int | None,
        installed: Callable[[str], bool],
        usage_provider: Callable[[], Any] | None = None,
        license_provider: Callable[[], Any] | None = None,
        app: Any | None = None,
        save: Callable[[dict], None] | None = None,
        delete: Callable[[str], None] | None = None,
        default_engine: str | None = None,
    ) -> None:
        assert config is not None, "config is required"
        assert chip, "chip is required"
        self._config = config
        self._app = app
        self._usage_provider = usage_provider
        self._license_provider = license_provider
        self._section = EngineSectionModel(
            config,
            _CatalogStore(catalog, installed, delete),
            chip=chip,
            ram_gb=ram_gb,
            default_engine=default_engine,
            on_engine_change=self.on_engine_change,
            save=save,
        )
        self._license_status = self._read_license()
        self._cloud_mode = _one_of(
            config.get(CONFIG_CLOUD_MODE, DEFAULT_CLOUD_MODE), CLOUD_MODES, DEFAULT_CLOUD_MODE
        )
        self._byok_provider = _one_of(
            config.get(CONFIG_BYOK_PROVIDER, DEFAULT_BYOK_PROVIDER),
            BYOK_PROVIDERS,
            DEFAULT_BYOK_PROVIDER,
        )
        self._original = self.as_config()

    # -- local models ----------------------------------------------------

    @property
    def section(self) -> EngineSectionModel:
        """The Wave 1 section this tab reuses, for the download sheet."""
        return self._section

    @property
    def rows(self) -> tuple[ModelRow, ...]:
        """One row per model this Mac can run, in catalog order."""
        active = self._section.active_model_id
        selected = self._section.selected_model_id
        return tuple(
            ModelRow(
                engine_id=choice.engine_id,
                model_id=choice.model_id,
                display_name=choice.display_name,
                size_bytes=choice.size_bytes,
                license=choice.license,
                installed=choice.installed,
                active=choice.model_id == active,
                recommended=choice.recommended,
                selected=choice.model_id == selected,
            )
            for choice in self._section.choices
        )

    @property
    def selected_model_id(self) -> str:
        return self._section.selected_model_id

    @property
    def active_model_id(self) -> str | None:
        return self._section.active_model_id

    def row(self, model_id: str) -> ModelRow:
        """The row for ``model_id``; an unknown id is a programming error."""
        for row in self.rows:
            if row.model_id == model_id:
                return row
        raise AssertionError(f"{model_id!r} is not offered on this machine")

    def select(self, model_id: str) -> bool:
        """Highlight a model, and activate it when it is installed."""
        return self._section.select(model_id)

    def delete(self, model_id: str) -> str | None:
        """Remove one model's files. Returns a refusal message, or None.

        Takes an explicit id, so Delete works on any installed row rather than
        only the highlighted one.
        """
        return self._section.delete(model_id)

    def on_download_finished(self, model_id: str) -> bool:
        """Re-read install state after a download and put the model to work."""
        return self._section.on_download_finished(model_id)

    def on_engine_change(self, engine_id: str, model_id: str) -> None:
        """Ask the running app to swap engines in the background. No restart."""
        reload_engine = getattr(self._app, "reload_engine", None) if self._app else None
        if reload_engine is None:
            logger.info(
                "Speech engine set to %s/%s; no running app to reload", engine_id, model_id
            )
            return
        reload_engine(engine_id, model_id)

    # -- cloud -----------------------------------------------------------

    @property
    def license_status(self) -> Any | None:
        """What the licence service last reported, or None when there is none."""
        return self._license_status

    @property
    def cloud_voice_entitled(self) -> bool:
        """Whether the plan includes cloud voice. The only gate on this tab."""
        return bool(getattr(self._license_status, "cloud_voice", False))

    @property
    def cloud_mode(self) -> str:
        return self._cloud_mode

    @property
    def cloud_engine_id(self) -> str | None:
        """The engine id behind the chosen mode, from the table. None when off."""
        return CLOUD_MODE_TO_ENGINE[self._cloud_mode]

    @property
    def cloud_options(self) -> tuple[CloudOption, ...]:
        """The three radio buttons, in order."""
        entitled = self.cloud_voice_entitled
        return tuple(
            CloudOption(
                mode=mode,
                title=CLOUD_MODE_LABELS[mode],
                note=(
                    MURMUR_CLOUD_LOCKED_NOTE
                    if mode == CLOUD_MODE_MURMUR and not entitled
                    else CLOUD_MODE_NOTES[mode]
                ),
                enabled=entitled if mode == CLOUD_MODE_MURMUR else True,
                selected=mode == self._cloud_mode,
            )
            for mode in CLOUD_MODES
        )

    @property
    def license_line(self) -> str | None:
        """The plan line, shown only under Murmur Cloud."""
        if self._cloud_mode != CLOUD_MODE_MURMUR:
            return None
        return format_license_line(self._license_status)

    @property
    def show_provider_popup(self) -> bool:
        return self._cloud_mode == CLOUD_MODE_OWN_KEY

    @property
    def byok_provider(self) -> str:
        return self._byok_provider

    @property
    def byok_provider_titles(self) -> tuple[str, ...]:
        return tuple(BYOK_PROVIDER_LABELS[provider] for provider in BYOK_PROVIDERS)

    @property
    def byok_provider_index(self) -> int:
        return BYOK_PROVIDERS.index(self._byok_provider)

    @property
    def byok_note(self) -> str | None:
        """Where the key goes, shown only under Own key."""
        if self._cloud_mode != CLOUD_MODE_OWN_KEY:
            return None
        return BYOK_NOTE.format(provider=BYOK_PROVIDER_LABELS[self._byok_provider])

    def set_cloud_mode(self, mode: str) -> bool:
        """Choose a cloud mode. Returns whether anything moved.

        Murmur Cloud without the ``cloud_voice`` entitlement is refused rather
        than stored: the radio is disabled, so reaching here at all means the
        entitlement went away under an open window.
        """
        assert mode in CLOUD_MODES, (
            f"Invalid cloud mode {mode!r}; expected one of {', '.join(CLOUD_MODES)}"
        )
        if mode == CLOUD_MODE_MURMUR and not self.cloud_voice_entitled:
            logger.info("Murmur Cloud refused: the current licence has no cloud voice")
            return False
        if mode == self._cloud_mode:
            return False
        self._cloud_mode = mode
        return True

    def set_byok_provider(self, provider: str) -> bool:
        """Choose the own-key provider. Returns whether anything moved."""
        assert provider in BYOK_PROVIDERS, (
            f"Unknown provider {provider!r}; expected one of {', '.join(BYOK_PROVIDERS)}"
        )
        if provider == self._byok_provider:
            return False
        self._byok_provider = provider
        return True

    def set_byok_provider_index(self, index: int) -> bool:
        """Choose the provider by popup row."""
        assert 0 <= index < len(BYOK_PROVIDERS), f"row {index} is out of range"
        return self.set_byok_provider(BYOK_PROVIDERS[index])

    # -- usage -----------------------------------------------------------

    @property
    def usage(self) -> UsageBlock | None:
        """This period's minutes and words, or None when nothing reports them."""
        if self._usage_provider is None:
            return None
        summary = self._usage_provider()
        if summary is None:
            return None
        allowance = getattr(summary, "allowance_minutes", None)
        return UsageBlock(
            title=USAGE_TITLE,
            period_label=str(getattr(summary, "period_label", "")),
            rows=(
                UsageRow(
                    label=USAGE_LABEL_CLOUD,
                    minutes=float(getattr(summary, "cloud_minutes", 0.0)),
                    words=int(getattr(summary, "cloud_words", 0)),
                    allowance_minutes=allowance,
                ),
                UsageRow(
                    label=USAGE_LABEL_LOCAL,
                    minutes=float(getattr(summary, "local_minutes", 0.0)),
                    words=int(getattr(summary, "local_words", 0)),
                    allowance_minutes=None,
                ),
            ),
        )

    # -- persistence -----------------------------------------------------

    def refresh(self) -> None:
        """Re-read install state from disk and the licence from its service."""
        self._section.refresh()
        self._license_status = self._read_license()

    def as_config(self) -> dict:
        """Every key this tab owns, at its current value."""
        return {
            CONFIG_CLOUD_MODE: self._cloud_mode,
            CONFIG_BYOK_PROVIDER: self._byok_provider,
        }

    def apply(self) -> dict:
        """The keys that differ from the config this model was built on.

        ``engine_id`` and ``model_id`` are never in here: picking a model is a
        live hot swap, already written and saved by the section.
        """
        current = self.as_config()
        return {key: value for key, value in current.items() if self._original[key] != value}

    def mark_saved(self) -> None:
        """Called once ``apply``'s dict has been persisted."""
        self._original = self.as_config()

    def _read_license(self) -> Any | None:
        if self._license_provider is None:
            return None
        return self._license_provider()


def _one_of(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """``value`` when the config holds one of ``allowed``, else ``default``.

    A config file edited by hand can name a mode that no longer exists; the
    tab shows the default rather than raising on a window that has to open.
    """
    if value in allowed:
        return value
    if value is not None:
        logger.warning("Ignoring unknown setting %r; using %r", value, default)
    return default


# -- The tab ------------------------------------------------------------------
#
# Everything below draws :class:`EngineTabModel`. AppKit is imported inside the
# methods that need it (and inside the ``ui.settings.base`` helpers), so this
# module stays importable in a headless test run.


def _make_radio(title: str, on: bool, theme: Any, action: Callable[[Any], None]) -> Any:
    """One radio button. AppKit groups radios that share a superview."""
    from Cocoa import NSButton, NSMakeRect, NSOffState, NSOnState

    button = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 22))
    button.setButtonType_(4)  # NSRadioButton
    button.setTitle_(str(title))
    button.setState_(NSOnState if on else NSOffState)
    button.setAppearance_(theme.control_appearance())
    button.sizeToFit()
    bind_action(button, action)
    return button


def _make_progress_bar(percent: float, theme: Any, width: int = 240) -> Any:
    """A determinate bar filled to ``percent``."""
    from Cocoa import NSMakeRect, NSProgressIndicator

    bar = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(0, 0, width, 12))
    bar.setStyle_(0)  # NSProgressIndicatorBarStyle
    bar.setIndeterminate_(False)
    bar.setMinValue_(0.0)
    bar.setMaxValue_(100.0)
    bar.setDoubleValue_(float(percent))
    bar.setAppearance_(theme.control_appearance())
    return bar


def _replace_arranged(stack: Any, views: list[Any]) -> None:
    """Swap everything a stack view holds for ``views``."""
    for existing in list(stack.arrangedSubviews()):
        stack.removeArrangedSubview_(existing)
        existing.removeFromSuperview()
    for view in views:
        if view is not None:
            stack.addArrangedSubview_(view)


class EngineTab:
    """Settings → Engine, rendered from :class:`EngineTabModel`.

    ``store`` is injectable so a test can hand in a :class:`ModelStore` over a
    temporary root; the window passes nothing and gets the real one.
    """

    identifier = TAB_ENGINE
    title = "Engine"

    def __init__(self, store: ModelStore | None = None) -> None:
        self._store = store
        self.model: EngineTabModel | None = None
        self._context: TabContext | None = None
        self._theme: Any = None
        self._view: Any = None
        self._rows_stack: Any = None
        self._usage_stack: Any = None
        self._cloud_buttons: dict[str, Any] = {}
        self._license_label: Any = None
        self._provider_row: Any = None
        self._provider_popup: Any = None
        self._byok_label: Any = None
        self._downloads: DownloadController | None = None
        self._downloading_id: str | None = None
        self._sheet: Any = None
        self._sheet_status: Any = None
        self._sheet_bar: Any = None

    # -- building --------------------------------------------------------

    def build(self, context: TabContext) -> Any:
        """Create the tab's view once, from the window's context."""
        assert context is not None, "context is required"
        from Cocoa import NSMakeRect, NSView

        self._context = context
        self._theme = context.theme
        store = self._store if self._store is not None else ModelStore()
        self._store = store
        self.model = EngineTabModel(
            context.config,
            catalog=store.catalog,
            chip=detect_chip(),
            ram_gb=detect_ram_gb(),
            installed=store.is_installed,
            delete=store.delete,
            usage_provider=context.service("usage"),
            license_provider=context.service("license"),
            app=context.app,
            save=context.save,
        )
        self._downloads = DownloadController(
            store,
            dispatch=main_thread_dispatcher(),
            on_change=self._download_changed,
        )

        self._rows_stack = stack_vertical([], spacing=ROW_SPACING)
        self._usage_stack = stack_vertical([], spacing=4)
        content = stack_vertical(
            [
                make_section_title("Local engine", self._theme),
                make_hint(
                    "Models are downloaded once and kept on this Mac. Switching "
                    "model reloads the engine in the background — no restart.",
                    self._theme,
                ),
                self._rows_stack,
                make_section_title("Cloud", self._theme),
                *self._build_cloud_controls(),
                make_section_title(USAGE_TITLE, self._theme),
                self._usage_stack,
            ],
            spacing=ROW_SPACING,
        )

        view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 520, 560))
        view.addSubview_(content)
        self._view = view
        self.refresh()
        return view

    def _build_cloud_controls(self) -> list[Any]:
        """The radio group, the licence line, and the own-key provider row."""
        rows: list[Any] = []
        for option in self.model.cloud_options:
            button = _make_radio(
                option.title,
                option.selected,
                self._theme,
                _bind(self._cloud_clicked, option.mode),
            )
            button.setEnabled_(option.enabled)
            self._cloud_buttons[option.mode] = button
            rows.append(button)
            rows.append(make_hint(option.note, self._theme))

        self._license_label = make_label(NOT_SIGNED_IN, self._theme, size=11)
        rows.append(self._license_label)

        self._provider_popup = make_popup(
            list(self.model.byok_provider_titles),
            self.model.byok_provider_index,
            self._theme,
            self._provider_chosen,
        )
        self._provider_row = stack_horizontal(
            [make_label("Provider", self._theme), self._provider_popup]
        )
        rows.append(self._provider_row)

        self._byok_label = make_hint(
            BYOK_NOTE.format(provider=BYOK_PROVIDER_LABELS[DEFAULT_BYOK_PROVIDER]), self._theme
        )
        rows.append(self._byok_label)
        return rows

    def _build_model_row(self, row: ModelRow) -> Any:
        """One model: name, detail line, and whichever buttons apply."""
        buttons: list[Any] = []
        if row.installed and not row.active:
            buttons.append(
                make_button("Use", self._theme, _bind(self._use_clicked, row.model_id), width=70)
            )
        if row.can_download:
            buttons.append(
                make_button(
                    "Download",
                    self._theme,
                    _bind(self._download_clicked, row.model_id),
                    width=100,
                )
            )
        if row.can_delete:
            buttons.append(
                make_button(
                    "Delete", self._theme, _bind(self._delete_clicked, row.model_id), width=80
                )
            )
        for button in buttons:
            button.setEnabled_(not self._is_downloading)
        labels = stack_vertical(
            [
                make_label(row.display_name, self._theme, size=12, bold=row.active),
                make_hint(row.detail, self._theme),
            ],
            spacing=2,
        )
        return stack_horizontal([labels, *buttons])

    def _build_usage_rows(self) -> list[Any]:
        block = self.model.usage
        if block is None:
            return []
        views: list[Any] = []
        if block.period_label:
            views.append(make_hint(block.period_label, self._theme))
        for row in block.rows:
            views.append(make_label(f"{row.label} · {row.text}", self._theme, size=11))
            if row.has_progress_bar:
                views.append(_make_progress_bar(row.percent, self._theme))
        return views

    # -- refreshing ------------------------------------------------------

    def refresh(self) -> None:
        """Re-read the model and redraw every control this tab owns."""
        if self.model is None or self._view is None:
            return
        self.model.refresh()
        _replace_arranged(self._rows_stack, [self._build_model_row(r) for r in self.model.rows])
        _replace_arranged(self._usage_stack, self._build_usage_rows())
        self._refresh_cloud()

    def _refresh_cloud(self) -> None:
        from Cocoa import NSOffState, NSOnState

        for option in self.model.cloud_options:
            button = self._cloud_buttons.get(option.mode)
            if button is None:
                continue
            button.setEnabled_(option.enabled)
            button.setState_(NSOnState if option.selected else NSOffState)

        line = self.model.license_line
        self._license_label.setStringValue_(line or "")
        self._license_label.setHidden_(line is None)

        note = self.model.byok_note
        self._provider_row.setHidden_(not self.model.show_provider_popup)
        self._provider_popup.selectItemAtIndex_(self.model.byok_provider_index)
        self._byok_label.setStringValue_(note or "")
        self._byok_label.setHidden_(note is None)

    # -- actions ---------------------------------------------------------

    @property
    def _is_downloading(self) -> bool:
        return self._downloads is not None and self._downloads.is_running

    def _use_clicked(self, model_id: str, sender: Any) -> None:
        self.model.select(model_id)
        self.refresh()

    def _delete_clicked(self, model_id: str, sender: Any) -> None:
        refusal = self.model.delete(model_id)
        if refusal is not None:
            self._alert(refusal)
            return
        self.refresh()

    def _cloud_clicked(self, mode: str, sender: Any) -> None:
        self.model.set_cloud_mode(mode)
        self._refresh_cloud()
        self._save_cloud()

    def _provider_chosen(self, sender: Any) -> None:
        self.model.set_byok_provider_index(sender.indexOfSelectedItem())
        self._refresh_cloud()
        self._save_cloud()

    def _save_cloud(self) -> None:
        """Persist the cloud keys as soon as they change, like the model rows."""
        changed = self.model.apply()
        if not changed or self._context is None:
            return
        self._context.save(changed)
        self.model.mark_saved()

    # -- the download sheet ----------------------------------------------

    def _download_clicked(self, model_id: str, sender: Any) -> None:
        if self._is_downloading:
            return
        row = self.model.row(model_id)
        self._downloading_id = model_id
        self._begin_sheet(row)
        self._downloads.start(model_id, total_bytes=row.size_bytes)
        self.refresh()

    def _cancel_clicked(self, sender: Any) -> None:
        self._downloads.cancel()

    def _begin_sheet(self, row: ModelRow) -> None:
        """A sheet with a determinate bar and a Cancel button."""
        from Cocoa import (
            NSBackingStoreBuffered,
            NSMakeRect,
            NSPanel,
            NSWindowStyleMaskTitled,
        )

        width, height = 380, 150
        sheet = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height), NSWindowStyleMaskTitled, NSBackingStoreBuffered, False
        )
        sheet.setTitle_("Downloading")
        self._theme.apply_window_theme(sheet)

        self._sheet_status = make_label("Ready to download", self._theme, size=11)
        self._sheet_bar = _make_progress_bar(0.0, self._theme, width=width - 40)
        cancel = make_button("Cancel", self._theme, self._cancel_clicked, width=96)
        body = stack_vertical(
            [
                make_label(row.display_name, self._theme, size=13, bold=True),
                self._sheet_status,
                self._sheet_bar,
                cancel,
            ]
        )
        sheet.contentView().addSubview_(body)
        self._sheet = sheet
        window = self._view.window() if self._view is not None else None
        if window is not None:
            window.beginSheet_completionHandler_(sheet, None)

    def _download_changed(self, state: Any) -> None:
        """Called on the main thread after every state change."""
        if self._sheet_status is not None:
            self._sheet_status.setStringValue_(state.status_line())
        if self._sheet_bar is not None:
            self._sheet_bar.setDoubleValue_(state.percent)
        if state.phase not in (PHASE_DONE, PHASE_FAILED, PHASE_CANCELLED):
            return
        if state.phase == PHASE_DONE and self._downloading_id is not None:
            self.model.on_download_finished(self._downloading_id)
        if state.phase == PHASE_FAILED:
            self._alert(state.status_line())
        self._downloading_id = None
        self._end_sheet()
        self.refresh()

    def _end_sheet(self) -> None:
        if self._sheet is None:
            return
        window = self._view.window() if self._view is not None else None
        if window is not None:
            window.endSheet_(self._sheet)
        self._sheet.orderOut_(None)
        self._sheet = None
        self._sheet_status = None
        self._sheet_bar = None

    def _alert(self, message: str) -> None:
        import ui_alerts

        ui_alerts.show_alert("Murmur", message)


def _bind(handler: Callable[..., None], value: Any) -> Callable[[Any], None]:
    """``handler(value, sender)`` as the one-argument callable actions take."""
    return lambda sender: handler(value, sender)


register_tab(EngineTab)


__all__ = [
    "BYOK_PROVIDERS",
    "BYOK_PROVIDER_LABELS",
    "CLOUD_MODES",
    "CLOUD_MODE_MURMUR",
    "CLOUD_MODE_OFF",
    "CLOUD_MODE_OWN_KEY",
    "CLOUD_MODE_TO_ENGINE",
    "CONFIG_BYOK_PROVIDER",
    "CONFIG_CLOUD_MODE",
    "DEFAULT_BYOK_PROVIDER",
    "DEFAULT_CLOUD_MODE",
    "CloudOption",
    "EngineTab",
    "EngineTabModel",
    "ModelRow",
    "UsageBlock",
    "UsageRow",
    "format_license_line",
]
