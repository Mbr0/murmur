"""Tests for the tabbed Settings shell.

Three things live here and none of them need a window server: the tab
registry (what is shown, in what order, and what happens when a tab's module
is not there), the :class:`~ui.settings.base.TabContext` every tab is handed,
and which tab the window opens on.
"""

import unittest
from unittest.mock import patch

import ui.settings.window as window_module
from ui.settings import (
    TABS,
    TAB_MODULES,
    TAB_ORDER,
    TabLifecycle,
    clear_tabs,
    load_tabs,
    register_tab,
    registered_tabs,
)
from ui.settings.base import (
    TAB_ACCOUNT,
    TAB_ENGINE,
    TAB_GENERAL,
    TAB_PRIVACY,
    TAB_SMART,
    TabContext,
)
from ui.settings.window import (
    CONFIG_LAST_TAB,
    SettingsWindowController,
    initial_tab,
)


def stub_tab(identifier, title=None, close=None):
    """A minimal class satisfying the SettingsTab protocol."""
    body = {
        "identifier": identifier,
        "title": title or identifier.title(),
        "build": lambda self, context: context,
        "refresh": lambda self: None,
    }
    if close is not None:
        body["close"] = close
    return type(f"Stub{identifier.title()}Tab", (), body)


class RegistryTestCase(unittest.TestCase):
    """The registry is module state; every test gets it empty and puts it back."""

    def setUp(self):
        self._saved = list(TABS)
        clear_tabs()

    def tearDown(self):
        TABS[:] = self._saved


class TabRegistryTests(RegistryTestCase):
    def test_registered_tabs_follow_display_order_not_registration_order(self):
        for identifier in (TAB_ACCOUNT, TAB_GENERAL, TAB_PRIVACY):
            register_tab(stub_tab(identifier))

        self.assertEqual(
            [cls.identifier for cls in registered_tabs()],
            [TAB_GENERAL, TAB_PRIVACY, TAB_ACCOUNT],
        )

    def test_registering_the_same_identifier_replaces_rather_than_duplicates(self):
        register_tab(stub_tab(TAB_GENERAL, title="First"))
        register_tab(stub_tab(TAB_GENERAL, title="Second"))

        tabs = registered_tabs()
        self.assertEqual(len(tabs), 1)
        self.assertEqual(tabs[0].title, "Second")

    def test_an_unknown_identifier_is_refused(self):
        with self.assertRaises(AssertionError):
            register_tab(stub_tab("shortcuts"))

    def test_a_tab_without_a_title_is_refused(self):
        with self.assertRaises(AssertionError):
            register_tab(type("Nameless", (), {"identifier": TAB_SMART, "title": ""}))

    def test_register_tab_returns_the_class_so_it_can_decorate(self):
        cls = stub_tab(TAB_ENGINE)
        self.assertIs(register_tab(cls), cls)


class LoadTabsTests(RegistryTestCase):
    """Wave 3 writes the five tab modules in parallel, so a missing one is
    normal and must not take the window down."""

    def setUp(self):
        super().setUp()
        self.imported = []
        self.by_module = {name: key for key, name in TAB_MODULES.items()}

    def importer(self, missing=()):
        def _import(module_name):
            self.imported.append(module_name)
            identifier = self.by_module[module_name]
            if identifier in missing:
                # ``name`` is what the real import machinery sets, and what
                # tells "this tab does not exist" from "this tab is broken".
                raise ModuleNotFoundError(
                    f"No module named {module_name!r}", name=module_name
                )
            register_tab(stub_tab(identifier))

        return _import

    def test_every_module_is_imported_and_registered(self):
        tabs = load_tabs(importer=self.importer())

        self.assertEqual([cls.identifier for cls in tabs], list(TAB_ORDER))
        self.assertEqual(sorted(self.imported), sorted(TAB_MODULES.values()))

    def test_a_missing_module_is_skipped_and_logged(self):
        with self.assertLogs("ui.settings", level="INFO") as captured:
            tabs = load_tabs(importer=self.importer(missing={TAB_SMART, TAB_ACCOUNT}))

        self.assertEqual(
            [cls.identifier for cls in tabs],
            [TAB_GENERAL, TAB_ENGINE, TAB_PRIVACY],
        )
        self.assertIn("smart", "\n".join(captured.output))

    def test_a_broken_module_is_not_swallowed(self):
        def explode(module_name):
            raise ImportError(f"{module_name} is broken")

        with self.assertRaises(ImportError):
            load_tabs(importer=explode)

    def test_a_tab_whose_own_import_is_missing_is_not_read_as_a_missing_tab(self):
        """The tab module is there; something it imports is not. That is a bug
        to see, not a tab to drop — otherwise a typo'd import hides a whole tab."""

        def explode(module_name):
            raise ModuleNotFoundError(
                "No module named 'services.nonexistent'", name="services.nonexistent"
            )

        with self.assertRaises(ModuleNotFoundError):
            load_tabs(importer=explode)

    def test_a_missing_submodule_of_the_tab_itself_still_counts_as_missing(self):
        def explode(module_name):
            raise ModuleNotFoundError(
                f"No module named {module_name}.helpers", name=f"{module_name}.helpers"
            )

        with self.assertLogs("ui.settings", level="INFO"):
            self.assertEqual((), load_tabs(importer=explode))

    def test_a_module_not_found_error_without_a_name_is_not_swallowed(self):
        def explode(module_name):
            raise ModuleNotFoundError("no idea which module")

        with self.assertRaises(ModuleNotFoundError):
            load_tabs(importer=explode)


class TabContextTests(unittest.TestCase):
    def context(self, **overrides):
        defaults = dict(
            config={"language": "fr"},
            save=lambda changed: None,
            app=None,
            theme=object(),
        )
        defaults.update(overrides)
        return TabContext(**defaults)

    def test_services_default_to_empty_and_service_returns_the_default(self):
        context = self.context()

        self.assertEqual(context.services, {})
        self.assertIsNone(context.service("license"))
        self.assertEqual(context.service("usage", "none"), "none")

    def test_service_returns_an_injected_provider(self):
        provider = object()
        context = self.context(services={"keychain": provider})

        self.assertIs(context.service("keychain"), provider)

    def test_app_call_is_a_no_op_without_a_running_app(self):
        self.assertIsNone(self.context().app_call("reload_hotkey", prompt=False))

    def test_app_call_forwards_arguments_to_the_app(self):
        calls = []

        class FakeApp:
            def reload_hotkey(self, prompt=True):
                calls.append(prompt)
                return "reloaded"

        context = self.context(app=FakeApp())

        self.assertEqual(context.app_call("reload_hotkey", prompt=False), "reloaded")
        self.assertEqual(calls, [False])

    def test_app_call_ignores_a_method_the_app_does_not_have(self):
        self.assertIsNone(self.context(app=object()).app_call("set_launch_at_login", True))


class InitialTabTests(unittest.TestCase):
    AVAILABLE = (TAB_GENERAL, TAB_ENGINE, TAB_PRIVACY)

    def test_the_requested_tab_wins(self):
        config = {CONFIG_LAST_TAB: TAB_ENGINE}

        self.assertEqual(initial_tab(config, self.AVAILABLE, TAB_PRIVACY), TAB_PRIVACY)

    def test_the_remembered_tab_is_used_when_none_is_requested(self):
        config = {CONFIG_LAST_TAB: TAB_ENGINE}

        self.assertEqual(initial_tab(config, self.AVAILABLE), TAB_ENGINE)

    def test_a_remembered_tab_that_is_not_available_falls_back_to_the_first(self):
        config = {CONFIG_LAST_TAB: TAB_ACCOUNT}

        self.assertEqual(initial_tab(config, self.AVAILABLE), TAB_GENERAL)

    def test_a_requested_tab_that_is_not_available_falls_back_to_the_remembered_one(self):
        config = {CONFIG_LAST_TAB: TAB_PRIVACY}

        self.assertEqual(initial_tab(config, self.AVAILABLE, TAB_SMART), TAB_PRIVACY)

    def test_nothing_opens_when_no_tab_registered(self):
        self.assertIsNone(initial_tab({}, ()))


class ControllerConfigTests(unittest.TestCase):
    """The controller's config half: merging a tab's changes and remembering
    the open tab. No AppKit is touched by any of it."""

    def controller(self, config=None):
        self.saved = []
        return SettingsWindowController(
            app=object(),
            config=config if config is not None else {"language": "auto"},
            save=self.saved.append,
            theme=object(),
        )

    def test_save_merges_changed_keys_into_the_live_config(self):
        controller = self.controller()

        controller.save({"language": "fr", "appearance_mode": "dark"})

        self.assertEqual(controller.config["language"], "fr")
        self.assertEqual(controller.config["appearance_mode"], "dark")
        self.assertEqual(self.saved, [controller.config])

    def test_saving_nothing_writes_nothing(self):
        controller = self.controller()

        controller.save({})

        self.assertEqual(self.saved, [])

    def test_remembering_a_tab_persists_the_last_tab_key(self):
        controller = self.controller()

        controller.remember_tab(TAB_PRIVACY)

        self.assertEqual(controller.config[CONFIG_LAST_TAB], TAB_PRIVACY)
        self.assertEqual(len(self.saved), 1)

    def test_remembering_the_same_tab_twice_writes_once(self):
        controller = self.controller()

        controller.remember_tab(TAB_ENGINE)
        controller.remember_tab(TAB_ENGINE)

        self.assertEqual(len(self.saved), 1)

    def test_the_context_carries_the_controllers_own_save(self):
        controller = self.controller()
        context = controller.context()

        context.save({"language": "nl"})

        self.assertEqual(controller.config["language"], "nl")
        self.assertIs(context.config, controller.config)
        self.assertIs(context.app, controller.app)


class TabTeardownTests(unittest.TestCase):
    """Closing the window must give back everything the tabs hold open.

    The Account tab polls Boske on a timer and the General tab can be holding a
    keyboard monitor; both outlive the view unless something calls ``close``.
    """

    def controller(self, tabs):
        controller = SettingsWindowController(
            app=None,
            config={"language": "auto"},
            save=lambda changed: None,
            theme=object(),
        )
        controller.tabs = tabs
        controller.identifiers = tuple(tabs)
        return controller

    def closing_tab(self, identifier, closed):
        cls = stub_tab(identifier, close=lambda self: closed.append(identifier))
        return cls()

    def test_closing_the_window_closes_every_tab(self):
        closed = []
        controller = self.controller(
            {
                TAB_GENERAL: self.closing_tab(TAB_GENERAL, closed),
                TAB_ACCOUNT: self.closing_tab(TAB_ACCOUNT, closed),
            }
        )

        controller.close()

        self.assertEqual(sorted(closed), sorted([TAB_GENERAL, TAB_ACCOUNT]))

    def test_the_window_delegate_route_closes_the_tabs_too(self):
        """The title-bar red button never reaches ``close``; it reaches this."""
        closed = []
        controller = self.controller({TAB_ACCOUNT: self.closing_tab(TAB_ACCOUNT, closed)})

        controller.window_will_close()

        self.assertEqual(closed, [TAB_ACCOUNT])

    def test_a_tab_without_a_close_is_skipped(self):
        closed = []
        controller = self.controller(
            {
                TAB_PRIVACY: stub_tab(TAB_PRIVACY)(),
                TAB_ACCOUNT: self.closing_tab(TAB_ACCOUNT, closed),
            }
        )

        controller.close_tabs()

        self.assertEqual(closed, [TAB_ACCOUNT])

    def test_a_tab_that_fails_to_close_is_logged_and_the_rest_still_close(self):
        closed = []

        def boom(self):
            raise RuntimeError("teardown went wrong")

        controller = self.controller(
            {
                TAB_GENERAL: stub_tab(TAB_GENERAL, close=boom)(),
                TAB_ACCOUNT: self.closing_tab(TAB_ACCOUNT, closed),
            }
        )

        with self.assertLogs("ui.settings.window", level="WARNING") as captured:
            controller.close_tabs()

        self.assertEqual(closed, [TAB_ACCOUNT])
        self.assertIn(TAB_GENERAL, "\n".join(captured.output))

    def test_closing_twice_is_harmless(self):
        closed = []
        controller = self.controller({TAB_ACCOUNT: self.closing_tab(TAB_ACCOUNT, closed)})

        controller.close()
        controller.close()

        self.assertEqual(closed, [TAB_ACCOUNT, TAB_ACCOUNT])

    def test_the_lifecycle_mixin_is_the_default_for_a_tab_holding_nothing(self):
        class QuietTab(TabLifecycle):
            identifier = TAB_PRIVACY
            title = "Privacy"

        tab = QuietTab()
        controller = self.controller({TAB_PRIVACY: tab})

        self.assertIsNone(tab.close())
        controller.close_tabs()  # no exception, nothing to give back


class ServicesTests(unittest.TestCase):
    """A ``services`` dict must reach every tab's context, end to end."""

    def tearDown(self):
        window_module._CONTROLLER = None

    def test_services_given_to_the_controller_reach_the_context(self):
        provider = object()
        controller = SettingsWindowController(
            app=None,
            config={"language": "auto"},
            save=lambda changed: None,
            theme=object(),
            services={"license": provider},
        )

        self.assertIs(controller.context().services["license"], provider)

    def test_open_settings_forwards_services_to_a_new_controller(self):
        window_module._CONTROLLER = None
        created = {}

        class StubController:
            def __init__(self, *, app=None, services=None):
                created["app"] = app
                created["services"] = services

            def show(self, tab):
                created["tab"] = tab

        with patch.object(window_module, "SettingsWindowController", StubController):
            window_module.open_settings(app=None, tab="account", services={"keychain": "x"})

        self.assertEqual(created["services"], {"keychain": "x"})
        self.assertEqual(created["tab"], "account")


if __name__ == "__main__":
    unittest.main()
