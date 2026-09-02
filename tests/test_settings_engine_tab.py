"""Tests for the Settings "Engine" tab state (Wave 3, E3b).

Everything here is the pure model: no AppKit, no network, no ModelStore on
disk. The tab is handed a catalog plus two callables (``installed`` and
``delete``), so a fake set of ids is enough to describe a machine.
"""

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from engines.model_store import ModelFile, ModelSpec
from services.model_profile_service import (
    CHIP_APPLE_SILICON,
    CHIP_INTEL,
    VOXTRAL_MIN_RAM_GB,
)
from ui.download_sheet import (
    CONFIG_ENGINE_ID,
    CONFIG_MODEL_ID,
    DownloadController,
    DownloadSheetState,
)
from ui.settings.engine_tab import (
    BYOK_PROVIDERS,
    CLOUD_DOWNGRADED_NOTICE,
    CLOUD_MODE_MURMUR,
    CLOUD_MODE_OFF,
    CLOUD_MODE_OWN_KEY,
    CLOUD_MODE_TO_ENGINE,
    CLOUD_MODES,
    CONFIG_BYOK_PROVIDER,
    CONFIG_CLOUD_MODE,
    DEFAULT_BYOK_PROVIDER,
    DEFAULT_CLOUD_MODE,
    FEATURE_CLOUD_VOICE,
    EngineTab,
    EngineTabModel,
    format_license_line,
)

TURBO_Q5 = ModelSpec(
    id="whispercpp-turbo-q5",
    engine="whispercpp",
    display_name="Whisper large-v3-turbo (quantised)",
    files=(ModelFile("turbo-q5.bin", 574_000_000, "a" * 64, "http://x/turbo-q5.bin"),),
    source="http://x",
    license="MIT",
)
TURBO = ModelSpec(
    id="whispercpp-turbo",
    engine="whispercpp",
    display_name="Whisper large-v3-turbo",
    files=(ModelFile("turbo.bin", 1_600_000_000, "b" * 64, "http://x/turbo.bin"),),
    source="http://x",
    license="MIT",
)
VOXTRAL = ModelSpec(
    id="voxtral-4bit",
    engine="voxtral_mlx",
    display_name="Voxtral Mini 4B Realtime (4-bit MLX)",
    files=(
        ModelFile("config.json", 1_000, "c" * 64, "http://x/config.json"),
        ModelFile("weights.bin", 3_100_000_000, "d" * 64, "http://x/weights.bin"),
    ),
    source="http://x",
    license="Apache-2.0",
)
CATALOG = (TURBO_Q5, TURBO, VOXTRAL)


@dataclass
class FakeLicense:
    """Shape of what ``services["license"]()`` returns in Wave 4."""

    pro: bool = True
    cloud_voice: bool = True
    expires_at: str = "31 Jan 2027"
    in_grace: bool = False


@dataclass
class FakeUsage:
    """Shape of the ``UsageSummary`` the usage service will return."""

    cloud_minutes: float = 0.0
    cloud_words: int = 0
    local_minutes: float = 0.0
    local_words: int = 0
    allowance_minutes: int | None = None
    period_label: str = "September 2026"


class FakeApp:
    """Stands in for ``MurmurApp`` and records hot-swap requests."""

    def __init__(self):
        self.reloads = []

    def reload_engine(self, engine_id, model_id):
        self.reloads.append((engine_id, model_id))


class Fixture:
    """A machine: which models are on disk, plus the config the tab writes."""

    def __init__(self, installed=()):
        self.installed_ids = set(installed)
        self.deleted = []
        self.config: dict = {}
        self.saves = 0
        self.saved: list[dict] = []
        self.app = FakeApp()
        self.license = FakeLicense()
        self.usage = None
        self.gate_asked: list[str] = []

    def pro_gate(self, feature):
        """Stands in for ``is_pro_feature_enabled``: the one question the UI asks."""
        self.gate_asked.append(feature)
        return bool(getattr(self.license, feature, False))

    def is_installed(self, model_id):
        return model_id in self.installed_ids

    def delete(self, model_id):
        self.deleted.append(model_id)
        self.installed_ids.discard(model_id)

    def save(self, changed):
        # Wave 2 narrowed this callback: a writer hands over only the keys it
        # owns, never the whole config it loaded (see persistence_service's
        # update_config note). Anything wider would revert other tabs' writes.
        assert isinstance(changed, dict) and changed is not self.config
        self.saved.append(changed)
        self.saves += 1

    def model(
        self,
        chip=CHIP_APPLE_SILICON,
        ram_gb=VOXTRAL_MIN_RAM_GB,
        default_engine="whispercpp",
        with_license=True,
        with_usage=False,
        with_app=True,
        pro_gate=True,
    ):
        """``pro_gate`` is the injected gate: True for the fixture's own, a
        callable to use that one, or False for a build with no gate at all."""
        if pro_gate is True:
            gate = self.pro_gate if with_license else None
        elif pro_gate is False:
            gate = None
        else:
            gate = pro_gate
        return EngineTabModel(
            self.config,
            catalog=CATALOG,
            chip=chip,
            ram_gb=ram_gb,
            installed=self.is_installed,
            usage_provider=(lambda: self.usage) if with_usage else None,
            license_provider=(lambda: self.license) if with_license else None,
            pro_gate=gate,
            app=self.app if with_app else None,
            save=self.save,
            delete=self.delete,
            default_engine=default_engine,
        )


class LocalModelRowTests(unittest.TestCase):
    def test_an_eligible_machine_gets_a_voxtral_row(self):
        model = Fixture().model(chip=CHIP_APPLE_SILICON, ram_gb=VOXTRAL_MIN_RAM_GB)
        self.assertEqual(
            [row.model_id for row in model.rows],
            ["whispercpp-turbo-q5", "whispercpp-turbo", "voxtral-4bit"],
        )

    def test_an_intel_machine_has_no_voxtral_row(self):
        model = Fixture().model(chip=CHIP_INTEL)
        self.assertEqual(
            [row.model_id for row in model.rows],
            ["whispercpp-turbo-q5", "whispercpp-turbo"],
        )

    def test_an_apple_machine_under_the_ram_bar_has_no_voxtral_row(self):
        model = Fixture().model(chip=CHIP_APPLE_SILICON, ram_gb=8)
        self.assertEqual(len(model.rows), 2)

    def test_a_row_reads_as_size_state_and_licence(self):
        fx = Fixture(installed=["whispercpp-turbo-q5"])
        model = fx.model()
        first, second = model.rows[0], model.rows[1]
        self.assertEqual(first.size_text, "574 MB")
        self.assertEqual(first.state_text, "Installed")
        self.assertEqual(
            first.detail, "574 MB · Installed · MIT · Recommended for this Mac"
        )
        self.assertEqual(second.state_text, "Not downloaded")
        self.assertEqual(second.detail, "1.6 GB · Not downloaded · MIT")

    def test_the_active_row_says_it_is_in_use(self):
        fx = Fixture(installed=["whispercpp-turbo-q5"])
        fx.config.update(
            {CONFIG_ENGINE_ID: "whispercpp", CONFIG_MODEL_ID: "whispercpp-turbo-q5"}
        )
        model = fx.model()
        self.assertEqual(model.rows[0].state_text, "In use")
        self.assertTrue(model.rows[0].active)

    def test_download_is_offered_only_on_a_missing_model(self):
        fx = Fixture(installed=["whispercpp-turbo-q5"])
        model = fx.model()
        self.assertFalse(model.rows[0].can_download)
        self.assertTrue(model.rows[1].can_download)

    def test_selecting_an_installed_row_hot_swaps_through_the_app(self):
        fx = Fixture(installed=["voxtral-4bit"])
        model = fx.model()

        self.assertTrue(model.select("voxtral-4bit"))

        self.assertEqual(fx.app.reloads, [("voxtral_mlx", "voxtral-4bit")])
        self.assertEqual(fx.config[CONFIG_ENGINE_ID], "voxtral_mlx")
        self.assertEqual(fx.config[CONFIG_MODEL_ID], "voxtral-4bit")
        self.assertTrue(model.rows[2].selected)

    def test_selecting_a_missing_row_only_moves_the_highlight(self):
        fx = Fixture()
        model = fx.model()
        self.assertFalse(model.select("voxtral-4bit"))
        self.assertEqual(fx.app.reloads, [])
        self.assertNotIn(CONFIG_MODEL_ID, fx.config)

    def test_a_finished_download_installs_and_activates(self):
        fx = Fixture()
        model = fx.model()
        fx.installed_ids.add("whispercpp-turbo")
        model.on_download_finished("whispercpp-turbo")
        self.assertEqual(fx.app.reloads, [("whispercpp", "whispercpp-turbo")])

    def test_no_app_means_no_reload_and_no_crash(self):
        fx = Fixture(installed=["whispercpp-turbo"])
        model = fx.model(with_app=False)
        self.assertTrue(model.select("whispercpp-turbo"))
        self.assertEqual(fx.config[CONFIG_MODEL_ID], "whispercpp-turbo")


class DeleteRowTests(unittest.TestCase):
    def test_delete_is_offered_on_an_installed_row_that_is_not_in_use(self):
        fx = Fixture(installed=["whispercpp-turbo-q5", "whispercpp-turbo"])
        fx.config.update(
            {CONFIG_ENGINE_ID: "whispercpp", CONFIG_MODEL_ID: "whispercpp-turbo-q5"}
        )
        model = fx.model()
        self.assertFalse(model.rows[0].can_delete)
        self.assertTrue(model.rows[1].can_delete)
        self.assertFalse(model.rows[2].can_delete)

    def test_deleting_a_row_that_is_not_the_selected_one_works(self):
        """The Wave 1 popup could only act on the highlighted model."""
        fx = Fixture(installed=["whispercpp-turbo-q5", "whispercpp-turbo"])
        fx.config.update(
            {CONFIG_ENGINE_ID: "whispercpp", CONFIG_MODEL_ID: "whispercpp-turbo-q5"}
        )
        model = fx.model()
        self.assertEqual(model.selected_model_id, "whispercpp-turbo-q5")

        self.assertIsNone(model.delete("whispercpp-turbo"))

        self.assertEqual(fx.deleted, ["whispercpp-turbo"])
        self.assertEqual(model.selected_model_id, "whispercpp-turbo-q5")
        self.assertFalse(model.rows[1].installed)

    def test_deleting_the_model_in_use_is_refused_with_a_message(self):
        fx = Fixture(installed=["whispercpp-turbo-q5"])
        fx.config.update(
            {CONFIG_ENGINE_ID: "whispercpp", CONFIG_MODEL_ID: "whispercpp-turbo-q5"}
        )
        model = fx.model()
        message = model.delete("whispercpp-turbo-q5")
        self.assertIsNotNone(message)
        self.assertIn("Murmur is using", message)
        self.assertEqual(fx.deleted, [])

    def test_deleting_a_model_that_is_not_downloaded_is_refused(self):
        fx = Fixture()
        model = fx.model()
        message = model.delete("whispercpp-turbo")
        self.assertIn("not downloaded", message)
        self.assertEqual(fx.deleted, [])

    def test_deleting_an_unknown_model_fails_fast(self):
        model = Fixture().model()
        with self.assertRaises(AssertionError):
            model.delete("not-in-catalog")


class CloudSectionTests(unittest.TestCase):
    def test_cloud_is_off_until_the_user_says_otherwise(self):
        model = Fixture().model()
        self.assertEqual(DEFAULT_CLOUD_MODE, CLOUD_MODE_OFF)
        self.assertEqual(model.cloud_mode, CLOUD_MODE_OFF)
        self.assertIsNone(model.cloud_engine_id)
        self.assertIsNone(model.license_line)
        self.assertIsNone(model.byok_note)
        self.assertFalse(model.show_provider_popup)

    def test_the_options_are_the_three_modes_in_order(self):
        model = Fixture().model()
        self.assertEqual([option.mode for option in model.cloud_options], list(CLOUD_MODES))
        self.assertEqual([option.selected for option in model.cloud_options], [True, False, False])

    def test_murmur_cloud_is_disabled_without_the_cloud_voice_entitlement(self):
        fx = Fixture()
        fx.license = FakeLicense(pro=True, cloud_voice=False)
        model = fx.model()
        options = {option.mode: option for option in model.cloud_options}
        self.assertFalse(options[CLOUD_MODE_MURMUR].enabled)
        self.assertTrue(options[CLOUD_MODE_OFF].enabled)
        self.assertTrue(options[CLOUD_MODE_OWN_KEY].enabled)
        self.assertFalse(model.cloud_voice_entitled)

    def test_murmur_cloud_is_disabled_when_nobody_is_signed_in(self):
        model = Fixture().model(with_license=False)
        options = {option.mode: option for option in model.cloud_options}
        self.assertFalse(options[CLOUD_MODE_MURMUR].enabled)

    def test_murmur_cloud_is_enabled_when_entitled(self):
        model = Fixture().model()
        options = {option.mode: option for option in model.cloud_options}
        self.assertTrue(options[CLOUD_MODE_MURMUR].enabled)

    def test_choosing_murmur_cloud_without_the_entitlement_is_refused(self):
        fx = Fixture()
        fx.license = FakeLicense(cloud_voice=False)
        model = fx.model()
        self.assertFalse(model.set_cloud_mode(CLOUD_MODE_MURMUR))
        self.assertEqual(model.cloud_mode, CLOUD_MODE_OFF)
        self.assertEqual(model.apply(), {})

    def test_switching_mode_is_reported_by_apply_and_left_to_the_window(self):
        fx = Fixture()
        model = fx.model()

        self.assertTrue(model.set_cloud_mode(CLOUD_MODE_MURMUR))

        self.assertEqual(model.cloud_mode, CLOUD_MODE_MURMUR)
        self.assertEqual(model.apply(), {CONFIG_CLOUD_MODE: CLOUD_MODE_MURMUR})
        self.assertNotIn(CONFIG_CLOUD_MODE, fx.config)

    def test_setting_the_mode_it_already_has_changes_nothing(self):
        model = Fixture().model()
        self.assertFalse(model.set_cloud_mode(CLOUD_MODE_OFF))
        self.assertEqual(model.apply(), {})

    def test_an_unknown_mode_fails_fast(self):
        model = Fixture().model()
        with self.assertRaises(AssertionError):
            model.set_cloud_mode("teleport")

    def test_the_licence_line_only_shows_under_murmur_cloud(self):
        fx = Fixture()
        model = fx.model()
        self.assertIsNone(model.license_line)
        model.set_cloud_mode(CLOUD_MODE_MURMUR)
        self.assertEqual(model.license_line, "Pro · Cloud voice included · Renews 31 Jan 2027")

    def test_the_licence_line_says_not_signed_in_without_a_provider(self):
        self.assertEqual(format_license_line(None), "Not signed in")

    def test_the_licence_line_names_the_grace_period(self):
        fx = Fixture()
        fx.license = FakeLicense(pro=True, cloud_voice=True, in_grace=True)
        model = fx.model()
        model.set_cloud_mode(CLOUD_MODE_MURMUR)
        self.assertEqual(
            model.license_line, "Pro · Cloud voice included · Grace period until 31 Jan 2027"
        )

    def test_a_lapsed_plan_turns_murmur_cloud_off_rather_than_leaving_it_on(self):
        """A config written while the plan was live, opened after it lapsed.

        Trusting the file would keep audio going to Murmur Cloud under a radio
        that is disabled but still selected — the setting says one thing and
        the app does another. The mode is downgraded, said out loud, and
        written back.
        """
        fx = Fixture()
        fx.config[CONFIG_CLOUD_MODE] = CLOUD_MODE_MURMUR
        fx.license = FakeLicense(pro=False, cloud_voice=False, expires_at=None)

        with self.assertLogs("ui.settings.engine_tab", level="INFO"):
            model = fx.model()

        self.assertEqual(model.cloud_mode, CLOUD_MODE_OFF)
        self.assertEqual(model.downgrade_notice, CLOUD_DOWNGRADED_NOTICE)
        self.assertEqual(model.apply(), {CONFIG_CLOUD_MODE: CLOUD_MODE_OFF})
        options = {option.mode: option for option in model.cloud_options}
        self.assertFalse(options[CLOUD_MODE_MURMUR].enabled)
        self.assertFalse(options[CLOUD_MODE_MURMUR].selected)
        self.assertTrue(options[CLOUD_MODE_OFF].selected)

    def test_a_hand_edited_config_cannot_switch_cloud_on_either(self):
        fx = Fixture()
        fx.config[CONFIG_CLOUD_MODE] = CLOUD_MODE_MURMUR

        with self.assertLogs("ui.settings.engine_tab", level="INFO"):
            model = fx.model(with_license=False)  # no licence service, no gate

        self.assertEqual(model.cloud_mode, CLOUD_MODE_OFF)
        self.assertIsNone(model.cloud_engine_id)

    def test_an_entitled_plan_keeps_the_configured_cloud_mode(self):
        fx = Fixture()
        fx.config[CONFIG_CLOUD_MODE] = CLOUD_MODE_MURMUR
        model = fx.model()

        self.assertEqual(model.cloud_mode, CLOUD_MODE_MURMUR)
        self.assertIsNone(model.downgrade_notice)
        self.assertEqual(model.apply(), {})

    def test_own_key_is_never_downgraded(self):
        fx = Fixture()
        fx.config[CONFIG_CLOUD_MODE] = CLOUD_MODE_OWN_KEY
        fx.license = FakeLicense(pro=False, cloud_voice=False)
        model = fx.model()

        self.assertEqual(model.cloud_mode, CLOUD_MODE_OWN_KEY)
        self.assertIsNone(model.downgrade_notice)
        self.assertEqual(model.apply(), {})

    def test_own_key_shows_the_provider_popup_and_points_at_the_account_tab(self):
        model = Fixture().model()
        model.set_cloud_mode(CLOUD_MODE_OWN_KEY)
        self.assertTrue(model.show_provider_popup)
        self.assertEqual(model.byok_provider, DEFAULT_BYOK_PROVIDER)
        self.assertEqual(model.byok_provider_index, 0)
        self.assertIn("Account tab", model.byok_note)
        self.assertIn("Mistral", model.byok_note)

    def test_the_provider_can_be_changed_and_applies(self):
        fx = Fixture()
        model = fx.model()
        model.set_cloud_mode(CLOUD_MODE_OWN_KEY)
        self.assertTrue(model.set_byok_provider("openai"))
        self.assertEqual(model.byok_provider_index, 1)
        self.assertEqual(
            model.apply(),
            {CONFIG_CLOUD_MODE: CLOUD_MODE_OWN_KEY, CONFIG_BYOK_PROVIDER: "openai"},
        )

    def test_an_unknown_provider_fails_fast(self):
        model = Fixture().model()
        with self.assertRaises(AssertionError):
            model.set_byok_provider("acme")

    def test_the_provider_titles_match_the_provider_ids(self):
        model = Fixture().model()
        self.assertEqual(len(model.byok_provider_titles), len(BYOK_PROVIDERS))
        self.assertEqual(model.byok_provider_titles[0], "Mistral")

    def test_the_mode_to_engine_table_covers_every_mode(self):
        self.assertEqual(set(CLOUD_MODE_TO_ENGINE), set(CLOUD_MODES))
        self.assertIsNone(CLOUD_MODE_TO_ENGINE[CLOUD_MODE_OFF])
        self.assertEqual(CLOUD_MODE_TO_ENGINE[CLOUD_MODE_MURMUR], "cloud")
        self.assertEqual(CLOUD_MODE_TO_ENGINE[CLOUD_MODE_OWN_KEY], "byok")

    def test_the_engine_id_is_read_from_the_table(self):
        model = Fixture().model()
        for mode in CLOUD_MODES:
            model._cloud_mode = mode  # bypass the entitlement gate on purpose
            self.assertEqual(model.cloud_engine_id, CLOUD_MODE_TO_ENGINE[mode])

    def test_a_configured_mode_is_read_back(self):
        fx = Fixture()
        fx.config[CONFIG_CLOUD_MODE] = CLOUD_MODE_OWN_KEY
        fx.config[CONFIG_BYOK_PROVIDER] = "openai"
        model = fx.model()
        self.assertEqual(model.cloud_mode, CLOUD_MODE_OWN_KEY)
        self.assertEqual(model.byok_provider, "openai")
        self.assertEqual(model.apply(), {})


class ProGateTests(unittest.TestCase):
    """Enablement is one question — ``is_pro_feature_enabled(feature)`` — asked
    of the injected gate. The UI never reads an entitlements field to decide
    what a user may click; a status line may still print one."""

    def test_the_gate_decides_and_the_entitlements_object_does_not(self):
        fx = Fixture()
        fx.license = FakeLicense(pro=False, cloud_voice=False)
        model = fx.model(pro_gate=lambda feature: True)

        options = {option.mode: option for option in model.cloud_options}
        self.assertTrue(options[CLOUD_MODE_MURMUR].enabled)
        self.assertTrue(model.cloud_voice_entitled)
        self.assertTrue(model.set_cloud_mode(CLOUD_MODE_MURMUR))

    def test_a_closed_gate_disables_murmur_cloud_however_generous_the_licence(self):
        fx = Fixture()
        fx.license = FakeLicense(pro=True, cloud_voice=True)
        model = fx.model(pro_gate=lambda feature: False)

        options = {option.mode: option for option in model.cloud_options}
        self.assertFalse(options[CLOUD_MODE_MURMUR].enabled)
        self.assertFalse(model.set_cloud_mode(CLOUD_MODE_MURMUR))

    def test_no_gate_at_all_means_every_pro_feature_is_off(self):
        fx = Fixture()
        fx.license = FakeLicense(pro=True, cloud_voice=True)
        model = fx.model(pro_gate=False)

        self.assertFalse(model.cloud_voice_entitled)
        options = {option.mode: option for option in model.cloud_options}
        self.assertFalse(options[CLOUD_MODE_MURMUR].enabled)

    def test_the_gate_is_asked_about_cloud_voice_by_name(self):
        fx = Fixture()
        fx.model().cloud_options

        self.assertIn(FEATURE_CLOUD_VOICE, fx.gate_asked)

    def test_a_gate_that_raises_closes_rather_than_takes_the_tab_down(self):
        def broken(feature):
            raise RuntimeError("the licence service fell over")

        fx = Fixture()
        with self.assertLogs("ui.settings.engine_tab", level="WARNING"):
            model = fx.model(pro_gate=broken)
            self.assertFalse(model.cloud_voice_entitled)

    def test_the_licence_line_still_prints_entitlement_fields(self):
        """A status line is description, not enablement: it may read the lease."""
        fx = Fixture()
        model = fx.model(pro_gate=lambda feature: True)
        model.set_cloud_mode(CLOUD_MODE_MURMUR)

        self.assertEqual(model.license_line, "Pro · Cloud voice included · Renews 31 Jan 2027")


class UsageBlockTests(unittest.TestCase):
    def test_no_provider_means_no_usage_block(self):
        model = Fixture().model(with_usage=False)
        self.assertIsNone(model.usage)

    def test_a_provider_with_nothing_to_report_means_no_usage_block(self):
        fx = Fixture()
        fx.usage = None
        model = fx.model(with_usage=True)
        self.assertIsNone(model.usage)

    def test_without_an_allowance_both_lines_are_plain(self):
        fx = Fixture()
        fx.usage = FakeUsage(
            cloud_minutes=12, cloud_words=1840, local_minutes=48.5, local_words=7200
        )
        block = fx.model(with_usage=True).usage
        self.assertEqual(block.period_label, "September 2026")
        cloud, local = block.rows
        self.assertEqual(cloud.label, "Murmur Cloud")
        self.assertEqual(cloud.text, "12 min · 1,840 words")
        self.assertIsNone(cloud.percent)
        self.assertFalse(cloud.has_progress_bar)
        self.assertEqual(local.label, "On this Mac")
        self.assertEqual(local.text, "48.5 min · 7,200 words")
        self.assertIsNone(local.percent)

    def test_an_allowance_turns_the_cloud_line_into_a_progress_bar(self):
        fx = Fixture()
        fx.usage = FakeUsage(
            cloud_minutes=30,
            cloud_words=4000,
            local_minutes=10,
            local_words=1000,
            allowance_minutes=60,
        )
        block = fx.model(with_usage=True).usage
        cloud, local = block.rows
        self.assertTrue(cloud.has_progress_bar)
        self.assertEqual(cloud.percent, 50.0)
        self.assertEqual(cloud.text, "30 of 60 min · 4,000 words")
        self.assertFalse(local.has_progress_bar)

    def test_going_over_the_allowance_stops_at_a_full_bar(self):
        fx = Fixture()
        fx.usage = FakeUsage(cloud_minutes=90, allowance_minutes=60)
        block = fx.model(with_usage=True).usage
        self.assertEqual(block.rows[0].percent, 100.0)

    def test_a_zero_allowance_is_not_a_division(self):
        fx = Fixture()
        fx.usage = FakeUsage(cloud_minutes=0, allowance_minutes=0)
        block = fx.model(with_usage=True).usage
        self.assertEqual(block.rows[0].percent, 0.0)


class ApplyTests(unittest.TestCase):
    def test_an_untouched_tab_applies_nothing(self):
        fx = Fixture()
        model = fx.model()
        self.assertEqual(model.apply(), {})
        self.assertEqual(fx.config, {})

    def test_only_the_keys_that_moved_come_back(self):
        fx = Fixture()
        model = fx.model()
        model.set_byok_provider("openai")
        self.assertEqual(model.apply(), {CONFIG_BYOK_PROVIDER: "openai"})

    def test_apply_leaves_saving_to_the_window(self):
        fx = Fixture()
        model = fx.model()
        model.set_cloud_mode(CLOUD_MODE_OWN_KEY)
        model.apply()
        self.assertEqual(fx.saves, 0)

    def test_apply_is_empty_once_the_window_has_saved(self):
        fx = Fixture()
        model = fx.model()
        model.set_cloud_mode(CLOUD_MODE_OWN_KEY)
        self.assertEqual(model.apply(), {CONFIG_CLOUD_MODE: CLOUD_MODE_OWN_KEY})
        model.mark_saved()
        self.assertEqual(model.apply(), {})

    def test_the_engine_keys_are_written_live_not_by_apply(self):
        fx = Fixture(installed=["whispercpp-turbo"])
        model = fx.model()
        model.select("whispercpp-turbo")
        self.assertEqual(fx.config[CONFIG_MODEL_ID], "whispercpp-turbo")
        self.assertEqual(fx.saves, 1)
        self.assertEqual(model.apply(), {})


class TabIdentityTests(unittest.TestCase):
    def test_the_tab_names_itself(self):
        self.assertEqual(EngineTab.identifier, "engine")
        self.assertEqual(EngineTab.title, "Engine")


class _FakeSheet:
    """Just enough NSPanel for ``_end_sheet`` to put it away."""

    def __init__(self):
        self.ordered_out = False

    def orderOut_(self, _sender):
        self.ordered_out = True


class _FakeStore:
    """A ModelStore that never gets as far as a byte."""

    def download(self, model_id, progress=None, cancel=None):  # pragma: no cover
        raise AssertionError("the worker is never started in these tests")


class CloseTests(unittest.TestCase):
    """Closing Settings takes the sheet down and leaves the download alone.

    A 1.6 GB model takes minutes, and the user who started it wants it whether
    or not the window is still up — killing it because they closed Settings is
    the regression this class exists for. What must go is the *sheet*: the
    worker used to push progress into views nobody could see. So the sheet is
    detached, the controller runs to completion, and its callbacks tolerate
    having no sheet to draw into. Only the sheet's own Cancel button cancels.
    """

    def _tab(self):
        tab = EngineTab()
        controller = DownloadController(
            _FakeStore(),
            dispatch=lambda run: None,  # the UI thread is not here
            spawn=lambda run: None,  # and neither is the worker
        )
        tab._downloads = controller
        return tab, controller

    def test_a_download_in_flight_is_left_running(self):
        tab, controller = self._tab()
        controller.start("whispercpp-turbo", total_bytes=1_600_000_000)
        self.assertTrue(controller.is_running)

        tab.close()

        self.assertFalse(controller._cancel.is_set())
        self.assertTrue(controller.is_running)

    def test_only_the_cancel_button_cancels(self):
        tab, controller = self._tab()
        controller.start("whispercpp-turbo")

        tab._cancel_clicked(None)

        self.assertTrue(controller._cancel.is_set())

    def test_the_sheet_is_taken_down(self):
        tab, controller = self._tab()
        controller.start("whispercpp-turbo")
        sheet = _FakeSheet()
        tab._sheet = sheet
        tab._sheet_status = object()
        tab._sheet_bar = object()
        tab._downloading_id = "whispercpp-turbo"

        tab.close()

        self.assertTrue(sheet.ordered_out)
        self.assertIsNone(tab._sheet)
        self.assertIsNone(tab._sheet_status)
        self.assertIsNone(tab._sheet_bar)
        # The id stays: the download is still running and the model it
        # installs still has to be put to work when it lands.
        self.assertEqual(tab._downloading_id, "whispercpp-turbo")

    def test_a_download_that_finishes_after_the_close_is_still_installed(self):
        tab, controller = self._tab()
        finished = []
        tab.model = SimpleNamespace(
            on_download_finished=lambda model_id: finished.append(model_id)
        )
        controller.start("whispercpp-turbo")
        tab._sheet = _FakeSheet()
        tab._sheet_status = object()
        tab._sheet_bar = object()
        tab._downloading_id = "whispercpp-turbo"

        tab.close()
        state = DownloadSheetState("whispercpp-turbo")
        state.mark_done()
        tab._download_changed(state)  # the worker's last hop, sheet long gone

        self.assertEqual(finished, ["whispercpp-turbo"])
        self.assertIsNone(tab._downloading_id)

    def test_closing_a_tab_with_nothing_open_is_a_no_op(self):
        # The window closes tabs whether or not they were ever built.
        EngineTab().close()

    def test_closing_twice_is_safe(self):
        tab, controller = self._tab()
        controller.start("whispercpp-turbo")
        tab._sheet = _FakeSheet()
        tab.close()
        tab.close()


if __name__ == "__main__":
    unittest.main()
