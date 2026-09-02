"""Tests for services.update_service.

Nothing here touches the network, ``codesign``, ``spctl`` or ``hdiutil``: the
opener and the runner are the two seams the module exposes, and both are faked.
The filesystem is real (temporary directories), so the rename-based swap is
exercised for what it is.
"""

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services.update_service import (
    DEFAULT_FEED_URL,
    TEAM_ID_ENV_VAR,
    UpdateError,
    UpdateFeed,
    UpdateFeedError,
    UpdateInfo,
    UpdateService,
    UpdateVerificationError,
    check_for_update,
    compare_versions,
    download_dmg,
    expected_team_id,
    install_update,
    is_newer,
    parse_team_identifier,
    verify_signature,
)

TEAM_ID = "AB12CD34EF"
OTHER_TEAM_ID = "ZZ99YY88XX"
DMG_BODY = b"pretend disk image" * 64


def _release(tag="v1.4.0", asset_name="Murmur-1.4.0.dmg", **overrides):
    payload = {
        "tag_name": tag,
        "body": "Fixes the thing.",
        "published_at": "2026-09-02T10:00:00Z",
        "draft": False,
        "assets": [
            {"name": "Murmur-1.4.0.dmg.sha256", "browser_download_url": "https://example.test/sum"},
            {
                "name": asset_name,
                "browser_download_url": f"https://example.test/{asset_name}",
                "size": len(DMG_BODY),
            },
        ],
    }
    payload.update(overrides)
    return payload


class _Response(io.BytesIO):
    """A urlopen stand-in: bytes plus a ``headers`` mapping, usable as a CM."""

    def __init__(self, body, headers=None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))} if headers is None else headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _opener_for(body, headers=None, error=None):
    seen = []

    def opener(url):
        seen.append(url)
        if error is not None:
            raise error
        return _Response(body, headers)

    opener.seen = seen
    return opener


def _json_opener(payload):
    return _opener_for(json.dumps(payload).encode("utf-8"))


def _mkdtemp_in(root):
    """Keep the updater's mount point inside the test's temporary directory."""

    real_mkdtemp = tempfile.mkdtemp

    def mkdtemp(prefix="", dir=None):  # noqa: A002 - mirrors tempfile.mkdtemp
        return real_mkdtemp(prefix=prefix, dir=dir or root)

    return mkdtemp


class FakeRunner:
    """Records argv and replays scripted results, keyed by the first two words."""

    def __init__(self, mountpoint_app_name="Murmur.app", failures=None, team_id=TEAM_ID):
        self.calls = []
        self.failures = failures or {}
        self.team_id = team_id
        self.mountpoint_app_name = mountpoint_app_name

    @property
    def commands(self):
        """Each call as tool plus first flag, paths dropped — asserts on ordering."""
        return [" ".join(part for part in call[:2] if not part.startswith("/")) for call in self.calls]

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        key = " ".join(argv[:2])

        if key in self.failures:
            code, message = self.failures[key]
            return subprocess.CompletedProcess(argv, code, stdout="", stderr=message)

        stderr = ""
        if key == "codesign -dv":
            team = "not set" if self.team_id is None else self.team_id
            stderr = f"Executable=/x\nIdentifier=com.canopystudio.murmur\nTeamIdentifier={team}\n"
        elif key == "hdiutil attach":
            mountpoint = Path(argv[argv.index("-mountpoint") + 1])
            app = mountpoint / self.mountpoint_app_name
            (app / "Contents" / "MacOS").mkdir(parents=True, exist_ok=True)
            (app / "Contents" / "MacOS" / "Murmur").write_text("new build")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr=stderr)


class VersionComparisonTests(unittest.TestCase):
    def test_orders_release_components(self):
        self.assertEqual(compare_versions("1.2.3", "1.2.4"), -1)
        self.assertEqual(compare_versions("1.3.0", "1.2.9"), 1)
        self.assertEqual(compare_versions("2.0.0", "1.99.99"), 1)
        self.assertEqual(compare_versions("1.2.3", "1.2.3"), 0)

    def test_tolerates_v_prefix_and_short_forms(self):
        self.assertEqual(compare_versions("v1.2.3", "1.2.3"), 0)
        self.assertEqual(compare_versions("1.2", "1.2.0"), 0)
        self.assertEqual(compare_versions("2", "1.9.9"), 1)

    def test_release_outranks_prerelease(self):
        self.assertEqual(compare_versions("1.0.0", "1.0.0-rc.1"), 1)
        self.assertEqual(compare_versions("1.0.0-alpha", "1.0.0-beta"), -1)
        self.assertEqual(compare_versions("1.0.0-rc.1", "1.0.0-rc.2"), -1)
        self.assertEqual(compare_versions("1.0.0-alpha", "1.0.0-alpha.1"), -1)

    def test_build_metadata_is_ignored(self):
        self.assertEqual(compare_versions("1.2.3+build.9", "1.2.3"), 0)

    def test_is_newer(self):
        self.assertTrue(is_newer("1.4.0", "1.3.9"))
        self.assertFalse(is_newer("1.3.9", "1.4.0"))
        self.assertFalse(is_newer("1.4.0", "1.4.0"))

    def test_garbage_version_raises(self):
        with self.assertRaises(ValueError):
            compare_versions("nightly", "1.0.0")


class FeedParsingTests(unittest.TestCase):
    def test_default_url_is_the_repo_release_feed(self):
        self.assertEqual(UpdateFeed().url, DEFAULT_FEED_URL)

    def test_parses_a_github_release(self):
        info = UpdateFeed.parse(_release())
        self.assertEqual(info.version, "1.4.0")
        self.assertEqual(info.dmg_url, "https://example.test/Murmur-1.4.0.dmg")
        self.assertEqual(info.notes, "Fixes the thing.")
        self.assertEqual(info.published_at, "2026-09-02T10:00:00Z")
        self.assertEqual(info.size_bytes, len(DMG_BODY))

    def test_fetch_reads_the_configured_url(self):
        opener = _json_opener(_release())
        feed = UpdateFeed(url="https://example.test/feed.json", opener=opener)
        self.assertEqual(feed.fetch().version, "1.4.0")
        self.assertEqual(opener.seen, ["https://example.test/feed.json"])

    def test_non_dmg_assets_are_rejected(self):
        with self.assertRaises(UpdateFeedError):
            UpdateFeed.parse(_release(asset_name="Murmur-1.4.0.zip"))

    def test_draft_release_is_rejected(self):
        with self.assertRaises(UpdateFeedError):
            UpdateFeed.parse(_release(draft=True))

    def test_missing_tag_is_rejected(self):
        payload = _release()
        del payload["tag_name"]
        with self.assertRaises(UpdateFeedError):
            UpdateFeed.parse(payload)

    def test_unparseable_tag_is_rejected(self):
        with self.assertRaises(UpdateFeedError):
            UpdateFeed.parse(_release(tag="nightly"))

    def test_broken_json_is_reported_as_a_feed_error(self):
        feed = UpdateFeed(opener=_opener_for(b"<html>nope</html>"))
        with self.assertRaises(UpdateFeedError):
            feed.fetch()

    def test_network_failure_is_reported_as_a_feed_error(self):
        feed = UpdateFeed(opener=_opener_for(b"", error=OSError("no route to host")))
        with self.assertRaises(UpdateFeedError):
            feed.fetch()


class CheckForUpdateTests(unittest.TestCase):
    def test_returns_info_when_the_feed_is_ahead(self):
        feed = UpdateFeed(opener=_json_opener(_release()))
        info = check_for_update("1.3.0", feed)
        self.assertIsNotNone(info)
        self.assertEqual(info.version, "1.4.0")

    def test_returns_none_when_current(self):
        feed = UpdateFeed(opener=_json_opener(_release()))
        self.assertIsNone(check_for_update("1.4.0", feed))

    def test_returns_none_when_ahead_of_the_feed(self):
        feed = UpdateFeed(opener=_json_opener(_release()))
        self.assertIsNone(check_for_update("2.0.0", feed))


class DownloadTests(unittest.TestCase):
    def test_writes_the_body_and_reports_progress(self):
        seen = []
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nested" / "Murmur.dmg"
            result = download_dmg(
                "https://example.test/Murmur.dmg",
                dest,
                progress=lambda done, total: seen.append((done, total)),
                opener=_opener_for(DMG_BODY),
            )
            self.assertEqual(result, dest)
            self.assertEqual(dest.read_bytes(), DMG_BODY)
        self.assertTrue(seen)
        self.assertEqual(seen[-1], (len(DMG_BODY), len(DMG_BODY)))

    def test_missing_content_length_reports_unknown_total(self):
        seen = []
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "Murmur.dmg"
            download_dmg(
                "https://example.test/Murmur.dmg",
                dest,
                progress=lambda done, total: seen.append((done, total)),
                opener=_opener_for(DMG_BODY, headers={}),
            )
        self.assertEqual(seen[-1], (len(DMG_BODY), None))

    def test_failure_leaves_no_partial_file(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "Murmur.dmg"
            with self.assertRaises(UpdateError):
                download_dmg(
                    "https://example.test/Murmur.dmg",
                    dest,
                    opener=_opener_for(b"", error=OSError("connection reset")),
                )
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [])


class TeamIdentifierTests(unittest.TestCase):
    def test_parses_from_stderr(self):
        result = subprocess.CompletedProcess([], 0, stdout="", stderr=f"TeamIdentifier={TEAM_ID}\n")
        self.assertEqual(parse_team_identifier(result), TEAM_ID)

    def test_ad_hoc_reads_as_absent(self):
        result = subprocess.CompletedProcess([], 0, stdout="", stderr="TeamIdentifier=not set\n")
        self.assertIsNone(parse_team_identifier(result))

    def test_env_var_supplies_the_expected_team(self):
        with patch.dict("os.environ", {TEAM_ID_ENV_VAR: f" {TEAM_ID} "}):
            self.assertEqual(expected_team_id(), TEAM_ID)


class VerificationTests(unittest.TestCase):
    def test_happy_path_runs_the_three_checks_in_order(self):
        runner = FakeRunner()
        verify_signature("/tmp/Murmur.dmg", runner=runner, expected_team=TEAM_ID)
        self.assertEqual(
            runner.commands,
            ["codesign --verify", "spctl --assess", "codesign -dv"],
        )
        self.assertIn("--deep", runner.calls[0])
        self.assertIn("--strict", runner.calls[0])
        self.assertIn("context:primary-signature", runner.calls[1])

    def test_broken_signature_aborts_before_gatekeeper(self):
        runner = FakeRunner(failures={"codesign --verify": (1, "code object is not signed at all")})
        with self.assertRaises(UpdateVerificationError):
            verify_signature("/tmp/Murmur.dmg", runner=runner, expected_team=TEAM_ID)
        self.assertEqual(runner.commands, ["codesign --verify"])

    def test_gatekeeper_rejection_aborts(self):
        runner = FakeRunner(failures={"spctl --assess": (3, "rejected")})
        with self.assertRaises(UpdateVerificationError):
            verify_signature("/tmp/Murmur.dmg", runner=runner, expected_team=TEAM_ID)
        self.assertEqual(runner.commands, ["codesign --verify", "spctl --assess"])

    def test_team_id_mismatch_aborts(self):
        runner = FakeRunner(team_id=OTHER_TEAM_ID)
        with self.assertRaises(UpdateVerificationError) as caught:
            verify_signature("/tmp/Murmur.dmg", runner=runner, expected_team=TEAM_ID)
        self.assertIn(OTHER_TEAM_ID, str(caught.exception))

    def test_ad_hoc_build_is_refused(self):
        runner = FakeRunner(team_id=None)
        with self.assertRaises(UpdateVerificationError):
            verify_signature("/tmp/Murmur.dmg", runner=runner, expected_team=TEAM_ID)

    def test_unset_expected_team_fails_fast(self):
        runner = FakeRunner()
        with patch.dict("os.environ", {TEAM_ID_ENV_VAR: ""}), patch(
            "services.update_service.read_build_info", return_value={}
        ), patch("services.update_service.EXPECTED_TEAM_ID", ""):
            with self.assertRaises(UpdateVerificationError) as caught:
                verify_signature("/tmp/Murmur.dmg", runner=runner)
        self.assertIn(TEAM_ID_ENV_VAR, str(caught.exception))


class InstallTests(unittest.TestCase):
    def _fixture(self, tmp):
        applications = Path(tmp) / "Applications"
        applications.mkdir()
        app = applications / "Murmur.app"
        (app / "Contents" / "MacOS").mkdir(parents=True)
        (app / "Contents" / "MacOS" / "Murmur").write_text("old build")
        dmg = Path(tmp) / "Murmur-1.4.0.dmg"
        dmg.write_bytes(DMG_BODY)
        return app, dmg

    def test_mounts_copies_swaps_and_relaunches(self):
        runner = FakeRunner()
        with TemporaryDirectory() as tmp:
            app, dmg = self._fixture(tmp)
            result = install_update(dmg, app, runner=runner, mount_root=tmp)
            self.assertEqual(result, app)
            self.assertEqual((app / "Contents" / "MacOS" / "Murmur").read_text(), "new build")
            leftovers = [p.name for p in app.parent.iterdir() if p.name != "Murmur.app"]
            self.assertEqual(leftovers, [])
        self.assertEqual(runner.commands, ["hdiutil attach", "hdiutil detach", "open"])

    def test_detaches_even_when_the_image_has_no_app(self):
        runner = FakeRunner(mountpoint_app_name="Something.app")
        with TemporaryDirectory() as tmp:
            app, dmg = self._fixture(tmp)
            with self.assertRaises(UpdateError):
                install_update(dmg, app, runner=runner, mount_root=tmp)
            self.assertEqual((app / "Contents" / "MacOS" / "Murmur").read_text(), "old build")
        self.assertEqual(runner.commands[:2], ["hdiutil attach", "hdiutil detach"])

    def test_mount_failure_leaves_the_app_alone(self):
        runner = FakeRunner(failures={"hdiutil attach": (1, "image not recognized")})
        with TemporaryDirectory() as tmp:
            app, dmg = self._fixture(tmp)
            with self.assertRaises(UpdateError):
                install_update(dmg, app, runner=runner, mount_root=tmp)
            self.assertEqual((app / "Contents" / "MacOS" / "Murmur").read_text(), "old build")

    def test_missing_dmg_raises(self):
        with TemporaryDirectory() as tmp:
            app, _ = self._fixture(tmp)
            with self.assertRaises(UpdateError):
                install_update(Path(tmp) / "absent.dmg", app, runner=FakeRunner(), mount_root=tmp)


class UpdateServiceTests(unittest.TestCase):
    def _service(self, tmp, runner, opener=None, **kwargs):
        applications = Path(tmp) / "Applications"
        applications.mkdir(exist_ok=True)
        app = applications / "Murmur.app"
        (app / "Contents" / "MacOS").mkdir(parents=True, exist_ok=True)
        (app / "Contents" / "MacOS" / "Murmur").write_text("old build")
        service = UpdateService(
            current_version="1.3.0",
            app_path=app,
            feed=UpdateFeed(opener=_json_opener(_release())),
            runner=runner,
            opener=opener or _opener_for(DMG_BODY),
            download_dir=Path(tmp),
            expected_team=TEAM_ID,
            **kwargs,
        )
        return service, app

    def test_check_returns_the_newer_release(self):
        with TemporaryDirectory() as tmp:
            service, _ = self._service(tmp, FakeRunner())
            info = service.check()
        self.assertEqual(info.version, "1.4.0")

    def test_happy_path_calls_download_verify_install_in_order(self):
        runner = FakeRunner()
        seen = []
        with TemporaryDirectory() as tmp:
            service, app = self._service(tmp, runner)
            info = service.check()
            with patch("services.update_service.tempfile.mkdtemp", _mkdtemp_in(tmp)):
                result = service.download_and_install(info, progress=lambda d, t: seen.append(d))
            self.assertEqual(result, app)
            self.assertEqual((app / "Contents" / "MacOS" / "Murmur").read_text(), "new build")
            self.assertEqual(list(Path(tmp).glob("*.dmg")), [])
        self.assertTrue(seen)
        self.assertEqual(
            runner.commands,
            [
                "codesign --verify",
                "spctl --assess",
                "codesign -dv",
                "hdiutil attach",
                "hdiutil detach",
                "open",
            ],
        )

    def test_verification_failure_aborts_before_install(self):
        runner = FakeRunner(team_id=OTHER_TEAM_ID)
        with TemporaryDirectory() as tmp:
            service, app = self._service(tmp, runner)
            with self.assertRaises(UpdateVerificationError):
                service.download_and_install(UpdateInfo(version="1.4.0", dmg_url="https://example.test/a.dmg"))
            self.assertEqual((app / "Contents" / "MacOS" / "Murmur").read_text(), "old build")
        self.assertNotIn("hdiutil attach", runner.commands)

    def test_download_failure_aborts_before_verification(self):
        runner = FakeRunner()
        with TemporaryDirectory() as tmp:
            service, app = self._service(tmp, runner, opener=_opener_for(b"", error=OSError("reset")))
            with self.assertRaises(UpdateError):
                service.download_and_install(UpdateInfo(version="1.4.0", dmg_url="https://example.test/a.dmg"))
            self.assertEqual((app / "Contents" / "MacOS" / "Murmur").read_text(), "old build")
        self.assertEqual(runner.commands, [])

    def test_app_path_outside_a_bundle_is_an_error(self):
        service = UpdateService(current_version="1.3.0", feed=UpdateFeed(opener=_json_opener(_release())))
        with patch("services.update_service.default_app_path", return_value=None):
            with self.assertRaises(UpdateError):
                _ = service.app_path


if __name__ == "__main__":
    unittest.main()
