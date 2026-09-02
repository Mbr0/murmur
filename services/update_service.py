#!/usr/bin/env python3
"""Signed-build updater (decision D4).

Sparkle 2 through PyObjC was the first option in D4. This repository vendors no
Sparkle framework, and whether Sparkle's Objective-C classes load from inside a
PyInstaller bundle cannot be established here, so D4 lands on the documented
fallback: a minimal updater that installs only a build whose Developer ID
signature and Team ID we recognise.

The chain, in order, is:

1. :class:`UpdateFeed` reads a JSON release feed (GitHub releases API by
   default) and turns it into an :class:`UpdateInfo`.
2. :func:`check_for_update` compares the feed version with the running version
   using semantic-version precedence.
3. :func:`download_dmg` streams the disk image to disk with progress.
4. :func:`verify_signature` runs ``codesign --verify --deep --strict``,
   ``spctl --assess --type open --context context:primary-signature`` and
   ``codesign -dv --verbose=4`` on the DMG, and refuses anything whose
   ``TeamIdentifier`` is not the expected one.
5. :func:`install_update` mounts the DMG, copies the app next to the running
   one, swaps by rename and relaunches.

Nothing installs before step 4 has passed: :meth:`UpdateService.download_and_install`
calls the verification and lets :class:`UpdateVerificationError` propagate.

Every subprocess call goes through an injectable ``runner`` and every network
read through an injectable ``opener``, so the tests never touch the network,
``codesign`` or ``hdiutil``.

Standard library only, by design: the updater has to keep working when the
bundle it is replacing is broken.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Release feed. GitHub's "latest release" JSON; injectable for tests and for a
#: future self-hosted appcast.
DEFAULT_FEED_URL = "https://api.github.com/repos/Mbr0/murmur/releases/latest"

#: Developer ID team the updater accepts. Empty on purpose in the repository:
#: a release build supplies it (see "Updater Team ID" in RELEASE_SIGNING.md),
#: either through the environment variable below or through the ``team_id``
#: field of ``Contents/Resources/build_info.json``, which
#: ``scripts/build_pyinstaller.sh`` writes from ``APPLE_TEAM_ID``.
EXPECTED_TEAM_ID = ""

#: Overrides :data:`EXPECTED_TEAM_ID` at runtime.
TEAM_ID_ENV_VAR = "MURMUR_EXPECTED_TEAM_ID"

#: Marker written into the bundle by the build script; the About text reads the
#: same file to label internal builds.
BUILD_INFO_NAME = "build_info.json"

_USER_AGENT = "Murmur-Updater"
_NETWORK_TIMEOUT_S = 30.0
_SUBPROCESS_TIMEOUT_S = 120.0
_MOUNT_TIMEOUT_S = 180.0
_CHUNK_BYTES = 256 * 1024

_VERSION_RE = re.compile(
    r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([0-9A-Za-z.\-]+))?(?:\+[0-9A-Za-z.\-]+)?\s*$"
)
_NUMERIC_RE = re.compile(r"^\d+$")
_TEAM_ID_RE = re.compile(r"^TeamIdentifier=(.+)$", re.MULTILINE)

#: ``codesign -dv`` prints this for an ad-hoc signature.
_TEAM_ID_ABSENT = "not set"


class UpdateError(Exception):
    """Any failure of the update flow."""


class UpdateFeedError(UpdateError):
    """The release feed could not be read or understood."""


class UpdateVerificationError(UpdateError):
    """The downloaded build is not a build we are willing to install."""


@dataclass(frozen=True)
class UpdateInfo:
    """One installable release."""

    version: str
    dmg_url: str
    notes: str = ""
    published_at: str = ""
    size_bytes: int | None = None


#: ``(bytes_done, total_bytes_or_None) -> None``.
ProgressCallback = Callable[[int, int | None], None]

#: ``(argv) -> CompletedProcess``; the seam over every subprocess call.
Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]

#: ``(url) -> file-like``; the seam over every network read.
Opener = Callable[[str], Any]


# --------------------------------------------------------------------------
# version comparison
# --------------------------------------------------------------------------


def _version_key(text: str) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
    """Sort key with semantic-version precedence.

    ``1.2.3`` outranks ``1.2.3-rc.1``: a release gets the ``(1,)`` marker and a
    pre-release ``(0, *identifiers)``. Numeric identifiers rank below
    alphanumeric ones, as SemVer 2.0.0 requires.
    """
    match = _VERSION_RE.match(text or "")
    if match is None:
        raise ValueError(f"not a version: {text!r}")

    release = (
        int(match.group(1)),
        int(match.group(2) or 0),
        int(match.group(3) or 0),
    )
    prerelease = match.group(4)
    if not prerelease:
        return release, (1,)

    identifiers: list[tuple[int, Any]] = []
    for part in prerelease.split("."):
        if _NUMERIC_RE.match(part):
            identifiers.append((0, int(part)))
        else:
            identifiers.append((1, part))
    return release, (0, *identifiers)


def compare_versions(left: str, right: str) -> int:
    """Return -1, 0 or 1 comparing two semantic versions. ``v`` prefix allowed."""
    left_key = _version_key(left)
    right_key = _version_key(right)
    return (left_key > right_key) - (left_key < right_key)


def is_newer(candidate: str, current: str) -> bool:
    """True when ``candidate`` is a strictly higher version than ``current``."""
    return compare_versions(candidate, current) > 0


# --------------------------------------------------------------------------
# feed
# --------------------------------------------------------------------------


def _urlopen(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(request, timeout=_NETWORK_TIMEOUT_S)  # noqa: S310 - https feed URL


class UpdateFeed:
    """A JSON release feed, shaped like the GitHub releases API."""

    def __init__(self, url: str = DEFAULT_FEED_URL, opener: Opener | None = None) -> None:
        assert url, "feed url is required"
        self.url = url
        self._opener = opener or _urlopen

    def fetch(self) -> UpdateInfo:
        """Read the feed and return the release it advertises."""
        try:
            with self._opener(self.url) as response:
                raw = response.read()
        except UpdateError:
            raise
        except Exception as error:  # noqa: BLE001 - urllib raises a wide family
            raise UpdateFeedError(f"could not read the release feed at {self.url}: {error}") from error

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise UpdateFeedError(f"release feed at {self.url} is not JSON: {error}") from error
        return self.parse(payload)

    @staticmethod
    def parse(payload: Any) -> UpdateInfo:
        """Turn a GitHub release object into an :class:`UpdateInfo`."""
        if not isinstance(payload, dict):
            raise UpdateFeedError(f"release feed must be a JSON object, got {type(payload).__name__}")
        if payload.get("draft"):
            raise UpdateFeedError("release feed points at a draft release")

        version = str(payload.get("tag_name") or payload.get("version") or "").strip()
        if not version:
            raise UpdateFeedError("release feed has no tag_name")
        try:
            _version_key(version)
        except ValueError as error:
            raise UpdateFeedError(str(error)) from error

        asset = _select_dmg_asset(payload)
        return UpdateInfo(
            version=version.lstrip("v"),
            dmg_url=str(asset["browser_download_url"]),
            notes=str(payload.get("body") or ""),
            published_at=str(payload.get("published_at") or ""),
            size_bytes=_as_int(asset.get("size")),
        )


def _select_dmg_asset(payload: dict) -> dict:
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise UpdateFeedError("release feed has no assets list")
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        url = asset.get("browser_download_url")
        if name.lower().endswith(".dmg") and url:
            return asset
    raise UpdateFeedError(f"release {payload.get('tag_name')!r} carries no .dmg asset")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def check_for_update(current_version: str, feed: UpdateFeed | None = None) -> UpdateInfo | None:
    """Return the advertised release when it is newer than ``current_version``."""
    assert current_version, "current_version is required"
    feed = feed or UpdateFeed()
    info = feed.fetch()
    try:
        newer = is_newer(info.version, current_version)
    except ValueError as error:
        raise UpdateFeedError(str(error)) from error
    return info if newer else None


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def download_dmg(
    url: str,
    dest: str | Path,
    progress: ProgressCallback | None = None,
    opener: Opener | None = None,
) -> Path:
    """Stream ``url`` into ``dest``, reporting progress, and return ``dest``.

    Writes to ``<dest>.part`` and renames, so a torn download never looks like
    a finished one.
    """
    assert url, "url is required"
    opener = opener or _urlopen
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")

    try:
        with opener(url) as response:
            total = _content_length(response)
            done = 0
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
    except UpdateError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as error:  # noqa: BLE001 - urllib and OSError family
        partial.unlink(missing_ok=True)
        raise UpdateError(f"download of {url} failed: {error}") from error

    os.replace(partial, dest)
    return dest


def _content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    return _as_int(getter("Content-Length"))


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        timeout=_MOUNT_TIMEOUT_S if argv and argv[0] == "hdiutil" else _SUBPROCESS_TIMEOUT_S,
    )


def bundle_resources_dir() -> Path | None:
    """``Contents/Resources`` of the running bundle, or None outside one.

    PyInstaller 6 puts ``sys._MEIPASS`` at ``Contents/Frameworks``; the sibling
    ``Contents/Resources`` is where the build script writes ``build_info.json``.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    base = Path(meipass)
    if base.name == "Resources":
        return base
    sibling = base.parent / "Resources"
    return sibling if sibling.is_dir() else base


def read_build_info() -> dict:
    """Read ``build_info.json`` from the running bundle; ``{}`` when absent."""
    resources = bundle_resources_dir()
    if resources is None:
        return {}
    path = resources / BUILD_INFO_NAME
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def expected_team_id() -> str:
    """Team ID this build will accept: env var, then build_info.json, then constant."""
    from_env = os.environ.get(TEAM_ID_ENV_VAR, "").strip()
    if from_env:
        return from_env
    from_bundle = str(read_build_info().get("team_id") or "").strip()
    if from_bundle:
        return from_bundle
    return EXPECTED_TEAM_ID.strip()


def _require_ok(result: subprocess.CompletedProcess[str], what: str) -> None:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise UpdateVerificationError(f"{what} (exit {result.returncode}): {detail}")


def parse_team_identifier(result: subprocess.CompletedProcess[str]) -> str | None:
    """Extract ``TeamIdentifier`` from ``codesign -dv`` output (it writes stderr)."""
    text = f"{result.stderr or ''}\n{result.stdout or ''}"
    match = _TEAM_ID_RE.search(text)
    if match is None:
        return None
    value = match.group(1).strip()
    return None if value == _TEAM_ID_ABSENT else value


def verify_signature(
    dmg_or_app_path: str | Path,
    runner: Runner | None = None,
    expected_team: str | None = None,
) -> None:
    """Raise :class:`UpdateVerificationError` unless the build is one we trust.

    Three gates: the signature is intact and strict, Gatekeeper assesses it as
    openable on its primary signature, and the Team ID matches. An ad-hoc build
    (``TeamIdentifier=not set``) is never acceptable, and a Developer ID build
    is refused outright when no expected Team ID is configured — silently
    trusting an unknown team would defeat the point of the check.
    """
    runner = runner or _run
    target = str(dmg_or_app_path)

    _require_ok(
        runner(["codesign", "--verify", "--deep", "--strict", target]),
        f"codesign rejected {target}",
    )
    _require_ok(
        runner(["spctl", "--assess", "--type", "open", "--context", "context:primary-signature", target]),
        f"Gatekeeper rejected {target}",
    )

    details = runner(["codesign", "-dv", "--verbose=4", target])
    _require_ok(details, f"could not read the signature of {target}")

    team = parse_team_identifier(details)
    if team is None:
        raise UpdateVerificationError(
            f"{target} carries no Developer ID Team ID (ad-hoc or unsigned); refusing to install it."
        )

    expected = expected_team_id() if expected_team is None else expected_team.strip()
    if not expected:
        raise UpdateVerificationError(
            f"{target} is signed by team {team!r} but no expected Team ID is configured. "
            f"Set {TEAM_ID_ENV_VAR} or the build_info.json team_id (see RELEASE_SIGNING.md)."
        )
    if team != expected:
        raise UpdateVerificationError(
            f"Team ID mismatch for {target}: signed by {team!r}, expected {expected!r}."
        )


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------


def default_app_path() -> Path | None:
    """The running ``.app`` bundle, or None when running from source."""
    if not getattr(sys, "frozen", False):
        return None
    for parent in Path(sys.executable).resolve().parents:
        if parent.suffix == ".app":
            return parent
    return None


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def install_update(
    dmg_path: str | Path,
    app_path: str | Path,
    runner: Runner | None = None,
    mount_root: str | Path | None = None,
    relaunch: bool = True,
) -> Path:
    """Mount ``dmg_path``, replace ``app_path`` with the app inside, relaunch.

    The swap is two renames on one volume — the new app moves into place only
    after the old one has moved aside — so the window in which no app exists at
    ``app_path`` is a single rename wide. macOS exposes no atomic directory
    exchange through Python.

    Call :func:`verify_signature` on the DMG first; this function assumes the
    image has already been vouched for.
    """
    runner = runner or _run
    dmg_path = Path(dmg_path)
    app_path = Path(app_path)
    if not dmg_path.is_file():
        raise UpdateError(f"{dmg_path} does not exist")

    mountpoint = Path(tempfile.mkdtemp(prefix="murmur-update-", dir=str(mount_root) if mount_root else None))
    staged = app_path.parent / f".{app_path.name}.incoming"
    _remove_tree(staged)

    try:
        attach = runner(
            [
                "hdiutil",
                "attach",
                "-nobrowse",
                "-readonly",
                "-noverify",
                "-mountpoint",
                str(mountpoint),
                str(dmg_path),
            ]
        )
        if attach.returncode != 0:
            detail = (attach.stderr or attach.stdout or "").strip()
            raise UpdateError(f"could not mount {dmg_path} (exit {attach.returncode}): {detail}")

        source = mountpoint / app_path.name
        if not source.is_dir():
            raise UpdateError(f"{dmg_path} does not contain {app_path.name}")
        shutil.copytree(source, staged, symlinks=True)
    finally:
        runner(["hdiutil", "detach", str(mountpoint), "-force"])
        try:
            mountpoint.rmdir()
        except OSError:
            pass

    previous = app_path.parent / f".{app_path.name}.previous"
    _remove_tree(previous)
    if app_path.exists():
        os.replace(app_path, previous)
    try:
        os.replace(staged, app_path)
    except OSError as error:
        if previous.exists() and not app_path.exists():
            os.replace(previous, app_path)
        _remove_tree(staged)
        raise UpdateError(f"could not move the new app into {app_path}: {error}") from error
    _remove_tree(previous)

    if relaunch:
        runner(["open", str(app_path)])
    return app_path


# --------------------------------------------------------------------------
# facade
# --------------------------------------------------------------------------


class UpdateService:
    """What the menu talks to: ``check()`` then ``download_and_install(info)``.

    The "Check for Updates…" menu item is wired elsewhere; this class holds no
    UI state and does no work until it is called.
    """

    def __init__(
        self,
        current_version: str,
        app_path: str | Path | None = None,
        feed: UpdateFeed | None = None,
        runner: Runner | None = None,
        opener: Opener | None = None,
        download_dir: str | Path | None = None,
        expected_team: str | None = None,
    ) -> None:
        assert current_version, "current_version is required"
        self.current_version = current_version
        self.feed = feed or UpdateFeed(opener=opener)
        self._app_path = Path(app_path) if app_path else None
        self._runner = runner
        self._opener = opener
        self._download_dir = Path(download_dir) if download_dir else None
        self._expected_team = expected_team

    @property
    def app_path(self) -> Path:
        """The bundle this service replaces."""
        path = self._app_path or default_app_path()
        if path is None:
            raise UpdateError(
                "Murmur is not running from an .app bundle; updates only apply to installed builds."
            )
        return path

    def check(self) -> UpdateInfo | None:
        """Return the newer release, or None when this build is current."""
        return check_for_update(self.current_version, self.feed)

    def download_and_install(self, info: UpdateInfo, progress: ProgressCallback | None = None) -> Path:
        """Download, verify, then install. Verification failure installs nothing."""
        assert info is not None, "info is required"
        app_path = self.app_path
        scratch = None if self._download_dir else Path(tempfile.mkdtemp(prefix="murmur-download-"))
        directory = self._download_dir or scratch
        assert directory is not None
        dmg_path = directory / f"Murmur-{info.version}.dmg"
        try:
            download_dmg(info.dmg_url, dmg_path, progress=progress, opener=self._opener)
            verify_signature(dmg_path, runner=self._runner, expected_team=self._expected_team)
            return install_update(dmg_path, app_path, runner=self._runner)
        finally:
            dmg_path.unlink(missing_ok=True)
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)


__all__ = [
    "BUILD_INFO_NAME",
    "DEFAULT_FEED_URL",
    "EXPECTED_TEAM_ID",
    "TEAM_ID_ENV_VAR",
    "UpdateError",
    "UpdateFeed",
    "UpdateFeedError",
    "UpdateInfo",
    "UpdateService",
    "UpdateVerificationError",
    "check_for_update",
    "compare_versions",
    "default_app_path",
    "download_dmg",
    "expected_team_id",
    "install_update",
    "is_newer",
    "parse_team_identifier",
    "read_build_info",
    "verify_signature",
]
