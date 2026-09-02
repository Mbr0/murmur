#!/usr/bin/env python3
"""Keychain-backed secret storage (Wave 3, decision D6).

Secrets never touch ``~/.murmur_config.json``. The Boske lease and the two
own-key (BYOK) API keys live in the login keychain as generic passwords, one
item per secret, all under one service name.

:class:`KeychainStore` implements the ``SecretStore`` protocol Wave 4's
``services/license_service.py`` consumes — ``get(name)``, ``set(name, value)``,
``delete(name)`` — plus ``has(name)`` for a UI that wants to show "key stored"
without reading the key back.

Four rules the implementation exists to enforce:

1. **The framework, not the CLI.** ``security add-generic-password -w <secret>``
   puts the secret in the process list, where any other user process can read
   it. Everything here goes through ``SecItemAdd`` / ``SecItemCopyMatching`` /
   ``SecItemUpdate`` / ``SecItemDelete`` in ``Security.framework``.
2. **The legacy login keychain, deliberately.** ``kSecAttrAccessible`` is set
   on every write, but the legacy file-based keychain these calls reach ignores
   it: only the data protection keychain honours an accessibility class.
   Selecting that one means adding ``kSecUseDataProtectionKeychain: True`` to
   every dictionary — and it is *not* set here on purpose. That keychain is
   entitlement-gated: a process without ``keychain-access-groups`` (or app
   groups) gets ``errSecMissingEntitlement`` (-34018) from every SecItem call,
   which is every unsigned and internal build, and every run from source.
   Measured, not assumed: with the flag, add and delete return -34018 here;
   without it, the same round trip returns 0. Turning it on is a signing
   change, not a code change, and it would strand every key already stored —
   the two keychains do not see each other's items. Revisit it only together
   with the entitlement and a migration.
3. **No secret in an exception or a log line.** Failures carry an OSStatus and
   the *item name*; the value is never formatted, repr'd or logged. This module
   imports no logger on purpose.
4. **The backend is a seam.** ``backend`` is any object with the four functions
   below, so tests drive a fake and never write to the real keychain. The
   default backend is resolved lazily, so importing this module on Linux CI —
   or anywhere without the framework — costs nothing and raises nothing.

The backend interface (plain values in, OSStatus out; CoreFoundation stays
inside the backend)::

    add(service: str, account: str, secret: bytes) -> int
    copy(service: str, account: str) -> tuple[int, bytes | None]
    update(service: str, account: str, secret: bytes) -> int
    delete(service: str, account: str) -> int

``PyObjCBackend`` is preferred when ``pyobjc-framework-Security`` is installed.
``CTypesBackend`` binds the same four C functions through :mod:`ctypes` and
needs no dependency at all, which is what keeps this working inside the
PyInstaller bundle whether or not the PyObjC bridge was collected.
"""

from __future__ import annotations

from typing import Any

#: One keychain service for the whole app; the item name is the account.
DEFAULT_SERVICE_NAME = "com.canopystudio.murmur"

#: Boske lease JWT obtained by the device-linking flow (D6).
ITEM_LEASE = "boske-lease"
#: Own-key (BYOK) API keys. Never Murmur Cloud credentials.
ITEM_BYOK_MISTRAL = "byok-mistral"
ITEM_BYOK_OPENAI = "byok-openai"

#: Every item this app owns, for "delete everything" flows.
ALL_ITEMS: tuple[str, ...] = (ITEM_LEASE, ITEM_BYOK_MISTRAL, ITEM_BYOK_OPENAI)

#: BYOK provider id -> keychain item name.
BYOK_ITEMS: dict[str, str] = {
    "mistral": ITEM_BYOK_MISTRAL,
    "openai": ITEM_BYOK_OPENAI,
}

ERR_SEC_SUCCESS = 0
ERR_SEC_USER_CANCELED = -128
ERR_SEC_NOT_AVAILABLE = -25291
ERR_SEC_AUTH_FAILED = -25293
ERR_SEC_DUPLICATE_ITEM = -25299
ERR_SEC_ITEM_NOT_FOUND = -25300
ERR_SEC_INTERACTION_NOT_ALLOWED = -25308
ERR_SEC_DECODE = -26275
ERR_SEC_MISSING_ENTITLEMENT = -34018

_STATUS_MEANINGS: dict[int, str] = {
    ERR_SEC_USER_CANCELED: "errSecUserCanceled — the user dismissed the keychain prompt",
    ERR_SEC_NOT_AVAILABLE: "errSecNotAvailable — no keychain is available",
    ERR_SEC_AUTH_FAILED: "errSecAuthFailed — the keychain refused this process",
    ERR_SEC_DUPLICATE_ITEM: "errSecDuplicateItem — the item already exists",
    ERR_SEC_ITEM_NOT_FOUND: "errSecItemNotFound — no such item",
    ERR_SEC_INTERACTION_NOT_ALLOWED: "errSecInteractionNotAllowed — the keychain is locked",
    ERR_SEC_DECODE: "errSecDecode — the stored item could not be decoded",
    ERR_SEC_MISSING_ENTITLEMENT: "errSecMissingEntitlement — this build lacks the keychain entitlement",
}


class KeychainError(Exception):
    """A Keychain call failed. Carries the OSStatus, never the secret."""

    def __init__(self, status: int, operation: str = "", item: str = "") -> None:
        self.status = int(status)
        self.operation = operation
        self.item = item
        meaning = _STATUS_MEANINGS.get(self.status, "unknown OSStatus")
        where = f"{operation} " if operation else ""
        which = f"item {item!r} " if item else ""
        super().__init__(f"Keychain {where}{which}failed ({self.status}: {meaning})")


class KeychainUnavailable(KeychainError):
    """Security.framework is not reachable from this process (non-macOS, mostly)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        Exception.__init__(self, f"macOS Keychain is unavailable: {reason}")
        self.status = ERR_SEC_NOT_AVAILABLE
        self.operation = ""
        self.item = ""


# --------------------------------------------------------------------------
# stores
# --------------------------------------------------------------------------


class KeychainStore:
    """``SecretStore`` over the macOS login keychain.

    One generic-password item per secret: service ``service_name``, account
    ``name``. ``backend`` is injected by tests; left None it is resolved once,
    on first use, by :func:`default_backend`.
    """

    def __init__(self, service_name: str = DEFAULT_SERVICE_NAME, backend: Any = None) -> None:
        assert service_name, "service_name is required"
        self.service_name = service_name
        self._backend = backend

    @property
    def backend(self) -> Any:
        """The Security binding in use; resolved on first access."""
        if self._backend is None:
            self._backend = default_backend()
        return self._backend

    def get(self, name: str) -> str | None:
        """The stored secret, or None when the item does not exist."""
        _check_name(name)
        status, raw = self.backend.copy(self.service_name, name)
        if status == ERR_SEC_ITEM_NOT_FOUND:
            return None
        if status != ERR_SEC_SUCCESS:
            raise KeychainError(status, "read", name)
        if raw is None:
            return None
        try:
            text: str | None = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is None:
            # Raised outside the ``except`` block on purpose: a
            # UnicodeDecodeError repeats the offending bytes, and it must not
            # become this error's ``__context__``.
            raise KeychainError(ERR_SEC_DECODE, "read", name)
        return text

    def set(self, name: str, value: str) -> None:
        """Store ``value``, replacing whatever the item held before."""
        _check_name(name)
        assert isinstance(value, str), "secret must be str"
        if not value:
            raise ValueError(f"refusing to store an empty secret for item {name!r}")
        secret = value.encode("utf-8")
        status = self.backend.add(self.service_name, name, secret)
        if status == ERR_SEC_DUPLICATE_ITEM:
            status = self.backend.update(self.service_name, name, secret)
        if status != ERR_SEC_SUCCESS:
            raise KeychainError(status, "write", name)

    def delete(self, name: str) -> None:
        """Remove the item. Removing what is not there is not an error."""
        _check_name(name)
        status = self.backend.delete(self.service_name, name)
        if status in (ERR_SEC_SUCCESS, ERR_SEC_ITEM_NOT_FOUND):
            return
        raise KeychainError(status, "delete", name)

    def has(self, name: str) -> bool:
        """Whether the item exists. Used by UI that must not read the secret."""
        _check_name(name)
        status, _raw = self.backend.copy(self.service_name, name)
        if status == ERR_SEC_ITEM_NOT_FOUND:
            return False
        if status != ERR_SEC_SUCCESS:
            raise KeychainError(status, "read", name)
        return True


class InMemorySecretStore:
    """``SecretStore`` that keeps secrets in a dict.

    For tests and for the Linux CI path, where no keychain exists. Nothing is
    written to disk, so secrets die with the process.
    """

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._values: dict[str, str] = dict(initial or {})

    def get(self, name: str) -> str | None:
        _check_name(name)
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        _check_name(name)
        assert isinstance(value, str), "secret must be str"
        if not value:
            raise ValueError(f"refusing to store an empty secret for item {name!r}")
        self._values[name] = value

    def delete(self, name: str) -> None:
        _check_name(name)
        self._values.pop(name, None)

    def has(self, name: str) -> bool:
        _check_name(name)
        return name in self._values

    def __repr__(self) -> str:  # pragma: no cover - defensive, never shows values
        return f"InMemorySecretStore(items={sorted(self._values)!r})"


def _check_name(name: str) -> None:
    assert isinstance(name, str), "item name must be str"
    assert name, "item name is required"


# --------------------------------------------------------------------------
# backend 1: PyObjC (pyobjc-framework-Security), when it is installed
# --------------------------------------------------------------------------


class PyObjCBackend:
    """``SecItem*`` through the PyObjC ``Security`` bridge.

    The bridge takes and returns Foundation objects, so the query dictionaries
    are plain Python dicts keyed by the framework's CFString constants.
    """

    def __init__(self, security: Any = None) -> None:
        self._sec = security if security is not None else _import_security()

    def _base(self, service: str, account: str) -> dict:
        sec = self._sec
        return {
            sec.kSecClass: sec.kSecClassGenericPassword,
            sec.kSecAttrService: service,
            sec.kSecAttrAccount: account,
        }

    def add(self, service: str, account: str, secret: bytes) -> int:
        sec = self._sec
        query = self._base(service, account)
        query[sec.kSecValueData] = secret
        query[sec.kSecAttrAccessible] = sec.kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        status, _ref = sec.SecItemAdd(query, None)
        return int(status)

    def copy(self, service: str, account: str) -> tuple[int, bytes | None]:
        sec = self._sec
        query = self._base(service, account)
        query[sec.kSecReturnData] = True
        query[sec.kSecMatchLimit] = sec.kSecMatchLimitOne
        status, data = sec.SecItemCopyMatching(query, None)
        if int(status) != ERR_SEC_SUCCESS or data is None:
            return int(status), None
        return int(status), bytes(data)

    def update(self, service: str, account: str, secret: bytes) -> int:
        sec = self._sec
        status = sec.SecItemUpdate(self._base(service, account), {sec.kSecValueData: secret})
        return int(status)

    def delete(self, service: str, account: str) -> int:
        return int(self._sec.SecItemDelete(self._base(service, account)))


def _import_security() -> Any:
    try:
        import Security  # type: ignore[import-not-found]  # noqa: N813 - framework name
    except ImportError as error:
        raise KeychainUnavailable(f"pyobjc-framework-Security is not installed ({error})") from None
    return Security


# --------------------------------------------------------------------------
# backend 2: ctypes against Security.framework, no dependency
# --------------------------------------------------------------------------

_CF_PATH = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_SEC_PATH = "/System/Library/Frameworks/Security.framework/Security"
_UTF8 = 0x08000100  # kCFStringEncodingUTF8


class CTypesBackend:
    """``SecItem*`` bound directly, so no PyObjC package is required.

    Every CoreFoundation object created here is released on the way out,
    including the ``CFDataRef`` ``SecItemCopyMatching`` hands back.
    """

    def __init__(self, cf_path: str = _CF_PATH, sec_path: str = _SEC_PATH) -> None:
        import ctypes

        self._ctypes = ctypes
        try:
            self._cf = ctypes.CDLL(cf_path)
            self._sec = ctypes.CDLL(sec_path)
        except OSError as error:
            raise KeychainUnavailable(f"Security.framework did not load ({error})") from None
        self._constants: dict[str, Any] = {}
        self._bind()

    def _bind(self) -> None:
        c = self._ctypes
        void_p, long_, int32, char_p = c.c_void_p, c.c_long, c.c_int32, c.c_char_p
        cf, sec = self._cf, self._sec

        cf.CFStringCreateWithBytes.restype = void_p
        cf.CFStringCreateWithBytes.argtypes = [void_p, char_p, long_, c.c_uint32, c.c_bool]
        cf.CFDataCreate.restype = void_p
        cf.CFDataCreate.argtypes = [void_p, char_p, long_]
        cf.CFDictionaryCreateMutable.restype = void_p
        cf.CFDictionaryCreateMutable.argtypes = [void_p, long_, void_p, void_p]
        cf.CFDictionarySetValue.argtypes = [void_p, void_p, void_p]
        cf.CFRelease.argtypes = [void_p]
        cf.CFDataGetBytePtr.restype = void_p
        cf.CFDataGetBytePtr.argtypes = [void_p]
        cf.CFDataGetLength.restype = long_
        cf.CFDataGetLength.argtypes = [void_p]

        for name in ("SecItemAdd", "SecItemCopyMatching", "SecItemUpdate"):
            fn = getattr(sec, name)
            fn.restype = int32
            fn.argtypes = [void_p, void_p]
        sec.SecItemDelete.restype = int32
        sec.SecItemDelete.argtypes = [void_p]

        self._key_callbacks = void_p.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")
        self._value_callbacks = void_p.in_dll(cf, "kCFTypeDictionaryValueCallBacks")
        self._true = void_p.in_dll(cf, "kCFBooleanTrue").value

    def _const(self, name: str) -> Any:
        """A CFStringRef global of Security.framework, looked up once."""
        if name not in self._constants:
            try:
                self._constants[name] = self._ctypes.c_void_p.in_dll(self._sec, name).value
            except ValueError as error:
                raise KeychainUnavailable(f"Security.framework has no {name} ({error})") from None
        return self._constants[name]

    def _string(self, text: str) -> Any:
        raw = text.encode("utf-8")
        ref = self._cf.CFStringCreateWithBytes(None, raw, len(raw), _UTF8, False)
        assert ref, "CFStringCreateWithBytes returned NULL"
        return ref

    def _data(self, raw: bytes) -> Any:
        ref = self._cf.CFDataCreate(None, raw, len(raw))
        assert ref, "CFDataCreate returned NULL"
        return ref

    def _dict(self, pairs: list[tuple[Any, Any]]) -> Any:
        ref = self._cf.CFDictionaryCreateMutable(
            None,
            0,
            self._ctypes.byref(self._key_callbacks),
            self._ctypes.byref(self._value_callbacks),
        )
        assert ref, "CFDictionaryCreateMutable returned NULL"
        for key, value in pairs:
            self._cf.CFDictionarySetValue(ref, key, value)
        return ref

    def _release(self, *refs: Any) -> None:
        for ref in refs:
            if ref:
                self._cf.CFRelease(ref)

    def _identity(
        self, service: str, account: str, owned: list[Any]
    ) -> list[tuple[Any, Any]]:
        """Class/service/account pairs. Created refs are appended to ``owned``.

        The framework's own constants are never appended: they are immortal and
        must not be released.
        """
        service_ref = self._string(service)
        account_ref = self._string(account)
        owned.extend((service_ref, account_ref))
        return [
            (self._const("kSecClass"), self._const("kSecClassGenericPassword")),
            (self._const("kSecAttrService"), service_ref),
            (self._const("kSecAttrAccount"), account_ref),
        ]

    def add(self, service: str, account: str, secret: bytes) -> int:
        owned: list[Any] = []
        try:
            pairs = self._identity(service, account, owned)
            data_ref = self._data(secret)
            owned.append(data_ref)
            pairs.append((self._const("kSecValueData"), data_ref))
            pairs.append(
                (
                    self._const("kSecAttrAccessible"),
                    self._const("kSecAttrAccessibleWhenUnlockedThisDeviceOnly"),
                )
            )
            query = self._dict(pairs)
            owned.append(query)
            return int(self._sec.SecItemAdd(query, None))
        finally:
            self._release(*owned)

    def copy(self, service: str, account: str) -> tuple[int, bytes | None]:
        owned: list[Any] = []
        out = self._ctypes.c_void_p()
        try:
            pairs = self._identity(service, account, owned)
            pairs.append((self._const("kSecReturnData"), self._true))
            pairs.append((self._const("kSecMatchLimit"), self._const("kSecMatchLimitOne")))
            query = self._dict(pairs)
            owned.append(query)
            status = int(self._sec.SecItemCopyMatching(query, self._ctypes.byref(out)))
            if status != ERR_SEC_SUCCESS or not out.value:
                return status, None
            owned.append(out.value)
            length = self._cf.CFDataGetLength(out)
            pointer = self._cf.CFDataGetBytePtr(out)
            return status, self._ctypes.string_at(pointer, length)
        finally:
            self._release(*owned)

    def update(self, service: str, account: str, secret: bytes) -> int:
        owned: list[Any] = []
        try:
            query = self._dict(self._identity(service, account, owned))
            owned.append(query)
            data_ref = self._data(secret)
            owned.append(data_ref)
            changes = self._dict([(self._const("kSecValueData"), data_ref)])
            owned.append(changes)
            return int(self._sec.SecItemUpdate(query, changes))
        finally:
            self._release(*owned)

    def delete(self, service: str, account: str) -> int:
        owned: list[Any] = []
        try:
            query = self._dict(self._identity(service, account, owned))
            owned.append(query)
            return int(self._sec.SecItemDelete(query))
        finally:
            self._release(*owned)


def default_backend() -> Any:
    """PyObjC's ``Security`` when installed, else the ctypes binding.

    Raises :class:`KeychainUnavailable` when neither is reachable, which is the
    normal outcome off macOS.
    """
    try:
        return PyObjCBackend()
    except KeychainUnavailable:
        return CTypesBackend()


__all__ = [
    "ALL_ITEMS",
    "BYOK_ITEMS",
    "DEFAULT_SERVICE_NAME",
    "ERR_SEC_DUPLICATE_ITEM",
    "ERR_SEC_ITEM_NOT_FOUND",
    "ERR_SEC_SUCCESS",
    "ITEM_BYOK_MISTRAL",
    "ITEM_BYOK_OPENAI",
    "ITEM_LEASE",
    "CTypesBackend",
    "InMemorySecretStore",
    "KeychainError",
    "KeychainStore",
    "KeychainUnavailable",
    "PyObjCBackend",
    "default_backend",
]
