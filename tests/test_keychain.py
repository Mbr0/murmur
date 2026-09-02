"""Tests for services.keychain.

The Security backend is the seam: every test here drives :class:`FakeBackend`,
so nothing touches the real login keychain. The one exception is
:class:`RealKeychainSmokeTest`, which is skipped unless
``MURMUR_KEYCHAIN_SMOKE=1`` and which writes only under a throwaway service
name it deletes again.

Two properties get their own tests because they are the point of the module:
a secret never appears in an exception (message, args or chained context), and
the module emits no log records at all.
"""

import os
import unittest
from unittest.mock import patch

from services.keychain import (
    ALL_ITEMS,
    BYOK_ITEMS,
    DEFAULT_SERVICE_NAME,
    ERR_SEC_DECODE,
    ERR_SEC_DUPLICATE_ITEM,
    ERR_SEC_ITEM_NOT_FOUND,
    ERR_SEC_SUCCESS,
    ITEM_BYOK_MISTRAL,
    ITEM_BYOK_OPENAI,
    ITEM_LEASE,
    CTypesBackend,
    InMemorySecretStore,
    KeychainError,
    KeychainStore,
    KeychainUnavailable,
    PyObjCBackend,
)

SECRET = "sk-live-do-not-leak-me-4f3a9c"
OTHER_SECRET = "sk-live-second-value-77b1"
ERR_SEC_AUTH_FAILED = -25293


class FakeBackend:
    """Stand-in for the four SecItem functions, over a dict.

    ``fail`` maps an operation name to the OSStatus it should return instead of
    doing the work, which is how the error-mapping tests are driven.
    """

    def __init__(self, fail: dict[str, int] | None = None) -> None:
        self.items: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.fail: dict[str, int] = dict(fail or {})

    def _record(self, operation: str, service: str, account: str) -> int | None:
        self.calls.append((operation, service, account))
        return self.fail.get(operation)

    def add(self, service: str, account: str, secret: bytes) -> int:
        forced = self._record("add", service, account)
        if forced is not None:
            return forced
        if (service, account) in self.items:
            return ERR_SEC_DUPLICATE_ITEM
        self.items[(service, account)] = bytes(secret)
        return ERR_SEC_SUCCESS

    def copy(self, service: str, account: str) -> tuple[int, bytes | None]:
        forced = self._record("copy", service, account)
        if forced is not None:
            return forced, None
        raw = self.items.get((service, account))
        if raw is None:
            return ERR_SEC_ITEM_NOT_FOUND, None
        return ERR_SEC_SUCCESS, raw

    def update(self, service: str, account: str, secret: bytes) -> int:
        forced = self._record("update", service, account)
        if forced is not None:
            return forced
        if (service, account) not in self.items:
            return ERR_SEC_ITEM_NOT_FOUND
        self.items[(service, account)] = bytes(secret)
        return ERR_SEC_SUCCESS

    def delete(self, service: str, account: str) -> int:
        forced = self._record("delete", service, account)
        if forced is not None:
            return forced
        if self.items.pop((service, account), None) is None:
            return ERR_SEC_ITEM_NOT_FOUND
        return ERR_SEC_SUCCESS

    def operations(self) -> list[str]:
        return [operation for operation, _service, _account in self.calls]


class ItemNameTest(unittest.TestCase):
    def test_item_names_are_the_documented_constants(self):
        self.assertEqual("boske-lease", ITEM_LEASE)
        self.assertEqual("byok-mistral", ITEM_BYOK_MISTRAL)
        self.assertEqual("byok-openai", ITEM_BYOK_OPENAI)
        self.assertEqual("com.canopystudio.murmur", DEFAULT_SERVICE_NAME)

    def test_all_items_and_byok_map_agree(self):
        self.assertEqual((ITEM_LEASE, ITEM_BYOK_MISTRAL, ITEM_BYOK_OPENAI), ALL_ITEMS)
        self.assertEqual({"mistral": ITEM_BYOK_MISTRAL, "openai": ITEM_BYOK_OPENAI}, BYOK_ITEMS)


class KeychainStoreTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.store = KeychainStore(service_name="test.murmur", backend=self.backend)

    def test_set_then_get_round_trips(self):
        self.store.set(ITEM_BYOK_OPENAI, SECRET)
        self.assertEqual(SECRET, self.store.get(ITEM_BYOK_OPENAI))
        self.assertEqual(SECRET.encode("utf-8"), self.backend.items[("test.murmur", ITEM_BYOK_OPENAI)])

    def test_get_missing_item_is_none(self):
        self.assertIsNone(self.store.get(ITEM_LEASE))

    def test_set_twice_updates_through_the_duplicate_path(self):
        self.store.set(ITEM_LEASE, SECRET)
        self.store.set(ITEM_LEASE, OTHER_SECRET)
        self.assertEqual(OTHER_SECRET, self.store.get(ITEM_LEASE))
        self.assertEqual(["add", "add", "update", "copy"], self.backend.operations())

    def test_delete_removes_the_item_and_is_idempotent(self):
        self.store.set(ITEM_BYOK_MISTRAL, SECRET)
        self.store.delete(ITEM_BYOK_MISTRAL)
        self.assertIsNone(self.store.get(ITEM_BYOK_MISTRAL))
        self.store.delete(ITEM_BYOK_MISTRAL)  # no such item is not an error

    def test_has_reports_presence_without_returning_the_secret(self):
        self.assertFalse(self.store.has(ITEM_LEASE))
        self.store.set(ITEM_LEASE, SECRET)
        self.assertTrue(self.store.has(ITEM_LEASE))

    def test_items_are_scoped_to_the_service_name(self):
        other = KeychainStore(service_name="other.service", backend=self.backend)
        self.store.set(ITEM_LEASE, SECRET)
        self.assertIsNone(other.get(ITEM_LEASE))

    def test_empty_secret_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.set(ITEM_LEASE, "")
        self.assertEqual([], self.backend.operations())

    def test_backend_is_resolved_lazily(self):
        with patch("services.keychain.default_backend", side_effect=AssertionError("resolved too early")):
            store = KeychainStore()  # constructing must not touch the framework
            self.assertEqual(DEFAULT_SERVICE_NAME, store.service_name)


class ErrorMappingTest(unittest.TestCase):
    def _store(self, fail: dict[str, int]) -> tuple[KeychainStore, FakeBackend]:
        backend = FakeBackend(fail=fail)
        return KeychainStore(service_name="test.murmur", backend=backend), backend

    def test_read_failure_raises_with_the_status(self):
        store, _backend = self._store({"copy": ERR_SEC_AUTH_FAILED})
        with self.assertRaises(KeychainError) as caught:
            store.get(ITEM_LEASE)
        self.assertEqual(ERR_SEC_AUTH_FAILED, caught.exception.status)
        self.assertIn("read", str(caught.exception))
        self.assertIn(ITEM_LEASE, str(caught.exception))

    def test_write_failure_raises_with_the_status(self):
        store, _backend = self._store({"add": ERR_SEC_AUTH_FAILED})
        with self.assertRaises(KeychainError) as caught:
            store.set(ITEM_LEASE, SECRET)
        self.assertEqual(ERR_SEC_AUTH_FAILED, caught.exception.status)
        self.assertIn("errSecAuthFailed", str(caught.exception))

    def test_update_failure_after_duplicate_raises(self):
        store, backend = self._store({"update": ERR_SEC_AUTH_FAILED})
        store.set(ITEM_LEASE, SECRET)
        with self.assertRaises(KeychainError):
            store.set(ITEM_LEASE, OTHER_SECRET)
        self.assertIn("update", backend.operations())

    def test_delete_failure_other_than_not_found_raises(self):
        store, _backend = self._store({"delete": ERR_SEC_AUTH_FAILED})
        with self.assertRaises(KeychainError) as caught:
            store.delete(ITEM_LEASE)
        self.assertEqual(ERR_SEC_AUTH_FAILED, caught.exception.status)

    def test_has_failure_other_than_not_found_raises(self):
        store, _backend = self._store({"copy": ERR_SEC_AUTH_FAILED})
        with self.assertRaises(KeychainError):
            store.has(ITEM_LEASE)

    def test_non_utf8_payload_becomes_a_decode_error(self):
        backend = FakeBackend()
        backend.items[("test.murmur", ITEM_LEASE)] = b"\xff\xfe\x00binary"
        store = KeychainStore(service_name="test.murmur", backend=backend)
        with self.assertRaises(KeychainError) as caught:
            store.get(ITEM_LEASE)
        self.assertEqual(ERR_SEC_DECODE, caught.exception.status)
        # The offending bytes must not travel with the exception.
        self.assertNotIn("binary", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)

    def test_unavailable_backend_reports_the_reason_without_a_secret(self):
        error = KeychainUnavailable("Security.framework did not load")
        self.assertIn("unavailable", str(error))
        self.assertIsInstance(error, KeychainError)


def _exception_text(error: BaseException) -> str:
    """Everything a traceback could print: message, args and the whole chain."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        parts.append(repr(current.args))
        parts.append(repr(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts)


class SecretNeverLeaksTest(unittest.TestCase):
    """The secret must not reach an exception, a repr or a log record."""

    def test_failing_operations_never_name_the_secret(self):
        for operation in ("add", "update", "copy", "delete"):
            with self.subTest(operation=operation):
                backend = FakeBackend(fail={operation: ERR_SEC_AUTH_FAILED})
                if operation == "update":
                    # ``set`` only reaches update when add reports a duplicate.
                    backend.items[("test.murmur", ITEM_LEASE)] = b"previous"
                store = KeychainStore(service_name="test.murmur", backend=backend)
                with self.assertRaises(KeychainError) as caught:
                    if operation in ("add", "update"):
                        store.set(ITEM_LEASE, SECRET)
                    elif operation == "copy":
                        store.get(ITEM_LEASE)
                    else:
                        store.delete(ITEM_LEASE)
                self.assertNotIn(SECRET, _exception_text(caught.exception))

    def test_store_repr_never_names_the_secret(self):
        backend = FakeBackend()
        store = KeychainStore(service_name="test.murmur", backend=backend)
        store.set(ITEM_LEASE, SECRET)
        self.assertNotIn(SECRET, repr(store))
        self.assertNotIn(SECRET, repr(InMemorySecretStore({ITEM_LEASE: SECRET})))

    def test_the_module_writes_no_log_records(self):
        backend = FakeBackend()
        store = KeychainStore(service_name="test.murmur", backend=backend)
        with self.assertNoLogs(level="DEBUG"):
            store.set(ITEM_LEASE, SECRET)
            store.get(ITEM_LEASE)
            store.has(ITEM_LEASE)
            store.delete(ITEM_LEASE)
            store.get(ITEM_LEASE)


class InMemorySecretStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = InMemorySecretStore()

    def test_round_trip_and_delete(self):
        self.assertIsNone(self.store.get(ITEM_LEASE))
        self.store.set(ITEM_LEASE, SECRET)
        self.assertEqual(SECRET, self.store.get(ITEM_LEASE))
        self.assertTrue(self.store.has(ITEM_LEASE))
        self.store.delete(ITEM_LEASE)
        self.assertFalse(self.store.has(ITEM_LEASE))
        self.store.delete(ITEM_LEASE)  # idempotent, like the real store

    def test_initial_values_and_empty_secret(self):
        store = InMemorySecretStore({ITEM_BYOK_OPENAI: SECRET})
        self.assertEqual(SECRET, store.get(ITEM_BYOK_OPENAI))
        with self.assertRaises(ValueError):
            store.set(ITEM_BYOK_OPENAI, "")


class StubSecurity:
    """Enough of PyObjC's ``Security`` module to check the query building."""

    kSecClass = "class"
    kSecClassGenericPassword = "genp"
    kSecAttrService = "svce"
    kSecAttrAccount = "acct"
    kSecValueData = "v_Data"
    kSecReturnData = "r_Data"
    kSecMatchLimit = "m_Limit"
    kSecMatchLimitOne = "m_LimitOne"
    kSecAttrAccessible = "pdmn"
    kSecAttrAccessibleWhenUnlockedThisDeviceOnly = "cku"

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], bytes] = {}
        self.queries: list[dict] = []

    def _key(self, query: dict) -> tuple[str, str]:
        return query[self.kSecAttrService], query[self.kSecAttrAccount]

    def SecItemAdd(self, query, _out):  # noqa: N802 - framework name
        self.queries.append(query)
        if self._key(query) in self.items:
            return ERR_SEC_DUPLICATE_ITEM, None
        self.items[self._key(query)] = bytes(query[self.kSecValueData])
        return ERR_SEC_SUCCESS, None

    def SecItemCopyMatching(self, query, _out):  # noqa: N802 - framework name
        self.queries.append(query)
        raw = self.items.get(self._key(query))
        if raw is None:
            return ERR_SEC_ITEM_NOT_FOUND, None
        return ERR_SEC_SUCCESS, raw

    def SecItemUpdate(self, query, changes):  # noqa: N802 - framework name
        self.queries.append(query)
        if self._key(query) not in self.items:
            return ERR_SEC_ITEM_NOT_FOUND
        self.items[self._key(query)] = bytes(changes[self.kSecValueData])
        return ERR_SEC_SUCCESS

    def SecItemDelete(self, query):  # noqa: N802 - framework name
        self.queries.append(query)
        if self.items.pop(self._key(query), None) is None:
            return ERR_SEC_ITEM_NOT_FOUND
        return ERR_SEC_SUCCESS


class PyObjCBackendTest(unittest.TestCase):
    def setUp(self):
        self.security = StubSecurity()
        self.store = KeychainStore(
            service_name="test.murmur", backend=PyObjCBackend(security=self.security)
        )

    def test_round_trip_through_the_bridge(self):
        self.store.set(ITEM_LEASE, SECRET)
        self.assertEqual(SECRET, self.store.get(ITEM_LEASE))
        self.store.set(ITEM_LEASE, OTHER_SECRET)
        self.assertEqual(OTHER_SECRET, self.store.get(ITEM_LEASE))
        self.store.delete(ITEM_LEASE)
        self.assertIsNone(self.store.get(ITEM_LEASE))

    def test_add_pins_the_item_to_this_device_while_unlocked(self):
        self.store.set(ITEM_LEASE, SECRET)
        add_query = self.security.queries[0]
        self.assertEqual(StubSecurity.kSecClassGenericPassword, add_query[StubSecurity.kSecClass])
        self.assertEqual(
            StubSecurity.kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            add_query[StubSecurity.kSecAttrAccessible],
        )
        self.assertEqual(SECRET.encode("utf-8"), add_query[StubSecurity.kSecValueData])

    def test_missing_pyobjc_package_reports_unavailable(self):
        with patch("services.keychain._import_security", side_effect=KeychainUnavailable("absent")):
            with self.assertRaises(KeychainUnavailable):
                PyObjCBackend()


class CTypesBackendTest(unittest.TestCase):
    def test_missing_framework_reports_unavailable(self):
        with self.assertRaises(KeychainUnavailable):
            CTypesBackend(cf_path="/nonexistent/CoreFoundation", sec_path="/nonexistent/Security")


@unittest.skipUnless(
    os.environ.get("MURMUR_KEYCHAIN_SMOKE") == "1",
    "set MURMUR_KEYCHAIN_SMOKE=1 to exercise the real login keychain",
)
class RealKeychainSmokeTest(unittest.TestCase):
    """Proves the framework binding works. Off by default: it writes a keychain item."""

    SERVICE = "com.canopystudio.murmur.test-smoke"
    ITEM = "smoke-throwaway"

    def setUp(self):
        self.store = KeychainStore(service_name=self.SERVICE)
        self.addCleanup(self.store.delete, self.ITEM)
        self.store.delete(self.ITEM)

    def test_set_get_delete_against_the_real_keychain(self):
        self.assertIsNone(self.store.get(self.ITEM))
        self.store.set(self.ITEM, SECRET)
        self.assertEqual(SECRET, self.store.get(self.ITEM))
        self.assertTrue(self.store.has(self.ITEM))

        self.store.set(self.ITEM, OTHER_SECRET)  # overwrite through the update path
        self.assertEqual(OTHER_SECRET, self.store.get(self.ITEM))

        self.store.delete(self.ITEM)
        self.assertIsNone(self.store.get(self.ITEM))
        self.assertFalse(self.store.has(self.ITEM))


if __name__ == "__main__":
    unittest.main()
