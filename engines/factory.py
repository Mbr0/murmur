#!/usr/bin/env python3
"""Build a configured :class:`~engines.base.Engine` from the app's config dict.

``engines.create_engine`` knows how to *instantiate* an engine; it does not
know what any of them need. whisper.cpp wants a path on disk, Murmur Cloud
wants a lease provider and an endpoint, own-key wants a Keychain item for the
provider the user picked. Somebody has to hold that knowledge, and if it lives
in ``murmur.py`` then ``murmur.py`` learns four engines' constructors — exactly
the engine-specific branching the folder rules forbid in app code.

So it lives here. :func:`build_engine` is the one place that maps config keys
onto constructor arguments, and the app calls it with the services it already
owns::

    engine = build_engine(
        "cloud",
        config=config,
        model_store=store,
        license_service=self.license,
        keychain=self.keychain,
    )

Everything it depends on is injected, so the whole module is testable with
fakes and imports nothing heavy: the concrete engine modules are still reached
through :func:`engines.create_engine`, which imports them lazily and honours
:func:`engines.register_engine` so a test never needs MLX or a binary on disk.

Config keys read here
---------------------

=====================  ====================================================
``model_id``           catalog id for a local engine; resolved through
                       :meth:`~engines.model_store.ModelStore.engine_model_path`
``cloud_base_url``     Boske proxy origin; defaults to
                       :data:`DEFAULT_CLOUD_BASE_URL`
``byok_provider``      own-key provider id (``mistral``, ``openai``)
``byok_model``         optional model override for that provider
=====================  ====================================================

``cloud_base_url`` is **not** in ``DEFAULT_CONFIG`` yet: the serial wiring step
adds it to :data:`services.persistence_service.DEFAULT_CONFIG` with
:data:`DEFAULT_CLOUD_BASE_URL` as its value. Until then — and whenever the key
is absent or blank — the default below is used, so an older config still
reaches the proxy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engines import create_engine
from engines.base import Engine

#: Boske proxy origin. **Placeholder — confirm with Boske** (decision D6);
#: ``https://`` is mandatory and enforced by the engines themselves, because
#: the lease travels to it as a bearer token.
DEFAULT_CLOUD_BASE_URL = "https://api.boske.app"

#: Config key holding the proxy origin. The same value is handed to
#: :class:`services.license_service.LicenseService`, so the two never disagree
#: about which host the lease belongs to.
CONFIG_CLOUD_BASE_URL = "cloud_base_url"

CONFIG_MODEL_ID = "model_id"
CONFIG_BYOK_PROVIDER = "byok_provider"
CONFIG_BYOK_MODEL = "byok_model"

#: Engine ids that transcribe on this Mac from a model on disk.
LOCAL_ENGINE_IDS = ("whispercpp", "voxtral_mlx")

#: Keychain item names for own-key credentials are ``byok-<provider id>``; the
#: Keychain store spells the same strings out as ``ITEM_BYOK_MISTRAL`` and
#: ``ITEM_BYOK_OPENAI``. Kept as a prefix here rather than imported, because
#: this module must not depend on a macOS-only module to build a local engine.
BYOK_ITEM_PREFIX = "byok-"


def byok_item_name(provider_id: str) -> str:
    """Keychain item name holding the API key for ``provider_id``.

    Must agree with ``services.keychain.ITEM_BYOK_*``; a mismatch reads as "no
    key stored" and silently drops the user back to the local engine.
    """
    cleaned = str(provider_id or "").strip().lower()
    assert cleaned, "provider_id is required"
    return f"{BYOK_ITEM_PREFIX}{cleaned}"


def cloud_base_url(config: dict[str, Any] | None) -> str:
    """The proxy origin from config, or the placeholder default."""
    raw = (config or {}).get(CONFIG_CLOUD_BASE_URL)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return DEFAULT_CLOUD_BASE_URL


def _required(config: dict[str, Any], key: str, engine_id: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"engine {engine_id!r} needs config[{key!r}]")
    return value.strip()


def _optional_kwargs(**pairs: Any) -> dict[str, Any]:
    """Drop the ``None`` values, so each engine keeps its own default."""
    return {name: value for name, value in pairs.items() if value is not None}


def build_engine(
    engine_id: str,
    *,
    config: dict[str, Any],
    model_store: Any,
    license_service: Any = None,
    keychain: Any = None,
    http_open: Callable | None = None,
) -> Engine:
    """Construct the engine ``engine_id`` wants, from config and services.

    The engine is returned **unloaded**: the caller decides when to pay for
    ``load()``, which for a local engine means spawning a server or reading
    gigabytes of weights.

    ``license_service`` is required for ``cloud`` and ``keychain`` for
    ``byok``; both are ignored otherwise, so the app may pass whatever it has.
    ``http_open`` is forwarded to the two network engines only — whisper.cpp
    also takes one, but its transport talks to a local ``whisper-server``, and
    handing it a caller's cloud transport would be a category error.

    Raises :class:`ValueError` for an unknown id, for a missing config key, and
    for a missing service the chosen engine cannot work without.
    """
    assert engine_id, "engine_id is required"
    assert config is not None, "config is required"

    if engine_id in LOCAL_ENGINE_IDS:
        assert model_store is not None, "model_store is required for a local engine"
        model_id = _required(config, CONFIG_MODEL_ID, engine_id)
        return create_engine(engine_id, model_path=model_store.engine_model_path(model_id))

    if engine_id == "cloud":
        if license_service is None:
            raise ValueError("engine 'cloud' needs a license_service to supply the lease")
        return create_engine(
            "cloud",
            base_url=cloud_base_url(config),
            # A callable, not a token: the lease is read at the moment it is
            # needed, so a sign-out or a refresh takes effect on the next
            # dictation without rebuilding the engine.
            lease_provider=lambda: license_service.current_lease_token(),
            **_optional_kwargs(http_open=http_open),
        )

    if engine_id == "byok":
        if keychain is None:
            raise ValueError("engine 'byok' needs a keychain to read the API key")
        provider = _required(config, CONFIG_BYOK_PROVIDER, engine_id)
        item = byok_item_name(provider)
        model = config.get(CONFIG_BYOK_MODEL)
        return create_engine(
            "byok",
            provider=provider,
            key_provider=lambda: keychain.get(item),
            **_optional_kwargs(
                model=model if isinstance(model, str) and model.strip() else None,
                http_open=http_open,
            ),
        )

    raise ValueError(f"Unknown engine id {engine_id!r}")


__all__ = [
    "BYOK_ITEM_PREFIX",
    "CONFIG_BYOK_MODEL",
    "CONFIG_BYOK_PROVIDER",
    "CONFIG_CLOUD_BASE_URL",
    "CONFIG_MODEL_ID",
    "DEFAULT_CLOUD_BASE_URL",
    "LOCAL_ENGINE_IDS",
    "build_engine",
    "byok_item_name",
    "cloud_base_url",
]
