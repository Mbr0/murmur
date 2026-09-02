#!/usr/bin/env python3
"""Engine package: the shared contract plus a lazy engine registry.

Concrete engines are never imported at package import time; they may need MLX
wheels or bundled binaries that are absent on a given machine.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from engines.base import (
    LANGUAGE_AUTO,
    Engine,
    EngineCapabilityError,
    EngineError,
    EngineInfo,
    EngineNotLoadedError,
    EngineUnavailableError,
    Hints,
    Partial,
    Segment,
    Transcript,
)

#: Engine ids that resolve to a module ``engines.<id>`` exposing ``ENGINE_CLASS``.
ENGINE_IDS = ("whispercpp", "voxtral_mlx", "whisper_openai")

_REGISTERED: dict[str, Callable[..., Engine]] = {}


def register_engine(engine_id: str, factory: Callable[..., Engine]) -> None:
    """Register a factory for ``engine_id``. Tests only; overrides the lazy import."""
    assert engine_id, "engine_id is required"
    assert factory is not None, "factory is required"
    _REGISTERED[engine_id] = factory


def unregister_engine(engine_id: str) -> None:
    """Remove a registered factory. Tests only; no-op when absent."""
    _REGISTERED.pop(engine_id, None)


def create_engine(engine_id: str, **kwargs) -> Engine:
    """Instantiate an engine by id, importing its module only when asked."""
    assert engine_id, "engine_id is required"
    factory = _REGISTERED.get(engine_id)
    if factory is not None:
        return factory(**kwargs)

    if engine_id not in ENGINE_IDS:
        raise ValueError(f"Unknown engine id {engine_id!r}; known ids: {', '.join(ENGINE_IDS)}")

    module_name = f"engines.{engine_id}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EngineUnavailableError(f"{module_name} cannot be imported on this machine: {exc}") from exc

    engine_class = getattr(module, "ENGINE_CLASS", None)
    if engine_class is None:
        raise EngineUnavailableError(f"{module_name} does not define ENGINE_CLASS")
    return engine_class(**kwargs)


__all__ = [
    "ENGINE_IDS",
    "LANGUAGE_AUTO",
    "Engine",
    "EngineCapabilityError",
    "EngineError",
    "EngineInfo",
    "EngineNotLoadedError",
    "EngineUnavailableError",
    "Hints",
    "Partial",
    "Segment",
    "Transcript",
    "create_engine",
    "register_engine",
    "unregister_engine",
]
