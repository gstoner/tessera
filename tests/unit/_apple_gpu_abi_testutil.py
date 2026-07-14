"""Shared Apple GPU C-ABI test helper.

``bind_or_skip`` binds a runtime symbol *by name* through the canonical
``APPLE_ABI`` registry (``_apple_gpu_dispatch``), so tests no longer hand-write
ctypes signatures at each call site — the registry is the single source of truth
and the ``test_apple_gpu_abi_registry`` guards keep it honest.

Use it in tests that **exercise** a symbol and should skip when no Apple GPU
runtime is present. Do NOT use it in tests that **assert a symbol's presence**
(those want a hard failure, not a skip) or that bind a symbol with a
deliberately non-registry signature.
"""
from __future__ import annotations

from typing import Callable

import pytest

from tessera._apple_gpu_dispatch import (
    apple_gpu_runtime,
    apple_gpu_skip_reason,
    bind_registered,
)


def bind_or_skip(symbol: str) -> Callable:
    """Return the bound Apple GPU runtime ``symbol`` (signature from
    ``APPLE_ABI``), or ``pytest.skip`` when the runtime dylib isn't available
    on this host."""
    if apple_gpu_runtime() is None:
        pytest.skip(apple_gpu_skip_reason() or "Apple GPU runtime unavailable")
    fn = bind_registered(symbol)
    if fn is None:  # runtime present but symbol didn't resolve
        pytest.skip(f"symbol {symbol!r} not resolvable in the Apple GPU runtime")
    return fn
