"""Bounded reuse of pure native MLIR replay passes for ROCm unary packages.

Only pass output is cached. Callers still compare the returned Tile text and
project every descriptor field on each invocation. The key binds exact input
IR, the pass, the tool and loaded libraries, and the complete environment.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from .scheduled_matmul import run_tessera_opt

_LIMIT_ENTRIES = 256
_LIMIT_BYTES = 16 * 1024 * 1024
_PASSES = {"--tessera-schedule-to-tile", "--canonicalize"}
_lock = threading.Lock()
_outputs: OrderedDict[str, tuple[str, int]] = OrderedDict()
_bytes = 0


def _identity(tool: Path) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    from .rocm_native import _tool_digest

    return str(tool.resolve()), _tool_digest(tool), tuple(sorted(os.environ.items()))


def clear() -> None:
    global _bytes
    with _lock:
        _outputs.clear()
        _bytes = 0


def run(tool: Path, source: str, option: str,
        *, execute: Callable[[Path, str, str], str] = run_tessera_opt) -> str:
    global _bytes
    if option not in _PASSES:
        raise ValueError("ROCm replay cache accepts only pure ancestry passes")
    before = _identity(tool)
    key = hashlib.sha256(repr((before, source, option, execute)).encode()).hexdigest()
    with _lock:
        cached = _outputs.get(key)
        if cached is not None:
            _outputs.move_to_end(key)
            return cached[0]
    # Exceptions never enter the cache. A tool or loader change during the
    # invocation also prevents retention of the output.
    result = execute(tool, source, option)
    size = len(result.encode())
    if size > _LIMIT_BYTES or _identity(tool) != before:
        return result
    with _lock:
        previous = _outputs.pop(key, None)
        if previous is not None:
            _bytes -= previous[1]
        _outputs[key] = result, size
        _bytes += size
        while len(_outputs) > _LIMIT_ENTRIES or _bytes > _LIMIT_BYTES:
            _, (_, removed) = _outputs.popitem(last=False)
            _bytes -= removed
    return result
