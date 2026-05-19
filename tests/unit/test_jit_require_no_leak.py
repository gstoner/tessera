"""Regression: ``tessera.require()`` is a true no-op outside an
explicit decoration / trace scope.

Findings audit (2026-05-19) flagged that the old implementation
unconditionally appended to a module-global list every time a
``@jit`` body fell back to eager Python — a quiet cross-call /
cross-test state leak.  The fix: ``require()`` only collects when
the caller has opened a :class:`collect_constraints` scope; the
module global is otherwise read-only.
"""

from __future__ import annotations

import threading

import pytest

from tessera.compiler.jit import (
    _ACTIVE_CONSTRAINTS,
    collect_constraints,
    require,
)
from tessera.compiler.constraints import Divisible


def test_require_outside_scope_is_a_no_op() -> None:
    """Calling ``require()`` outside any ``collect_constraints`` scope
    must not mutate the thread-local stack — that's the leak fix."""
    assert _ACTIVE_CONSTRAINTS.stack == []
    require(Divisible("K", 64))
    require(Divisible("M", 128))
    # No scope was open → no list was created and no constraints were
    # collected anywhere.
    assert _ACTIVE_CONSTRAINTS.stack == []


def test_require_inside_scope_collects_into_that_scope() -> None:
    with collect_constraints() as collected:
        require(Divisible("K", 64))
        require(Divisible("M", 128))
    assert len(collected) == 2
    assert collected[0].dim == "K"
    assert collected[1].dim == "M"
    # Scope exited cleanly — stack returns to empty.
    assert _ACTIVE_CONSTRAINTS.stack == []


def test_nested_scopes_collect_into_innermost() -> None:
    """Nesting works — each scope receives only its own
    ``require()`` calls."""
    with collect_constraints() as outer:
        require(Divisible("A", 1))
        with collect_constraints() as inner:
            require(Divisible("B", 2))
        require(Divisible("C", 3))
    assert [c.dim for c in outer] == ["A", "C"]
    assert [c.dim for c in inner] == ["B"]


def test_scope_pop_is_ordered() -> None:
    """The context manager raises if the stack was popped out of
    order — catches the failure mode where two unrelated scopes
    happen to share the thread-local stack."""
    cm = collect_constraints()
    cm.__enter__()
    try:
        # Corrupt the stack by popping it manually.
        _ACTIVE_CONSTRAINTS.stack.append([])
        with pytest.raises(AssertionError, match="popped out of order"):
            cm.__exit__(None, None, None)
        # Clean up so other tests see an empty stack.
        _ACTIVE_CONSTRAINTS.stack.clear()
    finally:
        # Best-effort cleanup on test failure paths.
        _ACTIVE_CONSTRAINTS.stack.clear()


def test_scope_is_thread_local() -> None:
    """A scope opened on one thread does NOT capture ``require()``
    calls from another thread — the previous global list would
    have leaked across threads on every call."""
    captured: dict[str, list] = {}

    def background() -> None:
        with collect_constraints() as scope:
            require(Divisible("BG", 8))
            captured["bg"] = list(scope)

    with collect_constraints() as main_scope:
        thread = threading.Thread(target=background)
        thread.start()
        thread.join()
        require(Divisible("MAIN", 4))

    # Background thread's scope only saw its own constraint.
    assert [c.dim for c in captured["bg"]] == ["BG"]
    # Main thread's scope only saw its own constraint — NOT the
    # background's, even though they ran concurrently.
    assert [c.dim for c in main_scope] == ["MAIN"]
