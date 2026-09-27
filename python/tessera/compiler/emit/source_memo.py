"""Memo of synthesized kernel source, invalidated by the code that produced it.

Sync ``AUTOTUNE-EMITTED-IDENTITY-2026-09-27`` (hot-path follow-up on PR #861).
Keying every compiled-artifact cache by the emitted source (so a changed
emitter compiles fresh and is stamped with what compiled) meant each launch of
an emitted lane re-ran its Python emitter just to learn the key: ~10 us per
``run()`` on the Mac for the ``mma.sync`` lanes, the same order as an sm_120
kernel (2-20 us). That cost landed in production dispatch and in every
end-to-end timing the arbiter races.

:func:`memoized` returns the value an emitter function produced for the same
arguments, as long as **the code that produced it is still the code a fresh
call would run**. The check is by object identity, per call:

* the emitter is looked up by *name* in its namespace at call time, never
  captured -- ``monkeypatch.setattr(module, "_synthesize_x", ...)``,
  assignment, or ``importlib.reload`` replace the function object, so the memo
  misses and the emitter runs again;
* so is every global it reaches by name, transitively: module-level functions
  (followed into their own modules), classes and constants it references, and
  a function reached as ``<tessera module>.<name>``. Rebinding any of them
  misses too, so patching a *helper* of an emitter is caught as well as
  patching the emitter.

What it does not see, stated so it is not read as covered: code reached
through an attribute of an object other than a ``tessera`` module (a method on
the region or on a class), and in-place mutation of a module-level container
(rebinding is seen; ``table[k] = v`` is not). An emitter whose reachable code
mentions ``environ`` or ``getenv`` is never memoized, because the environment
is an input the memo key does not hold.

Why a miss the memo cannot see is safe for Decision #11: the launch caches and
the arbiter's identity (``artifact_identity``) read the *same* memoized value,
so they cannot diverge -- the process runs the code it stamps. Only a stale
memo read by one side and a fresh emission by the other could stamp one
kernel's latency with another's name, and that pairing no longer exists.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable, Mapping

#: (namespace id, emitter name, args, kwargs) -> (snapshot, value)
_Snapshot = tuple[tuple[Any, str, Any], ...]
_MEMO: dict[tuple[Any, ...], tuple[_Snapshot, Any]] = {}
#: Bound so a caller that varies an argument without limit cannot grow it forever.
_MAX_ENTRIES = 4096
#: Names whose presence in reachable code means the result depends on the
#: process environment, which no memo key here holds.
_ENV_NAMES = frozenset({"environ", "getenv", "getenvb"})
_MISSING = object()


class _ReadsEnvironment(Exception):
    pass


class _ClassView:
    """``.get(name)`` on a class, through its MRO -- so a method patched on the
    class (or a subclass override) reads as a rebinding."""

    __slots__ = ("cls",)

    def __init__(self, cls: type) -> None:
        self.cls = cls

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self.cls, name, default)


def _code_names(code: types.CodeType) -> set[str]:
    names: set[str] = set()
    stack = [code]
    while stack:
        c = stack.pop()
        names.update(c.co_names)
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return names


def _snapshot(fn: Callable[..., Any], namespace: Any, name: str) -> _Snapshot:
    """Every ``(namespace, name, object)`` binding the emitter's result can
    depend on by name, starting with the emitter's own. Raises
    :class:`_ReadsEnvironment` when reachable code reads the environment."""
    entries: dict[tuple[int, str], tuple[Any, str, Any]] = {
        (id(namespace), name): (namespace, name, fn)}
    cls_view = namespace if isinstance(namespace, _ClassView) else None
    seen: set[int] = set()
    todo: list[Any] = [fn]

    def follow_module(mod: types.ModuleType, names: set[str]) -> None:
        # `<tessera module>.<name>` -- by a module global, or by a
        # function-local `from tessera... import name` (the module is then an
        # IMPORT_NAME constant and `name` an attribute read at call time).
        mod_ns = vars(mod)
        for attr in names:
            target = mod_ns.get(attr, _MISSING)
            if isinstance(target, types.FunctionType):
                entries[(id(mod_ns), attr)] = (mod_ns, attr, target)
                todo.append(target)

    while todo:
        f = todo.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        code = getattr(f, "__code__", None)
        glb = getattr(f, "__globals__", None)
        if code is None or glb is None:
            continue
        names = _code_names(code)
        if names & _ENV_NAMES:
            raise _ReadsEnvironment(getattr(f, "__qualname__", repr(f)))
        for n in names:
            if n.startswith("tessera"):
                imported = sys.modules.get(n)
                if isinstance(imported, types.ModuleType):
                    follow_module(imported, names)
            if cls_view is not None:
                # `self.<method>` on the emitter's own class.
                method = cls_view.get(n, _MISSING)
                if isinstance(method, types.FunctionType):
                    entries[(id(cls_view), n)] = (cls_view, n, method)
                    todo.append(method)
            obj = glb.get(n, _MISSING)
            if obj is _MISSING:
                continue
            if isinstance(obj, types.ModuleType):
                # Other modules (numpy, ctypes, ...) are not code this
                # checkout emits.
                if obj.__name__.startswith("tessera"):
                    follow_module(obj, names)
                continue
            entries[(id(glb), n)] = (glb, n, obj)
            if isinstance(obj, types.FunctionType):
                todo.append(obj)
    return tuple(entries.values())


def _still_current(snapshot: _Snapshot) -> bool:
    for ns, n, obj in snapshot:
        if ns.get(n, _MISSING) is not obj:
            return False
    return True


def _call(scope: Any, namespace: Any, name: str, fn: Callable[..., Any],
          call: Callable[[], Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        key = (scope, name, args, tuple(kwargs.items()))
        entry = _MEMO.get(key)
    except TypeError:                   # an unhashable argument: no memo
        return call()
    if entry is not None and _still_current(entry[0]):
        return entry[1]
    try:
        snapshot = _snapshot(fn, namespace, name)
    except _ReadsEnvironment:
        return call()
    value = call()
    # Store only if nothing it depends on was rebound while it ran; otherwise
    # which code produced `value` is unknown, and the next call re-emits.
    if _still_current(snapshot):
        if len(_MEMO) >= _MAX_ENTRIES:
            _MEMO.clear()
        _MEMO[key] = (snapshot, value)
    return value


def memoized(namespace: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    """``namespace[name](*args, **kwargs)`` (``namespace`` is a module's
    ``globals()``), served from the memo while the emitter and everything it
    reaches by name are the objects that produced the memoized value (module
    docstring). Arguments must be hashable to be memoized; an unhashable
    argument, or an emitter that reads the environment, is simply called every
    time. The value must be immutable (source text, a frozen ``KernelSource``):
    every caller shares it."""
    fn = namespace[name]
    return _call(id(namespace), namespace, name, fn,
                 lambda: fn(*args, **kwargs), args, kwargs)


def memoized_method(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """``obj.<name>(*args, **kwargs)`` for a *stateless* object (no instance
    attributes), memoized like :func:`memoized`: the method is resolved on the
    class at call time and its reachable globals are checked. An object that
    carries instance state is called every time -- that state is an input the
    key does not hold -- and so is one whose class sets ``source_memo_safe =
    False`` (it delegates to code reached through another object, which the
    name walk cannot see)."""
    cls = type(obj)
    fn = getattr(cls, name)
    if (getattr(obj, "__dict__", None) or not isinstance(fn, types.FunctionType)
            or not getattr(cls, "source_memo_safe", True)):
        return getattr(obj, name)(*args, **kwargs)
    return _call(obj, _ClassView(cls), name, fn,
                 lambda: fn(obj, *args, **kwargs), args, kwargs)


def clear() -> None:
    _MEMO.clear()


def size() -> int:
    return len(_MEMO)


__all__ = ["clear", "memoized", "memoized_method", "size"]
