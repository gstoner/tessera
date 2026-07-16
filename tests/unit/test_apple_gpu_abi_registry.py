"""Apple GPU C-ABI signature registry guards.

``_apple_gpu_dispatch.APPLE_ABI`` is the single source of truth for the Apple
GPU runtime C ABI — every exported symbol the Python side binds, mapped to its
canonical ctypes ``(argtypes, restype)``. Before it existed, ~19 test files and
8 production modules each hand-wrote the ctypes signature at the bind site, so a
renamed symbol or a changed signature was only ever caught on an actual Apple
GPU (every such test skips off-device). These guards close that gap:

* :func:`test_registry_wellformed` — the registry entries are structurally valid
  (runs anywhere).
* :func:`test_registry_covers_all_bind_symbol_call_sites` — **off-device drift
  net**: every literal ``bind_symbol("sym", argtypes, restype)`` in the tree must
  agree with the registry (symbol present + identical evaluated signature). This
  runs on ordinary CI with no GPU, so a signature that drifts from the registry —
  or a symbol bound but never registered — fails immediately.
* :func:`test_dylib_exports_resolve_every_registry_symbol` — **on-device ABI
  net**: when the runtime dylib is loaded, every registry symbol must resolve in
  it. Skips off-device (like the NVIDIA/ROCm runtime-symbol tests), but when it
  runs it covers the whole ABI in one place, catching a removed/renamed C export
  that no single feature test would notice.
"""
from __future__ import annotations

import ast
import ctypes
from pathlib import Path

import pytest

from tessera._apple_gpu_dispatch import (
    APPLE_ABI,
    apple_gpu_runtime,
    apple_gpu_skip_reason,
    bind_registered,
    expected_symbols,
)

_REPO = Path(__file__).resolve().parents[2]
_SCAN_ROOTS = ("python/tessera", "tests")


def _file_alias_ns(tree: ast.Module) -> dict:
    """Build an eval namespace for one file: ``ctypes`` plus every local
    ``name = <ctypes expr>`` alias defined anywhere in it (e.g. runtime.py's
    ``cf = ctypes.POINTER(ctypes.c_float)`` / ``ci = ctypes.POINTER(c_int64)``).
    Call sites bind through these aliases, so the guard must resolve them to
    compare signatures — assuming ``ctypes``-only would either miss the alias
    or, worse, guess the wrong pointee type."""
    ns: dict = {"ctypes": ctypes}
    pending = [(n.targets[0].id, ast.unparse(n.value))
               for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name)]
    for _ in range(3):  # a few passes so aliases-of-aliases resolve
        for name, expr in pending:
            if name in ns:
                continue
            try:
                ns[name] = eval(expr, dict(ns))  # noqa: S307 — trusted repo source
            except Exception:
                pass
    return ns


def _canon(argtypes_expr: str, restype_expr: str | None, ns: dict | None = None):
    """Evaluate an ``argtypes``/``restype`` source expression into a hashable
    canonical form: a tuple of ctypes type names + the restype name (or ``None``
    for void). ``ns`` supplies the file's aliases; defaults to ``ctypes`` only
    (used for the registry, whose entries are always explicit ctypes types)."""
    ns = ns or {"ctypes": ctypes}
    argtypes = eval(argtypes_expr, dict(ns))  # noqa: S307 — trusted repo source
    at = tuple(getattr(t, "__name__", repr(t)) for t in argtypes)
    if restype_expr in (None, "None"):
        return (at, None)
    rt = eval(restype_expr, dict(ns))  # noqa: S307
    return (at, None if rt is None else getattr(rt, "__name__", repr(rt)))


def _bind_symbol_call_sites():
    """Yield ``(symbol, argtypes_expr, restype_expr, relpath, lineno, ns)`` for
    every literal-symbol ``bind_symbol(...)`` call under the scanned roots, where
    ``ns`` is the file's alias namespace. Calls whose symbol is a
    variable/expression (dynamic dispatchers) are skipped — the registry is keyed
    by literal name."""
    for root in _SCAN_ROOTS:
        for path in (_REPO / root).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            ns = _file_alias_ns(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                if name != "bind_symbol":
                    continue
                args = node.args
                if not (args and isinstance(args[0], ast.Constant)
                        and isinstance(args[0].value, str)):
                    continue
                sym = args[0].value
                at = ast.unparse(args[1]) if len(args) > 1 else None
                rt = ast.unparse(args[2]) if len(args) > 2 else None
                for kw in node.keywords:
                    if kw.arg == "argtypes":
                        at = ast.unparse(kw.value)
                    elif kw.arg == "restype":
                        rt = ast.unparse(kw.value)
                if at is None:
                    continue
                yield sym, at, rt, str(path.relative_to(_REPO)), node.lineno, ns


# ── structural validity (runs anywhere) ──────────────────────────────────────

def test_registry_wellformed():
    assert APPLE_ABI, "registry is empty"
    assert set(expected_symbols()) == set(APPLE_ABI)
    for sym, entry in APPLE_ABI.items():
        assert isinstance(sym, str) and sym, sym
        assert isinstance(entry, tuple) and len(entry) == 2, sym
        argtypes, restype = entry
        assert isinstance(argtypes, tuple), f"{sym}: argtypes must be a tuple"
        # ctypes.sizeof raises TypeError for anything that isn't a ctypes type,
        # so it doubles as the "is a real ctypes type" check.
        for t in argtypes:
            ctypes.sizeof(t)
        if restype is not None:
            ctypes.sizeof(restype)


# ── off-device drift net (runs on ordinary CI, no GPU) ────────────────────────

def test_registry_covers_all_bind_symbol_call_sites():
    problems = []
    seen = set()
    for sym, at, rt, rel, ln, ns in _bind_symbol_call_sites():
        # skip this guard file's own examples if any appear as calls
        if rel.endswith("test_apple_gpu_abi_registry.py"):
            continue
        seen.add(sym)
        if sym not in APPLE_ABI:
            problems.append(f"{rel}:{ln} binds {sym!r} which is NOT in APPLE_ABI "
                            f"(register it in _apple_gpu_dispatch.APPLE_ABI)")
            continue
        reg_at, reg_rt = APPLE_ABI[sym]
        reg_canon = (tuple(getattr(t, "__name__", repr(t)) for t in reg_at),
                     None if reg_rt is None else getattr(reg_rt, "__name__", repr(reg_rt)))
        site_canon = _canon(at, rt, ns)
        if site_canon != reg_canon:
            problems.append(
                f"{rel}:{ln} binds {sym!r} with a signature that disagrees with "
                f"APPLE_ABI:\n     site     = {site_canon}\n     registry = {reg_canon}")
    assert not problems, "Apple GPU ABI drift:\n" + "\n".join(problems)


def test_registry_has_no_unused_entries_that_are_never_referenced():
    # Every registry symbol should be referenced somewhere (a call site or the
    # runtime string table). A symbol nothing binds is dead ABI surface worth a
    # look. We only assert the common case: it appears as a string in the tree.
    referenced = set()
    for root in _SCAN_ROOTS:
        for path in (_REPO / root).rglob("*.py"):
            try:
                txt = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for sym in APPLE_ABI:
                if sym in txt:
                    referenced.add(sym)
    orphaned = sorted(set(APPLE_ABI) - referenced)
    assert not orphaned, f"registry symbols referenced nowhere in-tree: {orphaned}"


# ── on-device ABI net (skips without the runtime dylib) ───────────────────────

@pytest.mark.hardware_apple_gpu
def test_dylib_exports_resolve_every_registry_symbol():
    unresolved = [sym for sym in expected_symbols() if bind_registered(sym) is None]
    assert not unresolved, (
        f"{len(unresolved)} registry symbol(s) do not resolve in the loaded "
        f"runtime dylib (renamed/removed C export?): {unresolved}\n"
        f"skip_reason={apple_gpu_skip_reason()}")
