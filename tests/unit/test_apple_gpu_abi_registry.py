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
                ns[name] = _safe_eval(expr, ns)
            except Exception:
                pass
    return ns


# ── The guard may parse repo source, but must never RUN it ───────────────────
# These evaluate expressions lifted out of the tree. One of them is
# `ctypes.CDLL('libamdhip64.so', mode=ctypes.RTLD_LOCAL)`, and evaluating it
# dlopens the HIP runtime as a side effect of a STATIC ABI check -- leaving it
# loaded, outside its normal initialization path, in the same process. On
# Princess-Luna that made a later live ROCm launch fail with rc 3 (a HIP
# allocation error) in the full sweep while passing in isolation, which read as
# a flaky ROCm test for as long as nobody ran the two files together.
#
# So evaluation is restricted to what this guard actually needs: ctypes TYPE
# expressions. `ctypes.c_float`, `ctypes.POINTER(...)`, a list or tuple of
# those, and names already resolved in this file's alias namespace. A call to
# anything else -- `CDLL`, `cdll.LoadLibrary`, an arbitrary function -- is
# refused before it runs, and the alias is simply left unresolved, exactly as
# it already was for every expression that failed to evaluate.
_CTYPES_TYPE_CALLS = frozenset({"POINTER", "CFUNCTYPE", "ARRAY"})


def _is_pure_ctypes_type_expr(node: ast.AST, ns: dict) -> bool:
    """Whether `node` builds a ctypes type with no side effects."""
    if isinstance(node, ast.Attribute):
        # ctypes.c_float / ctypes.RTLD_LOCAL — attribute reads are inert.
        return isinstance(node.value, ast.Name) and node.value.id in ns
    if isinstance(node, ast.Name):
        return node.id in ns
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_pure_ctypes_type_expr(e, ns) for e in node.elts)
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add)):
        # `ctypes.c_char * 8` is the ctypes array spelling, and real argtypes
        # are built with tuple arithmetic:
        #   (ctypes.c_void_p,) * 5 + (ctypes.c_int32,) * 14
        return (_is_pure_ctypes_type_expr(node.left, ns)
                and _is_pure_ctypes_type_expr(node.right, ns))
    if isinstance(node, ast.Call):
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _CTYPES_TYPE_CALLS):
            return False
        if not (isinstance(func.value, ast.Name) and func.value.id in ns):
            return False
        return all(_is_pure_ctypes_type_expr(a, ns) for a in node.args)
    return False


def _safe_eval(expr: str, ns: dict):
    """`eval` restricted to pure ctypes type expressions; raises otherwise."""
    node = ast.parse(expr, mode="eval").body
    if not _is_pure_ctypes_type_expr(node, ns):
        raise ValueError(f"refusing to execute non-type expression: {expr}")
    return eval(compile(ast.Expression(node), "<abi-guard>", "eval"), dict(ns))  # noqa: S307


def _canon(argtypes_expr: str, restype_expr: str | None, ns: dict | None = None):
    """Evaluate an ``argtypes``/``restype`` source expression into a hashable
    canonical form: a tuple of ctypes type names + the restype name (or ``None``
    for void). ``ns`` supplies the file's aliases; defaults to ``ctypes`` only
    (used for the registry, whose entries are always explicit ctypes types)."""
    ns = ns or {"ctypes": ctypes}
    argtypes = _safe_eval(argtypes_expr, ns)
    at = tuple(getattr(t, "__name__", repr(t)) for t in argtypes)
    if restype_expr in (None, "None"):
        return (at, None)
    rt = _safe_eval(restype_expr, ns)
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
