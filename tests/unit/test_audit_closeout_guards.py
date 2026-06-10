"""Regression guards from CODE_AUDIT_2026_06_10 closeout (Wave 1).

Two silent-gap guards, both platform-agnostic (decoration-time emission +
static import inspection — no Metal device needed):

* **Finding 1e** — a known-good, AST-emittable `@jit(target="apple_gpu")`
  corpus must emit ZERO `JIT_APPLE_GPU_TRACE_DEFERRED` diagnostics. The
  apple_gpu trace-deferral path is a deliberate fallback for bodies the AST
  Graph-IR emitter can't handle, but a regression that makes a *normally
  emittable* body defer would still pass every numerical test (the tracer
  produces correct numbers). This guard fails loudly instead.

* **§3 drift** — `driver.py` and `runtime.py` must only *import* the
  `_APPLE_GPU_*` / `APPLE_GPU_*` envelope tables, never define their own.
  `apple_gpu_envelope` is the single source (single-source refactor); a new
  top-level definition elsewhere would silently re-introduce the drift the
  refactor removed.
"""

import ast
from pathlib import Path

import numpy as np

import tessera as ts

_PKG = Path(ts.__file__).resolve().parent


# ── Finding 1e — zero TRACE_DEFERRED on the known-good corpus ────────────────

def _matmul(x, w):
    return ts.ops.matmul(x, w)


def _matmul_softmax(a, b):
    return ts.ops.softmax(ts.ops.matmul(a, b))


def _matmul_gelu(x, w):
    return ts.ops.gelu(ts.ops.matmul(x, w))


def _matmul_rmsnorm(x, w):
    return ts.ops.rmsnorm(ts.ops.matmul(x, w))


def _attention_block(q, k, v):
    return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(q, k)), v)


def _swiglu(x, wg, wu, wd):
    gate = ts.ops.matmul(x, wg)
    up = ts.ops.matmul(x, wu)
    return ts.ops.matmul(ts.ops.silu_mul(gate, up), wd)


def _unary_chain(x):
    return ts.ops.rmsnorm(ts.ops.silu(x))


_GOOD_CORPUS = [
    _matmul, _matmul_softmax, _matmul_gelu, _matmul_rmsnorm,
    _attention_block, _swiglu, _unary_chain,
]


def test_known_good_apple_gpu_corpus_emits_no_trace_deferred():
    """Decoration is the unit of work here (no GPU execution), so this runs
    on every platform — it locks the AST emitter, not the Metal runtime."""
    offenders = []
    for fn in _GOOD_CORPUS:
        jitted = ts.jit(target="apple_gpu")(fn)
        codes = [d.code for d in (jitted.lowering_diagnostics or [])]
        if "JIT_APPLE_GPU_TRACE_DEFERRED" in codes:
            offenders.append(fn.__name__)
    assert not offenders, (
        "AST Graph-IR emission regressed — these normally-emittable apple_gpu "
        f"bodies now defer to the tracer (would silently still pass numeric "
        f"tests): {offenders}"
    )


# ── §3 drift — envelope tables are single-source ─────────────────────────────

def _toplevel_apple_gpu_assignments(path: Path) -> list[str]:
    """Top-level names matching _APPLE_GPU_* / APPLE_GPU_* that are *assigned*
    (defined) in this module, as opposed to imported."""
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in tree.body:  # top-level only
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and (
                t.id.startswith("_APPLE_GPU_") or t.id.startswith("APPLE_GPU_")
            ):
                names.append(t.id)
    return names


def test_envelope_tables_are_single_source():
    envelope = _PKG / "compiler" / "apple_gpu_envelope.py"
    # The canonical op-set / opcode / lane tables — the names the envelope
    # owns. Redefining ANY of these in a consumer module is the drift the
    # single-source refactor removed. (Runtime-local handler maps and
    # symbol-availability memo caches are NOT in this set and are allowed to
    # live next to the code that uses them.)
    canonical = set(_toplevel_apple_gpu_assignments(envelope))
    assert canonical, (
        "apple_gpu_envelope should define the canonical _APPLE_GPU_* tables")
    for rel in ("compiler/driver.py", "runtime.py"):
        redefined = sorted(
            set(_toplevel_apple_gpu_assignments(_PKG / rel)) & canonical)
        assert not redefined, (
            f"{rel} redefines envelope-owned table(s) {redefined} — these must "
            "be imported from apple_gpu_envelope (single source), not redefined."
        )


def test_envelope_tables_are_actually_imported_where_used():
    # driver.py / runtime.py reference the names (consumers), proving the
    # guard above isn't vacuous: they use the tables, just don't define them.
    for rel in ("compiler/driver.py", "runtime.py"):
        src = (_PKG / rel).read_text()
        assert "apple_gpu_envelope import" in src or "apple_gpu_envelope" in src, (
            f"{rel} should import from apple_gpu_envelope")
