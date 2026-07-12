"""The backend manifest never silently lags the runtime (canonical primitives).

manifest_runtime_reconciliation parses the runtime's op-name dispatch gates and
cross-references the manifest. The invariant this locks: every **canonical
primitive** the runtime dispatches to a native device lane is *declared* native
in the manifest. A new gap here means a real op gained a runtime lane but its
manifest row still reads reference/missing — declare it (+ a fixture), or the
enablement dashboards under-count native coverage.
"""
from __future__ import annotations

from tessera.compiler import manifest_runtime_reconciliation as R


def test_no_manifest_runtime_reconciliation_gaps():
    gaps = R.reconciliation_gaps()
    assert gaps == [], (
        "canonical primitives the runtime dispatches natively but the manifest "
        "does NOT declare native — declare them in backend_manifest (+ a fixture) "
        "or fix the runtime overclaim:\n  "
        + "\n  ".join(
            f"{g.op}@{g.target} (manifest={g.manifest_status or 'missing'})"
            for g in gaps))


def test_runtime_dispatch_map_is_parsed():
    d = R.runtime_dispatch_map()
    # The parse must find real op-name-gated lanes on both native-execution
    # targets, and known op-gated ops must appear (guards against the extractor
    # silently breaking). NB: primary lanes dispatched by compiler_path rather
    # than an op-name gate (matmul, the main flash_attn) are out of scope — the
    # audit reconciles the op-name-gated family, which is where the lag occurs.
    assert d["rocm"] and d["x86"]
    assert "softmax" in d["rocm"] and "softmax" in d["x86"]
    assert "gelu" in d["rocm"]


def test_numpy_aliases_are_dispatched_but_not_gaps():
    """A numpy-style alias (``divide`` for ``div``) is accepted by the runtime but
    is not a canonical primitive — it must be dispatched yet never a gap (its
    canonical form is what the manifest tracks)."""
    d = R.runtime_dispatch_map()
    assert "divide" in d["x86"] or "divide" in d["rocm"]
    gaps = {(g.op, g.target) for g in R.reconciliation_gaps()}
    for alias in ("divide", "multiply", "subtract", "swish"):
        assert (alias, "rocm") not in gaps and (alias, "x86") not in gaps


def test_missing_runtime_raises_not_silent(monkeypatch):
    """Decision #26: a missing runtime source raises, never reports zero lanes."""
    import pytest
    monkeypatch.setattr(R, "_RUNTIME", R._RUNTIME.with_name("nope.py"))
    with pytest.raises(R.ReconciliationError):
        R.runtime_dispatch_map()
