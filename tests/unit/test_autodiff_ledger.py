"""Guards for the compiler-autodiff connection ledger (Phase 0).

The ledger is a *projection* over existing registries, not a new source of
truth (Decision #24). These tests lock that contract: its `python_reference`
column may not diverge from `primitive_coverage`, and its `ir_adjoint` column
must reflect the real native-vs-placeholder split in `AdjointInterface.cpp`.
"""

from __future__ import annotations

import pytest

from tessera.compiler import autodiff_ledger, generated_docs, primitive_coverage


def _coverage_diff_families() -> frozenset[str]:
    """Families primitive_coverage marks as having a complete VJP or JVP."""
    out = set()
    for name, cov in primitive_coverage.all_primitive_coverages().items():
        cs = cov.contract_status
        if cs.get("vjp") == "complete" or cs.get("jvp") == "complete":
            out.add(name)
    return frozenset(out)


def test_python_reference_reconciles_with_primitive_coverage() -> None:
    """The ledger must not invent or drop a differentiable family relative to
    the Decision #24 truth source — the whole point of it being a projection."""
    assert autodiff_ledger.python_reference_families() == _coverage_diff_families()


def test_native_and_placeholder_adjoints_are_disjoint_and_grounded() -> None:
    native, placeholder = autodiff_ledger._ir_adjoint_classes()
    # tanh/sigmoid are the W5 native static-shape adjoints; they must never be
    # miscounted as placeholder round-trips.
    assert {"tanh", "sigmoid"} <= native
    assert not (native & placeholder), "an op cannot be both native and placeholder"
    # The known Python-round-trip ops are placeholder, not native.
    assert {"gelu", "relu"} <= placeholder
    assert not ({"gelu", "relu"} & native)


def test_rows_are_consistent() -> None:
    rows = autodiff_ledger.collect_rows()
    assert rows, "ledger produced no rows"
    for r in rows:
        assert r.python_reference in {"yes", "no"}
        assert r.ir_adjoint in {"native", "placeholder", "none"}
        # A row exists only if differentiable OR carrying an IR adjoint.
        assert r.python_reference == "yes" or r.ir_adjoint != "none"
        # Backward-execution rungs are empty until Phases 3-4 wire them; a
        # non-empty one here means a real source appeared — update the ledger
        # provenance (and this guard) deliberately, don't let it slip in.
        assert not r.bwd_runtime_bound
        assert not r.bwd_oracle_proven
        assert not r.bwd_hardware_proven


def test_missing_source_raises_not_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision #26 / error-handling: a missing C++ source must raise a named
    error, never silently report zero IR adjoints."""
    monkeypatch.setattr(autodiff_ledger, "_ADJOINT_CPP",
                        autodiff_ledger._ADJOINT_CPP.with_name("does_not_exist.cpp"))
    with pytest.raises(autodiff_ledger.LedgerError):
        autodiff_ledger._ir_adjoint_classes()


def test_cpu_ir_oracle_families_are_proven_backward_capable() -> None:
    """Every family the ledger marks `bwd_cpu_ir_oracle` must actually have an
    IR adjoint (native or placeholder) — a family with no adjoint could not have
    an oracle-verified backward. Guards against the set drifting to a claim the
    pass can't back (the numerical proof itself is in
    test_autodiff_paired_cpu_oracle.py)."""
    native, placeholder = autodiff_ledger._ir_adjoint_classes()
    have_adjoint = native | placeholder
    for fam in autodiff_ledger._BWD_IR_ORACLE_CPU:
        assert fam in have_adjoint, (
            f"{fam!r} is marked bwd_cpu_ir_oracle but has no IR adjoint")
    proven = {r.family for r in autodiff_ledger.collect_rows() if r.bwd_cpu_ir_oracle}
    assert proven == set(autodiff_ledger._BWD_IR_ORACLE_CPU)


def test_registered_in_generated_docs() -> None:
    doc = generated_docs.get("autodiff_connection_ledger")
    assert doc.csv_path is not None
    assert doc.render_md().startswith("<!-- AUTO-GENERATED")
    # Canonical CSV round-trips through the renderer.
    assert "family,category,python_reference,ir_adjoint" in doc.render_csv()
