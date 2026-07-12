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
    assert not (native & placeholder), "an op cannot be both native and placeholder"
    # The native set is EXACTLY the buildAdjoint bodies that emit real Graph IR:
    # matmul's transposed matmuls + tanh/sigmoid's W5 closed forms. Nothing else.
    # This pins the classifier against the regression where LayerNormOp /
    # SoftmaxOp (hand-written defs that emit a `CustomAdjointCallOp` placeholder)
    # were miscounted as native merely because they had explicit definitions.
    assert native == {"matmul", "tanh", "sigmoid"}, (
        f"native adjoint set drifted: {sorted(native)} — a buildAdjoint that "
        "emits a CustomAdjointCallOp is a Python round-trip, not native"
    )
    # LayerNorm/Softmax emit CustomAdjointCallOp → placeholder, keyed by the
    # runtime VJP string ("layer_norm"/"softmax") so they land on the primitive.
    assert {"layer_norm", "softmax", "gelu", "relu"} <= placeholder
    assert not ({"layer_norm", "softmax", "gelu", "relu"} & native)


def test_rows_are_consistent() -> None:
    rows = autodiff_ledger.collect_rows()
    assert rows, "ledger produced no rows"
    for r in rows:
        assert r.python_reference in {"yes", "no"}
        assert r.ir_adjoint in {"native", "placeholder", "none"}
        # A row exists only if differentiable OR carrying an IR adjoint.
        assert r.python_reference == "yes" or r.ir_adjoint != "none"
        # Phase 4 (A2): the native backward rungs are now SOURCED from the
        # runtime execution matrix. Invariants: hardware_proven ⊆ runtime_bound
        # (a device-proven backward is necessarily runtime-bound), and every
        # target is a tracked backend. (oracle_proven stays empty until a native
        # backward oracle fixture is wired — Phase 4 Inc 3.)
        assert set(r.bwd_hardware_proven) <= set(r.bwd_runtime_bound)
        for t in (*r.bwd_runtime_bound, *r.bwd_hardware_proven):
            assert t in autodiff_ledger._TARGETS
        assert not r.bwd_oracle_proven


def test_flash_attn_backward_is_hardware_proven_on_rocm() -> None:
    """Phase 4 (A2): the gfx1151 flash_attn backward (matrix row
    `rocm_flash_attn_bwd_compiled`, covering MHA + GQA/MQA) lights up the native
    backward rungs, sourced from the execution matrix — the ledger's first
    `hardware_proven` backward."""
    rows = {r.family: r for r in autodiff_ledger.collect_rows()}
    fa = rows.get("flash_attn")
    assert fa is not None, "no flash_attn ledger row"
    assert "rocm" in fa.bwd_runtime_bound
    assert "rocm" in fa.bwd_hardware_proven


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
