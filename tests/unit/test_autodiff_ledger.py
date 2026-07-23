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
    # matmul's transposed matmuls, native add/multiply tensor algebra, and
    # tanh/sigmoid's W5 closed forms, comparison-backed ReLU, and the shared
    # normalization-statistics formulas. Nothing else.
    assert native == {
        "add", "broadcast", "gelu", "mul", "matmul", "reduce", "silu",
        "softmax", "tanh", "sigmoid", "relu", "rmsnorm", "layer_norm",
        "all_reduce", "all_gather", "reduce_scatter",
    }, (
        f"native adjoint set drifted: {sorted(native)} — a buildAdjoint that "
        "emits a CustomAdjointCallOp is a Python round-trip, not native"
    )
    assert not ({"layer_norm", "rmsnorm", "relu"} & placeholder)
    assert {"layer_norm", "rmsnorm", "relu"} <= native
    assert {"all_reduce", "all_gather", "reduce_scatter"} <= native


def test_rows_are_consistent() -> None:
    rows = autodiff_ledger.collect_rows()
    assert rows, "ledger produced no rows"
    for r in rows:
        assert r.python_reference in {"yes", "no"}
        assert r.ir_adjoint in {"native", "placeholder", "mixed", "none"}
        # A row exists only if differentiable OR carrying an IR adjoint.
        assert r.python_reference == "yes" or r.ir_adjoint != "none"
        # Phase 4 (A2): the native backward rungs are now SOURCED from the
        # Device verification requires both an oracle proof and runtime binding.
        verified = set(r.bwd_device_verified_jit) | set(r.bwd_device_verified_abi)
        assert verified <= set(r.bwd_runtime_bound)
        assert verified <= set(r.bwd_oracle_proven)
        assert not (set(r.bwd_device_verified_jit) & set(r.bwd_device_verified_abi))
        for t in (*r.bwd_runtime_bound, *verified):
            assert t in autodiff_ledger._TARGETS
        evidence = set(r.build_evidence)
        if r.python_reference == "yes":
            assert "python_reference=python-unit-registry" in evidence
        if r.ir_adjoint != "none":
            assert "ir_adjoint=llvm23-core" in evidence
        if r.bwd_cpu_ir_oracle:
            assert "bwd_cpu_ir_oracle=llvm23-core" in evidence
        for target in verified:
            assert any(item.startswith(f"device[{target}=") for item in evidence)


def test_flash_attn_backward_is_jit_verified_on_exact_rocm_target() -> None:
    """Phase 4 (A2): the gfx1151 flash_attn backward (matrix row
    `rocm_flash_attn_bwd_compiled`, covering MHA + GQA/MQA) lights up the native
    backward rungs, sourced from the execution matrix — the ledger's first
    exact-target device-verified backward."""
    rows = {r.family: r for r in autodiff_ledger.collect_rows()}
    fa = rows.get("flash_attn")
    assert fa is not None, "no flash_attn ledger row"
    assert "rocm_gfx1151" in fa.bwd_runtime_bound
    assert "rocm_gfx1151" in fa.bwd_oracle_proven
    assert "rocm_gfx1151" in fa.bwd_device_verified_jit


def test_rocm_aliases_composition_and_residual_policy_are_structured() -> None:
    from tessera.compiler import execution_matrix as em

    for alias in ("multi_head_attention", "gqa_attention", "mqa_attention"):
        assert em.has_native_backward(alias, "rocm")
        policy = em.backward_residual_policy(alias, "rocm")
        assert policy is not None
        assert policy["policy"] == "recompute_all"
        assert policy["implementation"] == "dedicated"

    assert em.has_native_backward("matmul", "rocm")
    matmul = em.backward_residual_policy("matmul", "rocm_gfx1151")
    assert matmul is not None
    assert matmul["policy"] == "save_inputs"
    assert matmul["implementation"] == "composition"
    compositions = em.backward_compositions()
    assert len(compositions) == 1
    assert compositions[0].component_paths == ("rocm_compiled", "rocm_compiled")


def test_device_verified_leaders_are_derived_from_rows() -> None:
    rendered = autodiff_ledger.render_markdown()
    rows = autodiff_ledger.collect_rows()
    leaders = {
        r.family for r in rows
        if r.bwd_device_verified_jit or r.bwd_device_verified_abi
    }
    section = rendered.split("### Device-verified leaders", 1)[1].split(
        "## Ledger", 1)[0]
    assert leaders
    for family in leaders:
        assert f"`{family}`" in section


def test_verified_backward_rows_name_build_and_fixture() -> None:
    from tessera.compiler import execution_matrix

    for row in execution_matrix.backward_rows():
        if not row.device_proof:
            continue
        assert row.evidence_target
        assert row.numerical_fixture
        assert row.proof_build


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
    rows = {row.family: row for row in autodiff_ledger.collect_rows()}
    for fam in autodiff_ledger._BWD_IR_ORACLE_CPU:
        assert fam in rows and rows[fam].ir_adjoint == "native", (
            f"{fam!r} is marked bwd_cpu_ir_oracle but has no native IR adjoint")
    proven = {r.family for r in autodiff_ledger.collect_rows() if r.bwd_cpu_ir_oracle}
    assert proven == set(autodiff_ledger._BWD_IR_ORACLE_CPU)


def test_reduce_adjoint_classification_is_kind_aware() -> None:
    kinds = autodiff_ledger._ir_adjoint_kind_classes()["reduce"]
    assert kinds == {
        "native": frozenset({"sum", "mean"}),
        "placeholder": frozenset({"max", "min"}),
    }
    rows = {row.family: row for row in autodiff_ledger.collect_rows()}
    assert rows["sum"].ir_adjoint == "native"
    assert rows["mean"].ir_adjoint == "native"
    assert rows["reduce"].ir_adjoint == "mixed"
    for family in ("max", "min", "amax", "amin"):
        assert rows[family].ir_adjoint == "placeholder"


def test_registered_in_generated_docs() -> None:
    doc = generated_docs.get("autodiff_connection_ledger")
    assert doc.csv_path is not None
    assert doc.render_md().startswith("<!-- AUTO-GENERATED")
    # Canonical CSV round-trips through the renderer.
    assert "family,category,python_reference,ir_adjoint" in doc.render_csv()
