"""End-to-end operator datatype-flow registry and generated-doc gates."""

from __future__ import annotations

from tessera.dtype import canonical_dtypes, planned_gated_dtypes
from tessera.compiler import audit, generated_docs
from tessera.compiler.dtype_flow_audit import (
    CSV_COLUMNS,
    REPORT_TARGETS,
    collect_dtype_flow_rows,
    render_csv,
    render_markdown,
)
from tessera.compiler.op_catalog import OP_SPECS
from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.compiler.tsol_coverage import all_tsol_op_names


def _row(op: str, dtype: str):
    return next(
        row for row in collect_dtype_flow_rows()
        if row.op_name == op and row.dtype == dtype
    )


def test_every_audited_operator_and_tsol_op_is_present() -> None:
    rows = collect_dtype_flow_rows()
    names = {row.op_name for row in rows}
    assert names == {row.op_name for row in audit.all_support_rows()}
    assert set(all_tsol_op_names()) <= names


def test_every_dtype_applicable_catalog_op_has_a_declared_storage() -> None:
    rows = collect_dtype_flow_rows()
    by_op = {name: [] for name in OP_SPECS}
    for row in rows:
        if row.op_name in by_op and row.dtype:
            by_op[row.op_name].append(row.dtype)
    coverage = all_primitive_coverages()
    missing = {
        name for name, dtypes in by_op.items()
        if (
            coverage[name].contract_status.get("dtype_layout_rule")
            != "not_applicable"
            and not dtypes
        )
    }
    assert missing == set()


def test_all_storage_and_accumulator_names_are_canonical_or_gated() -> None:
    allowed = canonical_dtypes() | planned_gated_dtypes()
    for row in collect_dtype_flow_rows():
        assert not row.dtype or row.dtype in allowed
        assert set(row.accumulators) <= allowed
        assert set(row.targets) == set(REPORT_TARGETS)


def test_matrix_accumulators_are_dtype_specific_not_policy_fallbacks() -> None:
    assert _row("matmul", "int8").accumulators == ("int32",)
    assert _row("matmul", "int4").accumulators == ("int32",)
    assert _row("matmul", "fp64").accumulators == ("fp64",)
    assert "fp32" in _row("matmul", "bf16").accumulators
    assert "fp32" in _row("matmul", "fp8_e4m3").accumulators


def test_amd_isa_legality_does_not_promote_runtime_support() -> None:
    assert _row("matmul", "fp8_e4m3").targets["rocm_gfx1151"].status == "unsupported"
    assert _row("matmul", "fp8_e4m3").targets["rocm_gfx1200"].status == "artifact_only"
    assert _row("matmul", "fp8_e4m3").targets["rocm_gfx1250"].status == "artifact_only"
    assert _row("matmul", "fp4_e2m1").targets["rocm_gfx1250"].status == "planned_gated"
    assert _row("matmul", "fp64").targets["rocm_gfx1251"].status == "planned_gated"
    assert _row("matmul", "int4").targets["rocm_gfx1151"].status == "device_verified_abi"


def test_derived_target_dtype_legality_does_not_claim_a_physical_kernel() -> None:
    derived = [
        state
        for row in collect_dtype_flow_rows()
        for state in row.targets.values()
        if state.source.endswith(":derived_legality")
    ]
    assert derived
    assert {state.status for state in derived} == {"legal_only"}


def test_tsol_reduced_storage_spectral_paths_remain_visible() -> None:
    for op in ("dct", "stft", "istft", "spectral_conv"):
        for dtype in ("fp16", "bf16"):
            row = _row(op, dtype)
            assert row.tsol
            assert row.targets["x86"].status == "device_verified_jit"
            assert row.targets["rocm_gfx1151"].status == "device_verified_jit"
    filter_dtypes = {
        row.dtype for row in collect_dtype_flow_rows()
        if row.op_name == "spectral_filter"
    }
    assert filter_dtypes == {"fp32", "complex64"}
    assert _row("spectral_filter", "complex64").physical_storage == "fp32x2_interleaved"
    assert _row("spectral_filter", "complex64").targets["x86"].status == "device_verified_jit"
    assert _row("fft", "complex64").targets["rocm_gfx1151"].status == "device_verified_jit"


def test_generated_csv_and_markdown_expose_each_layer_and_target() -> None:
    csv_text = render_csv()
    assert csv_text.splitlines()[0].split(",") == list(CSV_COLUMNS)
    assert "matmul,tessera.matmul" in csv_text
    markdown = render_markdown()
    assert "# Operator datatype flow" in markdown
    assert "## TSOL dtype flow" in markdown
    assert "## All-operator gap queue" in markdown
    assert "gfx1151" in markdown and "gfx1200" in markdown and "gfx1250" in markdown


def test_dtype_flow_is_registered_with_generated_docs() -> None:
    doc = generated_docs.get("dtype_flow")
    assert doc.csv_path is not None
    assert doc.also_gate_md
