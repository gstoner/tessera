from __future__ import annotations

import re
from pathlib import Path

from tessera.compiler.capabilities import TARGET_CAPABILITIES
from tessera.compiler.frontend.parser import lower_text_to_graph_ir
from tessera.compiler.graph_ir import _parse_mlir_tensor_type, tensor_ir_type
from tessera.compiler.op_catalog import OP_SPECS
from tessera.compiler.primitive_coverage import all_primitive_coverages


ROOT = Path(__file__).resolve().parents[2]

CANONICAL_DTYPES = {
    "fp64",
    "fp32",
    "fp16",
    "bf16",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp6_e2m3",
    "fp6_e3m2",
    "fp4_e2m1",
    "nvfp4",
    "int8",
    "int16",
    "int32",
    "int64",
    "bool",
}

ALIASES = {
    "f64": "fp64",
    "f32": "fp32",
    "f16": "fp16",
}

PLANNED_OR_GATED_DTYPES = {
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
    "int4",
    "mxfp8",
    "mxfp6",
    "mxfp4",
    "bfp8",
    "bfp4",
    "blockfp8",
    "blockfp4",
}

MLIR_CANONICAL = {
    "fp64": "f64",
    "fp32": "f32",
    "fp16": "f16",
    "bf16": "bf16",
    "fp8_e4m3": "xf8E4M3FN",
    "fp8_e5m2": "xf8E5M2",
    "fp6_e2m3": "!tessera.fp6_e2m3",
    "fp6_e3m2": "!tessera.fp6_e3m2",
    "fp4_e2m1": "!tessera.fp4_e2m1",
    "nvfp4": "!tessera.nvfp4",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "bool": "i1",
}


def _language_spec_dtype_terms() -> set[str]:
    text = (ROOT / "docs/spec/LANGUAGE_AND_IR_SPEC.md").read_text()
    match = re.search(r"Scalar dtype terminals:\n\n```text\n(?P<body>.*?)\n```", text, re.S)
    assert match is not None
    return set(match.group("body").split())


def _hardware_matrix_dtype_terms() -> set[str]:
    text = (ROOT / "docs/audit/hardware_dtype_support_matrix.md").read_text()
    code_terms = set(re.findall(r"`([^`]+)`", text))
    terms: set[str] = set()
    for item in code_terms:
        for part in item.split(","):
            token = part.strip()
            if token:
                terms.add(token)
    return terms


def test_canonical_dtype_names_round_trip_graph_ir_metadata():
    for dtype in sorted(CANONICAL_DTYPES):
        ir_type = tensor_ir_type((2, 3), dtype, layout="row_major")

        assert ir_type.dtype == dtype
        assert str(ir_type) == f"tensor<2x3x{MLIR_CANONICAL[dtype]}>"

        parsed = _parse_mlir_tensor_type(str(ir_type))
        assert parsed.shape == ("2", "3")
        assert parsed.dtype == dtype


def test_dtype_aliases_normalize_through_graph_ir():
    for alias, canonical in ALIASES.items():
        ir_type = tensor_ir_type((2,), alias)
        assert ir_type.dtype == canonical
        assert _parse_mlir_tensor_type(str(ir_type)).dtype == canonical


def test_textual_tensor_parser_accepts_current_canonical_dtypes_and_layout():
    for dtype in sorted(CANONICAL_DTYPES):
        module = lower_text_to_graph_ir(
            f"""
            module dtype_audit {{
              func identity(A: tensor<2x3x{dtype}; layout=row_major>) -> tensor<2x3x{dtype}; layout=row_major> {{
                return A;
              }}
            }}
            """
        )

        arg = module.functions[0].args[0]
        assert arg.ir_type.dtype == dtype
        assert arg.ir_type.layout == "row_major"
        assert 'tessera.layout = "row_major"' in arg.to_mlir()


def test_canonical_doc_tracks_current_and_planned_dtype_surface():
    doc = (ROOT / "docs/reference/tessera_tensor_attributes.md").read_text()

    for dtype in CANONICAL_DTYPES:
        assert f"`{dtype}`" in doc
    for alias in ALIASES:
        assert f"`{alias}`" in doc
    for dtype in PLANNED_OR_GATED_DTYPES:
        assert f"`{dtype}`" in doc
    assert 'math_mode="tf32"' in doc
    assert "JAX" in doc
    assert "ShapeDtypeStruct" in doc


def test_spec_and_hardware_audit_include_supported_or_planned_dtype_names():
    spec_terms = _language_spec_dtype_terms()
    matrix_terms = _hardware_matrix_dtype_terms()

    assert CANONICAL_DTYPES <= spec_terms
    assert {"uint8", "uint16", "uint32", "uint64", "int4"} <= spec_terms
    assert CANONICAL_DTYPES <= matrix_terms
    assert {"uint8", "uint16", "uint32", "uint64", "int4"} <= matrix_terms


def test_target_capability_dtypes_use_canonical_names_or_documented_aliases():
    allowed = CANONICAL_DTYPES | set(ALIASES)

    for target in TARGET_CAPABILITIES.values():
        assert set(target.supported_dtypes) <= allowed
        for op_capability in target.supported_ops.values():
            assert set(op_capability.dtypes) <= allowed


def test_every_catalog_op_has_explicit_dtype_layout_contract_status():
    """Every OP_SPECS entry must carry an explicit dtype_layout_rule value
    in the canonical status set.

    Sprint C (2026-05-11) flipped the 22 long-tail partials (control_flow,
    recurrent, sparse, linalg, moe, state_space, memory) to ``complete``,
    so OP_SPECS entries now read either ``complete`` or ``not_applicable``.
    The registry-level gap-reporting *infrastructure* (the ability to
    express ``partial`` / ``planned`` for new entries) is exercised by the
    standalone primitive coverage suite; this test guards only that every
    OP_SPECS entry is *explicitly classified*, not unexpectedly empty.
    """
    coverage = all_primitive_coverages()
    unknown = sorted(set(OP_SPECS) - set(coverage))
    assert unknown == []

    incomplete = {
        name: coverage[name].contract_status.get("dtype_layout_rule")
        for name in OP_SPECS
        if coverage[name].contract_status.get("dtype_layout_rule") not in {
            "complete",
            "partial",
            "planned",
            "not_applicable",
        }
    }
    assert incomplete == {}

    # OP_SPECS entries should all be classified — no implicit defaults.
    classified = {
        name: coverage[name].contract_status.get("dtype_layout_rule")
        for name in OP_SPECS
    }
    assert all(v in {"complete", "partial", "planned", "not_applicable"}
               for v in classified.values())
