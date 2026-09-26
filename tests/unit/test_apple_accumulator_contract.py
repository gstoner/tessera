"""APPLE-ACCUM-1 review fixes -- host-free contracts (no GPU needed).

* One accumulator spelling map across Python and C++ (``tessera.dtype``).
* The front door reads every carrier shape ``jit._serialized_numeric_policy``
  supports, refuses unknown ones, and stamps the registered fp32 default only
  for floating-point operands.
* The raw MSL emitters take no default accumulator.
"""
from __future__ import annotations

import subprocess

import pytest

from tessera.compiler.apple_fragment import canonical_accumulator_dtype
from tessera.compiler.graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
    NumericPolicy as GraphNumericPolicy,
)
from tessera.dtype import dtype_aliases


def _float_accumulator_spellings() -> dict[str, str]:
    """Every tessera.dtype spelling of fp32/fp16/bf16, with case folds."""
    out = {c: c for c in ("fp32", "fp16", "bf16")}
    for alias, canon in dtype_aliases().items():
        if canon in out:
            out[alias] = canon
    for name in list(out):
        out[name.upper()] = out[name]
    return out


def test_python_accumulator_spelling_is_tessera_dtype():
    for spelling, canon in _float_accumulator_spellings().items():
        assert canonical_accumulator_dtype(spelling) == canon, spelling
    assert canonical_accumulator_dtype("i32") == "int32"
    assert canonical_accumulator_dtype("not_a_dtype") == "not_a_dtype"


def _simdgroup_matmul(accum: str) -> str:
    return (
        "func.func @f(%a: tensor<16x16xf16>, %b: tensor<16x8xf16>) -> tensor<16x8xf32> {\n"
        f'  %c = "tessera.matmul"(%a, %b) {{numeric_policy = {{accum = "{accum}"}}}}\n'
        "      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf32>\n"
        "  return %c : tensor<16x8xf32>\n}\n")


def test_apple_accumulator_spellings_match_tessera_dtype(compiler_toolchain):
    """C++ appleAccumulatorType must accept exactly tessera.dtype's spellings:
    fp32/fp16 aliases lower, bf16 aliases are *recognized* and refused for the
    measured reason (never as an unknown name), and a non-dtype is unknown."""
    opt = compiler_toolchain.require_tessera_opt("tessera-matmul-to-apple-simdgroup")

    def run(accum: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(opt), "-", "--tessera-matmul-to-apple-simdgroup",
             "--allow-unregistered-dialect"],
            input=_simdgroup_matmul(accum), capture_output=True, text=True,
            check=False, timeout=60)

    for spelling, canon in _float_accumulator_spellings().items():
        proc = run(spelling)
        assert "not a Decision #15a accumulator" not in proc.stderr, spelling
        if canon == "bf16":
            assert proc.returncode != 0 and "is not bf16 accumulation" in proc.stderr
        else:
            assert proc.returncode == 0, (spelling, proc.stderr)
            elem = "f32" if canon == "fp32" else "f16"
            assert f"-> <{elem}>" in proc.stdout, spelling
    unknown = run("not_a_dtype")
    assert unknown.returncode != 0
    assert "not a Decision #15a accumulator" in unknown.stderr


def _module(storage_mlir: str, storage: str, *, kwargs=None, carried=None) -> GraphIRModule:
    ta = IRType(f"tensor<8x8x{storage_mlir}>", ("8", "8"), storage)
    tc = IRType("tensor<8x8xf32>", ("8", "8"), "fp32")
    op = IROp(result="c", op_name="tessera.matmul", operands=["%a", "%b"],
              operand_types=[ta.mlir_str, ta.mlir_str], result_type=tc.mlir_str,
              kwargs=dict(kwargs or {}))
    op.numeric_policy = carried
    return GraphIRModule(functions=[GraphIRFunction(
        name="f", args=[IRArg("a", ta), IRArg("b", ta)], result_types=[tc],
        body=[op], return_values=["%c"])])


def _stamped(module: GraphIRModule):
    from tessera.compiler.driver import materialize_matmul_accumulators

    return materialize_matmul_accumulators(module).functions[0].body[0].kwargs.get(
        "numeric_policy")


def test_front_door_reads_a_dict_carried_accumulator():
    """P1-2: a dict-form IROp.numeric_policy (what jit serializes) is read,
    not overwritten with the fp32 default."""
    assert _stamped(_module("f16", "fp16", carried={"accum": "fp16"})) == {"accum": "fp16"}
    # a dict with no accumulator falls through to the registered default
    assert _stamped(_module("f16", "fp16", carried={"storage": "fp16"})) == {"accum": "fp32"}


def test_front_door_reads_a_registry_policy_and_defaults_float_storage():
    from tessera.compiler.primitive_coverage import NumericPolicy

    assert _stamped(_module("bf16", "bf16", carried=NumericPolicy(
        storage="bf16", accum="fp16"))) == {"accum": "fp16"}
    assert _stamped(_module("f16", "fp16")) == {"accum": "fp32"}


def test_front_door_keeps_a_rendered_policy_unchanged():
    declared = {"accum": "bf16", "storage": "bf16"}
    assert _stamped(_module("bf16", "bf16", kwargs={"numeric_policy": declared})) == declared


def test_front_door_does_not_stamp_fp32_on_non_float_storage():
    """The registry's fp32 default ignores storage; an int8 matmul's documented
    accumulator is int32, so nothing is stamped and the backend refuses, named."""
    assert _stamped(_module("i8", "int8")) is None


@pytest.mark.parametrize("carried", [
    GraphNumericPolicy(storage="fp16"),   # accum defaults to "f32": ambiguous
    ("fp16", "fp32"),
    "fp16",
])
def test_front_door_refuses_an_unrecognized_carrier(carried):
    from tessera.compiler.driver import (
        AppleAccumulatorPolicyError,
        materialize_matmul_accumulators,
    )

    with pytest.raises(AppleAccumulatorPolicyError, match="APPLE_ACCUM_POLICY_UNRECOGNIZED"):
        materialize_matmul_accumulators(_module("f16", "fp16", carried=carried))


def test_front_door_refusal_is_recorded_not_raised_through_the_driver():
    from tessera.compiler.canonical_compile import canonical_compile

    art = canonical_compile(
        _module("f16", "fp16", carried=("fp16",)), target="apple_gpu",
        options={"apple_target_ir_mode": "value"}).to_runtime_artifact()
    assert "APPLE_ACCUM_POLICY_UNRECOGNIZED" in str(
        art.metadata.get("apple_value_target_ir_error"))


def test_raw_msl_emitters_take_no_default_accumulator():
    from tessera.compiler import msl_gemm_emit as emit

    with pytest.raises(TypeError):
        emit.emit_simdgroup_gemm_msl("f16")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        emit.emit_steel_gemm_msl("f16")  # type: ignore[call-arg]
    msl = emit.emit_steel_gemm_msl("f16", accum="fp32")
    with pytest.raises(TypeError):
        emit.validate_msl_gemm_structure(msl)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        emit.validate_steel_gemm_structure(msl)  # type: ignore[call-arg]
