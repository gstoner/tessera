"""Native mixed-adjoint contraction: typed inputs reach ordered FP32 FMA.

Compiler proof only. Public composed reverse and exact-device proof are
separate gates; no hardware MMA or performance claim follows from this test.
"""
import subprocess

import pytest


def source(a="f32", b="f16", output="f16", transpose_a=False, transpose_b=True,
           accumulator="f32"):
    return f"""module {{
  llvm.func @mixed_adjoint(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                           %m: i64, %n: i64, %k: i64) attributes {{nvvm.kernel}} {{
    "tile.scalar_matmul_kernel"(%a, %b, %d, %m, %n, %k) {{
      a_element_type = {a}, b_element_type = {b},
      output_element_type = {output}, accumulator_type = {accumulator},
      transpose_a = {str(transpose_a).lower()},
      transpose_b = {str(transpose_b).lower()}
    }} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }}
}}"""


def lower(compiler_toolchain, text, sm=120):
    pipeline = f"--lower-tile-to-nvidia=sm={sm}"
    tool = compiler_toolchain.require_tessera_opt(pipeline)
    return subprocess.run([str(tool), pipeline, "-"], input=text, text=True,
                          capture_output=True, timeout=60)


@pytest.mark.compiler_route
@pytest.mark.parametrize("half", ["f16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("orientation", [(False, False), (False, True), (True, False), (True, True)])
def test_mixed_contraction_preserves_wide_operand(
    compiler_toolchain, half, side, output, orientation,
):
    a, b = ("f32", half) if side == "a" else (half, "f32")
    result = lower(compiler_toolchain, source(a, b, output, *orientation))
    assert result.returncode == 0, result.stderr
    assert "tile.scalar_matmul_kernel" not in result.stdout
    assert "fma" in result.stdout
    assert "mma.sync" not in result.stdout
    assert "tf32" not in result.stdout
    assert "extf" in result.stdout or "fpext" in result.stdout
    assert "load" in result.stdout and "store" in result.stdout


@pytest.mark.compiler_route
@pytest.mark.parametrize("change,marker", [
    ({"accumulator": "f16"}, "requires f32 accumulation"),
    ({"a": "f64"}, "requires explicit f32/f16/bf16"),
])
def test_scalar_contract_rejects_changed_precision(compiler_toolchain, change, marker):
    result = lower(compiler_toolchain, source(**change))
    assert result.returncode != 0
    assert marker in result.stderr


@pytest.mark.compiler_route
def test_scalar_contract_requires_owning_target(compiler_toolchain):
    result = lower(compiler_toolchain, source(), sm=90)
    assert result.returncode != 0
    assert "requires exact sm_120" in result.stderr
