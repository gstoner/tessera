"""The row-program emitter's math admission table is closed and mirrored.

On both device routes a `math.*` op reaches a vendor library whose default
accuracy is a property of that library: libdevice's `__nv_sqrtf` took the
approximate path on sm_120 because MLIR never sets the precise-sqrt reflect
value. The emitter therefore admits a fixed set of math ops, each with a plan
that says *why* its result can be trusted, and refuses everything else rather
than passing it through unmeasured (Decision #21a). These tests hold the table,
its Python mirror and the recorder that measures it together.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tessera.compiler.native_row_program import (ADMITTED_MATH, MATH_AUDIT_DOMAINS, REFUSED_MATH,
                                                 declared_math, row_program_kernel,
                                                 row_unary_math_module)
from tessera.compiler.scheduled_matmul import find_tessera_opt

ROOT = Path(__file__).resolve().parents[2]
PASS_SOURCE = ROOT / "src/transforms/lib/RowProgramToGPUPass.cpp"
RECORDER = ROOT / "benchmarks/record_row_program_math_precision.py"
PLANS = {"rounding_explicit": "RoundingExplicit", "bit_exact": "BitExact", "measured": "Measured"}


def _cpp_table() -> dict[str, str]:
    body = PASS_SOURCE.read_text().split("kMathAdmission[] = {", 1)[1].split("};", 1)[0]
    entries = re.findall(r'\{"([a-z0-9_.]+)",\s*MathPlan::(\w+)', body)
    inverse = {cpp: name for name, cpp in PLANS.items()}
    return {op: inverse[plan] for op, plan in entries}


def test_the_python_mirror_matches_the_pass_table():
    """A row admitted in one place and not the other is how an unmeasured op
    slips back in: the pass would accept it and the audit would never sweep it."""
    assert _cpp_table() == ADMITTED_MATH


def test_every_admitted_op_has_a_host_reference_and_a_swept_domain():
    text = RECORDER.read_text()
    host = set(re.findall(r'"(math\.[a-z0-9_]+)": np\.', text))
    assert host == set(ADMITTED_MATH), "the recorder's host references must cover exactly the admitted set"
    assert set(MATH_AUDIT_DOMAINS) == set(ADMITTED_MATH)
    for op, domains in MATH_AUDIT_DOMAINS.items():
        assert domains, f"{op} declares no input domain to sweep"
        for lo, hi in domains:
            assert lo < hi, f"{op} domain {(lo, hi)} is empty"


def test_a_measured_plan_is_only_for_ops_the_route_cannot_round_explicitly():
    """`sqrt` has a correctly-rounded libdevice entry, so it must not be
    admitted as `measured`; the others have none, so they must not claim to be
    rounding-explicit."""
    assert ADMITTED_MATH["math.sqrt"] == "rounding_explicit"
    assert not {op for op, plan in ADMITTED_MATH.items() if plan == "rounding_explicit"} - {"math.sqrt"}


def _compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("tessera-opt required")
    help_text = subprocess.run([str(tool), "--help"], capture_output=True, text=True).stdout
    if "--tessera-row-program-to-gpu" not in help_text:
        pytest.skip("tessera-opt built without the row-program emitter")
    return tool


@pytest.mark.parametrize("op", sorted(ADMITTED_MATH))
def test_an_admitted_op_lowers_and_declares_its_plan(op):
    tool = _compiler()
    for backend in ("nvidia", "rocm"):
        kernel, lanes = row_program_kernel(row_unary_math_module(4, 8, op), entry="unary_math",
                                           backend=backend, compiler=tool)
        assert lanes == 8
        assert declared_math(kernel) == (f"{op}:{ADMITTED_MATH[op]}",)


def test_a_rounding_explicit_op_is_realized_as_the_correctly_rounded_call():
    tool = _compiler()
    body = lambda text: text.split("gpu.func @row_program(", 1)[1]
    nv, _ = row_program_kernel(row_unary_math_module(4, 8, "math.sqrt"), entry="unary_math",
                               backend="nvidia", compiler=tool)
    assert "llvm.call @__nv_fsqrt_rn" in body(nv) and "math.sqrt" not in body(nv)
    rocm, _ = row_program_kernel(row_unary_math_module(4, 8, "math.sqrt"), entry="unary_math",
                                 backend="rocm", compiler=tool)
    assert "llvm.intr.sqrt" in body(rocm) and "math.sqrt" not in body(rocm)


@pytest.mark.parametrize("op", ["math.rsqrt", "math.erf", "math.powf", "math.exp2",
                                "math.tanh", "math.log1p"])
def test_an_unadmitted_math_op_is_refused_by_name(op):
    """The refusal must name the op and the recorder, because the fix is to
    measure it — not to widen the table and hope."""
    tool = _compiler()
    operand = "%a, %a" if op == "math.powf" else "%a"
    module = f"""#map = affine_map<(d0, d1) -> (d0, d1)>
module {{
  func.func @unary_math(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {{
    %e = tensor.empty() : tensor<2x4xf32>
    %y = linalg.generic {{indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%x : tensor<2x4xf32>) outs(%e : tensor<2x4xf32>) {{
    ^bb0(%a: f32, %o: f32):
      %v = {op} {operand} : f32
      linalg.yield %v : f32
    }} -> tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }}
}}
"""
    result = subprocess.run([str(tool), "-", "--tessera-row-program-to-gpu=backend=nvidia entry=unary_math"],
                            input=module, capture_output=True, text=True)
    assert result.returncode != 0, "an unmeasured math op must not lower"
    assert f"`{op}` is not in the emitter's math admission table" in result.stderr
    assert "record_row_program_math_precision.py" in result.stderr
    assert "lowering failed without a diagnostic" not in result.stderr


def test_the_refused_set_is_disjoint_and_reasoned():
    """`tanh` and `log1p` were admitted until the audit measured them; they are
    recorded as refused *with the measurement* so nobody re-admits them on the
    assumption that nobody had looked."""
    assert not set(REFUSED_MATH) & set(ADMITTED_MATH)
    assert set(REFUSED_MATH) == {"math.tanh", "math.log1p"}
    for op, reason in REFUSED_MATH.items():
        assert len(reason) > 20 and op.split(".")[-1] not in reason.split()[0]
    table = PASS_SOURCE.read_text()
    for op in REFUSED_MATH:
        assert op in table, f"the pass must record why {op} is refused, not just omit it"
