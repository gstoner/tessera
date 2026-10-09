"""Native continuous operands of the target-neutral scaled product."""
import pytest

from tests.unit.test_native_scaled_transpose_export import compile_native, manifest


def floating_source(ta=False, tb=False, roles=(0, 1, 2, 3), permute=False):
    a = "9x2xf32" if ta else "2x9xf32"
    b = "5x9xf32" if tb else "9x5xf32"
    output = "5x2xf32" if permute else "2x5xf32"
    returned = ('%placed = "tessera.transpose"(%y) {permutation = array<i64: 1, 0>} '
                ': (tensor<2x5xf32>) -> tensor<5x2xf32>\n    return %placed : tensor<5x2xf32>'
                if permute else "return %y : tensor<2x5xf32>")
    return f'''module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
  func.func @floating_product(%a: tensor<{a}>, %b: tensor<{b}>,
                             %sa: tensor<2x3xf32>, %sb: tensor<3x2xf32>)
      -> tensor<{output}> attributes {{tessera.autodiff = "reverse",
          tessera.autodiff.wrt_indices = [{", ".join(map(str, roles))}]}} {{
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {{
      transposeA = {"true" if ta else "false"}, transposeB = {"true" if tb else "false"},
      numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
      scale_layout = {{granularity = "block", block = [4, 4], format = "fp32"}}
    }} : (tensor<{a}>, tensor<{b}>, tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x5xf32>
    {returned}
  }}
}}'''


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("roles", [(0,), (1,), (2,), (3,), (0, 1, 2, 3), (3, 1, 0, 2)])
def test_native_floating_adjoint_owns_requested_roles_and_excludes_own_capture(ta, tb, roles):
    result = compile_native(floating_source(ta, tb, roles))
    assert result.returncode == 0, result.stderr
    program = manifest(result.stdout)
    assert program["gradient_roles"] == list(roles)
    names = {0: "lhs_matrix", 1: "rhs_matrix", 2: "lhs_scale", 3: "rhs_scale"}
    for role, slot in zip(roles, program["outputs"], strict=True):
        member = program["steps"][slot - 5]
        assert member["gradient_argument"] == role
        assert member["gradient_role"] == names[role]
        assert member["inputs"] == [i for i in range(5) if i != role]
        assert program["buffers"][slot]["shape"] == program["buffers"][role]["shape"]
    assert "arith.extf" not in program["root_ir"]


def test_floating_matrix_adjoint_reaches_gpu_target():
    import os
    import subprocess
    opt = os.environ.get("TESSERA_OPT")
    if not opt:
        pytest.skip("matching compiler required")
    # Export keeps scale reductions first in SSA order; selecting role A is
    # resolved from the native manifest rather than hardcoding member order.
    result = compile_native(floating_source(permute=True))
    assert result.returncode == 0, result.stderr
    program = manifest(result.stdout)
    index = next(row["step"] for row in program["steps"]
                 if row.get("gradient_role") == "lhs_matrix")
    pipeline = ("builtin.module(tessera-autodiff-paired{export-scaled-transpose=true "
                f"select-scaled-transpose-member={index}" +
                "},tessera-graph-to-schedule,tessera-schedule-to-tile,"
                "tessera-rocm-executable{family=reduction input=tile output=target arch=gfx1201})")
    lowered = subprocess.run([opt, "--pass-pipeline=" + pipeline],
                             input=floating_source(permute=True), text=True,
                             capture_output=True, timeout=90)
    assert lowered.returncode == 0, lowered.stderr
    assert "gpu.func" in lowered.stdout
    assert "memref.load" in lowered.stdout and "memref.store" in lowered.stdout
    assert "tessera.autodiff.scaled_member" in lowered.stdout
    assert "tile.structured_reduction_kernel" not in lowered.stdout


def floating_batch_shapes(policy, ta=False, tb=False):
    prefixes = {
        "shared_rhs_rows": ((2, 3), (), (2, 3), ()),
        "shared_lhs": ((), (2, 3), (), (2, 3)),
        "independent_rhs": ((2, 3), (2, 3), (2, 3), (2, 3)),
        # Scales have their own broadcast prefixes, independently of matrices.
        "broadcast": ((2, 1), (3,), (1, 3), (2, 1)),
    }
    ap, bp, sp, tp = prefixes[policy]
    return (ap + ((9, 2) if ta else (2, 9)),
            bp + ((5, 9) if tb else (9, 5)),
            sp + (2, 3), tp + (3, 2))


def floating_batch_source(policy, ta=False, tb=False, roles=(0, 1, 2, 3)):
    shapes = floating_batch_shapes(policy, ta, tb)
    types = ["x".join(map(str, shape)) + "xf32" for shape in shapes]
    args = ", ".join(f"%{name}: tensor<{typ}>" for name, typ
                     in zip(("a", "b", "sa", "sb"), types, strict=True))
    signatures = ", ".join(f"tensor<{typ}>" for typ in types)
    return f'''module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
  func.func @floating_batch({args}) -> tensor<2x3x2x5xf32>
      attributes {{tessera.autodiff = "reverse",
                   tessera.autodiff.wrt_indices = [{", ".join(map(str, roles))}]}} {{
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {{
      batching = "{policy}", transposeA = {"true" if ta else "false"},
      transposeB = {"true" if tb else "false"},
      numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
      scale_layout = {{granularity = "block", block = [4, 4], format = "fp32"}}
    }} : ({signatures}) -> tensor<2x3x2x5xf32>
    return %y : tensor<2x3x2x5xf32>
  }}
}}'''


@pytest.mark.parametrize("policy", ["shared_rhs_rows", "shared_lhs", "independent_rhs", "broadcast"])
@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("roles", [(0,), (1,), (0, 1, 2, 3), (3, 1, 0, 2)])
def test_floating_batch_adjoint_preserves_shared_shapes_and_captures(policy, ta, tb, roles):
    result = compile_native(floating_batch_source(policy, ta, tb, roles))
    assert result.returncode == 0, result.stderr
    program = manifest(result.stdout)
    assert program["gradient_roles"] == list(roles)
    members = {member["output"]: member for member in program["steps"]}
    for role, slot in zip(roles, program["outputs"], strict=True):
        member = members[slot]
        assert member["gradient_argument"] == role
        assert member["inputs"] == [i for i in range(5) if i != role]
        assert program["buffers"][slot]["shape"] == list(floating_batch_shapes(policy, ta, tb)[role])
        assert program["buffers"][slot]["last_read"] == len(roles)
    for index, row in enumerate(program["buffers"][:5]):
        uses = [member["step"] for member in members.values() if index in member["inputs"]]
        assert row["last_read"] == (max(uses) if uses else -1)
