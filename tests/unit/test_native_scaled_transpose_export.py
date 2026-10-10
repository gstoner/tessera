"""Actual native scale-adjoint captures and output/lifetime ownership."""
import base64
import json
import os
import re
import subprocess
import pytest


def source(policy, roles=(2, 3), nk=False, *, shape=(2, 3, 7, 129, 256)):
    b0,b1,m,n,k = shape
    if len(shape) != 5 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("scale-transpose source requires five positive extents")
    prefix = f"{b0}x{b1}x"
    ap = prefix if policy != "shared_lhs" else ""
    bp = prefix if policy != "shared_rhs_rows" else ""
    a, b = ap+f"{m}x{k}xf8E4M3FN", bp+(f"{n}x{k}" if nk else f"{k}x{n}")+"xf8E4M3FN"
    g,c = (k+127)//128,(n+127)//128
    sa, sb = ap+f"{m}x{g}xf32", bp+f"{g}x{c}xf32"
    output = prefix+f"{m}x{n}xf32"
    args = ", ".join(f"%{name}: tensor<{typ}>"
                     for name, typ in zip(("a", "b", "sa", "sb"), (a, b, sa, sb), strict=True))
    return f'''module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
  func.func @scales({args}) -> tensor<{output}>
      attributes {{tessera.autodiff = "reverse",
                   tessera.autodiff.wrt_indices = [{", ".join(map(str, roles))}]}} {{
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {{
      batching = "{policy}", transposeB = {"true" if nk else "false"},
      numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
      scale_layout = {{granularity = "block", block = [128, 128], format = "fp32"}}
    }} : (tensor<{a}>, tensor<{b}>, tensor<{sa}>, tensor<{sb}>)
      -> tensor<{output}>
    return %y : tensor<{output}>
  }}
}}'''


def compile_native(text, options="export-scaled-transpose=true"):
    opt = os.environ.get("TESSERA_OPT")
    if not opt:
        pytest.skip("matching native MLIR compiler required")
    return subprocess.run([opt, "--tessera-autodiff-paired="+options],
                          input=text, text=True, capture_output=True, timeout=90)


def manifest(text):
    matches = re.findall(r'tessera.autodiff.scaled_program_json\s*=\s*"([A-Za-z0-9+/=]+)"', text)
    assert len(matches) == 1
    return json.loads(base64.b64decode(matches[0], validate=True))


@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("roles", [(2,), (3,), (2, 3), (3, 2)])
@pytest.mark.parametrize("nk", [False, True])
def test_native_transpose_export_captures_and_requested_order(policy, roles, nk):
    result = compile_native(source(policy, roles, nk))
    assert result.returncode == 0, result.stderr
    program = manifest(result.stdout)
    assert program["kind"] == "scale_vjp"
    assert program["gradient_roles"] == list(roles)
    assert program["argument_count"] == 5
    assert len(program["steps"]) == len(roles)
    assert "scf.for" in program["root_ir"] and "tensor.generate" in program["root_ir"]
    by_role = {step["gradient_role"]: step for step in program["steps"]}
    for role, inputs in (("lhs_scale", [0, 1, 3, 4]), ("rhs_scale", [0, 1, 2, 4])):
        if role in by_role:
            assert by_role[role]["operation"] == "tensor.generate"
            assert by_role[role]["inputs"] == inputs
    expected = [by_role["lhs_scale" if role == 2 else "rhs_scale"]["output"] for role in roles]
    assert program["outputs"] == expected
    for role, output in zip(roles, expected, strict=True):
        row = program["buffers"][output]
        assert row["shape"] == program["buffers"][role]["shape"]
        assert row["storage"] == "f32" and row["ownership"] == 2
        assert row["last_read"] == len(roles)
    # Complete lifetime information, including arguments not read by a
    # one-gradient request, must agree with the actual region captures.
    for index, row in enumerate(program["buffers"][:5]):
        uses = [step["step"] for step in program["steps"] if index in step["inputs"]]
        assert row["last_read"] == (max(uses) if uses else -1)


@pytest.mark.parametrize("roles", [(0,), (1,), (), (2, 2)])
def test_native_transpose_export_rejects_invalid_derivative_roles(roles):
    result = compile_native(source("independent_rhs", roles))
    assert result.returncode != 0
    assert "scaled transpose" in result.stderr


def test_native_transpose_member_projection_preserves_real_body():
    result = compile_native(source("shared_lhs", (3, 2)),
                            "export-scaled-transpose=true select-scaled-transpose-member=0")
    assert result.returncode == 0, result.stderr
    assert "tessera.autodiff.scaled_program_witness" in result.stdout
    assert 'operation = "tensor.generate"' in result.stdout
    assert 'tessera.autodiff.scale_adjoint = "lhs_scale"' in result.stdout
    # One actual outlined region, with its captures remapped to member args.
    assert len(re.findall(r"^  func.func private @scales__bwd__member_0\(",
                          result.stdout, flags=re.MULTILINE)) == 1


def test_native_transpose_export_requires_fresh_pair():
    first = compile_native(source("independent_rhs"), options="")
    assert first.returncode == 0, first.stderr
    second = compile_native(first.stdout)
    assert second.returncode != 0
    assert "fresh reverse pairing" in second.stderr


def test_native_transpose_export_preserves_frontend_argument_permutation():
    text = source("independent_rhs", (0, 1))
    start = text.index("func.func @scales(") + len("func.func @scales(")
    end = text.index(") ->", start)
    arguments = text[start:end].split(", ")
    text = text[:start] + ", ".join(arguments[2:] + arguments[:2]) + text[end:]
    result = compile_native(text)
    assert result.returncode == 0, result.stderr
    program = manifest(result.stdout)
    assert program["gradient_roles"] == [0, 1]
    steps = {step["gradient_role"]: step for step in program["steps"]}
    assert steps["lhs_scale"]["inputs"] == [1, 2, 3, 4]
    assert steps["rhs_scale"]["inputs"] == [0, 2, 3, 4]
    for role, output in zip((0, 1), program["outputs"], strict=True):
        assert program["buffers"][output]["shape"] == program["buffers"][role]["shape"]
