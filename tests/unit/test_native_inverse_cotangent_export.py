"""Native mapped reverse AD owns inverse-cotangent dependencies."""
import pytest

from tests.unit.test_native_scaled_transpose_export import compile_native, manifest, source


def mapped_source(policy="independent_rhs", roles=(2, 3), nk=False):
    text = source(policy, roles, nk, shape=(2, 3, 7, 19, 256))
    canonical = "2x3x7x19xf32"
    mapped = "7x2x3x19xf32"
    text = text.replace(") -> tensor<" + canonical + ">",
                        ") -> tensor<" + mapped + ">", 1)
    return text.replace("return %y : tensor<" + canonical + ">",
        '%mapped = "tessera.transpose"(%y) {permutation = array<i64: 2, 0, 1, 3>} '
        ': (tensor<' + canonical + '>) -> tensor<' + mapped + '>\n'
        '    return %mapped : tensor<' + mapped + '>')


@pytest.mark.parametrize("policy", ["shared_lhs", "shared_rhs_rows", "independent_rhs"])
@pytest.mark.parametrize("roles", [(2,), (3,), (2, 3), (3, 2)])
@pytest.mark.parametrize("nk", [False, True])
def test_native_inverse_cotangent_is_owned_before_scale_reductions(policy, roles, nk):
    result = compile_native(mapped_source(policy, roles, nk))
    assert result.returncode == 0, result.stderr
    program = manifest(result.stdout)
    assert program["gradient_roles"] == list(roles)
    seed = program["steps"][0]
    assert seed["operation"] == "tessera.transpose"
    assert seed["inputs"] == [4] and seed["output"] == 5
    assert seed["cotangent_source"] == 4
    assert "gradient_argument" not in seed
    assert seed["permutation"] == [1, 2, 0, 3]
    assert program["buffers"][4]["shape"] == [7, 2, 3, 19]
    private = program["buffers"][5]
    assert private["shape"] == [2, 3, 7, 19]
    assert (private["ownership"], private["first_write"], private["last_read"]) == (1, 0, len(roles))
    reductions = program["steps"][1:]
    assert all(step["operation"] == "tensor.generate" for step in reductions)
    assert all(step["inputs"][-1] == 5 and 4 not in step["inputs"] for step in reductions)
    for role, slot in zip(roles, program["outputs"], strict=True):
        assert program["buffers"][slot]["shape"] == program["buffers"][role]["shape"]


def test_inverse_cotangent_projects_to_real_native_graph_member():
    result = compile_native(mapped_source(), "export-scaled-transpose=true select-scaled-transpose-member=0")
    assert result.returncode == 0, result.stderr
    assert "tessera.autodiff.scaled_program_witness" in result.stdout
    assert "permutation = array<i64: 1, 2, 0, 3>" in result.stdout
    assert 'tessera.launch_bindings = ["member_input_0", "member_output"]' in result.stdout

def test_inverse_cotangent_reaches_native_gpu_target():
    import os
    import subprocess
    opt = os.environ.get("TESSERA_OPT")
    if not opt:
        pytest.skip("matching compiler required")
    pipeline = ("builtin.module(tessera-autodiff-paired{export-scaled-transpose=true "
                "select-scaled-transpose-member=0},tessera-graph-to-schedule,"
                "tessera-schedule-to-tile,tessera-rocm-executable{"
                "family=scalar_unary input=tile output=target arch=gfx1201})")
    result = subprocess.run([opt, "--pass-pipeline=" + pipeline],
                            input=mapped_source(), text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "gpu.func @tessera_tile_result_permutation_" in result.stdout
    assert "memref.load" in result.stdout and "memref.store" in result.stdout
    assert "tessera.rocm.program_member_json" in result.stdout


@pytest.fixture(scope="module")
def inverse_package():
    import os
    from tessera.compiler.native_scaled_program import package_native_scaled_vjp
    if not os.environ.get("TESSERA_ROCM_OPT"):
        pytest.skip("matching ROCm serialization toolchain required")
    return package_native_scaled_vjp(mapped_source())


@pytest.mark.parametrize("mutation", ["source", "boolean_source", "seed_binding", "premature_lifetime"])
def test_reencoded_inverse_manifest_retains_seed_lineage_and_private_lifetime(inverse_package, mutation):
    import base64
    import json
    from dataclasses import replace
    program = json.loads(inverse_package.program_json)
    members = [json.loads(row) for row in inverse_package.members_json]
    if mutation == "source":
        program["steps"][0]["cotangent_source"] = 3
    elif mutation == "boolean_source":
        program["steps"][0]["cotangent_source"] = True
    elif mutation == "seed_binding":
        program["steps"][0]["inputs"] = [3]
        members[0]["inputs"] = [3]
    else:
        program["buffers"][5]["last_read"] = 0
    encoded = json.dumps(program)
    witness = base64.b64encode(encoded.encode()).decode()
    for member in members:
        member["program_base64"] = witness
    broken = replace(inverse_package, program_json=encoded,
                     members_json=tuple(json.dumps(member) for member in members))
    with pytest.raises(ValueError):
        broken.validate()
