"""Explicit native broadcast carriers for mapped scaled-product sums."""
import copy
import json
import os

import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tests.unit.test_composed_scaled_jvp import composed, case as scalar_case


def case(side=0, depth=1, mode=None):
    _, values, _ = scalar_case((3, 5, 37))
    options = {} if mode is None else {"autodiff": mode, "wrt": ("sa0", "sb0", "sa1", "sb1")}
    scalar = ts.jit(target="rocm_gfx1201", **options)(composed)
    axes = tuple(0 if index == (2 if side == 0 else 4) else None for index in range(6))
    prefix = (2,) if depth == 1 else (2, 3)
    owner = scalar
    for _ in prefix:
        owner = vmap(owner, in_axes=axes)
    arrays = tuple(np.broadcast_to(value, (*prefix, *value.shape)).copy() if axis is not None
                   else value.copy() for value, axis in zip(values, axes, strict=True))
    directions = tuple(np.full_like(arrays[index], .03125) for index in (2, 3, 4, 5))
    return scalar, owner, arrays, directions, axes, prefix


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("mode", [None, "forward", "reverse"])
def test_projection_materializes_only_the_shared_result(side, depth, mode):
    scalar, owner, values, _, _, prefix = case(side, depth, mode)
    before = copy.deepcopy(scalar.graph_ir)
    graph = owner._specialized_autodiff_module(values, {})
    body = graph.functions[0].body
    assert [op.op_name for op in body] == ["tessera.scaled_matmul", "tessera.scaled_matmul",
                                         "tessera.broadcast", "tessera.add"]
    source = body[1-side]
    assert body[2].operands == ["%" + source.result_names[0]]
    assert tuple(map(int, source.inferred_type.shape)) == (3, 5)
    assert tuple(map(int, body[2].inferred_type.shape)) == (*prefix, 3, 5)
    assert scalar.graph_ir == before
    assert owner.frontend_differential(*values)


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching native compiler required")
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("mode", [None, "forward", "reverse"])
def test_native_broadcast_and_transpose_reduction_have_checked_storage(side, depth, mode):
    from tessera.compiler.native_scaled_program import (
        package_native_scaled_primal, package_native_scaled_jvp, package_native_scaled_vjp)
    _, owner, values, _, _, _ = case(side, depth, mode)
    graph = owner._specialized_autodiff_module(values, {})
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    builder = package_native_scaled_vjp if mode == "reverse" else package_native_scaled_jvp if mode == "forward" else package_native_scaled_primal
    package = builder(graph.to_mlir(target="rocm_gfx1201", canonical=True))
    package.validate()
    program = json.loads(package.program_json)
    carriers = [step for step in program["steps"] if step.get("lowering") == "structured_f32_carrier"]
    assert carriers
    if mode == "reverse":
        assert sum(step["operation"] == "tessera.reduce" for step in carriers) == depth
        for role, output in zip(program["gradient_roles"], program["outputs"], strict=True):
            assert program["buffers"][output]["shape"] == list(values[role].shape)
    else:
        assert any(step["operation"] == "tessera.broadcast" for step in carriers)

@pytest.fixture(scope="module")
def reverse_package():
    if not os.environ.get("TESSERA_OPT"):
        pytest.skip("matching native compiler required")
    from tessera.compiler.native_scaled_program import package_native_scaled_vjp
    _, owner, values, _, _, _ = case(0, 2, "reverse")
    graph = owner._specialized_autodiff_module(values, {})
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    return package_native_scaled_vjp(graph.to_mlir(target="rocm_gfx1201", canonical=True))


@pytest.mark.parametrize("mutation", ["axis", "algorithm", "count"])
def test_carrier_package_rejects_forged_binding(reverse_package, mutation):
    import base64
    from dataclasses import replace
    program = json.loads(reverse_package.program_json)
    members = [json.loads(raw) for raw in reverse_package.members_json]
    index = next(step["step"] for step in program["steps"] if step["operation"] == "tessera.reduce")
    if mutation == "axis":
        program["steps"][index]["axis"] = 8
    elif mutation == "algorithm":
        members[index]["scale_adjoint_schedule"] = "serial_per_scale_element"
    else:
        members[index]["scalars"][0] += 1
    # Keep the common witness internally consistent so the actual carrier
    # axis/algorithm/count checks, rather than a stale witness, reject it.
    witness = base64.b64encode(json.dumps(program).encode()).decode()
    for member in members:
        member["program_base64"] = witness
    corrupted = replace(reverse_package, program_json=json.dumps(program),
                        members_json=tuple(json.dumps(member) for member in members))
    with pytest.raises(ValueError, match="native scaled (carrier|reduction)"):
        corrupted.validate()
