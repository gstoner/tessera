"""Public nonleading map results remain compiler-owned Graph permutations."""
import copy

import pytest
from tessera.autodiff import vmap
from tessera.compiler.rocm_typed_scaled_native import supports_composed_scaled_primal
from tests.unit.test_native_typed_scaled_vmap import case


@pytest.mark.parametrize("policy", ["independent_rhs", "shared_lhs", "shared_rhs_rows"])
@pytest.mark.parametrize("fmt", ["fp32", "e8m0"])
@pytest.mark.parametrize("out_axis", [1, 2, -1])
def test_nonleading_output_projects_graph_and_certifies_scalar_map(policy, fmt, out_axis):
    scalar, leading, values, _ = case(policy, fmt, False, (2, 7, 19, 256))
    original = copy.deepcopy(scalar.graph_ir)
    owner = vmap(scalar, in_axes=leading._frontend_batch_axes, out_axes=out_axis)
    module = owner._specialized_autodiff_module(values, {})
    fn = module.functions[0]
    assert fn.body[-1].op_name == "tessera.transpose"
    axes = list(range(1, 3))
    axes.insert(out_axis % 3, 0)
    assert fn.body[-1].kwargs["permutation"] == axes
    assert fn.result_types[0].shape == tuple(str((2, 7, 19)[i]) for i in axes)
    assert supports_composed_scaled_primal(module)
    assert owner.frontend_differential(*values)
    assert scalar.graph_ir == original


def test_nested_result_placement_composes_each_map_level():
    scalar, _, flat, _ = case("independent_rhs", "e8m0", False, (6, 7, 19, 256))
    inner = vmap(scalar, in_axes=0, out_axes=1)
    outer = vmap(inner, in_axes=0, out_axes=2)
    values = tuple(value.reshape(2, 3, *value.shape[1:]) for value in flat)
    graph = outer._specialized_autodiff_module(values, {})
    assert outer._frontend_output_permutation == (2, 1, 0, 3)
    assert graph.functions[0].result_types[0].shape == ("7", "3", "2", "19")
    assert outer.frontend_differential(*values)


@pytest.mark.parametrize("axis", [True, 3, -4])
def test_invalid_output_axis_rejected_at_map_construction(axis):
    scalar, _, _, _ = case("independent_rhs", "fp32", False)
    with pytest.raises(ValueError, match="axis"):
        vmap(scalar, in_axes=0, out_axes=axis)

def test_identity_axes_with_wrong_rank_cannot_bypass_graph_admission():
    from tessera.compiler.native_vmap import project_result_axes
    _, owner, values, _ = case("independent_rhs", "fp32", False)
    module = owner._specialized_autodiff_module(values, {})
    original = copy.deepcopy(module)
    for axes in ((), (0,), (0, 1), (False, 1, 2)):
        with pytest.raises(ValueError, match="permute the result rank"):
            project_result_axes(module, axes)
    assert module == original
