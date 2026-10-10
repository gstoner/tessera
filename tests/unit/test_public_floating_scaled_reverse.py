"""Public Python tracing and semantic certification for native f32 reverse."""
import numpy as np
import pytest
import tessera as ts

from tests.device.rocm.test_floating_scaled_adjoint import inputs


def floating_python_source(ta=False, tb=False):
    a_axes = '"K", "M"' if ta else '"M", "K"'
    b_axes = '"N", "K"' if tb else '"K", "N"'
    return f'''def floating(a: ts.Tensor[{a_axes}, "fp32"],
             b: ts.Tensor[{b_axes}, "fp32"],
             sa: ts.Tensor["M", "G", "fp32"],
             sb: ts.Tensor["G", "C", "fp32"]):
    return ts.ops.scaled_matmul(a, b, sa, sb,
        transposeA={ta}, transposeB={tb},
        numeric_policy={{"accum":"fp32","execution_mode":"exact_per_block"}},
        scale_layout={{"granularity":"block","block":[4,4],"format":"fp32"}})
'''


def floating_owner(ta=False, tb=False, roles=("a", "b", "sa", "sb")):
    return ts.from_text(floating_python_source(ta, tb), target="rocm_gfx1201",
                        autodiff="reverse", wrt=roles)


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_public_continuous_reverse_retains_typed_tracer_graph(ta, tb):
    values = inputs(ta, tb)
    owner = floating_owner(ta, tb)
    certificate = owner.frontend_differential(*values[:4])
    assert certificate is owner.frontend_differential(*values[:4])
    graph = owner._traced_autodiff_module(values[:4], {})
    function = graph.functions[0]
    assert [arg.ir_type.dtype for arg in function.args] == ["fp32"] * 4
    assert tuple(function.result_types[0].shape) == ("2", "5")
    assert function.body[0].kwargs["transposeA"] == ta
    assert function.body[0].kwargs["transposeB"] == tb
    # Compare the semantic certificate's eager function with an independent
    # direct block-product oracle. This is frontend proof, not GPU evidence.
    a, b, sa, sb, _ = values
    a = a.T if ta else a
    b = b.T if tb else b
    expected = np.zeros((2, 5), np.float64)
    for group in range(3):
        sl = slice(group * 4, min(group * 4 + 4, 9))
        expected += (a[:, sl].astype(np.float64) @ b[sl].astype(np.float64)) * (
            sa[:, group, None] * sb[group, np.arange(5)//4][None])
    np.testing.assert_allclose(owner._fn(*values[:4]), expected, rtol=4e-5, atol=3e-6)


@pytest.mark.parametrize("matrix_slot", [0, 1])
def test_public_continuous_semantics_reject_mixed_matrix_storage(matrix_slot):
    values = list(inputs(False, False)[:4])
    values[matrix_slot] = values[matrix_slot].astype(np.float16)
    with pytest.raises(ValueError, match="E4M3FN or f32"):
        floating_owner()._fn(*values)


def test_continuous_graph_admission_does_not_claim_a_primal_kernel():
    from tessera.compiler.capabilities import supports_op, get_target_capability
    from tessera.compiler.rocm_typed_scaled_native import supports_typed_scaled
    cap = get_target_capability("rocm_gfx1201").op("scaled_matmul")
    assert "fp32" not in cap.dtypes and "fp32" in cap.graph_only_dtypes
    result = supports_op("rocm_gfx1201", "scaled_matmul", dtype="fp32", rank=2)
    assert result.supported and result.runtime_status == "artifact_only"
    owner = floating_owner()
    assert not supports_typed_scaled(owner.graph_ir)


def floating_batch_owner(policy, ta=False, tb=False, roles=("a", "b", "sa", "sb")):
    from tests.unit.test_native_floating_scaled_adjoint import floating_batch_shapes
    shapes = floating_batch_shapes(policy, ta, tb)
    annotations = [", ".join([*map(str, shape), '"fp32"']) for shape in shapes]
    names = ("a", "b", "sa", "sb")
    args = ", ".join(f"{name}: ts.Tensor[{annotation}]" for name, annotation
                     in zip(names, annotations, strict=True))
    source = f'''def floating_batch({args}):
    return ts.ops.scaled_matmul(a, b, sa, sb, batching="{policy}",
        transposeA={ta}, transposeB={tb},
        numeric_policy={{"accum":"fp32","execution_mode":"exact_per_block"}},
        scale_layout={{"granularity":"block","block":[4,4],"format":"fp32"}})
'''
    return ts.from_text(source, target="rocm_gfx1201", autodiff="reverse", wrt=roles)


@pytest.mark.parametrize("policy", ["shared_lhs", "shared_rhs_rows", "independent_rhs", "broadcast"])
@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_public_batched_continuous_reverse_traces_independent_prefixes(policy, ta, tb):
    from tests.device.rocm.test_floating_scaled_adjoint import batch_inputs
    values = batch_inputs(policy, ta, tb)
    owner = floating_batch_owner(policy, ta, tb)
    owner.frontend_differential(*values[:4])
    graph = owner._traced_autodiff_module(values[:4], {})
    function = graph.functions[0]
    assert tuple(function.result_types[0].shape) == ("2", "3", "2", "5")
    assert function.body[0].kwargs["batching"] == policy
    for arg, value in zip(function.args, values[:4], strict=True):
        assert tuple(arg.ir_type.shape) == tuple(map(str, value.shape))

def test_dtype_flow_reports_graph_legality_without_primal_kernel_promotion():
    from tessera.compiler.dtype_flow_audit import (
        REPORT_TARGETS, _capability_target_state, _target_state)
    from tessera.compiler.primitive_coverage import all_primitive_coverages
    assert "rocm_gfx1201" in REPORT_TARGETS
    state = _capability_target_state("tessera.scaled_matmul", "rocm_gfx1201", "fp32")
    assert state.status == "legal_only"
    coverage = all_primitive_coverages()["scaled_matmul"]
    physical = _target_state(coverage, "scaled_matmul", "tessera.scaled_matmul", "rocm_gfx1201", "fp32")
    assert physical.status == "unsupported"
    assert "dtype_absent" in physical.source


def test_graph_only_storage_is_canonical_and_cannot_be_claimed_as_primal_ready():
    from tessera.dtype import canonicalize_dtype
    from tessera.compiler.capabilities import TARGET_CAPABILITIES, supports_op
    for target in TARGET_CAPABILITIES.values():
        for op in target.supported_ops.values():
            assert len(set(op.graph_only_dtypes)) == len(op.graph_only_dtypes)
            assert not set(op.graph_only_dtypes).intersection(op.dtypes)
            for dtype in op.graph_only_dtypes:
                assert canonicalize_dtype(dtype) == dtype
                result = supports_op(target.name, op.op_name, dtype=dtype, rank=op.min_rank)
                assert result.supported and result.runtime_status == "artifact_only"
