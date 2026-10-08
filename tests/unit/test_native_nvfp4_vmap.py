"""Frontend batch metadata validation without packed-data conversion."""
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.native_vmap import batch_specs, project_batch
from tessera.compiler.nvfp4_tensor import NVFP4Tensor
from tessera.compiler.trace import trace, to_graph_ir_module


def product(a, b, sa, sb):
    return ts.ops.scaled_matmul(a, b, sa, sb,
        physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
        numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
        scale_layout={"granularity": "block", "block": [1, 16], "format": "ue4m3"})


def operands(independent=False):
    a = NVFP4Tensor(np.zeros((3, 7, 16), np.uint8), (3, 7, 31), 2)
    b = (NVFP4Tensor(np.zeros((3, 16, 5), np.uint8), (3, 31, 5), 1) if independent
         else NVFP4Tensor(np.zeros((16, 5), np.uint8), (31, 5), 0))
    sa = np.ones((3, 7, 2), np.uint8)
    sb = np.ones((3, 2, 5) if independent else (2, 5), np.uint8)
    return a, b, sa, sb


@pytest.mark.parametrize("independent", [False, True])
def test_batch_projection_preserves_scalar_graph_and_logical_dimensions(independent):
    values = operands(independent)
    axes = (0, 0, 0, 0) if independent else (0, None, 0, None)
    traced = trace(product, *batch_specs(values, axes))
    module = to_graph_ir_module(traced, name="product", target="nvidia_sm120")
    before = module.to_mlir(target="nvidia_sm120")
    projected = project_batch(module, values, axes)
    assert module.to_mlir(target="nvidia_sm120") == before
    assert projected.functions[0].args[0].ir_type.shape == ("3", "7", "31")
    assert projected.functions[0].result_types[0].shape == ("3", "7", "5")
    assert projected.functions[0].body[0].kwargs["batching"] == (
        "independent_rhs" if independent else "shared_rhs_rows")


@pytest.mark.parametrize("axes,out_axes", [((0, None, None, None), 0), (0, 1), (True, 0), ((0, None, 0, None), None)])
def test_native_vmap_rejects_unsupported_axes(axes, out_axes):
    scalar = ts.jit(product, target="nvidia_sm120")
    with pytest.raises(ValueError, match="native NVFP4 vmap"):
        vmap(scalar, in_axes=axes, out_axes=out_axes)


def test_batch_extent_mismatch_is_rejected_without_slicing():
    values = list(operands())
    values[2] = np.ones((4, 7, 2), np.uint8)
    with pytest.raises(ValueError, match="batch extents differ"):
        batch_specs(values, (0, None, 0, None))


MK = ts.Tensor["M", "K"]
KN = ts.Tensor["K", "N"]
MS = ts.Tensor["M", "S"]
SN = ts.Tensor["S", "N"]


def typed_product(a: MK, b: KN, sa: MS, sb: SN):
    return ts.ops.scaled_matmul(a, b, sa, sb,
        physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
        numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
        scale_layout={"granularity": "block", "block": [1, 16], "format": "ue4m3"})


def test_independent_batch_preserves_symbolic_constraints():
    from tessera.compiler.constraints import Range, TesseraConstraintError
    scalar = ts.jit(typed_product, target="nvidia_sm120")
    scalar.constraints.add(Range("M", 1, 6))
    owner = vmap(scalar, in_axes=0)
    with pytest.raises(TesseraConstraintError, match="M"):
        owner._enforce_call_time_constraints(operands(True), {})
    assert scalar._constraint_ir_args[0].dim_names == ("M", "K")


@pytest.mark.parametrize("axes", [None, (None, None, None, None)])
def test_unmapped_native_product_preserves_noop_vmap_semantics(axes):
    scalar = ts.jit(product, target="nvidia_sm120")
    assert vmap(scalar, in_axes=axes, out_axes=None) is scalar


def test_shared_lhs_batch_projection_preserves_scalar_owner():
    values = (NVFP4Tensor(np.zeros((7, 16), np.uint8), (7, 31), 1),
              operands(True)[1], np.ones((7, 2), np.uint8), operands(True)[3])
    axes = (None, 0, None, 0)
    traced = trace(product, *batch_specs(values, axes))
    module = to_graph_ir_module(traced, name="product", target="nvidia_sm120")
    before = module.to_mlir(target="nvidia_sm120")
    projected = project_batch(module, values, axes)
    assert module.to_mlir(target="nvidia_sm120") == before
    assert projected.functions[0].args[0].ir_type.shape == ("7", "31")
    assert projected.functions[0].args[1].ir_type.shape == ("3", "31", "5")
    assert projected.functions[0].result_types[0].shape == ("3", "7", "5")
    assert projected.functions[0].body[0].kwargs["batching"] == "shared_lhs"


def test_shared_lhs_vmap_retains_rhs_symbolic_constraints():
    from tessera.compiler.constraints import Range, TesseraConstraintError
    scalar = ts.jit(typed_product, target="nvidia_sm120")
    scalar.constraints.add(Range("N", 1, 4))
    owner = vmap(scalar, in_axes=(None, 0, None, 0))
    values = (NVFP4Tensor(np.zeros((7, 16), np.uint8), (7, 31), 1),
              operands(True)[1], np.ones((7, 2), np.uint8), operands(True)[3])
    with pytest.raises(TesseraConstraintError, match="N"):
        owner._enforce_call_time_constraints(values, {})
    assert scalar._constraint_ir_args[1].dim_names == ("K", "N")
