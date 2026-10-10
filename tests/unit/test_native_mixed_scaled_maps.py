"""Mixed leading map policies preserve each native broadcast level."""
import copy
import numpy as np
import pytest
from tessera.autodiff import vmap
from tests.unit.test_native_typed_scaled_vmap import case
from tessera.compiler.native_vmap import normalize_mixed_batch_inputs


def mixed_case(fmt, nk, cartesian=False, mode=None):
    scalar, _, flat, _ = case("independent_rhs", fmt, nk, (6, 3, 5, 256))
    if mode is not None:
        import tessera as ts
        scalar = ts.jit(target="rocm_gfx1201", autodiff=mode, wrt=("sa", "sb"))(scalar._fn)
    full = tuple(value.reshape(2, 3, *value.shape[1:]) for value in flat)
    inner_axes = (None, 0, None, 0)
    outer_axes = (0, None, 0, None) if cartesian else (0, 0, 0, 0)
    inner = vmap(scalar, in_axes=inner_axes)
    owner = vmap(inner, in_axes=outer_axes)
    values = tuple(value[:, 0] if role in (0, 2) else
                   value[0] if cartesian else value
                   for role, value in enumerate(full))
    values = tuple(np.ascontiguousarray(value) for value in values)
    expected = mixed_oracle(values, owner._frontend_batch_policies, fmt, nk)
    return scalar, inner, owner, values, expected


@pytest.mark.parametrize("fmt", ["fp32", "e8m0"])
@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("cartesian", [False, True])
def test_mixed_nested_projection_keeps_alias_storage_and_scalar_owners(fmt, nk, cartesian):
    scalar, inner, owner, values, expected = mixed_case(fmt, nk, cartesian)
    before = copy.deepcopy((scalar.graph_ir, inner.graph_ir))
    normalized = owner._ordered_inputs(values, {})
    for source, view in zip(values, normalized, strict=True):
        assert np.shares_memory(source, view)
        assert source.ctypes.data == view.ctypes.data
    graph = owner._specialized_autodiff_module(values, {})
    assert graph.functions[0].body[0].kwargs["batching"] == "broadcast"
    assert graph.functions[0].result_types[0].shape == ("2", "3", "3", "5")
    assert owner.frontend_differential(*values)
    assert (scalar.graph_ir, inner.graph_ir) == before
    assert expected.shape == (2, 3, 3, 5)


def test_mixed_map_extent_mismatch_is_rejected_before_capture():
    _, _, owner, values, _ = mixed_case("fp32", False)
    changed = (values[0], values[1][:1], values[2], values[3][:1])
    with pytest.raises(ValueError, match="extents differ"):
        owner._ordered_inputs(changed, {})


def test_mixed_map_source_constraint_checks_raw_outer_axes(monkeypatch):
    from tessera.compiler.constraints import Range, TesseraConstraintError
    scalar, _, _, values, _ = mixed_case("fp32", False)
    scalar.constraints.add(Range("M", 1, 2))
    owner = vmap(vmap(scalar, in_axes=(None, 0, None, 0)), in_axes=(0, 0, 0, 0))
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid scalar bound reached compiler")
    monkeypatch.setattr(owner, "_try_native_descriptor_call", forbidden)
    with pytest.raises(TesseraConstraintError, match="M"):
        owner(*values)


def mixed_oracle(values, policies, fmt, nk):
    normalized = normalize_mixed_batch_inputs(values, policies)
    a, b, sa, sb = normalized
    b = b.swapaxes(-1, -2) if nk else b
    sa = sa.astype(np.float64) if fmt == "fp32" else np.exp2(sa.astype(np.float64) - 127)
    sb = sb.astype(np.float64) if fmt == "fp32" else np.exp2(sb.astype(np.float64) - 127)
    sk, sn = (128, 128) if fmt == "fp32" else (32, 1)
    expected = np.zeros((2, 3, 3, 5), np.float64)
    for group in range(256 // sk):
        expected += (a[..., group*sk:(group+1)*sk].astype(np.float64) @
                     b[..., group*sk:(group+1)*sk, :].astype(np.float64)) * (
                         sa[..., group, None] * sb[..., group, np.arange(5)//sn][..., None, :])
    return expected
