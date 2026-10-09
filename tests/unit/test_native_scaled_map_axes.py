"""Non-leading scaled maps retain storage, scalar bounds, and adjoint axes."""
import copy

import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.native_vmap import normalize_mixed_batch_inputs, restore_mapped_gradient
from tests.unit.test_native_typed_scaled_vmap import case


def axis_case(fmt="fp32", nk=False, mode=None, nested=False, cartesian=False):
    scalar, _, flat, expected = case("independent_rhs", fmt, nk, (6 if nested else 3, 3, 5, 256))
    if mode is not None:
        scalar = ts.jit(target="rocm_gfx1201", autodiff=mode, wrt=("sa", "sb"))(scalar._fn)
    if not nested:
        axes = (1, 2, -1, 1)
        values = tuple(np.moveaxis(value, 0, axis) for value, axis in zip(flat, axes, strict=True))
        return scalar, vmap(scalar, in_axes=axes), values, expected
    full = tuple(value.reshape(2, 3, *value.shape[1:]) for value in flat)
    if cartesian:
        inner_axes = (None, 1, None, -1)
        outer_axes = (1, None, -1, None)
        canonical = tuple(value[:, 0] if role in (0, 2) else value[0]
                          for role, value in enumerate(full))
        # Independent cartesian float64 oracle, with A/sa shared along inner.
        a, b, sa, sb = canonical
        b = b.swapaxes(-1, -2) if nk else b
        scale_a = sa.astype(np.float64) if fmt == "fp32" else np.exp2(sa.astype(np.float64) - 127)
        scale_b = sb.astype(np.float64) if fmt == "fp32" else np.exp2(sb.astype(np.float64) - 127)
        sk, sn = (128, 128) if fmt == "fp32" else (32, 1)
        expected = np.zeros((2, 3, 3, 5), np.float64)
        for group in range(256 // sk):
            expected += (a[:, None, ..., group*sk:(group+1)*sk].astype(np.float64) @
                         b[None, ..., group*sk:(group+1)*sk, :].astype(np.float64)) * (
                scale_a[:, None, ..., group, None] *
                scale_b[None, ..., group, np.arange(5)//sn][..., None, :])
        values = tuple(np.moveaxis(value, 0, outer_axes[role] if outer_axes[role] is not None
                                   else inner_axes[role]) for role, value in enumerate(canonical))
    else:
        inner_axes = (1, -1, 0, 2)
        outer_axes = (2, 0, -1, 1)
        values = tuple(np.stack([np.moveaxis(plane, 0, inner_axes[role]) for plane in value],
                                axis=outer_axes[role]) for role, value in enumerate(full))
        expected = expected.reshape(2, 3, 3, 5)
    owner = vmap(vmap(scalar, in_axes=inner_axes), in_axes=outer_axes)
    return scalar, owner, values, expected


@pytest.mark.parametrize("fmt", ["fp32", "e8m0"])
@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("profile", ["single", "nested", "cartesian"])
def test_axis_projection_certifies_native_graph_and_preserves_storage(fmt, nk, profile):
    scalar, owner, values, expected = axis_case(fmt, nk, nested=profile != "single",
                                               cartesian=profile == "cartesian")
    before = copy.deepcopy(scalar.graph_ir)
    normalized = owner._ordered_inputs(values, {})
    for role, (source, view) in enumerate(zip(values, normalized, strict=True)):
        assert np.shares_memory(source, view)
        assert source.ctypes.data == view.ctypes.data
        np.testing.assert_array_equal(
            restore_mapped_gradient(view, source, owner._frontend_batch_policies, role), source)
    graph = owner._specialized_autodiff_module(values, {})
    assert graph.functions[0].body[0].kwargs["batching"] == "broadcast"
    assert tuple(map(int, graph.functions[0].result_types[0].shape)) == expected.shape
    from tessera.compiler.reference_typed_scaled_matmul import reference_typed_scaled_matmul
    np.testing.assert_allclose(reference_typed_scaled_matmul(
        *normalized, **graph.functions[0].body[0].kwargs), expected, rtol=4e-5, atol=2e-5)
    assert owner.frontend_differential(*values)
    assert scalar.graph_ir == before


@pytest.mark.parametrize("axis", [3, -4, True, 1.5])
def test_axis_bounds_refused_at_owner_construction(axis):
    scalar, _, _, _ = case("independent_rhs", "fp32", False)
    with pytest.raises(ValueError, match="axis|axes"):
        vmap(scalar, in_axes=(axis, 0, 0, 0))


def test_wrong_extent_and_rank_refused_before_compilation():
    _, owner, values, _ = axis_case()
    wrong = list(values)
    wrong[1] = wrong[1][..., :2]
    with pytest.raises(ValueError, match="extents differ"):
        owner._ordered_inputs(wrong, {})
    with pytest.raises(ValueError, match="rank"):
        normalize_mixed_batch_inputs([value[0] for value in values],
                                     owner._frontend_batch_policies)


@pytest.mark.parametrize("mode", [None, "forward", "reverse"])
def test_axis_projection_preserves_scalar_dimension_constraints(mode, monkeypatch):
    from tessera.compiler.constraints import Range, TesseraConstraintError
    scalar, _, values, _ = axis_case(mode=mode)
    scalar.constraints.add(Range("M", 1, 2))
    owner = vmap(scalar, in_axes=(1, 2, -1, 1))
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid scalar bound reached frontend/native compilation")
    monkeypatch.setattr(owner, "_specialized_autodiff_module", forbidden)
    monkeypatch.setattr(owner, "_try_native_descriptor_call", forbidden)
    with pytest.raises(TesseraConstraintError, match="M"):
        if mode == "forward":
            owner.native_jvp(*values, tangents=(values[2], values[3]))
        elif mode == "reverse":
            owner.native_backward(*values, out_cotangents=np.ones((3, 3, 5), np.float32))
        else:
            owner(*values)
