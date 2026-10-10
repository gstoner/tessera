"""Scaled JVP binds alias frames; unproved views fail before eager certification."""
import numpy as np
import pytest

from tessera._jit_boundary import TesseraJitError
from tests.unit.test_composed_scaled_jvp import case


@pytest.mark.parametrize("operand", [0, 2, "seed"])
@pytest.mark.parametrize("stride_kind", ["compact", "padded", "negative", "zero"])
def test_unproved_scaled_jvp_storage_fails_before_frontend(operand, stride_kind, monkeypatch):
    owner, values, seeds = case((3, 5, 256), wrt=("sa0",))
    values, seeds = list(values), list(seeds)
    original = seeds[0] if operand == "seed" else values[operand]
    backing = np.zeros(1, dtype=original.dtype)
    strides = list(original.strides)
    if stride_kind == "padded":
        strides[0] *= 2
    elif stride_kind == "negative":
        strides[-1] *= -1
    elif stride_kind == "zero":
        strides[-1] = 0
    forged = np.lib.stride_tricks.as_strided(backing, shape=original.shape,
                                            strides=tuple(strides))
    if operand == "seed":
        seeds[0] = forged
    else:
        values[operand] = forged
    def forbidden(*args, **kwargs):
        raise AssertionError("unproved storage reached frontend evaluation")
    monkeypatch.setattr(owner, "_specialized_autodiff_module", forbidden)
    monkeypatch.setattr(owner, "frontend_differential", forbidden)
    with pytest.raises(ValueError, match="backing allocation|whole-element strides"):
        owner.native_jvp(*values, tangents=seeds)


def test_scaled_jvp_does_not_implicitly_cast_tangent_storage(monkeypatch):
    owner, values, seeds = case((3, 5, 256), wrt=("sa0",))
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid seed dtype reached frontend evaluation")
    monkeypatch.setattr(owner, "_specialized_autodiff_module", forbidden)
    with pytest.raises(TesseraJitError, match="fp32 storage"):
        owner.native_jvp(*values, tangents=(seeds[0].astype(np.float16),))


def test_scaled_jvp_planner_receives_borrowed_strided_frame(monkeypatch):
    owner, values, seeds = case((3, 5, 256), wrt=("sa0", "sb1"))
    def padded(value):
        storage = np.empty((*value.shape[:-1], value.shape[-1]*2), value.dtype)
        view = storage[..., ::2]
        view[...] = value
        return view
    values, seeds = tuple(map(padded, values)), tuple(map(padded, seeds))
    monkeypatch.setenv("TESSERA_ROCM_CHIP", "gfx1201")
    monkeypatch.setattr(owner, "_compile_jvp_module", lambda *args: "host-binding-intercept")
    from tessera.compiler import native_jvp_plugins
    class Intercepted(Exception):
        pass
    def planner(**kwargs):
        for source, passed in zip(values, kwargs["primal_inputs"], strict=True):
            assert np.shares_memory(source, passed)
            assert source.strides == passed.strides
        raise Intercepted
    monkeypatch.setattr(native_jvp_plugins, "build_native_jvp_family_artifact", planner)
    def forbidden(*args, **kwargs):
        raise AssertionError("scaled JVP binding compacted a tensor in Python")
    # Eager frontend certification is an independent oracle, not production packing.
    owner.frontend_differential(*values)
    monkeypatch.setattr(np, "ascontiguousarray", forbidden)
    with pytest.raises(Intercepted):
        owner.native_jvp(*values, tangents=seeds)
