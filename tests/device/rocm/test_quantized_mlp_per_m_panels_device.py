"""Owning gfx1201 replay of per-active-M product images.

GFX1201-BOUNDED-PER-M-PANEL-20261030. An M-only bounded program at capacity
256 x (H 2048) carries a register gate image for M >= 1 and the static rule's
LDS images on 97..128 and 129..256. One retained owner serves every M; each
output is bitwise equal to the static program at that M, and the HIP owner
reports the image it selected.
"""
import subprocess

import numpy as np
import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tests.device.rocm.test_native_quantized_mlp_bounded_rows import provider
from tests.device.rocm.test_public_quantized_mlp_k_bound_device import static_at

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_rocm, pytest.mark.compiler_route]

H, K, N = 2048, 4096, 256
# Active M -> the gate image the owner must select (0 = base register panel).
SELECTION = {256: 2, 16: 0, 100: 1, 128: 1, 129: 2, 96: 0, 1: 0}


def test_one_owner_switches_gate_images_by_active_m():
    lib = provider()
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(256, H, K, N))
    native = fn.compile_native_quantized_program(*inputs(256, H, K, N), m_bound=256).native
    owner = native.prepare(inputs(256, H, K, N), runtime_library=lib)
    try:
        for index, (m, image) in enumerate(SELECTION.items()):
            roots = inputs(m, H, K, N, seed=1800 + index)
            owner.update(roots)
            generation, _ = owner.invoke()
            out = owner.read(generation)[0]
            assert owner.variant_info() == (0, 0, image, 0, 0, 0, 0), m
            np.testing.assert_array_equal(out.view(np.uint16), static_at(m, H, K, N, roots, lib).view(np.uint16))
    finally:
        owner.close()


def test_public_calls_bind_one_owner_across_the_ranges(monkeypatch):
    from tessera.compiler.rocm_quantized_mlp_program import NativeQuantizedMLPProgram
    lib = provider()
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(256, H, K, N))
    program = fn.compile_native_quantized_program(*inputs(256, H, K, N), m_bound=256)
    prepares = []
    original = NativeQuantizedMLPProgram.prepare

    def counted(self, values, *, runtime_library):
        if self is program.native:
            prepares.append(values[0].shape)
        return original(self, values, runtime_library=runtime_library)
    monkeypatch.setattr(NativeQuantizedMLPProgram, "prepare", counted)

    def forbidden(*a, **kw):
        pytest.fail("compiler or tracer ran during bounded public replay")
    roots = inputs(256, H, K, N, seed=1900)
    for m in (256, 16, 100, 200):
        x = inputs(m, H, K, N, seed=1900 + m)[0]
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "run", forbidden)
            patch.setattr(fn, "_trace_frontend_capture", forbidden)
            actual = fn(x, *roots[1:])
        assert fn._rocm_quantized_product_last_program is program
        np.testing.assert_array_equal(actual.view(np.uint16),
                                      static_at(m, H, K, N, [x, *roots[1:]], lib).view(np.uint16))
    assert len(prepares) == 1


@pytest.mark.parametrize("tamper", ["grid_tile", "overlap", "unframed_step"])
def test_native_variant_admission_refuses_tampered_variants(tamper, monkeypatch):
    from tessera.compiler import rocm_quantized_mlp_program as module
    lib = provider()
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(256, H, K, N))
    native = fn.compile_native_quantized_program(*inputs(256, H, K, N), m_bound=256).native
    original = module.PreparedQuantizedMLPProgram._variant_array

    def array(self, entries):
        rows = original(self, entries)
        if tamper == "grid_tile":
            rows[0].entries[0].grid_tile = 16          # no longer reproduces its geometry
        if tamper == "overlap":
            rows[1].min_rows = rows[0].max_rows        # ranges must be disjoint
        if tamper == "unframed_step":
            rows[0].step = 5                           # entries name another step
        return rows
    monkeypatch.setattr(module.PreparedQuantizedMLPProgram, "_variant_array", array)
    with pytest.raises(RuntimeError, match="status 1"):
        native.prepare(inputs(256, H, K, N), runtime_library=lib)


def test_h_bounded_owner_runs_masked_column_lds_variants():
    """A runtime-N gate product's LDS variants run an N that ends mid-tile."""
    lib = provider()
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(16, 512, K, N))
    native = fn.compile_native_quantized_program(*inputs(16, 512, K, N), m_bound=256, h_bound=2048).native
    owner = native.prepare(inputs(16, 512, K, N), runtime_library=lib)
    try:
        # Boxes over (M, H/16): 1 = 97..127 x 32..128, 3 = 129..256 x 63..124,
        # 4 = 129..256 x 125..128; 256 x 512 (N = 1024) stays on the base image.
        for index, (m, h, image) in enumerate(((200, 1008, 3), (100, 1040, 1), (16, 2048, 0),
                                               (256, 2048, 4), (256, 512, 0))):
            roots = inputs(m, h, K, N, seed=2000 + index)
            owner.update(roots)
            generation, _ = owner.invoke()
            assert owner.variant_info()[2] == image, (m, h)
            np.testing.assert_array_equal(owner.read(generation)[0].view(np.uint16),
                                          static_at(m, h, K, N, roots, lib).view(np.uint16))
    finally:
        owner.close()
