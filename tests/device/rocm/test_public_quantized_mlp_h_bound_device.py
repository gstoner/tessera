"""Owning gfx1201 replay of the hidden-width-bounded native quantized MLP.

GFX1201-BOUNDED-QUANTIZED-H-20261029. One explicit
``compile_native_quantized_program(..., m_bound=MB, h_bound=HB)`` call; every
later public call inside the bounds binds that one capacity program and its
retained owner without retracing or compiling. Every output is bitwise equal
to the static program compiled at that exact shape, and the small shape also
matches the independent staged oracle.
"""
import subprocess

import numpy as np
import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
from tests.device.rocm.test_native_quantized_mlp_bounded_rows import provider
from tests.device.rocm.test_public_quantized_mlp_k_bound_device import _REPLAY, static_at
from tests.device.rocm.test_public_quantized_mlp_n_bound_device import _raw_update
from tests.unit.test_native_quantized_mlp_program import source
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_rocm, pytest.mark.compiler_route]

# Traced (M, H, K, N), bounds, active (M, H) sequence at the traced K and N.
SHAPES = [
    ((3, 32, 48, 19), {"m_bound": 8, "h_bound": 64},
     ((3, 32), (8, 64), (1, 16), (5, 48), (2, 32))),
    ((37, 128, 4096, 256), {"m_bound": 64, "h_bound": 512},
     ((37, 128), (64, 512), (1, 16), (16, 304))),
    # GFX1201-BOUNDED-GLOBAL-PANEL-20261030: a capacity whose static rule takes
    # an LDS gate panel (256 x 4096); the runtime-N/H image is global.
    # GFX1201-BOUNDED-PER-M-PANEL-20261030: M = 100..256 run the LDS variants,
    # including an N that ends mid-tile (2H = 2016, 2080).
    ((16, 512, 4096, 256), {"m_bound": 256, "h_bound": 2048},
     ((16, 512), (256, 2048), (100, 1024), (200, 1008), (128, 1040))),
]


@pytest.mark.parametrize("traced,bounds,actives", SHAPES)
def test_public_h_bound_matches_static_programs_with_one_prepare(traced, bounds, actives, monkeypatch):
    from tessera.compiler.rocm_quantized_mlp_program import NativeQuantizedMLPProgram
    lib = provider()
    m0, h0, k, n = traced
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(m0, h0, k, n))
    program = fn.compile_native_quantized_program(*inputs(m0, h0, k, n), **bounds)
    assert program.native.h_bounded
    public = program.native
    prepares = []
    original = NativeQuantizedMLPProgram.prepare

    def counted(self, values, *, runtime_library):
        if self is public:
            prepares.append(values[1].shape)
        return original(self, values, runtime_library=runtime_library)
    monkeypatch.setattr(NativeQuantizedMLPProgram, "prepare", counted)

    def forbidden(*a, **kw):
        pytest.fail("compiler or tracer ran during bounded public replay")
    for index, (m, h) in enumerate(actives):
        roots = inputs(m, h, k, n, seed=1600 + index)
        before = [x.copy() for x in roots]
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "run", forbidden)
            patch.setattr(fn, "_trace_frontend_capture", forbidden)
            patch.setattr(fn, "_fn", forbidden)
            actual = fn(*roots)
        assert actual.shape == (m, n) and fn.execution_kind == "native_gpu"
        assert fn._rocm_quantized_product_last_program is program
        for root, held in zip(roots, before, strict=True):
            np.testing.assert_array_equal(root, held)
        expected = static_at(m, h, k, n, roots, lib)
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
        if k == 48:
            assert_compiled_chain(package_quantized_product_graph(source(m, h, k, n)), roots, actual, lib)
    assert prepares == [(2 * actives[0][1], k // 2)]
    report = fn.compile_report()
    assert report.target_decision["rocm_gfx1201"].startswith(
        "canonical_rocm_quantized_h_bounded_mlp_program; 7 ordered native packages")
    m, _ = actives[0]
    with pytest.raises(ValueError, match="H16 capacity"):
        fn(*inputs(m, bounds["h_bound"] + 16, k, n, seed=17))
    # Refusals kept the owner: the next call still binds it without a prepare.
    roots = inputs(m, 16, k, n, seed=19)
    np.testing.assert_array_equal(fn(*roots).view(np.uint16), static_at(m, 16, k, n, roots, lib).view(np.uint16))
    assert len(prepares) == 1


def _owner(lib):
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(3, 32, 48, 19))
    program = fn.compile_native_quantized_program(*inputs(3, 32, 48, 19), m_bound=8, h_bound=64).native
    return program, program.prepare(inputs(3, 32, 48, 19), runtime_library=lib)


def test_activation_only_update_keeps_the_resident_hidden_width():
    lib = provider()
    _, owner = _owner(lib)
    try:
        roots = inputs(6, 48, 48, 19, seed=31)
        owner.update(roots)
        assert (owner.active_m, owner.active_h) == (6, 48)
        x = inputs(4, 48, 48, 19, seed=32)[0]
        owner.update_activations(x)
        generation, _ = owner.invoke()
        out = owner.read(generation)[0]
        assert out.shape == (4, 19)
        np.testing.assert_array_equal(out.view(np.uint16),
                                      static_at(4, 48, 48, 19, [x, *roots[1:]], lib).view(np.uint16))
        assert owner.axes_info()[:2] == ((8, 0, 0, 64), (4, 0, 0, 48))
    finally:
        owner.close()


def test_native_hidden_update_admission():
    """The HIP owner re-checks every variable change against the axis frame."""
    lib = provider()
    _, owner = _owner(lib)
    try:
        roots = [np.ascontiguousarray(r) for r in inputs(3, 48, 48, 19)]
        # An H change needs all four H-dependent weight planes.
        assert _raw_update(owner, 3, 0, 0, (1, 2), (roots[1], roots[2]), h=3) == 1
        # H beyond capacity; and N is not framed here, so any N is refused.
        assert _raw_update(owner, 3, 0, 0, (1, 2, 4, 5), [roots[i] for i in (1, 2, 4, 5)], h=5) == 1
        assert _raw_update(owner, 3, 0, 19, (0,), (roots[0],)) == 1
        assert owner.axes_info()[1] == (3, 0, 0, 32)
        assert _raw_update(owner, 3, 0, 0, (1, 2, 4, 5), [roots[i] for i in (1, 2, 4, 5)], h=3) == 0
        assert owner.axes_info()[1] == (3, 0, 0, 48)
        generation, _ = owner.invoke()
        out = owner.read(generation)[0]
        x = inputs(3, 32, 48, 19)[0]
        np.testing.assert_array_equal(out.view(np.uint16),
                                      static_at(3, 48, 48, 19, [x, *roots[1:]], lib).view(np.uint16))
    finally:
        owner.close()


@pytest.mark.parametrize("tamper", ["silu_width_coef", "power_on_unframed_n", "duplicate_scalar"])
def test_native_hidden_frame_admission_refuses_tampered_frames(tamper, monkeypatch):
    from tessera.compiler import rocm_quantized_mlp_program as module
    lib = provider()
    program, owner = _owner(lib)
    owner.close()
    original = module._hidden_steps

    def steps(*args):
        entries = original(*args)
        if tamper == "silu_width_coef":
            entries[3] = module._ShapeStep(3, 1, 32, 8, 3, 0, 0)     # width 2H != capacity H
        if tamper == "power_on_unframed_n":
            entries[5] = module._ShapeStep(5, 0, 1, 12, 0, 256, 1)   # N is not framed
        if tamper == "duplicate_scalar":
            entries.append(module._ShapeStep(3, 0, 16, 9, 3, 0, 0))  # second entry for one scalar
        return entries
    monkeypatch.setattr(module, "_hidden_steps", steps)
    with pytest.raises(RuntimeError, match="status 1"):
        program.prepare(inputs(3, 32, 48, 19), runtime_library=lib)


def test_h_bounded_artifact_replays_in_a_fresh_process(tmp_path):
    import json
    import os
    import sys
    from tests.device.rocm.test_public_quantized_mlp_row_bound_device import _save_roots
    provider()
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(3, 32, 48, 19))
    fn.compile_native_quantized_program(*inputs(3, 32, 48, 19), m_bound=8, h_bound=64)
    expected = {}
    for index, (m, h) in enumerate(((3, 32), (8, 64), (1, 16), (5, 48))):
        roots = inputs(m, h, 48, 19, seed=1700 + index)
        name = f"roots-{index}.npz"
        _save_roots(tmp_path / name, roots)
        expected[name] = fn(*roots)
    (tmp_path / "artifact.json").write_text(fn.runtime_artifact().to_json())
    run = subprocess.run([sys.executable, "-c", _REPLAY, str(tmp_path)], capture_output=True, text=True,
                         env=dict(os.environ))
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert all(entry["ok"] for entry in report.values()), report
    for name, want in expected.items():
        np.testing.assert_array_equal(np.load(tmp_path / (name + ".npy")).view(np.uint16), want.view(np.uint16))
