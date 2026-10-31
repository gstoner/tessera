"""Owning gfx1201/gfx1151 replay of the column-bounded native quantized MLP.

GFX1201-BOUNDED-QUANTIZED-N-20261029. One explicit
``compile_native_quantized_program(..., m_bound=MB, n_bound=NB[, k_bound=KB])``
call; every later public call inside the bounds binds that one capacity
program and its retained owner without retracing or compiling. Every output
is bitwise equal to the static program compiled at that exact shape, and the
small shape also matches the independent staged oracle.
"""
import subprocess

import numpy as np
import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
from tests.device.rocm.test_native_quantized_mlp_bounded_rows import owning_target, provider, source
from tests.device.rocm.test_public_quantized_mlp_k_bound_device import _REPLAY, static_at
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_rocm, pytest.mark.compiler_route]

# H, traced (M, K, N), bounds, active (M, K, N) sequence.
SHAPES = [
    (32, (3, 48, 19), {"m_bound": 8, "n_bound": 40},
     ((3, 48, 19), (8, 48, 40), (1, 48, 1), (5, 48, 33), (2, 48, 19))),
    (32, (3, 48, 19), {"m_bound": 8, "k_bound": 96, "n_bound": 40},
     ((3, 48, 19), (8, 96, 40), (1, 16, 1), (5, 80, 33), (4, 32, 7))),
    (128, (37, 4096, 256), {"m_bound": 64, "k_bound": 12288, "n_bound": 512},
     ((37, 4096, 256), (64, 12288, 512), (1, 16, 1), (16, 8192, 300))),
]


@pytest.mark.parametrize("h,traced,bounds,actives", SHAPES)
def test_public_n_bound_matches_static_programs_with_one_prepare(h, traced, bounds, actives, monkeypatch):
    from tessera.compiler.rocm_quantized_mlp_program import NativeQuantizedMLPProgram
    lib = provider()
    m0, k0, n0 = traced
    fn = ts.jit(target=owning_target(), native_required=True)(make_mlp(m0, h, k0, n0))
    program = fn.compile_native_quantized_program(*inputs(m0, h, k0, n0), **bounds)
    assert program.native.n_bounded and program.native.k_bounded == ("k_bound" in bounds)
    public = program.native
    prepares = []
    original = NativeQuantizedMLPProgram.prepare

    def counted(self, values, *, runtime_library):
        if self is public:
            prepares.append(values[0].shape)
        return original(self, values, runtime_library=runtime_library)
    monkeypatch.setattr(NativeQuantizedMLPProgram, "prepare", counted)

    def forbidden(*a, **kw):
        pytest.fail("compiler or tracer ran during bounded public replay")
    for index, (m, k, n) in enumerate(actives):
        roots = inputs(m, h, k, n, seed=1400 + index)
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
        if h == 32:
            assert_compiled_chain(package_quantized_product_graph(source(m, h, k, n)), roots, actual, lib)
    assert prepares == [actives[0][:2]]
    report = fn.compile_report()
    assert report.target_decision[owning_target()].startswith(
        "canonical_rocm_quantized_n_bounded_mlp_program; 7 ordered native packages")
    m, k, _ = actives[0]
    with pytest.raises(ValueError, match="column capacity"):
        fn(*inputs(m, h, k, bounds["n_bound"] + 1, seed=17))
    # Refusals kept the owner: the next call still binds it without a prepare.
    roots = inputs(m, h, k, 2, seed=19)
    np.testing.assert_array_equal(fn(*roots).view(np.uint16), static_at(m, h, k, 2, roots, lib).view(np.uint16))
    assert len(prepares) == 1


def _owner(lib, *, h=32, **bounds):
    fn = ts.jit(target=owning_target(), native_required=True)(make_mlp(3, h, 48, 19))
    program = fn.compile_native_quantized_program(*inputs(3, h, 48, 19), m_bound=8, n_bound=40, **bounds).native
    return program, program.prepare(inputs(3, h, 48, 19), runtime_library=lib)


def test_activation_only_update_keeps_the_resident_columns():
    lib = provider()
    h = 32
    _, owner = _owner(lib)
    try:
        roots = inputs(6, h, 48, 33, seed=31)
        owner.update(roots)
        assert (owner.active_m, owner.active_n) == (6, 33)
        x = inputs(4, h, 48, 33, seed=32)[0]
        owner.update_activations(x)
        generation, _ = owner.invoke()
        out = owner.read(generation)[0]
        assert out.shape == (4, 33)
        np.testing.assert_array_equal(out.view(np.uint16),
                                      static_at(4, h, 48, 33, [x, *roots[1:]], lib).view(np.uint16))
        assert owner.axes_info()[:2] == ((8, 0, 40, 0), (4, 0, 33, 0))
    finally:
        owner.close()


def _raw_update(owner, m, g, n, slots, arrays, h=0):
    import ctypes as C
    update = owner.lib.tessera_rocm_program_update_axes
    update.argtypes = [C.c_uint64, C.c_int64, C.c_int64, C.c_int64, C.c_int64, C.c_uint32,
                       C.POINTER(C.c_uint32), C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
    update.restype = C.c_int
    count = len(slots)
    return update(owner.handle, m, g, n, h, count, (C.c_uint32*count)(*slots),
                  (C.c_void_p*count)(*[a.ctypes.data for a in arrays]),
                  (C.c_uint64*count)(*[a.nbytes for a in arrays]))


def test_native_axis_update_admission():
    """The HIP owner re-checks every variable change against the axis frame."""
    lib = provider()
    _, owner = _owner(lib)
    try:
        roots = [np.ascontiguousarray(r) for r in inputs(3, 32, 48, 33)]
        # An N change needs both down weight planes.
        assert _raw_update(owner, 3, 0, 33, (4,), (roots[4],)) == 1
        # N beyond capacity; and K is not framed here, so any G is refused.
        assert _raw_update(owner, 3, 0, 41, (4, 5), (roots[4], roots[5])) == 1
        assert _raw_update(owner, 3, 2, 19, (0,), (roots[0],)) == 1
        assert owner.axes_info()[1] == (3, 0, 19, 0)
        generation, _ = owner.invoke()
        assert owner.read(generation)[0].shape == (3, 19)
    finally:
        owner.close()


@pytest.mark.parametrize("tamper", ["drop_product_n", "expansion_coef", "power_on_unframed_k"])
def test_native_axis_frame_admission_refuses_tampered_frames(tamper, monkeypatch):
    from tessera.compiler import rocm_quantized_mlp_program as module
    lib = provider()
    program, owner = _owner(lib)
    owner.close()
    original = module._column_steps

    def steps(*args):
        entries = original(*args)
        if tamper == "drop_product_n":
            entries.pop(1)                                         # leaves N's step 6 scalar at capacity
            entries.append(module._ShapeStep(6, 1, 1, 1, 3, 0, 0))   # ... restated as an M scalar
        if tamper == "expansion_coef":
            entries[0] = module._ShapeStep(5, 0, 2, 4, 0, 256, 2)    # scalar 2N != capacity N
        if tamper == "power_on_unframed_k":
            entries[0] = module._ShapeStep(5, 0, 1, 6, 0, 256, 2)    # G is not framed
        return entries
    monkeypatch.setattr(module, "_column_steps", steps)
    with pytest.raises(RuntimeError, match="status 1"):
        program.prepare(inputs(3, 32, 48, 19), runtime_library=lib)


def test_n_bounded_artifact_replays_in_a_fresh_process(tmp_path):
    import json
    import os
    import sys
    from tests.device.rocm.test_public_quantized_mlp_row_bound_device import _save_roots
    provider()
    h = 32
    fn = ts.jit(target=owning_target(), native_required=True)(make_mlp(3, h, 48, 19))
    fn.compile_native_quantized_program(*inputs(3, h, 48, 19), m_bound=8, k_bound=96, n_bound=40)
    expected = {}
    for index, (m, k, n) in enumerate(((3, 48, 19), (8, 96, 40), (1, 16, 1), (5, 80, 33))):
        roots = inputs(m, h, k, n, seed=1500 + index)
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
    for path in tmp_path.glob("*.npy"):
        path.unlink()
    run = subprocess.run([sys.executable, "-c", _REPLAY, str(tmp_path)], capture_output=True, text=True,
                         env={**os.environ, "TESSERA_OPT": "/missing/compiler", "PATH": "/missing/tools"})
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert not any(entry["ok"] for entry in report.values()), report
    assert all("tessera-opt" in entry["error"] for entry in report.values()), report
