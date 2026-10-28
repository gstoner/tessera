"""Owning gfx1201 replay of the contraction-bounded native quantized MLP.

GFX1201-BOUNDED-QUANTIZED-K-20261028. One explicit
``compile_native_quantized_program(..., m_bound=MB, k_bound=KB)`` call; every
later public call at 1 <= M <= MB and K16-multiple K <= KB binds that one
capacity program and its retained owner without retracing or compiling.
Every output is bitwise equal to the static program compiled at that exact
(M, K), and the small shape also matches the independent staged oracle.
"""
import subprocess

import numpy as np
import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
from tests.device.rocm.test_native_quantized_mlp_bounded_rows import provider
from tests.unit.test_native_quantized_mlp_program import source
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_rocm, pytest.mark.compiler_route]

# (H, N), traced (M, K), (MB, KB), active (M, K) sequence.
SHAPES = [((32, 19), (3, 48), (8, 96), ((3, 48), (8, 96), (1, 16), (5, 80), (8, 48), (2, 96))),
          ((128, 256), (37, 4096), (64, 12288), ((37, 4096), (64, 12288), (1, 16), (16, 8192)))]


def static_at(m, h, k, n, roots, lib):
    with package_quantized_product_graph(source(m, h, k, n)).prepare(roots, runtime_library=lib) as owner:
        generation, _ = owner.invoke()
        return owner.read(generation)[0]


@pytest.mark.parametrize("dims,traced,bounds,actives", SHAPES)
def test_public_k_bound_matches_static_programs_with_one_prepare(dims, traced, bounds, actives, monkeypatch):
    from tessera.compiler.rocm_quantized_mlp_program import NativeQuantizedMLPProgram
    lib = provider()
    h, n = dims
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(traced[0], h, traced[1], n))
    program = fn.compile_native_quantized_program(*inputs(*traced[:1], h, traced[1], n),
                                                  m_bound=bounds[0], k_bound=bounds[1])
    assert program.native.k_bounded
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
    for index, (m, k) in enumerate(actives):
        roots = inputs(m, h, k, n, seed=1200 + index)
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
    # Every K change took the retained owner's full update; nothing re-prepared.
    assert prepares == [(actives[0][0], actives[0][1])]
    report = fn.compile_report()
    assert report.target_decision["rocm_gfx1201"].startswith(
        "canonical_rocm_quantized_k_bounded_mlp_program; 7 ordered native packages")
    with pytest.raises(ValueError, match="K16 capacity"):
        fn(*inputs(1, h, bounds[1] + 16, n, seed=17))
    with pytest.raises(ValueError, match="capacity"):
        fn(*inputs(bounds[0] + 1, h, 16, n, seed=18))
    # Refusals kept the owner: the next call still binds it without a prepare.
    roots = inputs(*actives[0][:1], h, actives[0][1], n, seed=19)
    np.testing.assert_array_equal(fn(*roots).view(np.uint16),
                                  static_at(actives[0][0], h, actives[0][1], n, roots, lib).view(np.uint16))
    assert len(prepares) == 1


def _owner(lib, *, m=3, k=48, mb=8, kb=96, h=32, n=19):
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(m, h, k, n))
    program = fn.compile_native_quantized_program(*inputs(m, h, k, n), m_bound=mb, k_bound=kb).native
    return program, program.prepare(inputs(m, h, k, n), runtime_library=lib)


def test_activation_only_update_keeps_the_resident_k():
    lib = provider()
    h, n = 32, 19
    _, owner = _owner(lib)
    try:
        assert owner.shape_info()[:4] == (8, 96, 3, 48)
        roots = inputs(6, h, 80, n, seed=31)
        owner.update(roots)
        assert (owner.active_m, owner.active_k) == (6, 80)
        x = inputs(4, h, 80, n, seed=32)[0]
        owner.update_activations(x)
        generation, _ = owner.invoke()
        np.testing.assert_array_equal(owner.read(generation)[0].view(np.uint16),
                                      static_at(4, h, 80, n, [x, *roots[1:]], lib).view(np.uint16))
        # The expanded weights were produced at K=80; a K=48 activation alone refuses
        # before any native write and leaves the owner usable.
        with pytest.raises(ValueError, match="differs from the resident weights"):
            owner.update_activations(inputs(4, h, 48, n)[0])
        generation, _ = owner.invoke()
        assert owner.read(generation)[0].shape == (4, n)
        assert owner.shape_info()[2:4] == (4, 80)
    finally:
        owner.close()


def _raw_update(owner, m, g, slots, arrays):
    import ctypes as C
    update = owner.lib.tessera_rocm_program_update_shape
    update.argtypes = [C.c_uint64, C.c_int64, C.c_int64, C.c_uint32, C.POINTER(C.c_uint32),
                       C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
    update.restype = C.c_int
    count = len(slots)
    return update(owner.handle, m, g, count, (C.c_uint32*count)(*slots),
                  (C.c_void_p*count)(*[a.ctypes.data for a in arrays]),
                  (C.c_uint64*count)(*[a.nbytes for a in arrays]))


def test_native_shape_update_admission():
    """The HIP owner re-checks every variable change against the frame."""
    lib = provider()
    h, n = 32, 19
    _, owner = _owner(lib)
    try:
        x = np.ascontiguousarray(inputs(3, h, 80, n)[0])
        # A K change needs every K-dependent original slot (x and both gate weight planes).
        assert _raw_update(owner, 3, 5, (0,), (x,)) == 1
        # G beyond capacity, and spans that disagree with the requested extents.
        assert _raw_update(owner, 3, 7, (0,), (np.ascontiguousarray(inputs(3, h, 112, n)[0]),)) == 1
        assert _raw_update(owner, 3, 3, (0,), (x,)) == 1
        assert owner.shape_info()[2:4] == (3, 48)
        generation, _ = owner.invoke()
        assert owner.read(generation)[0].shape == (3, n)
    finally:
        owner.close()


@pytest.mark.parametrize("tamper", ["unit", "coef", "drop_expansion", "grid"])
def test_native_shape_frame_admission_refuses_tampered_frames(tamper, monkeypatch):
    from tessera.compiler import rocm_quantized_mlp_program as module
    lib = provider()
    program, owner = _owner(lib)
    owner.close()
    original = module._shape_steps

    def steps(*args):
        entries = original(*args)
        if tamper == "coef":
            entries[3] = module._ShapeStep(2, 2, 8, 2, 3, 0, 0)          # K = 8 * G
        if tamper == "drop_expansion":
            entries.pop(1)                                              # step 1 has no entry
        if tamper == "grid":
            entries[1] = module._ShapeStep(1, 0, 1, 2, 0, 256, 512)     # capacity grid 12, not 1
        return entries
    monkeypatch.setattr(module, "_shape_steps", steps)
    if tamper == "unit":
        tail = module.PreparedQuantizedMLPProgram._shape_tail

        def broken(self):
            result = tail(self)
            self._shape_arrays[1][0] += 2                                 # x unit bytes
            return result
        monkeypatch.setattr(module.PreparedQuantizedMLPProgram, "_shape_tail", broken)
    with pytest.raises(RuntimeError, match="status 1"):
        program.prepare(inputs(3, 32, 48, 19), runtime_library=lib)


_REPLAY = "\n".join([
    "import sys,json,numpy as np,ml_dtypes",
    "from pathlib import Path",
    "from tessera import runtime as rt",
    "root=Path(sys.argv[1])",
    "artifact=rt.RuntimeArtifact.from_json((root/'artifact.json').read_text())",
    "out={}",
    "for name in sorted(p.name for p in root.glob('roots-*.npz')):",
    "    p=np.load(root/name)",
    "    roots=[p['x'].astype(ml_dtypes.bfloat16),p['packed'],p['scales'].view(ml_dtypes.float8_e4m3fn),p['globals_'],"
    "p['down_packed'],p['down_scales'].view(ml_dtypes.float8_e4m3fn),p['down_globals']]",
    "    result=rt.launch(artifact,roots)",
    "    if not result.get('ok'):",
    "        out[name]={'ok':False,'error':str(result.get('error') or result.get('reason') or result)[:400]}",
    "        continue",
    "    assert result.get('execution_kind')=='native_gpu' and len(result['component_receipts'])==7,result",
    "    np.save(root/(name+'.npy'),result['output']);out[name]={'ok':True}",
    "print(json.dumps(out))",
])


def test_k_bounded_artifact_replays_in_a_fresh_process(tmp_path):
    import json
    import os
    import sys
    from tests.device.rocm.test_public_quantized_mlp_row_bound_device import _save_roots
    provider()
    h, n = 32, 19
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(3, h, 48, n))
    fn.compile_native_quantized_program(*inputs(3, h, 48, n), m_bound=8, k_bound=96)
    expected = {}
    for index, (m, k) in enumerate(((3, 48), (8, 96), (1, 16), (5, 80))):
        roots = inputs(m, h, k, n, seed=1300 + index)
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
    # Replay re-derives the runtime-K expansion and both product projections.
    run = subprocess.run([sys.executable, "-c", _REPLAY, str(tmp_path)], capture_output=True, text=True,
                         env={**os.environ, "TESSERA_OPT": "/missing/compiler", "PATH": "/missing/tools"})
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert not any(entry["ok"] for entry in report.values()), report
    assert all("tessera-opt" in entry["error"] for entry in report.values()), report
