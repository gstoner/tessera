"""Owning gfx1201/gfx1151 public replay of the bounded-row native quantized MLP.

GFX1201-PUBLIC-QUANTIZED-ROW-BOUND-20261027. One explicit
``compile_native_quantized_program(..., m_bound=B)`` call; every later public
call at 1 <= M <= B binds that capacity program without retracing or compiling.
"""
import subprocess

import numpy as np
import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
from tests.device.rocm.test_native_quantized_mlp_bounded_rows import owning_target, provider, source, static_rows
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_rocm, pytest.mark.compiler_route]

SHAPES = [((32, 48, 19), 8, (3, 8, 1, 5)),
          ((16, 16, 1), 1, (1,)),
          ((128, 12288, 256), 256, (37, 256, 1))]


@pytest.mark.parametrize("dims,bound,actives", SHAPES)
def test_public_bound_replays_every_active_m_without_compiling(dims, bound, actives, monkeypatch):
    lib = provider()
    h, k, n = dims
    fn = ts.jit(target=owning_target(), native_required=True)(make_mlp(actives[0], h, k, n))
    program = fn.compile_native_quantized_program(*inputs(actives[0], h, k, n), m_bound=bound)
    assert program.bounded
    capacity = package_quantized_product_graph(source(bound, h, k, n))

    oracles = {m: package_quantized_product_graph(source(m, h, k, n)) for m in actives if m <= 64}

    def forbidden(*a, **kw):
        pytest.fail("compiler or tracer ran during bounded public replay")
    digest = program.native.binding_digest
    for index, m in enumerate(actives):
        roots = inputs(m, h, k, n, seed=600 + index)
        before = [x.copy() for x in roots]
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "run", forbidden)
            patch.setattr(fn, "_trace_frontend_capture", forbidden)
            patch.setattr(fn, "_fn", forbidden)
            actual = fn(*roots)
        assert actual.shape == (m, n)
        assert fn.execution_kind == "native_gpu"
        assert fn._rocm_quantized_product_last_program is program
        assert len(fn._rocm_quantized_product_last_receipts) == 7
        for root, held in zip(roots, before, strict=True):
            np.testing.assert_array_equal(root, held)
        expected = static_rows(capacity, roots, bound, lib, seed=900 + index)
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
        if m <= 64:
            # Independent staged math at the active shape.
            assert_compiled_chain(oracles[m], roots, actual, lib)
    assert program.native.binding_digest == digest
    report = fn.compile_report()
    assert report.target_decision[owning_target()].startswith(
        "canonical_rocm_quantized_bounded_mlp_program; 7 ordered native packages")
    assert fn.runtime_artifact().metadata["compiler_path"] == "canonical_rocm_quantized_bounded_mlp_program"
    with pytest.raises(ValueError, match="capacity"):
        fn(*inputs(bound + 1, h, k, n, seed=17))


def test_public_bound_refuses_changed_weight_shapes():
    provider()
    fn = ts.jit(target=owning_target(), native_required=True)(make_mlp(3, 32, 48, 19))
    fn.compile_native_quantized_program(*inputs(3, 32, 48, 19), m_bound=8)
    # The bound rebinds rows only: N, K and the weight roots stay fixed.
    with pytest.raises(ValueError, match="projection boundaries"):
        fn(*inputs(3, 32, 48, 21))
    roots = inputs(3, 32, 48, 19)
    roots[0] = inputs(3, 32, 64, 19)[0]
    with pytest.raises(ValueError, match="storage/shape"):
        fn(*roots)


def test_public_bound_retains_one_owner_across_calls(monkeypatch):
    """GFX1201-PUBLIC-QUANTIZED-CALL-OVERHEAD-20261028: one prepare serves every call.

    Byte-identical weights replace only the activation rows; new or in-place
    mutated weights take the owner's full update. Every output stays bitwise
    equal to the static capacity rows of the roots it was given.
    """
    from tessera.compiler.rocm_quantized_mlp_program import NativeQuantizedMLPProgram
    lib = provider()
    h, k, n, bound = 32, 48, 19, 8
    fn = ts.jit(target=owning_target(), native_required=True)(make_mlp(3, h, k, n))
    public = fn.compile_native_quantized_program(*inputs(3, h, k, n), m_bound=bound).native
    capacity = package_quantized_product_graph(source(bound, h, k, n))
    prepares = []
    original = NativeQuantizedMLPProgram.prepare

    def counted(self, values, *, runtime_library):
        if self is public:
            prepares.append(values[0].shape[0])
        return original(self, values, runtime_library=runtime_library)
    monkeypatch.setattr(NativeQuantizedMLPProgram, "prepare", counted)

    def check(roots, seed):
        expected = static_rows(capacity, roots, bound, lib, seed=seed)
        np.testing.assert_array_equal(fn(*roots).view(np.uint16), expected.view(np.uint16))

    weights = inputs(3, h, k, n, seed=40)[1:]
    for index, m in enumerate((3, 8, 1, 5)):
        check([inputs(m, h, k, n, seed=50 + index)[0], *weights], 950 + index)
    with pytest.raises(ValueError, match="capacity"):
        fn(*inputs(bound + 1, h, k, n, seed=17))
    check(inputs(4, h, k, n, seed=61), 961)                        # new weight set
    mutated = inputs(4, h, k, n, seed=61)
    mutated[1][0, 0] ^= 0x11                                         # mutate in place
    check(mutated, 962)
    check([inputs(2, h, k, n, seed=63)[0], *mutated[1:]], 963)       # held again
    # The capacity-bound refusal and both weight refreshes reused the owner.
    assert prepares == [3]


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


def _save_roots(path, roots):
    np.savez(path, x=roots[0].astype(np.float32), packed=roots[1], scales=roots[2].view(np.uint8),
             globals_=roots[3], down_packed=roots[4], down_scales=roots[5].view(np.uint8),
             down_globals=roots[6])


def test_bounded_artifact_replays_in_a_fresh_process(tmp_path):
    """GFX1201-PORTABLE-BOUNDED-QUANTIZED-RECORD-20261028.

    The serialized v5 artifact launches every active M <= B in a new process,
    bitwise equal to the public calls. Replay re-derives both arbitrary-M
    product images through the matching compiler, so a compiler-free process
    refuses rather than trusting the recorded projection.
    """
    import json
    import os
    import sys
    provider()
    h, k, n, bound = 32, 48, 19, 8
    fn = ts.jit(target=owning_target(), native_required=True)(make_mlp(3, h, k, n))
    fn.compile_native_quantized_program(*inputs(3, h, k, n), m_bound=bound)
    expected = {}
    for index, m in enumerate((3, 8, 1, 5)):
        roots = inputs(m, h, k, n, seed=700 + index)
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
        got = np.load(tmp_path / (name + ".npy"))
        assert got.shape == want.shape
        np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))

    for path in tmp_path.glob("*.npy"):
        path.unlink()
    run = subprocess.run([sys.executable, "-c", _REPLAY, str(tmp_path)], capture_output=True, text=True,
                         env={**os.environ, "TESSERA_OPT": "/missing/compiler", "PATH": "/missing/tools"})
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert not any(entry["ok"] for entry in report.values()), report
    assert all("tessera-opt" in entry["error"] for entry in report.values()), report
    assert not list(tmp_path.glob("*.npy"))
