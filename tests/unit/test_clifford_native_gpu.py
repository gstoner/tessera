"""The Clifford product family through the native GPU storage route (W6.4 GPU
route, 2026-09-16).

Host-free half (any host with ts-clifford-opt + tessera-opt): the kernel
skeleton carries a rank-1 `tessera_clifford` op; the dialect's own lowering
expands it, and the arena pipeline folds the tensors away, so the device
kernel is scalar arithmetic the compiler emitted -- with the grade-pruned
term count observable. Device half (owning ROCm / sm_120 host only): every
op of the family matches the standalone GA reference, and `runtime.launch`
reports native_gpu on the new rows.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tessera.compiler import native_clifford_gpu as ncg
from tessera.compiler.llvm_tools import llvm_bin_dir
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support.environment import (native_storage_target, nvidia_gpu_is_plausibly_present,
                                        rocm_gpu_is_plausibly_present)


def _tools():
    tool, clifford, llvm = find_tessera_opt(), ncg.find_ts_clifford_opt(), llvm_bin_dir()
    if tool is None or clifford is None or llvm is None:
        pytest.skip("tessera-opt, ts-clifford-opt and the matched LLVM tools are required")
    return tool, clifford, llvm


def _mulf(arena: str) -> int:
    return sum("arith.mulf" in line for line in arena.splitlines())


@pytest.mark.parametrize("op", sorted(ncg.CLIFFORD_GPU_OPS))
def test_skeleton_expands_to_a_scalar_kernel(op):
    from tessera.compiler.native_gpu_storage import replay_arena_ir
    tool, clifford, _ = _tools()
    grades = [1, 3] if op == "grade" else None
    skeleton, specs = ncg.clifford_gpu_skeleton(op, (6, 8), grades=grades)
    assert f'"tessera_clifford.{op}"' in skeleton
    expanded = ncg.expand_clifford_source(skeleton, clifford_opt=clifford)
    assert "tessera_clifford." not in expanded
    arena = replay_arena_ir(tool, expanded)
    assert "tensor." not in arena and "tessera_clifford." not in arena
    assert "gpu.func @clifford_" + op in arena and "__tessera_shared_bytes_" in arena
    scalar = ncg.CLIFFORD_GPU_OPS[op][1]
    assert specs[-2].shape == ((6, 1) if scalar else (6, 8)) and specs[-2].writable


def test_grade_restriction_prunes_the_emitted_kernel():
    """Cl(3,0) grade-2 output keeps 24 of the 64 products; the pruning is in
    the device code, not in a mask applied afterwards."""
    from tessera.compiler.native_gpu_storage import replay_arena_ir
    tool, clifford, _ = _tools()
    full = replay_arena_ir(tool, ncg.expand_clifford_source(ncg.clifford_gpu_skeleton("geo_product", (4, 8))[0], clifford_opt=clifford))
    pruned = replay_arena_ir(tool, ncg.expand_clifford_source(ncg.clifford_gpu_skeleton("geo_product", (4, 8), grades=[2])[0], clifford_opt=clifford))
    assert _mulf(full) == 64 and _mulf(pruned) == 24


@pytest.mark.parametrize("op,shape,grades,message", [
    ("exp", (4, 8), None, "no lowering"), ("geo_product", (4, 7), None, "shape"),
    ("grade", (4, 8), None, "grades"), ("wedge", (4, 8), [1], "grades applies"),
    ("geo_product", (1 << 24, 8), None, "bound"),
])
def test_out_of_envelope_requests_are_refused(op, shape, grades, message):
    with pytest.raises(ValueError, match=message):
        ncg.clifford_gpu_skeleton(op, shape, grades=grades)


# --- device half -----------------------------------------------------------

def _device_lane():
    target = native_storage_target()
    if target is None:
        pytest.skip("no CUDA or ROCm toolchain on this host")
    backend, chip = target
    present = nvidia_gpu_is_plausibly_present() if backend == "nvidia" else rocm_gpu_is_plausibly_present()
    if not present:
        pytest.skip(f"no {backend} device on this host")
    if backend == "rocm" and chip == "gfx1201" and os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1":
        pytest.skip("gfx1201 device proof requires TESSERA_GFX1201_DEVICE_PROOF=1")
    tool, clifford, llvm = _tools()
    return backend, chip, tool, llvm


_REF = {
    "geo_product": "clifford_geometric_product", "wedge": "clifford_wedge",
    "left_contract": "clifford_left_contraction", "inner": "clifford_inner", "norm": "clifford_norm",
    "rotor_sandwich": "clifford_rotor_sandwich", "reverse": "clifford_reverse",
    "grade_involute": "clifford_grade_involution", "conjugate": "clifford_conjugate",
    "hodge_star": "clifford_hodge_star",
}


def _reference(op, *arrays):
    import tessera._clifford_ops as ref
    fn = getattr(ref, _REF[op])
    flat = [x.reshape(-1, 8) for x in arrays]
    rows = [np.asarray(fn(*[f[i] for f in flat]), dtype=np.float32) for i in range(flat[0].shape[0])]
    return np.stack(rows).reshape(arrays[0].shape[:-1] + rows[0].shape).astype(np.float32)


@pytest.mark.parametrize("op", sorted(_REF))
@pytest.mark.parametrize("shape", [(8,), (37, 8), (3, 5, 8)])
def test_family_executes_on_the_device(op, shape):
    backend, chip, tool, llvm = _device_lane()
    rng = np.random.default_rng(hash((op, shape)) & 0xFFFF)
    arity = ncg.CLIFFORD_GPU_OPS[op][0]
    arrays = [rng.standard_normal(shape).astype(np.float32) for _ in range(arity)]
    if op == "rotor_sandwich":
        rotor = np.zeros(shape, np.float32); rotor[..., 0] = np.cos(0.3); rotor[..., 3] = np.sin(0.3)
        arrays[0] = rotor
    program = ncg.clifford_gpu_program(op, shape, backend=backend, chip=chip, compiler=tool, llvm_bin=llvm)
    out = program.run(*arrays)
    expect = _reference(op, *arrays)
    if ncg.CLIFFORD_GPU_OPS[op][1]:
        out = out.reshape(shape[:-1])
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)


def test_grade_pruned_product_and_projection_on_the_device():
    backend, chip, tool, llvm = _device_lane()
    rng = np.random.default_rng(9)
    a, b = (rng.standard_normal((11, 8)).astype(np.float32) for _ in range(2))
    full = _reference("geo_product", a, b)
    pruned = ncg.clifford_gpu_program("geo_product", (11, 8), backend=backend, chip=chip, compiler=tool, llvm_bin=llvm, grades=[2]).run(a, b)
    grade2 = [3, 5, 6]
    np.testing.assert_allclose(pruned[:, grade2], full[:, grade2], rtol=1e-5, atol=1e-5)
    assert np.all(pruned[:, [i for i in range(8) if i not in grade2]] == 0)
    kept = ncg.clifford_gpu_program("grade", (11, 8), backend=backend, chip=chip, compiler=tool, llvm_bin=llvm, grades=[0, 3]).run(a)
    np.testing.assert_array_equal(kept[:, [0, 7]], a[:, [0, 7]])
    assert np.all(kept[:, 1:7] == 0)


def test_runtime_launch_reports_the_native_gpu_row():
    from tessera import runtime as rt
    backend, chip, tool, llvm = _device_lane()
    target = "rocm" if backend == "rocm" else "nvidia_sm120"
    rng = np.random.default_rng(4)
    a, b = (rng.standard_normal((9, 8)).astype(np.float32) for _ in range(2))
    result = rt.launch(ncg.package_clifford_native("wedge", (9, 8), target=target), (a, b))
    assert result["ok"] and result["execution_kind"] == "native_gpu"
    assert result["compiler_path"] == f"{'rocm' if backend == 'rocm' else 'nvidia'}_clifford_native_compiled"
    np.testing.assert_allclose(np.asarray(result["output"]), _reference("wedge", a, b), rtol=1e-5, atol=1e-5)
    refused = rt.launch(ncg.package_clifford_native("wedge", (9, 8), target=target), (a[:2], b[:2]))
    assert refused["ok"] is False and "shape" in str(refused.get("reason", refused))
