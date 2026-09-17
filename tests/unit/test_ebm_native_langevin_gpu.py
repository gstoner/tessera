"""The EBM Langevin loop as one cooperative device kernel (2026-09-16).

Host-free half (any host with tessera-opt built with the EBM backend): one
tessera-opt invocation runs paired autodiff, the EBM lowering, tessera-to-
linalg, inlining and the row-program emitter; the result is a single
gpu.func with guarded per-lane loads, the K-step scf.for carrying the state
and both key words in registers, Philox inside the loop, and no tensor or
linalg op left. Device half (owning ROCm / sm_120 host): the loop is one
launch, bit-exact with the declared numpy policy for 1/5/12 steps, and
`runtime.launch` reports the new rows.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest

from tessera.compiler.llvm_tools import llvm_bin_dir
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.ebm import native_langevin as nl
from tests._support.environment import (native_storage_target, nvidia_gpu_is_plausibly_present,
                                        rocm_gpu_is_plausibly_present)


def _compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("tessera-opt required")
    import subprocess
    help_text = subprocess.run([str(tool), "--help"], capture_output=True, text=True).stdout
    if "--tessera-ebm-lower-langevin" not in help_text or "--tessera-row-program-to-gpu" not in help_text:
        pytest.skip("tessera-opt built without the EBM backend or the row-program emitter")
    return tool


def _kernel(shape=(4, 8), steps=3, temperature=0.7, backend="nvidia", manifold="euclidean", energy="quadratic",
            **kw):
    tool = _compiler()
    source, specs = nl.langevin_device_source(shape, eta=0.1, temperature=temperature, steps=steps,
                                              backend=backend, compiler=tool, manifold=manifold, energy=energy, **kw)
    body = source.split("gpu.func @row_program(", 1)[1]
    return source, specs, body


def test_single_driver_chain_emits_one_cooperative_kernel():
    source, specs, body = _kernel()
    assert source.count("gpu.func ") == 1 and "kernel attributes {known_block_size = array<i32: 8, 1, 1>" in source
    assert "tessera_ebm." not in body and "linalg." not in body and "tensor." not in body and "func.call" not in body
    assert "gpu.block_id" in body and "gpu.thread_id" in body
    # Guarded lane loads of y0 and x, the two key words, then ONE loop carrying
    # (state f32, key i64, key i64), then guarded stores of y and the key.
    assert body.count("llvm.load") == 4 and body.count("llvm.store") == 3
    loops = re.findall(r"scf\.for [^\n]*iter_args\(([^)]*)\) -> \(([^)]*)\)", body)
    assert len(loops) == 1 and loops[0][1] == "f32, i64, i64"
    # Philox-4x32-10 inside the loop body (CSE folds the constant-keyed round).
    assert body.count("arith.mului_extended") >= 18 and "math.log" in body and "math.cos" in body
    assert [s.name for s in specs] == ["y0", "x", "key", "y", "next_key", "scratch"]
    assert '\\22grid\\22:[4,1,1]' in source and '\\22block\\22:[8,1,1]' in source
    # The quadratic gradient is elementwise, so this energy needs no cross-lane
    # reduction: the emitter places no barrier at all.
    assert "gpu.barrier" not in body


def test_zero_temperature_kernel_has_no_noise_and_no_barriers():
    _, _, body = _kernel(temperature=0.0)
    assert "math.log" not in body and "arith.mului_extended" not in body and "gpu.barrier" not in body


def test_kernel_replays_through_the_arena_pipeline_for_both_backends():
    from tessera.compiler.native_gpu_storage import replay_arena_ir
    tool = _compiler()
    for backend in ("nvidia", "rocm"):
        source, _, _ = _kernel(shape=(3, 5), backend=backend)
        arena = replay_arena_ir(tool, source)
        assert "__tessera_shared_bytes_" in arena and "gpu.func @row_program" in arena
        assert "known_block_size = array<i32: 8, 1, 1>" in arena  # 5 features -> 8 lanes


def test_row_reduction_program_emits_the_ordered_shared_fold():
    """The quadratic gradient is elementwise, so the Langevin kernel carries no
    reduction; the emitter's reduction path is proven on a row-normalization
    program (feature-axis linalg.reduce broadcast back over the lanes)."""
    from tessera.compiler.native_row_program import row_normalize_module, row_program_kernel
    tool = _compiler()
    kernel, lanes = row_program_kernel(row_normalize_module(3, 6), entry="row_normalize", backend="rocm", compiler=tool)
    body = kernel.split("gpu.func @row_program(", 1)[1]
    assert lanes == 8 and body.count("gpu.barrier") == 3
    assert "llvm.mlir.addressof @row_reduction" in body and body.count("!llvm.ptr<3>") >= 4
    assert "linalg." not in body and "tensor." not in body and "llvm.intr.sqrt" in body and "math.sqrt" not in body
    # NVIDIA: convert-gpu-to-nvvm outlaws the LLVM math intrinsics, so the
    # emitter calls libdevice's rounding-explicit __nv_fsqrt_rn instead.
    nv, _ = row_program_kernel(row_normalize_module(3, 6), entry="row_normalize", backend="nvidia", compiler=tool)
    nv_body = nv.split("gpu.func @row_program(", 1)[1]  # the provenance attribute keeps the source text
    assert "llvm.call @__nv_fsqrt_rn" in nv_body and "math.sqrt" not in nv_body and "llvm.intr.sqrt" not in nv_body


@pytest.mark.parametrize("shape,message", [((4, 2000), "1024 features"), ((4,), "rows, features")])
def test_out_of_envelope_shapes_are_refused(shape, message):
    tool = _compiler()
    with pytest.raises(ValueError, match=message):
        nl.langevin_device_source(shape, eta=0.1, temperature=0.5, steps=1, backend="nvidia", compiler=tool)


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
    tool, llvm = _compiler(), llvm_bin_dir()
    if llvm is None:
        pytest.skip("matched LLVM tools required")
    return backend, chip, tool, llvm


def _pair(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32), rng.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("shape,steps", [((4, 8), 1), ((6, 8), 5), ((3, 5), 12), ((9, 100), 4)])
def test_device_loop_is_bit_exact_with_the_declared_policy(shape, steps):
    backend, chip, tool, llvm = _device_lane()
    y, x = _pair(shape, 40 + steps)
    out, key = nl.native_langevin_loop_device(y, x, [0x1234ABCD9876, 42], eta=0.1, temperature=0.7, steps=steps,
                                              backend=backend, chip=chip, compiler=tool, llvm_bin=llvm)
    expect, expect_key = nl.reference_langevin_loop(y, x, [0x1234ABCD9876, 42], eta=0.1, temperature=0.7, steps=steps)
    assert list(key) == list(expect_key)
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)
    assert not np.allclose(out, y - 0.1 * (y - x))


def test_device_descent_matches_the_independent_formula():
    backend, chip, tool, llvm = _device_lane()
    y, x = _pair((5, 7), 3)
    out, _ = nl.native_langevin_loop_device(y, x, [1, 1], eta=0.25, temperature=0.0, steps=1,
                                            backend=backend, chip=chip, compiler=tool, llvm_bin=llvm)
    np.testing.assert_allclose(out, y - 0.25 * (y - x), rtol=1e-6, atol=1e-6)


def test_device_loop_is_one_launch_and_launch_row_reports_native_gpu():
    from tessera import runtime as rt
    backend, chip, tool, llvm = _device_lane()
    y, x = _pair((4, 8), 11)
    target = "rocm" if backend == "rocm" else "nvidia_sm120"
    artifact = nl.package_ebm_langevin_native((4, 8), eta=0.1, temperature=0.3, steps=6, target=target)
    result = rt.launch(artifact, (y, x, [5, 6]))
    assert result["ok"] and result["execution_kind"] == "native_gpu"
    assert result["compiler_path"] == f"{'rocm' if backend == 'rocm' else 'nvidia'}_ebm_langevin_native_compiled"
    out, key = result["output"]
    expect, expect_key = nl.reference_langevin_loop(y, x, [5, 6], eta=0.1, temperature=0.3, steps=6)
    np.testing.assert_allclose(np.asarray(out), expect, rtol=1e-5, atol=1e-5)
    assert list(np.asarray(key)) == list(expect_key)
    # The whole 6-step loop was one kernel launch: the packaged program's
    # binding launched once for this call (per-call launches are counted by
    # the bound package).
    program = nl.ebm_langevin_program((4, 8), eta=0.1, temperature=0.3, steps=6, backend=backend, chip=chip,
                                      compiler=tool, llvm_bin=llvm)
    before = getattr(program.bound, "launch_count", None)
    program.run(y, x, [5, 6])
    if before is not None:
        assert program.bound.launch_count == before + 1


def _sequential_row_normalize(x):
    """The declared reduction order: lanes folded in index order in f32."""
    out = np.empty_like(x)
    for r in range(x.shape[0]):
        acc = np.float32(0.0)
        for v in x[r]:
            acc = np.float32(np.float32(v * v) + acc)
        out[r] = x[r] / np.float32(np.sqrt(acc))
    return out


@pytest.mark.parametrize("shape", [(3, 6), (5, 8), (2, 100), (7, 1024)])
def test_row_reduction_program_is_bit_exact_with_the_declared_order_on_device(shape):
    from tessera.compiler.native_gpu_tensor import TensorSpec
    from tessera.compiler.native_row_program import row_normalize_module, row_program_device
    backend, chip, tool, llvm = _device_lane()
    x = np.random.default_rng(shape[1]).standard_normal(shape).astype(np.float32)
    program = row_program_device(row_normalize_module(*shape), entry="row_normalize", rows=shape[0],
                                 specs=(TensorSpec("x", "fp32", shape, False), TensorSpec("y", "fp32", shape, True)),
                                 backend=backend, chip=chip, compiler=tool, llvm_bin=llvm, name="row normalize")
    out = program.run(x)
    np.testing.assert_array_equal(np.asarray(out), _sequential_row_normalize(x))
    np.testing.assert_allclose(np.linalg.norm(np.asarray(out), axis=1), 1.0, rtol=1e-5, atol=1e-5)


# --- N1 / M1 on the device --------------------------------------------------

@pytest.mark.parametrize("energy", nl.ENERGIES)
def test_every_energy_becomes_one_cooperative_kernel(energy):
    """The integrator is energy-agnostic: whatever the paired pass derives is
    lowered inside the same kernel, with no call and no host round-trip."""
    source, _, body = _kernel(energy=energy)
    assert source.count("gpu.func ") == 1
    assert "func.call" not in body and "linalg." not in body and "tensor." not in body
    assert "tessera.custom_adjoint_call" not in source  # a host VJP would be a per-step transfer
    if energy == "huber":
        assert "arith.select" in body      # the kink is in the device code
    if energy == "softplus":
        assert "math.exp" in body          # the stable sigmoid form


@pytest.mark.parametrize("energy", nl.ENERGIES)
def test_the_sphere_kernel_carries_its_reductions_and_status(energy):
    """Four ordered row reductions per step (entry norm, two projections, the
    retraction norm), each fenced by barriers, plus the per-row status word."""
    source, specs, body = _kernel(manifold="sphere", energy=energy)
    assert [s.name for s in specs] == ["y0", "x", "key", "y", "next_key", "status", "scratch"]
    assert body.count("gpu.barrier") == 12
    assert "llvm.mlir.addressof @row_reduction" in body
    loops = re.findall(r"scf\.for [^\n]*iter_args\(([^)]*)\) -> \(([^)]*)\)", body)
    assert loops[0][1] == "f32, i64, i64, i32"          # state, key, key, status
    assert body.count("__nv_fsqrt_rn") >= 1 or "llvm.intr.sqrt" in body   # the retraction
    assert "linalg." not in body and "tensor." not in body


def test_the_sphere_kernel_replays_for_both_backends():
    from tessera.compiler.native_gpu_storage import replay_arena_ir
    tool = _compiler()
    for backend in ("nvidia", "rocm"):
        source, _, _ = _kernel(shape=(3, 5), backend=backend, manifold="sphere")
        arena = replay_arena_ir(tool, source)
        assert "__tessera_shared_bytes_" in arena and "gpu.func @row_program" in arena


def _sphere_pair(shape, seed):
    v = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    unit = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    return unit, np.random.default_rng(seed + 1).standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("energy", nl.ENERGIES)
def test_device_energies_are_bit_exact_with_the_declared_policy(energy):
    backend, chip, tool, llvm = _device_lane()
    y, x = _pair((6, 8), 60)
    out, key = nl.native_langevin_loop_device(y, x, [11, 12], eta=0.1, temperature=0.5, steps=5,
                                              backend=backend, chip=chip, compiler=tool, llvm_bin=llvm,
                                              energy=energy)
    expect, expect_key = nl.reference_langevin_loop(y, x, [11, 12], eta=0.1, temperature=0.5, steps=5, energy=energy)
    assert list(key) == list(expect_key)
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape,steps,energy", [((4, 8), 1, "quadratic"), ((5, 16), 4, "huber"),
                                                ((3, 33), 7, "softplus"), ((2, 100), 3, "quadratic")])
def test_device_sphere_matches_the_reference_and_stays_on_the_sphere(shape, steps, energy):
    backend, chip, tool, llvm = _device_lane()
    y, x = _sphere_pair(shape, 70 + steps)
    out, key, status = nl.native_langevin_loop_device(y, x, [21, 22], eta=0.05, temperature=0.3, steps=steps,
                                                      backend=backend, chip=chip, compiler=tool, llvm_bin=llvm,
                                                      manifold="sphere", energy=energy)
    expect, expect_key, expect_status = nl.reference_langevin_loop(
        y, x, [21, 22], eta=0.05, temperature=0.3, steps=steps, manifold="sphere", energy=energy)
    assert list(key) == list(expect_key) and list(np.asarray(status)) == list(expect_status)
    # The row reductions are ordered, so the device fold matches the sequential
    # host fold; the tolerance covers only the f32 rounding of the projections.
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(out), axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_device_sphere_reports_the_entry_precondition_per_row():
    backend, chip, tool, llvm = _device_lane()
    y, x = _sphere_pair((4, 8), 80)
    bad = y.copy(); bad[1] *= 2.0
    _, _, status = nl.native_langevin_loop_device(bad, x, [1, 1], eta=0.05, temperature=0.0, steps=1,
                                                  backend=backend, chip=chip, compiler=tool, llvm_bin=llvm,
                                                  manifold="sphere")
    assert list(np.asarray(status)) == [0, 1, 0, 0]


def test_device_sphere_launch_row_returns_the_status():
    from tessera import runtime as rt
    backend, chip, tool, llvm = _device_lane()
    y, x = _sphere_pair((4, 8), 90)
    target = "rocm" if backend == "rocm" else "nvidia_sm120"
    artifact = nl.package_ebm_langevin_native((4, 8), eta=0.05, temperature=0.3, steps=4, target=target,
                                              manifold="sphere", energy="softplus")
    result = rt.launch(artifact, (y, x, [7, 8]))
    assert result["ok"] and result["execution_kind"] == "native_gpu" and len(result["output"]) == 3
    expect = nl.reference_langevin_loop(y, x, [7, 8], eta=0.05, temperature=0.3, steps=4,
                                        manifold="sphere", energy="softplus")
    np.testing.assert_allclose(np.asarray(result["output"][0]), expect[0], rtol=1e-5, atol=1e-5)


# --- M2: the bivector integrator on the device ------------------------------

@pytest.mark.parametrize("energy", nl.ENERGIES)
def test_the_bivector_kernel_reads_its_grade_mask_from_a_per_lane_table(energy):
    """The grade projection is the Clifford dialect's own compile-time keep
    mask; on the device the blade index IS the lane, so the mask becomes a
    private constant table each lane indexes — no Clifford op and no batch
    loop survive into the kernel."""
    source, specs, body = _kernel(manifold="bivector", energy=energy)
    assert [s.name for s in specs] == ["y0", "x", "key", "y", "next_key", "status", "scratch"]
    assert "tessera_clifford." not in body and "linalg." not in body and "tensor." not in body
    # Two tables: the keep mask and the negate mask of the diagonal blade map.
    assert body.count("row_feature_table") >= 2
    assert "llvm.mlir.global private constant @row_feature_table_0" in source
    loops = re.findall(r"scf\.for [^\n]*iter_args\([^)]*\) -> \(([^)]*)\)", body)
    assert loops[0] == "f32, i64, i64, i32"     # state, key, key, status
    # One ordered reduction only — the entry-grade leak check.
    assert body.count("gpu.barrier") == 3


def test_the_bivector_kernel_replays_for_both_backends():
    from tessera.compiler.native_gpu_storage import replay_arena_ir
    tool = _compiler()
    for backend in ("nvidia", "rocm"):
        source, _, _ = _kernel(backend=backend, manifold="bivector")
        arena = replay_arena_ir(tool, source)
        assert "__tessera_shared_bytes_" in arena and "gpu.func @row_program" in arena


@pytest.mark.parametrize("shape,message", [((4, 7), "2\\^n"), ((4, 8), "out of range")])
def test_bivector_semantic_keys_are_checked(shape, message):
    tool = _compiler()
    grade = 2 if shape[1] == 7 else 9
    with pytest.raises(ValueError, match=message):
        nl.langevin_device_source(shape, eta=0.1, temperature=0.3, steps=1, backend="nvidia",
                                  compiler=tool, manifold="bivector", grade=grade)


def _bivector_device_pair(rows=4, seed=61, grade=nl.BIVECTOR_GRADE):
    rng = np.random.default_rng(seed)
    return (nl.grade_projection(rng.standard_normal((rows, 8)).astype(np.float32), grade),
            rng.standard_normal((rows, 8)).astype(np.float32))


@pytest.mark.parametrize("energy", nl.ENERGIES)
@pytest.mark.parametrize("steps", [1, 6])
def test_device_bivector_matches_the_reference_and_stays_in_the_subspace(energy, steps):
    backend, chip, tool, llvm = _device_lane()
    y, x = _bivector_device_pair(seed=60 + steps)
    out, key, status = nl.native_langevin_loop_device(y, x, [31, 32], eta=0.05, temperature=0.3, steps=steps,
                                                      backend=backend, chip=chip, compiler=tool, llvm_bin=llvm,
                                                      manifold="bivector", energy=energy)
    expect, expect_key, expect_status = nl.reference_langevin_loop(
        y, x, [31, 32], eta=0.05, temperature=0.3, steps=steps, manifold="bivector", energy=energy)
    assert list(key) == list(expect_key) and list(np.asarray(status)) == list(expect_status)
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)
    off_grade = [i for i, g in enumerate(nl.blade_grades()) if g != 2]
    assert np.all(np.asarray(out)[:, off_grade] == 0.0)


def test_device_bivector_reports_the_entry_grade_per_row():
    backend, chip, tool, llvm = _device_lane()
    y, x = _bivector_device_pair(seed=63)
    bad = y.copy(); bad[2, 0] = 1.0
    _, _, status = nl.native_langevin_loop_device(bad, x, [1, 1], eta=0.05, temperature=0.0, steps=1,
                                                  backend=backend, chip=chip, compiler=tool, llvm_bin=llvm,
                                                  manifold="bivector")
    assert list(np.asarray(status)) == [0, 0, 1, 0]


def test_device_bivector_launch_row():
    from tessera import runtime as rt
    backend, chip, tool, llvm = _device_lane()
    y, x = _bivector_device_pair(seed=64)
    target = "rocm" if backend == "rocm" else "nvidia_sm120"
    artifact = nl.package_ebm_langevin_native((4, 8), eta=0.05, temperature=0.3, steps=5, target=target,
                                              manifold="bivector", energy="huber")
    result = rt.launch(artifact, (y, x, [8, 9]))
    assert result["ok"] and result["execution_kind"] == "native_gpu" and len(result["output"]) == 3
    expect = nl.reference_langevin_loop(y, x, [8, 9], eta=0.05, temperature=0.3, steps=5,
                                        manifold="bivector", energy="huber")
    np.testing.assert_allclose(np.asarray(result["output"][0]), expect[0], rtol=1e-5, atol=1e-5)


# --- the annealing schedule on the device ----------------------------------

def test_the_annealed_kernel_carries_its_temperature_in_registers():
    """Host-free: the cooling schedule must ride in the kernel, not force a
    launch per step. The loop's iter_args gain one f32 beside the state and the
    two key words, and no transcendental is needed — the temperature is cooled by
    a multiply, so the row-program emitter's math admission table (which refuses
    `math.powf`) is not in the way."""
    tool = _compiler()
    source, _ = nl.langevin_device_source((4, 8), eta=0.1, temperature=0.7, steps=3,
                                          backend="rocm", compiler=tool, anneal=0.5)
    assert source.count("gpu.func ") == 1
    body = source.split("gpu.func @row_program(", 1)[1]
    assert "linalg." not in body and "tensor." not in body and "math.powf" not in body
    carried = re.findall(r"iter_args\([^)]*\) -> \(([^)]*)\)", body)
    assert carried and carried[0].count("f32") == 2, carried   # state + temperature
    # The per-step noise scale's sqrt: the emitter pins it to the correctly
    # rounded realization for the route, so on ROCm it is the LLVM intrinsic and
    # `math.sqrt` is gone by here.
    assert "llvm.intr.sqrt" in body and "math.sqrt" not in body


@pytest.mark.parametrize("anneal", [1.0, 0.5])
def test_the_annealed_chain_matches_the_declared_policy_on_device(anneal):
    backend, chip, tool, llvm = _device_lane()
    rng = np.random.default_rng(int(anneal * 10) + 4)
    shape = (5, 16)
    y0 = rng.standard_normal(shape).astype(np.float32)
    x = rng.standard_normal(shape).astype(np.float32)
    key = [0x5150ABC, 11]
    got = nl.native_langevin_loop_device(y0, x, key, eta=0.1, temperature=0.6, steps=6,
                                         backend=backend, chip=chip, compiler=tool, llvm_bin=llvm,
                                         anneal=anneal)
    want, want_key = nl.reference_langevin_loop(y0, x, key, eta=0.1, temperature=0.6, steps=6,
                                                anneal=anneal)
    np.testing.assert_allclose(np.asarray(got[0]), want, rtol=1e-5, atol=1e-6)
    assert list(np.asarray(got[1])) == list(want_key)


def test_the_annealed_device_chain_agrees_with_the_cpu_lane():
    """Same schedule, two lanes: the device kernel and the CPU JIT must compute
    the same cooling chain, since both claim the same declared policy."""
    backend, chip, tool, llvm = _device_lane()
    rng = np.random.default_rng(88)
    y0 = rng.standard_normal((4, 32)).astype(np.float32)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    key = [7, 2]
    device = nl.native_langevin_loop_device(y0, x, key, eta=0.05, temperature=0.5, steps=8,
                                            backend=backend, chip=chip, compiler=tool, llvm_bin=llvm,
                                            anneal=0.7)
    host = nl.native_langevin_loop(y0, x, key, eta=0.05, temperature=0.5, steps=8, anneal=0.7)
    np.testing.assert_allclose(np.asarray(device[0]), np.asarray(host[0]), rtol=1e-5, atol=1e-6)
