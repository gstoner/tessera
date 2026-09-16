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


def _kernel(shape=(4, 8), steps=3, temperature=0.7, backend="nvidia"):
    tool = _compiler()
    source, specs = nl.langevin_device_source(shape, eta=0.1, temperature=temperature, steps=steps,
                                              backend=backend, compiler=tool)
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
