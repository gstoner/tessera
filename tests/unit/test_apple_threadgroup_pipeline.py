"""APPLE-PIPE-1 — one owner for Apple threadgroup memory.

Apple decides threadgroup placement in two places, and they must not drift:

  * the C++ ``tessera-apple-threadgroup-pipeline`` pass, which places shared
    ``!tile.buffer`` allocations into a per-function arena and claims
    ``!tile.pipeline_state`` rings as ping-pong staging;
  * ``msl_gemm_emit.materialize_apple_simdgroup_tile_msl``, which sizes the
    staged A/B tiles and edge scratch the emitted steel MSL actually declares.

If those two disagree, the IR would authorize a placement the kernel cannot
hold (or refuse one it can). These tests bind them to the same capacity, the
same ping-pong rule, and the same arena arithmetic.

The lit fixtures ``tests/tessera-ir/phase8/apple_threadgroup_pipeline*.mlir``
own the IR-shape and rejection-diagnostic contracts; this file owns the
cross-owner agreement.

This is an Apple-owned compiler-tool proof: it needs ``tessera-opt`` built with
the Apple backend, and no Metal device.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tessera.compiler.apple_fragment import (
    AppleFragmentError,
    select_apple_simdgroup_fragment,
)
from tessera.compiler.apple_target import AppleGPUArch, AppleGPUTargetProfile
from tessera.compiler.msl_gemm_emit import materialize_apple_simdgroup_tile_msl
from tests._support.environment import CompilerToolchain

pytestmark = [pytest.mark.compiler_tool, pytest.mark.compiler_apple]

# The Apple7 (M1-series) profile is this lane's exact device; the threadgroup
# ceiling is uniform across Apple4-Apple10 per the Metal Feature Set Tables.
TARGET = AppleGPUTargetProfile(arch=AppleGPUArch.APPLE7)

_LAYOUT = ('#tile.layout<shard = [32, 16] : [16, 1] on ["m", "m"], '
           "replica = [] : [] on [], offset = 0>")


def _module_with_allocs(byte_sizes: list[int]) -> str:
    allocs = "\n".join(
        f'  %a{i} = tile.alloc {{space = "smem", bytes = {n} : i64, '
        f"layout = {_LAYOUT}}} : !tile.buffer"
        for i, n in enumerate(byte_sizes))
    return f"module {{\n func.func @staged() {{\n{allocs}\n  return\n }}\n}}"


def _place(
    toolchain: CompilerToolchain, module: str
) -> subprocess.CompletedProcess[str]:
    tessera_opt = toolchain.require_tessera_opt("tessera-apple-threadgroup-pipeline")
    return subprocess.run(
        [str(tessera_opt), "-",
         "--pass-pipeline=builtin.module(tessera-apple-threadgroup-pipeline)"],
        input=module, capture_output=True, text=True, check=False, timeout=60)


def _arena_bytes(stdout: str) -> int | None:
    match = re.search(r"tessera_apple\.threadgroup_arena_bytes = (\d+)", stdout)
    return int(match.group(1)) if match else None


def _offsets(stdout: str) -> list[int]:
    return [int(m) for m in re.findall(
        r"tessera_apple\.threadgroup_offset = (\d+)", stdout)]


def test_ir_capacity_matches_the_target_profile(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """The pass's default ceiling is the profile's, not an independent constant."""
    proc = _place(compiler_toolchain, _module_with_allocs([256]))
    assert proc.returncode == 0, proc.stderr
    declared = re.search(
        r"tessera_apple\.threadgroup_capacity_bytes = (\d+)", proc.stdout)
    assert declared is not None, proc.stdout
    assert int(declared.group(1)) == TARGET.threadgroup_memory_capacity_bytes


def test_ir_arena_matches_the_materializer_resource_record(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """Placing the emitter's own staged bytes must reproduce its arena total."""
    artifact = materialize_apple_simdgroup_tile_msl(TARGET, "fp16", 32, 32, 16)
    resources = artifact.resources
    proc = _place(compiler_toolchain, _module_with_allocs([
        resources.staged_a_bytes,
        resources.staged_b_bytes,
        resources.edge_scratch_bytes,
    ]))
    assert proc.returncode == 0, proc.stderr
    assert _arena_bytes(proc.stdout) == resources.total_threadgroup_bytes
    # Placement is packed in declaration order, not aliased.
    assert _offsets(proc.stdout) == [
        0,
        resources.staged_a_bytes,
        resources.staged_a_bytes + resources.staged_b_bytes,
    ]


def test_materializer_refuses_a_descriptor_layout_that_drifts_from_its_tile() -> None:
    """The emitted MSL cannot silently replace compiler-owned staging bytes."""
    with pytest.raises(AppleFragmentError, match="E_PIPE_LAYOUT_MISMATCH"):
        materialize_apple_simdgroup_tile_msl(
            TARGET, "fp16", 32, 32, 16,
            staging_contract={
                "stage_depth": 2,
                "staged_a_bytes": 1,
                "staged_b_bytes": 2048,
                "edge_scratch_bytes": 256,
                "total_threadgroup_bytes": 4352,
            },
        )


def test_both_owners_reject_the_same_over_capacity_tile(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """An over-budget staged tile fails in the emitter and in the IR placer."""
    capacity = TARGET.threadgroup_memory_capacity_bytes
    # 128x128x64 fp16 double-buffered stages far past the 32 KiB ceiling.
    with pytest.raises(AppleFragmentError) as emitter_error:
        materialize_apple_simdgroup_tile_msl(TARGET, "fp16", 128, 128, 64)
    assert "APPLE_FRAGMENT_THREADGROUP_MEMORY_EXCEEDED" in str(emitter_error.value)

    proc = _place(compiler_toolchain, _module_with_allocs([capacity - 1024, 4096]))
    assert proc.returncode != 0
    assert "APPLE_THREADGROUP_MEMORY_EXCEEDED" in proc.stderr


def test_ping_pong_depth_is_the_double_buffering_the_emitter_declares(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """Depth 2 is accepted precisely because the emitted MSL is ping-pong."""
    artifact = materialize_apple_simdgroup_tile_msl(
        TARGET, "fp16", 32, 32, 16, double_buffer=True)
    assert artifact.resources.double_buffered
    # The emitted source declares two staged buffers per operand.
    assert "As[2]" in artifact.msl and "Bs[2]" in artifact.msl

    accepted = _place(
        compiler_toolchain,
        'module {\n func.func @ring() {\n'
        '  %s = tile.pipeline_init {depth = 2 : i64, stage = 0 : i64, '
        'phase = 1 : i64, role = "producer"} : !tile.pipeline_state\n'
        "  return\n }\n}")
    assert accepted.returncode == 0, accepted.stderr
    assert 'tessera_apple.stage_buffering = "ping_pong"' in accepted.stdout

    # Nothing in the emitter can realize a deeper ring, so the IR refuses it
    # rather than narrowing it behind the author's back.
    rejected = _place(
        compiler_toolchain,
        'module {\n func.func @ring() {\n'
        '  %s = tile.pipeline_init {depth = 3 : i64, stage = 0 : i64, '
        'phase = 1 : i64, role = "producer"} : !tile.pipeline_state\n'
        "  return\n }\n}")
    assert rejected.returncode != 0
    assert "APPLE_STAGE_DEPTH_UNSUPPORTED" in rejected.stderr


# Block-scaled formats take the packed A / packed B / scale A / scale B operand
# form; everything else takes the plain A / B form (`MatmulKernelOp::verify`).
_BLOCK_SCALED = {"nvfp4", "fp4_e2m1", "e2m3", "e3m2"}


def _mma_module(storage: str, accum: str) -> str:
    if storage in _BLOCK_SCALED:
        params = ("%a: !llvm.ptr, %b: !llvm.ptr, %sa: !llvm.ptr, "
                  "%sb: !llvm.ptr, %d: !llvm.ptr")
        operands = "%a, %b, %sa, %sb, %d, %m, %n, %k"
        types = ("!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, "
                 "i64, i64, i64")
    else:
        params = "%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr"
        operands = "%a, %b, %d, %m, %n, %k"
        types = "!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64"
    return f"""module {{
 llvm.func @mma({params}, %m: i64, %n: i64, %k: i64) {{
  tile.matmul_kernel {operands} {{
    numeric_policy = {{storage = "{storage}", accum = "{accum}"}},
    mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16,
      a = "{storage}", b = "{storage}", acc = "{accum}",
      a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
    epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
    warps = 1 : i64, staging = "global"
  }} : {types}
  llvm.return
 }}
}}"""


@pytest.mark.parametrize("storage", ["fp16", "bf16"])
def test_apple_mma_accepts_exactly_the_fragment_contract_dtypes(
    compiler_toolchain: CompilerToolchain, storage: str,
) -> None:
    """The IR gate admits a dtype only when the fragment selector does too."""
    # The Python owner accepts it...
    fragment = select_apple_simdgroup_fragment(TARGET, storage)
    assert fragment.accumulator_dtype == "fp32"
    # ...so the C++ capability gate must not reject it.
    proc = _place(compiler_toolchain, _mma_module(storage, "fp32"))
    assert proc.returncode == 0, proc.stderr
    assert "APPLE_MMA_STORAGE_UNSUPPORTED" not in proc.stderr


@pytest.mark.parametrize("storage", ["nvfp4", "fp4_e2m1", "fp6_e3m2", "fp8_e4m3", "int4"])
def test_apple_mma_rejects_sdk_gated_and_unrouted_dtypes(
    compiler_toolchain: CompilerToolchain, storage: str,
) -> None:
    """Low-precision cooperative matrix has no Apple route until the SDK lands.

    Both owners must refuse: a silent pass-through here would let an NVFP4
    descriptor reach Metal lowering with no kernel behind it.
    """
    with pytest.raises(AppleFragmentError) as fragment_error:
        select_apple_simdgroup_fragment(TARGET, storage)
    assert "APPLE_FRAGMENT_UNSUPPORTED_DTYPE" in str(fragment_error.value)

    accum = "int32" if storage == "int4" else "fp32"
    proc = _place(compiler_toolchain, _mma_module(storage, accum))
    assert proc.returncode != 0
    assert "APPLE_MMA_STORAGE_UNSUPPORTED" in proc.stderr


def test_capability_gate_covers_llvm_func_bodies(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """The gate walks any function-like op, not just `func.func`.

    Regression guard: an earlier revision walked `func::FuncOp` only, so every
    rejection silently skipped operations nested in an `llvm.func` body — the
    exact shape the packed-storage fixtures use.
    """
    proc = _place(compiler_toolchain, _mma_module("int4", "int32"))
    assert proc.returncode != 0, "llvm.func bodies must not bypass the gate"
    assert "APPLE_MMA_STORAGE_UNSUPPORTED" in proc.stderr


def test_loop_carried_pipeline_state_is_rooted_through_iter_args(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """A ring threaded by scf.for must not be called unrooted (PR #467 review).

    The canonical GEMM emits `tile.pipeline_advance %arg8` where `%arg8` is the
    K loop's region iter_arg, not the `pipeline_init` result. Membership-testing
    the raw operand reported APPLE_STAGE_UNROOTED_ADVANCE and rejected the very
    shared contract this pass exists to consume.
    """
    tessera_opt = compiler_toolchain.require_tessera_opt("tessera-apple-threadgroup-pipeline")
    gemm = """module {
  func.func @gemm(%a: tensor<64x32xf16>, %b: tensor<32x48xf16>) -> tensor<64x48xf32> {
    %0 = "tessera.matmul"(%a, %b)
        : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
    return %0 : tensor<64x48xf32>
  }
}"""
    proc = subprocess.run(
        [str(tessera_opt), "-", "--tessera-tiling",
         "--tessera-apple-threadgroup-pipeline", "--allow-unregistered-dialect"],
        input=gemm, capture_output=True, text=True, check=False, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "APPLE_STAGE_UNROOTED_ADVANCE" not in proc.stderr
    # The advance reports its rooted ring's mode, not a default.
    assert 'tile.pipeline_advance' in proc.stdout
    assert 'tessera_apple.stage_buffering = "ping_pong"' in proc.stdout


def test_advance_carries_the_rooted_rings_buffering_mode(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """A depth-1 ring's advances say `single`, not an assumed `ping_pong`.

    Stamping every advance ping_pong hands the emitter two contradictory
    physical schedules for one pipeline (PR #467 review).
    """
    proc = _place(
        compiler_toolchain,
        'module {\n func.func @ring() {\n'
        '  %s = tile.pipeline_init {depth = 1 : i64, stage = 0 : i64, '
        'phase = 1 : i64, role = "producer"} : !tile.pipeline_state\n'
        '  %t = tile.pipeline_advance %s '
        ': (!tile.pipeline_state) -> !tile.pipeline_state\n'
        "  return\n }\n}")
    assert proc.returncode == 0, proc.stderr
    advance = [ln for ln in proc.stdout.splitlines() if "pipeline_advance" in ln]
    assert advance and 'tessera_apple.stage_buffering = "single"' in advance[0]


def _mma_descriptor_module(a: str, acc: str, policy: str | None) -> str:
    policy_attr = f'numeric_policy = {{storage = "{policy}", accum = "fp32"}}, ' if policy else ""
    out = "f32" if acc in {"f32", "fp32"} else "i32"
    return f"""module {{
 llvm.func @m(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
              %m: i64, %n: i64, %k: i64) {{
  tile.matmul_kernel %a, %b, %d, %m, %n, %k {{
    {policy_attr}mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 32,
      a = "{a}", b = "{a}", acc = "{acc}",
      a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
    epilogue = #tile.epilogue<bias = false, activation = "none", output = "{out}">,
    warps = 1 : i64, staging = "global"
  }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
  llvm.return
 }}
}}"""


@pytest.mark.parametrize("policy", [None, "fp16"])
def test_mma_descriptor_is_authoritative_over_numeric_policy(
    compiler_toolchain: CompilerToolchain, policy: str | None,
) -> None:
    """An unrouted descriptor cannot pass by omitting or misstating its policy.

    Only `mma` is required by the op verifier, so an int4 descriptor with no
    numeric_policy — or with a laundering fp16 one — previously slipped through
    the Apple capability gate (PR #467 review).
    """
    proc = _place(compiler_toolchain, _mma_descriptor_module("int4", "int32", policy))
    assert proc.returncode != 0, "an int4 cooperative-matrix claim must be refused"
    assert "APPLE_MMA_STORAGE_UNSUPPORTED" in proc.stderr


def test_mma_descriptor_gate_admits_the_real_simdgroup_contract(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """fp16 operands with fp32 accumulation are exactly the Apple route."""
    proc = _place(compiler_toolchain, _mma_descriptor_module("f16", "f32", "fp16"))
    assert proc.returncode == 0, proc.stderr
    assert "APPLE_MMA_STORAGE_UNSUPPORTED" not in proc.stderr


def test_single_buffered_emitter_maps_to_depth_one(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """A non-double-buffered artifact is the 'single' staging mode, not depth 2."""
    artifact = materialize_apple_simdgroup_tile_msl(
        TARGET, "fp16", 32, 32, 16, double_buffer=False)
    assert not artifact.resources.double_buffered
    assert "As[2]" not in artifact.msl

    proc = _place(
        compiler_toolchain,
        'module {\n func.func @ring() {\n'
        '  %s = tile.pipeline_init {depth = 1 : i64, stage = 0 : i64, '
        'phase = 1 : i64, role = "producer"} : !tile.pipeline_state\n'
        "  return\n }\n}")
    assert proc.returncode == 0, proc.stderr
    assert 'tessera_apple.stage_buffering = "single"' in proc.stdout
