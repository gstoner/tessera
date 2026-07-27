"""APPLE-TILE-2 — the shared canonical M/N/K GEMM loop reaches an Apple route.

`CORE-GEMM-KLOOP-2026-07-25` made the shared Graph/Schedule->Tile contraction an
explicit M/N/K `scf.for` nest with an FP32 loop-carried accumulator, staged
`!tile.pipeline_state`, and ragged zero-pad guards. Apple ran `TilingPass` in
value mode only, so that contract had no Apple consumer at all.

`tessera-apple-canonical-gemm` recognizes the nest and re-forms it as one
`simdgroup_matrix` dispatch. The loop states the *semantics* (reduce over K in
FP32, zero-pad the ragged tail); the Apple steel GEMM already implements exactly
that reduction with its own cooperative staging, so reproducing the loop
statement-by-statement in MSL would be slower and no more correct.

**Standing incumbent rule.** Recognizing the loop changes no production route.
Value-mode Accelerate/MPS remains the incumbent for every measured row until a
two-run paired corpus shows the simdgroup route winning in the intended timing
domain. These tests prove the loop *reaches* an Apple route and that the route
is numerically correct on the exact device — not that it should be selected.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tessera.compiler.apple_target import AppleGPUArch, AppleGPUTargetProfile
from tessera.compiler.msl_gemm_emit import (
    apple_block_for_canonical_tile,
    dispatch_apple_simdgroup_tile_f16,
    materialize_apple_simdgroup_tile_msl,
)
from tests._support.environment import CompilerToolchain

pytestmark = [pytest.mark.compiler_tool, pytest.mark.compiler_apple]

TARGET = AppleGPUTargetProfile(arch=AppleGPUArch.APPLE7)


def _gemm_module(m: int, k: int, n: int, storage: str, accum: str = "f32") -> str:
    return f"""module {{
  func.func @gemm(%a: tensor<{m}x{k}x{storage}>, %b: tensor<{k}x{n}x{storage}>)
      -> tensor<{m}x{n}x{accum}> {{
    %0 = "tessera.matmul"(%a, %b)
        : (tensor<{m}x{k}x{storage}>, tensor<{k}x{n}x{storage}>)
        -> tensor<{m}x{n}x{accum}>
    return %0 : tensor<{m}x{n}x{accum}>
  }}
}}"""


def _lower(
    toolchain: CompilerToolchain, module: str
) -> subprocess.CompletedProcess[str]:
    """Tile through the *shared* canonical contract, then let Apple claim it."""
    tessera_opt = toolchain.require_tessera_opt()
    return subprocess.run(
        [str(tessera_opt), "-", "--tessera-tiling",
         "--tessera-apple-canonical-gemm", "--allow-unregistered-dialect"],
        input=module, capture_output=True, text=True, check=False, timeout=60)


def _descriptor(stdout: str) -> dict[str, str]:
    call = [line for line in stdout.splitlines()
            if "tessera_apple.gpu.kernel_call" in line]
    assert call, f"no Apple dispatch in:\n{stdout}"
    return dict(re.findall(r'([\w.]+) = "?([\w.]+)"?', call[0]))


def test_canonical_loop_becomes_one_apple_simdgroup_dispatch(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """The three-deep nest collapses to a single dispatch, not a loop of them."""
    proc = _lower(compiler_toolchain, _gemm_module(64, 32, 48, "f16"))
    assert proc.returncode == 0, proc.stderr
    # The nest is consumed, not left beside the dispatch.
    assert "scf.for" not in proc.stdout
    assert proc.stdout.count("tessera_apple.gpu.kernel_call") == 1

    descriptor = _descriptor(proc.stdout)
    assert descriptor["op_kind"] == "tile_simdgroup_gemm"
    assert descriptor["symbol"] == "tessera_apple_gpu_tile_simdgroup_gemm_f16"
    assert descriptor["status"] == "executable"
    # Provenance: the dispatch names the shared contract it answers.
    assert descriptor["tessera_apple.canonical_k_loop"] == "true"
    assert descriptor["tessera_apple.accumulate"] == "fp32"
    # The loop's own tile decision is carried, not re-derived downstream.
    assert descriptor["tessera_apple.tile_m"] == "16"
    assert descriptor["tessera_apple.tile_n"] == "16"
    assert descriptor["tessera_apple.tile_k"] == "16"
    # The ragged-tail guard is part of the contract Apple must honor.
    assert descriptor["tessera_apple.ragged_zero_pad"] == "true"


def test_bf16_canonical_loop_selects_the_bf16_symbol(
    compiler_toolchain: CompilerToolchain,
) -> None:
    proc = _lower(compiler_toolchain, _gemm_module(32, 16, 32, "bf16"))
    assert proc.returncode == 0, proc.stderr
    descriptor = _descriptor(proc.stdout)
    assert descriptor["symbol"] == "tessera_apple_gpu_tile_simdgroup_gemm_bf16"
    assert descriptor["dtype"] == "bf16"


def test_f32_canonical_loop_is_refused_not_silently_reformed(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """simdgroup_matrix has no f32 operand form, so f32 must not be claimed.

    This is the boundary that keeps the incumbent honest: f32 GEMM stays on the
    Accelerate/MPS value route rather than being quietly rerouted.
    """
    proc = _lower(compiler_toolchain, _gemm_module(32, 16, 32, "f32"))
    assert proc.returncode != 0
    assert "APPLE_CANONICAL_GEMM_DTYPE_UNSUPPORTED" in proc.stderr


def test_ordinary_loop_nest_is_not_misclaimed(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """Only the marked canonical reduction is claimed.

    A user-written loop containing a matmul carries no `canonical_k_step`
    marker and no staged pipeline state, so it must pass through untouched.
    """
    tessera_opt = compiler_toolchain.require_tessera_opt()
    module = """module {
  func.func @loop(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>,
                  %init: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init)
        -> (tensor<16x16xf32>) {
      %m = "tessera.matmul"(%a, %b)
          : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf32>
      scf.yield %m : tensor<16x16xf32>
    }
    return %r : tensor<16x16xf32>
  }
}"""
    proc = subprocess.run(
        [str(tessera_opt), "-", "--tessera-apple-canonical-gemm",
         "--allow-unregistered-dialect"],
        input=module, capture_output=True, text=True, check=False, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "tessera_apple.gpu.kernel_call" not in proc.stdout
    assert "scf.for" in proc.stdout


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("m,k,n", [(16, 16, 16), (32, 16, 32), (13, 16, 11)])
def test_canonical_loop_dispatch_executes_and_matches_the_oracle(
    compiler_toolchain: CompilerToolchain, m: int, k: int, n: int,
) -> None:
    """End to end: shared canonical loop -> Apple schedule -> Metal -> oracle.

    The tile decision the *compiler* put in the descriptor drives the emitted
    MSL, so this proves the loop-derived route executes, not merely that some
    simdgroup kernel does. The ragged `13x16x11` row exercises the zero-pad
    guard the canonical contract requires.
    """
    import numpy as np
    from tests._support.apple import require_apple_metal

    require_apple_metal()

    proc = _lower(compiler_toolchain, _gemm_module(m, k, n, "f16"))
    assert proc.returncode == 0, proc.stderr
    descriptor = _descriptor(proc.stdout)
    assert descriptor["tessera_apple.canonical_k_loop"] == "true"

    # The loop's tile is a *logical* step (clamped to the extent); Apple owns
    # rounding it to a physical threadgroup block on fragment granularity.
    block = apple_block_for_canonical_tile(
        int(descriptor["tessera_apple.tile_m"]),
        int(descriptor["tessera_apple.tile_n"]),
        int(descriptor["tessera_apple.tile_k"]),
    )
    artifact = materialize_apple_simdgroup_tile_msl(
        TARGET, "fp16", block.m, block.n, block.k)
    assert artifact.resources.total_threadgroup_bytes <= (
        TARGET.threadgroup_memory_capacity_bytes)

    a = (np.arange(m * k, dtype=np.float32).reshape(m, k) / 16).astype(np.float16)
    b = (np.arange(k * n, dtype=np.float32).reshape(k, n) / 32).astype(np.float16)
    out, native, record = dispatch_apple_simdgroup_tile_f16(
        artifact, a, b, return_provenance=True)

    # A numerically correct host recomputation cannot earn this rung.
    assert native is True
    assert record.device_time_ns is not None and record.device_time_ns > 0
    np.testing.assert_allclose(
        out, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-5, atol=1e-5)


# ── APPLE-ATTN-STREAM-1 ──────────────────────────────────────────────────────
#
# The streaming-attention consumer re-forms the shared KV-block recurrence into
# a dispatch on `tessera_apple_gpu_flash_attn_*` — the same ABI family the
# incumbent Graph-IR-direct route already uses. So parity has two halves:
# the compiler must carry the shared boundary semantics onto the descriptor,
# and that ABI must compute them correctly on the device.


def _attn_module(sq: int, sk: int, d: int, storage: str, causal: str) -> str:
    return f"""module attributes {{tessera.ir.version = "1.0",
    tessera.target = {{sm = 90 : i32, warps = 4 : i32, smem = 233472 : i64,
                      pipeline_stages = 2 : i32}}}} {{
  func.func @attn(%Q: tensor<{sq}x{d}x{storage}>, %K: tensor<{sk}x{d}x{storage}>,
                  %V: tensor<{sk}x{d}x{storage}>) -> tensor<{sq}x{d}xf32> {{
    %o = "tessera.flash_attn"(%Q, %K, %V)
        <{{operandSegmentSizes = array<i32: 1, 1, 1, 0>}}> {{
      causal = {causal}, head_dim = {d} : i64,
      tessera.tile_q = 64 : i32, tessera.tile_kv = 64 : i32
    }} : (tensor<{sq}x{d}x{storage}>, tensor<{sk}x{d}x{storage}>,
          tensor<{sk}x{d}x{storage}>) -> tensor<{sq}x{d}xf32>
    return %o : tensor<{sq}x{d}xf32>
  }}
}}"""


def _lower_attn(
    toolchain: CompilerToolchain, module: str
) -> subprocess.CompletedProcess[str]:
    tessera_opt = toolchain.require_tessera_opt()
    return subprocess.run(
        [str(tessera_opt), "-", "--allow-unregistered-dialect",
         "--tessera-tile-ir-lowering=tile-q=64 tile-kv=64 sm=90",
         "--tessera-apple-streaming-attention"],
        input=module, capture_output=True, text=True, check=False, timeout=60)


@pytest.mark.parametrize("causal", ["true", "false"])
def test_streaming_attention_carries_the_shared_boundary_semantics(
    compiler_toolchain: CompilerToolchain, causal: str,
) -> None:
    """Causal state reaches the dispatch from `boundary_mask`, not re-derived."""
    proc = _lower_attn(compiler_toolchain, _attn_module(64, 130, 64, "bf16", causal))
    assert proc.returncode == 0, proc.stderr
    descriptor = _descriptor(proc.stdout)
    assert descriptor["op_kind"] == "flash_attn"
    assert descriptor["tessera_apple.streaming_recurrence"] == "true"
    assert descriptor["tessera_apple.causal"] == causal
    # The logical KV length survives the ragged zero-fill the shared lowering
    # inserts (130 padded to 192), so masking cannot read the padding.
    assert descriptor["tessera_apple.logical_sk"] == "130"
    assert descriptor["tessera_apple.head_dim"] == "64"
    # The recurrence and its depth-3 staging are gone.
    assert "scf.for" not in proc.stdout
    assert "tile.pipeline_init" not in proc.stdout


@pytest.mark.hardware_apple_gpu
def test_streaming_attention_target_abi_matches_the_oracle_on_metal() -> None:
    """The ABI the re-formed dispatch targets computes attention correctly.

    This proves the second half of parity: the compiler-selected symbol family
    is numerically right on this device. The first half — that the descriptor
    carries the shared boundary semantics — is the host-free test above.
    """
    import numpy as np
    import tessera as ts
    from tests._support.apple import require_apple_metal

    require_apple_metal()

    @ts.jit(target="apple_gpu")
    def attention(q, k, v):
        return ts.ops.flash_attn(q, k, v)

    rng = np.random.default_rng(19)
    q = rng.standard_normal((1, 2, 8, 16), dtype=np.float32)
    k = rng.standard_normal((1, 2, 12, 16), dtype=np.float32)
    v = rng.standard_normal((1, 2, 12, 16), dtype=np.float32)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(16.0)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = np.matmul(exp / exp.sum(axis=-1, keepdims=True), v)
    np.testing.assert_allclose(attention(q, k, v), expected, rtol=1e-4, atol=1e-4)
