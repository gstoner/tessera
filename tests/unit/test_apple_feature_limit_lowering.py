"""P1 (2026-06-02) — Apple feature-limit-guided lowering.

APPLE_AUDIT "Next Work" item 3: *wire Apple feature limits into
schedule/tile/kernel selection*. The first concrete wire-up: the
threadgroup-tiled matmul→softmax kernel's N ceiling is no longer a
magic ``8192`` — it is **derived** from the per-arch threadgroup-memory
budget (`apple_target._ArchDefaults.threadgroup_memory_bytes`), since
the kernel holds one row of N fp32 scores in threadgroup memory.

These tests pin the derivation (so a future arch-limit change flows
through automatically) and confirm the runtime consumes it.
"""

from __future__ import annotations

import pytest

from tessera.compiler import apple_target as at


# ── The derivation itself ─────────────────────────────────────────────

def test_tiled_softmax_cap_is_threadgroup_memory_over_4():
    """N_max = threadgroup_memory_bytes // 4 (one fp32 score per slot)."""
    for arch in at.AppleGPUArch:
        floor = at.apple_arch_defaults(arch).threadgroup_memory_bytes
        cap = at.apple_threadgroup_tiled_softmax_n_cap(arch)
        assert cap == floor // 4


def test_tiled_softmax_cap_preserves_legacy_8192_on_current_arches():
    """Every current Apple arch has a 32 KB floor ⇒ the cap equals the
    8192 the runtime used to hardcode (behavior-preserving wire-up)."""
    for arch in at.AppleGPUArch:
        assert at.apple_threadgroup_tiled_softmax_n_cap(arch) == 8192


def test_tiled_softmax_cap_scales_with_higher_live_budget():
    """A live SKU reporting a larger threadgroup-memory budget raises
    the cap; the static floor is never undercut."""
    big = at.AppleRuntimeLimits(
        max_threadgroup_memory_bytes=64 * 1024,
        supports_packaged_ml=True, supports_metal4=True,
        apple_gpu_family=10)
    cap = at.apple_threadgroup_tiled_softmax_n_cap(
        at.AppleGPUArch.APPLE10, runtime_limits=big)
    assert cap == 64 * 1024 // 4  # 16384

    # A probe BELOW the floor must not drop the cap below the portable floor.
    small = at.AppleRuntimeLimits(
        max_threadgroup_memory_bytes=16 * 1024,  # < 32 KB floor
        supports_packaged_ml=True, supports_metal4=True,
        apple_gpu_family=10)
    cap2 = at.apple_threadgroup_tiled_softmax_n_cap(
        at.AppleGPUArch.APPLE10, runtime_limits=small)
    assert cap2 == 8192  # static floor wins


def test_elem_bytes_parameter():
    """Half-precision scores (2 bytes) would double the cap — the helper
    honors the bytes-per-score parameter."""
    cap = at.apple_threadgroup_tiled_softmax_n_cap(
        at.AppleGPUArch.APPLE10, elem_bytes=2)
    assert cap == 32 * 1024 // 2  # 16384


# ── Runtime consumes the derived cap ──────────────────────────────────

def test_runtime_helper_returns_derived_cap():
    from tessera import runtime as rt
    cap = rt._apple_threadgroup_tiled_softmax_n_cap()
    # On this host (static floor or a live probe ≥ floor) the cap is at
    # least the portable 8192 floor and is a multiple of 4-byte scores.
    assert cap >= 8192
    assert cap % 256 == 0


def test_runtime_helper_is_cached():
    from tessera import runtime as rt
    a = rt._apple_threadgroup_tiled_softmax_n_cap()
    b = rt._apple_threadgroup_tiled_softmax_n_cap()
    assert a == b


# ── P2 (2026-06-09) — fused-chain caps + bf16 gate + threads-per-row ──
# APPLE_AUDIT "Still Open" feature-table-driven selection: bf16 gating,
# head_dim ceilings, threadgroup sizing now consult the feature table.

def test_fused_chain_score_cap_is_stack_budget_over_elem_bytes():
    for arch in at.AppleGPUArch:
        assert at.apple_fused_chain_score_cap(arch) == 1024 // 4 == 256
        assert at.apple_fused_chain_score_cap(arch, elem_bytes=8) == 128


def test_flash_attn_head_dim_cap_matches_score_cap():
    for arch in at.AppleGPUArch:
        assert (at.apple_flash_attn_head_dim_cap(arch)
                == at.apple_fused_chain_score_cap(arch))


def test_threads_per_row_is_simdgroup_size():
    for arch in at.AppleGPUArch:
        assert (at.apple_threadgroup_threads_per_row(arch)
                == at.apple_arch_defaults(arch).simdgroup_size == 32)


def test_supports_native_bf16_static_arch_defaults():
    # MTLDataType.bfloat is Apple6+, so native bf16 is present on every
    # modeled arch — including M1 / Apple7.
    for arch in at.AppleGPUArch:
        assert at.apple_supports_native_bf16(arch), arch.name


def test_supports_native_bf16_live_family_wins():
    apple7 = at.AppleRuntimeLimits(0, False, False, apple_gpu_family=1007)
    pre = at.AppleRuntimeLimits(0, False, False, apple_gpu_family=1005)
    unknown = at.AppleRuntimeLimits(0, False, False, apple_gpu_family=-1)
    # A live family at/above the Apple6 bfloat floor (>= 1006) confirms
    # native bf16 — M1 / Apple7 (1007) qualifies.
    assert at.apple_supports_native_bf16(at.AppleGPUArch.APPLE7, runtime_limits=apple7)
    # A live family below the bfloat floor gates off even on a modern
    # static arch.
    assert not at.apple_supports_native_bf16(at.AppleGPUArch.APPLE10, runtime_limits=pre)
    # Unknown family falls back to the static arch default (native on apple7+).
    assert at.apple_supports_native_bf16(at.AppleGPUArch.APPLE7, runtime_limits=unknown)


def test_runtime_consumes_fused_score_cap_and_bf16_gate():
    from tessera import runtime as rt
    assert rt._apple_fused_score_cap() == at.apple_fused_chain_score_cap()
    assert rt._apple_fused_score_cap() is rt._apple_fused_score_cap()  # cached
    assert isinstance(rt._apple_gpu_supports_native_bf16(), bool)
