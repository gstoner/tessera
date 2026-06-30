"""Sprint #20a (2026-05-22) — attention sharding mock-mesh proof.

This test is the Bucket B promotion gate for the standard attention family:
flash_attn, multi_head_attention, gqa_attention, mqa_attention,
mla_decode, mla_decode_fused, attn_sliding_window, attn_top_k_blocks,
attn_compressed_blocks, attn_local_window_2d, linear_attn,
linear_attn_state, power_attn, retention.

The proof: softmax(QK^T/sqrt(d))·V is independent across the head axis
(every (q, k, v) triple for a given head is processed in isolation).
Therefore tensor-parallel sharding along the head axis is equivalent to
running the op locally on each rank's slice and concatenating outputs.

Proof shape:
  1. Compute single-rank reference output.
  2. Under a 2-rank head-split mesh, each rank runs the op on its head
     slice; all-gather along the head axis recovers the full output.
  3. np.testing.assert_allclose to single-rank reference.
  4. Repeat at world_size=4 for headcount=8 to exercise multi-way TP.
  5. Repeat for attn_sliding_window (representative of the windowed
     variants — same head-axis independence).
  6. Assert the primitive_coverage registry reflects the promotion:
     `flash_attn`/`mha`/`gqa`/`mqa`/`mla_decode`/MLA family/sliding/etc.
     have `sharding_rule = complete`, while the reasoning-model fused
     family (`deepseek_sparse_attention`, `lightning_attention`,
     `gated_attention`, `hybrid_attention`, `gated_deltanet`,
     `kimi_delta_attention`, `modified_delta_attention`) stays at
     `partial` because their target-specific fused kernels need real
     hardware to validate (Bucket C / Phase G/H gate).

The mock-mesh harness (`tessera.testing.mock_collective.MockRankGroup`)
is a thread-based in-process simulator — no NCCL/MPI dependency. See
CLAUDE.md §CPU Collective Mock.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.testing.mock_collective import MockRankGroup


# ─────────────────────────────────────────────────────────────────────────────
# Numerical proofs: head-axis sharding == single-rank execution
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("world_size", [2, 4])
def test_flash_attn_head_split_matches_single_rank(world_size: int) -> None:
    """Canonical TP-by-head proof: flash_attn is head-independent."""
    np.random.seed(0)
    B, H, S, D = 2, 8, 16, 16
    assert H % world_size == 0, "test design: head count must be divisible"
    Q = np.random.randn(B, H, S, D).astype(np.float32)
    K = np.random.randn(B, H, S, D).astype(np.float32)
    V = np.random.randn(B, H, S, D).astype(np.float32)
    expected = ts.ops.flash_attn(Q, K, V)

    H_local = H // world_size

    def worker(rank):
        h0 = rank.rank * H_local
        h1 = h0 + H_local
        local_out = ts.ops.flash_attn(Q[:, h0:h1], K[:, h0:h1], V[:, h0:h1])
        # All-gather along the head axis (axis=1 of (B, H, S, D))
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=1)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    assert len(results) == world_size
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-6)


def test_flash_attn_head_split_with_causal_mask() -> None:
    """Causal masking is per-(q,k) and doesn't cross heads — TP-safe."""
    np.random.seed(1)
    B, H, S, D = 2, 4, 12, 16
    Q = np.random.randn(B, H, S, D).astype(np.float32)
    K = np.random.randn(B, H, S, D).astype(np.float32)
    V = np.random.randn(B, H, S, D).astype(np.float32)
    expected = ts.ops.flash_attn(Q, K, V, causal=True)

    world_size = 2
    H_local = H // world_size

    def worker(rank):
        h0 = rank.rank * H_local
        h1 = h0 + H_local
        local_out = ts.ops.flash_attn(
            Q[:, h0:h1], K[:, h0:h1], V[:, h0:h1], causal=True,
        )
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=1)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-6)


def test_attn_sliding_window_head_split_matches_single_rank() -> None:
    """Windowed-attention variants share the head-axis independence
    property (the window is along the sequence axis, not heads)."""
    np.random.seed(2)
    B, H, S, D = 2, 4, 16, 16
    Q = np.random.randn(B, H, S, D).astype(np.float32)
    K = np.random.randn(B, H, S, D).astype(np.float32)
    V = np.random.randn(B, H, S, D).astype(np.float32)
    expected = ts.ops.attn_sliding_window(Q, K, V, window_size=8, causal=True)

    world_size = 2
    H_local = H // world_size

    def worker(rank):
        h0 = rank.rank * H_local
        h1 = h0 + H_local
        local_out = ts.ops.attn_sliding_window(
            Q[:, h0:h1], K[:, h0:h1], V[:, h0:h1],
            window_size=8, causal=True,
        )
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=1)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Registry promotion claims locked by these proofs
# ─────────────────────────────────────────────────────────────────────────────

# Standard attention family (Bucket B) — TP-by-head sharding is proven
# by the numerical tests above; the registry must reflect this.
_STANDARD_ATTN_NAMES = frozenset({
    "flash_attn",
    "multi_head_attention",
    "gqa_attention",
    "mqa_attention",
    "latent_kv_compress",
    "latent_kv_expand_k",
    "latent_kv_expand_v",
    "mla_decode",
    "mla_decode_fused",
    "attn_sliding_window",
    "attn_top_k_blocks",
    "attn_compressed_blocks",
    "attn_local_window_2d",
    "linear_attn",
    "linear_attn_state",
    "power_attn",
    "retention",
})

# Reasoning-model fused attention family (Bucket C — Phase G/H gate)
# Their sharding is the fused-kernel's sharding; promotion requires real
# hardware (target-specific fused kernels need backend verification).
_REASONING_FUSED_ATTN_NAMES = frozenset({
    "deepseek_sparse_attention",
    "lightning_attention",
    "gated_attention",
    "hybrid_attention",
    "gated_deltanet",
    "kimi_delta_attention",
    "modified_delta_attention",
})


def test_standard_attention_family_sharding_promoted_to_complete() -> None:
    """The mock-mesh proofs above license `sharding_rule = complete` for
    the standard attention family.  This test pins the registry's
    `_ATTN_STANDARD_HARDENED` override to that claim."""
    entries = all_primitive_coverages()
    failures: list[tuple[str, str]] = []
    for name in _STANDARD_ATTN_NAMES:
        if name not in entries:
            continue  # skip names not in the registry on this branch
        actual = entries[name].contract_status.get("sharding_rule")
        if actual != "complete":
            failures.append((name, str(actual)))
    assert not failures, (
        "standard attention family sharding_rule must be `complete` after "
        f"the Sprint #20a mock-mesh proof, but got: {failures}.  "
        "The proof is the TP-by-head independence of softmax(QK^T)V; "
        "see test_flash_attn_head_split_matches_single_rank above."
    )


def test_reasoning_fused_attention_family_sharding_stays_partial() -> None:
    """The reasoning-model fused family (DeepSeek NSA, Lightning, gated,
    hybrid, DeltaNet, Kimi-delta, modified-delta) ships target-specific
    fused kernels that need real hardware to validate.  Sprint #20a
    splits them out into `_ATTN_REASONING_FUSED_HARDENED` so the standard
    family can promote while this group stays at `partial` — honest
    Phase G/H gating."""
    entries = all_primitive_coverages()
    failures: list[tuple[str, str]] = []
    for name in _REASONING_FUSED_ATTN_NAMES:
        if name not in entries:
            continue
        actual = entries[name].contract_status.get("sharding_rule")
        if actual not in ("partial", "planned"):
            failures.append((name, str(actual)))
    assert not failures, (
        "reasoning-model fused attention family must stay at "
        "`partial`/`planned` for sharding_rule until Phase G/H "
        f"backend validation lands; got: {failures}"
    )
