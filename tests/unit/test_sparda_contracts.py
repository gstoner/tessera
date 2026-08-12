"""Executable mathematical contract for the SparDA review (arXiv 2606.04511).

Verifies the results recorded in docs/audit/compiler/SPARDA_REVIEW.md Part II
against the *code's* semantics (NVlabs/SparDA @ c8e235b), not the paper's prose
-- the two differ, and the divergences are recorded in the review's Sec. I.4.

Four contracts, each a reimplementation trap if taken on faith:

1.  The CUDA max-pool remap (kernel 5, stride 4, padding 1) over the compressed
    axis is *exactly* the block-overlap set of a 64-token block with 32/16
    mean-pool windows.
2.  The legacy stage-1 causal rule uses C-truncating division and is strictly
    causal; Python's floor division gives a DIFFERENT answer for small t.
3.  Softmax elision at inference is rank-equivalent to the softmax training
    objective (top-k is invariant to strictly monotone row transforms).
4.  Full-window-only compression leaves a decode lag of exactly
    (seqlen - kernel) mod stride tokens -- bounded by stride - 1 = 15, not the
    31 or 47 that looser readings suggest -- always inside the forced local
    window.

These numpy references are the oracles for any later Tessera compressed-key /
block-selection lane (review Sec. III.3); keep them dependency-free (no torch
on the fleet).
"""

from __future__ import annotations

import numpy as np
import pytest

# InfLLM-V2 substrate constants (models/minicpm/modeling_minicpm.py).
KERNEL = 32   # mean-pool window over keys
STRIDE = 16   # mean-pool stride (kernel > stride => windows overlap)
BLOCK = 64    # attention block, in tokens


def c_div(a: int, b: int) -> int:
    """C++ integer division: truncation toward zero.

    Python's ``//`` floors, which differs for negative numerators. The stage-1
    mask (csrc/flash_attn/src/mask.h) is C++, so the contract below must use
    this and not ``//``.
    """
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def pooled_window(block_idx: int, num_compressed: int) -> set[int]:
    """Compressed-key indices the CUDA max-pool reads for ``block_idx``.

    Mirrors max_pooling_1d.cuh: start = k*stride - padding (clamped at 0),
    end = start + kernel_size (clamped at the sequence end), with the Python
    remap stride = BLOCK // STRIDE, kernel = 5, padding = 1
    (infllm_v2/max_pooling_1d.py).
    """
    stride_p, kernel_p, pad_p = BLOCK // STRIDE, 5, 1
    start = max(block_idx * stride_p - pad_p, 0)
    end = min(block_idx * stride_p - pad_p + kernel_p, num_compressed)
    return set(range(start, max(start, end)))


def overlapping_compressed(block_idx: int, num_compressed: int) -> set[int]:
    """Ground truth: compressed keys whose 32/16 window intersects the block."""
    lo, hi = block_idx * BLOCK, block_idx * BLOCK + BLOCK
    return {
        j
        for j in range(num_compressed)
        if max(j * STRIDE, lo) < min(j * STRIDE + KERNEL, hi)
    }


def stage1_visible(t: int, j: int, causal_k_offset: int = 0) -> bool:
    """Legacy stage-1 visibility of compressed key ``j`` for query token ``t``.

    mask.h:285-288 -- col < (orig_row_idx - stride + 1) / stride + offset.
    """
    return j < c_div(t - STRIDE + 1, STRIDE) + causal_k_offset


def window_fully_past(t: int, j: int) -> bool:
    """Whether compressed key ``j``'s entire source window precedes ``t``."""
    return (j * STRIDE + KERNEL - 1) <= t


def num_full_windows(seqlen: int) -> int:
    """Compressed keys produced by full-window-only compression.

    calc_chunks_with_stride drops any trailing window shorter than KERNEL
    (valid_chunk_mask), so the tail is EXCLUDED, not partially pooled.
    """
    return max(0, (seqlen - KERNEL) // STRIDE + 1)


# ---------------------------------------------------------------------------
# 1. Pooling remap == exact block-overlap set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_idx", range(64))
def test_pooling_remap_is_exact_block_overlap(block_idx: int) -> None:
    n_comp = 512
    assert pooled_window(block_idx, n_comp) == overlapping_compressed(block_idx, n_comp)


def test_pooling_remap_tiles_without_fractional_boundary() -> None:
    """BLOCK / STRIDE == 4 exactly, so grid->block pooling never splits a cell."""
    assert BLOCK % STRIDE == 0
    # Interior blocks read exactly 5 compressed cells (4 fully inside + 1
    # straddling predecessor); block 0 reads 4 because of the clamp at zero.
    assert len(pooled_window(0, 512)) == 4
    for b in range(1, 64):
        assert len(pooled_window(b, 512)) == 5


# ---------------------------------------------------------------------------
# 2. Stage-1 causality: strictly causal, and C-division matters
# ---------------------------------------------------------------------------


def test_stage1_rule_is_whole_window_past() -> None:
    max_t = 4096
    for t in range(max_t):
        for j in range((max_t // STRIDE) + 2):
            assert stage1_visible(t, j) == window_fully_past(t, j), (t, j)


def test_python_floordiv_would_break_the_stage1_contract() -> None:
    """Pins the reimplementation trap: // is not the C++ semantics here.

    For t < STRIDE - 1 the numerator (t - 15) is negative, where Python floors
    toward -inf and C++ truncates toward zero.
    """
    disagreements = [t for t in range(STRIDE) if (t - STRIDE + 1) // STRIDE != c_div(t - STRIDE + 1, STRIDE)]
    assert disagreements, "expected C-vs-Python divergence for small t"
    t = disagreements[0]
    # Python's floor admits no keys (limit -1); C truncation gives limit 0.
    assert (t - STRIDE + 1) // STRIDE == -1
    assert c_div(t - STRIDE + 1, STRIDE) == 0


# ---------------------------------------------------------------------------
# 3. Softmax elision is rank-equivalent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(50))
def test_topk_invariant_under_softmax(seed: int) -> None:
    """Training matches softmax(s/sqrt(d)); inference takes top-k of raw s.

    modeling_minicpm.py:2261-2268 scores with a bare bmm -- no softmax, no
    1/sqrt(d) (review Sec. I.4). Legal because softmax is strictly monotone
    per row, so it cannot reorder.
    """
    rng = np.random.default_rng(seed)
    n, k, d = 257, 17, 128
    s = rng.standard_normal(n).astype(np.float32)
    shifted = s / np.sqrt(d)
    p = np.exp(shifted - shifted.max())
    p /= p.sum()
    assert set(np.argpartition(-s, k)[:k]) == set(np.argpartition(-p, k)[:k])


def test_topk_invariance_needs_strict_monotonicity() -> None:
    """Negative control: a non-monotone rescale DOES reorder.

    Guards the contract from being read as "any rescale is safe" -- e.g. a
    per-key bias (NOSA's CIS lane) is not rank-preserving.
    """
    s = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    biased = s + np.array([0.0, 5.0, 0.0], dtype=np.float32)
    assert int(np.argmax(s)) != int(np.argmax(biased))


# ---------------------------------------------------------------------------
# 4. Decode compressed-key lag is bounded and locally covered
# ---------------------------------------------------------------------------


def test_compressed_key_lag_has_closed_form_and_is_bounded_by_stride() -> None:
    """Uncovered tail == (seqlen - KERNEL) mod STRIDE, so it is < STRIDE.

    Derivation: with n = floor((L-K)/S) + 1 full windows, the last covered
    token is (n-1)*S + K - 1, so the gap is L - K - S*floor((L-K)/S), i.e.
    exactly (L - K) mod S. The worst case is therefore STRIDE - 1 == 15.

    Both looser readings are wrong and this test exists to pin that: "KERNEL-1
    = 31" (ignores that a window forms every STRIDE tokens) and
    "KERNEL-1+STRIDE = 47" (double-counts -- a bound that holds but is never
    attained). An earlier draft of the review asserted 47 and passed only
    because it tested `<=` against its own loose bound.
    """
    worst = 0
    for seqlen in range(KERNEL, 8192):
        n = num_full_windows(seqlen)
        covered_until = (n - 1) * STRIDE + KERNEL if n else 0
        gap = seqlen - covered_until
        assert gap == (seqlen - KERNEL) % STRIDE, seqlen
        worst = max(worst, gap)
    assert worst == STRIDE - 1 == 15


@pytest.mark.parametrize("local_blocks", [16, 32])
def test_lag_always_inside_forced_local_window(local_blocks: int) -> None:
    """The forced local window covers every token the compressed grid misses.

    This is the invariant that makes the one-step-stale K~_{l+1} cache safe at
    decode: the top-k path never needs the newest block because B_local covers
    it unconditionally. The paper never states it as an invariant.
    """
    for seqlen in range(KERNEL, 8192):
        n = num_full_windows(seqlen)
        covered_until = (n - 1) * STRIDE + KERNEL if n else 0
        window_start = max(0, ((seqlen - 1) // BLOCK - local_blocks) * BLOCK)
        assert covered_until >= window_start or seqlen <= local_blocks * BLOCK
