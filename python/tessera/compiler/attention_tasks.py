"""Phase E3 — Long-Context-Attention-style structure-keyed task family
(docs/audit/compiler/EVALUATOR_PLAN.md §9; LongCA-bench, ICLR'26).

LongCA's lesson: do not grade attention as one op — parameterize it by the axes
that actually determine cost/behavior (mask pattern × sequence length × variant),
because a single ``flash_attn`` cell hides where it breaks. This registers a
**matrix of grader tasks**, one per (mask × seqlen), each graded by the
Evaluator's vertical oracle against an independent causal/full attention
reference. Importing this module adds the tasks to ``compiler_grader`` (the
standard import-to-register pattern); nothing in the grader imports this, so it
stays optional.

Scope: full + causal masks on the Apple-GPU-native ``flash_attn`` path. Sliding-
window / block-sparse masks and the distributed-scale axis are follow-ups (and
the distributed axis needs the multi-rank lane).
"""

from __future__ import annotations

from typing import Any

import tessera as ts

from tessera.compiler.compiler_grader import CheckResult, task
from tessera.compiler.evaluator import Rung, evaluate


def _fa(q, k, v):
    return ts.ops.flash_attn(q, k, v)


def _fa_causal(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True)


_FA = ts.jit(target="apple_gpu")(_fa)
_FA_CAUSAL = ts.jit(target="apple_gpu")(_fa_causal)

# (mask, B, H, S, D) — the structure-keyed matrix.
_MATRIX: tuple[tuple[str, int, int, int, int], ...] = (
    ("full", 1, 2, 64, 16),
    ("full", 1, 2, 128, 16),
    ("full", 1, 4, 256, 32),
    ("causal", 1, 2, 64, 16),
    ("causal", 1, 2, 128, 16),
    ("causal", 1, 4, 256, 32),
)


def _attn_reference(q: Any, k: Any, v: Any, *, causal: bool) -> Any:
    """Independent numpy attention oracle: softmax(QKᵀ/√d)·V, optional causal."""
    import numpy as np

    d = q.shape[-1]
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(d)
    if causal:
        s = scores.shape[-1]
        future = np.triu(np.ones((s, s), dtype=bool), k=1)   # keys j > query i
        scores = np.where(future, -1e30, scores)
    m = scores.max(-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, v)


def _make_task(mask: str, b: int, h: int, s: int, d: int) -> Any:
    causal = mask == "causal"
    fn = _FA_CAUSAL if causal else _FA

    def _run(rng: Any) -> list[CheckResult]:
        import numpy as np

        q = rng.standard_normal((b, h, s, d)).astype(np.float32)
        k = rng.standard_normal((b, h, s, d)).astype(np.float32)
        v = rng.standard_normal((b, h, s, d)).astype(np.float32)
        oracle = _attn_reference(q, k, v, causal=causal)
        verdict = evaluate("apple_gpu", fn, (q, k, v), oracle, rtol=5e-3, atol=1e-3)
        return [CheckResult(
            f"{mask}/s{s} rung7+match",
            verdict.rung is Rung.HARDWARE_VERIFIED,
            verdict.detail,
        )]

    return _run


def attention_task_names() -> list[str]:
    return [f"attention/{mask}/s{s}" for (mask, _b, _h, s, _d) in _MATRIX]


# Register one grader task per (mask × seqlen) cell at import.
for _mask, _b, _h, _s, _d in _MATRIX:
    task(f"attention/{_mask}/s{_s}")(_make_task(_mask, _b, _h, _s, _d))


# ─────────────────────────────────────────────────────────────────────────────
# Structured-mask extension — sliding-window + block-sparse (RULER structure axis)
#
# ``flash_attn`` carries an optional additive ``attn_bias`` operand (see
# ``tessera.flash_attn`` and the DFlash sliding-layer substrate): the score
# matrix becomes ``softmax(scale·QKᵀ + attn_bias)·V``.  A structured mask is just
# a ``0 / -inf`` bias, so we reuse the SAME Apple-GPU flash_attn path and vary
# only the bias operand — "vary the structure, not the op", which is exactly
# RULER's lesson.  The achieved Evaluator rung is recorded in each check's
# detail: a cell that lands below ``HARDWARE_VERIFIED`` is the honest signal that
# a *native* masked-attention kernel (vs. the broadcast-bias reference path) is
# the next runtime gap — not a bug to hide.
# ─────────────────────────────────────────────────────────────────────────────

_NEG = -1.0e30


def _fa_bias(q, k, v, bias):
    return ts.ops.flash_attn(q, k, v, attn_bias=bias)


_FA_BIAS = ts.jit(target="apple_gpu")(_fa_bias)


def _sliding_window_bias(s: int, window: int, *, causal: bool) -> Any:
    """``0`` inside a band of ``window`` keys, ``-inf`` outside.

    Returns a ``(1, S, S)`` additive bias that broadcasts over the batch·head
    leading dims of the ``(B, H, S, D)`` score matrix.
    """
    import numpy as np

    i = np.arange(s)[:, None]
    j = np.arange(s)[None, :]
    keep = (j <= i) if causal else np.ones((s, s), dtype=bool)
    keep = keep & ((i - j) < window)              # within `window` past keys
    if not causal:
        keep = keep & ((j - i) < window)          # symmetric band
    return np.where(keep, 0.0, _NEG).astype(np.float32)[None]


def _block_sparse_bias(s: int, block: int) -> Any:
    """Block-diagonal mask: each query attends only within its own contiguous
    ``block`` of keys.  Returns a ``(1, S, S)`` additive bias."""
    import numpy as np

    blk = np.arange(s) // block
    keep = blk[:, None] == blk[None, :]
    return np.where(keep, 0.0, _NEG).astype(np.float32)[None]


def structured_mask_bias(mask: str, s: int, param: int) -> Any:
    """Build the additive ``(1, S, S)`` bias for a structured ``mask`` kind."""
    if mask == "sliding":
        return _sliding_window_bias(s, param, causal=True)
    if mask == "block":
        return _block_sparse_bias(s, param)
    raise ValueError(f"unknown structured mask {mask!r}")


def _attn_reference_biased(q: Any, k: Any, v: Any, bias: Any) -> Any:
    """Independent oracle: ``softmax(QKᵀ/√d + bias)·V`` over ``(B, H, S, D)``."""
    import numpy as np

    d = q.shape[-1]
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(d)
    scores = scores + bias[:, None]               # (1,S,S) → broadcast over heads
    m = scores.max(-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, v)


# (mask, B, H, S, D, param) — param = window (sliding) or block size (block).
_STRUCTURED_MATRIX: tuple[tuple[str, int, int, int, int, int], ...] = (
    ("sliding", 1, 2, 128, 16, 32),
    ("sliding", 1, 4, 256, 32, 64),
    ("block", 1, 2, 128, 16, 32),
    ("block", 1, 4, 256, 32, 64),
)


def _make_structured_task(
    mask: str, b: int, h: int, s: int, d: int, p: int
) -> Any:
    def _run(rng: Any) -> list[CheckResult]:
        import numpy as np

        q = rng.standard_normal((b, h, s, d)).astype(np.float32)
        k = rng.standard_normal((b, h, s, d)).astype(np.float32)
        v = rng.standard_normal((b, h, s, d)).astype(np.float32)
        bias = structured_mask_bias(mask, s, p)
        oracle = _attn_reference_biased(q, k, v, bias)
        verdict = evaluate(
            "apple_gpu", _FA_BIAS, (q, k, v, bias), oracle, rtol=5e-3, atol=1e-3
        )
        return [CheckResult(
            f"{mask}/s{s}/p{p} rung8+match",
            verdict.rung is Rung.HARDWARE_VERIFIED,
            verdict.detail,
        )]

    return _run


def structured_attention_task_names() -> list[str]:
    return [
        f"attention/{mask}/s{s}/p{p}"
        for (mask, _b, _h, s, _d, p) in _STRUCTURED_MATRIX
    ]


# Register one grader task per (structured-mask × seqlen × param) cell at import.
for _smask, _sb, _sh, _ss, _sd, _sp in _STRUCTURED_MATRIX:
    task(f"attention/{_smask}/s{_ss}/p{_sp}")(
        _make_structured_task(_smask, _sb, _sh, _ss, _sd, _sp)
    )
