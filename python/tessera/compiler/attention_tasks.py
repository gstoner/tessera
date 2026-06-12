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
