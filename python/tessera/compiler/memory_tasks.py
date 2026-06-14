"""Structure-keyed long-memory task family (EVALUATOR_PLAN.md §9; companion to
``attention_tasks``).

The long-memory benchmarks (RULER / LongMemEval / MemoryArena) reduce to a small
set of on-device primitives: *score* the query against the bank, *select* the
best entry, and *read* a (soft) weighted value.  This module grades those pieces
on the Apple-GPU-native path against independent numpy oracles, one grader task
per (op × bank-size).  Importing it registers the tasks into ``compiler_grader``
(the standard import-to-register pattern).

What this proves vs. what stays a gap:

* ``memory/score``    — ``query·keysᵀ`` runs on Metal (the matmul lane).
* ``memory/top1``     — top-1 select (argmax) runs on Metal (the argreduce lane).
* ``memory/soft_read``— full-bank ``softmax(QKᵀ/√d)·V`` runs on Metal (attention).

So *on-device scoring / selection / soft-read* are real (rung 8 here).  Hard
top-k (k>1) has no GPU kernel and the bank does not yet stay resident across
decode steps — those remain ``segmented_topk_gpu`` / ``resident_state_handle``
in ``benchmarks.long_memory_core.MEMORY_PRIMITIVE_GAPS``.
"""

from __future__ import annotations

from typing import Any

import tessera as ts

from tessera.compiler.compiler_grader import CheckResult, task
from tessera.compiler.evaluator import Rung, evaluate


def _score(query, keys_t):
    return ts.ops.matmul(query, keys_t)


def _top1(scores):
    return ts.ops.argmax(scores, axis=-1)


def _soft_read(q, k, v):
    return ts.ops.flash_attn(q, k, v)


_SCORE = ts.jit(target="apple_gpu")(_score)
_TOP1 = ts.jit(target="apple_gpu")(_top1)
_SOFT = ts.jit(target="apple_gpu")(_soft_read)

# (n_entries, key_dim, n_queries) — the structure-keyed bank-size axis.
_MEMORY_MATRIX: tuple[tuple[int, int, int], ...] = (
    (64, 32, 4),
    (256, 64, 8),
    (1024, 128, 4),
)


def _rung8(label: str, verdict: Any) -> list[CheckResult]:
    return [CheckResult(label, verdict.rung is Rung.HARDWARE_VERIFIED, verdict.detail)]


def _make_score_task(n: int, d: int, nq: int) -> Any:
    def _run(rng: Any) -> list[CheckResult]:
        import numpy as np

        keys = rng.standard_normal((n, d)).astype(np.float32)
        q = rng.standard_normal((nq, d)).astype(np.float32)
        kt = np.ascontiguousarray(keys.T)
        verdict = evaluate("apple_gpu", _SCORE, (q, kt), q @ keys.T,
                           rtol=5e-3, atol=1e-3)
        return _rung8(f"score/n{n} rung8+match", verdict)

    return _run


def _make_top1_task(n: int, d: int, nq: int) -> Any:
    def _run(rng: Any) -> list[CheckResult]:
        import numpy as np

        scores = rng.standard_normal((nq, n)).astype(np.float32)
        oracle = np.argmax(scores, axis=-1).astype(np.int64)
        verdict = evaluate("apple_gpu", _TOP1, (scores,), oracle, exact=True)
        return _rung8(f"top1/n{n} rung8+match", verdict)

    return _run


def _make_soft_read_task(n: int, d: int, nq: int) -> Any:
    def _run(rng: Any) -> list[CheckResult]:
        import numpy as np

        q = rng.standard_normal((1, 1, nq, d)).astype(np.float32)
        k = rng.standard_normal((1, 1, n, d)).astype(np.float32)
        v = rng.standard_normal((1, 1, n, d)).astype(np.float32)
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(d)
        m = scores.max(-1, keepdims=True)
        p = np.exp(scores - m)
        p = p / p.sum(-1, keepdims=True)
        oracle = np.einsum("bhqk,bhkd->bhqd", p, v)
        verdict = evaluate("apple_gpu", _SOFT, (q, k, v), oracle, rtol=5e-3, atol=1e-3)
        return _rung8(f"soft_read/n{n} rung8+match", verdict)

    return _run


def memory_task_names() -> list[str]:
    names: list[str] = []
    for (n, _d, _nq) in _MEMORY_MATRIX:
        names += [f"memory/score/n{n}", f"memory/top1/n{n}", f"memory/soft_read/n{n}"]
    return names


# Register one grader task per (op × bank-size) cell at import.
for _n, _d, _nq in _MEMORY_MATRIX:
    task(f"memory/score/n{_n}")(_make_score_task(_n, _d, _nq))
    task(f"memory/top1/n{_n}")(_make_top1_task(_n, _d, _nq))
    task(f"memory/soft_read/n{_n}")(_make_soft_read_task(_n, _d, _nq))
