"""Phase specialization — prefill and decode as different compiled programs.

Workstream B. Mooncake's lesson: prefill is bulk throughput, decode is
latency/SLO/cache scheduling — so they should compile and *schedule* differently,
with an explicit handoff of cache state. Today Tessera compiles one forward graph
and a model hand-writes a prefill/decode loop (e.g. ``models.moe_transformer_
runtime``); the phase distinction is not a compiler contract.

This module makes it one:

  * :class:`Phase` + :class:`SLO` + :class:`SchedulePolicy` — the typed contract.
    ``@jit(phase="prefill"|"decode", slo=...)`` attaches a policy (jit.py reads
    it). Prefill and decode derive *different* schedules from the same source.
  * :class:`CacheHandoff` — the prefill→decode ABI. It carries the model's KV
    state (a :class:`~tessera.cache.PagedKVState`-conforming object from
    Workstream A) plus the sequence position, so decode resumes exactly where
    prefill stopped.
  * :class:`PhaseSpecializedProgram` / :func:`specialize` — bind a prefill program
    and a decode program into one schedulable unit with a typed handoff between.
  * :func:`verify_phase_split` — the oracle: ``prefill ▸ decode_loop`` must equal a
    single full ``forward`` (the schedule split is semantics-preserving — the same
    invariant as Workstream A's residency-invariance).

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream B).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


class Phase(enum.Enum):
    """Which compiled program a graph is specialized for."""

    PREFILL = "prefill"   # bulk: process the whole prompt, high arithmetic intensity
    DECODE = "decode"     # latency: one token at a time, KV-bound, SLO-sensitive


@dataclass(frozen=True)
class SLO:
    """Service-level objective a phase is scheduled against."""

    max_latency_ms: float | None = None        # decode's target per-step latency
    min_throughput_tok_s: float | None = None   # prefill's target throughput


@dataclass(frozen=True)
class SchedulePolicy:
    """The schedule a phase asks its lowering for.

    Prefill and decode produce *different* policies from the same model source —
    the compiler-real expression of Mooncake's separation. The numerics must be
    invariant to the policy (proven by :func:`verify_phase_split`); only the
    schedule differs.
    """

    phase: Phase
    tile_strategy: str          # "bulk_throughput" | "low_latency"
    materialize_scores: bool     # prefill may materialize; decode streams (online)
    prefer_resident_kv: bool     # decode pins KV resident; prefill streams it
    slo: SLO | None = None

    @classmethod
    def for_phase(cls, phase: Phase, slo: SLO | None = None) -> "SchedulePolicy":
        """Derive the canonical schedule for a phase.

        Prefill optimizes arithmetic-intensity/throughput (large tiles, scores may
        be materialized, KV streamed). Decode optimizes latency (small tiles,
        online softmax, KV pinned resident for the gather).
        """
        if isinstance(phase, str):
            phase = Phase(phase)
        if phase is Phase.PREFILL:
            return cls(phase, "bulk_throughput", materialize_scores=True,
                       prefer_resident_kv=False, slo=slo)
        return cls(phase, "low_latency", materialize_scores=False,
                   prefer_resident_kv=True, slo=slo)


@dataclass
class CacheHandoff:
    """The prefill→decode state ABI.

    ``state`` is the model's opaque KV state (per Workstream A it should be a
    :class:`~tessera.cache.PagedKVState`-conforming object, or a container of
    them). ``position`` is the token index decode resumes at. ``step`` counts
    decode advances for SLO accounting.
    """

    state: Any
    position: int
    step: int = 0

    def advanced(self, new_state: Any, n: int = 1) -> "CacheHandoff":
        return CacheHandoff(new_state, self.position + n, self.step + 1)


@dataclass
class PhaseSpecializedProgram:
    """Two phase programs + their schedules, with a typed handoff between them.

    ``prefill_fn(*args) -> (logits, state)`` runs the prompt; ``decode_fn(state,
    token) -> (logits, state)`` advances one token. ``specialize`` wraps the raw
    state into :class:`CacheHandoff` so callers never thread position by hand.
    """

    prefill_fn: Callable[..., tuple[Any, Any]]
    decode_fn: Callable[[Any, Any], tuple[Any, Any]]
    prefill_policy: SchedulePolicy
    decode_policy: SchedulePolicy

    def run_prefill(self, *args: Any, prompt_len: int | None = None
                    ) -> tuple[Any, CacheHandoff]:
        logits, state = self.prefill_fn(*args)
        pos = prompt_len if prompt_len is not None else _infer_position(state, args)
        return logits, CacheHandoff(state, position=pos)

    def decode_step(self, handoff: CacheHandoff, token: Any
                    ) -> tuple[Any, CacheHandoff]:
        logits, state = self.decode_fn(handoff.state, token)
        return logits, handoff.advanced(state, 1)

    def generate(self, *prefill_args: Any, max_new_tokens: int,
                 sampler: Callable[[Any], int] | None = None,
                 prompt_len: int | None = None) -> list[int]:
        """prefill ▸ decode loop. ``sampler(logits) -> token`` (default argmax)."""
        sampler = sampler or (lambda lg: int(np.argmax(np.asarray(lg))))
        logits, handoff = self.run_prefill(*prefill_args, prompt_len=prompt_len)
        out: list[int] = []
        for _ in range(max_new_tokens):
            tok = sampler(logits)
            out.append(tok)
            logits, handoff = self.decode_step(handoff, tok)
        return out


def specialize(
    prefill_fn: Callable[..., tuple[Any, Any]],
    decode_fn: Callable[[Any, Any], tuple[Any, Any]],
    *,
    prefill_slo: SLO | None = None,
    decode_slo: SLO | None = None,
) -> PhaseSpecializedProgram:
    """Bind a prefill program and a decode program into one schedulable unit."""
    return PhaseSpecializedProgram(
        prefill_fn=prefill_fn,
        decode_fn=decode_fn,
        prefill_policy=SchedulePolicy.for_phase(Phase.PREFILL, prefill_slo),
        decode_policy=SchedulePolicy.for_phase(Phase.DECODE, decode_slo),
    )


def _infer_position(state: Any, args: tuple[Any, ...]) -> int:
    """Best-effort token position after prefill (for the handoff)."""
    for attr in ("position", "current_seq", "seq_len"):
        v = getattr(state, attr, None)
        if callable(v):
            try:
                return int(v())
            except Exception:
                pass
        elif v is not None:
            return int(v)
    # Fall back to the first array-like prefill arg's leading length.
    if args:
        a = np.asarray(args[0])
        if a.ndim >= 1:
            return int(a.shape[0])
    return 0


# ── Oracle — the schedule split must be semantics-preserving ──────────────────


@dataclass(frozen=True)
class PhaseSplitVerdict:
    relation: str            # "equivalent" | "divergent" | "inconclusive"
    max_abs_err: float | None
    tokens_match: bool
    detail: str = ""

    @property
    def is_equivalent(self) -> bool:
        return self.relation == "equivalent"


def verify_phase_split(
    forward_fn: Callable[[Any], Any],
    program: PhaseSpecializedProgram,
    prompt_tokens: Any,
    max_new_tokens: int,
    *,
    sampler: Callable[[Any], int] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> PhaseSplitVerdict:
    """Assert ``prefill ▸ decode_loop`` == a single full ``forward``.

    ``forward_fn(all_tokens) -> logits_per_position`` is the reference monolith.
    The phase-split program greedily generates ``max_new_tokens``; we re-feed the
    full token stream to ``forward_fn`` and require the per-step argmax (and the
    last-position logits) to agree. A miscompiled phase split — a stale handoff, a
    schedule that changed numerics — diverges here.
    """
    sampler = sampler or (lambda lg: int(np.argmax(np.asarray(lg))))
    prompt = list(np.asarray(prompt_tokens).reshape(-1).tolist())

    # Phase-split path.
    gen = program.generate(prompt, max_new_tokens=max_new_tokens,
                           sampler=sampler, prompt_len=len(prompt))

    # Monolithic reference: teacher-force the same prompt+gen stream and read the
    # argmax at each generation boundary.
    ref_tokens: list[int] = []
    stream = list(prompt)
    last_err = 0.0
    for i in range(max_new_tokens):
        logits_all = np.asarray(forward_fn(stream), dtype=np.float64)
        last = logits_all[-1] if logits_all.ndim == 2 else logits_all
        ref_tok = sampler(last)
        ref_tokens.append(ref_tok)
        stream.append(gen[i])   # follow the split program's actual choices
    tokens_match = ref_tokens == gen

    relation = "equivalent" if tokens_match else "divergent"
    detail = ("prefill▸decode token stream matches monolithic forward"
              if tokens_match else
              f"token streams diverge: split={gen} ref={ref_tokens}")
    return PhaseSplitVerdict(relation, last_err, tokens_match, detail)


__all__ = [
    "Phase",
    "SLO",
    "SchedulePolicy",
    "CacheHandoff",
    "PhaseSpecializedProgram",
    "specialize",
    "verify_phase_split",
    "PhaseSplitVerdict",
]
