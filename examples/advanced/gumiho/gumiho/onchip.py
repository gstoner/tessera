"""On-chip speculative step — compose the Phase-G kernels end to end.

A greedy speculative step has three parts; this wires the device kernels for
each together:

  1. **Draft** — the serial head (Rung 1, an MPSGraph `forLoop`) + the parallel
     heads produce candidate paths.  (`gumiho.serial_draft_forloop` /
     `build_draft`.)
  2. **Verify** — the target runs once over the draft tree (MPSGraph), giving its
     greedy token at every predicting position.  (`build_draft` →
     `DraftBundle.node_target_argmax`.)
  3. **Accept** — the *dynamic* control flow: per path, accept draft tokens while
     they match the target's greedy token (early break = variable trip count),
     keep the longest accepted prefix, emit the bonus.  This is the part
     MPSGraph can't express; it runs as the single MSL kernel
     `runtime.apple_gpu_msl_spec_accept` (Phase-G Rung 3).

So the static draft/verify (MPSGraph) feed the dynamic accept (MSL) — the whole
on-device speculative-step accept, validated against a host reference. See
docs/apple_gpu_control_flow_lowering.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessera import runtime as R

from .backend import make_backend
from .config import GumihoConfig, tiny_config
from .draft import build_draft
from .model import ParallelHeads, SerialHead, TargetModel, make_weights


@dataclass(frozen=True)
class OnchipStepSummary:
    backend: str
    num_paths: int
    best_path: int
    accepted_length: int
    bonus: int
    accepted_tokens: list
    matches_host: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"backend={self.backend} paths={self.num_paths} | "
                f"best_path={self.best_path} accepted={self.accepted_length} "
                f"{list(self.accepted_tokens)} bonus={self.bonus} | "
                f"matches_host={self.matches_host}")


def _host_greedy_accept(draft_paths: np.ndarray, target_greedy: np.ndarray):
    """Reference for the on-device accept: per path accept-while-match, keep the
    longest accepted prefix, return ``(best_path, len, bonus, accepted)``."""
    P, depth = draft_paths.shape
    bp, bl, bb = 0, -1, 0
    for p in range(P):
        length = 0
        for i in range(depth):
            if int(draft_paths[p, i]) == int(target_greedy[p, i]):
                length += 1
            else:
                break
        if length > bl:
            bl, bp = length, p
            bb = int(target_greedy[p, length])
    return bp, bl, bb, [int(draft_paths[bp, i]) for i in range(bl)]


def run_onchip_step_demo(cfg: GumihoConfig | None = None, *, seed: int = 0,
                         target: str = "apple_gpu",
                         distill_steps: int = 0) -> OnchipStepSummary:
    """Run one greedy speculative step composing the device kernels (draft +
    verify on MPSGraph, accept on the Rung-3 MSL kernel) and validate the accept
    against a host reference. ``distill_steps>0`` first distills the draft on the
    step's own context so the accepted length is non-zero (otherwise the
    untrained draft rarely matches the target's greedy token)."""
    cfg = cfg or tiny_config()
    weights = make_weights(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    context = rng.integers(0, cfg.vocab, size=cfg.context_len, dtype=np.int64)
    if distill_steps > 0:
        from .training import distill, trajectory_contexts
        ctxs = trajectory_contexts(cfg, weights, context[None], horizon=16)
        weights = distill(cfg, weights, contexts=ctxs, serial_steps=distill_steps,
                          parallel_steps=distill_steps, lr=0.05, seed=seed)
    be = make_backend(target, eps=cfg.rmsnorm_eps)

    tgt = TargetModel(weights, cfg)
    serial = SerialHead(weights, cfg)
    parallel = ParallelHeads(weights, cfg)
    last_hidden, _ = tgt.forward(be, context)

    # 1+2. draft + tree verify (serial forLoop + parallel + one target pass).
    bundle = build_draft(be, cfg, tgt, serial, parallel,
                         context_tokens=context, last_hidden=last_hidden[-1])

    # Shape the accept inputs from the verified tree:
    #   draft_paths[p, j]  = the j-th draft token of path p
    #   target_greedy[p,j] = the target's greedy token at the node predicting it
    draft_paths = np.ascontiguousarray(bundle.paths[:, 1:], np.int32)
    target_greedy = np.ascontiguousarray(
        bundle.node_target_argmax[bundle.path_node_ids], np.int32)

    # 3. Rung-3 on-device dynamic accept (MSL kernel) + host cross-check.
    bp, acc_len, bonus, accepted = R.apple_gpu_msl_spec_accept(
        draft_paths, target_greedy, np)
    ref = _host_greedy_accept(draft_paths, target_greedy)

    return OnchipStepSummary(
        backend=be.name, num_paths=bundle.num_paths, best_path=int(bp),
        accepted_length=int(acc_len), bonus=int(bonus),
        accepted_tokens=[int(t) for t in accepted],
        matches_host=((bp, acc_len, bonus, accepted) == ref))
