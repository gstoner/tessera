"""TracedHardMoELM — a fully tape-differentiable hard top-k MoE language model.

Demonstrates the G1 + G2 autodiff primitives end-to-end:

* **G1** ``ops.embedding`` — trainable token embeddings (gradient scatter-adds
  into the embedding table; previously raw numpy indexing was tape-invisible).
* **G2** ``ops.top_k_routing`` — *differentiable hard top-k routing*: only the
  top-``k`` experts get non-zero gate weight, yet gradient still flows to the
  router through the sparse-softmax jacobian. This is the piece that lets a
  hard-routed MoE (not just the dense soft-MoE of ``TracedMoEPolicy``) train.

Every op in ``logits`` is tape-traced, so ``adamw_step`` updates the embedding
table, router, experts, and head from real gradients. Hard-routing semantics
are exact (zero weight off the top-k); compute here is still dense (all experts
evaluated) — the compute-sparse dispatch would use ``ops.gather``/``scatter``
(also traceable), out of scope for this correctness demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessera import nn, ops


@dataclass(frozen=True)
class TracedHardMoEConfig:
    vocab_size: int = 32
    hidden: int = 16
    num_experts: int = 4
    top_k: int = 2
    expert_ffn: int = 32


class TracedHardMoELM(nn.Module):
    def __init__(self, cfg: TracedHardMoEConfig, *, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        V, H, E, F = cfg.vocab_size, cfg.hidden, cfg.num_experts, cfg.expert_ffn

        def par(shape, scale):
            return nn.Parameter((rng.standard_normal(shape) * scale).astype(np.float32))

        self.embed = par((V, H), 0.02)
        self.w_router = par((H, E), 1.0 / np.sqrt(H))
        for e in range(E):
            setattr(self, f"w_gate_{e}", par((H, F), 1.0 / np.sqrt(H)))
            setattr(self, f"w_down_{e}", par((F, H), 1.0 / np.sqrt(F)))
        self.w_out = par((H, V), 1.0 / np.sqrt(H))
        self._eye = np.eye(E, dtype=np.float32)

    def logits(self, ids):
        """Traced next-token logits ``(N, V)`` for flat token ids ``(N,)``."""
        cfg = self.cfg
        x = ops.embedding(self.embed, np.asarray(ids, np.int64))      # (N, H)  [G1]
        routing = ops.top_k_routing(ops.gemm(x, self.w_router), k=cfg.top_k)  # (N, E) [G2]
        y = None
        for e in range(cfg.num_experts):
            w = ops.gemm(routing, self._eye[:, e:e + 1])             # (N, 1) traced col-select
            wg = getattr(self, f"w_gate_{e}")
            wd = getattr(self, f"w_down_{e}")
            he = ops.gemm(ops.silu_mul(ops.gemm(x, wg), ops.gemm(x, wg)), wd)  # (N, H)
            ye = ops.mul(he, w)
            y = ye if y is None else ops.add(y, ye)
        return ops.gemm(ops.add(x, y), self.w_out)                   # (N, V)


def traced_ce_loss(logits, target_onehot):
    """Traced mean cross-entropy ``-mean_i sum_v onehot[i,v] log_softmax(logits)[i,v]``."""
    oh = np.asarray(target_onehot, np.float32)
    logp = ops.log_softmax(logits)                                  # (N, V)
    per = ops.reduce(ops.mul(logp, oh), op="sum", axis=1)           # (N,)
    neg = np.full(oh.shape[0], -1.0, np.float32)
    return ops.reduce(ops.mul(per, neg), op="mean")                 # scalar
