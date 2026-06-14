"""TracedMoEPolicy — a compact, fully tape-differentiable MoE policy.

This is the *trainable* MoE used by the RL loop: every op in its forward is a
tape-traced ``tessera.ops.*`` call and every weight is a 2-D ``nn.Parameter``,
so reverse-mode autodiff flows gradients to all router and expert parameters
(verified end-to-end in ``tests/unit/test_train_grpo.py``).

It is a **soft** MoE: all experts are evaluated and combined by the full router
softmax (no data-dependent top-k packing). This is the differentiable
counterpart to the reference ``Qwen3MoEModel`` — whose top-k hard routing runs
in numpy and is tape-invisible in the v1 tape (the same boundary PithTrain draws
by excluding the data-dependent MoE from ``torch.compile``). Use this model when
you need gradients; use the reference models for shape/inference graphs.

Design note (the tape gotcha this encodes): experts are individual 2-D
Parameters, NOT slices of one 3-D Parameter, because slicing a Parameter buffer
detaches it from the tape. The per-expert column-select of the router softmax is
done with a one-hot ``gemm`` (a constant matrix) rather than numpy indexing, for
the same reason.
"""

from __future__ import annotations

import numpy as np

from tessera import nn, ops


def _np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


class TracedMoEPolicy(nn.Module):
    """state features -> hidden -> soft-MoE FFN -> action logits, fully traced."""

    def __init__(self, state_dim: int, hidden: int, num_experts: int,
                 expert_ffn: int, num_actions: int, *, seed: int = 0) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        rng = np.random.default_rng(seed)
        d, H, E, F, A = state_dim, hidden, num_experts, expert_ffn, num_actions

        def par(shape, scale):
            return nn.Parameter((rng.standard_normal(shape) * scale).astype(np.float32))

        self.w_in = par((d, H), 1.0 / np.sqrt(d))
        self.w_router = par((H, E), 1.0 / np.sqrt(H))
        # Per-expert 2-D Parameters (registered individually so named_parameters
        # discovers them and the tape keeps each connected).
        for e in range(E):
            setattr(self, f"w_gate_{e}", par((H, F), 1.0 / np.sqrt(H)))
            setattr(self, f"w_down_{e}", par((F, H), 1.0 / np.sqrt(F)))
        self.w_out = par((H, A), 1.0 / np.sqrt(H))
        # Constant one-hot selector columns (not a Parameter).
        self._eye = np.eye(E, dtype=np.float32)

    def logits(self, states):
        """Traced action logits ``(N, A)``. Call inside an autodiff tape for grads."""
        x = np.asarray(states, dtype=np.float32)
        h = ops.silu_mul(ops.gemm(x, self.w_in), ops.gemm(x, self.w_in))   # (N, H)
        probs = ops.softmax(ops.gemm(h, self.w_router))                    # (N, E)
        y = None
        for e in range(self.num_experts):
            w = ops.gemm(probs, self._eye[:, e:e + 1])                     # (N, 1)
            wg = getattr(self, f"w_gate_{e}")
            wd = getattr(self, f"w_down_{e}")
            he = ops.gemm(ops.silu_mul(ops.gemm(h, wg), ops.gemm(h, wg)), wd)  # (N, H)
            ye = ops.mul(he, w)
            y = ye if y is None else ops.add(y, ye)
        return ops.gemm(ops.add(h, y), self.w_out)                         # (N, A)

    def logp(self, states, actions_onehot):
        """Traced log-prob of the taken actions ``(N,)``.

        Uses a one-hot action matrix + reduce (not numpy gather) so the chain
        stays tape-connected.
        """
        actions = np.asarray(actions_onehot, dtype=np.float32)
        logp_all = ops.log_softmax(self.logits(states))                   # (N, A)
        return ops.reduce(ops.mul(logp_all, actions), op="sum", axis=1)    # (N,)
