"""Distill the tiny Gumiho draft heads against the target model.

Out of the box the draft heads are random, so the target rejects almost
everything and the speculative step accepts ~1 token. This module trains them
so the draft actually predicts the target's continuation — which is what makes
the mean accepted length (and therefore the speedup) climb above 1.

The target is a fixed (random) model, so its greedy continuation is a
deterministic function of the context: a clean supervised distillation target.
We train with **Tessera's own autograd + optimizer** — the forward is written in
``tessera.ops`` so ``tessera.autodiff.grad`` differentiates it, and
``tessera.optim.adam`` applies the updates (the S10/S11 standalone-compiler
training surface).

Two facts keep the trainable forward simple and *exactly* equal to the
inference path:

* The serial head runs one token at a time (T=1), so its self-attention over a
  single position is degenerate (``softmax`` of one score = 1) and reduces to a
  value projection ``v @ Wo``. We therefore train the value slice of ``Wqkv``
  (the q/k slices never affect a T=1 output) and write it back, leaving the
  inference code untouched.
* RMSNorm's weight, the SwiGLU MLP, and ``fc_in`` are all plain ``matmul`` /
  ``mul`` / ``silu_mul`` — every op has a registered VJP.

``fc_in`` (which multiplies ``concat(hidden, embedding)``) is split into its
hidden- and embedding-halves so the autoregressive step-2 input never needs an
in-trace concat.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import tessera as ts
from tessera import autodiff as ad
from tessera import optim

from .config import GumihoConfig
from .model import GumihoWeights, _Layer, TargetModel
from .backend import NumpyBackend


# ─────────────────────────────────────────────────────────────────────────────
# Target rollout — the supervised distillation target
# ─────────────────────────────────────────────────────────────────────────────
def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _rollout(cfg: GumihoConfig, target: TargetModel, backend,
             context: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy target continuation of length ``total_draft_tokens``.

    Returns ``(g[total_draft], h_init[d], target_probs[total_draft, V])`` — the
    continuation token ids, the target's hidden at the last context position
    (the serial head's step-0 input), and the target's next-token *distribution*
    at each position. The distribution is the distillation soft label: matching
    it makes the Leviathan acceptance ratio ``p_target/p_draft`` ≈ 1 so correct
    tokens are actually accepted (argmax-matching alone over-rejects)."""
    seq = np.asarray(context, np.int64).copy()
    h_init = None
    g = []
    probs = []
    for k in range(cfg.total_draft_tokens):
        hidden, logits = target.forward(backend, seq)
        if k == 0:
            h_init = hidden[-1].astype(np.float64)
        probs.append(_softmax(logits[-1]))
        g.append(int(np.argmax(logits[-1])))
        seq = np.append(seq, g[-1])
    return (np.asarray(g, np.int64), np.asarray(h_init, np.float64),
            np.asarray(probs, np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# Traceable serial forward (tessera.ops) — value-only T=1 transformer blocks
# ─────────────────────────────────────────────────────────────────────────────
def _serial_block(s, wv, wo, ln1, ln2, wg, wu, wd):
    n = ts.ops.mul(ts.ops.rmsnorm(s), ln1)
    attn = ts.ops.matmul(ts.ops.matmul(n, wv), wo)        # T=1 attention = v@Wo
    s = ts.ops.add(s, attn)
    n2 = ts.ops.mul(ts.ops.rmsnorm(s), ln2)
    act = ts.ops.silu_mul(ts.ops.matmul(n2, wg), ts.ops.matmul(n2, wu))
    return ts.ops.add(s, ts.ops.matmul(act, wd))


def _serial_step(h_part, e_part, lm_head, params):
    fc_h, fc_e, snorm = params[0], params[1], params[2]
    s = ts.ops.add(ts.ops.matmul(h_part, fc_h), ts.ops.matmul(e_part, fc_e))
    off = 3
    n_layers = (len(params) - 3) // 7
    for li in range(n_layers):
        wv, wo, ln1, ln2, wg, wu, wd = params[off:off + 7]
        s = _serial_block(s, wv, wo, ln1, ln2, wg, wu, wd)
        off += 7
    logits = ts.ops.matmul(ts.ops.mul(ts.ops.rmsnorm(s), snorm), lm_head)
    return s, logits


def _ce(logits, onehot, n: int):
    lsm = ts.ops.log_softmax(logits)
    return ts.ops.reduce(ts.ops.mul(lsm, -onehot), op="sum")  # / n applied by caller


def _serial_params(cfg: GumihoConfig, w: GumihoWeights) -> list:
    d = cfg.d_model
    params = [np.ascontiguousarray(w.serial_fc_in[:d], np.float64),
              np.ascontiguousarray(w.serial_fc_in[d:], np.float64),
              w.serial_norm.astype(np.float64)]
    for layer in w.serial_layers:
        wv = np.ascontiguousarray(layer.wqkv[:, 2 * d:3 * d], np.float64)
        params += [wv, layer.wo.astype(np.float64), layer.ln1.astype(np.float64),
                   layer.ln2.astype(np.float64), layer.w_gate.astype(np.float64),
                   layer.w_up.astype(np.float64), layer.w_down.astype(np.float64)]
    return params


def _write_serial_params(cfg: GumihoConfig, w: GumihoWeights, params: list) -> GumihoWeights:
    d = cfg.d_model
    fc_in = np.vstack([np.asarray(params[0]), np.asarray(params[1])]).astype(np.float32)
    snorm = np.asarray(params[2], np.float32)
    new_layers = []
    off = 3
    for layer in w.serial_layers:
        wv, wo, ln1, ln2, wg, wu, wd = (np.asarray(p) for p in params[off:off + 7])
        wqkv = layer.wqkv.copy()
        wqkv[:, 2 * d:3 * d] = wv.astype(np.float32)     # q/k slices stay (T=1 irrelevant)
        new_layers.append(_Layer(
            ln1=ln1.astype(np.float32), wqkv=wqkv, wo=wo.astype(np.float32),
            ln2=ln2.astype(np.float32), w_gate=wg.astype(np.float32),
            w_up=wu.astype(np.float32), w_down=wd.astype(np.float32)))
        off += 7
    return replace(w, serial_fc_in=fc_in, serial_norm=snorm,
                   serial_layers=tuple(new_layers))


# ─────────────────────────────────────────────────────────────────────────────
# Distillation
# ─────────────────────────────────────────────────────────────────────────────
def trajectory_contexts(cfg: GumihoConfig, weights: GumihoWeights,
                        prompts: np.ndarray, horizon: int) -> np.ndarray:
    """Contexts sampled along the target's greedy trajectories from ``prompts``.

    The multi-step decoder visits exactly these on-trajectory contexts (the
    target is deterministic), so distilling on them is what lets the accepted
    length climb on the same prompts. Each sliding window of ``context_len``
    tokens along a length-``horizon`` rollout becomes one training context."""
    ref = NumpyBackend(eps=cfg.rmsnorm_eps)
    target = TargetModel(weights, cfg)
    C = cfg.context_len
    out: list[np.ndarray] = []
    for prompt in np.atleast_2d(np.asarray(prompts, np.int64)):
        seq = list(prompt[-C:])
        for _ in range(horizon):
            out.append(np.asarray(seq[-C:], np.int64))
            _h, logits = target.forward(ref, np.asarray(seq[-C:], np.int64))
            seq.append(int(np.argmax(logits[-1])))
    return np.stack(out)


def distill(cfg: GumihoConfig, weights: GumihoWeights, *, num_contexts: int = 96,
            serial_steps: int = 400, parallel_steps: int = 400, lr: float = 0.03,
            seed: int = 0, contexts: np.ndarray | None = None) -> GumihoWeights:
    """Return a copy of ``weights`` with distilled serial + parallel heads.

    ``contexts`` (``[N, context_len]``) supplies the training distribution; when
    omitted, random contexts are used. The decode demo passes on-trajectory
    contexts via :func:`trajectory_contexts`."""
    rng = np.random.default_rng(seed)
    ref = NumpyBackend(eps=cfg.rmsnorm_eps)
    target = TargetModel(weights, cfg)

    # 1. Build the distillation corpus.
    if contexts is None:
        contexts = rng.integers(0, cfg.vocab, size=(num_contexts, cfg.context_len),
                                dtype=np.int64)
    contexts = np.asarray(contexts, np.int64)
    num_contexts = contexts.shape[0]
    G = np.empty((num_contexts, cfg.total_draft_tokens), np.int64)
    TP = np.empty((num_contexts, cfg.total_draft_tokens, cfg.vocab), np.float64)
    h_init = np.empty((num_contexts, cfg.d_model), np.float64)
    roots = contexts[:, -1].astype(np.int64)
    for i in range(num_contexts):
        G[i], h_init[i], TP[i] = _rollout(cfg, target, ref, contexts[i])

    e_root = weights.embed[roots].astype(np.float64)
    e_g0 = weights.embed[G[:, 0]].astype(np.float64)
    # Soft labels = the target's distribution (distribution matching, not argmax).
    p0 = TP[:, 0, :]
    p1 = TP[:, 1, :]
    lm = weights.lm_head.astype(np.float64)
    N = num_contexts

    # 2. Train the serial head (2 autoregressive teacher-forced steps).
    def serial_loss(*params):
        s0, logits0 = _serial_step(h_init, e_root, lm, params)
        _s1, logits1 = _serial_step(s0, e_g0, lm, params)
        return ts.ops.reduce(
            ts.ops.add(ts.ops.mul(ts.ops.log_softmax(logits0), -p0),
                       ts.ops.mul(ts.ops.log_softmax(logits1), -p1)),
            op="sum")

    params = _serial_params(cfg, weights)
    grad_fn = ad.grad(serial_loss, argnums=tuple(range(len(params))))
    state = None
    for _ in range(serial_steps):
        grads = [np.asarray(g) / N for g in grad_fn(*params)]
        params, state = optim.adam(params, grads, state, lr=lr)
    weights = _write_serial_params(cfg, weights, params)

    # 3. Serial features for the parallel heads (detached numpy forward).
    feats = _serial_features(cfg, weights, ref, h_init, e_root, e_g0)

    # 4. Train each parallel head independently to predict its position.
    new_fc1, new_fc2 = list(weights.parallel_fc1), list(weights.parallel_fc2)
    for i in range(cfg.parallel_heads):
        soft = TP[:, cfg.serial_tokens + i, :]    # target distribution (soft label)

        def head_loss(fc1, fc2, _soft=soft):
            h = ts.ops.relu(ts.ops.matmul(feats, fc1))
            logits = ts.ops.matmul(ts.ops.matmul(h, fc2), lm)
            return ts.ops.reduce(ts.ops.mul(ts.ops.log_softmax(logits), -_soft), op="sum")

        p = [weights.parallel_fc1[i].astype(np.float64),
             weights.parallel_fc2[i].astype(np.float64)]
        gfn = ad.grad(head_loss, argnums=(0, 1))
        st = None
        for _ in range(parallel_steps):
            grads = [np.asarray(g) / N for g in gfn(*p)]
            p, st = optim.adam(p, grads, st, lr=lr)
        new_fc1[i] = np.asarray(p[0], np.float32)
        new_fc2[i] = np.asarray(p[1], np.float32)

    return replace(weights, parallel_fc1=tuple(new_fc1), parallel_fc2=tuple(new_fc2))


def _serial_features(cfg: GumihoConfig, weights: GumihoWeights, backend,
                     h_init, e_root, e_g0) -> np.ndarray:
    """Detached ``concat(serial_h0, serial_h1)`` for every context — the
    parallel heads' input. Uses the trained serial weights via plain numpy."""
    lm = weights.lm_head.astype(np.float64)
    params = _serial_params(cfg, weights)

    def step(h_part, e_part):
        fc_h, fc_e, snorm = params[0], params[1], params[2]
        s = np.asarray(h_part) @ fc_h + np.asarray(e_part) @ fc_e
        off = 3
        for _li in range(len(weights.serial_layers)):
            wv, wo, ln1, ln2, wg, wu, wd = params[off:off + 7]
            n = backend.rmsnorm(s, ln1)
            s = s + (n @ wv) @ wo
            n2 = backend.rmsnorm(s, ln2)
            s = s + backend.silu_mul(n2 @ wg, n2 @ wu) @ wd
            off += 7
        return s

    s0 = step(h_init, e_root)
    s1 = step(s0, e_g0)
    return np.concatenate([np.asarray(s0), np.asarray(s1)], axis=1)
