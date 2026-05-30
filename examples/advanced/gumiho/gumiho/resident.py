"""GPU-resident Gumiho serial draft — one command buffer per token.

The serial head is the hot inner loop: it runs autoregressively, and on the
non-resident path every one of its ~13 dense ops per step is a separate GPU
dispatch + host sync (the `@jit` per-op path). This module keeps the serial
weights **resident** on-device (uploaded once) and encodes a whole serial step
into **one** command buffer via :class:`runtime.AppleGPUEncodeSession`, so only
the sampled token id and the carry hidden cross back to the host each step.

It reuses the R2 encode ops — `bmm` / `rmsnorm` / `add` / `silu_mul` — including
the elementwise additions that complete the transformer-block surface. The math
is exactly the inference `SerialHead` (the T=1 self-attention reduces to a value
projection `v @ Wo`, so we use the value slice of `Wqkv`), and the result is
validated token-for-token against it. Degrades to the host `SerialHead` when
Metal is unavailable.

The FTA tree-verification phase stays host-orchestrated (top-k selection and the
prefix trie are data-dependent control flow), so this targets the serial draft —
the part that is a pure resident dense chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessera import runtime as R

from .config import GumihoConfig
from .model import GumihoWeights, SerialHead, TargetModel


# ─────────────────────────────────────────────────────────────────────────────
# Phase-G Rung 1 — the serial draft as a single MPSGraph forLoop dispatch
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ForLoopDraftResult:
    tokens: list
    matches_host: bool
    max_hidden_err: float
    dispatches: int               # 1 — the whole serial loop is one graph
    host_dispatch_equiv: int      # per-op dispatches the host loop would issue
    backend: str


def _pack_serial_weights(cfg: GumihoConfig, w: GumihoWeights) -> dict:
    d = cfg.d_model
    stk = lambda a: np.stack([getattr(layer, a) for layer in w.serial_layers])  # noqa: E731
    return dict(
        embed=w.embed, fc_in=w.serial_fc_in, ln1_all=stk("ln1"), ln2_all=stk("ln2"),
        wv_all=np.stack([layer.wqkv[:, 2 * d:3 * d] for layer in w.serial_layers]),
        wo_all=stk("wo"), wg_all=stk("w_gate"), wu_all=stk("w_up"),
        wd_all=stk("w_down"), snorm=w.serial_norm, lm_head=w.lm_head)


def serial_draft_forloop(cfg: GumihoConfig, weights: GumihoWeights, h_init, root_token):
    """Run the serial draft as one MPSGraph control-flow executable. Returns
    ``(tokens, hiddens)`` or ``None`` when the forLoop symbol is unavailable."""
    p = _pack_serial_weights(cfg, weights)
    return R.apple_gpu_cf_serial_draft(
        p["embed"], p["fc_in"], p["ln1_all"], p["ln2_all"], p["wv_all"], p["wo_all"],
        p["wg_all"], p["wu_all"], p["wd_all"], p["snorm"], p["lm_head"], h_init,
        int(root_token), cfg.serial_tokens, cfg.serial_layers, cfg.d_model,
        cfg.ffn_hidden, cfg.vocab, cfg.rmsnorm_eps, np)


def validate_serial_forloop(cfg: GumihoConfig, weights: GumihoWeights, *,
                            seed: int = 0) -> ForLoopDraftResult:
    """Run the serial-draft forLoop and check it matches the host SerialHead."""
    from .backend import NumpyBackend

    rng = np.random.default_rng(seed)
    ctx = rng.integers(0, cfg.vocab, size=cfg.context_len, dtype=np.int64)
    be = NumpyBackend(eps=cfg.rmsnorm_eps)
    tgt = TargetModel(weights, cfg)
    sh = SerialHead(weights, cfg)
    lh, _ = tgt.forward(be, ctx)
    h_init, root = lh[-1], int(ctx[-1])
    htoks, _, hhid = sh.generate(be, tgt, h_init, root)
    ops_equiv = (2 + 10 * cfg.serial_layers + 1) * cfg.serial_tokens

    res = serial_draft_forloop(cfg, weights, h_init, root)
    if res is None:                                   # off-Metal fallback
        return ForLoopDraftResult([int(t) for t in htoks], True, 0.0, 0,
                                  ops_equiv, "numpy")
    gtoks, ghid = res
    hh = np.stack([np.asarray(h, np.float64) for h in hhid])
    err = float(np.max(np.abs(ghid.astype(np.float64) - hh)))
    matches = (list(int(t) for t in gtoks) == list(int(t) for t in htoks)) and err < 1e-3
    return ForLoopDraftResult([int(t) for t in gtoks], matches, err, 1,
                              ops_equiv, "metal")


@dataclass(frozen=True)
class ResidentDraftResult:
    tokens: list
    matches_host: bool
    max_logit_abs_err: float
    command_buffers: int          # resident: 1 per serial token
    host_dispatch_equiv: int      # per-op dispatches the host path would issue
    backend: str


class ResidentSerialDraft:
    """Serial draft with resident weights + one command buffer per token."""

    # dense ops encoded per serial step (for the dispatch-reduction metric):
    # fc_in bmm + per layer (rmsnorm, wv bmm, wo bmm, add, rmsnorm, wg bmm,
    # wu bmm, silu_mul, wd bmm, add) + final rmsnorm + lm bmm.
    _OPS_PER_STEP_BASE = 2
    _OPS_PER_LAYER = 10

    def __init__(self, weights: GumihoWeights, cfg: GumihoConfig) -> None:
        self.w = weights
        self.cfg = cfg
        d = cfg.d_model
        self._dt = R.DeviceTensor
        self._metal = self._dt.is_metal()
        self._resident: list = []
        if self._metal:
            up = lambda a: self._keep(self._dt.from_numpy(np.ascontiguousarray(a, np.float32)))  # noqa: E731
            self._fc_in = up(weights.serial_fc_in[None])        # [1, 2d, d]
            self._snorm = up(weights.serial_norm)               # [d]
            self._lm = up(weights.lm_head[None])                # [1, d, V]
            self._layers = []
            for layer in weights.serial_layers:
                wv = layer.wqkv[:, 2 * d:3 * d]
                self._layers.append({
                    "ln1": up(layer.ln1), "ln2": up(layer.ln2),
                    "wv": up(wv[None]), "wo": up(layer.wo[None]),
                    "wg": up(layer.w_gate[None]), "wu": up(layer.w_up[None]),
                    "wd": up(layer.w_down[None]),
                })

    def _keep(self, t):
        if t is not None:
            self._resident.append(t)
        return t

    @property
    def ops_per_step(self) -> int:
        return self._OPS_PER_STEP_BASE + self._OPS_PER_LAYER * self.cfg.serial_layers + 1

    def generate(self, h_init: np.ndarray, root_token: int):
        """Return ``(tokens, hiddens, logits, command_buffers)``."""
        if not self._metal or any(t is None for t in self._resident):
            return self._host(h_init, root_token)
        d, ffn = self.cfg.d_model, self.cfg.ffn_hidden
        eps = self.cfg.rmsnorm_eps
        tokens, hiddens, logits_all = [], [], []
        h_t = np.ascontiguousarray(h_init, np.float32).reshape(d)
        y_t = int(root_token)
        cbufs = 0
        for _ in range(self.cfg.serial_tokens):
            e = self.w.embed[y_t].astype(np.float32)
            x = np.concatenate([h_t, e]).reshape(1, 1, 2 * d).astype(np.float32)
            dx = self._dt.from_numpy(x)
            sess = R.AppleGPUEncodeSession()
            if dx is None or not sess.available:
                return self._host(h_init, root_token)
            s2 = None
            with sess:
                s = sess.bmm(dx, self._fc_in)                   # [1,1,d]
                s2 = s.reshape_view(1, d) if s is not None else None
                for L in self._layers:
                    n1 = sess.rmsnorm(s2, L["ln1"], eps)
                    v = sess.bmm(n1.reshape_view(1, 1, d), L["wv"]) if n1 is not None else None
                    attn = sess.bmm(v, L["wo"]) if v is not None else None
                    s2 = sess.add(s2, attn.reshape_view(1, d)) if attn is not None else None
                    n2 = sess.rmsnorm(s2, L["ln2"], eps) if s2 is not None else None
                    gate = sess.bmm(n2.reshape_view(1, 1, d), L["wg"]) if n2 is not None else None
                    upp = sess.bmm(n2.reshape_view(1, 1, d), L["wu"]) if n2 is not None else None
                    act = (sess.silu_mul(gate.reshape_view(1, ffn), upp.reshape_view(1, ffn))
                           if gate is not None and upp is not None else None)
                    down = sess.bmm(act.reshape_view(1, 1, ffn), L["wd"]) if act is not None else None
                    s2 = sess.add(s2, down.reshape_view(1, d)) if down is not None else None
                sn = sess.rmsnorm(s2, self._snorm, eps) if s2 is not None else None
                logits_d = sess.bmm(sn.reshape_view(1, 1, d), self._lm) if sn is not None else None
            if logits_d is None or s2 is None:
                return self._host(h_init, root_token)
            cbufs += 1
            logits = np.asarray(logits_d.numpy()).reshape(self.cfg.vocab)
            h_t = np.asarray(s2.numpy()).reshape(d).copy()
            tok = int(np.argmax(logits))
            tokens.append(tok); hiddens.append(h_t.copy()); logits_all.append(logits)
            dx.free()
            y_t = tok
        return tokens, hiddens, logits_all, cbufs

    def _host(self, h_init, root_token):
        from .backend import NumpyBackend
        be = NumpyBackend(eps=self.cfg.rmsnorm_eps)
        target = TargetModel(self.w, self.cfg)
        sh = SerialHead(self.w, self.cfg)
        toks, lp, hid = sh.generate(be, target, h_init, root_token)
        # reconstruct per-step logits from log-probs is lossy; recompute argmax
        logits = [np.asarray(lp[k]) for k in range(len(toks))]
        return list(toks), [np.asarray(h) for h in hid], logits, 0

    def free(self) -> None:
        for t in self._resident:
            if t is not None:
                t.free()
        self._resident = []


def validate_resident_draft(cfg: GumihoConfig, weights: GumihoWeights, *,
                            seed: int = 0) -> ResidentDraftResult:
    """Run the resident serial draft and check it matches the host SerialHead."""
    from .backend import NumpyBackend
    rng = np.random.default_rng(seed)
    ctx = rng.integers(0, cfg.vocab, size=cfg.context_len, dtype=np.int64)
    be = NumpyBackend(eps=cfg.rmsnorm_eps)
    target = TargetModel(weights, cfg)
    last_hidden, _ = target.forward(be, ctx)
    h_init = last_hidden[-1]
    root = int(ctx[-1])

    # host reference (full SerialHead).
    sh = SerialHead(weights, cfg)
    host_tokens, host_lp, _host_hid = sh.generate(be, target, h_init, root)
    host_logits = [np.asarray(host_lp[k]) for k in range(len(host_tokens))]

    rd = ResidentSerialDraft(weights, cfg)
    tokens, _hid, logits, cbufs = rd.generate(h_init, root)
    backend = "metal" if rd._metal else "numpy"

    # Resident emits raw logits; host stores log-probs. Compare log-softmax of
    # both so the two are on the same footing.
    def logsm(z):
        z = np.asarray(z, np.float64)
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    max_err = 0.0
    if backend == "metal":
        for k in range(len(tokens)):
            max_err = max(max_err, float(np.max(np.abs(logsm(logits[k]) - host_logits[k]))))
    matches = (list(tokens) == list(host_tokens)) and (backend != "metal" or max_err < 1e-3)
    rd.free()
    return ResidentDraftResult(
        tokens=[int(t) for t in tokens], matches_host=bool(matches),
        max_logit_abs_err=max_err, command_buffers=cbufs,
        host_dispatch_equiv=rd.ops_per_step * cfg.serial_tokens, backend=backend)
