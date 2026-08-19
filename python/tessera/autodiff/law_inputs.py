"""AD-LAW-1 — declarative per-op input specs for the law sweep.

Each spec builds a deterministic, kink-avoiding sample point for one
registered op so ``laws.py`` can evaluate the adjoint and chain laws on it.
Growing coverage is one entry here per op — no engine changes.

Conventions:

- ``make(rng) -> (primals, kwargs)``; arrays float64 unless the op requires
  otherwise (integer labels etc.).
- **Kink avoidance is part of the spec's contract**: nonsmooth ops must be
  sampled strictly inside a smooth piece (``relu`` away from 0, ``maximum``
  away from ties), because the chain law compares against finite
  differences, which straddle the kink otherwise. Behaviour *at* the kink
  is Law 5's job (``nonsmooth.py``'s declared policies and
  ``test_nonsmooth_selection.py``), not this file's.
- ``diff_args`` lists the positional args the laws probe; ``None`` = all.
- ``zero_tangent_ok`` marks ops whose true derivative is 0 almost
  everywhere (``sign``): an identically-zero tangent stream is correct
  there, not a matched-zero symptom.

Ops in the registries with no entry here are reported by the sweep as
``no_spec`` — visible debt, never a silent skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = ["InputSpec", "LAW_INPUT_SPECS"]


@dataclass(frozen=True)
class InputSpec:
    make: Callable[[np.random.Generator], tuple[tuple, dict]]
    diff_args: Optional[tuple[int, ...]] = None
    probes: int = 4
    chain: bool = True
    zero_tangent_ok: bool = False
    rtol: Optional[float] = None   # override for rules that are internally
                                   # numeric (FD-based JVPs can't hit 1e-8)
    note: str = ""


def _away_from(x: np.ndarray, kink: float = 0.0, margin: float = 0.25) -> np.ndarray:
    """Push samples at least `margin` away from a kink point."""
    d = x - kink
    return kink + np.where(np.abs(d) < margin, np.sign(d) * margin + d, d)


def _unary(shape=(3, 4), lo=None, hi=None, kink=None):
    def make(rng):
        x = rng.standard_normal(shape)
        if kink is not None:
            x = _away_from(x, kink)
        if lo is not None or hi is not None:
            span = (hi - lo)
            x = lo + (span / 2) * (1 + np.tanh(x))  # smooth squash into (lo, hi)
        return (x,), {}
    return make


def _positive(shape=(3, 4), floor=0.3):
    def make(rng):
        return (floor + np.abs(rng.standard_normal(shape)),), {}
    return make


def _binary(shape=(3, 4), tie_gap=None):
    def make(rng):
        a = rng.standard_normal(shape)
        b = rng.standard_normal(shape)
        if tie_gap is not None:
            b = np.where(np.abs(a - b) < tie_gap, b + np.sign(b - a + 1e-3) * tie_gap, b)
        return (a, b), {}
    return make


def _norm_input(shape=(4, 8)):
    def make(rng):
        return (rng.standard_normal(shape),), {}
    return make


S = InputSpec  # local shorthand


LAW_INPUT_SPECS: dict[str, InputSpec] = {
    # ── smooth unary pointwise ───────────────────────────────────────────────
    "exp": S(_unary()),
    "log": S(_positive()),
    "log1p": S(_positive(floor=-0.5)),
    "expm1": S(_unary()),
    "sqrt": S(_positive()),
    "sin": S(_unary()),
    "cos": S(_unary()),
    "tan": S(_unary(lo=-1.2, hi=1.2)),
    "sinh": S(_unary()),
    "cosh": S(_unary()),
    "asin": S(_unary(lo=-0.9, hi=0.9)),
    "acos": S(_unary(lo=-0.9, hi=0.9)),
    "atan": S(_unary()),
    "erf": S(_unary()),
    "erfc": S(_unary()),
    "reciprocal": S(_positive()),
    "tanh": S(_unary()),
    "sigmoid": S(_unary()),
    "sigmoid_safe": S(_unary()),
    "softplus": S(_unary()),
    "silu": S(_unary()),
    "gelu": S(_unary()),
    "softcap": S(lambda rng: ((rng.standard_normal((3, 4)),), {"cap": 5.0})),

    # ── nonsmooth unary, sampled inside a smooth piece ───────────────────────
    "relu": S(_unary(kink=0.0)),
    "abs": S(_unary(kink=0.0)),
    "absolute": S(_unary(kink=0.0)),
    "sign": S(_unary(kink=0.0), zero_tangent_ok=True, chain=False,
              note="derivative 0 a.e. — zero tangents are correct"),
    "clamp": S(lambda rng: ((rng.standard_normal((3, 4)) * 0.3,),
                            {"min": -1.0, "max": 1.0}),
               note="interior samples (canonical `min`/`max` kwargs); "
                    "bound behaviour is Law 5"),

    # ── binary pointwise ─────────────────────────────────────────────────────
    "add": S(_binary()),
    "mul": S(_binary()),
    "maximum": S(_binary(tie_gap=0.2)),
    "minimum": S(_binary(tie_gap=0.2)),
    "atan2": S(_binary(tie_gap=0.3)),

    # ── reductions ───────────────────────────────────────────────────────────
    "sum": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "mean": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "prod": S(lambda rng: ((0.5 + np.abs(rng.standard_normal((3, 4))),), {"axis": -1})),
    "logsumexp": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "cumsum": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "var": S(_unary(shape=(3, 5))),
    "std": S(_unary(shape=(3, 5))),
    "amax": S(lambda rng: ((np.cumsum(0.3 + np.abs(rng.standard_normal((3, 4))), axis=-1),),
                           {"axis": -1}),
              note="strictly increasing rows — unique argmax, no ties"),
    "amin": S(lambda rng: ((np.cumsum(0.3 + np.abs(rng.standard_normal((3, 4))), axis=-1),),
                           {"axis": -1}),
              note="unique argmin by construction"),

    # ── linear / multilinear ─────────────────────────────────────────────────
    "matmul": S(lambda rng: ((rng.standard_normal((3, 4)),
                              rng.standard_normal((4, 2))), {})),
    "gemm": S(lambda rng: ((rng.standard_normal((3, 4)),
                            rng.standard_normal((4, 2))), {})),
    "transpose": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axes": (1, 0)})),
    "reshape": S(lambda rng: ((rng.standard_normal((3, 4)),), {"shape": (12,)})),

    # ── softmax family / normalization ───────────────────────────────────────
    "softmax": S(lambda rng: ((rng.standard_normal((3, 5)),), {"axis": -1})),
    "softmax_safe": S(lambda rng: ((rng.standard_normal((3, 5)),), {"axis": -1})),
    "log_softmax": S(lambda rng: ((rng.standard_normal((3, 5)),), {"axis": -1})),
    "layer_norm": S(_norm_input()),
    "rmsnorm": S(_norm_input()),

    # ── losses ───────────────────────────────────────────────────────────────
    "mse_loss": S(_binary(), note="both operands differentiable"),
    "cross_entropy_loss": S(
        lambda rng: ((rng.standard_normal((4, 6)),
                      rng.integers(0, 6, size=(4,))), {}),
        diff_args=(0,), chain=False,
        note="integer targets are non-differentiable"),
    "kl_divergence": S(
        lambda rng: ((
            np.log((lambda z: z / z.sum(axis=-1, keepdims=True))(
                0.2 + np.abs(rng.standard_normal((3, 5))))),
            (lambda z: z / z.sum(axis=-1, keepdims=True))(
                0.2 + np.abs(rng.standard_normal((3, 5)))),
        ), {}),
        note="arg0 is LOG-probs (p_log_probs), arg1 is probs — per the rule signature"),

    # ── AD-LAW-1c growth tranche ─────────────────────────────────────────────
    # unary long tail
    "rsqrt": S(_positive()),
    "lgamma": S(_positive(floor=0.5)),
    "digamma": S(_positive(floor=0.5)),
    "cast": S(lambda rng: ((rng.standard_normal((3, 4)),), {"dtype": "fp64"}),
              note="canonical dtype string, as the tape records it"),
    "div": S(lambda rng: ((rng.standard_normal((3, 4)),
                           1.0 + np.abs(rng.standard_normal((3, 4)))), {})),
    "sub": S(_binary()),
    "pow": S(lambda rng: ((0.5 + np.abs(rng.standard_normal((3, 4))),
                           rng.standard_normal((3, 4))), {}),
             note="positive base — d/dy needs log(x)"),
    "mod": S(lambda rng: ((0.3 + np.abs(rng.standard_normal((3, 4))) * 0.35,
                           2.0 + np.abs(rng.standard_normal((3, 4)))), {}),
             note="x ∈ (0.3, ~1.5), y > 2 ⇒ x mod y = x, far from wrap kinks"),
    "silu_mul": S(_binary()),

    # structural / view family (linear — adjoint is the shape transport)
    "permute": S(lambda rng: ((rng.standard_normal((2, 3, 4)),), {"axes": (2, 0, 1)})),
    "flatten": S(lambda rng: ((rng.standard_normal((2, 3, 4)),), {})),
    "squeeze": S(lambda rng: ((rng.standard_normal((2, 1, 4)),), {"axis": 1})),
    "unsqueeze": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": 0})),
    "expand": S(lambda rng: ((rng.standard_normal((1, 4)),), {"shape": (3, 4)})),
    "flip": S(lambda rng: ((rng.standard_normal((3, 4)),), {"axis": -1})),
    "roll": S(lambda rng: ((rng.standard_normal((3, 4)),), {"shift": 1, "axis": -1})),
    "tile": S(lambda rng: ((rng.standard_normal((2, 3)),), {"reps": (2, 2)})),
    "repeat": S(lambda rng: ((rng.standard_normal((2, 3)),), {"repeats": 2, "axis": 0})),
    "pad": S(lambda rng: ((rng.standard_normal((3, 4)),),
                          {"pad_width": ((1, 1), (0, 2))})),

    # selection / masking (index args non-differentiable)
    "where": S(lambda rng: ((rng.standard_normal((3, 4)) > 0.0,
                             rng.standard_normal((3, 4)),
                             rng.standard_normal((3, 4))), {}),
               diff_args=(1, 2), chain=False,
               note="cond is boolean, non-differentiable"),
    "masked_fill": S(lambda rng: ((rng.standard_normal((3, 4)),
                                   rng.standard_normal((3, 4)) > 0.5), {"value": 0.7}),
                     diff_args=(0,), chain=False),
    "gather": S(lambda rng: ((rng.standard_normal((4, 5)),
                              rng.integers(0, 4, size=(3,))), {"axis": 0}),
                diff_args=(0,), chain=False),
    "take": S(lambda rng: ((rng.standard_normal((4, 5)),
                            rng.integers(0, 4, size=(3,))), {"axis": 0}),
              diff_args=(0,), chain=False),
    "index_select": S(lambda rng: ((rng.standard_normal((4, 5)),
                                    rng.integers(0, 4, size=(3,))), {"axis": 0}),
                      diff_args=(0,), chain=False),

    # reductions / scans long tail
    "cumprod": S(lambda rng: ((0.5 + np.abs(rng.standard_normal((3, 4))),),
                              {"axis": -1})),
    "cummax": S(lambda rng: ((np.cumsum(0.3 + np.abs(rng.standard_normal((3, 4))), axis=-1),),
                             {"axis": -1}), note="strictly increasing — no ties"),
    "cummin": S(lambda rng: ((-np.cumsum(0.3 + np.abs(rng.standard_normal((3, 4))), axis=-1),),
                             {"axis": -1}), note="strictly decreasing — no ties"),
    "max": S(lambda rng: ((np.cumsum(0.3 + np.abs(rng.standard_normal((3, 4))), axis=-1),),
                          {"axis": -1}), note="unique argmax"),
    "min": S(lambda rng: ((np.cumsum(0.3 + np.abs(rng.standard_normal((3, 4))), axis=-1),),
                          {"axis": -1}), note="unique argmin"),
    "segment_reduce": S(lambda rng: ((rng.standard_normal((6, 3)),
                                      np.array([0, 0, 1, 1, 2, 2])), {"op": "sum"}),
                        diff_args=(0,), chain=False),

    # norms long tail
    "rmsnorm_safe": S(_norm_input()),
    "group_norm": S(lambda rng: ((rng.standard_normal((2, 4, 3, 3)), 2), {}),
                    diff_args=(0,), rtol=1e-5, chain=False,
                    note="num_groups is the second positional primal per the "
                         "rule signatures. Rules compute in fp32: the adjoint "
                         "law is exact at that precision and passes; an fp64 "
                         "central FD through fp32 arithmetic is noise-limited "
                         "at ~1e-2 relative, so the chain law is uninformative"),
    "instance_norm": S(lambda rng: ((rng.standard_normal((2, 3, 4, 4)),), {}),
                       rtol=1e-5, chain=False,
                       note="fp32-computing rules; see group_norm"),

    # losses long tail (targets differentiable only where the rule says so)
    "huber_loss": S(lambda rng: ((rng.standard_normal((3, 4)),
                                  rng.standard_normal((3, 4)) + 3.0), {}),
                    note="|pred−target| ≈ 3 ≫ delta=1 on most entries — but "
                         "mixed regions are fine, only |d|≈delta is a kink"),
    "mae_loss": S(lambda rng: ((rng.standard_normal((3, 4)),
                                rng.standard_normal((3, 4)) + 2.0), {}),
                  note="|pred−target| bounded away from the 0 kink"),
    "log_cosh_loss": S(_binary()),
    "smooth_l1_loss": S(lambda rng: ((rng.standard_normal((3, 4)),
                                      rng.standard_normal((3, 4)) + 3.0), {}),
                        note="away from the |d|=beta kink"),
    "binary_cross_entropy_loss": S(
        lambda rng: ((0.05 + 0.9 * rng.random((3, 4)),
                      (rng.random((3, 4)) > 0.5).astype(np.float64)), {}),
        diff_args=(0,)),
    "js_divergence": S(
        lambda rng: (tuple(
            (lambda z: z / z.sum(axis=-1, keepdims=True))(
                0.2 + np.abs(rng.standard_normal((3, 5))))
            for _ in range(2)), {}),
        diff_args=(0,), note="rule contract: gradient w.r.t. p only"),
    "z_loss": S(lambda rng: ((rng.standard_normal((3, 6)),), {})),
    "label_smoothed_cross_entropy": S(
        lambda rng: ((rng.standard_normal((4, 6)),
                      rng.integers(0, 6, size=(4,))), {}),
        diff_args=(0,), chain=False),

    # rope / position family
    "rope": S(lambda rng: ((rng.standard_normal((2, 2, 6, 8)),
                            rng.standard_normal((6, 4))), {}),
              diff_args=(0,), note="theta = position frequencies, held fixed"),
    "rope_merge": S(lambda rng: ((rng.standard_normal((2, 2, 6, 4)),
                                  rng.standard_normal((2, 2, 6, 4))), {})),

    # attention family (numeric-JVP rules — FD-limited tolerance)
    "flash_attn": S(lambda rng: ((rng.standard_normal((2, 2, 6, 4)),
                                  rng.standard_normal((2, 2, 6, 4)),
                                  rng.standard_normal((2, 2, 6, 4))), {}),
                    rtol=5e-5, probes=2, chain=False,
                    note="rule pair is numerically linearized internally"),
    "attn_sliding_window": S(lambda rng: ((rng.standard_normal((2, 2, 6, 4)),
                                           rng.standard_normal((2, 2, 6, 4)),
                                           rng.standard_normal((2, 2, 6, 4))),
                                          {"window_size": 3}),
                             rtol=5e-5, probes=2, chain=False),
    "linear_attn": S(lambda rng: ((rng.standard_normal((2, 2, 6, 4)),
                                   rng.standard_normal((2, 2, 6, 4)),
                                   rng.standard_normal((2, 2, 6, 4))), {}),
                     rtol=5e-5, probes=2, chain=False),
    "power_attn": S(lambda rng: ((rng.standard_normal((2, 2, 6, 4)),
                                  rng.standard_normal((2, 2, 6, 4)),
                                  rng.standard_normal((2, 2, 6, 4))), {}),
                    rtol=5e-5, probes=2, chain=False),
    "retention": S(lambda rng: ((rng.standard_normal((2, 2, 6, 4)),
                                 rng.standard_normal((2, 2, 6, 4)),
                                 rng.standard_normal((2, 2, 6, 4))), {}),
                   rtol=5e-5, probes=2, chain=False),

    # linear / matmul long tail
    "batched_gemm": S(lambda rng: ((rng.standard_normal((2, 3, 4)),
                                    rng.standard_normal((2, 4, 5))), {})),
    "qkv_projection": S(lambda rng: ((rng.standard_normal((2, 8)),
                                      rng.standard_normal((8, 24))), {})),
    "einsum": S(lambda rng: ((rng.standard_normal((3, 4)),
                              rng.standard_normal((4, 2))),
                             {"equation": "ij,jk->ik"}),
                chain=False,
                note="rules take the spec as an `equation` kwarg"),

    # spectral family (complex leaves; adjoint pairing is Re⟨a, conj b⟩)
    "fft": S(lambda rng: ((rng.standard_normal((3, 8))
                           + 1j * rng.standard_normal((3, 8)),), {"norm": "ortho"}),
             chain=False),
    "ifft": S(lambda rng: ((rng.standard_normal((3, 8))
                            + 1j * rng.standard_normal((3, 8)),), {"norm": "ortho"}),
              chain=False),
    "rfft": S(lambda rng: ((rng.standard_normal((3, 8)),), {"norm": "ortho"}),
              chain=False),
    "irfft": S(lambda rng: ((rng.standard_normal((3, 5))
                             + 1j * rng.standard_normal((3, 5)),), {"norm": "ortho"}),
               chain=False),
    # ── AD-LAW-1f: spectral transforms (JVPs now derived from the forward) ──
    # Non-default keys on purpose: the old hand-written JVPs ignored
    # axis/center/norm/onesided entirely, so a default-only probe would not
    # have caught them.
    "stft": S(lambda rng: ((rng.standard_normal(64), np.hanning(16)),
                           {"hop": 4, "center": True, "norm": "ortho"}),
              diff_args=(0, 1), chain=False,
              note="bilinear in (signal, window). `hop` travels as a kwarg: it "
                   "is the forward's 3rd positional but CONFIG, and the rules "
                   "declare it keyword-only — the AD-LAW-1d split"),
    "spectral_conv": S(lambda rng: ((rng.standard_normal(32),
                                     rng.standard_normal(32)),
                                    {"norm": "ortho"}),
                       diff_args=(0, 1), chain=False,
                       note="bilinear in (signal, kernel)"),
}


# ── Law 5: kink probes ───────────────────────────────────────────────────────
# The specs above deliberately sample *inside* a smooth piece, because Laws 1/3
# compare against finite differences that straddle a kink. Law 5 is the
# complement: it evaluates the rules **exactly at** the kink, where the Clarke
# subdifferential is a set and `nonsmooth.py` declares which element is
# returned (Decision #21a). A kink probe therefore asserts a *policy*, not a
# derivative — the only input where a legal-but-different selection is visible.


@dataclass(frozen=True)
class KinkSpec:
    """One at-the-kink probe for a declared-nonsmooth op.

    ``make`` returns ``(primals, kwargs)`` where at least one entry sits
    exactly on the kink/tie. ``kink_mask`` marks, per differentiable operand,
    which elements are at the kink (the only elements the policy governs).
    ``expected`` is the policy-mandated derivative value at those elements:
    a scalar for SUBGRAD_ZERO, or the string ``"split"`` for the
    mass-conserving SUBGRAD_SPLIT family (checked as an equal share whose
    total is 1).
    """

    make: Callable[[], tuple[tuple, dict]]
    kink_mask: Callable[[tuple], tuple]
    expected: object
    note: str = ""


def _at_zero_unary():
    def make():
        # Deliberately includes the kink itself plus both smooth sides, so a
        # rule that is right at 0 but wrong nearby still fails.
        return (np.array([[-1.5, -0.5, 0.0, 0.5, 1.5]]),), {}
    return make


def _mask_where_zero(primals):
    (x,) = primals
    return (np.asarray(x) == 0.0,)


KINK_SPECS: dict[str, KinkSpec] = {
    # SUBGRAD_ZERO family — one flat side; the declared selection is 0 at the
    # kink, matching a central-difference oracle that sees a flat plateau.
    "relu": KinkSpec(_at_zero_unary(), _mask_where_zero, 0.0),
    "abs": KinkSpec(_at_zero_unary(), _mask_where_zero, 0.0,
                    note="subdifferential at 0 is [-1,1]; midpoint declared"),
    "absolute": KinkSpec(_at_zero_unary(), _mask_where_zero, 0.0),
    "sign": KinkSpec(_at_zero_unary(), _mask_where_zero, 0.0),
    # clip/clamp: the kink is at each bound, not at 0.
    "clip": KinkSpec(
        lambda: ((np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]),),
                 {"min_val": -1.0, "max_val": 1.0}),
        lambda p: (np.isin(np.asarray(p[0]), (-1.0, 1.0)),), 0.0,
        note="strict interior only; grad 0 AT either bound"),
    "clamp": KinkSpec(
        lambda: ((np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]),),
                 {"min": -1.0, "max": 1.0}),
        lambda p: (np.isin(np.asarray(p[0]), (-1.0, 1.0)),), 0.0),
    # SUBGRAD_SPLIT family — a tie among competing arguments; the cotangent
    # is shared equally so the total mass is conserved.
    "maximum": KinkSpec(
        lambda: ((np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 5.0, 0.0]])), {}),
        lambda p: tuple(np.asarray(p[0]) == np.asarray(p[1]) for _ in range(2)),
        "split", note="element 0 is an exact tie"),
    "minimum": KinkSpec(
        lambda: ((np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 5.0, 0.0]])), {}),
        lambda p: tuple(np.asarray(p[0]) == np.asarray(p[1]) for _ in range(2)),
        "split"),
    "amax": KinkSpec(
        lambda: ((np.array([[3.0, 1.0, 3.0, 2.0]]),), {"axis": -1}),
        lambda p: (np.asarray(p[0]) == np.max(np.asarray(p[0]), axis=-1,
                                              keepdims=True),),
        "split", note="two-way tie for the maximum"),
    "max": KinkSpec(
        lambda: ((np.array([[3.0, 1.0, 3.0, 2.0]]),), {"axis": -1}),
        lambda p: (np.asarray(p[0]) == np.max(np.asarray(p[0]), axis=-1,
                                              keepdims=True),),
        "split", note="reduction form of the amax tie"),
    "min": KinkSpec(
        lambda: ((np.array([[1.0, 3.0, 1.0, 2.0]]),), {"axis": -1}),
        lambda p: (np.asarray(p[0]) == np.min(np.asarray(p[0]), axis=-1,
                                              keepdims=True),),
        "split", note="reduction form of the amin tie"),
    "amin": KinkSpec(
        lambda: ((np.array([[1.0, 3.0, 1.0, 2.0]]),), {"axis": -1}),
        lambda p: (np.asarray(p[0]) == np.min(np.asarray(p[0]), axis=-1,
                                              keepdims=True),),
        "split"),
}
