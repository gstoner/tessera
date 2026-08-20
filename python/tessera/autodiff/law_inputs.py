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

__all__ = ["GEO_LAW_INPUT_SPECS", "InputSpec", "LAW_INPUT_SPECS"]


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
    # Input-manifold declaration: some ops are only defined (and only
    # differentiable) on a submanifold — cholesky on symmetric PSD, for
    # example — so a raw Gaussian tangent leaves the domain and the rule
    # pair legitimately disagrees off-manifold. `tangent_project(i, t)`
    # maps a random tangent for primal `i` onto the manifold's tangent
    # space (e.g. symmetrization). This declares mathematics, not
    # tolerance: the projected tangents still exercise the full tangent
    # space of the domain.
    tangent_project: Optional[Callable[[int, np.ndarray], np.ndarray]] = None


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
    "istft": S(lambda rng: ((rng.standard_normal((13, 9))
                             + 1j * rng.standard_normal((13, 9)),
                             np.hanning(16) + 0.25),
                            {"hop": 4, "center": True, "norm": "ortho"}),
               diff_args=(0, 1), chain=False, rtol=1e-6,
               note="non-default config on purpose (AD-LAW-1h): the old JVP "
                    "pinned axis/onesided/norm and dropped center/length"),

    # ── AD-LAW-1g: quantize family (straight-through estimator) ─────────────
    # STE rules: the tangent flows through unchanged, so the derivative is
    # identically 1 regardless of scale/symmetric/format — verified, which is
    # what makes those swallowed keys benign. `chain=False`: an STE primal is
    # deliberately NOT the canonical forward's output (the forward returns a
    # (q, scale, zero_point) tuple), so the chain law's primal anchor does not
    # apply — this is a declared convention, not a defect.
    "quantize_int8": S(lambda rng: ((rng.standard_normal((3, 4)) * 3,), {}),
                       chain=False, note="STE; derivative is identity"),
    "quantize_int4": S(lambda rng: ((rng.standard_normal((3, 4)) * 3,), {}),
                       chain=False, note="STE; derivative is identity"),
    "quantize_fp4": S(lambda rng: ((rng.standard_normal((3, 4)),), {}),
                      chain=False, note="STE; derivative is identity"),
    "quantize_fp6": S(lambda rng: ((rng.standard_normal((3, 4)),), {}),
                      chain=False, note="STE; derivative is identity"),
    "quantize_nvfp4": S(lambda rng: ((rng.standard_normal((2, 16)),), {}),
                        chain=False, note="STE; derivative is identity"),
    # dequantize: linear in the container, with a per-block scale for nvfp4.
    "dequantize_nvfp4": S(
        lambda rng: ((rng.standard_normal(32),
                      (np.arange(4) + 1).astype(np.float64)), {"block_size": 8}),
        diff_args=(0,), chain=False,
        note="per-block scale array — the shape that crashed both modes "
             "before AD-LAW-1g"),
    # ── AD-LAW-1j spec growth: structural / shape ops ────────────────────────
    "cat": S(lambda rng: (([rng.standard_normal((2, 3)),
                            rng.standard_normal((2, 3))],), {"axis": 0})),
    "stack": S(lambda rng: (([rng.standard_normal((2, 3)),
                              rng.standard_normal((2, 3))],), {"axis": 0})),
    "chunk": S(lambda rng: ((rng.standard_normal((4, 3)),),
                            {"chunks": 2, "axis": 0})),
    "split": S(lambda rng: ((rng.standard_normal((4, 3)),),
                            {"indices_or_sections": 2, "axis": 0})),
    "view": S(lambda rng: ((rng.standard_normal((2, 6)),),
                           {"shape": (3, 4)})),
    "broadcast": S(lambda rng: ((rng.standard_normal((1, 4)),),
                                {"shape": (3, 4)})),
    "broadcast_to_axis": S(lambda rng: ((rng.standard_normal((3, 4)),),
                                        {"axis_size": 2, "axis": 0})),
    "select": S(lambda rng: ((rng.standard_normal((4, 3)),),
                             {"index": 1, "axis": 0})),
    "slice": S(lambda rng: ((rng.standard_normal((4, 5)),),
                            {"start_indices": (1, 0), "slice_sizes": (2, 3)})),
    "dynamic_slice": S(lambda rng: ((rng.standard_normal((4, 5)),),
                                    {"start_indices": (1, 0),
                                     "slice_sizes": (2, 3)})),
    "dynamic_update_slice": S(
        lambda rng: ((rng.standard_normal((4, 5)),
                      rng.standard_normal((2, 3))),
                     {"start_indices": (1, 0)})),
    "index_update": S(
        lambda rng: ((rng.standard_normal((4, 3)), np.array([0, 2]),
                      rng.standard_normal((2, 3))), {"axis": 0}),
        diff_args=(0, 2)),
    "scatter": S(
        lambda rng: ((rng.standard_normal((4, 3)), np.array([0, 2]),
                      rng.standard_normal((2, 3))), {"axis": 0}),
        diff_args=(0, 2)),
    "scatter_add": S(
        lambda rng: ((rng.standard_normal((4, 3)), np.array([0, 2]),
                      rng.standard_normal((2, 3))), {"axis": 0}),
        diff_args=(0, 2)),
    "scatter_reduce": S(
        lambda rng: ((rng.standard_normal((4, 3)), np.array([0, 2]),
                      rng.standard_normal((2, 3))),
                     {"axis": 0, "reduce": "sum"}),
        diff_args=(0, 2)),
    "masked_scatter": S(
        lambda rng: ((rng.standard_normal((3, 4)),
                      np.tile(np.array([True, False, True, False]), (3, 1)),
                      rng.standard_normal(6)), {}),
        diff_args=(0, 2)),
    "mor_scatter": S(
        lambda rng: ((rng.standard_normal((2, 3, 4)),
                      rng.standard_normal((2, 3, 4)),
                      rng.standard_normal((2, 3)) > 0), {}),
        diff_args=(0, 1)),
    # ── AD-LAW-1j: image / vision structural ────────────────────────────────
    "center_crop": S(lambda rng: ((rng.standard_normal((1, 2, 6, 6)),),
                                  {"size": (4, 4), "layout": "nchw"})),
    "patchify": S(lambda rng: ((rng.standard_normal((1, 2, 4, 4)),),
                               {"patch_size": 2, "layout": "nchw"})),
    "pixel_shuffle": S(lambda rng: ((rng.standard_normal((1, 4, 3, 3)),),
                                    {"upscale_factor": 2, "layout": "nchw"})),
    "pixel_unshuffle": S(lambda rng: ((rng.standard_normal((1, 1, 4, 4)),),
                                      {"downscale_factor": 2,
                                       "layout": "nchw"})),
    "image_normalize": S(lambda rng: ((rng.standard_normal((1, 2, 4, 4)),),
                                      {"mean": (0.1, 0.2), "std": (0.9, 1.1),
                                       "layout": "nchw"})),
    # ── AD-LAW-1j: linear algebra ────────────────────────────────────────────
    # cholesky's domain is symmetric PSD; its tangent space is the symmetric
    # matrices, so the probe tangents are symmetrized (an input-manifold
    # declaration — the rules are only claimed on that subspace).
    "cholesky": S(
        lambda rng: (((lambda a: a @ a.T + 3.0 * np.eye(3))(
            rng.standard_normal((3, 3))),), {}),
        tangent_project=lambda i, t: 0.5 * (t + np.swapaxes(t, -1, -2))),
    "qr": S(lambda rng: ((rng.standard_normal((4, 3))
                          + np.eye(4, 3) * 3.0,), {})),
    "svd": S(lambda rng: ((rng.standard_normal((3, 3))
                           + np.diag([3.0, 2.0, 1.0]),), {})),
    "tri_solve": S(
        lambda rng: ((np.tril(rng.standard_normal((3, 3)))
                      + 3.0 * np.eye(3),
                      rng.standard_normal((3, 2))), {"lower": True})),
    "weight_norm": S(lambda rng: ((rng.standard_normal((4, 3)),),
                                  {"axis": -1}),
                     rtol=2e-3,
                     note="float32 reference forward — central-difference "
                          "noise floor ~1e-3; defects in this rule showed "
                          "as O(1)"),
    "spectral_norm": S(lambda rng: ((rng.standard_normal((4, 3)),),
                                    {"n_iter": 8}), rtol=1e-5),
    # ── AD-LAW-1j: matmul-family projections ────────────────────────────────
    "factorized_matmul": S(lambda rng: ((rng.standard_normal((4, 3)),
                                         rng.standard_normal((3, 5))),
                                        {"rank": 2})),
    "linear_general": S(lambda rng: ((rng.standard_normal((2, 4)),
                                      rng.standard_normal((4, 3)),
                                      rng.standard_normal(3)), {"axis": -1})),
    "lora_linear": S(lambda rng: ((rng.standard_normal((2, 4)),
                                   rng.standard_normal((4, 3)),
                                   rng.standard_normal((4, 2)),
                                   rng.standard_normal((2, 3)),
                                   rng.standard_normal(3)), {"alpha": 1.0})),
    "latent_kv_compress": S(lambda rng: ((rng.standard_normal((2, 3, 4)),
                                          rng.standard_normal((4, 2))), {})),
    "latent_kv_expand_k": S(lambda rng: ((rng.standard_normal((2, 3, 2)),
                                          rng.standard_normal((2, 4))), {})),
    "latent_kv_expand_v": S(lambda rng: ((rng.standard_normal((2, 3, 2)),
                                          rng.standard_normal((2, 4))), {})),
    # ── AD-LAW-1j: clifford tensor lane (Cl(3,0) coefficient arrays) ────────
    "clifford_geometric_product": S(lambda rng: ((rng.standard_normal(8),
                                                  rng.standard_normal(8)),
                                                 {})),
    "clifford_wedge": S(lambda rng: ((rng.standard_normal(8),
                                      rng.standard_normal(8)), {})),
    "clifford_inner": S(lambda rng: ((rng.standard_normal(8),
                                      rng.standard_normal(8)), {})),
    "clifford_left_contraction": S(lambda rng: ((rng.standard_normal(8),
                                                 rng.standard_normal(8)),
                                                {})),
    "clifford_reverse": S(lambda rng: ((rng.standard_normal(8),), {})),
    "clifford_conjugate": S(lambda rng: ((rng.standard_normal(8),), {})),
    "clifford_grade_involution": S(lambda rng: ((rng.standard_normal(8),),
                                                {})),
    "clifford_grade_projection": S(lambda rng: ((rng.standard_normal(8),),
                                                {"grade": 1})),
    "clifford_hodge_star": S(lambda rng: ((rng.standard_normal(8),), {})),
    "clifford_norm": S(lambda rng: ((rng.standard_normal(8) + 0.5,), {})),
    "clifford_norm_squared": S(lambda rng: ((rng.standard_normal(8),), {})),
    "clifford_rotor_sandwich": S(
        lambda rng: ((_cl30_rotor(rng), rng.standard_normal(8)), {})),
    "clifford_exp": S(lambda rng: ((0.3 * rng.standard_normal(8),), {})),
    "clifford_log": S(lambda rng: ((_cl30_rotor(rng),), {})),
}


def _cl30_rotor(rng: np.random.Generator) -> np.ndarray:
    """Cl(3,0) rotor coefficients: cos θ + sin θ · B̂ on the grade-2 blades
    (masks 3, 5, 6). Used by the clifford tensor-lane specs whose domain is
    the rotor manifold (`clifford_log`, `clifford_rotor_sandwich`)."""
    theta = float(rng.uniform(0.2, 1.0))
    b = rng.standard_normal(3)
    b /= np.linalg.norm(b)
    c = np.zeros(8)
    c[0] = np.cos(theta)
    c[[3, 5, 6]] = np.sin(theta) * b
    return c


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
    tie_groups: int = 1
    """How many independent tie groups the probe contains.

    SUBGRAD_SPLIT conserves one unit of cotangent mass PER GROUP, so the
    engine compares the summed shares against this count. It defaulted to a
    hardcoded 1 before, which meant a probe with two tied rows would have
    summed to 2.0 and failed a correct rule — declared here so a future spec
    can carry more than one group.
    """
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


# ── Geometric registry (multivector) specs ──────────────────────────────────
# Inputs for the Law-3 adjoint sweep over `_VJPS_GEO`/`_JVPS_GEO` — the
# sweep the plan requires to run BEFORE the `CliffordTangent` absorption
# (AUTODIFF_NEXTGEN_PLAN §3.5 / §5 item 4). The pairing is the Frobenius
# inner product on coefficient vectors, which is the convention the VJPs
# themselves declare (`geometric/vjp.py` module docstring); the algebra is
# Cl(3, 0), the signature that convention is stated for.
#
# The `tessera.ga` import is deliberately lazy (inside each `make`) so
# importing this module never pulls the GA stack; the sweep already guards
# the geometric registry import the same way.


def _mv(rng: np.random.Generator, grades=None, scale: float = 1.0):
    from tessera.ga.multivector import Multivector
    from tessera.ga.signature import Cl

    alg = Cl(3, 0)
    coeffs = scale * rng.standard_normal(alg.dim)
    return Multivector(coeffs, alg, grades=grades)


def _geo_unary(**mv_kwargs):
    def make(rng):
        return (_mv(rng, **mv_kwargs),), {}
    return make


def _geo_binary():
    def make(rng):
        return (_mv(rng), _mv(rng)), {}
    return make


def _geo_rotor(rng: np.random.Generator):
    """A genuine rotor in Cl(3, 0): R = cos θ + sin θ · B̂ for a unit
    bivector B̂ (every bivector in 3D is simple, so B̂² = −1)."""
    import numpy as _np

    from tessera.ga.multivector import Multivector
    from tessera.ga.signature import Cl

    alg = Cl(3, 0)
    theta = float(rng.uniform(0.2, 1.2))
    b = rng.standard_normal(3)
    b = b / _np.linalg.norm(b)
    coeffs = _np.zeros(alg.dim)
    coeffs[0] = _np.cos(theta)
    # Grade-2 blade masks in Cl(3,0): popcount-2 indices 3 (e12), 5 (e13),
    # 6 (e23).
    coeffs[[3, 5, 6]] = _np.sin(theta) * b
    return Multivector(coeffs, alg)


def _geo_norm_input():
    def make(rng):
        # Keep |a| well away from the norm's declared subgradient-at-zero
        # convention: a random 8-coefficient Gaussian has |a| ≈ 2.6 a.s.,
        # but make the floor structural rather than probabilistic.
        mv = _mv(rng)
        import numpy as _np

        n = float(_np.sqrt(_np.sum(mv.coefficients ** 2)))
        if n < 0.5:  # pragma: no cover — measure-zero fallback, kept explicit
            mv = (1.0 / max(n, 1e-9)) * mv
        return (mv,), {}
    return make


GEO_LAW_INPUT_SPECS: dict[str, InputSpec] = {
    # linear, self-adjoint-by-declaration
    "add": S(_geo_binary()),
    "sub": S(_geo_binary()),
    "neg": S(_geo_unary()),
    "reverse": S(_geo_unary()),
    "grade_involution": S(_geo_unary()),
    "conjugate": S(_geo_unary()),
    "hodge_star": S(_geo_unary()),
    # a is differentiable; the grade selector is configuration
    "grade_projection": S(lambda rng: ((_mv(rng), 1), {}), diff_args=(0,)),
    # both the multivector and the scalar are differentiable
    "scalar_mul": S(lambda rng: ((_mv(rng), float(rng.uniform(0.5, 2.0))), {})),
    # bilinear
    "geometric_product": S(_geo_binary()),
    "wedge": S(_geo_binary()),
    "left_contraction": S(_geo_binary()),
    # scalar-valued
    "inner": S(_geo_binary()),
    "norm_squared": S(_geo_norm_input()),
    "norm": S(_geo_norm_input()),
    # rotor sandwich: a genuine rotor plus a general multivector
    "rotor_sandwich": S(lambda rng: ((_geo_rotor(rng), _mv(rng)), {})),
}
