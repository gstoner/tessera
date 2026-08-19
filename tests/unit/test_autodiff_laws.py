"""AD-LAW-1 — tests for the autodiff law engine, sweep, and math harness.

Design authority: docs/audit/compiler/AUTODIFF_NEXTGEN_PLAN.md §4/§7.

Covers:
- the §3 math-verification harness (11 checks incl. the #10a negative fixture
  that the naive εᵢ ↦ ε "surjection" is not an algebra map);
- the sweep itself: every op with a registered input spec must satisfy both
  laws (a new failure here is a real finding — triage it, do not raise the
  tolerance);
- the engine's teeth: a planted wrong VJP is caught by the adjoint law, and
  a matched-zero JVP/VJP pair passes the adjoint law but is caught by the
  chain law — the documented §3.5 completeness caveat, executable;
- paired-rule kwarg-default agreement: the sweep's first real finding was
  ``jvp_rmsnorm`` defaulting ``eps=1e-6`` against the forward's and
  ``vjp_rmsnorm``'s ``1e-5`` (J and Jᵀ silently differentiating two
  different functions). That class is now pinned registry-wide;
- dashboard determinism (the CSV is byte-gated by the drift gate).
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tessera.autodiff.law_inputs import LAW_INPUT_SPECS, InputSpec
from tessera.autodiff.laws import (
    LawResult,
    adjoint_check,
    chain_check,
    run_law_sweep,
)

REPO = Path(__file__).resolve().parents[2]


# ── the math harness ─────────────────────────────────────────────────────────


def test_nextgen_math_harness_passes():
    """All §3 mathematics of the plan, machine-checked (CORE_SUBSTRATE_VIEW
    §0.1 discipline). Run as a subprocess so the harness stays a standalone
    script with its own exit semantics."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "research/autodiff_nextgen/verify_autodiff_math.py")],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "11/11 checks passed" in proc.stdout


# ── the sweep ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sweep() -> list[LawResult]:
    return run_law_sweep()


def test_sweep_has_no_failures(sweep):
    flagged = [r for r in sweep if r.status in ("fail", "rule_error")]
    msg = "\n".join(f"{r.registry}/{r.op} [{r.law}] {r.status}: {r.detail}"
                    for r in flagged)
    assert not flagged, f"law violations (real findings — triage, don't tune):\n{msg}"


def test_sweep_reports_every_registered_op(sweep):
    """Claim integrity: the sweep must enumerate the whole registry — ops it
    cannot check appear as explicit no_spec/jvp_only/vjp_only rows."""
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _VJPS

    swept_tensor = {r.op for r in sweep if r.registry == "tensor"}
    assert set(_VJPS) | set(_JVPS) <= swept_tensor


def test_every_spec_op_is_actually_checked(sweep):
    """No spec may silently rot: each op with an input spec must produce a
    pass/fail adjoint row, not an error or skip bucket."""
    by_op = {r.op: r for r in sweep
             if r.registry == "tensor" and r.law == "adjoint"}
    for op in LAW_INPUT_SPECS:
        assert op in by_op, f"spec for {op!r} matches no registered op"
        assert by_op[op].status in ("pass", "fail"), (
            f"{op}: spec exists but adjoint status is {by_op[op].status} "
            f"({by_op[op].detail})")


def test_geometric_registry_is_enumerated(sweep):
    geo = [r for r in sweep if r.registry == "geometric"]
    assert len(geo) >= 16, "geometric registry rows missing from the sweep"


# ── the engine's teeth ───────────────────────────────────────────────────────


def _exp_jvp(primals, tangents, **_):
    (x,) = primals
    (dx,) = tangents
    e = np.exp(x)
    return e, dx * e


def _exp_vjp_wrong(dout, x, **_):
    return (dout * np.exp(x) * 1.01,)  # planted 1% error


def _spec_unary() -> InputSpec:
    return InputSpec(make=lambda rng: ((rng.standard_normal((3, 4)),), {}))


def test_adjoint_law_catches_planted_wrong_vjp():
    r = adjoint_check("exp<planted>", _spec_unary(), _exp_jvp, _exp_vjp_wrong)
    assert r.status == "fail", r


def _zero_jvp(primals, tangents, **_):
    (x,) = primals
    return np.exp(x), np.zeros_like(np.asarray(x))


def _zero_vjp(dout, x, **_):
    return (np.zeros_like(np.asarray(x)),)


def test_matched_zero_pair_demonstrates_law3_caveat():
    """The documented §3.5 completeness caveat, executable: a matched-zero
    JVP/VJP pair satisfies ⟨Jv,u⟩=⟨v,Jᵀu⟩ (0=0) on every probe — the engine
    flags the vacuous pairing rather than calling it a pass — and the chain
    law catches the wrong derivative outright."""
    spec = _spec_unary()
    r3 = adjoint_check("exp<matched-zero>", spec, _zero_jvp, _zero_vjp)
    assert r3.status == "fail" and "vacuous" in r3.detail, r3
    # The chain law anchors its FD on the canonical forward resolved from the
    # op name, so the synthetic pair borrows the real name: its primal half
    # (np.exp) matches the canonical forward — only the tangent is wrong,
    # which is exactly what the FD anchor must catch.
    r1 = chain_check("exp", spec, _zero_jvp)
    assert r1.status == "fail", r1


def test_chain_requires_canonical_forward():
    """Review hardening (#584): with no canonical forward to anchor the FD,
    the chain law must refuse (`not_applicable`) rather than silently degrade
    to self-consistency — a JVP that is the correct derivative of the wrong
    function would pass a self-anchored FD."""
    r = chain_check("no-such-op<synthetic>", _spec_unary(), _exp_jvp)
    assert r.status == "not_applicable" and "canonical forward" in r.detail, r


def test_chain_catches_wrong_function_jvp():
    """A JVP that self-consistently implements exp(2x) — right shape, right
    internal consistency, wrong function — must fail against the canonical
    exp forward (the P2 review scenario, executable)."""

    def wrong_fn_jvp(primals, tangents, **_):
        (x,) = primals
        (dx,) = tangents
        e = np.exp(2.0 * x)
        return e, 2.0 * dx * e

    r = chain_check("exp", _spec_unary(), wrong_fn_jvp)
    assert r.status == "fail", r


def test_zero_tangent_ok_respected():
    """`sign`-class ops (derivative 0 a.e.) must not be flagged as vacuous."""
    spec = InputSpec(make=lambda rng: ((rng.standard_normal((3, 4)),), {}),
                     zero_tangent_ok=True)

    def sign_jvp(primals, tangents, **_):
        (x,) = primals
        return np.sign(x), np.zeros_like(np.asarray(x))

    def sign_vjp(dout, x, **_):
        return (np.zeros_like(np.asarray(x)),)

    r = adjoint_check("sign<synthetic>", spec, sign_jvp, sign_vjp)
    assert r.status == "pass", r


# ── paired-default drift (the rmsnorm finding, pinned registry-wide) ─────────

# These pairs default one side to None as an "infer from inputs / required"
# convention; a *concrete* pair of unequal defaults is always a defect (it
# makes J and Jᵀ derivatives of two different functions). Pinned exactly: if
# this set changes, a human looks — additions are new findings, removals are
# fixes to record.
# The stft/istft entries left this set in AD-LAW-1f: their hand-written
# JVPs were deleted in favour of derivation from the forward, so there
# is no second signature to disagree with.
_KNOWN_NONE_CONVENTION = {
    ("masked_fill", "value"),
    ("unsqueeze", "axis"),
}


def _paired_default_mismatches():
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _VJPS

    concrete, none_side = [], set()
    for op in sorted(set(_JVPS) & set(_VJPS)):
        try:
            jd = {p.name: p.default
                  for p in inspect.signature(_JVPS[op]).parameters.values()
                  if p.default is not inspect.Parameter.empty}
            vd = {p.name: p.default
                  for p in inspect.signature(_VJPS[op]).parameters.values()
                  if p.default is not inspect.Parameter.empty}
        except (ValueError, TypeError):
            continue
        for k in sorted(set(jd) & set(vd)):
            if jd[k] == vd[k]:
                continue
            if jd[k] is None or vd[k] is None:
                none_side.add((op, k))
            else:
                concrete.append((op, k, jd[k], vd[k]))
    return concrete, none_side


def test_paired_rule_defaults_agree():
    concrete, none_side = _paired_default_mismatches()
    assert not concrete, (
        "paired JVP/VJP rules disagree on a concrete kwarg default — J and Jᵀ "
        f"are differentiating two different functions: {concrete}")
    assert none_side == _KNOWN_NONE_CONVENTION, (
        "the None-convention mismatch set changed — additions are new findings "
        f"to triage, removals are fixes to record here: {sorted(none_side)}")


# A paired rule "swallows" a kwarg when the sibling mode declares it
# keyword-only but this rule has no named parameter for it and catches
# unknowns as `**_` (the repo convention for *ignore*, vs `**kwargs` for
# *forward*). That is the `jvp_clamp` bug class: the canonical caller's
# kwargs silently vanish and the rule computes something else.
#
# Triage record (2026-08-18). FIXED in the same sweep: clamp, clip (alias
# deafness), add/mul unary-`scalar` (unshifted/unscaled primal), and the
# fft/ifft/rfft/irfft `norm` family (√n-wrong under `norm="ortho"`). Each
# fix carries a pinned regression test below.
#
# _BENIGN_SWALLOWS: verified against the canonical forward's signature —
# the swallowed name can never be sent by a canonical caller, or is
# side-internal protocol. Keyed with the recorded reason.
_BENIGN_SWALLOWS = {
    # AD-LAW-1g: straight-through estimators. The tangent flows through
    # unchanged, so the derivative is identically 1 no matter what these keys
    # say — verified for both modes across both settings of each key
    # (test_quantize_ste_derivative_is_key_invariant). The keys select the
    # quantization grid, which an STE deliberately does not see.
    ("quantize_fp4", "jvp", "scale"): "STE derivative is key-invariant (=1)",
    ("quantize_fp6", "jvp", "scale"): "STE derivative is key-invariant (=1)",
    ("quantize_nvfp4", "jvp", "scale"): "STE derivative is key-invariant (=1)",
    ("quantize_int4", "jvp", "symmetric"): "STE derivative is key-invariant (=1)",
    ("quantize_int8", "jvp", "symmetric"): "STE derivative is key-invariant (=1)",

    ("adam", "jvp", "_output_index"):
        "vjp-side multi-output cotangent selector, not a forward kwarg",
    ("rope_split", "jvp", "_output_index"):
        "vjp-side multi-output cotangent selector, not a forward kwarg",
    ("moe_dispatch", "jvp", "transport"):
        "reference forward is value-identity for every transport; the "
        "kwarg selects a mechanism, not a function",
    # NOTE: reasons must not spell out `ops.<name>` — the test_coverage
    # scanner regex-matches that pattern in strings/comments and would count
    # prose as a direct test reference, overstating coverage (Codex review
    # on the AD-LAW-1b PR).
    ("pow", "vjp", "exponent"):
        "the canonical forward `(x, y)` is binary positional; the jvp-side "
        "unary `exponent` form is a non-canonical extra entry point the "
        "tape never records",
    ("sub", "vjp", "scalar"):
        "the canonical forward `(x, y)` is binary with no scalar kwarg; the "
        "jvp-side `scalar` param is dead vocabulary",
}

# _OPEN_SWALLOW_FINDINGS: rules where one mode declares a keyword-only key
# the sibling swallows via `**_`. **This class is now CLOSED** across the
# registry — the triage record:
#   * AD-LAW-1b: clamp/clip alias deafness, add/mul unary-`scalar`, and the
#     fft/ifft/rfft/irfft `norm` family, fixed in place.
#   * AD-LAW-1f: stft and spectral_conv hand JVPs DELETED — both are
#     bilinear, so the JVP is derived from the forward (linear.py) and
#     honors every key by construction.
#   * AD-LAW-1g: the five quantize STE keys, verified derivative-invariant
#     and moved to _BENIGN_SWALLOWS with evidence.
#   * AD-LAW-1h: jvp_istft rewritten to honor every key by construction
#     (each term comes from the canonical forward) while preserving its
#     window quotient — see test_istft_jvp_honors_every_config_key.
# Do not add entries: fix the rule, or move it to _BENIGN_SWALLOWS with a
# recorded reason. A NEW entry here means the gate caught a fresh instance.
_OPEN_SWALLOW_FINDINGS: set[tuple[str, str, str]] = set()

_KNOWN_SWALLOWED_KWARGS = _OPEN_SWALLOW_FINDINGS | set(_BENIGN_SWALLOWS)


def _swallowed_kwarg_mismatches():
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _VJPS

    KW = inspect.Parameter.KEYWORD_ONLY
    VK = inspect.Parameter.VAR_KEYWORD
    VP = inspect.Parameter.VAR_POSITIONAL

    def info(fn):
        ps = list(inspect.signature(fn).parameters.values())
        kwonly = {p.name for p in ps if p.kind == KW}
        named = {p.name for p in ps if p.kind not in (VK, VP)}
        var = next((p.name for p in ps if p.kind == VK), None)
        return kwonly, named, var

    found = set()
    for op in sorted(set(_JVPS) & set(_VJPS)):
        try:
            jk, jn, jv = info(_JVPS[op])
            vk, vn, vv = info(_VJPS[op])
        except (ValueError, TypeError):
            continue
        if jv == "_":
            for name in vk - jn:
                found.add((op, "jvp", name))
        if vv == "_":
            for name in jk - vn:
                found.add((op, "vjp", name))
    return found


def test_swallowed_kwarg_findings_are_pinned():
    found = _swallowed_kwarg_mismatches()
    new = found - _KNOWN_SWALLOWED_KWARGS
    fixed = _KNOWN_SWALLOWED_KWARGS - found
    assert not new, (
        "NEW swallowed-kwarg mismatch (the jvp_clamp bug class — a canonical "
        f"caller's kwargs silently vanish into `**_`): {sorted(new)}")
    assert not fixed, (
        "swallowed-kwarg finding fixed — remove it from _KNOWN_SWALLOWED_KWARGS "
        f"to record the triage outcome: {sorted(fixed)}")


def test_clamp_jvp_honors_canonical_kwargs():
    """The fixed instance, pinned directly: jvp_clamp used `min_val`/`max_val`
    while the forward and vjp_clamp use `min`/`max`, so canonical kwargs fell
    into `**_` and the JVP silently computed the unclamped identity (primal
    AND tangent) — a matched-degenerate pair the adjoint law alone could not
    see (#10a negative fixture for the fix)."""
    from tessera.autodiff.jvp import _JVPS

    x = np.array([[-2.0, -0.5, 0.5, 2.0]])
    dx = np.ones_like(x)
    y, dy = _JVPS["clamp"]((x,), (dx,), min=-1.0, max=1.0)
    np.testing.assert_allclose(y, np.clip(x, -1.0, 1.0))
    np.testing.assert_allclose(dy, np.array([[0.0, 1.0, 1.0, 0.0]]))


def test_clip_jvp_honors_alias_kwargs():
    """`ops.clip` documents `min`/`max` as aliases for `min_val`/`max_val`
    and vjp_clip coalesces both — jvp_clip was deaf to the aliases, giving
    alias callers an unclipped primal and ungated tangent (#10a fixture)."""
    from tessera.autodiff.jvp import _JVPS

    x = np.array([[-2.0, -0.5, 0.5, 2.0]])
    dx = np.ones_like(x)
    for kw in ({"min": -1.0, "max": 1.0}, {"min_val": -1.0, "max_val": 1.0}):
        y, dy = _JVPS["clip"]((x,), (dx,), **kw)
        np.testing.assert_allclose(y, np.clip(x, -1.0, 1.0))
        np.testing.assert_allclose(dy, np.array([[0.0, 1.0, 1.0, 0.0]]))


def test_add_mul_unary_scalar_jvp_matches_forward():
    """`add(x, scalar=c)` / `mul(x, scalar=c)` — the JVPs swallowed `scalar`
    and returned unshifted/unscaled primals (mul's tangent was wrong too)."""
    from tessera.autodiff.jvp import _JVPS

    x = np.array([1.0, -2.0, 3.0])
    dx = np.array([0.5, 1.0, -1.0])
    y, dy = _JVPS["add"]((x,), (dx,), scalar=2.5)
    np.testing.assert_allclose(y, x + 2.5)
    np.testing.assert_allclose(dy, dx)
    y, dy = _JVPS["mul"]((x,), (dx,), scalar=-3.0)
    np.testing.assert_allclose(y, x * -3.0)
    np.testing.assert_allclose(dy, dx * -3.0)


@pytest.mark.parametrize("op", ["fft", "ifft", "rfft", "irfft"])
def test_fft_family_jvp_honors_norm(op):
    """The fft-family JVPs swallowed `norm`/`normalization`, so
    `norm="ortho"` callers got backward-normalized results — wrong by √n
    on both primal and tangent (#10a fixture)."""
    from tessera.autodiff.jvp import _JVPS

    rng = np.random.default_rng(11)
    if op in ("fft", "ifft"):
        x = rng.standard_normal((3, 8)) + 1j * rng.standard_normal((3, 8))
    elif op == "rfft":
        x = rng.standard_normal((3, 8))
    else:  # irfft consumes a half-spectrum
        x = rng.standard_normal((3, 5)) + 1j * rng.standard_normal((3, 5))
    dx = x * 0.0 + 1.0
    ref = getattr(np.fft, op)
    for kw in ({"norm": "ortho"}, {"normalization": "ortho"}):
        y, dy = _JVPS[op]((x,), (dx,), **kw)
        np.testing.assert_allclose(y, ref(x, norm="ortho"), atol=1e-12)
        np.testing.assert_allclose(dy, ref(dx, norm="ortho"), atol=1e-12)


def test_lgamma_digamma_jvps_are_no_longer_placeholders():
    """AD-LAW-1c findings (#10a fixtures): jvp_lgamma's derivative was a dead
    `if False` stub returning identically 0.0, and jvp_digamma was a literal
    `# placeholder` computing identity-with-slope-1 — forward mode through
    either op silently returned garbage while reverse mode was correct. Both
    now derive from the same polygamma helpers as their VJPs (one datum)."""
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _digamma_positive, _trigamma_positive
    from tessera import ops

    x = np.array([0.7, 1.3, 2.9, 6.1])
    dx = np.ones_like(x)

    y, dy = _JVPS["lgamma"]((x,), (dx,))
    assert np.any(dy != 0.0), "lgamma JVP derivative is still the zero stub"
    np.testing.assert_allclose(dy, _digamma_positive(x), rtol=1e-12)

    y, dy = _JVPS["digamma"]((x,), (dx,))
    fwd = getattr(ops.digamma, "__wrapped__", ops.digamma)
    np.testing.assert_allclose(y, fwd(x), rtol=1e-12)
    np.testing.assert_allclose(dy, _trigamma_positive(x), rtol=1e-12)
    assert not np.allclose(y, x), "digamma JVP primal is still the identity placeholder"


def test_polygamma_helpers_handle_negative_domain_without_recurrence():
    """Codex review on #586: the polygamma helpers advanced by 1 per iteration
    until reaching 8, so a valid input like -1e9+0.5 spun for ~10^9 steps —
    a live defect in the *reverse* path (vjp_lgamma/vjp_digamma already used
    these) that the new forward wiring would have inherited. Reflection
    formulas now handle x <= 0, matching the canonical forward exactly."""
    from tessera import ops
    from tessera.autodiff.vjp import _digamma_positive, _trigamma_positive

    far_negative = np.array([-1e9 + 0.5])
    # Correctness (not just speed): psi(1-x) ~ log(1e9), and -pi/tan(pi*x) -> 0.
    np.testing.assert_allclose(_digamma_positive(far_negative),
                               [np.log(1e9)], rtol=1e-6)
    # psi'(x) = pi^2/sin^2(pi x) - psi'(1-x); sin^2 = 1 at the half-integer.
    np.testing.assert_allclose(_trigamma_positive(far_negative),
                               [np.pi ** 2], rtol=1e-6)

    fwd = getattr(ops.digamma, "__wrapped__", ops.digamma)
    xs = np.array([-7.3, -2.5, -0.4, 0.3, 1.0, 2.5, 7.9, 8.1, 50.0])
    np.testing.assert_allclose(_digamma_positive(xs), fwd(xs), rtol=1e-12)

    h = 1e-5
    fd = (fwd(xs + h) - fwd(xs - h)) / (2 * h)
    np.testing.assert_allclose(_trigamma_positive(xs), fd, rtol=1e-5)

    # Poles (non-positive integers) are nan, per the canonical scalar impl.
    assert np.all(np.isnan(_digamma_positive(np.array([-3.0, 0.0]))))
    assert np.all(np.isnan(_trigamma_positive(np.array([-3.0, 0.0]))))


def test_cast_jvp_accepts_canonical_dtype_strings():
    """The tape records `ops.cast`'s canonical dtype STRING; jvp_cast fed it
    to `np.astype` and crashed on every canonically-spelled cast (#10a)."""
    from tessera.autodiff.jvp import _JVPS

    x = np.array([1.0, 2.0], dtype=np.float64)
    y, dy = _JVPS["cast"]((x,), (np.ones_like(x),), dtype="fp32")
    assert y.dtype == np.float32 and dy.dtype == np.float32


def test_rmsnorm_eps_defaults_match_forward():
    """The sweep's first finding, pinned directly: jvp_rmsnorm defaulted
    eps=1e-6 (the rmsnorm_safe value) against the forward's and the VJP's
    1e-5 (#10a negative fixture for the fix)."""
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _VJPS

    j_eps = inspect.signature(_JVPS["rmsnorm"]).parameters["eps"].default
    v_eps = inspect.signature(_VJPS["rmsnorm"]).parameters["eps"].default
    assert j_eps == v_eps == 1e-5
    j_safe = inspect.signature(_JVPS["rmsnorm_safe"]).parameters["eps"].default
    v_safe = inspect.signature(_VJPS["rmsnorm_safe"]).parameters["eps"].default
    assert j_safe == v_safe == 1e-6


# ── positional-call conventions (the tape's replay contract) ────────────────
#
# `_swallowed_kwarg_mismatches` above compares the two *rules* against each
# other. It cannot see the other half of the same bug class: a canonical key
# the FORWARD accepts that neither rule reads, and a canonical key passed
# POSITIONALLY, which the tape used to record nowhere at all. Both of those
# fail open — the rule runs on its own default and returns a wrong gradient
# with no error — so they get an executable gate, not just a signature scan.


def _tape_grads(name, args, kwargs, diff_idx):
    """Forward + backward one op through the tape; return the input cotangents."""
    from tessera import ops
    from tessera.autodiff.tape import tape

    with tape() as t:
        out = getattr(ops, name)(*args, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        arr = np.asarray(out)
        t.backward(out, cotangent=np.ones_like(arr, dtype=arr.dtype))
    return [
        None if t.cotangent.get(id(args[i])) is None
        else np.asarray(t.cotangent[id(args[i])])
        for i in diff_idx
    ]


def _positionalize(sig_names, args, kwargs):
    """Move as many kwargs as possible into positional slots, in the forward's
    own parameter order. Stops at the first gap — a later parameter cannot be
    passed positionally once an earlier one is missing."""
    pos, rest, moved = list(args), dict(kwargs), []
    for i in range(len(args), len(sig_names)):
        if sig_names[i] not in rest:
            break
        pos.append(rest.pop(sig_names[i]))
        moved.append(sig_names[i])
    return pos, rest, moved


# Forwards that genuinely refuse the positional spelling — the call is not
# legal, so there is no gradient to compare and nothing for autodiff to fix.
# Pinned exactly: an addition means a rule regressed, a removal means a
# forward's signature opened up and the pin is stale.
_KNOWN_POSITIONAL_REFUSALS = {
    ("softmax", "axis"): (
        "the canonical forward declares `axis` keyword-only "
        "(`softmax(x, *, axis=-1, ...)`), so the positional call never reaches "
        "autodiff at all"
    ),
}


def _positional_equivalence_findings():
    from tessera.autodiff.tape import _positional_params
    from tessera import ops

    diffs, raised, skipped, compared = [], {}, [], []
    for name in sorted(LAW_INPUT_SPECS):
        if name not in _VJPS_TABLE():
            continue
        fwd = getattr(ops, name, None)
        names = _positional_params(fwd) if fwd is not None else None
        if names is None:
            continue
        try:
            args, kwargs = LAW_INPUT_SPECS[name].make(np.random.default_rng(7))
        except Exception as exc:  # pragma: no cover - a broken spec, not a finding
            skipped.append((name, f"spec.make: {exc}"))
            continue
        pos, rest, moved = _positionalize(names, args, kwargs)
        if not moved:
            continue
        spec = LAW_INPUT_SPECS[name]
        diff_idx = spec.diff_args if spec.diff_args is not None else tuple(
            i for i, a in enumerate(args) if isinstance(a, np.ndarray))
        try:
            base = _tape_grads(name, list(args), kwargs, diff_idx)
        except Exception as exc:  # pragma: no cover - keyword form must work
            skipped.append((name, f"keyword form failed: {type(exc).__name__}: {exc}"))
            continue
        try:
            alt = _tape_grads(name, pos, rest, diff_idx)
        except Exception as exc:
            for key in moved:
                raised[(name, key)] = f"{type(exc).__name__}: {exc}"
            continue
        compared.append(name)
        for b, a in zip(base, alt):
            if (b is None) != (a is None):
                diffs.append((name, moved, "one form produced no cotangent"))
                break
            if b is not None and not np.allclose(
                np.asarray(b, dtype=np.complex128),
                np.asarray(a, dtype=np.complex128),
                rtol=1e-10, atol=1e-12,
            ):
                diffs.append((name, moved, "gradients differ"))
                break
    return diffs, raised, skipped, compared


def _VJPS_TABLE():
    from tessera.autodiff.vjp import _VJPS

    return _VJPS


def test_positional_call_matches_keyword_call():
    """Passing a configuration argument positionally must not change the
    gradient. It used to: `_describe` recorded a str/bool positional nowhere
    (so a positionally-spelled `"mean"` reduction got the *sum* adjoint —
    silently) and turned an
    int positional into a float64 literal that was then handed to the forward
    (`np.sum(x, axis=array(0.0))`) and replayed as an extra positional the
    keyword-only rule could not bind. The tape now routes positional
    configuration into the recorded kwargs under its canonical parameter name.

    This grows with `LAW_INPUT_SPECS`: one new spec entry extends the gate."""
    diffs, raised, skipped, compared = _positional_equivalence_findings()
    # A gate that silently stops exercising anything is worse than no gate:
    # every op here has a spec whose kwargs the forward also takes positionally.
    assert len(compared) >= 10, (
        f"the positional-equivalence gate only exercised {compared} — it has "
        "gone vacuous (a spec or signature-resolution change, not a fix)")
    assert not diffs, (
        "positional and keyword forms of the SAME call disagree on the "
        f"gradient (fail-open — the worst class): {diffs}")
    unexpected = {k: v for k, v in raised.items() if k not in _KNOWN_POSITIONAL_REFUSALS}
    assert not unexpected, (
        "the positional form of a canonical call raises where the keyword form "
        f"works: {unexpected}")
    stale = set(_KNOWN_POSITIONAL_REFUSALS) - set(raised)
    assert not stale, (
        "a pinned positional refusal no longer reproduces — record the fix by "
        f"removing it from _KNOWN_POSITIONAL_REFUSALS: {sorted(stale)}")
    assert not skipped, f"an op's own keyword form is broken: {skipped}"


def test_reduce_positional_semantic_key_reaches_the_rule():
    """The finding that motivated the routing, pinned directly (#10a negative
    fixture): a positionally-spelled reduction key is a *semantic key*.
    It was `_NON_ARRAY`, so it reached the forward but was recorded nowhere,
    and `vjp_reduce` ran on its own `op="sum"` default — a mean forward with a
    sum gradient, no error (Decision #21a: a semantic key may not default)."""
    from tessera import ops
    from tessera.autodiff.tape import tape

    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    grads = {}
    for label, call in (
        ("keyword", lambda t_x: ops.reduce(t_x, op="mean")),
        ("positional", lambda t_x: ops.reduce(t_x, "mean")),
    ):
        buf = x.copy()
        with tape() as t:
            y = call(buf)
            t.backward(y, cotangent=np.ones_like(np.asarray(y), dtype=np.float64))
        grads[label] = np.asarray(t.cotangent[id(buf)])
    np.testing.assert_allclose(grads["positional"], grads["keyword"])
    # The mean adjoint is 1/n, NOT the sum adjoint's 1.0 that the drop produced.
    np.testing.assert_allclose(grads["positional"], np.full((3, 4), 1.0 / 12.0))


def test_tri_solve_rules_honor_lower():
    """`tri_solve`'s canonical key is `lower`; both rules declared `upper` — a
    vocabulary no caller produces — so `lower=False` silently got the *lower*
    solve's adjoint. The rules also solved against the raw matrix instead of
    the triangle the forward selects. Checked against finite differences on a
    FULL matrix, which separates both halves (#10a negative fixture)."""
    from tessera import ops
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.tape import tape

    rng = np.random.default_rng(11)
    A = rng.standard_normal((4, 4)) + 4.0 * np.eye(4)  # NOT triangular
    b = rng.standard_normal(4)
    fwd = getattr(ops.tri_solve, "__wrapped__", ops.tri_solve)

    for lower in (True, False):
        buf = A.copy()
        with tape() as t:
            y = ops.tri_solve(buf, b, lower=lower)
            t.backward(y, cotangent=np.ones_like(np.asarray(y)))
        got = np.asarray(t.cotangent[id(buf)])
        fd = np.zeros_like(A)
        eps = 1e-6
        for i in range(4):
            for j in range(4):
                up, dn = A.copy(), A.copy()
                up[i, j] += eps
                dn[i, j] -= eps
                fd[i, j] = (np.asarray(fwd(up, b, lower=lower)).sum()
                            - np.asarray(fwd(dn, b, lower=lower)).sum()) / (2 * eps)
        np.testing.assert_allclose(got, fd, rtol=1e-5, atol=1e-6)

        dA = rng.standard_normal((4, 4))
        primal, tangent = _JVPS["tri_solve"]((A, b), (dA, np.zeros(4)), lower=lower)
        np.testing.assert_allclose(primal, np.asarray(fwd(A, b, lower=lower)))
        fd_dir = (np.asarray(fwd(A + eps * dA, b, lower=lower))
                  - np.asarray(fwd(A - eps * dA, b, lower=lower))) / (2 * eps)
        np.testing.assert_allclose(tangent, fd_dir, rtol=1e-5, atol=1e-7)


def test_segment_reduce_rules_honor_op():
    """Same class as `tri_solve`: the forward's key is `op`, both rules
    declared `reduce`, so every `op="max"`/`"mean"` call took the *sum* branch
    and returned 1.0 everywhere. The non-sum handling existed but was
    unreachable — and, once reached, forwarded `reduce=` to a forward that
    only knows `op=` and finite-differenced the integer segment ids. Now exact
    for every mode, including non-contiguous segment ids (#10a fixture)."""
    from tessera import ops
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.tape import tape

    fwd = getattr(ops.segment_reduce, "__wrapped__", ops.segment_reduce)
    rng = np.random.default_rng(2)
    for shape, seg in (((4,), np.array([0, 0, 1, 1])),
                       ((4, 3), np.array([0, 1, 1, 2])),
                       ((4,), np.array([0, 0, 7, 7]))):  # ids need not be dense
        x = rng.standard_normal(shape) + 2.0
        for op in ("sum", "mean", "max", "min", "prod"):
            buf = x.copy()
            with tape() as t:
                y = ops.segment_reduce(buf, seg, op=op)
                t.backward(y, cotangent=np.ones_like(np.asarray(y), dtype=np.float64))
            got = np.asarray(t.cotangent[id(buf)])
            fd = np.zeros(x.shape)
            eps = 1e-6
            for idx in np.ndindex(x.shape):
                up, dn = x.copy(), x.copy()
                up[idx] += eps
                dn[idx] -= eps
                fd[idx] = (np.asarray(fwd(up, seg, op=op)).sum()
                           - np.asarray(fwd(dn, seg, op=op)).sum()) / (2 * eps)
            np.testing.assert_allclose(got, fd, rtol=1e-5, atol=1e-5,
                                       err_msg=f"vjp {op} {shape}")
            dx = rng.standard_normal(x.shape)
            primal, tangent = _JVPS["segment_reduce"]((x, seg), (dx, None), op=op)
            np.testing.assert_allclose(primal, np.asarray(fwd(x, seg, op=op)))
            fd_dir = (np.asarray(fwd(x + eps * dx, seg, op=op))
                      - np.asarray(fwd(x - eps * dx, seg, op=op))) / (2 * eps)
            np.testing.assert_allclose(tangent, fd_dir, rtol=1e-5, atol=1e-5,
                                       err_msg=f"jvp {op} {shape}")


def test_scalar_operands_stay_positional():
    """The routing must not over-reach: a python scalar that is a genuine
    differentiable *operand* (a scalar factor in a product) still has to arrive as a
    positional input, including for the rules that rename their operands
    (`vjp_mod(dout, a, b)` over `mod(x, y)`) — the name mismatch is why this
    needs its own fixture."""
    from tessera import ops
    from tessera.autodiff.tape import tape

    cases = {
        "mul": (lambda t_x: ops.mul(t_x, -0.5), -0.5),
        "mod": (lambda t_x: ops.mod(t_x, 3), 1.0),
        "pow": (lambda t_x: ops.pow(t_x, 2.0), None),
    }
    x = np.array([[1.5, 2.5, 3.5]])
    for name, (call, expected) in cases.items():
        buf = x.copy()
        with tape() as t:
            y = call(buf)
            t.backward(y, cotangent=np.ones_like(np.asarray(y), dtype=np.float64))
        got = t.cotangent.get(id(buf))
        assert got is not None, f"{name}: scalar operand lost its cotangent"
        if expected is not None:
            np.testing.assert_allclose(np.asarray(got), np.full(x.shape, expected))
    # pow's gradient is 2x — spelled out rather than pinned to a constant.
    buf = x.copy()
    with tape() as t:
        y = ops.pow(buf, 2.0)
        t.backward(y, cotangent=np.ones_like(np.asarray(y), dtype=np.float64))
    np.testing.assert_allclose(np.asarray(t.cotangent[id(buf)]), 2.0 * x)


def test_forward_mode_positional_config_matches_keyword():
    """Forward mode had the same hole with a different face: a positional
    configuration argument became an extra *primal*, so a rule's `(x,) =
    primals` unpack raised `too many values to unpack` — which also masked the
    op's own diagnostic. Both modes now make the same operand/config split."""
    from tessera import ops
    from tessera.autodiff.jvp import _ACTIVE_JVP, _JVPTrace

    def tangent_of(call, x, dx):
        trace = _JVPTrace()
        trace.bind(x, dx)
        token = _ACTIVE_JVP.set(trace)
        try:
            return trace.tangents.get(id(call(x)))
        finally:
            _ACTIVE_JVP.reset(token)

    x = np.arange(12.0).reshape(3, 4)
    dx = np.ones_like(x)
    pairs = (
        (lambda t_x: ops.sum(t_x, axis=0), lambda t_x: ops.sum(t_x, 0)),
        (lambda t_x: ops.cumsum(t_x, axis=0), lambda t_x: ops.cumsum(t_x, 0)),
        (lambda t_x: ops.mean(t_x, axis=0, keepdims=True),
         lambda t_x: ops.mean(t_x, 0, True)),
    )
    for kw_call, positional_call in pairs:
        np.testing.assert_allclose(tangent_of(positional_call, x, dx),
                                   tangent_of(kw_call, x, dx))

    # A rejected configuration must still reach the caller as the op's own
    # diagnostic, not as an unpack error from the rule's signature.
    with pytest.raises(Exception, match="reduce"):
        tangent_of(lambda t_x: ops.reduce(t_x, "mean"), x, dx)


# A forward configuration key that NEITHER rule reads: it is not in the rule's
# named parameters and the rule catches unknowns with `**_` (the repo's
# convention for *ignore*, vs `**kwargs` for *forward*). This is the
# forward-vs-rule half of the `jvp_clamp` class, and it is what `tri_solve`
# and `segment_reduce` were. Every entry below is an OPEN finding awaiting a
# body read — some are benign (a storage-dtype key cannot change a gradient),
# but benign is a conclusion you reach by reading, not by assuming. Do not add
# entries: fix the rule, or move it to a named-benign set with the reason.
# Triage outcome recorded 2026-08-19 (AD-LAW-1d rebased onto main):
# ("fft"/"ifft"/"rfft"/"irfft", "jvp", "norm") were fixed by AD-LAW-1b
# (the JVPs now accept `norm`/`normalization` and normalize like the
# forward), so they leave this set rather than being re-pinned.
# AD-LAW-1g: the dequantize_nvfp4 block_size entries left this set —
# both modes now READ the key so the per-block scale expands exactly
# as the canonical forward blocks it (before, both crashed outright).
_OPEN_FORWARD_KEY_SWALLOWS = {
    ("adam", "jvp", "cast_updates_to_param_dtype"),
    ("adam", "jvp", "compute_dtype"),
    ("adam", "jvp", "state_dtype"),
    ("adam", "vjp", "cast_updates_to_param_dtype"),
    ("adam", "vjp", "compute_dtype"),
    ("adam", "vjp", "state_dtype"),
    ("all_reduce", "jvp", "axis"),
    ("all_reduce", "vjp", "axis"),
    ("all_to_all", "jvp", "axis"),
    ("all_to_all", "vjp", "axis"),
    ("conv2d", "vjp", "layout"),
    ("conv3d", "vjp", "layout"),
    ("dequant_matmul", "vjp", "backend"),
    ("dequantize_fp4", "jvp", "format"),
    ("dequantize_fp4", "vjp", "format"),
    ("dequantize_fp6", "jvp", "format"),
    ("dequantize_fp6", "vjp", "format"),
    ("dequantize_fp8", "vjp", "format"),
    ("grouped_gemm", "jvp", "kind"),
    ("grouped_gemm", "vjp", "kind"),
    ("image_resize", "jvp", "antialias"),
    ("momentum", "jvp", "cast_updates_to_param_dtype"),
    ("momentum", "jvp", "compute_dtype"),
    ("momentum", "jvp", "state_dtype"),
    ("momentum", "vjp", "cast_updates_to_param_dtype"),
    ("momentum", "vjp", "compute_dtype"),
    ("momentum", "vjp", "state_dtype"),
    ("nesterov", "jvp", "cast_updates_to_param_dtype"),
    ("nesterov", "jvp", "compute_dtype"),
    ("nesterov", "jvp", "state_dtype"),
    ("nesterov", "vjp", "cast_updates_to_param_dtype"),
    ("nesterov", "vjp", "compute_dtype"),
    ("nesterov", "vjp", "state_dtype"),
    ("online_softmax_state", "jvp", "axis"),
    ("online_softmax_state", "vjp", "axis"),
    ("quantize_fp4", "jvp", "format"),
    ("quantize_fp4", "vjp", "format"),
    ("quantize_fp6", "jvp", "format"),
    ("quantize_fp6", "vjp", "format"),
    ("quantize_fp8", "vjp", "format"),
    ("quantize_nvfp4", "jvp", "block_size"),
    ("quantize_nvfp4", "vjp", "block_size"),
}


def _forward_key_swallow_mismatches():
    from tessera import ops
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.tape import _innermost
    from tessera.autodiff.vjp import _VJPS

    VK, VP = inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL

    def sig_of(fn):
        try:
            return inspect.signature(_innermost(fn), follow_wrapped=False)
        except (TypeError, ValueError):
            return None

    found = set()
    for op in sorted(set(_VJPS) | set(_JVPS)):
        fwd = getattr(ops, op, None)
        fsig = sig_of(fwd) if fwd is not None else None
        if fsig is None:
            continue
        keys = []
        for i, p in enumerate(fsig.parameters.values()):
            if i == 0 or p.kind in (VK, VP):
                continue
            ann, default = p.annotation, p.default
            scalarish = (
                ann in (int, float, str, bool)
                or (isinstance(ann, str)
                    and ann.split("|")[0].strip() in ("int", "float", "str", "bool"))
                or isinstance(default, (int, float, str, bool))
            )
            if scalarish:
                keys.append(p.name)
        if not keys:
            continue
        for registry, table in (("vjp", _VJPS), ("jvp", _JVPS)):
            rule = table.get(op)
            rsig = sig_of(rule) if rule is not None else None
            if rsig is None:
                continue
            params = list(rsig.parameters.values())
            var_kw = next((p.name for p in params if p.kind == VK), None)
            if var_kw != "_":
                continue  # `**kwargs` forwards; only `**_` means *ignore*
            named = {p.name for p in params if p.kind not in (VK, VP)}
            for key in keys:
                if key not in named:
                    found.add((op, registry, key))
    return found


def test_forward_key_swallow_findings_are_pinned():
    """The forward-vs-rule scan. `_swallowed_kwarg_mismatches` compares the two
    rules against each other and so is blind to a key BOTH of them ignore —
    which is exactly what `tri_solve` (`lower` vs a declared-but-unproduced
    `upper`) and `segment_reduce` (`op` vs `reduce`) were, both silently
    returning a different function's gradient."""
    found = _forward_key_swallow_mismatches()
    new = found - _OPEN_FORWARD_KEY_SWALLOWS
    fixed = _OPEN_FORWARD_KEY_SWALLOWS - found
    assert not new, (
        "NEW forward-key swallow — the canonical forward accepts a key that "
        f"neither rule reads, so the rule differentiates something else: {sorted(new)}")
    assert not fixed, (
        "forward-key swallow finding fixed — remove it from "
        f"_OPEN_FORWARD_KEY_SWALLOWS to record the triage outcome: {sorted(fixed)}")


# ── dashboard determinism ────────────────────────────────────────────────────


def test_law_dashboard_is_deterministic():
    from tessera.compiler.law_audit import render_csv

    assert render_csv() == render_csv()


def test_law_dashboard_carries_no_floats():
    """The gated CSV must stay byte-stable across BLAS builds — statuses and
    integer probe counts only."""
    from tessera.compiler.law_audit import render_csv

    body = render_csv().splitlines()[1:]
    for line in body:
        registry, op, law, status, probes = line.split(",")
        assert status in ("pass", "fail", "rule_error", "no_spec",
                          "jvp_only", "vjp_only", "not_applicable"), line
        int(probes)


# ── Law 5: kink policy (AD-LAW-1e) ───────────────────────────────────────────


def test_every_declared_nonsmooth_op_has_a_kink_probe(sweep):
    """Decision #29: `NONSMOOTH_SELECTION` declares a policy per op; Law 5 is
    its consumer. An op that declares a policy nobody probes is an unconsumed
    declaration."""
    from tessera.autodiff.nonsmooth import NONSMOOTH_SELECTION

    kink_rows = {r.op: r for r in sweep if r.law == "kink"}
    for op in NONSMOOTH_SELECTION:
        assert op in kink_rows, f"{op} declares a policy but has no Law-5 row"
        assert kink_rows[op].status == "pass", (
            f"{op}: kink law {kink_rows[op].status} — {kink_rows[op].detail}")


def test_kink_law_catches_wrong_side_selection():
    """A rule that returns grad 1 AT the relu kink violates the declared
    SUBGRAD_ZERO policy — legal as *a* Clarke element, wrong as *the declared*
    one, and invisible to Laws 1/3 which sample away from the kink."""
    from tessera.autodiff.law_inputs import KINK_SPECS
    from tessera.autodiff.laws import kink_check

    def inclusive_relu_vjp(dout, x, **_):
        return (np.asarray(dout) * (np.asarray(x) >= 0).astype(np.float64),)

    r = kink_check("relu", KINK_SPECS["relu"], inclusive_relu_vjp, None)
    assert r.status == "fail" and "declared" in r.detail, r


def test_kink_law_catches_hard_select_instead_of_split():
    """SUBGRAD_SPLIT requires an equal share among tied arguments; a hard
    argmax select passes every smooth-point test but breaks the declared
    policy."""
    from tessera.autodiff.law_inputs import KINK_SPECS
    from tessera.autodiff.laws import kink_check

    def hard_select_vjp(dout, x, *, axis=None, **_):
        a = np.asarray(x)
        first = np.zeros_like(a)
        np.put_along_axis(first, np.argmax(a, axis=-1)[..., None], 1.0, axis=-1)
        return (np.asarray(dout) * first,)

    r = kink_check("amax", KINK_SPECS["amax"], hard_select_vjp, None)
    assert r.status == "fail" and "not equal" in r.detail, r


def test_kink_law_catches_unconserved_tie_mass():
    """Regression for a flaw in the law itself: an earlier conservation check
    accepted `total == n_tied * share`, which is vacuous when the shares are
    equal — so a rule handing EVERY tied element the full cotangent passed.
    The target is 1.0 exactly (one tie group per probe)."""
    from tessera.autodiff.law_inputs import KINK_SPECS
    from tessera.autodiff.laws import kink_check

    def full_mass_vjp(dout, x, *, axis=None, **_):
        a = np.asarray(x)
        m = (a == np.max(a, axis=-1, keepdims=True)).astype(np.float64)
        return (np.asarray(dout) * m,)

    r = kink_check("amax", KINK_SPECS["amax"], full_mass_vjp, None)
    assert r.status == "fail" and "mass not conserved" in r.detail, r


def test_kink_law_fails_closed_on_undeclared_policy():
    """#21a: an op with no declared selection must error, never default."""
    from tessera.autodiff.law_inputs import KINK_SPECS
    from tessera.autodiff.laws import kink_check

    r = kink_check("not_a_nonsmooth_op", KINK_SPECS["relu"],
                   lambda dout, x, **_: (dout,), None)
    assert r.status == "rule_error" and "undeclared" in r.detail, r


# ── AD-LAW-1f: spectral JVPs derived from the forward ────────────────────────


@pytest.mark.parametrize("op,linear_args", [
    ("stft", (0, 1)), ("spectral_conv", (0, 1)),
])
def test_spectral_ops_are_declared_multilinear(op, linear_args):
    """The declaration that makes derivation legal — verified numerically,
    not assumed: f(a+s·a′, b) == f(a,b) + s·f(a′,b) for each linear slot."""
    from tessera import ops
    from tessera.autodiff.linear import MULTILINEAR_PRIMITIVES

    assert MULTILINEAR_PRIMITIVES[op] == linear_args
    fwd = getattr(getattr(ops, op), "__wrapped__", getattr(ops, op))
    rng = np.random.default_rng(17)
    if op == "stft":
        args, kw = [rng.standard_normal(64), np.hanning(16)], {"hop": 4}
        alt = [rng.standard_normal(64), np.hanning(16) * 0.7]
    elif op == "spectral_conv":
        args, kw = [rng.standard_normal(32), rng.standard_normal(32)], {}
        alt = [rng.standard_normal(32), rng.standard_normal(32)]
    s = 2.7
    for i in linear_args:
        shifted = list(args)
        shifted[i] = args[i] + s * alt[i]
        swapped = list(args)
        swapped[i] = alt[i]
        np.testing.assert_allclose(
            fwd(*shifted, **kw), fwd(*args, **kw) + s * fwd(*swapped, **kw),
            atol=1e-9, err_msg=f"{op} is not linear in arg {i}")


def test_spectral_jvps_are_derived_from_the_forward():
    """AD-LAW-1f: the hand-written spectral JVPs reimplemented the transform
    and honored NONE of the forward's configuration — `jvp_stft` hardcoded
    n_fft=512/hop=128 and ignored the positional window and hop entirely, so
    it RAISED on any canonical call, while `vjp_stft` was a faithful
    transpose honoring every key. J and Jᵀ were derivatives of different
    functions. Now derived from the forward, so the keys are honored by
    construction."""
    from tessera import ops
    from tessera.autodiff.jvp import _JVPS

    rng = np.random.default_rng(3)
    x, win, hop = rng.standard_normal(64), np.hanning(16), 4
    fwd = getattr(ops.stft, "__wrapped__", ops.stft)

    # 1. The canonical call no longer raises, and the primal IS the forward.
    y, _ = _JVPS["stft"]((x, win), (np.zeros_like(x), np.zeros_like(win)),
                         hop=hop)
    np.testing.assert_allclose(y, fwd(x, win, hop=hop), atol=1e-12)

    # 2. Non-default keys are honored — the exact thing the old rule ignored.
    kw = {"hop": hop, "center": True, "norm": "ortho"}
    dx, dw = rng.standard_normal(64), rng.standard_normal(16)
    y, dy = _JVPS["stft"]((x, win), (dx, dw), **kw)
    np.testing.assert_allclose(y, fwd(x, win, **kw), atol=1e-12)
    eps = 1e-6
    fd = (fwd(x + eps * dx, win + eps * dw, **kw)
          - fwd(x - eps * dx, win - eps * dw, **kw)) / (2 * eps)
    np.testing.assert_allclose(dy, fd, atol=1e-7)


def test_spectral_ops_pass_the_adjoint_law(sweep):
    """The hand-written VJPs are faithful transposes; the derived JVPs must
    pair with them exactly."""
    rows = {r.op: r for r in sweep if r.law == "adjoint"}
    for op in ("stft", "spectral_conv"):
        assert rows[op].status == "pass", f"{op}: {rows[op].status} {rows[op].detail}"


def test_istft_jvp_honors_every_config_key():
    """AD-LAW-1h closes the last pinned swallow group.

    `jvp_istft` used to reimplement the transform: it pinned axis=-1,
    onesided=True, norm="backward" and dropped `center`/`length`, so it
    differentiated a different function than the forward for any non-default
    configuration — while `vjp_istft` honored all of them. It could not
    simply be derived (as stft/spectral_conv were in AD-LAW-1f) because
    istft's normalized overlap-add is NOT linear in the window.

    The rewrite takes each term from the canonical forward itself — the
    spectrum term is the forward applied to the tangent (exact, since the
    map is linear there), and the window term is the quotient rule with only
    the FFT-free window overlap sums computed locally. So every key is
    honored by construction, and the window quotient is preserved.
    """
    import itertools

    from tessera import ops
    from tessera.autodiff.jvp import _JVPS

    fwd = getattr(ops.istft, "__wrapped__", ops.istft)
    rng = np.random.default_rng(21)
    X = rng.standard_normal((13, 9)) + 1j * rng.standard_normal((13, 9))
    win = np.hanning(16) + 0.25
    dX = rng.standard_normal((13, 9)) + 1j * rng.standard_normal((13, 9))
    dwin = rng.standard_normal(16)
    eps = 1e-6

    # The keys must actually change the answer, or this proves nothing.
    assert not np.allclose(fwd(X, win, hop=4, norm="backward"),
                           fwd(X, win, hop=4, norm="ortho"))
    assert (fwd(X, win, hop=4, center=True).shape
            != fwd(X, win, hop=4).shape)

    for center, norm, length in itertools.product(
            (False, True), ("backward", "ortho"), (None, 40)):
        cfg = {"hop": 4, "center": center, "norm": norm, "length": length}
        y, dy = _JVPS["istft"]((X, win), (dX, dwin), **cfg)
        np.testing.assert_allclose(y, fwd(X, win, **cfg), atol=1e-12,
                                   err_msg=f"primal mismatch at {cfg}")
        fd = (fwd(X + eps * dX, win + eps * dwin, **cfg)
              - fwd(X - eps * dX, win - eps * dwin, **cfg)) / (2 * eps)
        rel = np.max(np.abs(dy - fd)) / (np.max(np.abs(fd)) + 1e-12)
        assert rel < 1e-6, f"tangent vs FD rel={rel:.2e} at {cfg}"

    # The window tangent — which a naive derivation would have zeroed — is
    # still computed, and a non-default `axis` on rank-3 input works.
    _, dy = _JVPS["istft"]((X, win), (np.zeros_like(X), dwin), hop=4)
    assert np.any(np.abs(dy) > 1e-9), "window tangent is identically zero"

    X3 = rng.standard_normal((2, 13, 9)) + 1j * rng.standard_normal((2, 13, 9))
    dX3 = rng.standard_normal((2, 13, 9)) + 1j * rng.standard_normal((2, 13, 9))
    for axis in (-1, 2):
        cfg = {"hop": 4, "axis": axis}
        y, dy = _JVPS["istft"]((X3, win), (dX3, dwin), **cfg)
        fd = (fwd(X3 + eps * dX3, win + eps * dwin, **cfg)
              - fwd(X3 - eps * dX3, win - eps * dwin, **cfg)) / (2 * eps)
        assert np.max(np.abs(dy - fd)) / (np.max(np.abs(fd)) + 1e-12) < 1e-6


def test_swallow_class_is_closed():
    """The headline of AD-LAW-1b/f/g/h: no rule in either registry now
    swallows a key its sibling declares. A new entry means a regression."""
    assert _OPEN_SWALLOW_FINDINGS == set()


# ── AD-LAW-1g: the quantize family ───────────────────────────────────────────


@pytest.mark.parametrize("op,key,values", [
    ("quantize_int8", "symmetric", (True, False)),
    ("quantize_int4", "symmetric", (True, False)),
    ("quantize_fp4", "scale", (None, 4.0)),
    ("quantize_fp6", "scale", (None, 4.0)),
    ("quantize_nvfp4", "scale", (None, 4.0)),
])
def test_quantize_ste_derivative_is_key_invariant(op, key, values):
    """The evidence behind classifying these swallows benign: a
    straight-through estimator's derivative is identically 1, so the keys
    that select the quantization grid cannot change it. Verified in BOTH
    modes across both settings — not argued from the docstring."""
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _VJPS

    x = np.random.default_rng(11).standard_normal(8) * 3
    seen_j, seen_v = [], []
    for v in values:
        _, dy = _JVPS[op]((x,), (np.ones_like(x),), **{key: v})
        (g,) = _VJPS[op](np.ones_like(x), x, **{key: v})
        seen_j.append(np.asarray(dy, dtype=np.float64))
        seen_v.append(np.asarray(g, dtype=np.float64))
    np.testing.assert_allclose(seen_j[0], seen_j[1], atol=0)
    np.testing.assert_allclose(seen_v[0], seen_v[1], atol=0)
    np.testing.assert_allclose(seen_j[0], np.ones_like(x), atol=0)


def test_dequantize_nvfp4_handles_per_block_scales():
    """AD-LAW-1g's hard defect: nvfp4's canonical forward takes a per-block
    scale array, but both rules reached `float(scale)` and raised
    `TypeError: only 0-dimensional arrays ...` — so every canonical
    multi-block call was unusable in either mode. Now broadcast per block,
    with `block_size` read rather than swallowed."""
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _VJPS

    q = np.ones(32)
    scales = (np.arange(4) + 1).astype(np.float64)   # 4 blocks of 8
    (g,) = (_VJPS["dequantize_nvfp4"](np.ones_like(q), q, scales,
                                      block_size=8)[:1])
    _, dy = _JVPS["dequantize_nvfp4"]((q, scales),
                                      (np.ones_like(q), np.zeros_like(scales)),
                                      block_size=8)
    # Each block carries its own scale, and the modes agree.
    np.testing.assert_allclose(np.asarray(g)[::8], scales)
    np.testing.assert_allclose(np.asarray(dy)[::8], scales)
    # The scalar-scale path is unchanged.
    (gs,) = _VJPS["dequantize_fp4"](np.ones(6), np.arange(6.0),
                                    np.float32(0.5))[:1]
    np.testing.assert_allclose(np.asarray(gs), np.full(6, 0.5))


def test_dequantize_rule_contradicts_stub_forward():
    """Recorded open question, NOT silently resolved (claim integrity).

    Every `dequantize_*` reference forward is an identity stub: it validates
    the format key and returns the container as fp32, ignoring `scale`. The
    AD rules instead implement the true scaled inverse (d/dq = scale). Both
    cannot be right, and which one moves is a quantization-track decision
    about completing the forward — not something the autodiff layer should
    invent. This test PINS the contradiction so it stays visible: it fails
    the day either side changes, which is exactly when a human should look.
    """
    from tessera import ops
    from tessera.autodiff.vjp import _VJPS

    # eps is deliberately coarse: the stub forward casts to float32, so a
    # 1e-4 step lands in cancellation noise.
    q, scale, eps = np.arange(6.0), np.float32(0.5), 1e-2
    fwd = getattr(ops.dequantize_fp4, "__wrapped__", ops.dequantize_fp4)

    # The stub forward is the identity in the container.
    np.testing.assert_allclose(fwd(q, scale), q.astype(np.float32))
    fd = (fwd(q + eps, scale) - fwd(q - eps, scale)) / (2 * eps)
    np.testing.assert_allclose(fd, np.ones_like(q), atol=1e-2)

    # The rule assumes the scaled inverse.
    (g,) = _VJPS["dequantize_fp4"](np.ones_like(q), q, scale)[:1]
    np.testing.assert_allclose(np.asarray(g), np.full(6, float(scale)))


# ── AD-LAW-2: Laws 2 and 4, the AD-WEIL-1 gate ───────────────────────────────


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_law2_homomorphism_is_exact_on_polynomials(order):
    """Law 2: evaluating a polynomial with only the algebra's ring operations
    reproduces its EXACT Taylor coefficients. Nilpotency removes truncation
    error entirely for degree <= k, so the only residual is float rounding —
    machine precision, not a tuned tolerance."""
    from tessera.autodiff.laws import homomorphism_check

    r = homomorphism_check(order)
    assert r.status == "pass", r
    assert r.max_rel_residual < 1e-13, r.max_rel_residual


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_law4_jet_equals_nested_duals(order):
    """Law 4: order-k jets agree with the k-nested dual tower they would
    replace, through random programs including exp/log/tanh/sin/cos. This is
    the differential proof Decision #31 requires before the nested route can
    ever be retired."""
    from tessera.autodiff.laws import quotient_check

    r = quotient_check(order)
    assert r.status == "pass", r


def test_law2_is_not_vacuous():
    """Regression for a flaw in the law itself: the first draft compared
    `W.mul(f, g)` against `W.mul(f, g)` — trivially equal — and used an
    absolute threshold on values a random program can make large. A corrupted
    `mul` must now fail."""
    from tessera.autodiff import laws as L
    from tessera.autodiff.algebra import TruncatedJet

    real = L.homomorphism_check(3)
    assert real.status == "pass"

    original = TruncatedJet.mul

    def corrupted_mul(self, a, b):
        """Truncated Cauchy product with the top coefficient scaled — the
        kind of subtle product error a vacuous law would miss."""
        k = self.order
        out = [np.zeros_like(np.asarray(a[0])) for _ in range(k + 1)]
        for i in range(k + 1):
            for j in range(k + 1 - i):
                out[i + j] = out[i + j] + a[i] * b[j]
        out[-1] = out[-1] * 0.5
        return out

    try:
        TruncatedJet.mul = corrupted_mul           # type: ignore[assignment]
        broken = L.homomorphism_check(3)
    finally:
        TruncatedJet.mul = original                # type: ignore[assignment]
    assert broken.status == "fail", "Law 2 did not catch a corrupted product"


def test_law4_catches_a_wrong_recurrence():
    """A jet recurrence that is subtly wrong must fail against the nested
    reference — that is what makes Law 4 a migration gate rather than a
    smoke test."""
    from tessera.autodiff import algebra as A
    from tessera.autodiff.laws import quotient_check

    good = A.SCALAR_RECURRENCES["tanh"]

    def wrong_jet(W, u):
        w = good.jet(W, u)
        if len(w) > 2:
            w[2] = w[2] * 1.05                    # 5% error at order 2
        return w

    saved = dict(A.SCALAR_RECURRENCES)
    try:
        # Restrict the table so the random program MUST select the corrupted
        # primitive — with a dozen entries it might otherwise never pick it,
        # and the mutation test would pass for the wrong reason.
        A.SCALAR_RECURRENCES.clear()
        A.SCALAR_RECURRENCES["tanh"] = A.ScalarRecurrence(
            good.value, good.derivative_expr, wrong_jet)
        r = quotient_check(3)
    finally:
        A.SCALAR_RECURRENCES.clear()
        A.SCALAR_RECURRENCES.update(saved)
    assert r.status == "fail", "Law 4 did not catch a wrong recurrence"


def test_jet_dimension_is_k_plus_one_not_two_to_the_k():
    """The cost claim behind the whole program, measured rather than asserted:
    an order-k jet carries k+1 coefficients, while the k-nested dual tower it
    replaces carries 2^k. This is why 'run AutodiffPass twice' is the wrong
    higher-order path."""
    from tessera.autodiff.algebra import TruncatedJet, _NestedDual

    for k in (1, 2, 3, 4, 5):
        jet = TruncatedJet(k).lift(np.asarray(0.5), np.asarray(1.0))
        assert len(jet) == k + 1

        # A GENERIC element of the k-fold tensor product is a full binary
        # tree of depth k — one component per subset of {ε₁..ε_k}. (An
        # earlier version built a left-spine, which has only k+1 leaves and
        # would have "confirmed" the claim for the wrong reason.)
        def generic(depth):
            if depth == 0:
                return 1.0
            return _NestedDual(generic(depth - 1), generic(depth - 1))

        def count(node):
            if isinstance(node, _NestedDual):
                return count(node.a) + count(node.b)
            return 1

        assert count(generic(k)) == 2 ** k
    # At k=5 that is 6 coefficients versus 32 — a 5.3x gap that widens.


def test_istft_window_tangent_at_the_denominator_floor():
    """Covers the region every other istft test dodges (review finding).

    The forward divides by `max(OLA(w**2), 1e-12)`. A window with exact zero
    endpoints — `np.hanning(n)`, the canonical choice — puts the first and
    last output positions inside that floor, where the window derivative is
    one-sided and enormous. Every other test here uses `hanning + 0.25`,
    which avoids the region entirely, so nothing exercised it.

    Two things are pinned: the tangent is *correct for the clamped function*
    (machine-precision relative agreement with a central difference, which is
    the claim the rule's docstring makes), and its magnitude really is
    floor-driven — so a future change that quietly regularizes the floor will
    show up here rather than in a user's optimizer.
    """
    from tessera import ops
    from tessera.autodiff.jvp import _JVPS, _window_overlap_sums

    fwd = getattr(ops.istft, "__wrapped__", ops.istft)
    rng = np.random.default_rng(5)
    X = rng.standard_normal((13, 9)) + 1j * rng.standard_normal((13, 9))
    win = np.hanning(16)                     # exact zeros at both ends
    dwin = rng.standard_normal(16)
    assert win[0] == 0.0 and win[-1] == 0.0

    # The floor is genuinely active at position 0.
    b_w, _, _ = _window_overlap_sums(win, dwin, 64, 4, 16)
    assert b_w[0] == 1e-12, "probe no longer reaches the clamped region"

    _, dy = _JVPS["istft"]((X, win), (np.zeros_like(X), dwin), hop=4)
    eps = 1e-6
    fd = (fwd(X, win + eps * dwin, hop=4)
          - fwd(X, win - eps * dwin, hop=4)) / (2 * eps)

    rel = np.abs(dy - fd) / (np.abs(fd) + 1e-30)
    assert rel.max() < 1e-6, f"relative disagreement {rel.max():.2e}"

    # Floor-driven magnitude: orders of magnitude above the interior.
    assert abs(dy[0]) > 1e6 > abs(dy[8:-8]).max()


def test_derivative_datum_is_declared_once():
    """Review finding: `_NestedScalarOps._derivative` used to restate every
    derivative as a hardcoded if-chain, duplicating what the registry already
    declared — inside the module whose thesis is one datum per primitive. It
    now evaluates the registry's single `derivative_expr`, so the scalar and
    tower paths cannot drift, and the table is extensible.
    """
    import numpy as _np

    from tessera.autodiff import algebra as A

    # One declaration, two evaluations: scalar `pointwise` is derived from it.
    for name, rec in A.SCALAR_RECURRENCES.items():
        _, df = rec.pointwise
        x = 0.6
        analytic = {
            "exp": _np.exp(x),
            "log": 1.0 / x,
            "tanh": 1.0 - _np.tanh(x) ** 2,
            "sin": _np.cos(x),
            "cos": -_np.sin(x),
            "sqrt": 0.5 / _np.sqrt(x),
            "reciprocal": -1.0 / x ** 2,
            "sinh": _np.cosh(x),
            "cosh": _np.sinh(x),
            "atan": 1.0 / (1.0 + x * x),
            "expm1": _np.exp(x),
            "log1p": 1.0 / (1.0 + x),
            "sigmoid": (lambda s: s * (1 - s))(1.0 / (1.0 + _np.exp(-x))),
        }[name]
        _np.testing.assert_allclose(df(x), analytic, rtol=1e-14)

    # Registering a NEW primitive extends the nested reference automatically;
    # the old if-chain raised KeyError, which degraded Law 4 to `rule_error`.
    from tessera.autodiff.laws import quotient_check

    def _jet_sqrt(W, u):
        k = W.order
        w = [_np.sqrt(u[0])] + [_np.zeros_like(_np.asarray(u[0]))
                                for _ in range(k)]
        for n in range(1, k + 1):
            acc = u[n]
            for j in range(1, n):
                acc = acc - w[j] * w[n - j]
            w[n] = acc / (2 * w[0])
        return w

    # A name deliberately NOT already in the table, so this probes
    # extensibility rather than mutating a shipped entry.
    A.SCALAR_RECURRENCES["synthetic_root"] = A.ScalarRecurrence(
        _np.sqrt,
        lambda o, x: o.mul(0.5, o.reciprocal(
            o.apply("synthetic_root", x))),
        _jet_sqrt)
    try:
        r = quotient_check(3)
    finally:
        del A.SCALAR_RECURRENCES["synthetic_root"]
    assert r.status == "pass", f"new primitive broke Law 4: {r.detail}"


def test_no_dead_nested_apply_helper():
    """`_nd_apply` was dead AND wrong (it passed `dfn` as its own derivative,
    valid only for exp). Deleted; this pins that it stays deleted rather than
    being resurrected as a plausible-looking helper."""
    from tessera.autodiff import algebra as A

    assert not hasattr(A, "_nd_apply")


def test_tie_mass_target_follows_the_declared_group_count():
    """Review finding: the SPLIT branch hardcoded a target of 1.0, so a probe
    with two tie groups would sum to 2.0 and fail a correct rule. The count is
    now declared per probe."""
    from tessera.autodiff.law_inputs import KinkSpec
    from tessera.autodiff.laws import kink_check

    def two_row_probe():
        return (np.array([[3.0, 1.0, 3.0], [2.0, 2.0, 0.5]]),), {"axis": -1}

    spec = KinkSpec(
        two_row_probe,
        lambda p: (np.asarray(p[0]) == np.max(np.asarray(p[0]), axis=-1,
                                              keepdims=True),),
        "split",
        tie_groups=2,          # one tie per row
    )
    from tessera.autodiff.vjp import _VJPS
    from tessera.autodiff.jvp import _JVPS

    r = kink_check("amax", spec, _VJPS["amax"], _JVPS.get("amax"))
    assert r.status == "pass", r

    # Declaring the wrong count must fail — the check still has teeth.
    wrong = KinkSpec(two_row_probe, spec.kink_mask, "split", tie_groups=1)
    r2 = kink_check("amax", wrong, _VJPS["amax"], _JVPS.get("amax"))
    assert r2.status == "fail" and "mass not conserved" in r2.detail, r2


# ── AD-WEIL-1 + Law 6 ────────────────────────────────────────────────────────


def test_generic_substrate_reproduces_clifford_exactly():
    """W6.3's named open question, answered with evidence.

    The integrated plan asks whether one finite-multiplication-table
    substrate can carry both arbitrary commutative nilpotent Weil algebras
    and the Clifford algebras `ga/signature.py` implements, and W6.4 says to
    treat the reuse as a hypothesis to PROVE. Both families are monomial —
    a product of basis elements is one basis element times a scalar — so a
    single structure-constant table serves both. Cross-checked against `ga`'s
    own `geometric_product` as oracle on every allowed signature.
    """
    from tessera.autodiff.algebra import (TruncatedJet, clifford_table,
                                          weil_table)
    from tessera.ga import Multivector, geometric_product
    from tessera.ga.signature import V1_ALLOWED_SIGNATURES, Cl

    rng = np.random.default_rng(11)

    # Weil half: the generic table reproduces TruncatedJet's product exactly.
    for k in (1, 2, 3, 4, 5):
        A, W = weil_table(k), TruncatedJet(k)
        a = [np.asarray(v) for v in rng.standard_normal(k + 1)]
        b = [np.asarray(v) for v in rng.standard_normal(k + 1)]
        for lhs, rhs in zip(A.mul(a, b), W.mul(a, b)):
            assert float(lhs) == float(rhs)

    # Clifford half: exact against the GA oracle, on both allowed signatures.
    for (p, q, r) in sorted(V1_ALLOWED_SIGNATURES):
        alg, A = Cl(p, q, r), clifford_table(p, q, r)
        assert A.dim == alg.dim
        for _ in range(10):
            ca, cb = rng.standard_normal(alg.dim), rng.standard_normal(alg.dim)
            ref = np.asarray(
                geometric_product(Multivector(ca, alg),
                                  Multivector(cb, alg)).coefficients,
                dtype=np.float64)
            got = np.array([float(v) for v in
                            A.mul([np.asarray(v) for v in ca],
                                  [np.asarray(v) for v in cb])])
            np.testing.assert_allclose(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("name", sorted(__import__(
    "tessera.autodiff.algebra", fromlist=["x"]).SCALAR_RECURRENCES))
def test_ode_table_entry_matches_registered_jvp_and_nested_duals(name):
    """AD-WEIL-1's core claim, per primitive: ONE datum reproduces the
    existing first-order rule AND extends to every order.

    k=1 is checked against the *registered production JVP* (so the table
    cannot drift from what the compiler actually uses), and k=2..4 against
    the nested-dual tower (so the higher orders are not self-certified).
    """
    import math

    from tessera.autodiff.algebra import (SCALAR_RECURRENCES, TruncatedJet,
                                          _NestedScalarOps,
                                          nested_dual_derivative)
    from tessera.autodiff.jvp import _JVPS

    # Domain-aware evaluation points.
    x = {"log": 1.7, "log1p": 0.6, "sqrt": 2.3, "reciprocal": 1.9,
         "sigmoid": 0.4, "atan": 0.5}.get(name, 0.37)

    W = TruncatedJet(4)
    w = SCALAR_RECURRENCES[name].jet(
        W, W.lift(np.asarray(x), np.asarray(1.0)))

    jvp = _JVPS.get(name)
    if jvp is not None:
        _, dy = jvp((np.asarray([x]),), (np.asarray([1.0]),))
        np.testing.assert_allclose(float(np.ravel(dy)[0]), float(w[1]),
                                   rtol=1e-12,
                                   err_msg=f"{name}: k=1 differs from the "
                                           f"registered JVP")

    ops = _NestedScalarOps()
    for n in (2, 3, 4):
        ref = nested_dual_derivative(
            lambda t, o, _n=name: o._apply(_n, t), x, n)
        got = float(w[n]) * math.factorial(n)
        np.testing.assert_allclose(got, ref, rtol=1e-10,
                                   err_msg=f"{name}: order {n} differs from "
                                           f"the nested tower")


def test_conditioning_envelope_is_measured_not_assumed():
    """The plan's §3.8 gates IR investment on this measurement, and the
    measurement CORRECTS the plan: monomial jets were expected to go
    conditioning-limited "past ~order 10-15". They do not — the Taylor
    convention keeps the recurrences on scaled coefficients, so relative
    accuracy holds at ~1e-16 through k=30. The real limits are elsewhere:
    recovering `k! * w_k` overflows float64 near k=175, and a small radius of
    convergence inflates the coefficients themselves.
    """
    import math

    from tessera.autodiff.algebra import SCALAR_RECURRENCES, TruncatedJet

    for k in (4, 12, 20, 30):
        W = TruncatedJet(k)
        w = SCALAR_RECURRENCES["exp"].jet(
            W, W.lift(np.asarray(0.7), np.asarray(1.0)))
        got = float(w[k]) * math.factorial(k)
        rel = abs(got - math.exp(0.7)) / math.exp(0.7)
        assert rel < 1e-13, f"order {k} degraded to {rel:.2e}"

    # Accuracy holds even where the coefficients are huge (small radius).
    W = TruncatedJet(20)
    w = SCALAR_RECURRENCES["reciprocal"].jet(
        W, W.lift(np.asarray(0.15), np.asarray(1.0)))
    exact = [(-1) ** n / 0.15 ** (n + 1) for n in range(21)]
    assert abs(float(w[20])) > 1e16          # dynamic range, not error
    for n in range(21):
        assert abs(float(w[n]) - exact[n]) / abs(exact[n]) < 1e-13


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_law6_enclosure_contains_the_exact_jet(order):
    """Law 6a: an outward-rounded interval is a certificate — the exact jet
    must lie inside it. One escape is a failure, because a certificate that
    can miss the truth is worse than none."""
    from tessera.autodiff.laws import enclosure_check

    r = enclosure_check(order)
    assert r.status == "pass", r
    assert r.max_rel_residual < 1e-12, f"interval too loose: {r.detail}"


def test_law6_unbiasedness_checks_convergence_not_tolerance():
    """Law 6b: the randomized-jet Laplacian estimator is unbiased, asserted
    through the SHAPE of its convergence (error falls like 1/sqrt(N)). A
    biased estimator plateaus, which a single-N tolerance check would
    accept."""
    from tessera.autodiff.laws import unbiasedness_check

    r = unbiasedness_check()
    assert r.status == "pass", r


def test_law6_enclosure_catches_an_unsound_interval():
    """Mutation: an interval algebra that rounds INWARD produces intervals
    that can miss the exact value — exactly the failure a certificate must
    never have."""
    from tessera.autodiff import algebra as A
    from tessera.autodiff.laws import enclosure_check

    original = A.TaylorModel._widen
    try:
        # Round inward instead of outward.
        A.TaylorModel._widen = lambda self, lo, hi: (
            np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf))
        r = enclosure_check(3, trials=200)
    finally:
        A.TaylorModel._widen = original
    assert r.status == "fail", "unsound (inward-rounded) intervals passed"


def test_estimator_is_deterministic_under_a_fixed_seed():
    """Decision #18: Philox streams make the estimator replayable, which is
    what turns 'unbiased estimator' from folklore into a testable contract."""
    from tessera.autodiff.algebra import hutchinson_laplacian
    from tessera.rng import RNGKey

    def quad(lifted, W):
        sq = W.mul(lifted, lifted)
        return [sq[i] + 3.0 * lifted[i] for i in range(len(lifted))]

    a = hutchinson_laplacian(quad, np.zeros(5), RNGKey(7), 32)
    b = hutchinson_laplacian(quad, np.zeros(5), RNGKey(7), 32)
    assert a == b
