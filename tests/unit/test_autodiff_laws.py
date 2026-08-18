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
_KNOWN_NONE_CONVENTION = {
    ("istft", "hop"),
    ("istft", "n_fft"),
    ("masked_fill", "value"),
    ("stft", "hop"),
    ("stft", "n_fft"),
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

# _OPEN_SWALLOW_FINDINGS: still awaiting triage/fix. stft/istft/
# spectral_conv ride the spectral-family review (AD-TSOL-STFT-BWD-1 is the
# owning queue item for that family's backward contracts); the quantize
# family rides the quantization-semantics review (STE primal conventions).
# Do not add entries — fix the rule, or move to _BENIGN_SWALLOWS with a
# recorded reason.
_OPEN_SWALLOW_FINDINGS = {
    ("istft", "jvp", "axis"),
    ("istft", "jvp", "center"),
    ("istft", "jvp", "hop_length"),
    ("istft", "jvp", "length"),
    ("istft", "jvp", "norm"),
    ("istft", "jvp", "normalization"),
    ("istft", "jvp", "onesided"),
    ("quantize_fp4", "jvp", "scale"),
    ("quantize_fp6", "jvp", "scale"),
    ("quantize_int4", "jvp", "symmetric"),
    ("quantize_int8", "jvp", "symmetric"),
    ("quantize_nvfp4", "jvp", "scale"),
    ("spectral_conv", "jvp", "axis"),
    ("spectral_conv", "jvp", "norm"),
    ("spectral_conv", "jvp", "normalization"),
    ("stft", "jvp", "axis"),
    ("stft", "jvp", "center"),
    ("stft", "jvp", "hop_length"),
    ("stft", "jvp", "norm"),
    ("stft", "jvp", "normalization"),
    ("stft", "jvp", "onesided"),
    ("stft", "jvp", "pad_mode"),
}

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
