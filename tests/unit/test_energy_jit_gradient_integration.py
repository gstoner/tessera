"""Slice 1 (2026-05-22) — M6 EBM gradient integration on
``EnergyCompiledCallable``.

Before this slice: users had to manually call
``make_gradient_program(E)`` after decorating with ``@energy_jit``.
After this slice: ``E.grad_y(env)``, ``E.refine(y0, T, eta)``, and
``E.fused_report()`` are first-class methods on the device_verified_jit callable.

This file pins:

  1. The lazy ``gradient_program`` property builds once per instance
     and is cached across grad_y / refine calls.
  2. ``grad_y`` produces gradients that match a finite-difference
     reference for a representative whitelist op set.
  3. ``refine`` performs T deterministic descent steps using the same
     cached program (no rebuild between iterations).
  4. ``fused_report`` returns a dict with the documented shape that
     external tooling can serialise + diff.
  5. The integration round-trips: build → grad → refine → re-evaluate
     forward, with the energy strictly decreasing over the refinement.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import energy
from tessera.compiler.energy_jit import (
    energy_jit, EnergyCompiledCallable,
)


# ─────────────────────────────────────────────────────────────────────────────
# gradient_program: lazy + cached
# ─────────────────────────────────────────────────────────────────────────────


class TestGradientProgramLifecycle:
    def test_property_is_lazy(self) -> None:
        """Building the gradient program is deferred until first
        access — so a decoration that never needs gradients pays
        zero build cost."""
        @energy_jit(target="apple_gpu")
        def E(y, W):
            return energy.quadratic(y, W)

        # The device_verified_jit callable exists; the cached attr does not yet.
        assert isinstance(E, EnergyCompiledCallable)
        assert not hasattr(E, "_gradient_program") or \
            E._gradient_program is None

        # First access triggers the build.
        prog = E.gradient_program
        assert prog is not None
        assert hasattr(E, "_gradient_program")

    def test_property_is_cached(self) -> None:
        @energy_jit(target="apple_gpu")
        def E(y, W):
            return energy.quadratic(y, W)

        p1 = E.gradient_program
        p2 = E.gradient_program
        # Identity check — same instance, not a re-build.
        assert p1 is p2
        # build_call_count must stay at 1 after multiple accesses.
        assert p1.build_call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# grad_y vs finite-difference reference
# ─────────────────────────────────────────────────────────────────────────────


class TestGradY:
    def _finite_diff_grad_y(
        self, fn, env, y_name, eps=1e-4,
    ) -> np.ndarray:
        """Central-difference reference for ∂E/∂y."""
        y = np.asarray(env[y_name], dtype=np.float64).copy()
        params = {k: v for k, v in env.items() if k != y_name}
        out = np.zeros_like(y)
        flat = y.reshape(-1)
        flat_out = out.reshape(-1)
        for i in range(flat.size):
            saved = float(flat[i])
            flat[i] = saved + eps
            e_plus = float(np.asarray(fn(y, **params)).sum())
            flat[i] = saved - eps
            e_minus = float(np.asarray(fn(y, **params)).sum())
            flat[i] = saved
            flat_out[i] = (e_plus - e_minus) / (2.0 * eps)
        return out.astype(np.float32)

    def test_quadratic_matches_finite_difference(self) -> None:
        @energy_jit(target="apple_gpu")
        def E(y, W):
            return energy.quadratic(y, W)

        rng = np.random.default_rng(0)
        y = rng.standard_normal((4,)).astype(np.float32)
        W = rng.standard_normal((4, 4)).astype(np.float32)
        env = {"y": y, "W": W}

        g = E.grad_y(env)
        ref = self._finite_diff_grad_y(
            lambda y_, W: energy.quadratic(y_, W), env, "y",
        )
        np.testing.assert_allclose(g, ref, rtol=1e-2, atol=1e-2)

    def test_chain_matches_finite_difference(self) -> None:
        """Multi-op chain: norm_sq + relu + inner + reduce_sum."""
        @energy_jit(target="apple_gpu")
        def E(y):
            q = energy.norm_sq(y)
            r = energy.relu(y)
            s = energy.inner(r, y)
            return energy.reduce_sum(s)

        rng = np.random.default_rng(7)
        y = rng.standard_normal((6,)).astype(np.float32)
        env = {"y": y}

        g = E.grad_y(env)
        ref = self._finite_diff_grad_y(
            lambda y_: (
                energy.reduce_sum(
                    energy.inner(energy.relu(y_), y_),
                )
            ),
            env, "y",
        )
        np.testing.assert_allclose(g, ref, rtol=5e-2, atol=5e-2)


# ─────────────────────────────────────────────────────────────────────────────
# refine: T-step descent, build-once
# ─────────────────────────────────────────────────────────────────────────────


class TestRefine:
    def test_refine_decreases_energy(self) -> None:
        """The whole point of the refinement loop — the energy of
        the iterate goes DOWN as we descend."""
        @energy_jit(target="apple_gpu")
        def E(y, W):
            return energy.quadratic(y, W)

        rng = np.random.default_rng(42)
        # Build a positive-definite W so the quadratic has a minimum.
        A = rng.standard_normal((4, 4)).astype(np.float32)
        W = (A @ A.T + 1.0 * np.eye(4, dtype=np.float32))
        y0 = rng.standard_normal((4,)).astype(np.float32) * 2.0

        E_init = float(energy.quadratic(y0, W).sum())
        y_refined = E.refine(y0, T=10, eta=0.05, params={"W": W})
        E_final = float(energy.quadratic(y_refined, W).sum())

        assert E_final < E_init, (
            f"refine did not decrease energy: {E_init:.4f} → {E_final:.4f}"
        )

    def test_refine_reuses_gradient_program(self) -> None:
        """``refine`` must build the gradient program exactly once
        and reuse it across T steps."""
        @energy_jit(target="apple_gpu")
        def E(y, W):
            return energy.quadratic(y, W)

        rng = np.random.default_rng(0)
        W = (rng.standard_normal((3, 3)).astype(np.float32))
        W = W @ W.T + 0.5 * np.eye(3, dtype=np.float32)
        y0 = rng.standard_normal((3,)).astype(np.float32)

        E.refine(y0, T=5, eta=0.01, params={"W": W})
        # build_call_count from gradient_program should still be 1.
        assert E.gradient_program.build_call_count == 1

    def test_refine_is_deterministic(self) -> None:
        """Same y0 + same params + same T + same eta ⇒ identical
        trajectory.  This is the determinism contract the future
        fused MSL kernel will satisfy."""
        @energy_jit(target="apple_gpu")
        def E(y, W):
            return energy.quadratic(y, W)

        rng = np.random.default_rng(3)
        W = (rng.standard_normal((4, 4)).astype(np.float32))
        W = W @ W.T + 0.1 * np.eye(4, dtype=np.float32)
        y0 = rng.standard_normal((4,)).astype(np.float32)

        a = E.refine(y0, T=8, eta=0.02, params={"W": W})
        b = E.refine(y0, T=8, eta=0.02, params={"W": W})
        np.testing.assert_array_equal(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# fused_report
# ─────────────────────────────────────────────────────────────────────────────


class TestFusedReport:
    def test_quadratic_report_is_forward_and_grad(self) -> None:
        @energy_jit(target="apple_gpu")
        def E(y, W):
            return energy.quadratic(y, W)

        r = E.fused_report()
        # Schema check — all documented fields present.
        for key in (
            "program_id", "source", "target", "dtype",
            "arg_names", "return_ref",
            "forward_ops", "gradient_chain",
            "fusion_class", "errors",
        ):
            assert key in r, f"fused_report missing field {key!r}"

        # quadratic has a closed-form VJP, so the chain fuses.
        assert r["fusion_class"] == "forward_and_grad"
        assert r["errors"] == []
        assert r["arg_names"] == ["y", "W"]
        # Forward op set matches the function.
        op_names = [step["op_name"] for step in r["forward_ops"]]
        assert "energy_quadratic" in op_names
        # Every step in the gradient chain has a VJP.
        assert all(step["has_vjp"] for step in r["gradient_chain"])

    def test_report_is_json_serialisable(self) -> None:
        """Status dashboards and autotuners need to serialise the
        report; lock that it round-trips through json."""
        import json

        @energy_jit(target="apple_gpu")
        def E(y):
            return energy.reduce_sum(energy.norm_sq(y))

        r = E.fused_report()
        s = json.dumps(r)
        assert "energy_norm_sq" in s
        loaded = json.loads(s)
        assert loaded["fusion_class"] == "forward_and_grad"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: integration is alive — build → grad → refine → re-evaluate
# ─────────────────────────────────────────────────────────────────────────────


def test_end_to_end_decorate_grad_refine_reevaluate() -> None:
    """The whole point of Slice 1: a user writes
    ``@energy_jit`` and never touches ``make_gradient_program``;
    grad_y, refine, fused_report, and compile_report all work on
    the same instance."""
    @energy_jit(target="apple_gpu")
    def E(y, W):
        return energy.quadratic(y, W)

    rng = np.random.default_rng(11)
    A = rng.standard_normal((5, 5)).astype(np.float32)
    W = A @ A.T + np.eye(5, dtype=np.float32)
    y0 = rng.standard_normal((5,)).astype(np.float32) * 1.5
    env = {"y": y0, "W": W}

    # All four lane surfaces on the same callable.
    assert E(y0, W) is not None                      # forward works
    g = E.grad_y(env)                                 # gradient works
    assert g.shape == y0.shape
    y_refined = E.refine(y0, T=20, eta=0.05, params={"W": W})  # refinement works
    report = E.compile_report()                       # CompileReport works
    fused = E.fused_report()                          # fused report works

    # Energy actually decreased.
    E_init = float(energy.quadratic(y0, W).sum())
    E_final = float(energy.quadratic(y_refined, W).sum())
    assert E_final < E_init

    # Gradient program built exactly once across all the calls above.
    assert E.gradient_program.build_call_count == 1

    # Reports are stable: report_hash() is deterministic across calls.
    assert report.report_hash() == E.compile_report().report_hash()
    assert fused["fusion_class"] == "forward_and_grad"
