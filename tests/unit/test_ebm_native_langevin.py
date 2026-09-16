"""The traceable energy loop through the MLIR/LLVM backbone
(W4-PRODUCT-1 / AD-SOLVER-IFT-1 acceptance, 2026-09-16).

Acceptance from the GA/EBM review: native forward and gradient agree with
independent formulas, fixed-key samples agree with the declared policy, and
the complete loop executes without per-step host gradient transfers. Here the
gradient is the compiler's (paired autodiff on the Graph IR energy), the noise
is Philox drawn inside the compiled loop, and one JIT invocation runs all K
steps. Skips only when libtessera_jit lacks the EBM lane.

Since the nonlinear/manifold slice (N1, M1) the same lane carries three
energies — quadratic, Huber and softplus, each exercising one more adjoint on
the Graph IR path — and the sphere integrator, whose two singularities are
reported in a per-row status word rather than repaired silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tessera.ebm import native_langevin as nl

pytestmark = pytest.mark.skipif(
    not nl.has_native_langevin(),
    reason="libtessera_jit built without the EBM lane (TESSERA_BUILD_EBM_BACKEND=ON)",
)


def _pair(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32), rng.standard_normal(shape).astype(np.float32)


def test_energy_matches_the_independent_formula():
    y, x = _pair((5, 7), 1)
    np.testing.assert_allclose(nl.native_quadratic_energy(y, x), 0.5 * np.sum((x - y) ** 2, axis=1), rtol=1e-5, atol=1e-6)


def test_gradient_step_matches_the_independent_formula():
    """Temperature 0: y1 = y0 - eta * (y0 - x), with the gradient derived by
    the compiler from the energy, not written by hand."""
    y, x = _pair((6, 4), 2)
    out, key = nl.native_langevin_loop(y, x, [7, 9], eta=0.25, temperature=0.0, steps=1)
    np.testing.assert_allclose(out, y - 0.25 * (y - x), rtol=1e-6, atol=1e-6)
    assert list(key) == [7, 10]


def test_descent_converges_to_the_context():
    y, x = _pair((3, 8), 3)
    out, _ = nl.native_langevin_loop(y, x, [1, 1], eta=0.5, temperature=0.0, steps=40)
    np.testing.assert_allclose(out, x, atol=1e-5)


@pytest.mark.parametrize("shape,steps", [((4, 8), 1), ((4, 8), 5), ((3, 5), 12)])
def test_fixed_key_samples_match_the_declared_policy(shape, steps):
    """The whole loop runs in one compiled call and its samples are bit-exact
    with the numpy mirror of the declared Philox / Box-Muller policy."""
    y, x = _pair(shape, 4 + steps)
    before = jb.invocation_count()
    out, key = nl.native_langevin_loop(y, x, [0x1234ABCD9876, 42], eta=0.1, temperature=0.7, steps=steps)
    assert jb.invocation_count() == before + 1  # one native call for all K steps
    expect, expect_key = nl.reference_langevin_loop(y, x, [0x1234ABCD9876, 42], eta=0.1, temperature=0.7, steps=steps)
    assert list(key) == list(expect_key)
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)
    assert not np.allclose(out, y - 0.1 * (y - x))  # noise actually entered


def test_different_keys_give_different_samples_and_same_key_repeats():
    y, x = _pair((4, 8), 9)
    a, _ = nl.native_langevin_loop(y, x, [1, 0], eta=0.1, temperature=1.0, steps=2)
    b, _ = nl.native_langevin_loop(y, x, [2, 0], eta=0.1, temperature=1.0, steps=2)
    c, _ = nl.native_langevin_loop(y, x, [1, 0], eta=0.1, temperature=1.0, steps=2)
    assert not np.allclose(a, b) and np.array_equal(a, c)


def test_runtime_launch_reports_the_native_cpu_row():
    from tessera import runtime as rt
    y, x = _pair((4, 8), 11)
    result = rt.launch(nl.package_ebm_langevin_cpu((4, 8), eta=0.1, temperature=0.3, steps=3), (y, x, [5, 6]))
    assert result["ok"] and result["execution_kind"] == "native_cpu"
    assert result["compiler_path"] == "cpu_ebm_langevin_llvm_jit"
    out, key = result["output"]
    expect, expect_key = nl.reference_langevin_loop(y, x, [5, 6], eta=0.1, temperature=0.3, steps=3)
    np.testing.assert_allclose(np.asarray(out), expect, rtol=1e-5, atol=1e-5)
    assert list(np.asarray(key)) == list(expect_key)


@pytest.mark.parametrize("kwargs", [dict(eta=0.0, temperature=1.0, steps=1), dict(eta=0.1, temperature=-1.0, steps=1),
                                    dict(eta=0.1, temperature=1.0, steps=0)])
def test_out_of_envelope_requests_are_refused(kwargs):
    y, x = _pair((2, 2), 0)
    with pytest.raises(ValueError):
        nl.native_langevin_loop(y, x, [0, 0], **kwargs)


# --- N1: nonlinear energies -------------------------------------------------

@pytest.mark.parametrize("energy", nl.ENERGIES)
def test_each_energy_matches_its_independent_formula(energy):
    y, x = _pair((5, 9), 21)
    np.testing.assert_allclose(nl.native_quadratic_energy(y, x, energy=energy),
                               nl.reference_energy(y, x, energy=energy), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("energy", nl.ENERGIES)
def test_each_energy_descends_on_its_own_compiler_derived_gradient(energy):
    """T = 0 is one gradient step: y - eta * dE/dy with the compiler's gradient,
    which must equal the independently written derivative of that energy."""
    y, x = _pair((4, 8), 22)
    got, _ = nl.native_langevin_loop(y, x, [1, 1], eta=0.25, temperature=0.0, steps=1, energy=energy)
    want = (y - np.float32(0.25) * nl.reference_gradient(y, x, energy=energy)).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    # The gradients genuinely differ between the energies, so this is not a
    # test that passes for any of them: Huber clips and softplus saturates.
    if energy != "quadratic":
        assert not np.allclose(nl.reference_gradient(y, x, energy=energy),
                               nl.reference_gradient(y, x, energy="quadratic"), atol=1e-3)


@pytest.mark.parametrize("energy", nl.ENERGIES)
@pytest.mark.parametrize("steps", [1, 5])
def test_each_energy_samples_match_the_declared_policy(energy, steps):
    y, x = _pair((4, 8), 23 + steps)
    got = nl.native_langevin_loop(y, x, [0x99AA1234, 7], eta=0.1, temperature=0.6, steps=steps, energy=energy)
    want = nl.reference_langevin_loop(y, x, [0x99AA1234, 7], eta=0.1, temperature=0.6, steps=steps, energy=energy)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)
    assert list(got[1]) == list(want[1])


def test_the_huber_energy_is_the_clipped_gradient_outside_its_delta():
    """The kink is the point of admitting Huber: beyond delta the gradient is
    constant, which a quadratic energy never produces."""
    y = np.array([[5.0, 0.1, -7.0, 0.2]], np.float32)
    x = np.zeros((1, 4), np.float32)
    got, _ = nl.native_langevin_loop(y, x, [1, 1], eta=1.0, temperature=0.0, steps=1, energy="huber")
    np.testing.assert_allclose(got, y - np.array([[1.0, 0.1, -1.0, 0.2]], np.float32), rtol=1e-6, atol=1e-6)


def test_the_softplus_energy_saturates_rather_than_overflowing():
    """The stable adjoint (sigmoid) must stay finite where exp(x)/(1+exp(x))
    would overflow: a large positive difference gives gradient 1, a large
    negative one gives 0."""
    y = np.array([[80.0, -80.0, 0.0]], np.float32)
    x = np.zeros((1, 3), np.float32)
    got, _ = nl.native_langevin_loop(y, x, [1, 1], eta=1.0, temperature=0.0, steps=1, energy="softplus")
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, y - np.array([[1.0, 0.0, 0.5]], np.float32), rtol=1e-5, atol=1e-5)
    assert np.all(np.isfinite(nl.native_quadratic_energy(y, x, energy="softplus")))


# --- M1: the sphere integrator ----------------------------------------------

def _unit(shape, seed):
    v = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.mark.parametrize("energy", nl.ENERGIES)
def test_sphere_samples_match_the_reference_and_stay_on_the_sphere(energy):
    y, x = _unit((4, 8), 31), _pair((4, 8), 32)[0]
    got = nl.native_langevin_loop(y, x, [3, 4], eta=0.05, temperature=0.4, steps=4,
                                  manifold="sphere", energy=energy)
    want = nl.reference_langevin_loop(y, x, [3, 4], eta=0.05, temperature=0.4, steps=4,
                                      manifold="sphere", energy=energy)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)
    assert list(got[1]) == list(want[1]) and list(got[2]) == list(want[2]) == [0, 0, 0, 0]
    np.testing.assert_allclose(np.linalg.norm(np.asarray(got[0]), axis=1), 1.0, rtol=1e-5, atol=1e-5)
    # The tangent projection is what makes this different from the euclidean
    # step, so the two must not agree.
    flat = nl.reference_langevin_loop(y, x, [3, 4], eta=0.05, temperature=0.4, steps=4, energy=energy)
    assert not np.allclose(got[0], flat[0], atol=1e-3)


def test_sphere_at_zero_temperature_is_monotone_projected_descent():
    y, x = _unit((4, 8), 33), _pair((4, 8), 34)[0]
    got = nl.native_langevin_loop(y, x, [1, 1], eta=0.05, temperature=0.0, steps=12, manifold="sphere")
    before, after = nl.reference_energy(y, x), nl.reference_energy(np.asarray(got[0]), x)
    assert np.all(after < before) and list(got[2]) == [0, 0, 0, 0]
    np.testing.assert_allclose(np.linalg.norm(np.asarray(got[0]), axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_sphere_reports_the_entry_precondition_per_row_without_repairing_it():
    """Decision #21a: a state that is not on the sphere is reported, never
    silently renormalized, and only the offending row is flagged."""
    y, x = _unit((4, 8), 35), _pair((4, 8), 36)[0]
    bad = y.copy(); bad[2] *= 1.5
    got = nl.native_langevin_loop(bad, x, [1, 1], eta=0.05, temperature=0.0, steps=1, manifold="sphere")
    assert list(got[2]) == [0, 0, 1, 0]
    want = nl.reference_langevin_loop(bad, x, [1, 1], eta=0.05, temperature=0.0, steps=1, manifold="sphere")
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)


def test_sphere_status_accumulates_over_the_loop():
    y, x = _unit((3, 6), 37), _pair((3, 6), 38)[0]
    bad = y.copy(); bad[1] *= 3.0
    # The first step flags row 1; later steps see a retracted (unit) row, so the
    # accumulated word still carries the violation.
    got = nl.native_langevin_loop(bad, x, [1, 1], eta=0.05, temperature=0.3, steps=5, manifold="sphere")
    want = nl.reference_langevin_loop(bad, x, [1, 1], eta=0.05, temperature=0.3, steps=5, manifold="sphere")
    assert list(got[2]) == list(want[2]) and got[2][1] & 1


def test_runtime_launch_carries_the_energy_and_manifold():
    from tessera import runtime as rt
    y, x = _unit((4, 8), 39), _pair((4, 8), 40)[0]
    artifact = nl.package_ebm_langevin_cpu((4, 8), eta=0.05, temperature=0.2, steps=3,
                                           manifold="sphere", energy="softplus")
    result = rt.launch(artifact, (y, x, [2, 2]))
    assert result["ok"] and result["compiler_path"] == "cpu_ebm_langevin_llvm_jit"
    out = result["output"]
    want = nl.reference_langevin_loop(y, x, [2, 2], eta=0.05, temperature=0.2, steps=3,
                                      manifold="sphere", energy="softplus")
    assert len(out) == 3
    np.testing.assert_allclose(np.asarray(out[0]), want[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kwargs", [dict(manifold="bivector"), dict(manifold="hyperbolic"), dict(energy="l1")])
def test_unadmitted_energies_and_manifolds_are_refused(kwargs):
    y, x = _pair((4, 8), 41)
    with pytest.raises(ValueError):
        nl.native_langevin_loop(y, x, [1, 1], eta=0.1, temperature=0.1, steps=1, **kwargs)
