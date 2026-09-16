"""The traceable quadratic energy loop through the MLIR/LLVM backbone
(W4-PRODUCT-1 / AD-SOLVER-IFT-1 acceptance, 2026-09-16).

Acceptance from the GA/EBM review: native forward and gradient agree with
independent formulas, fixed-key samples agree with the declared policy, and
the complete loop executes without per-step host gradient transfers. Here the
gradient is the compiler's (paired autodiff on the Graph IR energy), the noise
is Philox drawn inside the compiled loop, and one JIT invocation runs all K
steps. Skips only when libtessera_jit lacks the EBM lane.
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
