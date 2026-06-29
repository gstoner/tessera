"""EBM energy / step-compute lane on x86 AVX-512 (P7 follow-up) — the
tensor-clean tessera.ebm_* graph ops: energy_quadratic / inner_step /
refinement / self_verify. All compose on the device binary + reduce kernels (no
new kernel): the diff/square/reduce on AVX-512, the scalar scale / argmin gather
on the host. Reachable via `compiler_path="x86_ebm_compute_compiled"`. Validated
vs tessera.ebm.energy. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.ebm.energy import (
    energy_quadratic as _energy_quadratic,
    inner_step as _inner_step,
    refinement as _refinement,
    self_verify as _self_verify,
)


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, n_operands, kwargs):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_ebm_compute_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, *arrs, **kwargs):
    res = rt.launch(_art(rt, op, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_ebm_compute_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(29)


def _rn(*shape):
    return _RNG.standard_normal(shape).astype(np.float32)


def test_energy_quadratic():
    rt = _rt_or_skip()
    x, y = _rn(4, 6), _rn(4, 6)
    got = _run(rt, "tessera.ebm_energy_quadratic", x, y)
    ref = _energy_quadratic(x, y)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_energy_quadratic_3d():
    rt = _rt_or_skip()
    x, y = _rn(3, 5, 2), _rn(3, 5, 2)        # reduces over axes 1..
    got = _run(rt, "tessera.ebm_energy_quadratic", x, y)
    ref = _energy_quadratic(x, y)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_inner_step():
    rt = _rt_or_skip()
    y, grad = _rn(5, 4), _rn(5, 4)
    eta = 0.3
    got = _run(rt, "tessera.ebm_inner_step", y, grad, eta=eta)
    ref = _inner_step(y, grad, eta)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_refinement():
    rt = _rt_or_skip()
    y0, grad = _rn(6, 3), _rn(6, 3)
    got = _run(rt, "tessera.ebm_refinement", y0, grad, eta=0.1, T=5)
    ref = _refinement(y0, grad, eta=0.1, T=5)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_refinement_zero_steps_is_identity():
    rt = _rt_or_skip()
    y0, grad = _rn(2, 2), _rn(2, 2)
    np.testing.assert_array_equal(
        _run(rt, "tessera.ebm_refinement", y0, grad, eta=0.5, T=0), y0)


def test_self_verify_hard_argmin():
    rt = _rt_or_skip()
    energies = _rn(4, 3)
    candidates = _rn(4, 3, 5)
    got = _run(rt, "tessera.ebm_self_verify", energies, candidates)
    ref = _self_verify(energies, candidates)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_self_verify_soft_min():
    rt = _rt_or_skip()
    energies = _rn(4, 3)
    candidates = _rn(4, 3, 5)
    got = _run(rt, "tessera.ebm_self_verify", energies, candidates, beta=2.0)
    ref = _self_verify(energies, candidates, beta=2.0)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)
