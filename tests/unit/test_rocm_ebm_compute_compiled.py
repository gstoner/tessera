"""EBM energy / step-compute lane on AMD ROCm gfx1151 (P7 follow-up) — the
tensor-clean tessera.ebm_* graph ops: energy_quadratic / inner_step /
refinement / self_verify. All compose on the gfx1151 binary + reduce kernels
(no new kernel): the diff/square/reduce on-device, the scalar scale / argmin
gather on the host. Reachable via `compiler_path="rocm_ebm_compute_compiled"`.
Validated vs tessera.ebm.energy on gfx1151. Skip-clean: tessera-opt not built /
no GPU.
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


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, n_operands, kwargs):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_ebm_compute_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, *arrs, **kwargs):
    res = rt.launch(_art(rt, op, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_ebm_compute_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(29)


def _rn(*shape):
    return _RNG.standard_normal(shape).astype(np.float32)


def test_energy_quadratic():
    rt = _rocm_or_skip()
    x, y = _rn(4, 6), _rn(4, 6)
    np.testing.assert_allclose(
        _run(rt, "tessera.ebm_energy_quadratic", x, y),
        np.asarray(_energy_quadratic(x, y)), rtol=1e-4, atol=1e-4)


def test_inner_step_and_refinement():
    rt = _rocm_or_skip()
    y, grad = _rn(5, 4), _rn(5, 4)
    np.testing.assert_allclose(
        _run(rt, "tessera.ebm_inner_step", y, grad, eta=0.3),
        np.asarray(_inner_step(y, grad, 0.3)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        _run(rt, "tessera.ebm_refinement", y, grad, eta=0.1, T=5),
        np.asarray(_refinement(y, grad, eta=0.1, T=5)), rtol=1e-5, atol=1e-5)


def test_self_verify_hard_and_soft():
    rt = _rocm_or_skip()
    energies = _rn(4, 3)
    candidates = _rn(4, 3, 5)
    np.testing.assert_allclose(
        _run(rt, "tessera.ebm_self_verify", energies, candidates),
        np.asarray(_self_verify(energies, candidates)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        _run(rt, "tessera.ebm_self_verify", energies, candidates, beta=2.0),
        np.asarray(_self_verify(energies, candidates, beta=2.0)),
        rtol=1e-4, atol=1e-4)
