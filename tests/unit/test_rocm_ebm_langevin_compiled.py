"""EBM Langevin sampling lane on AMD ROCm gfx1151 (P7 sampling) — one Langevin
step that DRAWS its Gaussian noise on-device from counter-based Philox-4x32-10
(the P6 generator) via the COMPILER-GENERATED kernel
(generate-rocm-ebm-langevin-kernel): out = y − η·grad + noise_scale·z.
Reachable via `compiler_path="rocm_ebm_langevin_compiled"`. Validated vs
tessera.ebm.langevin_step_philox on gfx1151 (the device f32 Box-Muller agrees
with the reference to f32 tolerance). Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.ebm.energy import langevin_step_philox


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_ebm_langevin_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["y", "g"], "output_name": "o",
        "ops": [{"op_name": "tessera.ebm.langevin_step", "result": "o",
                 "operands": ["y", "g"], "kwargs": kwargs}]})


def _run(rt, y, grad, **kwargs):
    res = rt.launch(_art(rt, kwargs), (y, grad))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_ebm_langevin_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(31)


def test_langevin_matches_reference():
    rt = _rocm_or_skip()
    y = _RNG.standard_normal((4, 5)).astype(np.float32)
    grad = _RNG.standard_normal((4, 5)).astype(np.float32)
    key, counter = [0x1234, 0x5678], [7, 1, 2, 3]
    got = _run(rt, y, grad, eta=0.1, noise_scale=0.3, key=key, counter=counter)
    ref = langevin_step_philox(y, grad, eta=0.1, noise_scale=0.3,
                               key=np.array(key, np.uint32),
                               counter=np.array(counter, np.uint32))
    # device f32 Box-Muller vs the reference's f64 — agree to f32 tolerance.
    np.testing.assert_allclose(got, np.asarray(ref), rtol=2e-3, atol=2e-3)


def test_langevin_zero_noise_is_gradient_descent():
    rt = _rocm_or_skip()
    y = _RNG.standard_normal((6,)).astype(np.float32)
    grad = _RNG.standard_normal((6,)).astype(np.float32)
    got = _run(rt, y, grad, eta=0.25, noise_scale=0.0, key=[1, 2],
               counter=[0, 0, 0, 0])
    np.testing.assert_allclose(got, (y - 0.25 * grad).astype(np.float32),
                               rtol=1e-5, atol=1e-5)
