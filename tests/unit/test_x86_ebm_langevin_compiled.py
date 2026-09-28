"""EBM Langevin sampling lane on x86 AVX-512 (P7 sampling) — one Langevin step
that DRAWS its Gaussian noise on-device from counter-based Philox-4x32-10 (the
P6 generator): out = y − η·grad + noise_scale·z. The op is
``tessera.ebm.langevin_step_philox`` (operands y, grad, seed, counter), as the
frontend emits it for ``ops.ebm_langevin_step_philox`` (ODS triage WIRE slice
4, 2026-09-27). Reachable via
`compiler_path="x86_ebm_langevin_compiled"`. Validated byte-for-byte vs
tessera.ebm.langevin_step_philox. Skip-clean: libtessera_x86_elementwise.so not
built.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.ebm.energy import langevin_step_philox


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _seed(key) -> np.ndarray:
    """The Graph op's 1 x i64 seed for a (k0, k1) u32 key (low word first)."""
    return np.array([int(key[0]) | (int(key[1]) << 32)], np.int64)


def _art(rt, attrs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_ebm_langevin_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["y", "g", "seed", "ctr"], "output_name": "o",
        "ops": [{"op_name": "tessera.ebm.langevin_step_philox", "result": "o",
                 "operands": ["y", "g", "seed", "ctr"], "kwargs": attrs}]})


def _run(rt, y, grad, *, key, counter, **attrs):
    attrs.setdefault("temperature", 1.0)
    res = rt.launch(_art(rt, attrs),
                    (y, grad, _seed(key), np.asarray(counter, np.int64)))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_ebm_langevin_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(31)


def test_langevin_matches_reference():
    rt = _rt_or_skip()
    y = _RNG.standard_normal((4, 5)).astype(np.float32)
    grad = _RNG.standard_normal((4, 5)).astype(np.float32)
    key, counter = [0x1234, 0x5678], [7, 1, 2, 3]
    got = _run(rt, y, grad, eta=0.1, noise_scale=0.3, key=key, counter=counter)
    ref = langevin_step_philox(y, grad, eta=0.1, noise_scale=0.3,
                               key=np.array(key, np.uint32),
                               counter=np.array(counter, np.uint32))
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_langevin_zero_noise_is_gradient_descent():
    rt = _rt_or_skip()
    y = _RNG.standard_normal((6,)).astype(np.float32)
    grad = _RNG.standard_normal((6,)).astype(np.float32)
    got = _run(rt, y, grad, eta=0.25, noise_scale=0.0, key=[1, 2],
               counter=[0, 0, 0, 0])
    np.testing.assert_allclose(got, (y - 0.25 * grad).astype(np.float32),
                               rtol=1e-5, atol=1e-5)


def test_langevin_counter_changes_noise():
    rt = _rt_or_skip()
    y = np.zeros((8,), np.float32)
    grad = np.zeros((8,), np.float32)
    a = _run(rt, y, grad, eta=0.5, noise_scale=1.0, key=[9, 9], counter=[0, 0, 0, 0])
    b = _run(rt, y, grad, eta=0.5, noise_scale=1.0, key=[9, 9], counter=[100, 0, 0, 0])
    assert not np.allclose(a, b)        # different counter -> different draw


def test_langevin_default_noise_scale_is_sqrt_2_eta_t():
    """Absent noise_scale the op's amplitude is sqrt(2*eta*T) (TesseraOps.td)."""
    rt = _rt_or_skip()
    y = np.zeros((16,), np.float32)
    grad = np.zeros((16,), np.float32)
    key, counter = [3, 4], [5, 0, 0, 0]
    got = _run(rt, y, grad, eta=0.02, temperature=2.0, key=key, counter=counter)
    ref = langevin_step_philox(y, grad, eta=0.02, noise_scale=float(np.sqrt(0.08)),
                               key=np.array(key, np.uint32),
                               counter=np.array(counter, np.uint32))
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_langevin_refuses_missing_temperature_and_bad_counter():
    """eta / temperature are semantic (#21a); a counter word outside u32 is
    refused, not truncated."""
    rt = _rt_or_skip()
    y = np.zeros((4,), np.float32)
    art = _art(rt, {"eta": 0.1})
    res = rt.launch(art, (y, y, _seed([1, 2]), np.zeros(4, np.int64)))
    assert res["ok"] is False and "temperature" in str(res.get("reason"))
    art = _art(rt, {"eta": 0.1, "temperature": 1.0})
    res = rt.launch(art, (y, y, _seed([1, 2]), np.array([1 << 32, 0, 0, 0], np.int64)))
    assert res["ok"] is False and "counter" in str(res.get("reason"))


def test_frontend_emitted_op_is_what_the_executor_consumes():
    """Producer -> consumer: the op record @jit emits for
    ``ops.ebm_langevin_step_philox`` is launched unchanged on this lane."""
    import tessera as ts
    from tessera import ops

    @ts.jit
    def step(y, g, seed, ctr):
        return ops.ebm_langevin_step_philox(y, g, seed, ctr, eta=0.1,
                                            temperature=0.5, noise_scale=0.3)

    rt = _rt_or_skip()
    y = _RNG.standard_normal((3, 7)).astype(np.float32)
    grad = _RNG.standard_normal((3, 7)).astype(np.float32)
    seed, ctr = _seed([11, 22]), np.array([9, 1, 2, 3], np.int64)
    traced = step.runtime_artifact().metadata
    assert [o["op_name"] for o in traced["ops"]] == ["tessera.ebm.langevin_step_philox"]
    art = _art(rt, {})
    art.metadata["arg_names"] = list(traced["arg_names"])
    art.metadata["output_name"] = traced["output_name"]
    art.metadata["ops"] = traced["ops"]
    res = rt.launch(art, (y, grad, seed, ctr))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(step(y, grad, seed, ctr))  # the eager reference lane
    np.testing.assert_allclose(np.asarray(res["output"]), ref, rtol=1e-5, atol=1e-5)
