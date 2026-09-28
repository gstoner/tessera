"""ODS triage WIRE slice 4 (GOV-ODS-CONSUMER-1): ``tessera.ebm.langevin_step_philox``
has a producer and a consumer.

* Producer: ``ops.ebm_langevin_step_philox(y, grad, seed, counter, *, eta,
  temperature, noise_scale=None)`` (op catalog) -- ``@jit`` emits the 4-operand
  Graph op.
* Consumer: the x86 / ROCm compiled executors (``runtime._EBM_LANGEVIN_OPS``),
  which now accept exactly this op and read seed / counter from its operands.

This file is host-free: the executor's operand/attribute mapping is exercised
through ``_execute_ebm_langevin`` with a recording kernel stand-in. The device
numerics live in ``test_{x86,rocm}_ebm_langevin_compiled.py`` (Zen 5 / gfx1151).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import ops
from tessera import runtime as rt
from tessera.ebm.energy import langevin_step_philox


@ts.jit
def _step(y, g, seed, ctr):
    return ops.ebm_langevin_step_philox(y, g, seed, ctr, eta=0.1, temperature=0.5,
                                        noise_scale=0.3)


@ts.jit
def _step_default_noise(y, g, seed, ctr):
    return ops.ebm_langevin_step_philox(y, g, seed, ctr, eta=0.02, temperature=2.0)


def _record():
    seen: dict = {}

    def fake(y, grad, eta, ns, k0, k1, c0, c1, c2, c3, np_mod):
        seen.update(eta=eta, ns=ns, key=(k0, k1), ctr=(c0, c1, c2, c3))
        return np.asarray(y) - eta * np.asarray(grad)

    return seen, fake


def _launch_traced(fn, args):
    md = fn.runtime_artifact().metadata
    art = rt.RuntimeArtifact(metadata={"arg_names": md["arg_names"],
                                       "output_name": md["output_name"],
                                       "ops": md["ops"]})
    seen, fake = _record()
    out = rt._execute_ebm_langevin(art, args, fake, "test_lane")
    return seen, out


def test_frontend_emits_the_four_operand_philox_op():
    text = _step.ir_text()
    assert "tessera.ebm.langevin_step_philox" in text
    assert 'tessera.effect_kind = "random"' in text
    assert 'tessera.stochastic_identity = "seed_counter"' in text
    (op,) = _step.runtime_artifact().metadata["ops"]
    assert op["op_name"] == "tessera.ebm.langevin_step_philox"
    assert op["operands"] == ["y", "g", "seed", "ctr"]
    assert op["kwargs"] == {"eta": 0.1, "temperature": 0.5, "noise_scale": 0.3}


def test_executor_consumes_the_traced_op_and_maps_seed_and_counter():
    y = np.zeros((2, 3), np.float32)
    seed = np.array([(0xCAFE << 32) | 0xBEEF], np.int64)
    ctr = np.array([7, 1, 2, 3], np.int64)
    seen, _ = _launch_traced(_step, (y, y, seed, ctr))
    assert seen["key"] == (0xBEEF, 0xCAFE)          # low word first
    assert seen["ctr"] == (7, 1, 2, 3)
    assert seen["eta"] == pytest.approx(0.1) and seen["ns"] == pytest.approx(0.3)


def test_absent_noise_scale_is_sqrt_two_eta_temperature():
    y = np.zeros((4,), np.float32)
    seen, _ = _launch_traced(_step_default_noise,
                             (y, y, np.array([5], np.int64), np.zeros(4, np.int64)))
    assert seen["ns"] == pytest.approx(np.sqrt(2 * 0.02 * 2.0))


def test_negative_seed_splits_as_twos_complement():
    y = np.zeros((4,), np.float32)
    seen, _ = _launch_traced(_step, (y, y, np.array([-1], np.int64), np.zeros(4, np.int64)))
    assert seen["key"] == (0xFFFFFFFF, 0xFFFFFFFF)


def _art(op_name, operands, kwargs):
    return rt.RuntimeArtifact(metadata={
        "arg_names": ["y", "g", "seed", "ctr"][:len(operands)], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": operands,
                 "kwargs": kwargs}]})


def test_the_host_noise_op_is_no_longer_accepted():
    """The executors ran Philox semantics under `tessera.ebm.langevin_step`,
    whose third operand is HOST noise; that name is now refused (#31)."""
    y = np.zeros((4,), np.float32)
    _, fake = _record()
    with pytest.raises(ValueError, match="langevin_step_philox"):
        rt._execute_ebm_langevin(
            _art("tessera.ebm.langevin_step", ["y", "g", "seed"],
                 {"eta": 0.1, "noise_scale": 0.3}),
            (y, y, y), fake, "test_lane")


@pytest.mark.parametrize("kwargs,counter,match", [
    ({"eta": 0.1}, [0, 0, 0, 0], "temperature"),
    ({"temperature": 1.0}, [0, 0, 0, 0], "eta"),
    ({"eta": 0.1, "temperature": 1.0}, [1 << 32, 0, 0, 0], "counter"),
    ({"eta": 0.1, "temperature": 1.0}, [-1, 0, 0, 0], "counter"),
    ({"eta": 0.1, "temperature": 1.0, "noise_scale": -0.5}, [0, 0, 0, 0], "noise_scale"),
])
def test_executor_refuses_rather_than_defaults(kwargs, counter, match):
    y = np.zeros((4,), np.float32)
    _, fake = _record()
    with pytest.raises(ValueError, match=match):
        rt._execute_ebm_langevin(
            _art("tessera.ebm.langevin_step_philox", ["y", "g", "seed", "ctr"], kwargs),
            (y, y, np.array([1], np.int64), np.array(counter, np.int64)), fake, "test_lane")


def test_eager_reference_matches_the_numpy_philox_reference():
    rng = np.random.default_rng(4)
    y = rng.standard_normal((3, 5)).astype(np.float32)
    g = rng.standard_normal((3, 5)).astype(np.float32)
    seed = np.array([(22 << 32) | 11], np.int64)
    ctr = np.array([9, 1, 2, 3], np.int64)
    got = ops.ebm_langevin_step_philox(y, g, seed, ctr, eta=0.1, temperature=0.5,
                                       noise_scale=0.3)
    ref = langevin_step_philox(y, g, eta=0.1, noise_scale=0.3,
                               key=np.array([11, 22], np.uint32),
                               counter=np.array([9, 1, 2, 3], np.uint32))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
