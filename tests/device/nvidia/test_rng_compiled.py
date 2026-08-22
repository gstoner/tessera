"""Exact-device proof for the typed NVIDIA Philox/key-counter package."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import rng_device as reference


def _runtime_or_skip():
    from tessera import runtime

    if runtime._load_nvidia_rng_runtime() is None:
        pytest.skip("libtessera_nvidia_rng.so not built")
    try:
        runtime._nvidia_philox_uniform(0, 0, 1)
    except RuntimeError as exc:
        pytest.skip(f"no usable NVIDIA GPU: {exc}")
    return runtime


def _artifact(runtime, op_name, kwargs, operands=()):
    names = [f"arg{index}" for index in range(len(operands))]
    return runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_rng_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": names,
        "output_name": "output",
        "ops": [{
            "op_name": op_name,
            "result": "output",
            "operands": names,
            "kwargs": kwargs,
        }],
    })


@pytest.mark.parametrize("count", (0, 1, 3, 4, 17, 1003))
def test_uniform_core_is_bit_exact_for_ragged_counts(count):
    runtime = _runtime_or_skip()
    seed = 0x123456789ABCDEF0
    counter = 0x100000013
    actual = runtime._nvidia_philox_uniform(seed, counter, count)
    expected = reference.philox_uniform(seed, counter, count)
    np.testing.assert_array_equal(actual, expected)


def test_uniform_range_and_explicit_key_counter_are_bit_exact():
    runtime = _runtime_or_skip()
    key = np.array([0x1234, 0x55], dtype=np.uint64)
    counter = np.array([19], dtype=np.uint64)
    artifact = _artifact(
        runtime, "tessera.rng_philox_uniform",
        {"shape": [17], "lo": -2.0, "hi": 3.0}, (key, counter))
    result = runtime.launch(artifact, (key, counter))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "nvidia_rng_compiled"
    expected = reference.uniform(int(key[0]) ^ int(key[1]), 17, -2.0, 3.0, 19)
    np.testing.assert_array_equal(np.asarray(result["output"]), expected)


def test_normal_transform_matches_reference_and_statistics():
    runtime = _runtime_or_skip()
    artifact = _artifact(
        runtime, "tessera.rng_normal",
        {"seed": 7, "counter_base": 11, "shape": [100],
         "mean": 2.0, "std": 0.5})
    actual = np.asarray(runtime.launch(artifact, ())["output"])
    expected = reference.normal(7, 100, 2.0, 0.5, 11)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=4e-6)

    large = np.asarray(runtime.launch(_artifact(
        runtime, "tessera.rng_normal", {"seed": 1, "shape": [200000]}),
        ())["output"])
    assert abs(float(large.mean())) < 1.0e-2
    assert abs(float(large.std()) - 1.0) < 1.0e-2


def test_dropout_replays_exact_uniform_mask_and_eval_is_identity():
    runtime = _runtime_or_skip()
    x = np.linspace(-3.0, 3.0, 20003, dtype=np.float32)
    kwargs = {"seed": 3, "counter_base": 29, "p": 0.3, "training": True}
    actual = np.asarray(runtime.launch(
        _artifact(runtime, "tessera.dropout", kwargs, (x,)), (x,))["output"])
    expected = x * reference.dropout_mask(3, x.size, 0.3, 29)
    np.testing.assert_array_equal(actual, expected)
    keep_rate = float(np.count_nonzero(actual) / np.count_nonzero(x))
    assert abs(keep_rate - 0.7) < 0.02

    evaluation = np.asarray(runtime.launch(_artifact(
        runtime, "tessera.dropout", {**kwargs, "training": False}, (x,)),
        (x,))["output"])
    np.testing.assert_array_equal(evaluation, x)


def test_same_key_counter_is_deterministic_and_counter_changes_stream():
    runtime = _runtime_or_skip()
    first = runtime._nvidia_philox_uniform(9, 41, 4099)
    second = runtime._nvidia_philox_uniform(9, 41, 4099)
    different = runtime._nvidia_philox_uniform(9, 42, 4099)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
