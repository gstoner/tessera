"""AR-DECODE-PROGRAM-1 S1b: row argmax/argmin on sm_120.

The kernel is compiled Graph -> Schedule -> Tile -> NVIDIA by the native
passes (``NativeArgReduce.h``) and checked bitwise against NumPy, whose
first-occurrence tie-break and first-NaN rule the contract declares, and
against the greedy tokens of the S0 decode oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_cuda_host_ready

PIPELINE = "tessera-nvidia-pipeline-sm120"
NAMES = ("x", "indices")


@pytest.fixture(autouse=True)
def _sm120():
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    from tessera import runtime as rt
    assert rt._nvidia_device_name() == "sm_120"


_PACKAGES: dict = {}


def _run(kind, x, *, axis=-1, keepdims=False):
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native as native
    key = (kind, x.shape, axis, keepdims)
    if key not in _PACKAGES:
        _PACKAGES[key] = native.package_argreduce(
            kind, NAMES, x.shape, axis=axis, keepdims=keepdims, pipeline_name=PIPELINE)
    package = _PACKAGES[key]
    expected = getattr(np, kind)(x, axis=axis, keepdims=keepdims)
    out = np.full(np.shape(expected), -7, np.int32)
    artifact = rt.RuntimeArtifact(
        tile_ir=package.tile_ir, target_ir=package.target_ir, metadata={"target": "nvidia_sm120"},
        native_image=package.image, launch_descriptor=package.descriptor)
    columns = x.size if axis is None else x.shape[-1]
    bindings = {"x": np.ascontiguousarray(x, np.float32), "indices": out,
                "Rows": x.size // columns, "Cols": columns}
    result = rt.launch(rt.RuntimeArtifact.from_json(artifact.to_json()), bindings)
    assert result.get("ok") is True, result
    return out, np.asarray(expected, np.int32)


def test_greedy_tokens_match_the_decode_oracle():
    """argmax over the oracle's per-step fp32 logits reproduces its greedy tokens."""
    from tessera.models import ar_decode as A
    trace = A.reference_decode(A.AR_DECODE_TINY, A.tiny_fixture_weights(),
                               A.tiny_fixture_contract(), A.TINY_PROMPT, 32)
    logits = trace.logits.astype(np.float32)
    got, want = _run("argmax", logits)
    np.testing.assert_array_equal(got, want)
    assert got.tolist() == list(trace.tokens)


@pytest.mark.parametrize("kind", ["argmax", "argmin"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 31), (3, 128), (5, 129), (2, 1000), (1, 151936), (7, 3, 33)])
def test_random_rows_match_numpy(kind, shape):
    x = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)
    got, want = _run(kind, x)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("kind", ["argmax", "argmin"])
def test_ties_take_the_first_occurrence(kind):
    # Every row holds its extreme value at several columns spread across
    # threads and warps, so the combine order is exercised, not only the scan.
    x = np.zeros((4, 1000), np.float32)
    extreme = 5.0 if kind == "argmax" else -5.0
    for row, cols in enumerate([(999, 3, 500), (128, 0, 129), (640, 641, 33), (31, 32, 997)]):
        x[row, list(cols)] = extreme
    got, want = _run(kind, x)
    np.testing.assert_array_equal(got, want)
    assert got.tolist() == [3, 0, 33, 31]
    # Signed zeros compare equal: an all-zero row with -0.0 first returns 0.
    z = np.zeros((1, 300), np.float32)
    z[0, 0] = -0.0
    z[0, 200] = 0.0
    assert _run(kind, z)[0].tolist() == [0]


@pytest.mark.parametrize("kind", ["argmax", "argmin"])
def test_first_nan_wins_and_infinities_are_values(kind):
    x = np.random.default_rng(7).standard_normal((4, 700)).astype(np.float32)
    x[0, [650, 129]] = np.nan                       # first NaN, across warps
    x[1, :] = -np.inf if kind == "argmax" else np.inf  # all one infinity: index 0
    x[2, 444] = np.inf if kind == "argmax" else -np.inf
    x[3, :] = np.nan                               # all NaN: index 0
    got, want = _run(kind, x)
    np.testing.assert_array_equal(got, want)
    assert got.tolist() == [129, 0, 444, 0]


@pytest.mark.parametrize("keepdims", [False, True])
def test_flattened_and_keepdims_forms(keepdims):
    x = np.random.default_rng(11).standard_normal((6, 5, 9)).astype(np.float32)
    got, want = _run("argmax", x, axis=None, keepdims=keepdims)
    np.testing.assert_array_equal(got, want)
    got, want = _run("argmin", x, axis=-1, keepdims=keepdims)
    np.testing.assert_array_equal(got, want)


def test_mismatched_buffer_is_refused():
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native as native
    package = native.package_argreduce("argmax", NAMES, (2, 64), pipeline_name=PIPELINE)
    artifact = rt.RuntimeArtifact(
        tile_ir=package.tile_ir, target_ir=package.target_ir, metadata={"target": "nvidia_sm120"},
        native_image=package.image, launch_descriptor=package.descriptor)
    bindings = {"x": np.zeros((2, 63), np.float32), "indices": np.zeros(2, np.int32),
                "Rows": 2, "Cols": 64}
    try:
        result = rt.launch(artifact, bindings)
    except (RuntimeError, ValueError):
        return
    assert result.get("ok") is not True, result
    assert not bindings["indices"].any()
