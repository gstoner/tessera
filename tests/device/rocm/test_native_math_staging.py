"""Exact-device ownership and changed-input proof for native math staging."""
import ctypes as C

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.device.rocm.test_native_math_package_jit import FUNCTIONS, target as _target_fixture
from tests.device.rocm.test_native_math_widening import FUNCTIONS as WIDENING
from benchmarks.rocm.benchmark_native_math_schedule import expected


@pytest.fixture
def target():
    return _target_fixture.__wrapped__()


def stats(lib):
    values = [C.c_uint64() for _ in range(4)]
    assert lib.tessera_rocm_movement_stats(*(C.byref(v) for v in values)) == 0
    return tuple(v.value for v in values)


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("kind", ["sqrt", "add", "cumsum"])
def test_staging_reuses_capacity_and_updates_every_input(target, storage, kind, monkeypatch):
    monkeypatch.setenv("TESSERA_ROCM_NATIVE_MATH", "1")
    monkeypatch.setenv("TESSERA_ROCM_MATH_STAGING_REUSE", "1")
    lib = rt._load_rocm_native_movement_runtime()
    assert lib is not None and hasattr(lib, "tessera_rocm_math_launch"), "matching native library required"
    assert lib.tessera_rocm_movement_clear_current() == 0
    dtype = {"f32": np.float32, "f16": np.float16,
             "bf16": pytest.importorskip("ml_dtypes").bfloat16}[storage]
    rng = np.random.default_rng(606)
    values = [rng.uniform(.25, 1.5, (3, 17)).astype(dtype)]
    if kind == "add":
        values.append(rng.uniform(.5, 1.5, (3, 17)).astype(dtype))
    fn = ts.jit(target=target)((FUNCTIONS if storage == "f32" else WIDENING)[kind])
    before = stats(lib)
    result = fn(*values)
    assert fn.execution_kind == "native_gpu"
    np.testing.assert_allclose(result, expected(kind, [v.astype(np.float32) for v in values]), rtol=2e-5, atol=2e-5)
    warm = stats(lib)
    assert warm[0] - before[0] == len(values) + 1
    for _ in range(7):
        for index, value in enumerate(values):
            value[:] = rng.uniform(.25 + index * .1, 1.5, value.shape).astype(dtype)
        np.testing.assert_allclose(fn(*values), expected(kind, [v.astype(np.float32) for v in values]), rtol=2e-5, atol=2e-5)
    after = stats(lib)
    assert after[0] == warm[0] and after[1] == warm[1]
    assert after[2] - warm[2] == 7 * (len(values) + 1)
    assert after[3] - warm[3] == 7
    monkeypatch.setenv("TESSERA_ROCM_MATH_STAGING_REUSE", "0")
    np.testing.assert_allclose(fn(*values), expected(kind, [v.astype(np.float32) for v in values]), rtol=2e-5, atol=2e-5)
    control = stats(lib)
    assert control[0] - after[0] == len(values) + 1
    assert control[1] - after[1] == 2 * (len(values) + 1)
    assert lib.tessera_rocm_movement_clear_current() == 0
