"""Exact-host AVX-512 execution proof for EGGROLL W2 rank-1 correction."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera import rng
from tessera.stdlib.es import low_rank_correction


@pytest.mark.compiler_avx512
@pytest.mark.parametrize(
    "population,rows,in_dim,out_dim",
    ((4, 3, 8, 6), (2, 2, 513, 277)),
)
def test_native_es_low_rank_correction_on_zen5(
    population: int, rows: int, in_dim: int, out_dim: int
) -> None:
    from tessera import runtime as rt

    library = rt._load_x86_elementwise()
    if library is None or not hasattr(
        library, "tessera_x86_avx512_es_low_rank_correction_f32"
    ):
        pytest.skip("AVX-512 EGGROLL image is unavailable")
    function = library.tessera_x86_avx512_es_low_rank_correction_f32
    generator = np.random.default_rng(7)
    values = generator.standard_normal(
        (population, rows, in_dim), dtype=np.float32
    )
    members = np.arange(8, 8 + population, dtype=np.int64)
    key = rng.RNGKey(
        seed_high=0x0123456789ABCDEF,
        seed_low=0x1234567890ABCDEF,
    )
    key_storage = np.array([key.seed_high, key.seed_low], dtype=np.uint64).view(
        np.int64
    )
    output = np.empty((population, rows, out_dim), dtype=np.float32)
    f32p = ctypes.POINTER(ctypes.c_float)
    i64p = ctypes.POINTER(ctypes.c_int64)
    status = function(
        values.ctypes.data_as(f32p),
        members.ctypes.data_as(i64p),
        key_storage.ctypes.data_as(i64p),
        output.ctypes.data_as(f32p),
        population,
        rows,
        in_dim,
        out_dim,
        3,
        ctypes.c_float(0.02),
        1,
    )
    assert status == 0
    expected = low_rank_correction(
        values,
        members,
        key,
        out_dim=out_dim,
        rank=1,
        sigma=0.02,
        epoch=3,
        antithetic=True,
    )
    np.testing.assert_allclose(output, expected, rtol=4e-5, atol=3e-6)


@pytest.mark.compiler_avx512
def test_runtime_launch_consumes_the_x86_es_package() -> None:
    from tessera import runtime as rt

    if not rt._x86_elementwise_available():
        pytest.skip("AVX-512 EGGROLL image is unavailable")
    key = rng.RNGKey(seed_high=17, seed_low=23)
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 17.0
    members = np.array([4, 5], dtype=np.int64)
    key_storage = np.array([key.seed_high, key.seed_low], dtype=np.int64)
    artifact = rt.RuntimeArtifact(
        metadata={
            "target": "x86",
            "compiler_path": "x86_es_low_rank_compiled",
            "executable": True,
            "execution_kind": "native_cpu",
            "arg_names": ["x", "member_ids", "key"],
            "ops": [
                {
                    "op_name": "tessera.es_low_rank_correction",
                    "operands": ["x", "member_ids", "key"],
                    "kwargs": {
                        "out_dim": 5,
                        "rank": 1,
                        "epoch": 2,
                        "sigma": 0.03,
                        "antithetic": True,
                    },
                }
            ],
        }
    )
    result = rt.launch(artifact, (values, members, key_storage))
    assert result["ok"] is True, result.get("reason")
    expected = low_rank_correction(
        values, members, key, out_dim=5, rank=1, sigma=0.03,
        epoch=2, antithetic=True,
    )
    np.testing.assert_allclose(result["output"], expected, rtol=4e-5, atol=3e-6)
