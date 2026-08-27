"""Numerical gates for the x86 PR #568 physical family consumers."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from tessera.game import (
    subset_mobius,
    subset_zeta,
    superset_mobius,
    superset_zeta,
)
from tessera.solvers_ops import tridiagonal_solve
from tessera.compiler.backend_manifest import manifest_for
from tessera.compiler.x86_native import X86_AVX512_ARCHITECTURE, host_supports_architecture


def _library() -> ctypes.CDLL:
    if not host_supports_architecture(X86_AVX512_ARCHITECTURE):
        pytest.skip("x86 physical reference-tier image requires AVX-512 host support")
    candidates = (
        Path(__file__).parents[2] / "build/src/compiler/codegen/tessera_x86_backend/libtessera_x86_elementwise.so",
        Path(__file__).parents[2] / "build-x86/src/compiler/codegen/tessera_x86_backend/libtessera_x86_elementwise.so",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        pytest.skip("x86 physical reference-tier image is not built")
    return ctypes.CDLL(str(path))


_F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")


@pytest.mark.parametrize("batch", [1, 7, 8, 17])
@pytest.mark.parametrize("n", [1, 8, 31])
def test_batch_vector_thomas_matches_fp64_reference(batch: int, n: int) -> None:
    lib = _library()
    kernel = lib.tessera_x86_avx512_tridiagonal_solve_f32
    kernel.argtypes = [_F32, _F32, _F32, _F32, _F32, ctypes.c_int64, ctypes.c_int64]
    kernel.restype = ctypes.c_int
    rng = np.random.default_rng(batch * 100 + n)
    dl = rng.uniform(-0.2, 0.2, n).astype(np.float32)
    du = rng.uniform(-0.2, 0.2, n).astype(np.float32)
    dl[0] = 0.0
    du[-1] = 0.0
    diagonal = (2.0 + np.abs(dl) + np.abs(du)).astype(np.float32)
    rhs = rng.normal(size=(batch, n)).astype(np.float32)
    output = np.empty_like(rhs)
    assert kernel(dl, diagonal, du, rhs, output, batch, n) == 0
    expected = np.asarray(tridiagonal_solve(dl, diagonal, du, rhs))
    np.testing.assert_array_equal(output, expected)


def test_batch_vector_thomas_fails_closed_on_zero_pivot() -> None:
    lib = _library()
    kernel = lib.tessera_x86_avx512_tridiagonal_solve_f32
    kernel.argtypes = [_F32, _F32, _F32, _F32, _F32, ctypes.c_int64, ctypes.c_int64]
    kernel.restype = ctypes.c_int
    diagonal = np.array([0.0, 2.0], dtype=np.float32)
    off = np.zeros(2, dtype=np.float32)
    rhs = np.ones((1, 2), dtype=np.float32)
    assert kernel(off, diagonal, off, rhs, np.empty_like(rhs), 1, 2) != 0


@pytest.mark.parametrize(
    ("reference", "half", "sign"),
    [
        (subset_zeta, 1, 1),
        (subset_mobius, 1, -1),
        (superset_zeta, 0, 1),
        (superset_mobius, 0, -1),
    ],
)
def test_shared_coalition_butterfly_matches_fp64_reference(reference, half: int, sign: int) -> None:
    lib = _library()
    kernel = lib.tessera_x86_avx512_coalition_butterfly_f32
    kernel.argtypes = [
        _F32,
        _F32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    kernel.restype = ctypes.c_int
    rng = np.random.default_rng(568 + half * 10 + sign)
    values = rng.normal(size=(3, 32)).astype(np.float32)
    output = np.empty_like(values)
    assert kernel(values, output, 3, 32, half, sign) == 0
    np.testing.assert_array_equal(output, np.asarray(reference(values)))


def test_shared_coalition_butterfly_rejects_non_power_of_two() -> None:
    lib = _library()
    kernel = lib.tessera_x86_avx512_coalition_butterfly_f32
    kernel.argtypes = [
        _F32,
        _F32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    kernel.restype = ctypes.c_int
    values = np.ones((2, 7), dtype=np.float32)
    assert kernel(values, np.empty_like(values), 2, 7, 1, 1) != 0


@pytest.mark.parametrize(
    ("op_name", "symbol"),
    [
        ("tridiagonal_solve", "tessera_x86_avx512_tridiagonal_solve_f32"),
        ("game_subset_zeta", "tessera_x86_avx512_coalition_butterfly_f32"),
    ],
)
def test_reference_tier_manifest_records_jit_and_stable_symbol_evidence(
    op_name: str, symbol: str
) -> None:
    entry = next(item for item in manifest_for(op_name) if item.target == "x86")
    # A shipped symbol plus a numerical fixture proves the executable JIT
    # lane.  ABI promotion remains gated on a clean exact-device packet and
    # the repository-wide hardware-evidence allow-list.
    assert entry.status == "device_verified_jit"
    assert entry.runtime_symbol == symbol
    assert entry.execute_compare_fixture == "tests/unit/test_x86_reference_tier_compiled.py"
