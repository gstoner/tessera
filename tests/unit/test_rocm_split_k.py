"""ROCM-SPLIT-K-1: cross-workgroup split-K on the gfx1201 typed route.

Host-free half. The decision has ONE production authority -- the C++
`selectGfx1201SplitK` in PMPasses.cpp -- and one declared oracle,
`rocm_tiling.select_split_k`, which `scheduled_matmul.verify_matmul_projection`
recomputes on every package (Decision #31). These tests pin the oracle's
behaviour (so a wrong model fails a test rather than hiding, the #29a lesson
from the old `k > 4096` predicate), then check that the two deciders agree on
real native lowerings and that every consumer fails closed on a split that
arrives without its reduction order (Decision #21a).

Device execution and timing are in `test_rocm_gfx1201_scheduled.py` and the
recorded packet; nothing here is device evidence.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tessera.compiler.rocm_tiling import SPLIT_K_MIN_SLICE_K, select_split_k
from tessera.compiler.scheduled_matmul import (
    _rocm_profile,
    find_tessera_opt,
    rocm_split_k,
)


def _gfx1201(m: int, n: int, k: int, *, macro=(16, 16), block_k=32,
             dtype="fp16", dynamic=False):
    return select_split_k(m, n, k, macro_tile=macro, block_k=block_k,
                          profile=_rocm_profile("gfx1201"), dtype=dtype,
                          dynamic=dynamic)


# ── the oracle's behaviour ──────────────────────────────────────────────────


def test_router_gate_shape_splits_in_two() -> None:
    """16x256x2048: 16 output tiles against 32 WGPs -- the shape the old
    `k > 4096` model answered False for."""
    assert _gfx1201(16, 256, 2048) == (2, None)
    assert _gfx1201(16, 256, 2048, dtype="bf16") == (2, None)


def test_slice_count_fills_the_machine_as_a_power_of_two() -> None:
    # One tile wants 32 slices; K=8192 gives 32 slices of exactly 256.
    assert _gfx1201(16, 16, 8192) == (32, None)
    # Five tiles want ceil(32/5) = 7 -> rounded down to 4.
    assert _gfx1201(16, 80, 4096) == (4, None)


def test_occupied_machine_is_never_split() -> None:
    """Negative case (Decision #10a): 256 tiles is not occupancy-short."""
    assert _gfx1201(1024, 1024, 2048, macro=(64, 64)) == (1, None)
    assert _gfx1201(512, 512, 4096) == (1, None)  # 32x32 = 1024 tiles


def test_unaligned_k_falls_back_with_a_reason_not_silently() -> None:
    slices, reason = _gfx1201(16, 256, 2050)
    assert slices == 1
    assert reason is not None and "K=2050" in reason


def test_minimum_slice_guard_bounds_the_split() -> None:
    # K=256 would give two 128-wide slices, below the guard.
    slices, reason = _gfx1201(16, 256, 256)
    assert slices == 1 and reason is not None
    # K=512 gives exactly two slices of the guard size.
    assert _gfx1201(16, 256, 2 * SPLIT_K_MIN_SLICE_K) == (2, None)
    # The guard also caps the count: 1 tile wants 32, K=2048 allows 8.
    assert _gfx1201(16, 16, 2048) == (8, None)


def test_no_split_without_a_macro_k_block_or_static_shape() -> None:
    assert _gfx1201(16, 256, 2048, block_k=0) == (1, None)
    assert _gfx1201(16, 256, 2048, dynamic=True) == (1, None)


def test_only_gfx1201_float_storage_is_split() -> None:
    """gfx1151 has no split-K evidence and the fp8/integer storages have no
    device proof of a split; the oracle states that scope, as does C++."""
    router = dict(macro_tile=(16, 16), dynamic=False)
    assert rocm_split_k(16, 256, 2048, target="rocm_gfx1201", storage="f16", **router) == (2, "ordered")
    assert rocm_split_k(16, 256, 2048, target="rocm_gfx1201", storage="bf16", **router) == (2, "ordered")
    assert rocm_split_k(16, 256, 2048, target="rocm_gfx1151", storage="f16", **router) == (1, "")
    for storage in ("e4m3", "int8", "int4"):
        assert rocm_split_k(16, 256, 2048, target="rocm_gfx1201", storage=storage, **router) == (1, "")


# ── authority vs oracle, on real native lowerings ──────────────────────────


needs_opt = pytest.mark.skipif(find_tessera_opt() is None,
                               reason="requires a built tessera-opt")


def _lower(shape, **kwargs):
    from tessera.compiler import scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module
    return scheduled_matmul.lower_scheduled_matmul(
        _module(target="rocm", shape=shape, **kwargs), target="rocm_gfx1201")


@needs_opt
@pytest.mark.parametrize("shape,expected", [
    ((16, 2048, 256), 2),     # (m, k, n): the router gate
    ((16, 2050, 256), 1),     # ragged K: occupancy asks, nothing aligns
    ((16, 8192, 16), 32),     # one tile
    ((64, 4096, 64), 2),      # 16 tiles
    ((1024, 2048, 1024), 1),  # occupied
    ((17, 19, 23), 1),        # K too small for a macro K block
])
def test_native_schedule_and_oracle_agree(shape, expected) -> None:
    from tessera.compiler.scheduled_matmul import verify_matmul_projection
    artifact = _lower(shape)
    verify_matmul_projection(artifact)
    assert artifact.split_k == expected
    assert artifact.split_k_reduction == ("ordered" if expected > 1 else "")
    assert ("tessera.split_k" in artifact.tile_ir) is (expected > 1)


@needs_opt
def test_artifact_that_disagrees_with_its_tile_ir_is_refused() -> None:
    artifact = _lower((16, 2048, 256))
    for forged in (replace(artifact, split_k=4), replace(artifact, split_k=1, split_k_reduction="")):
        with pytest.raises(ValueError, match="split-K"):
            forged.validate()


@needs_opt
def test_projection_refuses_when_the_oracle_and_the_authority_diverge(monkeypatch) -> None:
    """The differential half of Decision #31: if the Python predicate and the
    C++ Schedule ever answer differently, packaging stops."""
    from tessera.compiler import scheduled_matmul
    artifact = _lower((16, 2048, 256))
    scheduled_matmul.verify_matmul_projection(artifact)
    monkeypatch.setattr(scheduled_matmul, "rocm_split_k", lambda *a, **k: (4, "ordered"))
    with pytest.raises(ValueError, match="oracle disagrees"):
        scheduled_matmul.verify_matmul_projection(artifact)


@needs_opt
def test_tile_artifact_must_carry_the_pair() -> None:
    artifact = _lower((16, 2048, 256))
    dropped = artifact.tile_ir.replace(', tessera.split_k_reduction = "ordered"', "")
    assert dropped != artifact.tile_ir
    with pytest.raises(ValueError, match="split-K"):
        replace(artifact, tile_ir=dropped).validate()
    with pytest.raises(ValueError):
        replace(artifact, split_k_reduction="atomic").validate()


@needs_opt
def test_split_epilogue_stays_the_programs() -> None:
    """The Tile op states the PROGRAM's epilogue under a split; the generator
    moves it to the ordered reduction. Dropping it here would be the
    'fallback must compute the same program' failure."""
    artifact = _lower((16, 2048, 256), activation="gelu", bias=True)
    assert artifact.split_k == 2
    assert 'activation = "gelu"' in artifact.tile_ir and "bias = true" in artifact.tile_ir


@needs_opt
def test_lds_staging_refuses_a_split_schedule() -> None:
    from tessera.compiler import rocm_native
    artifact = _lower((16, 2048, 256))
    with pytest.raises(ValueError, match="ROCM-SPLIT-K-1"):
        rocm_native.package_scheduled_matmul(
            artifact, pipeline_name="tessera-lower-to-rocm", staging="lds")
