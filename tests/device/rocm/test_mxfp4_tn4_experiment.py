"""Exact-device, manual TN4 prefill proof on a ragged N tile."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tessera import runtime as rt
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.benchmark_gfx1201_mxfp4_safe_epilogue import _candidate_engine
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera.compiler.rocm_mxfp4_tn4_experiment import package_folded_tn4_experiment


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("shape", [(65, 144, 64), (257, 144, 128)])
def test_tn4_ragged_output_matches_exact_k32(shape: tuple[int, int, int]) -> None:
    assert rt._rocm_live_arch() == "gfx1201"
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    case = base.Case("prefill", *shape)
    inputs = base._logical_inputs(case)
    folded = prepare_folded_weights(
        inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True,
    )
    assert folded.lossless
    exact = base._tessera_engine(hip, case, inputs, 1, None, None)
    tn4 = _candidate_engine(hip, case, inputs, 1, folded, tn4=True)
    try:
        np.testing.assert_array_equal(
            tn4.output().view(np.uint16), exact.output().view(np.uint16),
        )
    finally:
        tn4.close()
        exact.close()


def test_tn4_refuses_narrow_n_before_compilation() -> None:
    from tessera.compiler import rocm_mxfp4 as mx

    packed = mx.pack_e2m1_codes(np.ones((80, 64), dtype=np.uint8))
    folded = prepare_folded_weights(
        packed, np.full((2, 80), 127, dtype=np.uint8), allow_approximate=True,
    )
    with pytest.raises(ValueError, match="N >= 128"):
        package_folded_tn4_experiment(65, 80, 64, folded)
