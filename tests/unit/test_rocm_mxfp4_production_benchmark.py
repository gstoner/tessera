"""Host-free contracts for the matched gfx1201 MXFP4 benchmark."""
from __future__ import annotations

import numpy as np

from benchmarks.rocm.benchmark_gfx1201_mxfp4_production import (
    Case,
    _fragment_order,
    _logical_inputs,
    _parse_case,
)
from tessera.compiler import rocm_mxfp4 as mx


def test_matched_input_uses_canonical_physical_layouts() -> None:
    case = Case("decode", 3, 32, 64)
    inputs = _logical_inputs(case)
    packed = inputs["packed_row_major"]
    codes = mx.unpack_e2m1_codes(packed)

    np.testing.assert_array_equal(
        _fragment_order(packed, case.n, case.k),
        mx.to_fragment_order(packed),
    )
    assert inputs["a"].shape == (case.m, case.k)
    assert packed.shape == (case.n, case.k // 2)
    assert codes.shape == (case.n, case.k)
    assert inputs["b_scale"].shape == (case.k // 32, case.n)


def test_matched_input_keeps_row_reference_fold_exact() -> None:
    inputs = _logical_inputs(Case("decode", 2, 32, 64))
    scales = inputs["b_scale"]
    reference = inputs["row_reference"]
    delta = reference[None, :].astype(np.int16) - scales.astype(np.int16)
    assert int(delta.min()) == 0
    assert int(delta.max()) <= 2


def test_case_parser_keeps_workload_separate_from_shape() -> None:
    assert _parse_case("prefill:256x5120x8704") == Case(
        "prefill", 256, 5120, 8704
    )
