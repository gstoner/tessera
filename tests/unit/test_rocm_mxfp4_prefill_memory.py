"""Count model-level packed/expanded weight residency without selecting a route."""
from __future__ import annotations

import pytest

from tessera.compiler.rocm_mxfp4_prefill_memory import (
    MXFP4LayerShape, assess_prefill_weight_residency,
)


def test_model_residency_keeps_packed_decode_and_expanded_prefill() -> None:
    layer = MXFP4LayerShape(17408, 5120, count=32)
    packed = 32 * (17408 * 5120 // 2 + (5120 // 32 + 1) * 17408)
    expanded = 32 * (17408 * 5120 + 17408)
    refused = assess_prefill_weight_residency(
        (layer,), available_extra_bytes=expanded - 1,
    )
    assert refused["packed_weight_and_scale_bytes"] == packed
    assert refused["expanded_weight_and_reference_bytes"] == expanded
    assert refused["incremental_bytes"] == expanded
    assert refused["selection_state"] == "refused_budget"
    admitted = assess_prefill_weight_residency(
        (layer,), available_extra_bytes=expanded,
    )
    assert admitted["budget_allows_trial"] is True
    assert admitted["selection_state"] == "manual_trial_only"
    prefill_only = assess_prefill_weight_residency(
        (layer,), available_extra_bytes=expanded - packed,
        retain_packed_for_decode=False,
    )
    assert prefill_only["incremental_bytes"] == expanded - packed


@pytest.mark.parametrize("shape", [
    MXFP4LayerShape(48, 64), MXFP4LayerShape(80, 128),
])
def test_ragged_supported_n_shapes_have_exact_byte_accounting(shape: MXFP4LayerShape) -> None:
    report = assess_prefill_weight_residency((shape,), available_extra_bytes=0)
    assert report["packed_weight_and_scale_bytes"] == (
        shape.n * shape.k // 2 + (shape.k // 32 + 1) * shape.n
    )
    assert report["budget_allows_trial"] is False


def test_model_residency_refuses_malformed_shapes() -> None:
    with pytest.raises(ValueError, match="N16"):
        MXFP4LayerShape(17, 64)
    with pytest.raises(ValueError, match="nonnegative"):
        assess_prefill_weight_residency((MXFP4LayerShape(48, 64),), available_extra_bytes=-1)
