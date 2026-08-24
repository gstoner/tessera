"""Exact-device proofs for the canonical scheduled SM120 matmul package.

The implementation helpers remain beside the host-free scheduled-contract
tests, but NVIDIA-TEST-6 requires every hardware node to be collected from a
device root.  Exporting marked aliases here keeps one implementation while
making the proof environment explicit.
"""

from __future__ import annotations

import pytest

from tests.unit import test_scheduled_matmul_consumers as _shared


test_sm120_typed_scheduled_matmul_executes_exact_artifact = (
    pytest.mark.hardware_nvidia(
        _shared._sm120_typed_scheduled_matmul_executes_exact_artifact
    )
)
test_sm120_macro_cta_reuses_shared_panels_exact_device = (
    pytest.mark.hardware_nvidia(
        _shared._sm120_macro_cta_reuses_shared_panels_exact_device
    )
)
test_sm120_macro_cta_k_tail_exact_device = pytest.mark.hardware_nvidia(
    _shared._sm120_macro_cta_k_tail_exact_device
)
test_sm120_scheduled_epilogue_reduced_output_exact_device = (
    pytest.mark.hardware_nvidia(
        _shared._sm120_scheduled_epilogue_reduced_output_exact_device
    )
)
test_sm120_macro_cta_bf16_exact_device = pytest.mark.hardware_nvidia(
    _shared._sm120_macro_cta_bf16_exact_device
)
test_sm120_bounded_dynamic_strided_matmul_exact_device = (
    pytest.mark.hardware_nvidia(
        _shared._sm120_bounded_dynamic_strided_matmul_exact_device
    )
)
