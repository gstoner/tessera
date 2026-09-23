"""Host-array certification and fail-closed fast-epilogue codegen."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.rocm_mxfp4_folded import (
    certify_folded_safe_scales,
    emit_mxfp4_folded_prefill_hip,
)


def test_safe_certificate_binds_input_bytes_and_bounds() -> None:
    scales = np.array([1.0, 0.5, -2.0], dtype=np.float32)
    refs = np.array([126, 127, 128], dtype=np.uint8)
    certificate = certify_folded_safe_scales(scales, refs)
    assert certificate["minimum_abs_product"] == 0.25
    assert certificate["maximum_abs_product"] == 4.0
    assert len(str(certificate["activation_scale_sha256"])) == 64
    assert len(str(certificate["row_reference_sha256"])) == 64
    changed = scales.copy()
    changed[0] = 1.5
    assert certify_folded_safe_scales(changed, refs)["activation_scale_sha256"] != (
        certificate["activation_scale_sha256"]
    )


@pytest.mark.parametrize("scales,refs", [
    ([0.0], [127]),
    ([float("inf")], [127]),
    ([float("nan")], [127]),
    ([1.0], [0]),
    ([1.0], [255]),
    ([2.0**-127], [1]),
    ([2.0**127], [254]),
])
def test_safe_certificate_refuses_extreme_or_reserved_scales(
    scales: list[float], refs: list[int],
) -> None:
    with pytest.raises(ValueError, match="safe folded epilogue"):
        certify_folded_safe_scales(
            np.array(scales, dtype=np.float32), np.array(refs, dtype=np.uint8),
        )


def test_safe_epilogue_has_distinct_codegen_without_double_fallback() -> None:
    ordinary = emit_mxfp4_folded_prefill_hip(full_k64=True)
    safe = emit_mxfp4_folded_prefill_hip(full_k64=True, safe_epilogue=True)
    assert "#if !0" in ordinary
    assert "#if !1" in safe
    assert "const float combined_scale = row_scale * activation_scale;" in safe
    assert "O[m * N + n] = (__bf16)scaled;" in safe
    assert ordinary != safe
