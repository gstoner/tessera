"""The manual BN128 source widens B staging and all N consumers together."""
from __future__ import annotations

from tessera.compiler.rocm_mxfp4_tn4_experiment import emit_folded_tn4_experiment_hip


def test_tn4_widens_lds_compute_epilogue_and_grid_source() -> None:
    source = emit_folded_tn4_experiment_hip()
    assert "sB[128 * 80]" in source
    assert "floatx8 acc[4][4]" in source
    assert "blockIdx.x * 128" in source
    assert "for (int q = 0; q < 2; ++q)" in source
    assert "sB + (slot / 4) * 80 + off" in source
    assert "fragment_i32x2 af[4], bf[4]" in source
    assert source.count("for (int j = 0; j < 4; ++j)") == 4
    assert source.count("wn * 64 + j * 16") == 2
