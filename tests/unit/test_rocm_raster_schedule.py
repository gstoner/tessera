"""ROCM-RASTER-1 schedule-to-generated-kernel wiring."""

from __future__ import annotations

import subprocess

import pytest

from tests._support.compiler_tool import require_tessera_opt


def _generate(order: str = "row_major", group: int = 1) -> str:
    opt = require_tessera_opt("--generate-wmma-gemm-kernel")
    source = f"""
module {{
  "tessera_rocm.wmma_gemm"() {{
    name = "gemm", m = 16 : i64, n = 16 : i64, k = 16 : i64,
    mt = 2 : i64, nt = 4 : i64,
    schedule_raster_order = "{order}",
    schedule_raster_group = {group} : i64
  }} : () -> ()
}}
"""
    result = subprocess.run(
        [str(opt), "-", "--generate-wmma-gemm-kernel"],
        input=source, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_row_major_is_identity_and_is_carried_on_generated_kernel():
    output = _generate()
    column = _generate("column_major")
    assert 'tessera.rocm.schedule_raster_order = "row_major"' in output
    assert "tessera.rocm.schedule_raster_group = 1" in output
    # The identity uses the native 2-D grid coordinates directly; it does not
    # flatten and re-divide the launch grid.
    assert output.count("gpu.block_id") == 2
    assert column.count("arith.remui") > output.count("arith.remui")


@pytest.mark.parametrize("order", ["column_major", "grouped_m", "grouped_n"])
def test_non_identity_rasters_materialize_remap(order: str):
    output = _generate(order, 8)
    assert f'tessera.rocm.schedule_raster_order = "{order}"' in output
    assert "tessera.rocm.schedule_raster_group = 8" in output
    assert "arith.remui" in output


def test_invalid_raster_fails_closed():
    opt = require_tessera_opt("--generate-wmma-gemm-kernel")
    source = """
module {
  "tessera_rocm.wmma_gemm"() {
    name = "gemm", m = 16 : i64, n = 16 : i64, k = 16 : i64,
    schedule_raster_order = "morton"
  } : () -> ()
}
"""
    result = subprocess.run(
        [str(opt), "-", "--generate-wmma-gemm-kernel"],
        input=source, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "schedule raster order must be" in result.stderr
