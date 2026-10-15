"""Native register scale-group defaults must fit the Graph's legal granularity."""
from __future__ import annotations

import subprocess
import pytest
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, lower_blockscale
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def _compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("requires production compiler")
    help_text = subprocess.run([str(tool), "--help"], capture_output=True, text=True, check=True).stdout
    if "--generate-wmma-gemm-kernel" not in help_text:
        pytest.skip("requires native ROCm generator in production compiler")
    return tool


def _kernel(scale_k, panels=None):
    tool = _compiler()
    shape = BlockScaleShape(7, 19, scale_k * 2, scale_k, 1, "nk", "bf16")
    tile = lower_blockscale(shape, tessera_opt=tool).tile_ir
    options = "via-tile=true"
    if panels is not None:
        options += f" scale-group-panels={panels}"
    return run_tessera_opt(tool, tile, "--generate-wmma-gemm-kernel=" + options)


@pytest.mark.compiler_route
@pytest.mark.parametrize("scale_k", [16, 48, 80])
def test_implicit_panels_fit_an_odd_instruction_group(scale_k):
    automatic = _kernel(scale_k)
    assert "tessera.rocm.scale_group_panels = 1 : i64" in automatic
    assert automatic == _kernel(scale_k, 1)


@pytest.mark.compiler_route
@pytest.mark.parametrize("scale_k", [32, 128])
def test_existing_even_group_default_is_unchanged(scale_k):
    automatic = _kernel(scale_k)
    assert "tessera.rocm.scale_group_panels = 2 : i64" in automatic
    assert automatic == _kernel(scale_k, 2)


@pytest.mark.compiler_route
@pytest.mark.parametrize("scale_k", [16, 48, 80])
def test_explicit_incompatible_panel_request_still_rejects(scale_k):
    with pytest.raises(RuntimeError, match="must divide the group's"):
        _kernel(scale_k, 2)
