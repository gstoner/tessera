"""gfx1151 F16-decode scaled product: compiler boundary and instruction census.

GFX1151-QUANTIZED-MLP-20261030, slice 2. RDNA 3.5 has no FP8 WMMA, so the
quantized MLP's K16 fp32-block-scaled E4M3 x E4M3 product decodes each fragment
exactly to f16 and runs V_WMMA_F32_16X16X16_F16 with fp32 scale-group partials.
Compile-only: these tests stop at a gfx1151 HSACO and its disassembly; no
gfx1151 execution or numerical result is claimed here.
"""
import re

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route


def _member(m=16, h=32, k=64, n=32):
    """The MLP's gate scaled-product member Graph, retargeted to gfx1151."""
    from tessera.compiler.rocm_quantized_product_program import _export
    from tests.unit.test_native_quantized_mlp_program import source
    text = _export(find_tessera_opt(), source(m, h, k, n))["member_graphs"][2]
    assert text.count('"rocm_gfx1201"') == 1 and text.count('tessera.arch = "gfx1201"') == 1
    return text.replace('"rocm_gfx1201"', '"rocm_gfx1151"').replace(
        'tessera.arch = "gfx1201"', 'tessera.arch = "gfx1151"')


def _tile(text):
    tool = find_tessera_opt()
    return run_tessera_opt(tool, run_tessera_opt(tool, text, "--tessera-graph-to-schedule"),
                           "--tessera-schedule-to-tile")


@pytest.mark.usefixtures("production_compiler")
def test_gfx1151_scaled_product_takes_the_register_panel_and_a_wmma_descriptor():
    schedule = run_tessera_opt(find_tessera_opt(), _member(256, 512, 4096, 256), "--tessera-graph-to-schedule")
    record = next(line for line in schedule.splitlines() if "schedule.matmul" in line)
    # The register panel: one wave, never the LDS body (none exists on gfx1151).
    assert 'staging = "lds"' not in record and "warps = 1" in record
    assert 'arch = "gfx1151"' in record
    tile = _tile(_member())
    assert 'family = "wmma"' in tile and 'a = "e4m3", b = "e4m3"' in tile


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("m,h,k,n", [(16, 32, 64, 32), (37, 128, 4096, 256)])
def test_gfx1151_scaled_product_compiles_to_f16_wmma_only(rocm_image_toolchain, m, h, k, n):
    from tessera.compiler.rocm_native import _compile_native_tile_ir
    from tests._support.rocm_isa import disassemble
    target, _, payload, *_ = _compile_native_tile_ir(
        _tile(_member(m, h, k, n)), directive="tessera_rocm.scaled_wmma_gemm",
        family="matmul", architecture="gfx1151")
    assert payload.startswith(b"\x7fELF")
    text = disassemble(payload, chip="gfx1151")
    wmma = re.findall(r"\bv_wmma_\w+", text)
    assert wmma and set(wmma) == {"v_wmma_f32_16x16x16_f16"}
    assert not re.search(r"\bv_\w*(fp8|bf8)\w*", text)


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("change", [
    # Only the fp32-scaled E4M3 pair decodes; E5M2 has no gfx1151 form.
    lambda text: text.replace("f8E4M3FN", "f8E5M2"),
])
def test_gfx1151_refuses_other_fp8_forms(change):
    text = change(_member())
    with pytest.raises(RuntimeError):
        _tile(text)


@pytest.mark.usefixtures("production_compiler")
def test_gfx1201_scaled_product_keeps_the_fp8_instruction(rocm_image_toolchain):
    from tessera.compiler.rocm_native import _compile_native_tile_ir
    from tests._support.rocm_isa import disassemble
    text = _member().replace('"rocm_gfx1151"', '"rocm_gfx1201"').replace(
        'tessera.arch = "gfx1151"', 'tessera.arch = "gfx1201"')
    _, _, payload, *_ = _compile_native_tile_ir(
        _tile(text), directive="tessera_rocm.scaled_wmma_gemm", family="matmul", architecture="gfx1201")
    assert set(re.findall(r"\bv_wmma_\w+", disassemble(payload, chip="gfx1201"))) == {
        "v_wmma_f32_16x16x16_fp8_fp8"}


def test_panel_oracle_mirrors_gfx1151_register_only():
    from tessera.compiler.rocm_fp8_blockscale import BlockScalePanel, BlockScaleShape, blockscale_panel_oracle
    shape = BlockScaleShape(256, 4096, 4096, 16, 1, "nk")
    assert blockscale_panel_oracle(shape).staging == "lds"            # gfx1201 takes LDS here
    assert blockscale_panel_oracle(shape, arch="gfx1151") == BlockScalePanel("global", 32, 32, 1)
    assert blockscale_panel_oracle(shape, arch="gfx1151", min_m=1) == BlockScalePanel("global", 16, 32, 1)


@pytest.mark.usefixtures("production_compiler")
def test_gfx1151_large_scaled_schedule_matches_the_oracle():
    from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, verify_blockscale_schedule
    schedule = run_tessera_opt(find_tessera_opt(), _member(256, 2048, 4096, 256), "--tessera-graph-to-schedule")
    # Gate N = 2H = 4096: gfx1201 would take LDS; gfx1151 keeps the register panel.
    panel = verify_blockscale_schedule(BlockScaleShape(256, 4096, 4096, 16, 1, "nk", "bf16"), schedule)
    assert panel.staging == "global"
