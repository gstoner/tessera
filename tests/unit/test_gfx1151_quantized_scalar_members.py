"""gfx1151 compile-only admission of the quantized MLP's scalar members.

GFX1151-QUANTIZED-MLP-20261030. RDNA 3.5 has no FP8 WMMA or FP8 convert, but
the K16 quantizer is scalar code: on gfx1151 its E4M3 cast lowers to the
software encode proven bit-identical to RDNA4's V_CVT_PK_FP8_F32 (see
tests/device/rocm/test_e4m3_software_encode.py). These tests stop at a
serialized gfx1151 HSACO; no gfx1151 execution is claimed here.
"""
import json
import subprocess

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_k16_quant_graph import source as gfx1201_source
from tests.unit.test_native_k16_quant_target import member

pytestmark = pytest.mark.compiler_route


def source(rows=3, columns=32):
    return gfx1201_source(rows, columns).replace(
        'tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"',
        'tessera.target = "rocm_gfx1151", tessera.arch = "gfx1151"', 1)


def stages(rows=3, columns=32):
    tool = find_tessera_opt()
    tile = run_tessera_opt(tool, source(rows, columns),
        "--pass-pipeline=builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile)")
    target = run_tessera_opt(tool, tile, "--lower-tile-to-rocm=arch=gfx1151")
    gpu = run_tessera_opt(tool, target, "--lower-tessera-target-to-rocdl")
    return tile, target, gpu


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("rows,columns", [(3, 32), (19, 12288)])
def test_gfx1151_quantizer_lowers_with_the_software_e4m3_encode(rows, columns):
    tile, target, gpu = stages(rows, columns)
    assert 'arch = "gfx1151"' in tile and "tessera_rocm.row_region_kernel" in target
    # No FP8 instruction, and the cast is gone: the software encode replaced it.
    assert "rocdl.cvt.pk.fp8" not in gpu and "arith.truncf" not in gpu
    assert member(target) == member(gpu)
    info = member(gpu)
    assert info["architecture"] == "gfx1151"
    assert info["profile"] == "gfx1151_block_quant_bf16_e4m3_f32_k16_v1"
    groups = rows * columns // 16
    assert info["scalars"] == [groups] and info["geometry"] == [(groups + 255) // 256, 1, 1, 256, 1, 1]


@pytest.mark.usefixtures("production_compiler")
def test_gfx1151_and_gfx1201_quantizers_have_distinct_identities():
    tool = find_tessera_opt()
    pipeline = "--pass-pipeline=builtin.module(tessera-graph-to-schedule)"
    a = run_tessera_opt(tool, source(), pipeline)
    b = run_tessera_opt(tool, gfx1201_source(), pipeline)
    assert 'arch = "gfx1151"' in a and 'arch = "gfx1201"' in b
    hash_of = lambda text: text.split('hash = "')[1].split('"')[0]
    assert hash_of(a) != hash_of(b)


@pytest.mark.usefixtures("production_compiler")
def test_quantizer_refuses_a_mismatched_target_and_arch():
    text = source().replace('tessera.arch = "gfx1151"', 'tessera.arch = "gfx1201"', 1)
    with pytest.raises(RuntimeError, match="explicit BF16/max/scale/normalize/E4M3 Graph contract"):
        run_tessera_opt(find_tessera_opt(), text, "--tessera-graph-to-schedule")


@pytest.mark.compiler_rocm
def test_gfx1151_quantizer_serializes_as_a_gfx1151_hsaco(compiler_toolchain):
    _, _, gpu = stages()
    pipeline = ("builtin.module(gpu.module(convert-scf-to-cf,"
        "convert-gpu-to-rocdl{chipset=gfx1151},reconcile-unrealized-casts),"
        "rocdl-attach-target{chip=gfx1151},gpu-module-to-binary)")
    result = subprocess.run([str(compiler_toolchain.require_mlir_opt()), f"--pass-pipeline={pipeline}"],
                            input=gpu, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    from tessera.compiler.rocm_native import _extract_hsaco
    payload = _extract_hsaco(result.stdout)
    assert payload.startswith(b"\x7fELF") and len(payload) > 1024


def test_gfx1151_executable_package_waits_for_device_promotion(rocm_image_toolchain):
    """The packager compiles for the Graph's architecture, but an executable
    gfx1151 image needs that family promoted by exact-device proof first."""
    from tessera.compiler.rocm_block_quant_native import lower_block_quant_text, package_block_quant_recipe
    recipe = lower_block_quant_text(source())
    with pytest.raises(ValueError, match="no promoted family plugins for gfx1151"):
        package_block_quant_recipe(recipe)
