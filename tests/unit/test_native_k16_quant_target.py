"""Mixed-width K16 Target SSA, checked native frame, and image serialization."""
import base64
import json
import re
import subprocess

import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_k16_quant_graph import source

pytestmark = pytest.mark.compiler_route


def stages(rows=3,columns=32):
    tool = find_tessera_opt()
    tile = run_tessera_opt(tool,source(rows,columns),
        "--pass-pipeline=builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile)")
    target = run_tessera_opt(tool,tile,"--lower-tile-to-rocm=arch=gfx1201")
    gpu = run_tessera_opt(tool,target,"--lower-tessera-target-to-rocdl")
    return tile,target,gpu


def member(text):
    raw = re.search(r'tessera.rocm.block_quant_member_json = "([A-Za-z0-9+/=]+)"',text).group(1)
    return json.loads(base64.b64decode(raw))


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("rows,columns",[(3,32),(19,12288),(256,12288)])
def test_quantizer_target_keeps_storage_widths_and_two_output_lifetimes(rows,columns):
    tile,target,gpu = stages(rows,columns)
    assert "tessera_rocm.row_region_kernel" in target
    assert "tile.buffer_view" not in target and "tile.store_element" not in target
    assert target.count("llvm.load") == 2 and target.count("llvm.store") == 2
    assert "arith.bitcast" in target and "f8E4M3FN to i8" in target
    assert "gpu.func" in gpu and "rocdl.cvt.pk.fp8.f32" in gpu
    assert "arith.truncf" not in gpu and "tessera_rocm.row_region_kernel" not in gpu
    assert "memref<?xbf16>" in gpu and "memref<?xi8>" in gpu and "memref<?xf32>" in gpu
    a,b = member(target),member(gpu)
    assert a == b
    assert b["profile"] == "gfx1201_block_quant_bf16_e4m3_f32_k16_v1"
    assert b["inputs"] == [0] and b["outputs"] == [1,2]
    groups = rows*columns//16
    assert b["scalars"] == [groups]
    assert b["geometry"] == [(groups+255)//256,1,1,256,1,1]
    program = json.loads(base64.b64decode(b["program_base64"]))
    assert program["kind"] == "block_quant" and program["outputs"] == [1,2]
    assert [buf["bytes"] for buf in program["buffers"]] == [rows*columns*2,rows*columns,groups*4]
    assert [buf["storage"] for buf in program["buffers"]] == ["bf16","f8E4M3FN","f32"]
    assert [buf["ownership"] for buf in program["buffers"]] == [0,2,2]
    assert [buf["first_write"] for buf in program["buffers"]] == [-1,0,0]
    assert [buf["last_read"] for buf in program["buffers"]] == [0,1,1]
    assert "tessera.reduce" in program["root_ir"] and "tessera.cast" in program["root_ir"]
    assert "tile.row_region_kernel" in tile


@pytest.mark.usefixtures("production_compiler")
def test_quantizer_target_rejects_sibling_architecture():
    tile,_,_ = stages()
    with pytest.raises(RuntimeError,match="isolated gfx1201"):
        run_tessera_opt(find_tessera_opt(),tile,"--lower-tile-to-rocm=arch=gfx1151")


@pytest.mark.usefixtures("production_compiler")
def test_quantizer_native_frame_cannot_be_replaced_after_export():
    _,target,_ = stages()
    changed = re.sub(r'(tessera.rocm.block_quant_member_json = ")[A-Za-z0-9+/=]+',r'\1e30=',target)
    assert changed != target
    with pytest.raises(RuntimeError,match="frame changed"):
        run_tessera_opt(find_tessera_opt(),changed,"--lower-tessera-target-to-rocdl")


@pytest.mark.compiler_rocm
def test_quantizer_gpu_body_serializes_as_real_gfx1201_hsaco(compiler_toolchain):
    _,_,gpu = stages()
    tool = compiler_toolchain.require_mlir_opt()
    pipeline = ("builtin.module(gpu.module(convert-scf-to-cf,"
        "convert-gpu-to-rocdl{chipset=gfx1201},reconcile-unrealized-casts),"
        "rocdl-attach-target{chip=gfx1201},gpu-module-to-binary)")
    result = subprocess.run([str(tool),f"--pass-pipeline={pipeline}"],input=gpu,
                            text=True,capture_output=True,check=False)
    assert result.returncode == 0,result.stderr
    from tessera.compiler.rocm_native import _extract_hsaco
    payload = _extract_hsaco(result.stdout)
    assert payload.startswith(b"\x7fELF") and len(payload)>1024
