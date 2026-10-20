"""Checked contiguous loads preserve the existing bounded Tile fragment contract."""
import subprocess

import pytest

PAIRS = (
    ("f16", "f16"), ("bf16", "bf16"),
    ("e4m3", "e4m3"), ("e5m2", "e5m2"),
    ("e4m3", "e5m2"), ("e5m2", "e4m3"),
    ("int8", "int8"), ("int4", "int4"),
)
TYPES = {"f16": "f16", "bf16": "bf16", "e4m3": "f8E4M3FN",
         "e5m2": "f8E5M2", "int8": "i8", "int4": "i8"}
SHAPES = ((1, 1, 1), (7, 9, 7), (8, 8, 8), (9, 7, 9),
          (15, 15, 15), (16, 16, 16), (7, 16, 16), (16, 7, 16),
          (1, 16, 8), (16, 1, 8), (15, 16, 9), (16, 15, 7))


def fragment_source(a, b, *, bounded=True,
                    orders=("row_major", "col_major")):
    """An unmasked full output makes invalid input-fragment elements visible."""
    accum = "i32" if a.startswith("int") else "f32"
    a_args = "%a_mem, %zero, %zero" + (", %m, %k" if bounded else "")
    b_args = "%b_mem, %zero, %zero" + (", %k, %n" if bounded else "")
    indices = ", index" * (4 if bounded else 2)
    layout = '#tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>'
    return f"""
!fa = !tile.fragment<m = 16, n = 16, k = 16, elem = "{a}", acc = "{accum}", role = "a", layout = "row_major", family = "auto">
!fb = !tile.fragment<m = 16, n = 16, k = 16, elem = "{b}", acc = "{accum}", role = "b", layout = "col_major", family = "auto">
!fc = !tile.fragment<m = 16, n = 16, k = 16, elem = "{accum}", acc = "{accum}", role = "acc", layout = "row_major", family = "auto">
#mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "{a}", b = "{b}", acc = "{accum}", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
module {{
  gpu.module @bounded_fragment_mod {{
    gpu.func @bounded_fragment_values(
        %a_mem: memref<?x{TYPES[a]}>, %b_mem: memref<?x{TYPES[b]}>,
        %d_mem: memref<?x{accum}>, %m: index, %n: index, %k: index) kernel {{
      %zero = arith.constant 0 : index
      %av = tile.view {a_args} {{
        tile.layout = {layout},
        tile.memory = #tile.memory_layout<space = "gmem", order = "{orders[0]}", leading_dim = 16>
      }} : (memref<?x{TYPES[a]}>{indices}) -> !tile.tile
      %bv = tile.view {b_args} {{
        tile.layout = {layout},
        tile.memory = #tile.memory_layout<space = "gmem", order = "{orders[1]}", leading_dim = 16>
      }} : (memref<?x{TYPES[b]}>{indices}) -> !tile.tile
      %af = tile.fragment_pack %av {{role = "a", mma = #mma}} : (!tile.tile) -> !fa
      %bf = tile.fragment_pack %bv {{role = "b", mma = #mma}} : (!tile.tile) -> !fb
      %z = tile.fragment_zero {{role = "acc", mma = #mma}} : !fc
      %c = tile.mma %af, %bf, %z {{mma = #mma}} : (!fa, !fb, !fc) -> !fc
      %out = tile.fragment_unpack %c {{tile.layout = {layout}, mma = #mma}} : (!fc) -> !tile.tile
      "tile.store"(%out, %d_mem, %zero, %zero) {{
        tile.layout = {layout},
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      }} : (!tile.tile, memref<?x{accum}>, index, index) -> ()
      gpu.return
    }}
  }}
}}
"""


def lower_fragment(tools, text, arch, *, rocdl=False):
    pipeline = f"builtin.module(lower-tile-to-rocm{{arch={arch}}}"
    pipeline += ",lower-tessera-target-to-rocdl)" if rocdl else ")"
    flag = "--pass-pipeline=" + pipeline
    compiler = tools.require_tessera_opt(flag)
    result = subprocess.run([str(compiler), "-", flag], input=text, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.compiler_route
@pytest.mark.parametrize("arch,a,b",
    [("gfx1201", a, b) for a, b in PAIRS]
    + [("gfx1151", t, t) for t in ("f16", "bf16", "int8")]
    + [("gfx942", t, t) for t in ("f16", "bf16")])
@pytest.mark.parametrize("bounded", [False, True])
def test_contiguous_fragment_retains_checked_vector_load(
        compiler_toolchain, arch, a, b, bounded):
    lowered = lower_fragment(compiler_toolchain,
        fragment_source(a, b, bounded=bounded), arch)
    assert lowered.count("vector.load") == 2
    if bounded:
        assert "scf.if" in lowered
        assert "memref.load" in lowered  # Partial K still has a scalar fallback.
    else:
        assert "memref.load" not in lowered


@pytest.mark.compiler_route
@pytest.mark.parametrize("orders,expected",
    [(("col_major", "col_major"), 1), (("row_major", "row_major"), 1),
     (("col_major", "row_major"), 0)])
def test_strided_fragment_keeps_scalar_elements(compiler_toolchain, orders, expected):
    lowered = lower_fragment(compiler_toolchain,
        fragment_source("e4m3", "e4m3", orders=orders), "gfx1201")
    assert lowered.count("vector.load") == expected
    assert "memref.load" in lowered


@pytest.mark.compiler_route
@pytest.mark.parametrize("scale_k,scale_n", [(128,128),(32,1)])
def test_scaled_register_entry_binds_one_wave_geometry(compiler_toolchain,scale_k,scale_n):
    from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape,lower_blockscale
    import re
    program=lower_blockscale(BlockScaleShape(17,19,256,scale_k,scale_n,"nk","bf16"))
    flag="--generate-wmma-gemm-kernel=via-tile=true"
    compiler=compiler_toolchain.require_tessera_opt(flag)
    result=subprocess.run([str(compiler),"-",flag],input=program.tile_ir,
        text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert re.search(r"known_block_size\s*=\s*array<i32:\s*32,\s*1,\s*1>",result.stdout)
