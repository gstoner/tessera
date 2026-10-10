"""Native Tile full-K folded epilogue proof; not Graph/Target route closure."""
import ctypes as ct
import os
import ml_dtypes
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import rocm_native
from benchmarks.rocm.benchmark_gfx1201_fp8_blockscale import Hip, Launch, memref

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 device gate")]

SOURCE = """
#l = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>
#ar = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>
#bc = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 0>
!fa = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">
!fb = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">
!fc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
 gpu.module @folded_scale_native_mod {
  gpu.func @folded_scale_native(%a: memref<?xf8E4M3FN>, %b: memref<?xf8E4M3FN>,
     %as: memref<?xf32>, %ref: memref<?xi8>, %o: memref<?xbf16>,
     %m: index, %n: index, %k: index) kernel {
   %c0 = arith.constant 0 : index
   %c16 = arith.constant 16 : index
   %z = tile.fragment_zero : !fc
   %acc = scf.for %ki = %c0 to %k step %c16 iter_args(%p = %z) -> (!fc) {
    %av = tile.view %a, %c0, %ki, %k {tile.layout = #l, tile.memory = #ar} : (memref<?xf8E4M3FN>, index, index, index) -> !tile.tile
    %bv = tile.view %b, %ki, %c0, %k {tile.layout = #l, tile.memory = #bc} : (memref<?xf8E4M3FN>, index, index, index) -> !tile.tile
    %af = tile.fragment_pack %av : (!tile.tile) -> !fa
    %bf = tile.fragment_pack %bv : (!tile.tile) -> !fb
    %r = tile.mma %af, %bf, %p : (!fa, !fb, !fc) -> !fc
    scf.yield %r : !fc
   }
   %scaled = tile.fragment_folded_scale %acc scales(%as, %ref) at(%c0, %c0) bounds(%m, %n) : !fc, memref<?xf32>, memref<?xi8>
   %v = tile.fragment_unpack %scaled {tile.layout = #l} : (!fc) -> !tile.tile
   tile.store %v, %o, %c0, %c0, %m, %n, %n {tile.layout = #l, tile.memory = #ar, tile.epilogue = #tile.epilogue<bias = false, activation = "none", output = "bf16">} : !tile.tile, memref<?xbf16>, index, index, index, index, index
   gpu.return
  }
 }
}
"""

@pytest.mark.parametrize("case", ["normal", "overflow_scale", "underflow_scale", "zero_partial", "zero_reference", "ragged"])
def test_native_folded_scale_matches_wide_oracle(case):
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = (9, 11, 64) if case == "ragged" else (16, 16, 64)
    value = (2.0**-9 if case == "overflow_scale" else 448.0 if case == "underflow_scale"
             else 0.0 if case == "zero_partial" else 1.0)
    a = np.full((16,k), value, dtype=ml_dtypes.float8_e4m3fn)
    b = a.copy()
    scales = np.full(m, 2.0**20 if case in ("overflow_scale","zero_partial")
                     else 2.0**-30 if case == "underflow_scale" else 0.5, np.float32)
    refs = np.full(n, 240 if case in ("overflow_scale","zero_partial")
                   else 1 if case == "underflow_scale" else 0 if case == "zero_reference" else 128, np.uint8)
    target, backend, image, *_ = rocm_native._compile_native_tile_ir(
        SOURCE, directive="gpu.module", family="matmul",
        architecture="gfx1201", schedule_kernel=True)
    assert "tile.fragment_folded_scale" not in backend
    assert "f64" in target and "scf.if" in target
    hip = Hip()
    pointers = [hip.upload(a), hip.upload(b), hip.upload(scales), hip.upload(refs),
                hip.upload(np.full((m,n), -101, ml_dtypes.bfloat16))]
    launch = None
    try:
        args = []
        for pointer, size in zip(pointers, (16*k,16*k,m,n,m*n)):
            args += memref(pointer,size)
        args += [ct.c_int64(m),ct.c_int64(n),ct.c_int64(k)]
        launch = Launch(hip,image,"folded_scale_native",args,(1,1,1),(32,1,1))
        launch()
        got = hip.download(pointers[-1], np.zeros((m,n),ml_dtypes.bfloat16))
        partial = a[:m].astype(np.float64) @ b[:n].astype(np.float64).T
        want = (partial * np.where(refs == 0, 0.0, np.exp2(refs.astype(np.int64)-127))[None,:]
                * scales.astype(np.float64)[:,None]).astype(np.float32).astype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))
    finally:
        if launch is not None:
            hip.check(hip.lib.hipModuleUnload(launch.module))
        for pointer in pointers:
            hip.check(hip.lib.hipFree(pointer))
