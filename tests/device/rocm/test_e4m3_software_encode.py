"""The software f32 -> E4M3FN encode is bit-identical to RDNA4's instruction.

GFX1151-QUANTIZED-MLP-20261030. gfx1151 (RDNA 3.5) has no V_CVT_PK_FP8_F32,
so `lower-tessera-target-to-rocdl` lowers its E4M3 casts to an integer-only
encode. The same encode can be forced on gfx1201 (`e4m3-encode=software`),
where the hardware instruction is the reference: both images run over every
one of the 2^32 f32 bit patterns and must produce identical bytes.
"""
import ctypes as ct
import subprocess

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_native import _extract_hsaco
from benchmarks.rocm.benchmark_gfx1201_fp8_blockscale import Hip, Launch, memref

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_route]

# The kernel encodes the f32 whose bit pattern is base + global index.
KERNEL = """
module attributes {{tessera.arch = "{arch}"}} {{
  gpu.module @e4m3_mod {{
    gpu.func @e4m3_all(%dst: memref<?xf8E4M3FN>, %base: i32, %n: index) kernel {{
      %bid = gpu.block_id x
      %tid = gpu.thread_id x
      %c256 = arith.constant 256 : index
      %row = arith.muli %bid, %c256 : index
      %i = arith.addi %row, %tid : index
      %ok = arith.cmpi ult, %i, %n : index
      scf.if %ok {{
        %wide = arith.index_cast %i : index to i32
        %bits = arith.addi %base, %wide : i32
        %v = arith.bitcast %bits : i32 to f32
        %e = arith.truncf %v : f32 to f8E4M3FN
        memref.store %e, %dst[%i] : memref<?xf8E4M3FN>
      }}
      gpu.return
    }}
  }}
}}
"""
CHUNK = 1 << 26


def lower(tools, arch, encode):
    text = KERNEL.format(arch=arch)
    flag = f"--pass-pipeline=builtin.module(lower-tessera-target-to-rocdl{{e4m3-encode={encode}}})"
    lowered = subprocess.run([str(tools.require_tessera_opt(flag)), "-", flag], input=text, text=True,
                             capture_output=True, timeout=60)
    assert lowered.returncode == 0, lowered.stderr
    return lowered.stdout


def image(tools, arch, encode):
    pipeline = ("builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
                f"reconcile-unrealized-casts),rocdl-attach-target{{chip={arch}}},gpu-module-to-binary)")
    result = subprocess.run([str(tools.require_mlir_opt()), "--pass-pipeline=" + pipeline],
                            input=lower(tools, arch, encode), text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return _extract_hsaco(result.stdout)


def test_software_encode_matches_v_cvt_pk_fp8_f32_on_every_f32(compiler_toolchain):
    if rt._rocm_live_arch() != "gfx1201":
        pytest.skip("the reference instruction is RDNA4's V_CVT_PK_FP8_F32 (owning gfx1201)")
    assert "rocdl.cvt.pk.fp8.f32" in lower(compiler_toolchain, "gfx1201", "auto")
    assert "rocdl.cvt.pk.fp8.f32" not in lower(compiler_toolchain, "gfx1201", "software")
    hip = Hip()
    images = {encode: image(compiler_toolchain, "gfx1201", encode) for encode in ("auto", "software")}
    outputs = {encode: np.zeros(CHUNK, dtype=np.uint8) for encode in images}
    pointers = {encode: hip.upload(outputs[encode]) for encode in images}
    # Known encodings prove the sweep ran and read real bytes (not a hollow green).
    known = {0x3F800000: 0x38, 0x43E00000: 0x7E, 0xC3E00000: 0xFE, 0x7FC00000: 0xFF,
             0x3B000000: 0x01, 0x00000001: 0x00, 0x80000000: 0x80, 0x7F800000: 0x7F}
    differing = compared = nonzero = 0
    seen = {}
    for chunk in range((1 << 32) // CHUNK):
        base = ct.c_int32(np.int32(np.uint32(chunk * CHUNK)))
        for encode, img in images.items():
            outputs[encode].fill(0xAA)
            hip.check(hip.lib.hipMemcpy(pointers[encode], outputs[encode].ctypes.data_as(ct.c_void_p),
                                        ct.c_size_t(CHUNK), 1))
            Launch(hip, img, "e4m3_all", memref(pointers[encode], CHUNK) + [base, ct.c_int64(CHUNK)],
                   (CHUNK // 256, 1, 1), (256, 1, 1))()
        hip.check(hip.lib.hipDeviceSynchronize())
        got = {encode: hip.download(pointers[encode], outputs[encode]) for encode in images}
        differing += int(np.count_nonzero(got["auto"] != got["software"]))
        compared += got["auto"].size
        nonzero += int(np.count_nonzero(got["software"]))
        for bits in known:
            if chunk * CHUNK <= bits < (chunk + 1) * CHUNK:
                seen[bits] = (int(got["auto"][bits - chunk * CHUNK]), int(got["software"][bits - chunk * CHUNK]))
    assert compared == 1 << 32 and nonzero > (1 << 31)
    assert seen == {bits: (code, code) for bits, code in known.items()}
    assert differing == 0


def test_gfx1151_lowering_emits_no_fp8_instruction(compiler_toolchain):
    """RDNA 3.5 has no FP8 convert: its default lowering is the software encode."""
    lowered = lower(compiler_toolchain, "gfx1151", "auto")
    assert "rocdl.cvt.pk.fp8" not in lowered and "arith.truncf" not in lowered
    assert "arith.bitcast" in lowered
