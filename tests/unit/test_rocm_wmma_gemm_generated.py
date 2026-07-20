"""Stage K (step 2) — the compiler GENERATES the executing WMMA GEMM.

The milestone: "the Tessera compiler, not a hand-written kernel, produced the
executing GEMM." A `tessera_rocm.wmma_gemm` matmul directive is expanded by the
`generate-wmma-gemm-kernel` pass into a fragment-materialized RDNA WMMA kernel
(fragment loads + `tessera_rocm.wmma` + accumulator stores — the kernel BODY is
emitted by the pass, not authored), which then lowers through Stage J (real
`rocdl.wmma`) and Stage I (→ hsaco) and executes:

    "tessera_rocm.wmma_gemm"{m=16,n=16,k=16}
      --(tessera-opt --generate-wmma-gemm-kernel)--> gpu.func{frag loads + wmma + stores}
      --(--lower-tessera-target-to-rocdl  [Stage J])--> rocdl.wmma
      --(mlir-opt: convert-scf-to-cf, convert-gpu-to-rocdl, reconcile,
         rocdl-attach-target{gfx1151}, gpu-module-to-binary  [Stage I])--> hsaco
      --(hipModuleLoadData + launch)--> executes

The result is compared to BOTH numpy AND the shipped `device_verified_abi`
hand-written kernel (`tessera_rocm_wmma_gemm_f16`), the on-silicon oracle. On
gfx1151: vs numpy ~2e-7, **vs the oracle 0.0 (bit-identical)** — same instruction,
layout, and accumulation order, but compiler-generated.

`test_rocm_wmma_gemm_via_mlir.py` is the companion that locks the target IR with a
hand-authored kernel; this test proves the pass reproduces it.

Skip-clean: tessera-opt / mlir-opt / oracle lib not built, or no usable AMD GPU.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"
ORACLE_LIB = (REPO / "build" / "src" / "compiler" / "codegen"
              / "Tessera_ROCM_Backend" / "runtime" / "hip"
              / "libtessera_rocm_gemm.so")
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
TYPED_FRAGMENT_FIXTURE = (
    REPO / "src/compiler/codegen/Tessera_ROCM_Backend/test/rocm"
    / "gfx1151_tile_fragment_store.mlir"
)

_DIRECTIVE = '''
module {
  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : i64, n = 16 : i64, k = 16 : i64} : () -> ()
}
'''

_PORTABLE_TILE_KERNEL = '''
module {
  func.func @gemm(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                  %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}
'''

_PORTABLE_TILE_EPILOGUE = '''
module {
  func.func @gemm_epilogue(%a: !llvm.ptr, %b: !llvm.ptr, %bias: !llvm.ptr,
                           %d: !llvm.ptr, %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %bias, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = true, activation = "silu", output = "f16">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}
'''


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-23/bin/mlir-opt", "/opt/homebrew/opt/llvm@23/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    return shutil.which("mlir-opt")


def _hip():
    rocm_lib = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(rocm_lib, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    try:
        return ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None


def _extract_hsaco(text: str) -> bytes:
    i = text.index('bin = "') + len('bin = "')
    out = bytearray()
    j = i
    hexd = "0123456789abcdefABCDEF"
    simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
    while j < len(text):
        c = text[j]
        if c == '"':
            break
        if c == "\\":
            nx = text[j + 1:j + 3]
            if len(nx) == 2 and nx[0] in hexd and nx[1] in hexd:
                out.append(int(nx, 16)); j += 3; continue
            if text[j + 1] in simple:
                out.append(simple[text[j + 1]]); j += 2; continue
        out.append(ord(c)); j += 1
    return bytes(out)


def _need_tools():
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    mo = _find_mlir_opt()
    if mo is None:
        pytest.skip("mlir-opt not found (set TESSERA_MLIR_OPT or install LLVM 23)")
    return mo


def test_generate_pass_emits_fragment_materialized_kernel():
    """No GPU needed: the pass turns the directive into a gpu.func whose body is
    the problem-size-generic, fragment-materialized WMMA kernel — a runtime
    (M,N,K) signature, an scf.for K-loop, masked fragment loads + tessera_rocm.
    wmma + masked stores."""
    _need_tools()
    r = subprocess.run([str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel"],
                       input=_DIRECTIVE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # Problem-size-generic ABI: dynamic memrefs + runtime M,N,K index args.
    assert "gpu.func @gemm" in out and "kernel" in out
    assert "memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf32>" in out
    assert "index, %arg4: index, %arg5: index" in out
    # Dispatch: tileFull ? fast : edge, each = aligned main K-loop + a masked
    # tail panel (scf.if needTail) for ragged K. So two scf.for K-loops (fast
    # main, edge main); each path carries a wmma in the main loop and one in the
    # tail panel -> four tessera_rocm.wmma total.
    #   - fast — interior: contiguous vector.load A, unmasked B, unmasked store
    #   - edge — ragged M/N: coalesced loads at clamped row/col + a vector
    #            arith.select to zero OOB fragments; masked stores
    #   - tail — ragged K (K%16!=0): per-element clamp-and-select, run once
    assert "vector.load" in out                 # coalesced A fragment (fast+edge)
    assert "arith.select" in out                 # edge zeroing + tail/clamp masks
    assert out.count("scf.for") == 2            # fast-main + edge-main K-loops
    assert out.count("tessera_rocm.wmma") == 4  # 2 paths x (main + tail panel)
    assert "scf.if" in out                       # tileFull dispatch + ragged-K tail
    assert '"tessera_rocm.wmma_gemm"' not in out   # directive consumed


def test_portable_tile_kernel_reuses_fragment_materialized_generator():
    """The launch-level portable contract reaches the identical production
    gfx1151 generator rather than a second backend-specific kernel body."""
    _need_tools()
    r = subprocess.run([str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel"],
                       input=_PORTABLE_TILE_KERNEL, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "tile.matmul_kernel" not in r.stdout
    assert "gpu.func @gemm" in r.stdout
    assert "vector.load" in r.stdout
    assert "tessera_rocm.wmma" in r.stdout
    assert "memref.store" in r.stdout


def test_portable_tile_epilogue_preserves_abi_and_output_conversion():
    """Bias remains the third portable operand, while SiLU and f16 conversion
    are fused on the register accumulator before the final store."""
    _need_tools()
    r = subprocess.run([str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel"],
                       input=_PORTABLE_TILE_EPILOGUE,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert ("gpu.func @gemm_epilogue(%arg0: memref<?xf16>, "
            "%arg1: memref<?xf16>, %arg2: memref<?xf32>, "
            "%arg3: memref<?xf16>") in out
    assert "arith.addf" in out       # bias
    # SiLU uses the shared bounded-tanh arithmetic form, avoiding a target
    # device-math libcall while preserving the portable epilogue contract.
    assert "arith.maximumf" in out
    assert "arith.minimumf" in out
    assert "arith.divf" in out
    assert "arith.truncf" in out     # f32 accumulator -> f16 output
    # One hoisted bias load in each mutually exclusive fast/edge branch. Before
    # the cleanup this was emitted once per accumulator element (16 total).
    assert out.count("memref.load %arg2") == 2
    assert "tile.fragment_contract" not in out


@pytest.mark.parametrize("dtype", ["f16", "bf16", "int8", "int4"])
def test_typed_tile_fragment_fixture_executes_and_matches_numpy(dtype):
    """The literal view -> pack -> zero -> MMA -> unpack -> store chain reaches
    ROCDL, assembles for gfx1151, launches, and writes the logical 16x16 tile."""
    mlir_opt = _need_tools()
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    fixture = TYPED_FRAGMENT_FIXTURE.read_text()
    storage_dtype = np.float16
    if dtype == "bf16":
        storage_dtype = pytest.importorskip("ml_dtypes").bfloat16
        fixture = fixture.replace("f16", "bf16")
    elif dtype in ("int8", "int4"):
        fixture = fixture.replace("memref<256xf16>", "memref<256xi8>")
        fixture = fixture.replace("memref<256xf32>", "memref<256xi32>")
        fixture = fixture.replace(
            'a = "f16", b = "f16", acc = "f32"',
            f'a = "{dtype}", b = "{dtype}", acc = "i32"')
        storage_dtype = np.int8
    lowered = subprocess.run(
        [str(TESSERA_OPT), "-", "--allow-unregistered-dialect",
         "--pass-pipeline=builtin.module(lower-tile-to-rocm{arch=gfx1151},"
         "lower-tessera-target-to-rocdl)"],
        input=fixture, capture_output=True, text=True)
    assert lowered.returncode == 0, lowered.stderr
    intrinsic = (f"rocdl.wmma.f32.16x16x16.{dtype}"
                 if dtype in ("f16", "bf16")
                 else f"rocdl.wmma.i32.16x16x16.iu{dtype[-1]}")
    assert intrinsic in lowered.stdout
    pipeline = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )
    serialized = subprocess.run(
        [mlir_opt, f"--pass-pipeline={pipeline}"], input=lowered.stdout,
        capture_output=True, text=True)
    assert serialized.returncode == 0, serialized.stderr
    hsaco = _extract_hsaco(serialized.stdout)

    rng = np.random.default_rng(1151)
    if dtype in ("int8", "int4"):
        low, high = (-8, 8) if dtype == "int4" else (-16, 17)
        a = rng.integers(low, high, size=(16, 16), dtype=np.int8)
        b = np.asfortranarray(
            rng.integers(low, high, size=(16, 16), dtype=np.int8))
    else:
        a = (rng.standard_normal((16, 16)) * 0.4).astype(storage_dtype)
        b = np.asfortranarray(
            (rng.standard_normal((16, 16)) * 0.4).astype(storage_dtype))
    if hip.hipInit(0) != 0:
        pytest.skip("hipInit failed")
    module = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), hsaco) != 0:
        pytest.skip("no usable gfx1151 device (module load failed)")
    function = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(
        ctypes.byref(function), module, b"fragment_store") == 0
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for ptr, nbytes in ((da, a.nbytes), (db, b.nbytes), (dd, 16 * 16 * 4)):
        assert hip.hipMalloc(ctypes.byref(ptr), nbytes) == 0
    hip.hipMemcpy(da, a.ctypes.data_as(ctypes.c_void_p), a.nbytes, 1)
    hip.hipMemcpy(db, b.ctypes.data_as(ctypes.c_void_p), b.nbytes, 1)

    def descriptor(ptr, size):
        return [ctypes.c_void_p(ptr.value), ctypes.c_void_p(ptr.value),
                ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]

    args = descriptor(da, 256) + descriptor(db, 256) + descriptor(dd, 256)
    argv = (ctypes.c_void_p * len(args))()
    for i, arg in enumerate(args):
        argv[i] = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    assert launch(function, 1, 1, 1, 32, 1, 1, 0, None, argv, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    output_dtype = np.int32 if dtype in ("int8", "int4") else np.float32
    out = np.zeros((16, 16), output_dtype)
    hip.hipMemcpy(out.ctypes.data_as(ctypes.c_void_p), dd, out.nbytes, 2)
    for ptr in (da, db, dd):
        hip.hipFree(ptr)
    if dtype in ("int8", "int4"):
        np.testing.assert_array_equal(
            out, a.astype(np.int32) @ b.astype(np.int32))
    else:
        tolerance = 2e-3 if dtype == "f16" else 2e-2
        np.testing.assert_allclose(
            out, a.astype(np.float32) @ b.astype(np.float32),
            rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "source", [_DIRECTIVE, _PORTABLE_TILE_KERNEL],
    ids=["target-directive", "portable-tile-kernel"],
)
@pytest.mark.parametrize("shape", [(16, 16, 16), (33, 17, 31)],
                         ids=["aligned", "ragged"])
def test_compiler_generated_gemm_matches_numpy_and_oracle(source, shape):
    mlir_opt = _need_tools()
    hip = _hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")

    # directive -> generate kernel -> Stage J (real rocdl.wmma)
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel",
         "--lower-tessera-target-to-rocdl"],
        input=source, capture_output=True, text=True)
    assert gen.returncode == 0, gen.stderr
    assert "rocdl.wmma.f32.16x16x16.f16" in gen.stdout
    # Stage I: finish-lower + attach + serialize
    pipeline = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )
    ser = subprocess.run([mlir_opt, f"--pass-pipeline={pipeline}"],
                         input=gen.stdout, capture_output=True, text=True)
    assert ser.returncode == 0, ser.stderr
    assert "bin = " in ser.stdout
    hsaco = _extract_hsaco(ser.stdout)
    assert hsaco[:4] == b"\x7fELF"

    rng = np.random.default_rng(7)
    M, N, Kd = shape
    A = (rng.standard_normal((M, Kd)) * 0.4).astype(np.float16)
    B = (rng.standard_normal((Kd, N)) * 0.4).astype(np.float16)

    if hip.hipInit(0) != 0:
        pytest.skip("hipInit failed")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU (module load failed)")
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") == 0
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for d, nbytes in ((da, A.nbytes), (db, B.nbytes),
                      (dd, M * N * np.dtype(np.float32).itemsize)):
        if hip.hipMalloc(ctypes.byref(d), nbytes) != 0:
            pytest.skip("hipMalloc failed")
    hip.hipMemcpy(da, A.ctypes.data_as(ctypes.c_void_p), A.nbytes, 1)
    hip.hipMemcpy(db, B.ctypes.data_as(ctypes.c_void_p), B.nbytes, 1)

    # Problem-size-generic kernel ABI: 3 dynamic-memref descriptors (5 fields:
    # alloc, aligned, offset, size, stride) + runtime M,N,K as i64. The grid
    # is one wave per 16x16 output tile; M=N=K=16 -> grid (1,1,1).
    def mr(p, size):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]

    args = (mr(da, M * Kd) + mr(db, Kd * N) + mr(dd, M * N)
            + [ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(Kd)])
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    gx, gy = (N + 15) // 16, (M + 15) // 16
    assert launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    D = np.zeros((M, N), dtype=np.float32)
    hip.hipMemcpy(D.ctypes.data_as(ctypes.c_void_p), dd, D.nbytes, 2)
    for d in (da, db, dd):
        hip.hipFree(d)

    assert float(np.max(np.abs(D - A.astype(np.float32) @ B.astype(np.float32)))) < 5e-2

    if not ORACLE_LIB.is_file():
        pytest.skip("oracle lib not built: ninja -C build tessera_rocm_gemm")
    lib = ctypes.CDLL(str(ORACLE_LIB), mode=ctypes.RTLD_GLOBAL)
    ofn = lib.tessera_rocm_wmma_gemm_f16
    ofn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    ofn.restype = ctypes.c_int
    Do = np.zeros((M, N), dtype=np.float32)
    orc = ofn(A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
              Do.ctypes.data_as(ctypes.c_void_p), M, N, Kd)
    if orc == 2:
        pytest.skip("oracle: no usable AMD GPU")
    assert orc == 0
    # The compiler-GENERATED GEMM matches the hand-written oracle bit-for-bit.
    assert float(np.max(np.abs(D - Do))) == 0.0, \
        "compiler-generated GEMM != hand-written oracle"


def test_generate_pass_rejects_non_16_tile():
    """Honest scope: only the 16x16x16 tile is implemented — other extents are a
    named error, not a silent wrong kernel."""
    _need_tools()
    bad = _DIRECTIVE.replace("k = 16", "k = 32")
    r = subprocess.run([str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel"],
                       input=bad, capture_output=True, text=True)
    assert r.returncode != 0
    assert "16x16x16" in r.stderr
