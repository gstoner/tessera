"""Stage D — RDNA WMMA matrix-core GEMM executes through the C-ABI bridge.

Builds on Stage C (`test_runtime_abi_rocm_launch_bridge.py`): instead of a naive
scalar GEMM, the launched kernel uses the real RDNA 3.5 (gfx1151 / gfx1100 under
WSL) **WMMA matrix instruction** — `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`,
the same `v_wmma_f32_16x16x16_f16` that `rocdl_emit.py` emits — to compute a
16×16×16 `f32 ← f16` GEMM, and execute-and-compares it against a host reference.

This is the first real **`backend_kernel` execution** for a Tessera matmul on
non-Apple silicon: the WMMA op runs through `tsrLaunchKernel` and matches the
oracle. Per the bring-up plan we bring up the `f32←f16` combo first (bf16 has
documented gfx115x bugs). The operand/accumulator fragment layout matches the
grounded mapping in `rocdl_emit.py` (col = lane&15, row = 2·e + lane>>4).

Skip-clean: no hipcc / no built runtime lib / no usable GPU (HIP probe).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._support.rocm_build import tessera_runtime_lib_path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_INCLUDE = REPO_ROOT / "src" / "runtime" / "include"


def _hipcc() -> str | None:
    return shutil.which("hipcc") or (
        "/opt/rocm/bin/hipcc" if Path("/opt/rocm/bin/hipcc").is_file() else None)


_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

typedef float  float8 __attribute__((ext_vector_type(8)));

// One 16x16x16 WMMA tile: D[16x16] = A[16x16] @ B[16x16], row-major, f16 in /
// f32 acc, one 32-lane wave. The fragment layout is the architecture's, not
// ours -- the same two layouts `tessera_rocm_gemm.cpp` specializes at HIPRTC
// time (docs/reference/isa/rdna, RDNA3.5 vs RDNA4 §7.12.2):
//   gfx11 (RDNA3/3.5): each lane holds all 16 K values of its A row / B column
//     (duplicated across the two half-waves); output row = 2*e + (lane>>4).
//   gfx12 (RDNA4): each half-wave owns one K half, so a lane holds 8 values at
//     k = 8*(lane>>4) + i, the builtin is the `_gfx12` one, and the output rows
//     are contiguous per half-wave: row = 8*(lane>>4) + e.
// hipcc compiles for the host's own GPU, so the choice is made per device pass;
// before this the harness was gfx11-only and failed to compile on gfx1201
// ("needs target feature wmma-256b-insts").
#if defined(__gfx1200__) || defined(__gfx1201__)
typedef __fp16 halfk __attribute__((ext_vector_type(8)));
#define TSR_K_PER_LANE 8
#define TSR_WMMA_F16 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
#else
typedef __fp16 halfk __attribute__((ext_vector_type(16)));
#define TSR_K_PER_LANE 16
#define TSR_WMMA_F16 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
#endif
__global__ void tessera_rocm_wmma_gemm_f16_kernel(const __fp16* A, const __fp16* B,
                                                  float* D) {
  int l = threadIdx.x, lane = l & 15, half = l >> 4;
  const int k0 = (TSR_K_PER_LANE == 8) ? 8 * half : 0;
  halfk a, b; float8 c = {0,0,0,0,0,0,0,0};
  for (int i = 0; i < TSR_K_PER_LANE; ++i) {
    a[i] = A[lane*16 + k0 + i]; b[i] = B[(k0 + i)*16 + lane];
  }
  c = TSR_WMMA_F16(a, b, c);
  for (int e = 0; e < 8; ++e) {
    int r = (TSR_K_PER_LANE == 8) ? 8 * half + e : e*2 + half;
    D[r*16 + lane] = c[e];
  }
}

// Launcher: maps (rocm*, tessera_rocm_wmma_gemm_f16) to the WMMA launch.
// buffers = {A(f16), B(f16), D(f32)}; dims = {M,N,K} (this kernel: 16,16,16).
static TsrStatus rocm_wmma_launcher(const char* target, const char* name,
                                    const tsrGpuLaunchParams* p, void*) {
  if (std::strncmp(target, "rocm", 4) != 0) return TSR_STATUS_NOT_FOUND;
  if (std::strcmp(name, "tessera_rocm_wmma_gemm_f16") != 0) return TSR_STATUS_NOT_FOUND;
  if (p->num_buffers != 3 || p->num_dims != 3) return TSR_STATUS_INVALID_ARGUMENT;
  if (p->dims[0] != 16 || p->dims[1] != 16 || p->dims[2] != 16)
    return TSR_STATUS_INVALID_ARGUMENT;  // this proof is a single 16x16x16 tile
  const __fp16* A = (const __fp16*)p->buffers[0];
  const __fp16* B = (const __fp16*)p->buffers[1];
  float* D = (float*)p->buffers[2];
  size_t sh = 16 * 16 * sizeof(__fp16), sf = 16 * 16 * sizeof(float);
  __fp16 *dA = nullptr, *dB = nullptr; float *dD = nullptr;
  if (hipMalloc(&dA, sh) != hipSuccess) return TSR_STATUS_INTERNAL;
  if (hipMalloc(&dB, sh) != hipSuccess) { hipFree(dA); return TSR_STATUS_INTERNAL; }
  if (hipMalloc(&dD, sf) != hipSuccess) { hipFree(dA); hipFree(dB); return TSR_STATUS_INTERNAL; }
  TsrStatus rc = TSR_STATUS_SUCCESS;
  do {
    if (hipMemcpy(dA, A, sh, hipMemcpyHostToDevice) != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
    if (hipMemcpy(dB, B, sh, hipMemcpyHostToDevice) != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
    tessera_rocm_wmma_gemm_f16_kernel<<<dim3(1), dim3(32)>>>(dA, dB, dD);
    if (hipGetLastError() != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
    if (hipDeviceSynchronize() != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
    if (hipMemcpy(D, dD, sf, hipMemcpyDeviceToHost) != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
  } while (0);
  hipFree(dA); hipFree(dB); hipFree(dD);
  return rc;
}

static bool gpu_usable() {
  float* d = nullptr;
  if (hipMalloc(&d, sizeof(float)) != hipSuccess) return false;
  float z = 0.0f;
  bool ok = hipMemcpy(d, &z, sizeof(float), hipMemcpyHostToDevice) == hipSuccess
            && hipDeviceSynchronize() == hipSuccess;
  hipFree(d);
  return ok;
}

int main() {
  if (!gpu_usable()) { std::printf("SKIP_NO_DEVICE\n"); return 0; }
  if (tsrRegisterGpuLauncher(rocm_wmma_launcher, nullptr) != TSR_STATUS_SUCCESS) return 5;
  if (tsrInit() != TSR_STATUS_SUCCESS) return 6;
  tsrDevice dev = nullptr; if (tsrGetDevice(0, &dev) != TSR_STATUS_SUCCESS) return 7;
  tsrStream s = nullptr;   if (tsrCreateStream(dev, &s) != TSR_STATUS_SUCCESS) return 8;

  tsrCompileOptions opt{}; opt.target = "rocm_%CHIP%";
  tsrArtifact art = nullptr;
  if (tsrCompileArtifact("tessera_rocm_wmma_gemm_f16", &opt, &art) != TSR_STATUS_SUCCESS) return 9;
  tsrKernel k = nullptr;
  if (tsrGetKernel(art, "tessera_rocm_wmma_gemm_f16", &k) != TSR_STATUS_SUCCESS) return 10;

  const int N = 16;
  __fp16 A[N*N], B[N*N]; float D[N*N], ref[N*N];
  for (int i = 0; i < N*N; ++i) { A[i] = (__fp16)(((i % 7) - 3) * 0.1f);
                                  B[i] = (__fp16)(((i % 5) - 2) * 0.25f); }
  for (int m = 0; m < N; ++m) for (int n = 0; n < N; ++n) {
    float sacc = 0.0f;
    for (int kk = 0; kk < N; ++kk) sacc += (float)A[m*N+kk] * (float)B[kk*N+n];
    ref[m*N+n] = sacc;
  }
  std::memset(D, 0, sizeof(D));
  void* bufs[3] = { A, B, D };
  int64_t dims[3] = { N, N, N };
  tsrGpuLaunchParams p{}; p.buffers = bufs; p.num_buffers = 3; p.dims = dims; p.num_dims = 3;
  void* args[1] = { &p };
  TsrStatus st = tsrLaunchKernel(s, k, args, 1);
  if (st != TSR_STATUS_SUCCESS) { std::fprintf(stderr, "launch=%d\n", (int)st); return 11; }

  float maxerr = 0.0f;
  for (int i = 0; i < N*N; ++i) { float d = std::fabs(D[i] - ref[i]); if (d > maxerr) maxerr = d; }
  if (maxerr > 1e-2f) { std::fprintf(stderr, "WMMA maxerr=%g too high\n", maxerr); return 12; }

  // Negative: unregistered kernel still reports UNIMPLEMENTED.
  tsrArtifact art2 = nullptr;
  if (tsrCompileArtifact("not_a_real_kernel", &opt, &art2) != TSR_STATUS_SUCCESS) return 13;
  tsrKernel k2 = nullptr;
  if (tsrGetKernel(art2, "not_a_real_kernel", &k2) != TSR_STATUS_SUCCESS) return 14;
  if (tsrLaunchKernel(s, k2, args, 1) != TSR_STATUS_UNIMPLEMENTED) return 15;

  std::printf("OK maxerr=%.3g\n", maxerr);
  tsrDestroyKernel(k2); tsrDestroyArtifact(art2);
  tsrDestroyKernel(k); tsrDestroyArtifact(art);
  tsrDestroyStream(s); tsrShutdown();
  return 0;
}
"""


def test_rocm_wmma_gemm_executes_and_compares_through_bridge(tmp_path):
    hipcc = _hipcc()
    if hipcc is None:
        pytest.skip("hipcc (ROCm) not found")
    runtime_lib = tessera_runtime_lib_path()
    if runtime_lib is None:
        pytest.skip("build libtessera_runtime.a (ninja -C build tessera_runtime)")

    src = tmp_path / "wmma_exec.cpp"
    obj = tmp_path / "wmma_exec.o"
    binp = tmp_path / "wmma_exec"
    chip = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
    src.write_text(_HARNESS.replace("%CHIP%", chip))
    # Two-step: hipcc's clang driver tries to compile a `.a` passed positionally.
    r = subprocess.run([hipcc, "-std=c++17", "-O2", "-I", str(RUNTIME_INCLUDE),
                        "-c", str(src), "-o", str(obj)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"hipcc compile failed:\n{r.stderr[:4000]}"
    r = subprocess.run([hipcc, str(obj), str(runtime_lib), "-lpthread", "-o", str(binp)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"hipcc link failed:\n{r.stderr[:4000]}"

    r = subprocess.run([str(binp)], capture_output=True, text=True, timeout=120)
    out = r.stdout.strip()
    if out == "SKIP_NO_DEVICE":
        pytest.skip("no usable AMD GPU (HIP probe failed)")
    assert r.returncode == 0, (
        f"harness exit {r.returncode} (non-zero = failing step)\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")
    assert out.startswith("OK"), r.stdout
