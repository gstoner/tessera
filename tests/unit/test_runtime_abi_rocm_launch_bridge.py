"""Stage C — ROCm/HIP C-ABI GPU launch bridge (first non-Apple real launch).

The ROCm analog of ``test_runtime_abi_gpu_launch_bridge.py`` (the Apple G7
proof). A backend registers a ``tsrGpuLauncherFn`` mapping a kernel NAME on a
``rocm`` target to a native HIP launch; ``tsrLaunchKernel`` routes the GPU
artifact kernel to it. Here a hipcc-compiled harness:

  * registers a launcher that runs a real ``__global__`` GEMM on the AMD GPU
    (hipMalloc / H2D / launch / sync / D2H) over the params' buffers + dims,
  * compiles a ``rocm`` artifact for that kernel, launches it through
    ``tsrLaunchKernel``, and verifies the GPU output equals ``A @ B``,
  * confirms an unregistered kernel name still reports ``UNIMPLEMENTED`` (the
    bridge never silently succeeds).

This is the **first non-Apple kernel to execute through Tessera's C-ABI launch
bridge** — Strix Halo bring-up Stage C (see
``docs/audit/backend/rocm/STRIX_HALO_EXECUTION_PLAN.md``).

Skip-clean: no hipcc / no built runtime lib / no usable GPU (the harness probes
HIP and prints ``SKIP_NO_DEVICE`` — robust to the WSL quirk where
``hipGetDeviceCount`` reports 0 yet launches still work).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_LIB = REPO_ROOT / "build" / "src" / "runtime" / "libtessera_runtime.a"
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

// One thread per output element — a correct (not fast) GEMM, enough to prove the
// launch bridge executes a real kernel and the result matches A @ B.
__global__ void tessera_rocm_gemm_f32_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m < M && n < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
    C[m * N + n] = acc;
  }
}

// Backend GPU launcher: maps (rocm*, tessera_rocm_gemm_f32) to a real HIP launch
// over the params' host buffers {A,B,C} and dims {M,N,K}.
static TsrStatus rocm_launcher(const char* target, const char* name,
                               const tsrGpuLaunchParams* p, void*) {
  if (std::strncmp(target, "rocm", 4) != 0) return TSR_STATUS_NOT_FOUND;
  if (std::strcmp(name, "tessera_rocm_gemm_f32") != 0) return TSR_STATUS_NOT_FOUND;
  if (p->num_buffers != 3 || p->num_dims != 3) return TSR_STATUS_INVALID_ARGUMENT;
  const float* A = (const float*)p->buffers[0];
  const float* B = (const float*)p->buffers[1];
  float* C = (float*)p->buffers[2];
  int M = (int)p->dims[0], N = (int)p->dims[1], K = (int)p->dims[2];
  size_t sA = (size_t)M * K * sizeof(float);
  size_t sB = (size_t)K * N * sizeof(float);
  size_t sC = (size_t)M * N * sizeof(float);
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (hipMalloc(&dA, sA) != hipSuccess) return TSR_STATUS_INTERNAL;
  if (hipMalloc(&dB, sB) != hipSuccess) { hipFree(dA); return TSR_STATUS_INTERNAL; }
  if (hipMalloc(&dC, sC) != hipSuccess) { hipFree(dA); hipFree(dB); return TSR_STATUS_INTERNAL; }
  TsrStatus rc = TSR_STATUS_SUCCESS;
  do {
    if (hipMemcpy(dA, A, sA, hipMemcpyHostToDevice) != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
    if (hipMemcpy(dB, B, sB, hipMemcpyHostToDevice) != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    tessera_rocm_gemm_f32_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    if (hipGetLastError() != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
    if (hipDeviceSynchronize() != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
    if (hipMemcpy(C, dC, sC, hipMemcpyDeviceToHost) != hipSuccess) { rc = TSR_STATUS_INTERNAL; break; }
  } while (0);
  hipFree(dA); hipFree(dB); hipFree(dC);
  return rc;
}

// Probe whether a usable GPU exists. hipGetDeviceCount is unreliable under WSL
// (reports 0 while launches still work), so probe with a real malloc + 1-thread
// launch + sync round-trip instead.
static bool gpu_usable() {
  float* d = nullptr;
  if (hipMalloc(&d, sizeof(float)) != hipSuccess) return false;
  float zero = 0.0f;
  bool ok = hipMemcpy(d, &zero, sizeof(float), hipMemcpyHostToDevice) == hipSuccess
            && hipDeviceSynchronize() == hipSuccess;
  hipFree(d);
  return ok;
}

int main() {
  if (!gpu_usable()) { std::printf("SKIP_NO_DEVICE\n"); return 0; }

  if (tsrRegisterGpuLauncher(rocm_launcher, nullptr) != TSR_STATUS_SUCCESS) return 5;
  if (tsrInit() != TSR_STATUS_SUCCESS) return 6;
  tsrDevice dev = nullptr; if (tsrGetDevice(0, &dev) != TSR_STATUS_SUCCESS) return 7;
  tsrStream s = nullptr;   if (tsrCreateStream(dev, &s) != TSR_STATUS_SUCCESS) return 8;

  tsrCompileOptions opt{}; opt.target = "rocm";
  tsrArtifact art = nullptr;
  if (tsrCompileArtifact("tessera_rocm_gemm_f32", &opt, &art) != TSR_STATUS_SUCCESS) return 9;
  tsrKernel k = nullptr;
  if (tsrGetKernel(art, "tessera_rocm_gemm_f32", &k) != TSR_STATUS_SUCCESS) return 10;

  const int M = 4, N = 3, K = 5;
  float A[M * K], B[K * N], C[M * N];
  for (int i = 0; i < M * K; ++i) A[i] = ((i % 7) - 3) * 0.1f;
  for (int i = 0; i < K * N; ++i) B[i] = ((i % 5) - 2) * 0.2f;
  std::memset(C, 0, sizeof(C));
  void* bufs[3] = { A, B, C };
  int64_t dims[3] = { M, N, K };
  tsrGpuLaunchParams p{}; p.buffers = bufs; p.num_buffers = 3; p.dims = dims; p.num_dims = 3;
  void* args[1] = { &p };
  TsrStatus st = tsrLaunchKernel(s, k, args, 1);
  if (st != TSR_STATUS_SUCCESS) { std::fprintf(stderr, "launch=%d\n", (int)st); return 11; }

  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    float ref = 0.0f;
    for (int kk = 0; kk < K; ++kk) ref += A[m * K + kk] * B[kk * N + n];
    if (std::fabs(C[m * N + n] - ref) > 1e-3f) {
      std::fprintf(stderr, "mismatch %d,%d: %f vs %f\n", m, n, C[m * N + n], ref);
      return 12;
    }
  }

  // Negative: a kernel the launcher does not recognize still reports
  // UNIMPLEMENTED — the bridge never silently succeeds.
  tsrArtifact art2 = nullptr;
  if (tsrCompileArtifact("not_a_real_kernel", &opt, &art2) != TSR_STATUS_SUCCESS) return 13;
  tsrKernel k2 = nullptr;
  if (tsrGetKernel(art2, "not_a_real_kernel", &k2) != TSR_STATUS_SUCCESS) return 14;
  if (tsrLaunchKernel(s, k2, args, 1) != TSR_STATUS_UNIMPLEMENTED) return 15;

  std::printf("OK\n");
  tsrDestroyKernel(k2); tsrDestroyArtifact(art2);
  tsrDestroyKernel(k); tsrDestroyArtifact(art);
  tsrDestroyStream(s); tsrShutdown();
  return 0;
}
"""


def test_rocm_launch_bridge_runs_gemm_through_c_abi(tmp_path):
    hipcc = _hipcc()
    if hipcc is None:
        pytest.skip("hipcc (ROCm) not found")
    if not RUNTIME_LIB.is_file():
        pytest.skip("build libtessera_runtime.a (ninja -C build tessera_runtime)")

    src_path = tmp_path / "rocm_launch_bridge.cpp"
    obj_path = tmp_path / "rocm_launch_bridge.o"
    bin_path = tmp_path / "rocm_launch_bridge"
    src_path.write_text(_HARNESS)
    # Two-step: compile then link. hipcc's clang driver tries to *compile* a `.a`
    # passed as a positional arg, so compile to an object first, then link.
    r = subprocess.run(
        [hipcc, "-std=c++17", "-O2", "-I", str(RUNTIME_INCLUDE),
         "-c", str(src_path), "-o", str(obj_path)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"hipcc compile failed:\n{r.stderr[:4000]}"
    r = subprocess.run(
        [hipcc, str(obj_path), str(RUNTIME_LIB), "-lpthread", "-o", str(bin_path)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"hipcc link failed:\n{r.stderr[:4000]}"

    r = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=120)
    out = r.stdout.strip()
    if out == "SKIP_NO_DEVICE":
        pytest.skip("no usable AMD GPU (HIP probe failed)")
    assert r.returncode == 0, (
        f"harness exit {r.returncode} (non-zero = failing step)\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")
    assert out == "OK", r.stdout
