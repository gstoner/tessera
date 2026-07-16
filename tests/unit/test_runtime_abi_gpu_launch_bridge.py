"""G7 — C-ABI GPU launch bridge (pluggable launcher hook).

Closes the runtime-audit gap "tsrLaunchKernel returns UNIMPLEMENTED for GPU
kernels". The core runtime stays backend-agnostic: a backend registers a
``tsrGpuLauncherFn`` (mapping a kernel NAME on a target to its native symbol),
and ``tsrLaunchKernel`` routes GPU artifact kernels to it. NVIDIA/ROCm register
their launchers when hardware lights up; here we prove the Apple GPU path
end-to-end on a real Metal device.

Two layers:
  * **Source/contract guard** (runs everywhere) — the launcher hook + params
    struct + routing exist in the header and runtime. Locks the ABI even on
    non-Darwin CI where the harness can't run.
  * **End-to-end** (Darwin + Metal) — a C++ harness dlopens the Apple GPU
    runtime dylib, registers a launcher that calls
    ``tessera_apple_gpu_mps_matmul_f32``, compiles an ``apple_gpu`` artifact for
    that kernel, launches it through ``tsrLaunchKernel``, and verifies the GPU
    output equals ``A @ B``. A second, unregistered kernel name still reports
    ``TSR_STATUS_UNIMPLEMENTED`` (the bridge never silently succeeds).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_LIB = REPO_ROOT / "build" / "src" / "runtime" / "libtessera_runtime.a"
RUNTIME_INCLUDE = REPO_ROOT / "src" / "runtime" / "include"
RUNTIME_SRC = REPO_ROOT / "src" / "runtime" / "src" / "tessera_runtime.cpp"
RUNTIME_HDR = REPO_ROOT / "src" / "runtime" / "include" / "tessera" / "tessera_runtime.h"


# ── contract guard (platform-agnostic) ─────────────────────────────────────── #

def test_gpu_launcher_hook_exists_in_header():
    hdr = RUNTIME_HDR.read_text()
    assert "tsrGpuLauncherFn" in hdr
    assert "tsrRegisterGpuLauncher" in hdr
    types = (RUNTIME_INCLUDE / "tessera" / "tsr_types.h").read_text()
    assert "tsrGpuLaunchParams" in types


def test_launch_kernel_routes_gpu_to_registered_launcher():
    """The GPU branch of tsrLaunchKernel must consult the registered launcher
    AND keep the honest UNIMPLEMENTED fallback when none is registered / the
    launcher declines the kernel."""
    src = RUNTIME_SRC.read_text()
    i = src.find(" tsrLaunchKernel(")
    body = src[src.find("{", i):src.find("\n}", i)]
    assert "g_gpu_launcher" in body, "GPU branch must consult the registered launcher"
    assert "no native C-ABI launch bridge" in body, "must keep the honest fallback"
    assert "tsrRegisterGpuLauncher" in src


# ── end-to-end (Darwin + Metal) ────────────────────────────────────────────── #

def _cxx() -> str | None:
    return shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")


_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cmath>

typedef void (*gemm_fn)(const float*, const float*, float*, int, int, int);
static gemm_fn g_gemm = nullptr;

// A backend GPU launcher: maps (apple_gpu, tessera_apple_gpu_mps_matmul_f32) to
// the dlopen'd Apple runtime symbol and runs it over the params' buffers/dims.
static TsrStatus apple_launcher(const char* target, const char* name,
                                const tsrGpuLaunchParams* p, void*) {
  if (std::strcmp(target, "apple_gpu") != 0) return TSR_STATUS_NOT_FOUND;
  if (std::strcmp(name, "tessera_apple_gpu_mps_matmul_f32") != 0) return TSR_STATUS_NOT_FOUND;
  if (!g_gemm) return TSR_STATUS_UNIMPLEMENTED;
  if (p->num_buffers != 3 || p->num_dims != 3) return TSR_STATUS_INVALID_ARGUMENT;
  g_gemm((const float*)p->buffers[0], (const float*)p->buffers[1],
         (float*)p->buffers[2], (int)p->dims[0], (int)p->dims[1], (int)p->dims[2]);
  return TSR_STATUS_SUCCESS;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: harness <apple_runtime.dylib>\n"); return 2; }
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
  g_gemm = (gemm_fn)dlsym(lib, "tessera_apple_gpu_mps_matmul_f32");
  if (!g_gemm) { std::fprintf(stderr, "dlsym failed\n"); return 4; }

  if (tsrRegisterGpuLauncher(apple_launcher, nullptr) != TSR_STATUS_SUCCESS) return 5;
  if (tsrInit() != TSR_STATUS_SUCCESS) return 6;
  tsrDevice dev = nullptr; if (tsrGetDevice(0, &dev) != TSR_STATUS_SUCCESS) return 7;
  tsrStream s = nullptr;   if (tsrCreateStream(dev, &s) != TSR_STATUS_SUCCESS) return 8;

  tsrCompileOptions opt{}; opt.target = "apple_gpu";
  tsrArtifact art = nullptr;
  if (tsrCompileArtifact("tessera_apple_gpu_mps_matmul_f32", &opt, &art) != TSR_STATUS_SUCCESS) return 9;
  tsrKernel k = nullptr;
  if (tsrGetKernel(art, "tessera_apple_gpu_mps_matmul_f32", &k) != TSR_STATUS_SUCCESS) return 10;

  const int M = 4, N = 3, K = 5;
  float A[M*K], B[K*N], C[M*N];
  for (int i = 0; i < M*K; ++i) A[i] = ((i % 7) - 3) * 0.1f;
  for (int i = 0; i < K*N; ++i) B[i] = ((i % 5) - 2) * 0.2f;
  std::memset(C, 0, sizeof(C));
  void* bufs[3] = { A, B, C };
  int64_t dims[3] = { M, N, K };
  tsrGpuLaunchParams p{}; p.buffers = bufs; p.num_buffers = 3; p.dims = dims; p.num_dims = 3;
  void* args[1] = { &p };
  TsrStatus st = tsrLaunchKernel(s, k, args, 1);
  if (st != TSR_STATUS_SUCCESS) { std::fprintf(stderr, "launch=%d\n", (int)st); return 11; }

  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    float ref = 0.0f;
    for (int kk = 0; kk < K; ++kk) ref += A[m*K+kk] * B[kk*N+n];
    if (std::fabs(C[m*N+n] - ref) > 1e-3f) {
      std::fprintf(stderr, "mismatch %d,%d: %f vs %f\n", m, n, C[m*N+n], ref);
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


@pytest.mark.hardware_apple_gpu
@pytest.mark.integration
def test_gpu_launch_bridge_runs_apple_gemm_through_c_abi(tmp_path):
    assert RUNTIME_LIB.is_file(), "build libtessera_runtime.a before the ABI integration test"
    cxx = _cxx()
    assert cxx is not None, "C++ compiler unavailable on the ABI integration host"
    # Locate the loadable Apple GPU runtime dylib (Python compiles/caches it).
    from tessera import _apple_gpu_backend as agb
    assert agb.is_available(), "Apple GPU runtime unavailable on the hardware test host"
    import tessera.runtime as rt
    dylib = getattr(rt._load_apple_gpu_runtime(), "_name", None)
    assert dylib and os.path.isfile(dylib), "Apple GPU runtime dylib not found"

    src_path = tmp_path / "gpu_launch_bridge.cpp"
    bin_path = tmp_path / "gpu_launch_bridge"
    src_path.write_text(_HARNESS)
    compile_cmd = [cxx, "-std=c++17", "-O2", "-I", str(RUNTIME_INCLUDE),
                   str(src_path), str(RUNTIME_LIB), "-lpthread", "-o", str(bin_path)]
    r = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"compile failed:\n{r.stderr[:4000]}"
    r = subprocess.run([str(bin_path), dylib], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (
        f"harness exit {r.returncode} (non-zero = failing step)\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")
    assert r.stdout.strip() == "OK", r.stdout
