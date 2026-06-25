"""Stage D (NVIDIA) — sm_120 mma.sync GEMM executes through the C-ABI bridge.

The consumer-Blackwell counterpart to ``test_rocm_wmma_execute_compare.py``. The
launched kernel is the **Tessera-emitted PTX** from
``ptx_emit.emit_mma_sync_matmul_ptx`` (NOT a hand-written nvcc kernel): a
warp-level ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` that computes a
16×8×16 ``f32 ← bf16`` GEMM. The C-ABI launcher loads that emitted PTX via the
CUDA Driver API (``cuModuleLoadDataEx`` — runs ptxas in-driver), launches it
through ``tsrLaunchKernel``, and execute-and-compares against a host reference.

This is the first real **`backend_kernel` execution** for a Tessera matmul on
NVIDIA silicon: the emitted PTX runs through the runtime bridge and matches the
oracle, closing the rung ladder emit → assemble → launch → compare on sm_120.

Skip-clean: no nvcc / no built runtime lib / no usable GPU (CUDA probe).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tessera.compiler import ptx_emit as P

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_LIB = REPO_ROOT / "build" / "src" / "runtime" / "libtessera_runtime.a"
RUNTIME_INCLUDE = REPO_ROOT / "src" / "runtime" / "include"


def _nvcc() -> str | None:
    return shutil.which("nvcc") or (
        "/usr/local/cuda/bin/nvcc"
        if Path("/usr/local/cuda/bin/nvcc").is_file()
        else None
    )


# The launcher loads the emitted PTX (path passed as argv[1]) via the Driver API
# and runs it through tsrLaunchKernel. buffers = {A(bf16), B(bf16), D(f32)};
# dims = {M,N,K} (this kernel: 16,8,16).
_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#define M 16
#define N 8
#define K 16

static const char* g_ptx_path = nullptr;
static CUfunction  g_fn = nullptr;
static bool        g_loaded = false;

// bf16 round-to-nearest-even, packed in uint16 (host side, no SDK dependency).
static unsigned short f2bf16(float f){
  unsigned int x; std::memcpy(&x,&f,4);
  unsigned int r=(x>>16)&1u, t=0x7fffu+r; x+=t; return (unsigned short)(x>>16);
}
static float bf162f(unsigned short h){ unsigned int x=((unsigned int)h)<<16; float f; std::memcpy(&f,&x,4); return f; }

static bool ensure_module() {
  if (g_loaded) return g_fn != nullptr;
  g_loaded = true;
  FILE* fp = std::fopen(g_ptx_path, "rb");
  if (!fp) return false;
  std::fseek(fp,0,SEEK_END); long sz=std::ftell(fp); std::fseek(fp,0,SEEK_SET);
  std::vector<char> ptx(sz+1); if (std::fread(ptx.data(),1,sz,fp)!=(size_t)sz){std::fclose(fp);return false;} ptx[sz]=0; std::fclose(fp);
  char log[8192]; log[0]=0;
  CUjit_option o[]={CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* ov[]={(void*)log,(void*)(size_t)sizeof(log)};
  CUmodule mod;
  if (cuModuleLoadDataEx(&mod, ptx.data(), 2, o, ov) != CUDA_SUCCESS) {
    std::fprintf(stderr,"cuModuleLoadDataEx: %s\n", log); return false;
  }
  if (cuModuleGetFunction(&g_fn, mod, "tessera_mma_m16n8k16_bf16") != CUDA_SUCCESS) return false;
  return true;
}

static TsrStatus nvidia_mma_launcher(const char* target, const char* name,
                                     const tsrGpuLaunchParams* p, void*) {
  if (std::strncmp(target, "nvidia", 6) != 0) return TSR_STATUS_NOT_FOUND;
  if (std::strcmp(name, "tessera_mma_m16n8k16_bf16") != 0) return TSR_STATUS_NOT_FOUND;
  if (p->num_buffers != 3 || p->num_dims != 3) return TSR_STATUS_INVALID_ARGUMENT;
  if (p->dims[0] != M || p->dims[1] != N || p->dims[2] != K)
    return TSR_STATUS_INVALID_ARGUMENT;   // this proof is a single 16x8x16 tile
  if (!ensure_module()) return TSR_STATUS_INTERNAL;

  const unsigned short* A = (const unsigned short*)p->buffers[0];
  const unsigned short* B = (const unsigned short*)p->buffers[1];
  float* D = (float*)p->buffers[2];
  size_t sA=M*K*2, sB=K*N*2, sD=M*N*4;
  CUdeviceptr dA=0,dB=0,dD=0;
  if (cuMemAlloc(&dA,sA)!=CUDA_SUCCESS) return TSR_STATUS_INTERNAL;
  if (cuMemAlloc(&dB,sB)!=CUDA_SUCCESS){cuMemFree(dA);return TSR_STATUS_INTERNAL;}
  if (cuMemAlloc(&dD,sD)!=CUDA_SUCCESS){cuMemFree(dA);cuMemFree(dB);return TSR_STATUS_INTERNAL;}
  TsrStatus rc = TSR_STATUS_SUCCESS;
  do {
    if (cuMemcpyHtoD(dA,A,sA)!=CUDA_SUCCESS){rc=TSR_STATUS_INTERNAL;break;}
    if (cuMemcpyHtoD(dB,B,sB)!=CUDA_SUCCESS){rc=TSR_STATUS_INTERNAL;break;}
    void* args[]={&dA,&dB,&dD};
    if (cuLaunchKernel(g_fn, 1,1,1, 32,1,1, 0, 0, args, 0)!=CUDA_SUCCESS){rc=TSR_STATUS_INTERNAL;break;}
    if (cuCtxSynchronize()!=CUDA_SUCCESS){rc=TSR_STATUS_INTERNAL;break;}
    if (cuMemcpyDtoH(D,dD,sD)!=CUDA_SUCCESS){rc=TSR_STATUS_INTERNAL;break;}
  } while(0);
  cuMemFree(dA); cuMemFree(dB); cuMemFree(dD);
  return rc;
}

static bool gpu_usable() {
  if (cuInit(0)!=CUDA_SUCCESS) return false;
  int n=0; if (cuDeviceGetCount(&n)!=CUDA_SUCCESS || n<1) return false;
  CUdevice dev; if (cuDeviceGet(&dev,0)!=CUDA_SUCCESS) return false;
  CUcontext ctx; if (cuDevicePrimaryCtxRetain(&ctx,dev)!=CUDA_SUCCESS) return false;
  if (cuCtxSetCurrent(ctx)!=CUDA_SUCCESS) return false;
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr,"usage: %s <ptx>\n", argv[0]); return 2; }
  g_ptx_path = argv[1];
  if (!gpu_usable()) { std::printf("SKIP_NO_DEVICE\n"); return 0; }
  if (tsrRegisterGpuLauncher(nvidia_mma_launcher, nullptr) != TSR_STATUS_SUCCESS) return 5;
  if (tsrInit() != TSR_STATUS_SUCCESS) return 6;
  tsrDevice dev=nullptr; if (tsrGetDevice(0,&dev)!=TSR_STATUS_SUCCESS) return 7;
  tsrStream s=nullptr;   if (tsrCreateStream(dev,&s)!=TSR_STATUS_SUCCESS) return 8;

  tsrCompileOptions opt{}; opt.target = "nvidia_sm120";
  tsrArtifact art=nullptr;
  if (tsrCompileArtifact("tessera_mma_m16n8k16_bf16", &opt, &art)!=TSR_STATUS_SUCCESS) return 9;
  tsrKernel k=nullptr;
  if (tsrGetKernel(art, "tessera_mma_m16n8k16_bf16", &k)!=TSR_STATUS_SUCCESS) return 10;

  // A row-major MxK bf16, B col-major KxN bf16, D row-major MxN f32.
  unsigned short A[M*K], B[K*N]; float D[M*N], ref[M*N];
  float fA[M*K], fB[K*N];
  for (int i=0;i<M*K;i++){ float v=(((i*37+11)%201)-100)/100.0f; A[i]=f2bf16(v); fA[i]=bf162f(A[i]); }
  for (int i=0;i<K*N;i++){ float v=(((i*53+7)%201)-100)/100.0f;  B[i]=f2bf16(v); fB[i]=bf162f(B[i]); }
  for (int m=0;m<M;m++) for (int n=0;n<N;n++){
    float a=0; for(int kk=0;kk<K;kk++) a += fA[m*K+kk]*fB[kk + n*K];   // B col-major
    ref[m*N+n]=a;
  }
  std::memset(D,0,sizeof(D));
  void* bufs[3]={A,B,D};
  int64_t dims[3]={M,N,K};
  tsrGpuLaunchParams p{}; p.buffers=bufs; p.num_buffers=3; p.dims=dims; p.num_dims=3;
  void* args[1]={&p};
  TsrStatus st = tsrLaunchKernel(s,k,args,1);
  if (st != TSR_STATUS_SUCCESS) { std::fprintf(stderr,"launch=%d\n",(int)st); return 11; }

  float maxerr=0.0f;
  for (int i=0;i<M*N;i++){ float d=std::fabs(D[i]-ref[i]); if(d>maxerr) maxerr=d; }
  if (maxerr > 1e-2f) { std::fprintf(stderr,"mma.sync maxerr=%g too high\n", maxerr); return 12; }

  // Negative: unregistered kernel still reports UNIMPLEMENTED.
  tsrArtifact art2=nullptr;
  if (tsrCompileArtifact("not_a_real_kernel", &opt, &art2)!=TSR_STATUS_SUCCESS) return 13;
  tsrKernel k2=nullptr;
  if (tsrGetKernel(art2, "not_a_real_kernel", &k2)!=TSR_STATUS_SUCCESS) return 14;
  if (tsrLaunchKernel(s,k2,args,1) != TSR_STATUS_UNIMPLEMENTED) return 15;

  std::printf("OK maxerr=%.3g\n", maxerr);
  tsrDestroyKernel(k2); tsrDestroyArtifact(art2);
  tsrDestroyKernel(k); tsrDestroyArtifact(art);
  tsrDestroyStream(s); tsrShutdown();
  return 0;
}
"""


def test_nvidia_mma_sync_gemm_executes_and_compares_through_bridge(tmp_path):
    nvcc = _nvcc()
    if nvcc is None:
        pytest.skip("nvcc (CUDA toolkit) not found")
    if not RUNTIME_LIB.is_file():
        pytest.skip("build libtessera_runtime.a (ninja -C build tessera_runtime)")

    # The kernel under test IS Tessera's emitted PTX — the whole point.
    ptx = P.emit_mma_sync_matmul_ptx()
    assert P.validate_mma_sync_ptx_structure(ptx) == []
    ptx_path = tmp_path / "tessera_mma.ptx"
    ptx_path.write_text(ptx)

    cuda_root = Path(nvcc).resolve().parents[1]
    stubs = cuda_root / "lib64" / "stubs"
    lib64 = cuda_root / "lib64"
    wsl_lib = Path("/usr/lib/wsl/lib")   # WSL ships the real libcuda here

    src = tmp_path / "nvmma_exec.cpp"
    obj = tmp_path / "nvmma_exec.o"
    binp = tmp_path / "nvmma_exec"
    src.write_text(_HARNESS)

    # Compile (two-step: nvcc's driver would try to compile the .a positionally).
    r = subprocess.run(
        [nvcc, "-std=c++17", "-O2", "-I", str(RUNTIME_INCLUDE),
         "-c", str(src), "-o", str(obj)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"nvcc compile failed:\n{r.stderr[:4000]}"

    link_libdirs = [str(d) for d in (stubs, wsl_lib) if d.is_dir()]
    r = subprocess.run(
        [nvcc, str(obj), str(RUNTIME_LIB),
         *[f"-L{d}" for d in link_libdirs], "-lcuda", "-lpthread", "-o", str(binp)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"nvcc link failed:\n{r.stderr[:4000]}"

    # Run; real libcuda comes from WSL lib / toolkit lib64 at load time.
    run_libdirs = [str(d) for d in (wsl_lib, lib64) if d.is_dir()]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        run_libdirs + [env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    r = subprocess.run([str(binp), str(ptx_path)],
                       capture_output=True, text=True, timeout=120, env=env)
    out = r.stdout.strip()
    if out == "SKIP_NO_DEVICE":
        pytest.skip("no usable NVIDIA GPU (CUDA probe failed)")
    assert r.returncode == 0, (
        f"harness exit {r.returncode} (non-zero = failing step)\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")
    assert out.startswith("OK"), r.stdout
