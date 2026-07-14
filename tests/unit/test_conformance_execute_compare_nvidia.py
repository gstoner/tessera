"""Stage D (NVIDIA) — sm_120 mma.sync GEMM executes through the SHIPPED C-ABI bridge.

The consumer-Blackwell counterpart to ``test_rocm_wmma_execute_compare.py``. The
launched kernel is the **Tessera-emitted PTX** from
``ptx_emit.emit_mma_sync_matmul_ptx`` (NOT a hand-written nvcc kernel): a
warp-level ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` that computes a
16×8×16 ``f32 ← bf16`` GEMM.

C2 tail (COMPILER_REFACTOR_PLAN): the launcher is no longer an inline throwaway —
it is the **shipped** ``tessera_nvidia_ptx_launch.cpp`` (the counterpart to Apple's
``apple_gpu_runtime.mm``), compiled straight from source here. The harness registers
the emitted PTX (``tessera_nvidia_ptx_register``), registers the shipped launcher on
the ``tsrRegisterGpuLauncher`` seam (``tessera_nvidia_register_ptx_launcher``), then
drives it through ``tsrLaunchKernel`` and execute-and-compares against a host
reference. The shipped bridge driver-JITs the PTX (``cuModuleLoadDataEx``, cached by
kernel name) and launches it (``cuLaunchKernel``).

This is the first real **`backend_kernel` execution** for a Tessera matmul on
NVIDIA silicon: the emitted PTX runs through the shipped runtime bridge and matches
the oracle, closing the rung ladder emit → assemble → launch → compare on sm_120.

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
# The shipped C2-tail launch bridge (compiled from source into the harness).
BRIDGE_DIR = (REPO_ROOT / "src" / "compiler" / "codegen"
              / "tessera_gpu_backend_NVIDIA" / "runtime" / "cuda")
BRIDGE_SRC = BRIDGE_DIR / "tessera_nvidia_ptx_launch.cpp"


def _nvcc() -> str | None:
    return shutil.which("nvcc") or (
        "/usr/local/cuda/bin/nvcc"
        if Path("/usr/local/cuda/bin/nvcc").is_file()
        else None
    )


# The harness registers the emitted PTX (read from argv[1]) with the SHIPPED
# bridge, registers the shipped launcher on the tsrRegisterGpuLauncher seam, then
# runs it through tsrLaunchKernel. buffers = {A(bf16), B(bf16), D(f32)};
# dims = {M,N,K} (this kernel: 16,8,16). The launcher body lives in the shipped
# tessera_nvidia_ptx_launch.cpp (compiled alongside this harness) — NOT inline.
_HARNESS = r"""
#include "tessera/tessera_runtime.h"
#include "tessera_nvidia_ptx_launch.h"
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#define M 16
#define N 8
#define K 16

// bf16 round-to-nearest-even, packed in uint16 (host side, no SDK dependency).
static unsigned short f2bf16(float f){
  unsigned int x; std::memcpy(&x,&f,4);
  unsigned int r=(x>>16)&1u, t=0x7fffu+r; x+=t; return (unsigned short)(x>>16);
}
static float bf162f(unsigned short h){ unsigned int x=((unsigned int)h)<<16; float f; std::memcpy(&f,&x,4); return f; }

static bool read_file(const char* path, std::string& out) {
  FILE* fp = std::fopen(path, "rb");
  if (!fp) return false;
  std::fseek(fp,0,SEEK_END); long sz=std::ftell(fp); std::fseek(fp,0,SEEK_SET);
  std::vector<char> buf(sz+1);
  if (std::fread(buf.data(),1,sz,fp)!=(size_t)sz){std::fclose(fp);return false;}
  buf[sz]=0; std::fclose(fp); out.assign(buf.data()); return true;
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
  if (!gpu_usable()) { std::printf("SKIP_NO_DEVICE\n"); return 0; }
  std::string ptx;
  if (!read_file(argv[1], ptx)) return 3;
  // Hand the emitted PTX to the shipped bridge, then register it on the seam.
  if (tessera_nvidia_ptx_register("tessera_mma_m16n8k16_bf16", ptx.c_str()) != 0) return 4;
  if (tessera_nvidia_register_ptx_launcher() != 0) return 5;
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

  // ---- general aligned mma.sync GEMM (K-loop + grid-tiled), PTX from argv[2] ----
  // A multi-tile, multi-K-iteration shape (32x16x32) through the SAME seam, proving
  // the C2-tail-breadth emitter beyond the single 16x8x16 tile.
  {
    std::string gptx;
    if (argc < 3 || !read_file(argv[2], gptx)) return 16;
    if (tessera_nvidia_ptx_register("tessera_mma_gemm_bf16", gptx.c_str()) != 0) return 17;
    const int M2=32, N2=16, K2=32;
    std::vector<unsigned short> gA(M2*K2), gB(K2*N2);
    std::vector<float> gD(M2*N2, 0.0f), gref(M2*N2), gfA(M2*K2), gfB(K2*N2);
    for (int i=0;i<M2*K2;i++){ float v=(((i*29+5)%201)-100)/100.0f; gA[i]=f2bf16(v); gfA[i]=bf162f(gA[i]); }
    for (int i=0;i<K2*N2;i++){ float v=(((i*41+3)%201)-100)/100.0f; gB[i]=f2bf16(v); gfB[i]=bf162f(gB[i]); }
    for (int m=0;m<M2;m++) for (int n=0;n<N2;n++){
      float a=0; for(int kk=0;kk<K2;kk++) a += gfA[m*K2+kk]*gfB[kk + n*K2];  // B col-major
      gref[m*N2+n]=a;
    }
    tsrArtifact ga=nullptr; if (tsrCompileArtifact("tessera_mma_gemm_bf16",&opt,&ga)!=TSR_STATUS_SUCCESS) return 18;
    tsrKernel gk=nullptr;   if (tsrGetKernel(ga,"tessera_mma_gemm_bf16",&gk)!=TSR_STATUS_SUCCESS) return 19;
    void* gbufs[3]={gA.data(),gB.data(),gD.data()};
    int64_t gdims[3]={M2,N2,K2};
    tsrGpuLaunchParams gp{}; gp.buffers=gbufs; gp.num_buffers=3; gp.dims=gdims; gp.num_dims=3;
    void* gargs[1]={&gp};
    if (tsrLaunchKernel(s,gk,gargs,1)!=TSR_STATUS_SUCCESS) return 20;
    float gmax=0.0f; for (int i=0;i<M2*N2;i++){ float d=std::fabs(gD[i]-gref[i]); if(d>gmax)gmax=d; }
    if (gmax > 1e-2f) { std::fprintf(stderr,"general GEMM maxerr=%g\n",gmax); return 21; }

    // Oversized aligned shape (M*K = 32784*65536 > 2^31): the emitted kernel's
    // 32-bit index math would wrap, so the bridge must reject BEFORE allocating,
    // honestly (INVALID_ARGUMENT), not silently corrupt (PR #291 review). Reuses
    // the small gbufs — the guard fires before any pointer is dereferenced.
    int64_t odims[3]={32784, 8, 65536};
    tsrGpuLaunchParams op2{}; op2.buffers=gbufs; op2.num_buffers=3; op2.dims=odims; op2.num_dims=3;
    void* oargs[1]={&op2};
    if (tsrLaunchKernel(s,gk,oargs,1) != TSR_STATUS_INVALID_ARGUMENT) return 22;

    tsrDestroyKernel(gk); tsrDestroyArtifact(ga);
  }

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

    # The kernels under test ARE Tessera's emitted PTX — the whole point.
    ptx = P.emit_mma_sync_matmul_ptx()
    assert P.validate_mma_sync_ptx_structure(ptx) == []
    ptx_path = tmp_path / "tessera_mma.ptx"
    ptx_path.write_text(ptx)
    # The general aligned-M/N/K GEMM (C2-tail breadth), passed as argv[2].
    gptx = P.emit_mma_sync_gemm_ptx(dtype="bf16")
    assert P.validate_mma_sync_gemm_ptx_structure(gptx) == []
    gptx_path = tmp_path / "tessera_mma_gemm.ptx"
    gptx_path.write_text(gptx)

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
         "-I", str(BRIDGE_DIR), "-c", str(src), "-o", str(obj)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"nvcc compile failed:\n{r.stderr[:4000]}"

    # Compile the SHIPPED bridge source (not an inline copy) into the harness.
    bridge_obj = tmp_path / "tessera_nvidia_ptx_launch.o"
    r = subprocess.run(
        [nvcc, "-std=c++17", "-O2", "-I", str(RUNTIME_INCLUDE),
         "-I", str(BRIDGE_DIR), "-c", str(BRIDGE_SRC), "-o", str(bridge_obj)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"nvcc bridge compile failed:\n{r.stderr[:4000]}"

    link_libdirs = [str(d) for d in (stubs, wsl_lib) if d.is_dir()]
    r = subprocess.run(
        [nvcc, str(obj), str(bridge_obj), str(RUNTIME_LIB),
         *[f"-L{d}" for d in link_libdirs], "-lcuda", "-lpthread", "-o", str(binp)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"nvcc link failed:\n{r.stderr[:4000]}"

    # Run; real libcuda comes from WSL lib / toolkit lib64 at load time.
    run_libdirs = [str(d) for d in (wsl_lib, lib64) if d.is_dir()]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        run_libdirs + [env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    r = subprocess.run([str(binp), str(ptx_path), str(gptx_path)],
                       capture_output=True, text=True, timeout=120, env=env)
    out = r.stdout.strip()
    if out == "SKIP_NO_DEVICE":
        pytest.skip("no usable NVIDIA GPU (CUDA probe failed)")
    assert r.returncode == 0, (
        f"harness exit {r.returncode} (non-zero = failing step)\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")
    assert out.startswith("OK"), r.stdout
