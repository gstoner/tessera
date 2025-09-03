import os, pathlib, textwrap
from ..ir import IRModule
from . import codegen_tessera

def _write_bridge_script(out_dir: pathlib.Path, mlir_path: pathlib.Path, arch: str):
    repo = os.environ.get("TESSERA_REPO_ROOT", "")
    backend_dir = pathlib.Path(repo) / "src/compiler/codegen/Tessera_ROCM_Backend"
    script = out_dir / "bridge_run.sh"
    body = f"""#!/usr/bin/env bash
set -euo pipefail
REPO="{repo}"
BACKEND_DIR="{backend_dir}"
INPUT="{mlir_path}"
ARCH="${{ARCH:-{arch}}}"
echo "[bridge] Using Tessera_ROCM_Backend at $BACKEND_DIR"
echo "[bridge] Input IR: $INPUT"
echo "[bridge] ARCH=$ARCH"
# Example pipeline: adapt to your repo tool
# mlir-opt "$INPUT" -pass-pipeline="builtin.module(tessera-lower,convert-to-rocdl)" | \\
#   tessera-rocm-cc --arch=$ARCH -o kernel.hsaco
echo "[bridge] TODO: replace with your backend's CLI invocation"
"""
    script.write_text(body + "\n")
    os.chmod(script, 0o755)
    return script

def _emit_naive_hip(ir: IRModule, out_dir: pathlib.Path, arch: str):
    f = ir.funcs[0]
    src = """#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void k___FNAME__(float* A, float* B, float* C, int M, int N, int K) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= N) return;
  float acc = 0.0f;
  for (int k=0;k<K;++k) acc += A[i*K + k] * B[k*N + j];
  C[i*N + j] = acc;
}
static inline float frand() { return (float)rand()/RAND_MAX; }
int main() {
  int M = getenv("M")?atoi(getenv("M")):128;
  int N = getenv("N")?atoi(getenv("N")):128;
  int K = getenv("K")?atoi(getenv("K")):128;
  size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N;
  float *A=(float*)malloc(szA*sizeof(float));
  float *B=(float*)malloc(szB*sizeof(float));
  float *C=(float*)malloc(szC*sizeof(float));
  for (size_t i=0;i<szA;++i) A[i]=frand();
  for (size_t i=0;i<szB;++i) B[i]=frand();
  for (size_t i=0;i<szC;++i) C[i]=0.0f;
  float *dA,*dB,*dC;
  hipMalloc(&dA, szA*sizeof(float));
  hipMalloc(&dB, szB*sizeof(float));
  hipMalloc(&dC, szC*sizeof(float));
  hipMemcpy(dA,A,szA*sizeof(float),hipMemcpyHostToDevice);
  hipMemcpy(dB,B,szB*sizeof(float),hipMemcpyHostToDevice);
  dim3 block(16,16);
  dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
  hipLaunchKernelGGL(k___FNAME__, grid, block, 0, 0, dA,dB,dC,M,N,K);
  hipMemcpy(C,dC,szC*sizeof(float),hipMemcpyDeviceToHost);
  double sum=0.0; for (size_t i=0;i<szC;++i) sum+=C[i];
  printf("OK ROCm naive  M=%d N=%d K=%d  checksum=%.6f\n",M,N,K,sum);
  hipFree(dA); hipFree(dB); hipFree(dC); free(A); free(B); free(C);
  return 0;
}
""".replace("__FNAME__", f.name)
    (out_dir / f"{f.name}.hip.cpp").write_text(src)
    (out_dir / "Makefile").write_text(f"""HIPCC ?= hipcc
ARCH ?= {arch}
CFLAGS ?= -O3 --amdgpu-target=$(ARCH)
all: {f.name}
{f.name}: {f.name}.hip.cpp
	$(HIPCC) $(CFLAGS) -o {f.name} {f.name}.hip.cpp
clean:
	rm -f {f.name}
""" )

def emit(ir: IRModule, out_dir: str, impl: str = "bridge", arch: str = None):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f = ir.funcs[0]
    arch = arch or os.environ.get("ARCH","gfx90a")
    mlir = out / f"{f.name}.tessera.mlir"
    mlir.write_text(codegen_tessera._emit_func_text(f))
    if impl == "naive":
        _emit_naive_hip(ir, out, arch)
        print(f"[rocm] naive HIP emitted. Try: make -C {out}")
    else:
        script = _write_bridge_script(out, mlir, arch)
        print(f"[rocm] bridge script: {script}")
    return str(out)
