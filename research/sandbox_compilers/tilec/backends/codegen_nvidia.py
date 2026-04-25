import os, pathlib, textwrap
from ..ir import IRModule
from . import codegen_tessera

def _write_bridge_script(out_dir: pathlib.Path, mlir_path: pathlib.Path, arch: str):
    repo = os.environ.get("TESSERA_REPO_ROOT", "")
    backend_dir = pathlib.Path(repo) / "src/compiler/codegen/tessera_gpu_backend_NVIDIA"
    script = out_dir / "bridge_run.sh"
    body = f"""#!/usr/bin/env bash
set -euo pipefail
REPO="{repo}"
BACKEND_DIR="{backend_dir}"
INPUT="{mlir_path}"
ARCH="${{ARCH:-{arch}}}"
echo "[bridge] Using tessera_gpu_backend_NVIDIA at $BACKEND_DIR"
echo "[bridge] Input IR: $INPUT"
echo "[bridge] ARCH=$ARCH"
# Example pipeline: adapt to your repo tool
# mlir-opt "$INPUT" -pass-pipeline="builtin.module(tessera-lower,convert-to-nvvm)" | \\
#   tessera-nv-cc --arch=$ARCH -o kernel.cubin
echo "[bridge] TODO: replace with your backend's CLI invocation"
"""
    script.write_text(body + "\n")
    os.chmod(script, 0o755)
    return script

def _emit_naive_cuda(ir: IRModule, out_dir: pathlib.Path, arch: str):
    f = ir.funcs[0]
    body = []
    for op in f.body:
        if op["op"] == "matmul":
            body.append("""
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float acc = 0.0f;
        for (int k=0;k<K;++k) acc += A[i*K + k] * B[k*N + j];
        C[i*N + j] = acc;
    }
            """)
        elif op["op"] == "add":
            body.append("""
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        C[i*N + j] = C[i*N + j] + BIAS[i*N + j];
    }
            """)
    cu = ("""#include <cuda_runtime.h>
#include <stdio.h>
__global__ void k___FNAME__(float* A, float* B, float* C, int M, int N, int K) {
__BODY__
}
static inline float frand() { return (float)rand()/RAND_MAX; }
int main(){
    int M=getenv("M")?atoi(getenv("M")):128;
    int N=getenv("N")?atoi(getenv("N")):128;
    int K=getenv("K")?atoi(getenv("K")):128;
    size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N;
    float *A=(float*)malloc(szA*sizeof(float));
    float *B=(float*)malloc(szB*sizeof(float));
    float *C=(float*)malloc(szC*sizeof(float));
    for (size_t i=0;i<szA;++i) A[i]=frand();
    for (size_t i=0;i<szB;++i) B[i]=frand();
    for (size_t i=0;i<szC;++i) C[i]=0.0f;
    float *dA,*dB,*dC;
    cudaMalloc(&dA, szA*sizeof(float));
    cudaMalloc(&dB, szB*sizeof(float));
    cudaMalloc(&dC, szC*sizeof(float));
    cudaMemcpy(dA,A,szA*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,szB*sizeof(float),cudaMemcpyHostToDevice);
    dim3 block(16,16);
    dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
    k___FNAME___<<<grid,block>>>(dA,dB,dC,M,N,K);
    cudaMemcpy(C,dC,szC*sizeof(float),cudaMemcpyDeviceToHost);
    double sum=0.0; for (size_t i=0;i<szC;++i) sum+=C[i];
    printf("OK NVIDIA naive  M=%d N=%d K=%d  checksum=%.6f\n",M,N,K,sum);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(A); free(B); free(C);
    return 0;
}
""").replace("__FNAME__", f.name).replace("__BODY__", "\n".join(body))
    (out_dir / f"{f.name}.cu").write_text(cu)
    (out_dir / "Makefile").write_text(f"""NVCC ?= nvcc
ARCH ?= {arch}
CFLAGS ?= -O3 -arch=$(ARCH)
all: {f.name}
{f.name}: {f.name}.cu
	$(NVCC) $(CFLAGS) -o {f.name} {f.name}.cu
clean:
	rm -f {f.name}
""")

def emit(ir: IRModule, out_dir: str, impl: str = "bridge", arch: str = None):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f = ir.funcs[0]
    arch = arch or os.environ.get("ARCH","sm90")
    mlir = out / f"{f.name}.tessera.mlir"
    mlir.write_text(codegen_tessera._emit_func_text(f))
    if impl == "naive":
        _emit_naive_cuda(ir, out, arch)
        print(f"[nvidia] naive CUDA emitted. Try: make -C {out}")
    else:
        script = _write_bridge_script(out, mlir, arch)
        print(f"[nvidia] bridge script: {script}")
    return str(out)
