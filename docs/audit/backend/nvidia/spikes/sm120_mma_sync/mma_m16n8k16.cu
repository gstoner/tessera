// Spike #6 — sm_120 mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32 single-tile GEMM.
// One warp computes D[16x8] = A[16x16] * B[16x8] (bf16 in, f32 accumulate).
// Proves the instruction executes correctly on consumer Blackwell (CC 12.0).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define M 16
#define N 8
#define K 16

// A: MxK row-major bf16, B: KxN col-major bf16 (i.e. B[k + n*K]), D: MxN row-major f32.
__global__ void mma_tile(const __nv_bfloat16* A, const __nv_bfloat16* B, float* D) {
    int lane = threadIdx.x;            // 0..31, single warp
    int gid  = lane >> 2;              // groupID 0..7
    int tig  = lane & 3;               // threadID-in-group 0..3

    // --- Load A fragment: 4x .b32, each packs 2 consecutive-col bf16. ---
    // a0: row=gid,    col=2*tig (+1)
    // a1: row=gid+8,  col=2*tig (+1)
    // a2: row=gid,    col=2*tig+8 (+1)
    // a3: row=gid+8,  col=2*tig+8 (+1)
    auto packA = [&](int row, int col) -> unsigned {
        __nv_bfloat16 lo = A[row*K + col];
        __nv_bfloat16 hi = A[row*K + col + 1];
        __nv_bfloat162 v = __halves2bfloat162(lo, hi);
        return *reinterpret_cast<unsigned*>(&v);
    };
    unsigned a0 = packA(gid,   2*tig);
    unsigned a1 = packA(gid+8, 2*tig);
    unsigned a2 = packA(gid,   2*tig+8);
    unsigned a3 = packA(gid+8, 2*tig+8);

    // --- Load B fragment: 2x .b32, each packs 2 consecutive-row bf16. ---
    // b0: row=2*tig (+1),   col=gid
    // b1: row=2*tig+8 (+1), col=gid
    auto packB = [&](int row, int col) -> unsigned {
        __nv_bfloat16 lo = B[row + col*K];        // col-major
        __nv_bfloat16 hi = B[(row+1) + col*K];
        __nv_bfloat162 v = __halves2bfloat162(lo, hi);
        return *reinterpret_cast<unsigned*>(&v);
    };
    unsigned b0 = packB(2*tig,   gid);
    unsigned b1 = packB(2*tig+8, gid);

    float d0=0,d1=0,d2=0,d3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),
          "f"(d0),"f"(d1),"f"(d2),"f"(d3));

    // --- Store D fragment: 4x f32. ---
    // d0: row=gid,    col=2*tig
    // d1: row=gid,    col=2*tig+1
    // d2: row=gid+8,  col=2*tig
    // d3: row=gid+8,  col=2*tig+1
    D[(gid)*N   + 2*tig]   = d0;
    D[(gid)*N   + 2*tig+1] = d1;
    D[(gid+8)*N + 2*tig]   = d2;
    D[(gid+8)*N + 2*tig+1] = d3;
}

int main() {
    __nv_bfloat16 *hA=(__nv_bfloat16*)malloc(M*K*2), *hB=(__nv_bfloat16*)malloc(K*N*2);
    float *hD=(float*)malloc(M*N*4), *ref=(float*)malloc(M*N*4);
    float *fA=(float*)malloc(M*K*4), *fB=(float*)malloc(K*N*4);

    srand(1234);
    for(int i=0;i<M*K;i++){ float v=((rand()%201)-100)/100.0f; hA[i]=__float2bfloat16(v); fA[i]=__bfloat162float(hA[i]); }
    for(int i=0;i<K*N;i++){ float v=((rand()%201)-100)/100.0f; hB[i]=__float2bfloat16(v); fB[i]=__bfloat162float(hB[i]); }

    // CPU reference using the bf16-rounded values, f32 accumulate. B is col-major.
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        float acc=0; for(int k=0;k<K;k++) acc += fA[m*K+k]*fB[k + n*K];
        ref[m*N+n]=acc;
    }

    __nv_bfloat16 *dA,*dB; float *dD;
    CK(cudaMalloc(&dA,M*K*2)); CK(cudaMalloc(&dB,K*N*2)); CK(cudaMalloc(&dD,M*N*4));
    CK(cudaMemcpy(dA,hA,M*K*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB,K*N*2,cudaMemcpyHostToDevice));
    mma_tile<<<1,32>>>(dA,dB,dD);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hD,dD,M*N*4,cudaMemcpyDeviceToHost));

    double maxabs=0; int bad=0;
    for(int i=0;i<M*N;i++){ double e=fabs(hD[i]-ref[i]); if(e>maxabs)maxabs=e; if(e>1e-2)bad++; }
    printf("mma.sync.m16n8k16 single-tile GEMM (16x16x8, bf16->f32) on sm_120\n");
    printf("max abs error vs CPU ref = %g, elements over 1e-2 = %d / %d\n", maxabs, bad, M*N);
    printf("sample D[0..3] = %.4f %.4f %.4f %.4f   ref = %.4f %.4f %.4f %.4f\n",
           hD[0],hD[1],hD[2],hD[3], ref[0],ref[1],ref[2],ref[3]);
    printf("RESULT: %s\n", bad==0 ? "PASS (execute-and-compare matches reference)" : "FAIL");
    return bad==0?0:1;
}
