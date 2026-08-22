// probe.cu — Phase 0 capability + numeric probe (GATING).
//
// Decides, on the actual sm_120 silicon, which mma.sync variants (a) assemble,
// (b) execute without an illegal-instruction fault, and (c) produce a
// numerically correct fragment vs a CPU reference. Emits capability_matrix.json.
//
// The PIVOTAL question this answers (INT4-first, per plan §1): does native
//   mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
// execute on consumer Blackwell sm_120, or is it absent/emulated as it is on
// Hopper?  No result elsewhere in the study is trusted for a variant that fails
// here (Decision #21, fail-closed).
//
// Build:  nvcc -arch=sm_120a -o probe probe.cu
// Run:    ./probe > capability_matrix.json
//
// NOTE: this probe is deliberately self-contained (no cuBLAS, no external deps).
// A variant that fails to COMPILE is handled at build time: guard each native
// path behind a macro so the file always builds and simply reports the guarded
// variant as "compiled_out" if nvcc rejects the instruction for this arch.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
  } } while(0)

// A single warp performs one m16n8kK MMA on a tiny, fixed input and writes the
// 16x8 s32 (or f32) accumulator to global memory. The host compares to a CPU
// reference. "Correct" is the only thing that promotes a variant to usable.

// ---- helper: check a launch actually executed (catches illegal-instruction) --
static bool launch_ok(const char* what) {
  cudaError_t e = cudaGetLastError();          // launch config errors
  if (e != cudaSuccess) { fprintf(stderr,"[%s] launch: %s\n",what,cudaGetErrorString(e)); return false; }
  e = cudaDeviceSynchronize();                 // runtime faults (e.g. illegal instr)
  if (e != cudaSuccess) { fprintf(stderr,"[%s] exec: %s\n",what,cudaGetErrorString(e)); return false; }
  return true;
}

// ======================= FP16 m16n8k16 (sanity baseline) =====================
// A: 16x16 f16 row-major, B: 16x8 f16 col-major, C: 16x8 f32. Fragment layouts
// per PTX ISA "Matrix multiply-accumulate operation using mma instruction".
__global__ void mma_fp16_m16n8k16(const __half* A, const __half* B, float* C) {
  int lane = threadIdx.x & 31;
  // Load A fragment (8 f16 = 4 regs of .b32), B fragment (4 f16 = 2 regs).
  // Layout math per the ISA fragment tables; validated on-box by the CPU cmp.
  uint32_t a[4], b[2]; float c[4] = {0,0,0,0};
  const uint32_t* A32 = reinterpret_cast<const uint32_t*>(A);
  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  // A: row = lane%16 ... two k-groups; see ISA. Kept explicit for validation.
  int gr = lane >> 2, tc = lane & 3;
  a[0] = A32[(gr)      *8 + tc];
  a[1] = A32[(gr+8)    *8 + tc];
  a[2] = A32[(gr)      *8 + tc + 4];
  a[3] = A32[(gr+8)    *8 + tc + 4];
  b[0] = B32[(tc)      *4 + gr%4]; // placeholder col-major mapping; CPU cmp gates
  b[1] = B32[(tc+4)    *4 + gr%4];
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  // store per-lane accumulator (row = gr + 8*(i/2), col = 2*tc + i%2)
  for (int i=0;i<4;i++){ int row = gr + 8*(i/2); int col = 2*tc + (i%2); C[row*8+col]=c[i]; }
}

// ======================= INT8 m16n8k32 ======================================
__global__ void mma_int8_m16n8k32(const int8_t* A, const int8_t* B, int32_t* C) {
  int lane = threadIdx.x & 31;
  uint32_t a[4], b[2]; int32_t c[4] = {0,0,0,0};
  const uint32_t* A32 = reinterpret_cast<const uint32_t*>(A); // 4 int8 per b32
  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  int gr = lane >> 2, tc = lane & 3;
  a[0]=A32[gr*8+tc];   a[1]=A32[(gr+8)*8+tc];
  a[2]=A32[gr*8+tc+4]; a[3]=A32[(gr+8)*8+tc+4];
  b[0]=B32[tc*4+gr%4]; b[1]=B32[(tc+4)*4+gr%4];
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  for (int i=0;i<4;i++){ int row=gr+8*(i/2); int col=2*tc+(i%2); C[row*8+col]=c[i]; }
}

// ======================= INT4 m16n8k64  (THE PIVOTAL VARIANT) ================
// Guarded so the file always builds. If nvcc for sm_120a rejects the s4 MMA,
// build with -DNO_INT4_MMA and the probe reports it "compiled_out".
#ifndef NO_INT4_MMA
__global__ void mma_int4_m16n8k64(const uint32_t* A, const uint32_t* B, int32_t* C) {
  int lane = threadIdx.x & 31;
  // int4 packed 8-per-b32. A frag: 4 regs (16x64 s4), B frag: 2 regs (8x64 s4).
  uint32_t a[4], b[2]; int32_t c[4] = {0,0,0,0};
  int gr = lane >> 2, tc = lane & 3;
  a[0]=A[gr*8+tc];   a[1]=A[(gr+8)*8+tc];
  a[2]=A[gr*8+tc+4]; a[3]=A[(gr+8)*8+tc+4];
  b[0]=B[tc*4+gr%4]; b[1]=B[(tc+4)*4+gr%4];
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  for (int i=0;i<4;i++){ int row=gr+8*(i/2); int col=2*tc+(i%2); C[row*8+col]=c[i]; }
}
#endif

// ------------------------------ CPU references ------------------------------
static void cpu_gemm_f(const std::vector<float>&A,const std::vector<float>&B,
                       std::vector<float>&C,int M,int N,int K){
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=A[m*K+k]*B[n*K+k];C[m*N+n]=s;}
}
static void cpu_gemm_i(const std::vector<int>&A,const std::vector<int>&B,
                       std::vector<int>&C,int M,int N,int K){
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){int s=0;for(int k=0;k<K;k++)s+=A[m*K+k]*B[n*K+k];C[m*N+n]=s;}
}

// emit one JSON row
static void emit(const char* name,const char* ptx,const char* status,double err){
  static bool first=true;
  printf("%s\n  {\"variant\":\"%s\",\"ptx\":\"%s\",\"status\":\"%s\",\"max_abs_err\":%g}",
         first?"":",",name,ptx,status,err);
  first=false;
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("{\n\"device\":\"%s\",\"cc\":\"%d.%d\",\"l2_bytes\":%zu,\"variants\":[",
         p.name,p.major,p.minor,(size_t)p.l2CacheSize);
  bool sm120 = (p.major==12);
  fprintf(stderr,"device=%s cc=%d.%d sm120=%s\n",p.name,p.major,p.minor,sm120?"yes":"no");

  // ---- FP16 sanity ----
  {
    int M=16,N=8,K=16; std::vector<float> Af(M*K),Bf(N*K),Cref(M*N);
    for(auto&x:Af)x=(rand()%7-3); for(auto&x:Bf)x=(rand()%7-3);
    cpu_gemm_f(Af,Bf,Cref,M,N,K);
    std::vector<__half> Ah(M*K),Bh(N*K); for(int i=0;i<M*K;i++)Ah[i]=__float2half(Af[i]);
    for(int i=0;i<N*K;i++)Bh[i]=__float2half(Bf[i]);
    __half *dA,*dB; float* dC; CK(cudaMalloc(&dA,Ah.size()*2));CK(cudaMalloc(&dB,Bh.size()*2));
    CK(cudaMalloc(&dC,M*N*4));
    CK(cudaMemcpy(dA,Ah.data(),Ah.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,Bh.data(),Bh.size()*2,cudaMemcpyHostToDevice));
    mma_fp16_m16n8k16<<<1,32>>>(dA,dB,dC);
    bool ok=launch_ok("fp16"); std::vector<float>Cg(M*N); double err=1e9;
    if(ok){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));err=0;
      for(int i=0;i<M*N;i++)err=fmax(err,fabs(Cg[i]-Cref[i]));}
    emit("fp16_m16n8k16","mma.sync.m16n8k16.f32.f16.f16.f32",
         ok?(err<1e-1?"native_ok":"ran_wrong_layout"):"exec_fail",err);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
  }

  // ---- INT8 ----
  {
    int M=16,N=8,K=32; std::vector<int>Ai(M*K),Bi(N*K),Cref(M*N);
    for(auto&x:Ai)x=rand()%7-3; for(auto&x:Bi)x=rand()%7-3;
    cpu_gemm_i(Ai,Bi,Cref,M,N,K);
    std::vector<int8_t>Ab(M*K),Bb(N*K); for(int i=0;i<M*K;i++)Ab[i]=Ai[i];
    for(int i=0;i<N*K;i++)Bb[i]=Bi[i];
    int8_t*dA,*dB; int32_t*dC; CK(cudaMalloc(&dA,Ab.size()));CK(cudaMalloc(&dB,Bb.size()));
    CK(cudaMalloc(&dC,M*N*4));
    CK(cudaMemcpy(dA,Ab.data(),Ab.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,Bb.data(),Bb.size(),cudaMemcpyHostToDevice));
    mma_int8_m16n8k32<<<1,32>>>(dA,dB,dC);
    bool ok=launch_ok("int8"); std::vector<int>Cg(M*N); double err=1e9;
    if(ok){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));err=0;
      for(int i=0;i<M*N;i++)err=fmax(err,(double)abs(Cg[i]-Cref[i]));}
    emit("int8_m16n8k32","mma.sync.m16n8k32.s32.s8.s8.s32",
         ok?(err==0?"native_ok":"ran_wrong_layout"):"exec_fail",err);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
  }

  // ---- INT4 (pivotal) ----
#ifdef NO_INT4_MMA
  emit("int4_m16n8k64","mma.sync.m16n8k64.s32.s4.s4.s32","compiled_out",-1);
#else
  {
    int M=16,N=8,K=64; std::vector<int>Ai(M*K),Bi(N*K),Cref(M*N);
    for(auto&x:Ai)x=rand()%7-4; for(auto&x:Bi)x=rand()%7-4;   // in [-4,3] fits s4
    cpu_gemm_i(Ai,Bi,Cref,M,N,K);
    // pack 8 s4 per u32 (row-major A: MxK, B col-major: NxK)
    auto pack=[&](const std::vector<int>&v,int rows,int cols){
      std::vector<uint32_t> o(rows*cols/8,0);
      for(int r=0;r<rows;r++)for(int c=0;c<cols;c++){int val=v[r*cols+c]&0xF;
        int idx=(r*cols+c); o[idx/8]|=(uint32_t)val<<((idx%8)*4);} return o;};
    auto Ap=pack(Ai,M,K); auto Bp=pack(Bi,N,K);
    uint32_t*dA,*dB; int32_t*dC; CK(cudaMalloc(&dA,Ap.size()*4));CK(cudaMalloc(&dB,Bp.size()*4));
    CK(cudaMalloc(&dC,M*N*4));
    CK(cudaMemcpy(dA,Ap.data(),Ap.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,Bp.data(),Bp.size()*4,cudaMemcpyHostToDevice));
    mma_int4_m16n8k64<<<1,32>>>(dA,dB,dC);
    bool ok=launch_ok("int4"); std::vector<int>Cg(M*N); double err=1e9;
    if(ok){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));err=0;
      for(int i=0;i<M*N;i++)err=fmax(err,(double)abs(Cg[i]-Cref[i]));}
    // status semantics: exec_fail => native s4 NOT available on sm_120 (Hopper-like
    // removal); ran_wrong_layout => instruction exists, our fragment map needs the
    // on-box fix; native_ok => the paper's key premise holds on sm_120.
    emit("int4_m16n8k64","mma.sync.m16n8k64.s32.s4.s4.s32",
         ok?(err==0?"native_ok":"ran_wrong_layout"):"exec_fail",err);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
  }
#endif

  printf("\n]}\n");
  return 0;
}
