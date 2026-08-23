// bench.cu — Phase 1 (correctness) + Phase 2 (clean timing) driver.
//
// Emits one results.jsonl row per (kernel, dtype, N). CUDA-event timing only —
// NEVER profiler time (plan §C1/C2). Every kernel is VALIDATED against a
// reference before it is timed (plan §C6); a kernel that fails validation emits a
// row with "status":"WRONG" and latency null, and its timing is not trusted.
//
// INT4-first (per chosen emphasis): the native mma.sync.m16n8k64.s4 path is the
// centerpiece; fp16/int8 are context. cuBLASLt is the library floor (§C8).
//
// Build: see Makefile (nvcc -arch=sm_120a ... -lcublasLt)
// Run:   ./bench --sizes 512,1024,2048,4096,8192 --iters 200 --warmup 20 > results.jsonl
//
// HONESTY NOTE: the hand-PTX tiled kernels below are reference skeletons adapted
// from Borowski & Osinski Listings 1–3. They MUST pass the in-binary correctness
// check on sm_120 before their latencies mean anything — that check is the whole
// point of Phase 1 and protects the study from an untested kernel. WMMA baselines
// use nvcuda::wmma and are reliable. cuBLASLt is authoritative.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublasLt.h>

using namespace nvcuda;

#define CK(x) do{cudaError_t _cuda_status=(x); if(_cuda_status!=cudaSuccess){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(_cuda_status),__LINE__);exit(1);} }while(0)

// The cuBLASLt floor is configured outside the timed region.  It receives the
// same logical A[M,K] / B[N,K] (physically row-major, consumed as B^T) contract
// as the hand-written `.row.col` kernels.  This keeps setup/autotuning and host
// conversion out of the event measurement.
struct CublasLtF16Floor {
  cublasLtHandle_t handle{}; cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{},b{},c{},d{}; cublasLtMatmulPreference_t pref{};
  cublasLtMatmulHeuristicResult_t heuristic{}; void* workspace{};
  bool ready=false;
  bool init(int M,int N,int K) {
    cublasOperation_t transa=CUBLAS_OP_N, transb=CUBLAS_OP_T;
    cublasLtOrder_t order=CUBLASLT_ORDER_ROW;
    size_t workspace_bytes=4<<20; int returned=0;
    if(cublasLtCreate(&handle)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transa,sizeof(transa))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&transb,sizeof(transb))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,M,K,K)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,N,K,K)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,M,N,N)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutCreate(&d,CUDA_R_32F,M,N,N)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutSetAttribute(a,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutSetAttribute(b,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutSetAttribute(c,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutSetAttribute(d,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulPreferenceCreate(&pref)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes))!=CUBLAS_STATUS_SUCCESS ||
       cudaMalloc(&workspace,workspace_bytes)!=cudaSuccess ||
       cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,d,pref,1,&heuristic,&returned)!=CUBLAS_STATUS_SUCCESS || returned!=1) return false;
    ready=true; return true;
  }
  bool launch(const half* A,const half* B,float* C) const {
    float alpha=1.f,beta=0.f;
    return ready && cublasLtMatmul(handle,op,&alpha,A,a,B,b,&beta,C,c,C,d,
                                   &heuristic.algo,workspace,4<<20,0)==CUBLAS_STATUS_SUCCESS;
  }
  ~CublasLtF16Floor(){
    if(workspace) cudaFree(workspace); if(pref) cublasLtMatmulPreferenceDestroy(pref);
    if(d) cublasLtMatrixLayoutDestroy(d); if(c) cublasLtMatrixLayoutDestroy(c);
    if(b) cublasLtMatrixLayoutDestroy(b); if(a) cublasLtMatrixLayoutDestroy(a);
    if(op) cublasLtMatmulDescDestroy(op); if(handle) cublasLtDestroy(handle);
  }
};

struct CublasLtI8Floor {
  cublasLtHandle_t handle{}; cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{},b{},c{},d{}; cublasLtMatmulPreference_t pref{};
  cublasLtMatmulHeuristicResult_t heuristic{}; void* workspace{};
  bool ready=false;
  bool init(int M,int N,int K) {
    cublasOperation_t transa=CUBLAS_OP_N, transb=CUBLAS_OP_T;
    cublasLtOrder_t order=CUBLASLT_ORDER_ROW;
    size_t workspace_bytes=4<<20; int returned=0;
    if(cublasLtCreate(&handle)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transa,sizeof(transa))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&transb,sizeof(transb))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutCreate(&a,CUDA_R_8I,M,K,K)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutCreate(&b,CUDA_R_8I,N,K,K)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutCreate(&c,CUDA_R_32I,M,N,N)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutCreate(&d,CUDA_R_32I,M,N,N)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutSetAttribute(a,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutSetAttribute(b,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutSetAttribute(c,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatrixLayoutSetAttribute(d,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order))!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulPreferenceCreate(&pref)!=CUBLAS_STATUS_SUCCESS ||
       cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes))!=CUBLAS_STATUS_SUCCESS ||
       cudaMalloc(&workspace,workspace_bytes)!=cudaSuccess ||
       cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,d,pref,1,&heuristic,&returned)!=CUBLAS_STATUS_SUCCESS || returned!=1) return false;
    ready=true; return true;
  }
  bool launch(const int8_t* A,const int8_t* B,int32_t* C) const {
    int32_t alpha=1,beta=0;
    return ready && cublasLtMatmul(handle,op,&alpha,A,a,B,b,&beta,C,c,C,d,
                                   &heuristic.algo,workspace,4<<20,0)==CUBLAS_STATUS_SUCCESS;
  }
  ~CublasLtI8Floor(){
    if(workspace) cudaFree(workspace); if(pref) cublasLtMatmulPreferenceDestroy(pref);
    if(d) cublasLtMatrixLayoutDestroy(d); if(c) cublasLtMatrixLayoutDestroy(c);
    if(b) cublasLtMatrixLayoutDestroy(b); if(a) cublasLtMatrixLayoutDestroy(a);
    if(op) cublasLtMatmulDescDestroy(op); if(handle) cublasLtDestroy(handle);
  }
};

// ----------------------------- timing core ----------------------------------
struct Stat { double median_ms, p05, p95, cov; };

template <class Launch>
Stat time_kernel(Launch launch, int warmup, int iters, int batch) {
  cudaEvent_t start, stop;
  CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
  for (int i=0;i<warmup*batch;i++) launch();
  CK(cudaDeviceSynchronize());
  std::vector<float> ms(iters);
  for (int i=0;i<iters;i++){ CK(cudaEventRecord(start)); for(int j=0;j<batch;j++) launch(); CK(cudaEventRecord(stop));
    CK(cudaEventSynchronize(stop)); CK(cudaEventElapsedTime(&ms[i],start,stop)); ms[i]/=batch; }
  std::sort(ms.begin(),ms.end());
  double sum=0; for(float v:ms)sum+=v; double mean=sum/iters;
  double var=0; for(float v:ms)var+=(v-mean)*(v-mean); var/=iters;
  Stat st; st.median_ms=ms[iters/2]; st.p05=ms[(int)(0.05*iters)];
  st.p95=ms[(int)(0.95*iters)]; st.cov=(mean>0)?sqrt(var)/mean:0;
  cudaEventDestroy(start); cudaEventDestroy(stop); return st;
}

// ----------------------------- references -----------------------------------
// FP16: relative Frobenius error vs fp32 CPU. INT: bit-exact vs int CPU.
static double rel_err_f(const std::vector<float>&got,const std::vector<float>&ref){
  double num=0,den=0; for(size_t i=0;i<ref.size();i++){double d=got[i]-ref[i];num+=d*d;den+=ref[i]*ref[i];}
  return den>0?sqrt(num/den):sqrt(num);
}
static double sampled_err_f(const std::vector<float>&got,const std::vector<float>&A,
                            const std::vector<float>&B,int M,int N,int K){
  double worst=0;
  for(int s=0;s<64;s++){int m=(s*97)%M,n=(s*193)%N;float ref=0;
    for(int k=0;k<K;k++)ref+=A[m*K+k]*B[n*K+k]; worst=fmax(worst,fabs(got[m*N+n]-ref));}
  return worst;
}
static long sampled_err_i(const std::vector<int>&got,const std::vector<int>&A,
                          const std::vector<int>&B,int M,int N,int K){
  long worst=0;
  for(int s=0;s<64;s++){int m=(s*97)%M,n=(s*193)%N;int ref=0;
    for(int k=0;k<K;k++)ref+=A[m*K+k]*B[n*K+k];worst=std::max(worst,(long)labs(got[m*N+n]-ref));}
  return worst;
}
static long sampled_err_i_float(const std::vector<float>&got,const std::vector<int>&A,
                                const std::vector<int>&B,int M,int N,int K){
  long worst=0;
  for(int s=0;s<64;s++){int m=(s*97)%M,n=(s*193)%N,ref=0;
    for(int k=0;k<K;k++)ref+=A[m*K+k]*B[n*K+k];worst=std::max(worst,(long)labs((long)llround(got[m*N+n])-ref));}
  return worst;
}

// =================== FP16 WMMA baseline (reliable) ==========================
// Minimal tiled WMMA GEMM: C[MxN] = A[MxK] * B[KxN], row-major A, col-major B.
// Block computes a 16x16 output tile per warp; kept simple (no double buffer) so
// it is unquestionably correct as the *reference baseline*. PTX variants are the
// ones under study; the WMMA baseline only needs to be right, not fast.
__global__ void wmma_fp16(const half* A,const half* B,float* C,int M,int N,int K){
  int warp = (blockIdx.x*blockDim.x+threadIdx.x)/32;
  int tilesN = N/16; int tr=warp/tilesN, tc=warp%tilesN;
  if(tr*16>=M||tc*16>=N) return;
  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
  wmma::fragment<wmma::accumulator,16,16,16,float> c;
  wmma::fill_fragment(c,0.0f);
  for(int k=0;k<K;k+=16){
    wmma::load_matrix_sync(a,A+tr*16*K+k,K);
    wmma::load_matrix_sync(b,B+tc*16*K+k,K);   // B col-major: [N][K]
    wmma::mma_sync(c,a,b,c);
  }
  wmma::store_matrix_sync(C+tr*16*N+tc*16,c,N,wmma::mem_row_major);
}

// Distinct symbol solely for NCU attribution of the explicit pre-expanded INT4
// baseline.  Its compute is deliberately identical to wmma_fp16.
__global__ void wmma_fp16_int4_emulated(const half* A,const half* B,float* C,int M,int N,int K){
  int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32, tilesN=N/16, tr=warp/tilesN, tc=warp%tilesN;
  if(tr*16>=M||tc*16>=N) return;
  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
  wmma::fragment<wmma::accumulator,16,16,16,float> c;
  wmma::fill_fragment(c,0.0f);
  for(int k=0;k<K;k+=16){wmma::load_matrix_sync(a,A+tr*16*K+k,K);wmma::load_matrix_sync(b,B+tc*16*K+k,K);wmma::mma_sync(c,a,b,c);}
  wmma::store_matrix_sync(C+tr*16*N+tc*16,c,N,wmma::mem_row_major);
}

// =================== native INT4 PTX (THE pivotal kernel) ===================
// Skeleton adapted from Listings 1–2 (mma.sync.m16n8k64.s4). Simplified to a
// single accumulator tile so the correctness gate can validate the MMA + packing
// on-box; the double-buffered/3-stage variants layer on top once this is green.
// Guarded: build -DNO_INT4_MMA if nvcc rejects s4 for sm_120a.
#ifndef NO_INT4_MMA
__global__ void int4_native_k64(const uint32_t* A,const uint32_t* B,int32_t* C,
                                int M,int N,int K){
  // A: MxK int4 row-major packed 8/word => row stride K/8 words.
  // B: NxK int4 col-major packed => col stride K/8 words.
  int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32; int lane=threadIdx.x&31;
  int tilesN=N/8; int tr=warp/tilesN, tc=warp%tilesN;
  if(tr*16>=M||tc*8>=N) return;
  int gr=lane>>2, tcq=lane&3;
  int32_t c[4]={0,0,0,0};
  int Kw=K/8; // words per row
  for(int k=0;k<K;k+=64){
    int kw=k/8;
    uint32_t a[4],b[2];
    // Canonical m16n8k64 map: lane group owns the output row pair/B column;
    // thread-in-group selects an eight-int4 K fragment.  B is physically
    // column-major, so its word stride is Kw per output column.
    a[0]=A[(tr*16+gr)*Kw+kw+tcq];      a[1]=A[(tr*16+gr+8)*Kw+kw+tcq];
    a[2]=A[(tr*16+gr)*Kw+kw+tcq+4];    a[3]=A[(tr*16+gr+8)*Kw+kw+tcq+4];
    b[0]=B[(tc*8+gr)*Kw+kw+tcq];       b[1]=B[(tc*8+gr)*Kw+kw+tcq+4];
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  }
  for(int i=0;i<4;i++){int row=tr*16+gr+8*(i/2);int col=tc*8+2*tcq+(i%2);C[row*N+col]=c[i];}
}

// Three shared-memory stages, filled with cp.async.  One warp/block owns one
// m16n8 tile so every stage has an unambiguous producer/consumer barrier.  The
// ring holds 16x64 A (128 words) plus 64x8 B in physical column-major order
// (64 words).  It is intentionally a correctness-first pipeline, not yet a
// claim that this staging depth is the selected raster policy.
__device__ __forceinline__ void cp_async_16(void* dst,const void* src){
  unsigned smem=(unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(smem),"l"(src));
}
__device__ __forceinline__ void cp_async_commit(){ asm volatile("cp.async.commit_group;"); }
// Leave the two newer groups in flight.  With three primed slots this waits
// only for the oldest producer before its slot is consumed, preserving the
// intended copy/compute overlap through the ring.
__device__ __forceinline__ void cp_async_wait_two_pending(){ asm volatile("cp.async.wait_group 2;"); }

__global__ void int4_native_k64_3stage(const uint32_t* A,const uint32_t* B,int32_t* C,
                                       int M,int N,int K){
  __shared__ uint32_t stage[3][192];
  int lane=threadIdx.x; int tr=blockIdx.x/(N/8), tc=blockIdx.x%(N/8);
  int gr=lane>>2, tcq=lane&3, Kw=K/8, tiles=K/64;
  auto copy_tile=[&](int tile,int slot){
    int kw=tile*8;
    // 32 contiguous A chunks, then two 16-byte chunks for each B column.
    for(int chunk=lane;chunk<48;chunk+=32){
      const uint32_t* src;
      if(chunk<32) { int row=chunk/2, half=chunk%2;
        src=A+(tr*16+row)*Kw+kw+half*4; }
      else { int q=chunk-32; src=B+(tc*8+q/2)*Kw+kw+(q%2)*4; }
      cp_async_16(&stage[slot][chunk*4],src);
    }
    cp_async_commit();
  };
  int primed=tiles<3?tiles:3;
  for(int t=0;t<primed;t++) copy_tile(t,t);
  int32_t c[4]={0,0,0,0};
  for(int tile=0;tile<tiles;tile++){
    cp_async_wait_two_pending(); __syncthreads();
    int slot=tile%3; uint32_t* s=stage[slot];
    uint32_t a0=s[gr*8+tcq],a1=s[(gr+8)*8+tcq];
    uint32_t a2=s[gr*8+tcq+4],a3=s[(gr+8)*8+tcq+4];
    uint32_t b0=s[128+gr*8+tcq],b1=s[128+gr*8+tcq+4];
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    __syncthreads();
    if(tile+3<tiles) copy_tile(tile+3,slot);
  }
  for(int i=0;i<4;i++){int row=tr*16+gr+8*(i/2);int col=tc*8+2*tcq+(i%2);C[row*N+col]=c[i];}
}
#endif

// ====================== native INT8 PTX =====================================
// One warp owns a 16x8 output tile.  A is row-major and B is physically
// column-major (NxK), matching the `.row.col` mma.sync operand contract.
__global__ void int8_native_k32(const int8_t* A,const int8_t* B,int32_t* C,
                                int M,int N,int K){
  int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32; int lane=threadIdx.x&31;
  int tilesN=N/8; int tr=warp/tilesN, tc=warp%tilesN;
  if(tr*16>=M||tc*8>=N) return;
  int gr=lane>>2, tcq=lane&3;
  int32_t c[4]={0,0,0,0};
  int Kw=K/4;
  const uint32_t* A32=reinterpret_cast<const uint32_t*>(A);
  const uint32_t* B32=reinterpret_cast<const uint32_t*>(B);
  for(int k=0;k<K;k+=32){
    int kw=k/4;
    uint32_t a0=A32[(tr*16+gr)*Kw+kw+tcq];
    uint32_t a1=A32[(tr*16+gr+8)*Kw+kw+tcq];
    uint32_t a2=A32[(tr*16+gr)*Kw+kw+tcq+4];
    uint32_t a3=A32[(tr*16+gr+8)*Kw+kw+tcq+4];
    uint32_t b0=B32[(tc*8+gr)*Kw+kw+tcq];
    uint32_t b1=B32[(tc*8+gr)*Kw+kw+tcq+4];
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  for(int i=0;i<4;i++){int row=tr*16+gr+8*(i/2);int col=tc*8+2*tcq+(i%2);C[row*N+col]=c[i];}
}

// ----------------------------- probe gating ---------------------------------
// Fail-closed contract (plan §Phase 0): a family is timed ONLY if it is in the
// --enable set that run.sh derives from capability_matrix.json. This is what
// keeps an exec_fail (illegal-instruction) variant from ever being launched
// here — launching it would sticky-corrupt the CUDA context and abort the rest
// of the run, discarding already-valid rows (e.g. FP16). Default "ALL" is for
// standalone use; run.sh always passes an explicit set.
static std::string g_enable="ALL";
static bool enabled(const char* fam){
  return g_enable=="ALL" || g_enable.find(fam)!=std::string::npos;
}

// ----------------------------- JSONL emit -----------------------------------
static const char* g_dev="?"; static const char* g_ver="study-0.1";
static const char* g_representation="native";
static const char* g_validation="exact_cpu";
static const char* g_timing_scope="kernel";
static void emit_row(const char*kernel,const char*dtype,int n,const Stat*st,
                     double tflops,double ai_theo,const char*status,int locked){
  // Decision #12 canonical fields: backend, op, shape, dtype, latency_ms,
  // tflops, memory_bw_gb_s, device, tessera_version (+ study-specific extras).
  printf("{\"backend\":\"nvidia\",\"kernel\":\"%s\",\"dtype\":\"%s\",\"op\":\"gemm\",\"n\":%d,",
         kernel,dtype,n);
  printf("\"shape\":[%d,%d,%d],",n,n,n);
  if(st) printf("\"latency_ms\":%.6f,\"p05_ms\":%.6f,\"p95_ms\":%.6f,\"cov\":%.5f,\"tflops\":%.6f,",
                st->median_ms,st->p05,st->p95,st->cov,tflops);
  else   printf("\"latency_ms\":null,\"cov\":null,\"tflops\":null,");
  printf("\"ai_theoretical\":%.4f,\"memory_bw_gb_s\":null,\"device\":\"%s\",",ai_theo,g_dev);
  printf("\"tessera_version\":\"%s\",\"representation\":\"%s\",\"validation\":\"%s\",\"timing_scope\":\"%s\",\"clocks_locked\":%s,\"status\":\"%s\"}\n",
         g_ver, g_representation, g_validation, g_timing_scope, locked?"true":"false", status);
}

static double tflops_of(int n,double ms){return (2.0*n*n*n)/(ms/1e3)/1e12;}
static double ai_theo(const char*dt,int n){
  double bpe = (!strcmp(dt,"fp16")||!strcmp(dt,"bf16"))?2:(!strcmp(dt,"int8"))?1:0.5;
  double bytes = (double)n*n*(2*bpe+4); // A+B operand + C(int32/fp32) once
  return (2.0*n*n*n)/bytes;
}

int main(int argc,char**argv){
  std::vector<int> sizes={512,1024,2048,4096,8192};
  int iters=200,warmup=20,locked=0,exact_max_n=1024,batch=10;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--iters"))iters=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--warmup"))warmup=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--clocks-locked"))locked=1;
    else if(!strcmp(argv[i],"--exact-max-n"))exact_max_n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--batch"))batch=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--enable"))g_enable=argv[++i];
    else if(!strcmp(argv[i],"--sizes")){sizes.clear();char*t=strtok(argv[++i],",");
      while(t){sizes.push_back(atoi(t));t=strtok(nullptr,",");}}
  }
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); g_dev=strdup(p.name);
  fprintf(stderr,"device=%s cc=%d.%d iters=%d warmup=%d batch=%d locked=%d\n",
          p.name,p.major,p.minor,iters,warmup,batch,locked);

  for(int n:sizes){
    int M=n,N=n,K=n;
    bool exact_cpu=n<=exact_max_n;
    g_validation=exact_cpu?"exact_cpu":"sampled_cpu_64";
    // ---- FP16 WMMA baseline (validate then time) ----
    if(!enabled("fp16_wmma") && !enabled("cublaslt_fp16")){
      emit_row("fp16_wmma","fp16",n,nullptr,0,ai_theo("fp16",n),"SKIPPED_BY_PROBE",locked);
      g_timing_scope="library_call";
      emit_row("cublaslt_fp16","fp16",n,nullptr,0,ai_theo("fp16",n),"SKIPPED_BY_PROBE",locked);
      g_timing_scope="kernel";
    } else {
      std::vector<float> Af(M*K),Bf(N*K),Cref;
      for(auto&x:Af)x=(rand()%7-3)*0.5f; for(auto&x:Bf)x=(rand()%7-3)*0.5f;
      if(exact_cpu){Cref.resize(M*N);for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++){float s=0;for(int k=0;k<K;k++)s+=Af[m*K+k]*Bf[nn*K+k];Cref[m*N+nn]=s;}}
      std::vector<half>Ah(M*K),Bh(N*K); for(int i=0;i<M*K;i++)Ah[i]=__float2half(Af[i]);
      for(int i=0;i<N*K;i++)Bh[i]=__float2half(Bf[i]);
      half*dA,*dB; float*dC; CK(cudaMalloc(&dA,Ah.size()*2));CK(cudaMalloc(&dB,Bh.size()*2));CK(cudaMalloc(&dC,M*N*4));
      CK(cudaMemcpy(dA,Ah.data(),Ah.size()*2,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dB,Bh.data(),Bh.size()*2,cudaMemcpyHostToDevice));
      int warps=(M/16)*(N/16); int threads=256; int blocks=(warps*32+threads-1)/threads;
      std::vector<float>Cg(M*N);
      if(enabled("fp16_wmma")) {
      auto L=[&]{ wmma_fp16<<<blocks,threads>>>(dA,dB,dC,M,N,K); };
      L(); cudaError_t ex=cudaDeviceSynchronize();
      double err=1e9;
      if(ex==cudaSuccess){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));err=exact_cpu?rel_err_f(Cg,Cref):sampled_err_f(Cg,Af,Bf,M,N,K);}
      if(ex==cudaSuccess&&err<2e-2){Stat st=time_kernel(L,warmup,iters,batch);
        emit_row("fp16_wmma","fp16",n,&st,tflops_of(n,st.median_ms),ai_theo("fp16",n),"OK",locked);}
      else emit_row("fp16_wmma","fp16",n,nullptr,0,ai_theo("fp16",n),
                    ex==cudaSuccess?"WRONG":"EXEC_FAIL",locked);
      } else emit_row("fp16_wmma","fp16",n,nullptr,0,ai_theo("fp16",n),"SKIPPED_BY_PROBE",locked);

      // ---- cuBLASLt FP16 floor: same data/reference, descriptor setup excluded ----
      g_timing_scope="library_call";
      if(!enabled("cublaslt_fp16")) {
        emit_row("cublaslt_fp16","fp16",n,nullptr,0,ai_theo("fp16",n),"SKIPPED_BY_PROBE",locked);
      } else {
        CublasLtF16Floor lt; bool lt_ok=lt.init(M,N,K);
        if(lt_ok) lt_ok=lt.launch(dA,dB,dC) && cudaDeviceSynchronize()==cudaSuccess;
        double lt_err=1e9;
        if(lt_ok){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));lt_err=exact_cpu?rel_err_f(Cg,Cref):sampled_err_f(Cg,Af,Bf,M,N,K);}
        if(lt_ok&&lt_err<2e-2){
          auto LL=[&]{ if(!lt.launch(dA,dB,dC)){fprintf(stderr,"cuBLASLt launch failed\n");exit(1);} };
          Stat st=time_kernel(LL,warmup,iters,batch);
          emit_row("cublaslt_fp16","fp16",n,&st,tflops_of(n,st.median_ms),ai_theo("fp16",n),"OK",locked);
        } else emit_row("cublaslt_fp16","fp16",n,nullptr,0,ai_theo("fp16",n),
                        lt_ok?"WRONG":"EXEC_FAIL",locked);
      }
      g_timing_scope="kernel";
      cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
    // ---- INT4 pre-expanded FP16 baseline (independent of native s4) ----
    // Expansion and host-to-device copies happen before timing.  Its traffic
    // contract is consequently FP16 operands plus an FP32 output, despite the
    // logical INT4 input values; it is a compute-only comparison baseline.
    if(!enabled("int4_wmma_preexpanded_fp16")) {
      emit_row("int4_wmma_preexpanded_fp16","int4",n,nullptr,0,ai_theo("fp16",n),"SKIPPED_BY_PROBE",locked);
    } else {
      std::vector<int> Ai(M*K),Bi(N*K),Cref;
      for(auto&x:Ai)x=rand()%7-4; for(auto&x:Bi)x=rand()%7-4;
      if(exact_cpu){Cref.resize(M*N);for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++){int s=0;for(int k=0;k<K;k++)s+=Ai[m*K+k]*Bi[nn*K+k];Cref[m*N+nn]=s;}}
      std::vector<half> Ae(M*K),Be(N*K);
      for(int i=0;i<M*K;i++)Ae[i]=__float2half((float)Ai[i]);
      for(int i=0;i<N*K;i++)Be[i]=__float2half((float)Bi[i]);
      half *dAe,*dBe; float*dCe;
      CK(cudaMalloc(&dAe,Ae.size()*sizeof(half)));CK(cudaMalloc(&dBe,Be.size()*sizeof(half)));CK(cudaMalloc(&dCe,M*N*sizeof(float)));
      CK(cudaMemcpy(dAe,Ae.data(),Ae.size()*sizeof(half),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dBe,Be.data(),Be.size()*sizeof(half),cudaMemcpyHostToDevice));
      int warps16=(M/16)*(N/16),threads16=256,blocks16=(warps16*32+threads16-1)/threads16;
      auto LE=[&]{wmma_fp16_int4_emulated<<<blocks16,threads16>>>(dAe,dBe,dCe,M,N,K);};
      LE(); cudaError_t exe=cudaDeviceSynchronize(); long maxee=1<<30;
      std::vector<float> Cge(M*N);
      if(exe==cudaSuccess){CK(cudaMemcpy(Cge.data(),dCe,M*N*sizeof(float),cudaMemcpyDeviceToHost));maxee=0;
        maxee=exact_cpu?0:sampled_err_i_float(Cge,Ai,Bi,M,N,K);if(exact_cpu)for(int i=0;i<M*N;i++)maxee=std::max(maxee,(long)labs((long)llround(Cge[i])-Cref[i]));}
      g_representation="preexpanded_fp16";
      if(exe==cudaSuccess&&maxee==0){Stat st=time_kernel(LE,warmup,iters,batch);
        emit_row("int4_wmma_preexpanded_fp16","int4",n,&st,tflops_of(n,st.median_ms),ai_theo("fp16",n),"OK",locked);}
      else emit_row("int4_wmma_preexpanded_fp16","int4",n,nullptr,0,ai_theo("fp16",n),
                    exe==cudaSuccess?"WRONG":"EXEC_FAIL",locked);
      g_representation="native"; cudaFree(dAe);cudaFree(dBe);cudaFree(dCe);
    }
    // ---- INT4 native (pivotal): validate then time ----
    // Fail-closed: only launched if run.sh put int4_ptx_mma_k64 in --enable,
    // i.e. Phase 0 reported native_ok. This prevents an exec_fail (illegal
    // instruction) variant from sticky-corrupting the context and aborting the
    // remaining sizes (which would discard already-valid FP16 rows).
#ifndef NO_INT4_MMA
    if(!enabled("int4_ptx_mma_k64")){
      emit_row("int4_ptx_mma_k64","int4",n,nullptr,0,ai_theo("int4",n),"SKIPPED_BY_PROBE",locked);
    } else {
      std::vector<int>Ai(M*K),Bi(N*K),Cref;
      for(auto&x:Ai)x=rand()%7-4; for(auto&x:Bi)x=rand()%7-4;
      if(exact_cpu){Cref.resize(M*N);for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++){int s=0;for(int k=0;k<K;k++)s+=Ai[m*K+k]*Bi[nn*K+k];Cref[m*N+nn]=s;}}
      auto pack=[&](const std::vector<int>&v,int rows,int cols){std::vector<uint32_t>o((size_t)rows*cols/8,0);
        for(size_t i=0;i<v.size();i++){o[i/8]|=(uint32_t)(v[i]&0xF)<<((i%8)*4);}return o;};
      auto Ap=pack(Ai,M,K),Bp=pack(Bi,N,K);
      uint32_t*dA,*dB; int32_t*dC; CK(cudaMalloc(&dA,Ap.size()*4));CK(cudaMalloc(&dB,Bp.size()*4));CK(cudaMalloc(&dC,M*N*4));
      CK(cudaMemcpy(dA,Ap.data(),Ap.size()*4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dB,Bp.data(),Bp.size()*4,cudaMemcpyHostToDevice));
      int warps=(M/16)*(N/8); int threads=256; int blocks=(warps*32+threads-1)/threads;
      auto L=[&]{ int4_native_k64<<<blocks,threads>>>(dA,dB,dC,M,N,K); };
      L(); cudaError_t ex=cudaDeviceSynchronize();
      std::vector<int>Cg(M*N); long maxe=1<<30;
      if(ex==cudaSuccess){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));maxe=0;
        maxe=exact_cpu?0:sampled_err_i(Cg,Ai,Bi,M,N,K);if(exact_cpu)for(int i=0;i<M*N;i++)maxe=std::max(maxe,(long)labs(Cg[i]-Cref[i]));}
      if(ex==cudaSuccess&&maxe==0){Stat st=time_kernel(L,warmup,iters,batch);
        emit_row("int4_ptx_mma_k64","int4",n,&st,tflops_of(n,st.median_ms),ai_theo("int4",n),"OK",locked);}
      else emit_row("int4_ptx_mma_k64","int4",n,nullptr,0,ai_theo("int4",n),
                    ex==cudaSuccess?"WRONG":"EXEC_FAIL",locked);

      // Three-stage cp.async candidate.  It is intentionally separately
      // validated: a green scalar INT4 kernel never grants this pipeline a
      // timing claim.
      if(!enabled("int4_ptx_3stage")) {
        emit_row("int4_ptx_3stage","int4",n,nullptr,0,ai_theo("int4",n),"SKIPPED_BY_PROBE",locked);
      } else {
        int blocks=(M/16)*(N/8);
        auto L3=[&]{ int4_native_k64_3stage<<<blocks,32>>>(dA,dB,dC,M,N,K); };
        L3(); cudaError_t ex3=cudaDeviceSynchronize(); long maxe3=1<<30;
        if(ex3==cudaSuccess){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));maxe3=0;
          maxe3=exact_cpu?0:sampled_err_i(Cg,Ai,Bi,M,N,K);if(exact_cpu)for(int i=0;i<M*N;i++)maxe3=std::max(maxe3,(long)labs(Cg[i]-Cref[i]));}
        if(ex3==cudaSuccess&&maxe3==0){Stat st=time_kernel(L3,warmup,iters,batch);
          emit_row("int4_ptx_3stage","int4",n,&st,tflops_of(n,st.median_ms),ai_theo("int4",n),"OK",locked);}
        else emit_row("int4_ptx_3stage","int4",n,nullptr,0,ai_theo("int4",n),
                      ex3==cudaSuccess?"WRONG":"EXEC_FAIL",locked);
      }
      cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
#else
    emit_row("int4_ptx_mma_k64","int4",n,nullptr,0,ai_theo("int4",n),"COMPILED_OUT",locked);
#endif
    // ---- INT8 native: validate then time ----
    if(!enabled("int8_ptx_mma_k32") && !enabled("cublaslt_int8")){
      emit_row("int8_ptx_mma_k32","int8",n,nullptr,0,ai_theo("int8",n),"SKIPPED_BY_PROBE",locked);
      g_timing_scope="library_call";
      emit_row("cublaslt_int8","int8",n,nullptr,0,ai_theo("int8",n),"SKIPPED_BY_PROBE",locked);
      g_timing_scope="kernel";
    } else {
      std::vector<int>Ai(M*K),Bi(N*K),Cref;
      for(auto&x:Ai)x=rand()%7-3; for(auto&x:Bi)x=rand()%7-3;
      if(exact_cpu){Cref.resize(M*N);for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++){int s=0;for(int k=0;k<K;k++)s+=Ai[m*K+k]*Bi[nn*K+k];Cref[m*N+nn]=s;}}
      std::vector<int8_t>Ab(M*K),Bb(N*K);
      for(int i=0;i<M*K;i++)Ab[i]=Ai[i]; for(int i=0;i<N*K;i++)Bb[i]=Bi[i];
      int8_t*dA,*dB; int32_t*dC; CK(cudaMalloc(&dA,Ab.size()));CK(cudaMalloc(&dB,Bb.size()));CK(cudaMalloc(&dC,M*N*4));
      CK(cudaMemcpy(dA,Ab.data(),Ab.size(),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dB,Bb.data(),Bb.size(),cudaMemcpyHostToDevice));
      int warps=(M/16)*(N/8); int threads=256; int blocks=(warps*32+threads-1)/threads;
      std::vector<int>Cg(M*N);
      if(enabled("int8_ptx_mma_k32")) {
      auto L=[&]{ int8_native_k32<<<blocks,threads>>>(dA,dB,dC,M,N,K); };
      L(); cudaError_t ex=cudaDeviceSynchronize();
      long maxe=1<<30;
      if(ex==cudaSuccess){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));maxe=0;
        maxe=exact_cpu?0:sampled_err_i(Cg,Ai,Bi,M,N,K);if(exact_cpu)for(int i=0;i<M*N;i++)maxe=std::max(maxe,(long)labs(Cg[i]-Cref[i]));}
      if(ex==cudaSuccess&&maxe==0){Stat st=time_kernel(L,warmup,iters,batch);
        emit_row("int8_ptx_mma_k32","int8",n,&st,tflops_of(n,st.median_ms),ai_theo("int8",n),"OK",locked);}
      else emit_row("int8_ptx_mma_k32","int8",n,nullptr,0,ai_theo("int8",n),
                    ex==cudaSuccess?"WRONG":"EXEC_FAIL",locked);
      } else emit_row("int8_ptx_mma_k32","int8",n,nullptr,0,ai_theo("int8",n),"SKIPPED_BY_PROBE",locked);

      // ---- cuBLASLt INT8 floor: same signed-storage and B^T contract ----
      g_timing_scope="library_call";
      if(!enabled("cublaslt_int8")) {
        emit_row("cublaslt_int8","int8",n,nullptr,0,ai_theo("int8",n),"SKIPPED_BY_PROBE",locked);
      } else {
        CublasLtI8Floor lt; bool lt_ok=lt.init(M,N,K);
        if(lt_ok) lt_ok=lt.launch(dA,dB,dC) && cudaDeviceSynchronize()==cudaSuccess;
        long lt_err=1<<30;
        if(lt_ok){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));lt_err=0;
          lt_err=exact_cpu?0:sampled_err_i(Cg,Ai,Bi,M,N,K);if(exact_cpu)for(int i=0;i<M*N;i++)lt_err=std::max(lt_err,(long)labs(Cg[i]-Cref[i]));}
        if(lt_ok&&lt_err==0){
          auto LL=[&]{if(!lt.launch(dA,dB,dC)){fprintf(stderr,"cuBLASLt INT8 launch failed\n");exit(1);}};
          Stat st=time_kernel(LL,warmup,iters,batch);
          emit_row("cublaslt_int8","int8",n,&st,tflops_of(n,st.median_ms),ai_theo("int8",n),"OK",locked);
        } else emit_row("cublaslt_int8","int8",n,nullptr,0,ai_theo("int8",n),
                        lt_ok?"WRONG":"EXEC_FAIL",locked);
      }
      g_timing_scope="kernel";
      cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
    // Remaining fair-floor work: an end-to-end packed-INT4 emulation path (the
    // labelled WMMA candidate above intentionally excludes expansion). The
    // three-stage path remains non-selected until a full packet proves a win.
  }
  return 0;
}
