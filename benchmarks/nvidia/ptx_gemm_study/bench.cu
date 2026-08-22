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

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

// ----------------------------- timing core ----------------------------------
struct Stat { double median_ms, p05, p95, cov; };

template <class Launch>
Stat time_kernel(Launch launch, int warmup, int iters) {
  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  for (int i=0;i<warmup;i++) launch();
  CK(cudaDeviceSynchronize());
  std::vector<float> ms(iters);
  for (int i=0;i<iters;i++){ CK(cudaEventRecord(s)); launch(); CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e)); CK(cudaEventElapsedTime(&ms[i],s,e)); }
  std::sort(ms.begin(),ms.end());
  double sum=0; for(float v:ms)sum+=v; double mean=sum/iters;
  double var=0; for(float v:ms)var+=(v-mean)*(v-mean); var/=iters;
  Stat st; st.median_ms=ms[iters/2]; st.p05=ms[(int)(0.05*iters)];
  st.p95=ms[(int)(0.95*iters)]; st.cov=(mean>0)?sqrt(var)/mean:0;
  cudaEventDestroy(s); cudaEventDestroy(e); return st;
}

// ----------------------------- references -----------------------------------
// FP16: relative Frobenius error vs fp32 CPU. INT: bit-exact vs int CPU.
static double rel_err_f(const std::vector<float>&got,const std::vector<float>&ref){
  double num=0,den=0; for(size_t i=0;i<ref.size();i++){double d=got[i]-ref[i];num+=d*d;den+=ref[i]*ref[i];}
  return den>0?sqrt(num/den):sqrt(num);
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
    // fragment maps mirror probe.cu (validated by the correctness gate)
    a[0]=A[(tr*16+gr)*Kw+kw+tcq];      a[1]=A[(tr*16+gr+8)*Kw+kw+tcq];
    a[2]=A[(tr*16+gr)*Kw+kw+tcq+4];    a[3]=A[(tr*16+gr+8)*Kw+kw+tcq+4];
    b[0]=B[(tc*8+tcq)*Kw+kw+gr%4];     b[1]=B[(tc*8+tcq+4)*Kw+kw+gr%4];
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  }
  for(int i=0;i<4;i++){int row=tr*16+gr+8*(i/2);int col=tc*8+2*tcq+(i%2);C[row*N+col]=c[i];}
}
#endif

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
  printf("\"tessera_version\":\"%s\",\"clocks_locked\":%s,\"status\":\"%s\"}\n",
         g_ver, locked?"true":"false", status);
}

static double tflops_of(int n,double ms){return (2.0*n*n*n)/(ms/1e3)/1e12;}
static double ai_theo(const char*dt,int n){
  double bpe = (!strcmp(dt,"fp16")||!strcmp(dt,"bf16"))?2:(!strcmp(dt,"int8"))?1:0.5;
  double bytes = (double)n*n*(2*bpe+4); // A+B operand + C(int32/fp32) once
  return (2.0*n*n*n)/bytes;
}

int main(int argc,char**argv){
  std::vector<int> sizes={512,1024,2048,4096,8192};
  int iters=200,warmup=20,locked=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--iters"))iters=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--warmup"))warmup=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--clocks-locked"))locked=1;
    else if(!strcmp(argv[i],"--enable"))g_enable=argv[++i];
    else if(!strcmp(argv[i],"--sizes")){sizes.clear();char*t=strtok(argv[++i],",");
      while(t){sizes.push_back(atoi(t));t=strtok(nullptr,",");}}
  }
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); g_dev=strdup(p.name);
  fprintf(stderr,"device=%s cc=%d.%d iters=%d warmup=%d locked=%d\n",
          p.name,p.major,p.minor,iters,warmup,locked);

  for(int n:sizes){
    int M=n,N=n,K=n;
    // ---- FP16 WMMA baseline (validate then time) ----
    if(!enabled("fp16_wmma")){
      emit_row("fp16_wmma","fp16",n,nullptr,0,ai_theo("fp16",n),"SKIPPED_BY_PROBE",locked);
    } else {
      std::vector<float> Af(M*K),Bf(N*K),Cref(M*N);
      for(auto&x:Af)x=(rand()%7-3)*0.5f; for(auto&x:Bf)x=(rand()%7-3)*0.5f;
      for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++){float s=0;for(int k=0;k<K;k++)s+=Af[m*K+k]*Bf[nn*K+k];Cref[m*N+nn]=s;}
      std::vector<half>Ah(M*K),Bh(N*K); for(int i=0;i<M*K;i++)Ah[i]=__float2half(Af[i]);
      for(int i=0;i<N*K;i++)Bh[i]=__float2half(Bf[i]);
      half*dA,*dB; float*dC; CK(cudaMalloc(&dA,Ah.size()*2));CK(cudaMalloc(&dB,Bh.size()*2));CK(cudaMalloc(&dC,M*N*4));
      CK(cudaMemcpy(dA,Ah.data(),Ah.size()*2,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dB,Bh.data(),Bh.size()*2,cudaMemcpyHostToDevice));
      int warps=(M/16)*(N/16); int threads=256; int blocks=(warps*32+threads-1)/threads;
      auto L=[&]{ wmma_fp16<<<blocks,threads>>>(dA,dB,dC,M,N,K); };
      L(); cudaError_t ex=cudaDeviceSynchronize();
      std::vector<float>Cg(M*N); double err=1e9;
      if(ex==cudaSuccess){CK(cudaMemcpy(Cg.data(),dC,M*N*4,cudaMemcpyDeviceToHost));err=rel_err_f(Cg,Cref);}
      if(ex==cudaSuccess&&err<2e-2){Stat st=time_kernel(L,warmup,iters);
        emit_row("fp16_wmma","fp16",n,&st,tflops_of(n,st.median_ms),ai_theo("fp16",n),"OK",locked);}
      else emit_row("fp16_wmma","fp16",n,nullptr,0,ai_theo("fp16",n),
                    ex==cudaSuccess?"WRONG":"EXEC_FAIL",locked);
      cudaFree(dA);cudaFree(dB);cudaFree(dC);
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
      std::vector<int>Ai(M*K),Bi(N*K),Cref(M*N);
      for(auto&x:Ai)x=rand()%7-4; for(auto&x:Bi)x=rand()%7-4;
      for(int m=0;m<M;m++)for(int nn=0;nn<N;nn++){int s=0;for(int k=0;k<K;k++)s+=Ai[m*K+k]*Bi[nn*K+k];Cref[m*N+nn]=s;}
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
        for(int i=0;i<M*N;i++)maxe=std::max(maxe,(long)labs(Cg[i]-Cref[i]));}
      if(ex==cudaSuccess&&maxe==0){Stat st=time_kernel(L,warmup,iters);
        emit_row("int4_ptx_mma_k64","int4",n,&st,tflops_of(n,st.median_ms),ai_theo("int4",n),"OK",locked);}
      else emit_row("int4_ptx_mma_k64","int4",n,nullptr,0,ai_theo("int4",n),
                    ex==cudaSuccess?"WRONG":"EXEC_FAIL",locked);
      cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
#else
    emit_row("int4_ptx_mma_k64","int4",n,nullptr,0,ai_theo("int4",n),"COMPILED_OUT",locked);
#endif
    // TODO(on-box): int8_wmma, int8_ptx_mma_k32, int4_wmma (emulated baseline),
    //   int4_ptx_3stage, cuBLASLt fp16/int8 — same validate-then-time pattern.
    //   Kept out of the initial skeleton to keep the pivotal INT4 path reviewable.
  }
  return 0;
}
