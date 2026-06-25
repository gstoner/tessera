// Prototype: general mma.sync GEMM via NVRTC (mirror of ROCm's HIPRTC path).
// D[M,N] f32 = A[M,K] @ B[K,N], inputs 16-bit (bf16 or f16), row-major.
// Each warp computes one 16x8 output tile, looping K in steps of 16; ragged
// edges zero-padded. Validates across shapes for both dtypes.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda.h>
#include <nvrtc.h>

// Kernel source: two extern "C" kernels (f16, bf16), identical but for the mma
// operand type. 16-bit operands are loaded as raw bits and packed into .b32 —
// no cuda_fp16/bf16 headers needed (keeps NVRTC self-contained).
static const char* kSrc = R"NVRTC(
extern "C" __global__ void gemm_TYPE(const unsigned short* A,
                                     const unsigned short* B,
                                     float* D, int M, int N, int K) {
  int mtile = blockIdx.x * 16;
  int ntile = blockIdx.y * 8;
  int lane = threadIdx.x;
  int gid = lane >> 2, tig = lane & 3;
  float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
  for (int k0 = 0; k0 < K; k0 += 16) {
    // A fragment (row-major MxK): each reg packs (col,col+1) of one row.
    auto la = [&](int r, int c)->unsigned {
      int rr=mtile+r, cc=k0+c;
      unsigned lo = (rr<M && cc<K)   ? A[rr*K+cc]   : 0u;
      unsigned hi = (rr<M && cc+1<K) ? A[rr*K+cc+1] : 0u;
      return (hi<<16)|lo;
    };
    unsigned a0=la(gid,2*tig), a1=la(gid+8,2*tig), a2=la(gid,2*tig+8), a3=la(gid+8,2*tig+8);
    // B fragment (row-major KxN, .col layout): each reg packs (row,row+1) of one col.
    auto lb = [&](int r, int c)->unsigned {
      int rr=k0+r, cc=ntile+c;
      unsigned lo = (rr<K   && cc<N) ? B[rr*N+cc]     : 0u;
      unsigned hi = (rr+1<K && cc<N) ? B[(rr+1)*N+cc] : 0u;
      return (hi<<16)|lo;
    };
    unsigned b0=lb(2*tig,gid), b1=lb(2*tig+8,gid);
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.TYPE.TYPE.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  auto st=[&](int r,int c,float v){ int rr=mtile+r,cc=ntile+c; if(rr<M&&cc<N) D[rr*N+cc]=v; };
  st(gid,2*tig,d0); st(gid,2*tig+1,d1); st(gid+8,2*tig,d2); st(gid+8,2*tig+1,d3);
}
)NVRTC";

#define DK(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorString(r,&s);printf("CU %s @%d:%s\n",#x,__LINE__,s);exit(1);}}while(0)
#define NK(x) do{nvrtcResult r=(x); if(r!=NVRTC_SUCCESS){printf("NVRTC %s @%d:%s\n",#x,__LINE__,nvrtcGetErrorString(r));exit(1);}}while(0)

static unsigned short f2bf16(float f){unsigned x;memcpy(&x,&f,4);unsigned r=(x>>16)&1u,t=0x7fffu+r;x+=t;return (unsigned short)(x>>16);}
static float bf162f(unsigned short h){unsigned x=((unsigned)h)<<16;float f;memcpy(&f,&x,4);return f;}
// minimal f16 (round-toward-zero is fine for test magnitudes; use RNE-ish)
static unsigned short f2f16(float f){
  unsigned x; memcpy(&x,&f,4);
  unsigned sign=(x>>16)&0x8000u; int exp=(int)((x>>23)&0xff)-127+15; unsigned man=x&0x7fffffu;
  if(exp<=0) return (unsigned short)sign;
  if(exp>=31) return (unsigned short)(sign|0x7c00u);
  return (unsigned short)(sign | (exp<<10) | (man>>13));
}
static float f162f(unsigned short h){
  unsigned sign=(h&0x8000u)<<16; int exp=(h>>10)&0x1f; unsigned man=h&0x3ffu;
  if(exp==0){ if(man==0){unsigned b=sign;float f;memcpy(&f,&b,4);return f;} }
  unsigned e=(unsigned)(exp-15+127); unsigned b=sign|(e<<23)|(man<<13); float f; memcpy(&f,&b,4); return f;
}

static CUfunction load(CUdevice dev, const char* type, const char* fname) {
  std::string src(kSrc);
  // substitute TYPE
  for(size_t p; (p=src.find("TYPE"))!=std::string::npos; ) src.replace(p,4,type);
  nvrtcProgram prog; NK(nvrtcCreateProgram(&prog, src.c_str(), "gemm.cu",0,0,0));
  int maj,min; DK(cuDeviceGetAttribute(&maj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev));
  DK(cuDeviceGetAttribute(&min,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev));
  char arch[32]; snprintf(arch,sizeof(arch),"--gpu-architecture=compute_%d%d",maj,min);
  const char* opts[]={arch};
  nvrtcResult c=nvrtcCompileProgram(prog,1,opts);
  if(c!=NVRTC_SUCCESS){ size_t n; nvrtcGetProgramLogSize(prog,&n); std::vector<char> log(n); nvrtcGetProgramLog(prog,log.data()); printf("NVRTC log:\n%s\n",log.data()); exit(1);}
  size_t psz; NK(nvrtcGetPTXSize(prog,&psz)); std::vector<char> ptx(psz); NK(nvrtcGetPTX(prog,ptx.data()));
  CUmodule mod; DK(cuModuleLoadData(&mod, ptx.data()));
  CUfunction fn; DK(cuModuleGetFunction(&fn,mod,fname));
  nvrtcDestroyProgram(&prog);
  return fn;
}

static double run_shape(CUdevice dev, CUfunction fn, int M,int N,int K, bool bf16){
  std::vector<unsigned short> A(M*K),B(K*N); std::vector<float> D(M*N,0),ref(M*N,0),fA(M*K),fB(K*N);
  srand(M*1000+N*10+K);
  for(int i=0;i<M*K;i++){float v=((rand()%201)-100)/100.f; A[i]=bf16?f2bf16(v):f2f16(v); fA[i]=bf16?bf162f(A[i]):f162f(A[i]);}
  for(int i=0;i<K*N;i++){float v=((rand()%201)-100)/100.f; B[i]=bf16?f2bf16(v):f2f16(v); fB[i]=bf16?bf162f(B[i]):f162f(B[i]);}
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){float a=0;for(int k=0;k<K;k++)a+=fA[m*K+k]*fB[k*N+n];ref[m*N+n]=a;}
  CUdeviceptr dA,dB,dD; DK(cuMemAlloc(&dA,M*K*2));DK(cuMemAlloc(&dB,K*N*2));DK(cuMemAlloc(&dD,M*N*4));
  DK(cuMemcpyHtoD(dA,A.data(),M*K*2)); DK(cuMemcpyHtoD(dB,B.data(),K*N*2));
  int mt=(M+15)/16, nt=(N+7)/8; void* args[]={&dA,&dB,&dD,&M,&N,&K};
  DK(cuLaunchKernel(fn, mt,nt,1, 32,1,1, 0,0,args,0)); DK(cuCtxSynchronize());
  DK(cuMemcpyDtoH(D.data(),dD,M*N*4));
  cuMemFree(dA);cuMemFree(dB);cuMemFree(dD);
  double mx=0; for(int i=0;i<M*N;i++){double e=fabs(D[i]-ref[i]); if(e>mx)mx=e;}
  return mx;
}

int main(){
  DK(cuInit(0)); CUdevice dev; DK(cuDeviceGet(&dev,0)); CUcontext ctx; DK(cuDevicePrimaryCtxRetain(&ctx,dev)); DK(cuCtxSetCurrent(ctx));
  CUfunction fbf=load(dev,"bf16","gemm_bf16"), ff16=load(dev,"f16","gemm_f16");
  int shapes[][3]={{16,8,16},{16,16,16},{32,24,48},{64,64,64},{17,9,31},{128,128,256},{1,1,16},{100,50,200}};
  int bad=0;
  for(auto&s:shapes){
    double ebf=run_shape(dev,fbf,s[0],s[1],s[2],true);
    double tol_bf = s[2]<=256?1e-1:5e-1;  // bf16 ~3 mantissa bits; scale with K
    printf("bf16 %4dx%4dx%4d  maxerr=%.4g  %s\n",s[0],s[1],s[2],ebf, ebf<tol_bf?"ok":"BAD"); if(ebf>=tol_bf)bad++;
  }
  for(auto&s:shapes){
    double ef=run_shape(dev,ff16,s[0],s[1],s[2],false);
    double tol_f = s[2]<=256?1e-2:5e-2;
    printf("f16  %4dx%4dx%4d  maxerr=%.4g  %s\n",s[0],s[1],s[2],ef, ef<tol_f?"ok":"BAD"); if(ef>=tol_f)bad++;
  }
  printf("RESULT: %s\n", bad==0?"PASS":"FAIL");
  return bad?1:0;
}
