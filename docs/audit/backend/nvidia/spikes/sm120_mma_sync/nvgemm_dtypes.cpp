// Multi-dtype mma.sync GEMM prototype (NVRTC), across shapes.
// Covers: bf16 & f16 (m16n8k16), tf32 (m16n8k8), fp8 e4m3 & e5m2 (m16n8k32).
// D[M,N] f32 = A[M,K] @ B[K,N], row-major; ragged edges zero-padded.
// Each warp computes one 16x8 output tile, K-looped; execute-and-compare vs a
// host reference that quantizes inputs to the SAME dtype the GPU consumes.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda.h>
#include <nvrtc.h>

#define DK(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorString(r,&s);printf("CU %s @%d:%s\n",#x,__LINE__,s);exit(1);}}while(0)
#define NK(x) do{nvrtcResult r=(x); if(r!=NVRTC_SUCCESS){printf("NVRTC %s @%d:%s\n",#x,__LINE__,nvrtcGetErrorString(r));exit(1);}}while(0)

// ── kernels ──────────────────────────────────────────────────────────────────
// 16-bit (bf16/f16): m16n8k16, each .b32 packs 2 contiguous 16-bit elems.
static const char* kSrc16 = R"NVRTC(
extern "C" __global__ void gemm(const unsigned short* A, const unsigned short* B,
                                float* D, int M, int N, int K) {
  int mt=blockIdx.x*16, nt=blockIdx.y*8, lane=threadIdx.x, gid=lane>>2, tig=lane&3;
  float d0=0,d1=0,d2=0,d3=0;
  for (int k0=0;k0<K;k0+=16){
    auto la=[&](int r,int c)->unsigned{int rr=mt+r,cc=k0+c;
      unsigned lo=(rr<M&&cc<K)?A[rr*K+cc]:0u, hi=(rr<M&&cc+1<K)?A[rr*K+cc+1]:0u; return (hi<<16)|lo;};
    auto lb=[&](int r,int c)->unsigned{int rr=k0+r,cc=nt+c;
      unsigned lo=(rr<K&&cc<N)?B[rr*N+cc]:0u, hi=(rr+1<K&&cc<N)?B[(rr+1)*N+cc]:0u; return (hi<<16)|lo;};
    unsigned a0=la(gid,2*tig),a1=la(gid+8,2*tig),a2=la(gid,2*tig+8),a3=la(gid+8,2*tig+8);
    unsigned b0=lb(2*tig,gid),b1=lb(2*tig+8,gid);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.TYPE.TYPE.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<N)D[rr*N+cc]=v;};
  st(gid,2*tig,d0);st(gid,2*tig+1,d1);st(gid+8,2*tig,d2);st(gid+8,2*tig+1,d3);
}
)NVRTC";

// tf32: m16n8k8, A holds 4 tf32 (one per .b32, the f32 bits), B holds 2.
static const char* kSrcTf32 = R"NVRTC(
extern "C" __global__ void gemm(const float* A, const float* B,
                                float* D, int M, int N, int K) {
  int mt=blockIdx.x*16, nt=blockIdx.y*8, lane=threadIdx.x, gid=lane>>2, tig=lane&3;
  float d0=0,d1=0,d2=0,d3=0;
  for (int k0=0;k0<K;k0+=8){
    auto la=[&](int r,int c)->unsigned{int rr=mt+r,cc=k0+c; float v=(rr<M&&cc<K)?A[rr*K+cc]:0.f; unsigned u; memcpy(&u,&v,4); return u;};
    auto lb=[&](int r,int c)->unsigned{int rr=k0+r,cc=nt+c; float v=(rr<K&&cc<N)?B[rr*N+cc]:0.f; unsigned u; memcpy(&u,&v,4); return u;};
    unsigned a0=la(gid,tig),a1=la(gid+8,tig),a2=la(gid,tig+4),a3=la(gid+8,tig+4);
    unsigned b0=lb(tig,gid),b1=lb(tig+4,gid);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<N)D[rr*N+cc]=v;};
  st(gid,2*tig,d0);st(gid,2*tig+1,d1);st(gid+8,2*tig,d2);st(gid+8,2*tig+1,d3);
}
)NVRTC";

// fp8 (e4m3/e5m2): m16n8k32, each .b32 packs 4 fp8 bytes.
static const char* kSrcF8 = R"NVRTC(
extern "C" __global__ void gemm(const unsigned char* A, const unsigned char* B,
                                float* D, int M, int N, int K) {
  int mt=blockIdx.x*16, nt=blockIdx.y*8, lane=threadIdx.x, gid=lane>>2, tig=lane&3;
  float d0=0,d1=0,d2=0,d3=0;
  for (int k0=0;k0<K;k0+=32){
    auto la=[&](int r,int c)->unsigned{int rr=mt+r; unsigned w=0;
      for(int j=0;j<4;j++){int cc=k0+c+j; unsigned b=(rr<M&&cc<K)?A[rr*K+cc]:0u; w|=b<<(8*j);} return w;};
    auto lb=[&](int r,int c)->unsigned{int cc=nt+c; unsigned w=0;
      for(int j=0;j<4;j++){int rr=k0+r+j; unsigned b=(rr<K&&cc<N)?B[rr*N+cc]:0u; w|=b<<(8*j);} return w;};
    unsigned a0=la(gid,4*tig),a1=la(gid+8,4*tig),a2=la(gid,4*tig+16),a3=la(gid+8,4*tig+16);
    unsigned b0=lb(4*tig,gid),b1=lb(4*tig+16,gid);
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.TYPE.TYPE.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<N)D[rr*N+cc]=v;};
  st(gid,2*tig,d0);st(gid,2*tig+1,d1);st(gid+8,2*tig,d2);st(gid+8,2*tig+1,d3);
}
)NVRTC";

// ── host quant/dequant (match the bits the hardware consumes) ─────────────────
static unsigned short f2bf16(float f){unsigned x;memcpy(&x,&f,4);unsigned r=(x>>16)&1u,t=0x7fffu+r;x+=t;return (unsigned short)(x>>16);}
static float bf162f(unsigned short h){unsigned x=((unsigned)h)<<16;float f;memcpy(&f,&x,4);return f;}
static unsigned short f2f16(float f){unsigned x;memcpy(&x,&f,4);unsigned s=(x>>16)&0x8000u;int e=(int)((x>>23)&0xff)-112;unsigned m=x&0x7fffffu;
  if(e<=0)return(unsigned short)s; if(e>=31)return(unsigned short)(s|0x7c00u);
  unsigned mm=m>>13; if(m&0x1000u)mm++; return (unsigned short)(s|(e<<10)|mm);}
static float f162f(unsigned short h){unsigned s=(h&0x8000u)<<16;int e=(h>>10)&0x1f;unsigned m=h&0x3ffu;
  if(e==0){if(m==0){unsigned b=s;float f;memcpy(&f,&b,4);return f;} float f=(float)m/1024.f*6.1035e-5f; unsigned b; memcpy(&b,&f,4); b|=s; memcpy(&f,&b,4); return f;}
  unsigned e2=(unsigned)(e-15+127),b=s|(e2<<23)|(m<<13);float f;memcpy(&f,&b,4);return f;}
static float quant_tf32(float f){unsigned x;memcpy(&x,&f,4);unsigned r=(x>>13)&1u;x+= (0xfffu+r); x&=~0x1fffu; float o; memcpy(&o,&x,4); return o;}

// generic fp8 (ebits=exp, mbits=mantissa, bias) round-to-nearest-even, OCP-ish,
// no inf/NaN for our [-1,1] test range. Returns the byte; deq decodes it back.
static unsigned char f2fp8(float f, int ebits, int mbits, int bias){
  unsigned x; memcpy(&x,&f,4); unsigned s=(x>>31)&1u; int e=(int)((x>>23)&0xff)-127; unsigned m=x&0x7fffffu;
  unsigned sbit = s<<(ebits+mbits);
  int maxe=(1<<ebits)-1;                       // top exponent code (reserved here for clamp)
  if(((x>>23)&0xff)==0) return (unsigned char)sbit;             // zero/denorm input -> 0
  int eo=e+bias;
  unsigned full=(1u<<23)|m;                     // implicit leading 1
  int shift=23-mbits;
  unsigned mant=full>>shift, rem=full&((1u<<shift)-1), half=1u<<(shift-1);
  if(rem>half || (rem==half && (mant&1u))) mant++;             // round-to-nearest-even
  if(mant>=(2u<<mbits)){ mant>>=1; eo++; }                      // mantissa carry
  if(eo<=0) return (unsigned char)sbit;                         // underflow -> 0 (no denorms)
  if(eo>=maxe){ unsigned mm=(1u<<mbits)-1;                      // clamp to max normal
    return (unsigned char)(sbit|((unsigned)(maxe-1)<<mbits)|mm); }
  return (unsigned char)(sbit|((unsigned)eo<<mbits)|(mant&((1u<<mbits)-1)));
}
static float fp82f(unsigned char b, int ebits, int mbits, int bias){
  unsigned s=(b>>(ebits+mbits))&1u; unsigned e=(b>>mbits)&((1u<<ebits)-1); unsigned m=b&((1u<<mbits)-1);
  float val;
  if(e==0){ val = (float)m/(float)(1<<mbits) * powf(2.f, (float)(1-bias)); }
  else { val = (1.f + (float)m/(float)(1<<mbits)) * powf(2.f,(float)((int)e-bias)); }
  return s?-val:val;
}

// ── NVRTC load ────────────────────────────────────────────────────────────────
static CUfunction load(CUdevice dev, const char* src_in, const char* type){
  std::string src(src_in);
  if(type) for(size_t p;(p=src.find("TYPE"))!=std::string::npos;) src.replace(p,4,type);
  nvrtcProgram prog; NK(nvrtcCreateProgram(&prog,src.c_str(),"g.cu",0,0,0));
  int maj,min; DK(cuDeviceGetAttribute(&maj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev));
  DK(cuDeviceGetAttribute(&min,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev));
  char arch[40]; snprintf(arch,sizeof(arch),"--gpu-architecture=compute_%d%d",maj,min);
  const char* o[]={arch};
  nvrtcResult c=nvrtcCompileProgram(prog,1,o);
  if(c!=NVRTC_SUCCESS){size_t n;nvrtcGetProgramLogSize(prog,&n);std::vector<char>l(n);nvrtcGetProgramLog(prog,l.data());printf("NVRTC log:\n%s\n",l.data());exit(1);}
  size_t ps;NK(nvrtcGetPTXSize(prog,&ps));std::vector<char>ptx(ps);NK(nvrtcGetPTX(prog,ptx.data()));
  CUmodule mod;DK(cuModuleLoadData(&mod,ptx.data()));CUfunction fn;DK(cuModuleGetFunction(&fn,mod,"gemm"));
  nvrtcDestroyProgram(&prog);return fn;
}

enum Dt{BF16,F16,TF32,E4M3,E5M2};
static double run(CUdevice dev, CUfunction fn, Dt dt, int M,int N,int K){
  std::vector<float> fA(M*K),fB(K*N),ref(M*N,0),D(M*N,0);
  srand(M*131+N*17+K*7+dt);
  for(int i=0;i<M*K;i++) fA[i]=((rand()%201)-100)/100.f;
  for(int i=0;i<K*N;i++) fB[i]=((rand()%201)-100)/100.f;
  // quantize to the dtype, keep dequantized floats for the reference
  size_t esz; std::vector<unsigned char> qA,qB; std::vector<float> tA,tB;
  if(dt==TF32){ tA.resize(M*K); tB.resize(K*N);
    for(int i=0;i<M*K;i++){tA[i]=quant_tf32(fA[i]);} for(int i=0;i<K*N;i++){tB[i]=quant_tf32(fB[i]);}
  } else if(dt==BF16||dt==F16){ qA.resize(M*K*2); qB.resize(K*N*2); tA.resize(M*K); tB.resize(K*N);
    for(int i=0;i<M*K;i++){unsigned short h=dt==BF16?f2bf16(fA[i]):f2f16(fA[i]); memcpy(&qA[i*2],&h,2); tA[i]=dt==BF16?bf162f(h):f162f(h);}
    for(int i=0;i<K*N;i++){unsigned short h=dt==BF16?f2bf16(fB[i]):f2f16(fB[i]); memcpy(&qB[i*2],&h,2); tB[i]=dt==BF16?bf162f(h):f162f(h);}
  } else { int eb=dt==E4M3?4:5, mb=dt==E4M3?3:2, bias=dt==E4M3?7:15;
    qA.resize(M*K); qB.resize(K*N); tA.resize(M*K); tB.resize(K*N);
    for(int i=0;i<M*K;i++){unsigned char b=f2fp8(fA[i],eb,mb,bias); qA[i]=b; tA[i]=fp82f(b,eb,mb,bias);}
    for(int i=0;i<K*N;i++){unsigned char b=f2fp8(fB[i],eb,mb,bias); qB[i]=b; tB[i]=fp82f(b,eb,mb,bias);}
  }
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){float a=0;for(int k=0;k<K;k++)a+=tA[m*K+k]*tB[k*N+n];ref[m*N+n]=a;}
  CUdeviceptr dA,dB,dD; size_t bA,bB;
  if(dt==TF32){bA=M*K*4;bB=K*N*4;} else if(dt==BF16||dt==F16){bA=M*K*2;bB=K*N*2;} else {bA=M*K;bB=K*N;}
  DK(cuMemAlloc(&dA,bA));DK(cuMemAlloc(&dB,bB));DK(cuMemAlloc(&dD,M*N*4));
  if(dt==TF32){DK(cuMemcpyHtoD(dA,tA.data(),bA));DK(cuMemcpyHtoD(dB,tB.data(),bB));}
  else{DK(cuMemcpyHtoD(dA,qA.data(),bA));DK(cuMemcpyHtoD(dB,qB.data(),bB));}
  int mt=(M+15)/16,nt=(N+7)/8; void* args[]={&dA,&dB,&dD,&M,&N,&K};
  DK(cuLaunchKernel(fn,mt,nt,1,32,1,1,0,0,args,0));DK(cuCtxSynchronize());
  DK(cuMemcpyDtoH(D.data(),dD,M*N*4)); cuMemFree(dA);cuMemFree(dB);cuMemFree(dD);
  double mx=0,rel=0; for(int i=0;i<M*N;i++){double e=fabs(D[i]-ref[i]); if(e>mx)mx=e;}
  // relative-to-K-scale bound
  return mx;
}

int main(){
  DK(cuInit(0));CUdevice dev;DK(cuDeviceGet(&dev,0));CUcontext ctx;DK(cuDevicePrimaryCtxRetain(&ctx,dev));DK(cuCtxSetCurrent(ctx));
  CUfunction fbf=load(dev,kSrc16,"bf16"), ff16=load(dev,kSrc16,"f16"),
             ftf=load(dev,kSrcTf32,0), fe4=load(dev,kSrcF8,"e4m3"), fe5=load(dev,kSrcF8,"e5m2");
  struct{const char*n;Dt dt;CUfunction fn;double tol;} K[]={
    {"bf16",BF16,fbf,2e-1},{"f16",F16,ff16,5e-2},{"tf32",TF32,ftf,1e-2},
    {"e4m3",E4M3,fe4,3.0},{"e5m2",E5M2,fe5,6.0}};
  int shapes[][3]={{16,8,16},{16,16,16},{32,24,48},{64,64,64},{17,9,31},{128,128,256},{100,50,200}};
  int bad=0;
  for(auto&kd:K){
    printf("== %s ==\n", kd.n);
    for(auto&s:shapes){
      // align K to the dtype's k-step is handled by the kernel (zero-padded); use raw shapes
      double e=run(dev,kd.fn,kd.dt,s[0],s[1],s[2]);
      double tol=kd.tol*(s[2]>256?5:1);
      printf("  %4dx%4dx%4d maxerr=%.4g %s\n",s[0],s[1],s[2],e, e<tol?"ok":"BAD");
      if(!(e<tol))bad++;
    }
  }
  printf("RESULT: %s\n", bad==0?"PASS":"FAIL");
  return bad?1:0;
}
