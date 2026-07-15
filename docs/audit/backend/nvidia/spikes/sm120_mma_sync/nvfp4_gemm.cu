// NVFP4 (e2m1 + ue4m3 block scale) m16n8k64 warp MMA numerical oracle.
// One warp: D[16x8] f32 = A[16x64] @ B[64x8], fp4 e2m1 operands, UNIT block scales.
// Host builds per-lane fragments per the m16n8k64 fp4 layout, uploads them, the
// kernel runs the block-scale mma, host gathers D and compares to a fp4 reference.
// PTX ISA 9.3 grounds the 4X selector ABI: all four scale bytes participate,
// byte-id is zero, A is supplied by the lower lane pair and B by lane 0.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define M 16
#define N 8
#define K 64
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s @%d:%s\n",#x,__LINE__,cudaGetErrorString(e));return 1;}}while(0)

// e2m1 magnitudes by (exp,mant): see PTX fp4. code = s<<3 | e<<1 | m.
__host__ __device__ static float e2m1_mag(int e,int m){
  if(e==0) return m?0.5f:0.0f;
  return (1.0f + 0.5f*m) * (float)(1<<(e-1));
}
__host__ static float e2m1_dec(unsigned code){
  int s=(code>>3)&1,e=(code>>1)&3,m=code&1; float v=e2m1_mag(e,m); return s?-v:v;
}
__host__ static unsigned e2m1_enc(float f){
  int s=f<0; float a=fabsf(f); unsigned best=0; float bd=1e30f;
  for(int c=0;c<8;c++){ float v=e2m1_mag((c>>1)&3,c&1); float d=fabsf(a-v); if(d<bd){bd=d;best=c;} }
  return (s<<3)|best;
}

// ue4m3 scale = 1.0 -> exp=bias(7), mant=0 -> 0b0111000 = 0x38.
#define UE4M3_ONE 0x38u

__host__ static float ue4m3_dec(unsigned code) {
  unsigned exponent = (code >> 3) & 0xf, mantissa = code & 7;
  if (exponent == 0)
    return ldexpf((float)mantissa / 8.0f, -6);
  return ldexpf(1.0f + (float)mantissa / 8.0f, (int)exponent - 7);
}

__host__ static unsigned pack_scale4(const unsigned values[4]) {
  return values[0] | (values[1] << 8) | (values[2] << 16) |
         (values[3] << 24);
}

__global__ void nvfp4_tile(const unsigned* A, const unsigned* B,
                           const unsigned* SFa, const unsigned* SFb, float* D) {
  int lane=threadIdx.x;
  unsigned a0=A[lane*4+0],a1=A[lane*4+1],a2=A[lane*4+2],a3=A[lane*4+3];
  unsigned b0=B[lane*2+0],b1=B[lane*2+1];
  unsigned sfa=SFa[lane], sfb=SFb[lane];
  float d0=0,d1=0,d2=0,d3=0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
    "f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
    "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
      "r"(sfa), "n"(0), "n"(0), "r"(sfb), "n"(0), "n"(0));
  D[lane*4+0]=d0; D[lane*4+1]=d1; D[lane*4+2]=d2; D[lane*4+3]=d3;
}

int main(int argc, char** argv){
  bool mapped_scales = argc > 1 && strcmp(argv[1], "mapped") == 0;
  unsigned scale_byte = (argc>1 && !mapped_scales)
      ? (unsigned)strtoul(argv[1],0,16) : UE4M3_ONE;
  unsigned scaleA[M][4], scaleB[4][N];
  const unsigned scale_codes[3] = {0x30u, 0x38u, 0x40u};
  for (int m=0;m<M;m++) for (int block=0;block<4;block++)
    scaleA[m][block] = mapped_scales ? scale_codes[(m + block) % 3] : scale_byte;
  for (int block=0;block<4;block++) for (int n=0;n<N;n++)
    scaleB[block][n] = mapped_scales ? scale_codes[(2*block + n) % 3] : scale_byte;
  // logical fp4 matrices (codes 0..15) + decoded float values
  unsigned Ac[M*K], Bc[K*N]; float Af[M*K], Bf[K*N], ref[M*N];
  srand(7);
  for(int i=0;i<M*K;i++){ float v=((rand()%13)-6)*0.5f; Ac[i]=e2m1_enc(v); Af[i]=e2m1_dec(Ac[i]); }
  for(int i=0;i<K*N;i++){ float v=((rand()%13)-6)*0.5f; Bc[i]=e2m1_enc(v); Bf[i]=e2m1_dec(Bc[i]); }
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float a=0; for(int k=0;k<K;k++) {
    int block=k/16;
    a+=Af[m*K+k]*ue4m3_dec(scaleA[m][block])*
       Bf[k*N+n]*ue4m3_dec(scaleB[block][n]);
  } ref[m*N+n]=a; }

  // Build per-lane fragments per hypothesised m16n8k64 fp4 layout.
  // a0:(row=gid,col=8tig+j) a1:(row=gid+8,..) a2:(row=gid,col=8tig+32+j) a3:(row=gid+8,..)
  // b0:(col=gid,row=8tig+j) b1:(col=gid,row=8tig+32+j); j=0..7 in bits[4j..4j+3].
  unsigned hA[32*4]={0}, hB[32*2]={0}, hSFa[32], hSFb[32];
  float hD[32*4];
  for(int lane=0;lane<32;lane++){
    int gid=lane>>2, tig=lane&3;
    auto packA=[&](int row,int col0)->unsigned{ unsigned w=0; for(int j=0;j<8;j++){int c=col0+j; w|=(Ac[row*K+c]&0xf)<<(4*j);} return w; };
    hA[lane*4+0]=packA(gid,   8*tig);
    hA[lane*4+1]=packA(gid+8, 8*tig);
    hA[lane*4+2]=packA(gid,   8*tig+32);
    hA[lane*4+3]=packA(gid+8, 8*tig+32);
    auto packB=[&](int col,int row0)->unsigned{ unsigned w=0; for(int j=0;j<8;j++){int r=row0+j; w|=(Bc[r*N+col]&0xf)<<(4*j);} return w; };
    hB[lane*2+0]=packB(gid, 8*tig);
    hB[lane*2+1]=packB(gid, 8*tig+32);
    unsigned a_values[4] = {0,0,0,0}, b_values[4] = {0,0,0,0};
    // thread-id-a=0 selects tig 0/1: lower lane supplies row gid, upper
    // lane supplies row gid+8. thread-id-b=0 selects tig 0 for column gid.
    if(tig==0) for(int block=0;block<4;block++) {
      a_values[block]=scaleA[gid][block]; b_values[block]=scaleB[block][gid];
    }
    if(tig==1) for(int block=0;block<4;block++)
      a_values[block]=scaleA[gid+8][block];
    hSFa[lane]=pack_scale4(a_values);
    hSFb[lane]=pack_scale4(b_values);
  }

  unsigned *dA,*dB,*dSa,*dSb; float* dD;
  CK(cudaMalloc(&dA,sizeof(hA))); CK(cudaMalloc(&dB,sizeof(hB)));
  CK(cudaMalloc(&dSa,sizeof(hSFa))); CK(cudaMalloc(&dSb,sizeof(hSFb))); CK(cudaMalloc(&dD,sizeof(hD)));
  CK(cudaMemcpy(dA,hA,sizeof(hA),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB,sizeof(hB),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSa,hSFa,sizeof(hSFa),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSb,hSFb,sizeof(hSFb),cudaMemcpyHostToDevice));
  nvfp4_tile<<<1,32>>>(dA,dB,dSa,dSb,dD);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hD,dD,sizeof(hD),cudaMemcpyDeviceToHost));

  // gather: d0:(gid,2tig) d1:(gid,2tig+1) d2:(gid+8,2tig) d3:(gid+8,2tig+1)
  float Dg[M*N];
  for(int lane=0;lane<32;lane++){ int gid=lane>>2,tig=lane&3;
    Dg[(gid)*N+2*tig]=hD[lane*4+0]; Dg[(gid)*N+2*tig+1]=hD[lane*4+1];
    Dg[(gid+8)*N+2*tig]=hD[lane*4+2]; Dg[(gid+8)*N+2*tig+1]=hD[lane*4+3]; }
  double mx=0; int bad=0;
  for(int i=0;i<M*N;i++){ double e=fabs(Dg[i]-ref[i]); if(e>mx)mx=e; if(e>1e-3)bad++; }
  printf("NVFP4 m16n8k64 block_scale (%s scales) data-path test\n",
         mapped_scales ? "mapped non-uniform" : "uniform");
  printf("max abs error vs fp4 ref = %g, elements off = %d/%d\n", mx, bad, M*N);
  printf("sample D[0..3]=%.3f %.3f %.3f %.3f  ref=%.3f %.3f %.3f %.3f\n",
         Dg[0],Dg[1],Dg[2],Dg[3], ref[0],ref[1],ref[2],ref[3]);
  printf("RESULT: %s\n", bad==0?"PASS (data + UE4M3 scale ABI verified on sm_120a)":"FAIL");
  return bad?1:0;
}
