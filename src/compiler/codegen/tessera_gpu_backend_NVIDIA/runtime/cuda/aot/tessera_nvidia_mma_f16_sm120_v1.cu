// Canonical source for the version-1 SM120 f16 production GEMM artifact.
//
// CMake compiles this exact file to the adjacent versioned cubin and embeds the
// same text in libtessera_nvidia_gemm.so for its NVRTC fallback.  Keep the
// entry name and physical ABI in lockstep with tessera_nvidia_mma_gemm_f16.
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<N)D[rr*N+cc]=v;};
  st(gid,2*tig,d0);st(gid,2*tig+1,d1);st(gid+8,2*tig,d2);st(gid+8,2*tig+1,d3);
}
