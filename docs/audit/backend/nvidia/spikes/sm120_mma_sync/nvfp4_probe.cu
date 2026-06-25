// Availability + grammar probe for the warp-level NVFP4 block-scale MMA on sm_120a.
// Compile: nvcc -arch=sm_120a -ptx nvfp4_probe.cu   (ptxas via -cubin to assemble)
// We try the documented m16n8k64 mxf4nvf4 block_scale form and let ptxas judge.
#include <cstdint>

extern "C" __global__ void probe(const unsigned* Ain, const unsigned* Bin,
                                 const unsigned* SFa, const unsigned* SFb,
                                 float* Dout) {
  int lane = threadIdx.x;
  unsigned a0=Ain[lane*4+0],a1=Ain[lane*4+1],a2=Ain[lane*4+2],a3=Ain[lane*4+3];
  unsigned b0=Bin[lane*2+0],b1=Bin[lane*2+1];
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
  Dout[lane*4+0]=d0; Dout[lane*4+1]=d1; Dout[lane*4+2]=d2; Dout[lane*4+3]=d3;
}
