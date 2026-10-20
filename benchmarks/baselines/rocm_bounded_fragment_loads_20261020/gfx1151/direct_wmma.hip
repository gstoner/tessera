#include <hip/hip_runtime.h>
#include <cstdint>
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Bits16 = short __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
__global__ void direct_half(const _Float16 *a,const _Float16 *b,float *o) {
  int lane=threadIdx.x%16, group=threadIdx.x/16;
  Half16 av,bv; Float8 zero={0,0,0,0,0,0,0,0};
  for(int j=0;j<16;++j) {av[j]=a[lane*16+j];bv[j]=b[j*16+lane];}
  Float8 d=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av,bv,zero);
  for(int j=0;j<8;++j)o[(2*j+group)*16+lane]=d[j];
}
__global__ void direct_bfloat(const short *a,const short *b,float *o) {
  int lane=threadIdx.x%16, group=threadIdx.x/16;
  Bits16 av,bv; Float8 zero={0,0,0,0,0,0,0,0};
  for(int j=0;j<16;++j) {av[j]=a[lane*16+j];bv[j]=b[j*16+lane];}
  Float8 d=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av,bv,zero);
  for(int j=0;j<8;++j)o[(2*j+group)*16+lane]=d[j];
}
extern "C" int direct_run(const void *a,const void *b,float *o,int bf) {
  void *aa=nullptr,*bb=nullptr,*oo=nullptr;
  auto result=hipMalloc(&aa,512);if(result)return int(result);
  result=hipMalloc(&bb,512);if(result){hipFree(aa);return int(result);}
  result=hipMalloc(&oo,1024);if(result){hipFree(aa);hipFree(bb);return int(result);}
  result=hipMemcpy(aa,a,512,hipMemcpyHostToDevice);
  if(!result)result=hipMemcpy(bb,b,512,hipMemcpyHostToDevice);
  if(!result){if(bf)hipLaunchKernelGGL(direct_bfloat,dim3(1),dim3(32),0,0,(const short*)aa,(const short*)bb,(float*)oo);else hipLaunchKernelGGL(direct_half,dim3(1),dim3(32),0,0,(const _Float16*)aa,(const _Float16*)bb,(float*)oo);result=hipGetLastError();}
  if(!result)result=hipDeviceSynchronize();
  if(!result)result=hipMemcpy(o,oo,1024,hipMemcpyDeviceToHost);
  hipFree(aa);hipFree(bb);hipFree(oo);return int(result);
}
