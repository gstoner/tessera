// stage_probe.cu -- copy-only validation for the INT4 three-stage pipeline.
//
// It moves three 16x64 A + 64x8 B packed-int4 tiles through the exact cp.async
// ring layout used by int4_native_k64_3stage, then compares every staged word
// on the host.  No MMA instruction is present: a failure here is transport or
// synchronization, not a fragment-map ambiguity.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;} } while(0)

__device__ __forceinline__ void cp16(void* dst,const void* src){
  unsigned smem=(unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(smem),"l"(src));
}
__device__ __forceinline__ void commit(){asm volatile("cp.async.commit_group;");}
__device__ __forceinline__ void wait_all(){asm volatile("cp.async.wait_all;");}

__global__ void copy_three(const uint32_t* A,const uint32_t* B,uint32_t* out){
  __shared__ uint32_t stage[3][192];
  int lane=threadIdx.x, Kw=32; // K=256 logical int4 elements => 32 words.
  for(int tile=0;tile<3;tile++){
    int kw=tile*8;
    for(int chunk=lane;chunk<48;chunk+=32){
      const uint32_t* src;
      if(chunk<32) { int row=chunk/2, half=chunk%2; src=A+row*Kw+kw+half*4; }
      else {int q=chunk-32; src=B+(q/2)*Kw+kw+(q%2)*4;}
      cp16(&stage[tile][chunk*4],src);
    }
    commit();
  }
  wait_all(); __syncthreads();
  for(int slot=0;slot<3;slot++)
    for(int i=lane;i<192;i+=32) out[slot*192+i]=stage[slot][i];
}

int main(){
  constexpr int Kw=32;
  std::vector<uint32_t> a(16*Kw),b(8*Kw),got(3*192),expect(3*192);
  for(size_t i=0;i<a.size();i++) a[i]=0xA0000000u+(uint32_t)i;
  for(size_t i=0;i<b.size();i++) b[i]=0xB0000000u+(uint32_t)i;
  for(int tile=0;tile<3;tile++){
    int kw=tile*8;
    for(int row=0;row<16;row++) for(int w=0;w<8;w++) expect[tile*192+row*8+w]=a[row*Kw+kw+w];
    for(int col=0;col<8;col++) for(int w=0;w<8;w++) expect[tile*192+128+col*8+w]=b[col*Kw+kw+w];
  }
  uint32_t *da,*db,*dg; CK(cudaMalloc(&da,a.size()*4));CK(cudaMalloc(&db,b.size()*4));CK(cudaMalloc(&dg,got.size()*4));
  CK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
  copy_three<<<1,32>>>(da,db,dg); CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(got.data(),dg,got.size()*4,cudaMemcpyDeviceToHost));
  int errors=0; for(size_t i=0;i<got.size();i++) if(got[i]!=expect[i]){if(errors++<4)fprintf(stderr,"word %zu got=%08x expect=%08x\n",i,got[i],expect[i]);}
  cudaFree(da);cudaFree(db);cudaFree(dg);
  printf("stage_copy_3 %s (%d mismatches)\n",errors?"WRONG":"OK",errors); return errors?1:0;
}
