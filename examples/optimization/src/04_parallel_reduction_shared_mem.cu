
// CUDA block reduction with warp shuffles + shared memory. Launch with N large.
#include <cstdio>
#include <vector>
#include <random>

__inline__ __device__ float warp_reduce_sum(float val){
  for (int offset=16; offset>0; offset/=2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void block_reduce(const float* __restrict__ x, float* __restrict__ out, int N){
  extern __shared__ float smem[];
  float sum = 0.f;
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += gridDim.x*blockDim.x)
    sum += x[i];
  sum = warp_reduce_sum(sum);
  if ((threadIdx.x & 31)==0) smem[threadIdx.x>>5] = sum;
  __syncthreads();
  if (threadIdx.x < blockDim.x/32){
    float v = smem[threadIdx.x];
    v = warp_reduce_sum(v);
    if (threadIdx.x==0) out[blockIdx.x] = v;
  }
}

int main(){
  const int N = 1<<24;
  std::vector<float> h(N,1.0f);
  float *dX,*dY;
  cudaMalloc(&dX,N*sizeof(float));
  cudaMalloc(&dY,1024*sizeof(float));
  cudaMemcpy(dX,h.data(),N*sizeof(float),cudaMemcpyHostToDevice);
  dim3 blk(256), gr(1024);
  size_t sh = blk.x/32*sizeof(float);
  block_reduce<<<gr,blk,sh>>>(dX,dY,N);
  std::vector<float> partial(1024);
  cudaMemcpy(partial.data(), dY, 1024*sizeof(float), cudaMemcpyDeviceToHost);
  double s=0; for(auto v:partial) s+=v;
  printf("sum ~ %.0f (expected %d)\\n", s, N);
  cudaFree(dX); cudaFree(dY);
  return 0;
}
