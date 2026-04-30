
// H100 WGMMA sanity: run FMA fallback kernel and WGMMA kernel, compare rough outputs.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <random>

extern "C" void power_attn_forward_bf16_deg2(const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*, __nv_bfloat16*, int,int,int,int,int);
extern "C" void power_attn_forward_wgmma_bf16_deg2(const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*, __nv_bfloat16*, int,int,int,int,int);

int main(){
  int B=1,H=2,S=256,D=64,Dh=64;
  size_t sz = (size_t)B*H*S*D;
  std::vector<__nv_bfloat16> hQ(sz), hK(sz), hV(sz), hO1(sz), hO2(sz);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5,0.5);
  for (size_t i=0;i<sz;i++){
    hQ[i] = __float2bfloat16(dist(rng));
    hK[i] = __float2bfloat16(dist(rng));
    hV[i] = __float2bfloat16(dist(rng));
  }
  __nv_bfloat16 *dQ,*dK,*dV,*dO1,*dO2;
  cudaMalloc(&dQ, sz*sizeof(__nv_bfloat16));
  cudaMalloc(&dK, sz*sizeof(__nv_bfloat16));
  cudaMalloc(&dV, sz*sizeof(__nv_bfloat16));
  cudaMalloc(&dO1, sz*sizeof(__nv_bfloat16));
  cudaMalloc(&dO2, sz*sizeof(__nv_bfloat16));
  cudaMemcpy(dQ,hQ.data(),sz*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice);
  cudaMemcpy(dK,hK.data(),sz*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice);
  cudaMemcpy(dV,hV.data(),sz*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice);

  // Launches (grid over BH, simple shared size)
  dim3 grid(B*H), block(256);
  size_t shmem = (size_t)(D*(D+1)/2)*Dh*sizeof(float) + (size_t)(D*(D+1)/2)*sizeof(float) + (size_t)(D*(D+1)/2)*sizeof(float);

  // Fallback
  power_attn_forward_bf16_deg2<<<grid,block,shmem>>>(dQ,dK,dV,dO1,B,H,S,D,Dh);
  cudaDeviceSynchronize();
  // WGMMA
  #if __CUDA_ARCH__ >= 900 && defined(TESSERA_ENABLE_WGMMA)
  power_attn_forward_wgmma_bf16_deg2<<<grid,block,shmem>>>(dQ,dK,dV,dO2,B,H,S,D,Dh);
  cudaDeviceSynchronize();
  #else
  cudaMemset(dO2,0,sz*sizeof(__nv_bfloat16));
  #endif

  cudaMemcpy(hO1.data(), dO1, sz*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(hO2.data(), dO2, sz*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  // Compare a few elements
  double max_abs=0, mean_abs=0;
  for (int i=0;i<1000;i++){
    size_t idx = i % sz;
    float a = __bfloat162float(hO1[idx]);
    float b = __bfloat162float(hO2[idx]);
    double d = fabs(a-b);
    max_abs = std::max(max_abs, d);
    mean_abs += d;
  }
  mean_abs /= 1000.0;
  printf("Sanity: mean_abs=%.6f max_abs=%.6f\\n", mean_abs, max_abs);

  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO1); cudaFree(dO2);
  return 0;
}
