
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static void ck(cudaError_t e, const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(1);} }

// Applies Rotary Position Embeddings (RoPE) to Q or K matrix of shape [S, D], D even.
// q_out[i, 2j:2j+2] = [ qcos - qsin ; qsin + qcos ] using angle = (i * inv_theta^(2j/D))
__global__ void rope_apply_kernel(const float* __restrict__ q_in,
                                  float* __restrict__ q_out,
                                  int S, int D,
                                  float base_theta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // sequence index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // half-dim index (j over D/2)
  if (i >= S || 2*j+1 >= D) return;

  float pos = float(i);
  float dim = float(2*j);
  // theta_j = base^( - dim / D )
  float angle = pos * powf(base_theta, - dim / float(D));
  float cs = cosf(angle);
  float sn = sinf(angle);

  int idx0 = i*D + 2*j;
  int idx1 = idx0 + 1;

  float x0 = q_in[idx0];
  float x1 = q_in[idx1];

  // [x0', x1'] = [x0 * cs - x1 * sn, x0 * sn + x1 * cs]
  q_out[idx0] = x0 * cs - x1 * sn;
  q_out[idx1] = x0 * sn + x1 * cs;
}

int main(int argc, char** argv){
  // CLI: --S, --D, --theta
  int S=4096, D=128;
  float theta=10000.f;
  for (int i=1;i<argc;++i){
    if (!strcmp(argv[i],"--S") && i+1<argc) S = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--D") && i+1<argc) D = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--theta") && i+1<argc) theta = atof(argv[++i]);
  }
  size_t sz = (size_t)S*D;
  float *in=nullptr, *out=nullptr;
  ck(cudaMalloc(&in, sz*sizeof(float)), "malloc in");
  ck(cudaMalloc(&out, sz*sizeof(float)), "malloc out");
  ck(cudaMemset(in, 0, sz*sizeof(float)), "memset in");
  dim3 block(128,2);
  dim3 grid((S+block.x-1)/block.x, (D/2 + block.y-1)/block.y);
  // warmup
  rope_apply_kernel<<<grid,block>>>(in,out,S,D,theta);
  ck(cudaDeviceSynchronize(),"sync");
  // timed
  const int iters=20;
  float ms_total=0.f;
  for (int it=0; it<iters; ++it){
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    rope_apply_kernel<<<grid,block>>>(in,out,S,D,theta);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0.f; cudaEventElapsedTime(&ms,s,e);
    ms_total += ms;
    cudaEventDestroy(s); cudaEventDestroy(e);
  }
  printf("{\"S\":%d,\"D\":%d,\"theta\":%.1f,\"ms_avg\":%.6f}\n", S,D,theta, ms_total/iters);
  cudaFree(in); cudaFree(out);
  return 0;
}
