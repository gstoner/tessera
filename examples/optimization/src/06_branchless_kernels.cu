
// Branchless selection: y[i] = (x[i] > t) ? a : b
#include <cstdio>
__global__ void branchless_select(const float* x, float* y, float a, float b, float t, int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N){
    float xi = x[i];
    // Convert predicate to mask: 0.0f or 1.0f
    float m = __int2float_rn(__float_as_int(xi > t));
    y[i] = m*a + (1.0f - m)*b;
  }
}
int main(){ printf("Compile and run on CUDA device.\\n"); return 0; }
