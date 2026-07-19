#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <mma.h>

static_assert(sizeof(double) == 8 && sizeof(double2) == 16);
static_assert(sizeof(float) == 4 && sizeof(float4) == 16);
static_assert(sizeof(__half) == 2 && sizeof(__half2) == 4);
static_assert(sizeof(__nv_bfloat16) == 2 && sizeof(__nv_bfloat162) == 4);
static_assert(sizeof(__nv_fp8_e4m3) == 1 && sizeof(__nv_fp8x2_e4m3) == 2);
static_assert(sizeof(__nv_fp8_e5m2) == 1 && sizeof(__nv_fp8x4_e5m2) == 4);
static_assert(sizeof(__nv_fp6_e2m3) == 1 && sizeof(__nv_fp6x2_e2m3) == 2);
static_assert(sizeof(__nv_fp6_e3m2) == 1 && sizeof(__nv_fp6x4_e3m2) == 4);
static_assert(sizeof(__nv_fp4_e2m1) == 1 && sizeof(__nv_fp4x2_e2m1) == 1);
static_assert(sizeof(__nv_fp4x4_e2m1) == 2);

// Compilation proves CUDA's scalar/vector storage and conversion surface. It
// is deliberately separate from the Tensor Core fragment fixtures: TF32 is an
// fp32 math mode, while FP6/FP4 packed C++ types do not by themselves prove a
// matrix instruction, scale layout, or Tessera runtime ABI.
extern "C" __global__ void tessera_sm120_dtype_surface(
    const double *f64, const double2 *f64x2,
    const float *f32, const float4 *f32x4,
    const __half *f16, const __half2 *f16x2,
    const __nv_bfloat16 *bf16, const __nv_bfloat162 *bf16x2,
    const __nv_fp8_e4m3 *e4m3, const __nv_fp8x2_e4m3 *e4m3x2,
    const __nv_fp8_e5m2 *e5m2, const __nv_fp8x4_e5m2 *e5m2x4,
    const __nv_fp6_e2m3 *e2m3, const __nv_fp6x2_e2m3 *e2m3x2,
    const __nv_fp6_e3m2 *e3m2, const __nv_fp6x4_e3m2 *e3m2x4,
    const __nv_fp4_e2m1 *e2m1, const __nv_fp4x4_e2m1 *e2m1x4,
    float *tf32_out) {
  if (blockIdx.x || threadIdx.x)
    return;
  tf32_out[0] = nvcuda::wmma::__float_to_tf32(f32[0]);
  // Keep every ABI operand live without imposing arithmetic semantics on the
  // storage-only low-precision types.
  tf32_out[1] = static_cast<float>(
      f64[0] + f64x2[0].x + f32x4[0].x + __half2float(f16[0]) +
      __half2float(f16x2[0].x) + __bfloat162float(bf16[0]) +
      __bfloat162float(bf16x2[0].x));
  (void)e4m3; (void)e4m3x2; (void)e5m2; (void)e5m2x4;
  (void)e2m3; (void)e2m3x2; (void)e3m2; (void)e3m2x4;
  (void)e2m1; (void)e2m1x4;
}
