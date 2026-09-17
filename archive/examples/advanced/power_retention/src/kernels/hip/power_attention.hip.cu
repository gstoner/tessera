
#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__)
// ===== MI300X gfx94*: MFMA projection scaffold (BF16->FP32) =====
static __device__ inline void mfma_m64n64k16_bf16_fp32(float* c, const void* a, const void* b){
  // Guarded inline assembly placeholder for gfx94; replace with real 'v_mfma_f32_16x16x16bf16' sequences.
  asm volatile("// mfma placeholder");
}
#endif

extern "C" __global__ void hip_noop() {}
