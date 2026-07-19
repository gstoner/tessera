#include <cuda_runtime.h>

// Compile-only CUDA Math API coverage. This proves that the named CUDA 13.3
// integer, conversion, and packed-SIMD representatives are accepted for
// sm_120a; it does not claim a Tessera Target-IR producer or runtime route.
extern "C" __global__ void tessera_sm120_intrinsic_surface(
    int a, int b, unsigned int ua, unsigned int ub, float x, unsigned int *out) {
  if (blockIdx.x || threadIdx.x)
    return;

  out[0] = static_cast<unsigned int>(abs(a) + min(a, b) + max(a, b));
  out[1] = __brev(ua) ^ __byte_perm(ua, ub, 0x5410u);
  out[2] = static_cast<unsigned int>(__clz(a) + __ffs(b) + __popc(ua));
  out[3] = __funnelshift_l(ua, ub, 37u);
  out[4] = static_cast<unsigned int>(__dp2a_lo(a, b, 7));
  out[5] = static_cast<unsigned int>(__dp4a(a, b, 11));
  out[6] = static_cast<unsigned int>(__float2int_rn(x));
  out[7] = static_cast<unsigned int>(__float2int_rd(x));
  out[8] = static_cast<unsigned int>(__float2int_ru(x));
  out[9] = static_cast<unsigned int>(__float2int_rz(x));
  out[10] = static_cast<unsigned int>(__float_as_int(__int_as_float(a)));
  out[11] = __vadd2(ua, ub);
  out[12] = __vadd4(ua, ub);
  out[13] = __vaddss4(ua, ub);
  out[14] = __vabsdiffs4(ua, ub);
  out[15] = __vcmpeq4(ua, ub);
}
