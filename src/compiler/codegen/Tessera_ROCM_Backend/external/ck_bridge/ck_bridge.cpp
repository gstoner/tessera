#include "ck_bridge.h"
#include <iostream>
#ifdef TESSERA_HAVE_CK
// Include CK headers if available; example only
// #include <ck/library/tensor_operation_instance/gpu/device_gemm_instance.hpp>
#endif

namespace tessera::ck {
bool run_gemm_fp16(const void* A, const void* B, void* C, GemmConfig cfg){
#ifdef TESSERA_HAVE_CK
  // TODO: Create and run a concrete CK DeviceGemm for row/row/row layouts.
  (void)A; (void)B; (void)C; (void)cfg;
  std::cout << "[CK] running DeviceGemm FP16\n";
  return true;
#else
  std::cerr << "[CK] not available\n"; return false;
#endif
}
} // namespace tessera::ck
