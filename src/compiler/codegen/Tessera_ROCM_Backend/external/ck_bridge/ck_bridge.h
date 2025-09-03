#pragma once
#include <cstdint>
namespace tessera::ck {
struct GemmConfig { int64_t M,N,K, lda,ldb,ldc; bool rowMA=true,rowMB=true,rowMC=true; };
bool run_gemm_fp16(const void* A, const void* B, void* C, GemmConfig cfg);
} // namespace tessera::ck
