
#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { TS_OK=0, TS_ERR_INVALID=1, TS_ERR_OOM=2, TS_ERR_BACKEND=3 } tsStatus;

typedef struct {
  void* ptr; int device; int dtype; int rank; long long shape[8];
} tsTensor;

// GEMM + BatchNorm runtime entry points (CPU or GPU-backend)
tsStatus tesseraMatmul(const tsTensor* A, const tsTensor* B, tsTensor* C);
tsStatus tesseraBatchNorm(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                          const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y, float eps);

#ifdef __cplusplus
}
#endif
