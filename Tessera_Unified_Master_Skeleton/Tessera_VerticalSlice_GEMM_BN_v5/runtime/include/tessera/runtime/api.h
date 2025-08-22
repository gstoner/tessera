#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { TS_OK=0, TS_ERR_INVALID=1, TS_ERR_OOM=2, TS_ERR_BACKEND=3 } tsStatus;

typedef struct {
  void* ptr;
  int device;    // -1 = host, >=0 = CUDA device id (in this slice)
  int dtype;     // 0 = f32 (assumed)
  int rank;
  long long shape[8];
} tsTensor;

// Allocate/free on host or CUDA device (in this slice: host only for simplicity)
tsStatus tesseraAllocHost(tsTensor* t, int dtype, int rank, const long long* shape);
tsStatus tesseraFreeHost(tsTensor* t);

// Matmul: C = A @ B (rank-2)
tsStatus tesseraMatmul(const tsTensor* A, const tsTensor* B, tsTensor* C);

// BatchNorm (inference) along last dim
tsStatus tesseraBatchNorm(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                          const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y, float eps);

#ifdef __cplusplus
}
#endif
