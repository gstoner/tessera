#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { TS_OK=0, TS_ERR_INVALID=1, TS_ERR_OOM=2, TS_ERR_BACKEND=3 } tsStatus;

typedef struct {
  void* ptr;
  int device;    // -1 = host, >=0 = device id
  int dtype;     // 0 = f32
  int rank;
  long long shape[8];
} tsTensor;

typedef struct { void* impl; int device; } tsStream;

// Device management
tsStatus tesseraSetDevice(int device);

// Host allocations
tsStatus tesseraAllocHost(tsTensor* t, int dtype, int rank, const long long* shape);
tsStatus tesseraFreeHost(tsTensor* t);

// Device allocations (CUDA path)
tsStatus tesseraAllocDevice(tsTensor* t, int device, int dtype, int rank, const long long* shape);
tsStatus tesseraFreeDevice(tsTensor* t);

// Copies (size inferred from tensor shape/dtype)
tsStatus tesseraCopyHostToDevice(const tsTensor* srcHost, tsTensor* dstDevice);
tsStatus tesseraCopyDeviceToHost(const tsTensor* srcDevice, tsTensor* dstHost);

// Streams
tsStatus tesseraStreamCreate(int device, tsStream* out);
tsStatus tesseraStreamDestroy(tsStream* s);
tsStatus tesseraStreamSync(tsStream* s);

// Ops (Async + Sync wrappers)
tsStatus tesseraMatmulAsync(const tsTensor* A, const tsTensor* B, tsTensor* C, tsStream* stream);
tsStatus tesseraBatchNormAsync(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                               const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y,
                               float eps, tsStream* stream);

static inline tsStatus tesseraMatmul(const tsTensor* A, const tsTensor* B, tsTensor* C){
  tsStream s{nullptr, A?A->device:-1}; return tesseraMatmulAsync(A,B,C,&s) | tesseraStreamSync(&s);
}
static inline tsStatus tesseraBatchNorm(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                               const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y, float eps){
  tsStream s{nullptr, X?X->device:-1}; return tesseraBatchNormAsync(X,Mean,Var,Gamma,Beta,Y,eps,&s) | tesseraStreamSync(&s);
}

#ifdef __cplusplus
}
#endif
