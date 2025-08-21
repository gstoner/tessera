#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  TS_OK = 0,
  TS_ERR_INVALID = 1,
  TS_ERR_OOM = 2,
  TS_ERR_BACKEND = 3,
} tsStatus;

typedef struct {
  void* ptr;
  int   device;   // -1 for host
  int   dtype;    // enum placeholder
  int   rank;
  int64_t shape[8];
} tsTensor;

tsStatus tesseraInit(void);
tsStatus tesseraShutdown(void);

// memory
tsStatus tesseraAlloc(tsTensor* t, int device, int dtype, int rank, const int64_t* shape);
tsStatus tesseraFree(tsTensor* t);

// streams
typedef struct { int device; int id; } tsStream;
tsStatus tesseraStreamCreate(int device, tsStream* out);
tsStatus tesseraStreamDestroy(tsStream* s);
tsStatus tesseraStreamSync(tsStream* s);

// simple op stub (CPU fallback)
tsStatus tesseraMatmul(const tsTensor* A, const tsTensor* B, tsTensor* C);

#ifdef __cplusplus
}
#endif
