#include "native_program_runtime.h"
#include "NVFP4ScaleBounds.h"

extern "C" int tessera_rocm_program_nvfp4_scale_bounds(
    const uint8_t *scales, uint64_t bytes, uint64_t rows, uint64_t columns,
    const int64_t *offsets, uint64_t offsetCount,
    uint8_t *bounds, uint64_t boundsBytes) try {
  return tessera::rocm::nvfp4ScaleBounds(scales,bytes,rows,columns,
                                      offsets,offsetCount,bounds,boundsBytes);
} catch (...) { return 12; }
