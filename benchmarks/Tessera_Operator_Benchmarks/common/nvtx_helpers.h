#pragma once
#if defined(OPBENCH_WITH_NVTX)
#include <nvToolsExt.h>
struct NvtxRange {
  nvtxRangeId_t id;
  NvtxRange(const char* name){ id = nvtxRangeStartA(name); }
  ~NvtxRange(){ nvtxRangeEnd(id); }
};
#else
struct NvtxRange {
  NvtxRange(const char*) {}
};
#endif
