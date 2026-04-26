#include "tprof/tprof_runtime.h"
#ifdef TPROF_WITH_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
namespace tprof {
void nvtx_push(const char* name) {
#ifdef TPROF_WITH_NVTX
  nvtxRangePushA(name ? name : "tprof");
#else
  (void)name;
#endif
}
void nvtx_pop() {
#ifdef TPROF_WITH_NVTX
  nvtxRangePop();
#endif
}
} // namespace tprof
