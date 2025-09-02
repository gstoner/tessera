#include "tprof/tprof_runtime.h"

namespace tprof {

bool cupti_init() {
#ifdef TPROF_WITH_CUPTI
  // TODO: wire CUPTI initialization
  return true;
#else
  return false;
#endif
}

void cupti_shutdown() {
#ifdef TPROF_WITH_CUPTI
  // TODO: CUPTI finalize
#endif
}

} // namespace tprof
