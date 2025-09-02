#include "tprof/tprof_runtime.h"
namespace tprof {
bool cupti_init() {
#ifdef TPROF_WITH_CUPTI
  // TODO: real CUPTI init
  return true;
#else
  return false;
#endif
}
void cupti_shutdown() {
#ifdef TPROF_WITH_CUPTI
  // TODO: finalize
#endif
}
} // namespace tprof
