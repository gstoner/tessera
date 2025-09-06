#include "passes.h"
namespace tessera { namespace ebt {
void registerEBTCanonicalizePipeline() { /* TODO */ }
void registerEBTLowerPipeline() { /* TODO */ }
void registerEBTPasses() { registerEBTCanonicalizePipeline(); registerEBTLowerPipeline(); }
}} // namespace
