#pragma once
#include <cstdint>
namespace tessera { namespace ebt {
// Creates a pass that materializes candidate (K) and step (T) loops in the IR.
// Impl note: match the high-level EBT pattern and emit scf.for with attrs.
void registerEBTMaterializeLoopsPass();
}} // ns
