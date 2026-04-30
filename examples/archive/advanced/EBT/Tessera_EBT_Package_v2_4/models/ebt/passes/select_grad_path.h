#pragma once
namespace tessera { namespace ebt {
// Register a pass that swaps autodiff VJP grad with custom JVP calls when enabled.
void registerEBTSelectGradPathPass();
}} // ns
