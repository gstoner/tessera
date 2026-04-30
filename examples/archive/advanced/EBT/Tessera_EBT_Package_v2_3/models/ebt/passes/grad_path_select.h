#pragma once
namespace tessera { namespace ebt {
// Creates a pass that replaces tessera.autodiff.grad_y with ebt.energy_*_jvp when enabled.
void registerEBTSelectGradPathPass(/*bool preferJVP*/);
}} // ns
