#include "grad_path_select.h"

// Scaffold: In your tree, implement a RewritePattern that matches
//   tessera.autodiff.grad_y(@energy_primal(...))
// and replaces with a call to the appropriate @energy_*_jvp(...) if preferJVP.
// Use MLIR pattern infra and set benefit so it runs after autodiff materialization.

namespace tessera { namespace ebt {
void registerEBTSelectGradPathPass(/*bool preferJVP*/) {}
}} // ns
