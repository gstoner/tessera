#include "select_grad_path.h"
// Implementation sketch:
// - Pattern: m_Op("tessera.autodiff.grad_y", operands=[h,y,...]) → replace with
//   a call/callee to @energy_bilinear_jvp or @energy_mlp_jvp based on annotations.
// - Annotate with {ebt.grad_path = "jvp"} on the site for debugability.
// - When disabled, leave the autodiff op intact and annotate {ebt.grad_path = "vjp"}.
namespace tessera { namespace ebt {
void registerEBTSelectGradPathPass() {}
}} // ns
