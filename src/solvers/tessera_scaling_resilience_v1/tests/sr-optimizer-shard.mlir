// RUN: tessera-opt-sr -tessera-optimizer-shard %s | FileCheck %s
%p = "tessera.param"() : () -> tensor<*xf16>
%m = "tessera.opt.state"() {kind = "adam_m"} : () -> !tessera.opt.state
"tessera.optimizer.shard"(%p, %m) { axis = "data", policy = "zero2" } :
  (tensor<*xf16>, !tessera.opt.state) -> (tensor<*xf16>, !tessera.opt.state)
// CHECK: zero2