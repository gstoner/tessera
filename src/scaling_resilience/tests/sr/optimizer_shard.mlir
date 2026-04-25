// RUN: tessera-opt -tessera-optimizer-shard %s | FileCheck %s --check-prefix=SHARD
%p = "tessera.param"() : () -> tensor<*xf16>
%m = "tessera.opt.state"() {kind = "adam_m"} : () -> !tessera.opt.state
"tessera.optimizer.shard"(%p, %m) { axis = "data", policy = "zero2" } :
  (tensor<*xf16>, !tessera.opt.state) -> (tensor<*xf16>, !tessera.opt.state)
// SHARD: sr.sharded