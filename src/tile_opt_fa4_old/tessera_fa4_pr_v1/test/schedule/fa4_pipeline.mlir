// RUN: tessera-opt %s -tessera-lower-schedule | FileCheck %s
tessera.schedule @fa4_pipeline {
  %w_load  = tessera.schedule.warp "load", 1
  %w_mma   = tessera.schedule.warp "mma", 1
  %w_smx   = tessera.schedule.warp "softmax", 8
  %w_corr  = tessera.schedule.warp "correction", 4
  %w_epi   = tessera.schedule.warp "epilogue", 2
  tessera.schedule.pipe %w_load, %w_mma, %w_smx, %w_corr, %w_epi { buffering = {K=3, V=3, S=2, O=2} }
  tessera.schedule.policy "persistent", 1, "static"
}
// CHECK: async.execute
// CHECK: barrier
