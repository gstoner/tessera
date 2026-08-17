// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-legality)' -verify-diagnostics %s
//
// SO-2: a structured LDS operation may not carry an anonymous pipeline
// state.  The ROCm consumer must be able to resolve the state back to a
// producer/consumer tile.role declaration before target conversion.

func.func @anonymous_lds_pipeline_state_is_rejected() {
  %state = tile.pipeline_init {
    depth = 2 : i64,
    stage = 0 : i64,
    phase = 1 : i64,
    role = "producer"
  } : !tile.pipeline_state
  %buffer = tile.alloc {
    space = "smem",
    bytes = 4096 : i64,
    layout = #tile.layout<shard = [64, 32] : [32, 1] on ["lds", "waveid"],
                          replica = [] : [] on [], offset = 0>
  } : !tile.buffer

  // expected-error @+1 {{ROCM_WAVE_LDS_ROLE_UNRESOLVED}}
  "test.rocm_lds_role_probe"(%buffer, %state)
      : (!tile.buffer, !tile.pipeline_state) -> ()
  tile.dealloc %buffer : !tile.buffer
  return
}
