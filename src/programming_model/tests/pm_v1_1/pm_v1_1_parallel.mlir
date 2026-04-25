// RUN: tessera-opt %s -pm-v1_1-verify | FileCheck %s

module {
  // CHECK: schedule.mesh.define
  schedule.mesh.define @M dims = [2,4,2] axis_names = ["data","model","pipe"]

  // CHECK: schedule.mesh.region {{.*}} axis = "data"
  schedule.mesh.region @M { axis = "data" } {
    // CHECK: schedule.pipeline.region { schedule = "1f1b", micro_batches = 8 }
    schedule.pipeline.region { schedule = "1f1b", micro_batches = 8 } {
      // CHECK: schedule.stage
      schedule.stage @s0 { devices = ["gpu:0"] } { }
      schedule.stage @s1 { devices = ["gpu:1"] } { }
    }
  }

  // MoE hooks
  // CHECK: moe.plan
  moe.plan { a2a_bucket = 131072, pack_cast = "bf16->fp8" }
  // CHECK: moe.token_limiter.create
  %lim = moe.token_limiter.create { max_tokens_in_flight = 4, refill = 1 }
}
