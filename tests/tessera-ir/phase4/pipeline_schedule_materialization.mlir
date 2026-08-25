// RUN: tessera-opt --tessera-pipeline --allow-unregistered-dialect --verify-each=false %s | FileCheck %s
//
// The verified 1F1B Schedule Object is already materialized by the producer.
// The lowering pipeline preserves its dependency carrier and stamps the same
// digest on the owning function and inserted communication operations.

module attributes {
  tessera.schedule_digest = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  tessera.pipeline_schedule_schema = "tessera.pipeline_schedule.v1",
  tessera.pipeline_steps = [{action_id = "root", clock = 0, depends_on = [], micro_batch = 0, phase = "F", rank = 0, stage = 0}],
  tessera.pp_num_stages = 2,
  tessera.pp_num_micro_batches = 3,
  tessera.pp_interleaved = false
} {
  func.func @pipeline(%x: tensor<64x128xbf16>,
                      %w0: tensor<128x256xbf16>,
                      %w1: tensor<256x128xbf16>) -> tensor<64x128xbf16> {
    %a = "tessera.matmul"(%x, %w0) :
      (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
    %b = "tessera.matmul"(%a, %w1) :
      (tensor<64x256xbf16>, tensor<256x128xbf16>) -> tensor<64x128xbf16>
    return %b : tensor<64x128xbf16>
  }
}

// CHECK: sym_name = "pipeline"
// CHECK: tessera.pipeline.send
// CHECK-SAME: tessera.schedule_digest = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
// CHECK: tessera.pipeline.recv
// CHECK-SAME: tessera.schedule_digest = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
// CHECK: }) {tessera.schedule_digest = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
// CHECK: }) {tessera.pipeline_schedule_schema = "tessera.pipeline_schedule.v1"
// CHECK-SAME: tessera.pipeline_steps = [{action_id = "root"
// CHECK-SAME: tessera.schedule_digest = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
