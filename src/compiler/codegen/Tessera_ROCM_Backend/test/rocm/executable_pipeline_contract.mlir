// RUN: %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=softmax input=tile output=target arch=gfx1151})' %s | FileCheck %s

// The family plugin is selected by typed pipeline configuration. The producer,
// Target IR consumer, and backend code generator are durable module attributes;
// no Python caller owns or repeats the pass sequence.
module {
}

// CHECK: module attributes {
// CHECK-DAG: tessera.pipeline.arch = "gfx1151"
// CHECK-DAG: tessera.pipeline.backend_codegen = "rocdl_hsaco"
// CHECK-DAG: tessera.pipeline.family = "softmax"
// CHECK-DAG: tessera.pipeline.output = "target"
// CHECK-DAG: tessera.pipeline.schema = "tessera.executable_pipeline.v1"
// CHECK-DAG: tessera.pipeline.target_ir_consumer = "tessera_rocm"
// CHECK-DAG: tessera.pipeline.tile_producer = "content_addressed_tile"
