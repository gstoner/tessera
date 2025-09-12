\
// RUN: tessera-opt %s -tessera-pipeline-overlap | FileCheck %s
// CHECK: tessera.neighbors.pipeline.config
"tessera.neighbors.pipeline.config"() {stages = 3, overlap = "eager", double_buffer = true, reuse = "lines"} : () -> ()
