// RUN: tessera-opt %s -tessera-lower-schedule | FileCheck %s
// CHECK: tessera.queue.create
// CHECK: tessera.queue.push
// CHECK: tessera.queue.pop
tessera.schedule @fa4_pipeline { }
