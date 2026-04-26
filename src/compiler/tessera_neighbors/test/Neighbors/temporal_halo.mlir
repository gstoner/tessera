\
// RUN: tessera-opt %s -tessera-halo-infer | FileCheck %s
// CHECK: time = 1
%h = "tessera.neighbors.halo.region"(%arg0) {time = 1} : (tensor<?x?xf32>) -> !tessera.neighbors.halo
