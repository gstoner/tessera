\
// RUN: tessera-opt %s -tessera-topology-dynamic | FileCheck %s
// CHECK: replan
// NOTE: this is a placeholder to ensure pass runs and inserts replan hooks.
module {
  // imagine a topology mutation here
}
