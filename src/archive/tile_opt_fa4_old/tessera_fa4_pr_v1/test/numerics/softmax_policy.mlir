// RUN: tessera-opt %s -tessera-apply-softmax-policy | FileCheck %s
tessera.numerics.softmax "poly3", 2.0e-3
// CHECK: poly3
