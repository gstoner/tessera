// RUN: not tessera-opt %s -tessera-apply-softmax-policy 2>&1 | FileCheck %s
tessera.numerics.softmax "poly3", -1.0
// CHECK: rescale_threshold must be > 0
