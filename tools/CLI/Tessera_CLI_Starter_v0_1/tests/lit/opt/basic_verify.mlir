\
// RUN: tessera-opt %s --to=schedule --verify --dump=tile --dump-dir %t | FileCheck %s
// CHECK: (tessera-opt) normalized IR
module {}
