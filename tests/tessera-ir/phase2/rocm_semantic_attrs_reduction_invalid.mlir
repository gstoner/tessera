// RUN: not tessera-opt %s 2>&1 | FileCheck %s
//
// W1.1b — `reduction` selects WHAT THE KERNEL COMPUTES, so an unknown policy
// must be an error rather than silently becoming the default. "median" is a
// real reduction elsewhere in the tree, which is exactly why a bare StrAttr
// here was dangerous: it is a plausible value that this kernel cannot perform.

func.func @bad_reduction() {
  // CHECK: attribute 'reduction' failed to satisfy constraint: ROCm reduction policy
  "tessera_rocm.binary_loss"() {
    name = "bce", dtype = "f32", reduction = "median"
  } : () -> ()
  return
}
