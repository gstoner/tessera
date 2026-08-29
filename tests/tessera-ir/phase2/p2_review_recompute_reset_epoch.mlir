// P2 review follow-up (2026-08-29, PR #640 review) — a value released twice.
//
// A checkpoint resets liveBytes to zero, which forgives every value defined
// before it. The last-use table, however, still holds those values' deaths at
// their later ordinals. Subtracting them there debits bytes belonging to
// POST-checkpoint values instead, so the counter sinks toward zero and the
// pass under-checkpoints a program whose live set really does exceed the
// budget — defeating the memory bound the pass exists to enforce.
//
// Below, %p is defined before the first checkpoint and dies after it. Measured
// on this input at this budget: 3 checkpoints while the double release was
// present, 5 once each value is released only within its own epoch. (At a 2 MB
// budget both spellings answer 2, which is why this fixture pins 1 MB — the
// defect is only observable where the forgiven bytes change a decision.)
//
// RUN: tessera-opt --tessera-insert-recompute="memory-budget-mb=1" \
// RUN:   --allow-unregistered-dialect %s | FileCheck %s

// CHECK: tessera_sr.num_checkpoints = 5
// CHECK: @value_crossing_a_checkpoint_is_released_once
func.func @value_crossing_a_checkpoint_is_released_once(%a: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  %p = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %q = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %r = "tessera.add"(%p, %q)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %n1 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %use_p = "tessera.add"(%p, %n1)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %n2 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %n3 = "tessera.gelu"(%a) : (tensor<512x512xf32>) -> tensor<512x512xf32>
  %s1 = "tessera.add"(%use_p, %n2)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s2 = "tessera.add"(%s1, %n3)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %s3 = "tessera.add"(%s2, %r)
      : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  return %s3 : tensor<512x512xf32>
}
