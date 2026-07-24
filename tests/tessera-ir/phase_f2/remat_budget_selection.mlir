// RUN: tessera-opt --tessera-activation-rematerialization %s | FileCheck %s
//
// A function-level budget is the production-pipeline contract: the same pass
// that follows AutodiffPass selects long-lived pure activations, then sinks
// their recomputation.  No frontend tessera.recompute marker is required.

module {
  // CHECK: func.func @budget_selected
  // CHECK-SAME: tessera.remat_auto_selected = 2
  // CHECK-SAME: tessera.remat_budget_bytes = 1048576
  // CHECK-SAME: tessera.remat_budget_mb = 1
  // CHECK-SAME: tessera.remat_budget_source = "explicit_function"
  // CHECK-SAME: tessera.remat_peak_after_bytes = 0
  // CHECK-SAME: tessera.remat_peak_before_bytes = 8388608
  // CHECK-SAME: tessera.remat_selected_cost_ns = 2097152
  // CHECK-SAME: tessera.rematerialized = 2
  func.func @budget_selected(%x: tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32>
      attributes {tessera.remat_budget_mb = 1 : i32} {
    %square = arith.mulf %x, %x : tensor<1024x1024xf32>
    %neg = arith.negf %x : tensor<1024x1024xf32>
    // CHECK: arith.negf
    // CHECK-NEXT: arith.mulf
    // CHECK-NEXT: arith.addf
    %sum = arith.addf %square, %neg : tensor<1024x1024xf32>
    return %sum : tensor<1024x1024xf32>
  }

  // Equal-size/lifetime activations use measured recompute latency as the
  // deciding signal. Only the cheap producer is needed to meet this budget.
  // CHECK-LABEL: func.func @measured_cost_prefers_cheap
  // CHECK-SAME: tessera.remat_auto_selected = 1
  // CHECK-SAME: tessera.remat_budget_bytes = 1048576
  // CHECK-SAME: tessera.remat_peak_after_bytes = 1048576
  // CHECK-SAME: tessera.remat_peak_before_bytes = 2097152
  // CHECK-SAME: tessera.remat_selected_cost_ns = 25
  func.func @measured_cost_prefers_cheap(
      %x: tensor<512x512xf32>) -> tensor<512x512xf32>
      attributes {tessera.remat_budget_mb = 1 : i32} {
    %expensive = arith.mulf %x, %x
        {tessera.remat_cost_ns = 900 : i64} : tensor<512x512xf32>
    %cheap = arith.negf %x
        {tessera.remat_cost_ns = 25 : i64} : tensor<512x512xf32>
    // CHECK: arith.negf
    // CHECK-NEXT: arith.addf
    %sum = arith.addf %expensive, %cheap : tensor<512x512xf32>
    return %sum : tensor<512x512xf32>
  }

  // CHECK-NOT: tessera.recompute

  // A production training graph may omit a hand-authored activation budget.
  // The pass derives one from the device envelope and explicitly marked model
  // state:
  //   usable = 20 MiB * 90% = 18 MiB
  //   state  = 4 MiB * (parameter + gradient + 2 optimizer copies) + 1 MiB
  //          = 17 MiB
  //   activation budget = 1 MiB
  // CHECK-LABEL: func.func @model_derived_budget
  // CHECK-SAME: tessera.model_parameter_bytes = 4194304
  // CHECK-SAME: tessera.model_state_bytes = 17825792
  // CHECK-SAME: tessera.remat_auto_selected = 2
  // CHECK-SAME: tessera.remat_budget_bytes = 1048576
  // CHECK-SAME: tessera.remat_budget_source = "model_device_envelope"
  // CHECK-SAME: tessera.remat_peak_after_bytes = 0
  // CHECK-SAME: tessera.remat_peak_before_bytes = 8388608
  func.func @model_derived_budget(
      %weight: tensor<1024x1024xf32> {tessera.model.parameter},
      %x: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
      attributes {
        tessera.device_memory_capacity_bytes = 20971520 : i64,
        tessera.device_memory_reserve_basis_points = 1000 : i32,
        tessera.model_gradient_copies = 1 : i32,
        tessera.model_optimizer_state_copies = 2 : i32,
        tessera.model_persistent_bytes = 1048576 : i64
      } {
    %square = arith.mulf %x, %x : tensor<1024x1024xf32>
    %neg = arith.negf %x : tensor<1024x1024xf32>
    %sum = arith.addf %square, %neg : tensor<1024x1024xf32>
    return %sum : tensor<1024x1024xf32>
  }

  // Dynamic parameter storage participates only through an explicit,
  // conservative ABI bound. With no gradient/optimizer copies and no reserve,
  // 10 MiB capacity - 2 MiB parameter storage leaves 8 MiB for activations.
  // CHECK-LABEL: func.func @bounded_dynamic_parameter
  // CHECK-SAME: tessera.model_parameter_bytes = 2097152
  // CHECK-SAME: tessera.model_state_bytes = 2097152
  // CHECK-SAME: tessera.remat_budget_bytes = 8388608
  // CHECK-SAME: tessera.remat_budget_source = "model_device_envelope"
  func.func @bounded_dynamic_parameter(
      %weight: tensor<?xf32> {
        tessera.model.parameter,
        tessera.model.parameter_bytes_bound = 2097152 : i64
      },
      %x: tensor<16xf32>) -> tensor<16xf32>
      attributes {
        tessera.device_memory_capacity_bytes = 10485760 : i64,
        tessera.device_memory_reserve_basis_points = 0 : i32,
        tessera.model_gradient_copies = 0 : i32,
        tessera.model_optimizer_state_copies = 0 : i32
      } {
    return %x : tensor<16xf32>
  }

  // A zero-byte derived activation budget is a real budget, not "disabled".
  // CHECK-LABEL: func.func @zero_activation_budget
  // CHECK-SAME: tessera.remat_auto_selected = 1
  // CHECK-SAME: tessera.remat_budget_bytes = 0
  // CHECK-SAME: tessera.remat_peak_after_bytes = 0
  // CHECK-SAME: tessera.remat_peak_before_bytes = 4194304
  func.func @zero_activation_budget(
      %weight: tensor<1024x1024xf32> {tessera.model.parameter},
      %x: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
      attributes {
        tessera.device_memory_capacity_bytes = 16777216 : i64,
        tessera.device_memory_reserve_basis_points = 0 : i32,
        tessera.model_gradient_copies = 1 : i32,
        tessera.model_optimizer_state_copies = 2 : i32
      } {
    %neg = arith.negf %x : tensor<1024x1024xf32>
    %sum = arith.addf %neg, %x : tensor<1024x1024xf32>
    return %sum : tensor<1024x1024xf32>
  }

  // An explicit function budget is authoritative even if dormant model/device
  // inputs would be invalid. Production launchers can attach a model envelope
  // without overriding a deliberate per-function tuning decision.
  // CHECK-LABEL: func.func @explicit_budget_precedes_model_envelope
  // CHECK-SAME: tessera.remat_budget_bytes = 2097152
  // CHECK-SAME: tessera.remat_budget_mb = 2
  // CHECK-SAME: tessera.remat_budget_source = "explicit_function"
  func.func @explicit_budget_precedes_model_envelope(
      %x: tensor<16xf32>) -> tensor<16xf32>
      attributes {
        tessera.device_memory_capacity_bytes = -1 : i64,
        tessera.remat_budget_mb = 2 : i32
      } {
    return %x : tensor<16xf32>
  }
}
