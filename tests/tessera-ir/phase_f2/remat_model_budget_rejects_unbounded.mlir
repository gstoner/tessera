// RUN: tessera-opt --tessera-activation-rematerialization --verify-diagnostics --split-input-file %s
//
// A dynamic model parameter has no statically knowable storage footprint.
// Model-derived budgeting must reject it unless the frontend supplies an
// explicit conservative byte bound; guessing here can overcommit the device.

module {
  // expected-error @+1 {{REMAT_MODEL_BUDGET_INVALID: model parameter argument 0 has a dynamic or unsupported type and requires a non-negative tessera.model.parameter_bytes_bound}}
  func.func @unbounded_dynamic_parameter(
      %weight: tensor<?x1024xf32> {tessera.model.parameter},
      %x: tensor<1024xf32>) -> tensor<1024xf32>
      attributes {
        tessera.device_memory_capacity_bytes = 8589934592 : i64
      } {
    return %x : tensor<1024xf32>
  }
}

// -----

module {
  // expected-error @+1 {{REMAT_MODEL_BUDGET_INVALID: model-derived memory-budget inputs must be non-negative and reserve basis points must be <= 10000}}
  func.func @invalid_reserve()
      attributes {
        tessera.device_memory_capacity_bytes = 8589934592 : i64,
        tessera.device_memory_reserve_basis_points = 10001 : i32
      } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{REMAT_MODEL_BUDGET_INVALID: model-derived memory-budget arithmetic overflows signed i64}}
  func.func @state_copy_overflow()
      attributes {
        tessera.device_memory_capacity_bytes = 8589934592 : i64,
        tessera.model_gradient_copies = 9223372036854775807 : i64
      } {
    return
  }
}
