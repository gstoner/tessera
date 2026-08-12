// RUN: tessera-opt --tessera-symdim-equality --split-input-file --verify-diagnostics %s

// M >= 1 and N == M has an integer solution under the witnesses.
func.func @presburger_ok() attributes {
  tessera.dim_sizes = {M = 8 : i64, N = 8 : i64},
  tessera.presburger_constraints = {
    version = 1 : i64,
    symbols = ["M", "N"],
    constraints = [
      {relation = "ge", coefficients = array<i64: 1, 0>, constant = -1 : i64},
      {relation = "eq", coefficients = array<i64: -1, 1>, constant = 0 : i64},
      {relation = "mod", coefficients = array<i64: 1, 0>, constant = 0 : i64,
       modulus = 4 : i64, remainder = 0 : i64}
    ]
  }
} {
  return
}

// -----

// expected-error @+1 {{SYMDIM_PRESBURGER_UNSATISFIABLE}}
func.func @presburger_unsatisfiable() attributes {
  tessera.dim_sizes = {M = 8 : i64, N = 7 : i64},
  tessera.presburger_constraints = {
    version = 1 : i64,
    symbols = ["M", "N"],
    constraints = [
      {relation = "ge", coefficients = array<i64: 1, 0>, constant = -1 : i64},
      {relation = "eq", coefficients = array<i64: -1, 1>, constant = 0 : i64}
    ]
  }
} {
  return
}

// -----

// expected-error @+1 {{SYMDIM_PRESBURGER_UNSATISFIABLE}}
func.func @presburger_divisibility_unsatisfiable() attributes {
  tessera.dim_sizes = {M = 6 : i64},
  tessera.presburger_constraints = {
    version = 1 : i64,
    symbols = ["M"],
    constraints = [
      {relation = "mod", coefficients = array<i64: 1>, constant = 0 : i64,
       modulus = 4 : i64, remainder = 0 : i64}
    ]
  }
} {
  return
}

// -----

// expected-error @+1 {{SYMDIM_PRESBURGER_MALFORMED}}
func.func @presburger_malformed() attributes {
  tessera.presburger_constraints = {
    version = 1 : i64,
    symbols = ["M", "N"],
    constraints = [
      {relation = "eq", coefficients = array<i64: 1>, constant = 0 : i64}
    ]
  }
} {
  return
}
