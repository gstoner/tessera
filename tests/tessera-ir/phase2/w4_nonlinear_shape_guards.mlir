// RUN: tessera-opt --tessera-symdim-equality --verify-diagnostics %s

module {
  func.func @valid() attributes {
    tessera.dim_sizes = {B = 2 : i64, M = 8 : i64, N = 16 : i64},
    tessera.nonlinear_shape_guards = {
      version = 1 : i64,
      symbols = ["B", "M", "N"],
      constraints = [{relation = "eq", constant = 0 : i64, terms = [
        {coefficient = 1 : i64, powers = array<i64: 1, 1, 0>},
        {coefficient = -1 : i64, powers = array<i64: 0, 0, 1>}
      ]}]
    }
  } {
    return
  }

  // expected-error@+1 {{SYMDIM_NONLINEAR_GUARD_VIOLATION: polynomial eq row evaluated to -2}}
  func.func @invalid() attributes {
    tessera.dim_sizes = {B = 2 : i64, M = 7 : i64, N = 16 : i64},
    tessera.nonlinear_shape_guards = {
      version = 1 : i64,
      symbols = ["B", "M", "N"],
      constraints = [{relation = "eq", constant = 0 : i64, terms = [
        {coefficient = 1 : i64, powers = array<i64: 1, 1, 0>},
        {coefficient = -1 : i64, powers = array<i64: 0, 0, 1>}
      ]}]
    }
  } {
    return
  }

  // expected-error@+1 {{SYMDIM_NONLINEAR_GUARD_INCOMPLETE: symbol 'M' has no concrete tessera.dim_sizes witness}}
  func.func @incomplete() attributes {
    tessera.dim_sizes = {B = 2 : i64, N = 16 : i64},
    tessera.nonlinear_shape_guards = {
      version = 1 : i64,
      symbols = ["B", "M", "N"],
      constraints = [{relation = "eq", constant = 0 : i64, terms = [
        {coefficient = 1 : i64, powers = array<i64: 1, 1, 0>},
        {coefficient = -1 : i64, powers = array<i64: 0, 0, 1>}
      ]}]
    }
  } {
    return
  }
}
