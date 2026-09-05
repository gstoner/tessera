// RUN: tessera-opt --tessera-symdim-equality %s -split-input-file -verify-diagnostics >/dev/null
func.func @frontend(%a: tensor<?x?xf32> {tessera.dim_names = ["M", "K"]},
                    %b: tensor<?x?xf32> {tessera.dim_names = ["K", "N"]}) -> tensor<?x?xf32> {
  %r = tessera.matmul %a, %b {tessera.dim_names_lhs = ["M", "K"], tessera.dim_names_rhs = ["K", "N"]} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
// -----
func.func @frontend_bad(%a: tensor<?x?xf32> {tessera.dim_names = ["M", "Q"]},
                        %b: tensor<?x?xf32> {tessera.dim_names = ["K", "N"]}) -> tensor<?x?xf32> {
  // expected-error @+1 {{SYMDIM_FLOW_INCONSISTENCY}}
  %r = tessera.matmul %a, %b {tessera.dim_names_lhs = ["M", "K"], tessera.dim_names_rhs = ["K", "N"]} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
