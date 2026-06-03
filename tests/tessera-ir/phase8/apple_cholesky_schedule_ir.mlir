// L-series linalg pilot — L2: tessera.cholesky through Graph→Schedule.
//
// DistributionLoweringPass is generic — it outlines the function body into a
// schedule.mesh.region and yields the escaping values.  This fixture proves
// cholesky traverses the Schedule layer identically to matmul (the spine
// needed no cholesky-specific code), and that the result is dominance-clean
// (the region yields %chol and the return consumes the region result).
//
// RUN: tessera-opt --tessera-distribution-lowering='mesh-axes=dp mesh-sizes=1' --allow-unregistered-dialect %s | FileCheck %s --check-prefix=OPT
// RUN: tessera-opt %s | FileCheck %s --check-prefix=NOOP

// OPT-LABEL: func.func @chol_step
// OPT:       schedule.mesh.define
// OPT-SAME:  "dp"
// OPT:       %[[R:.*]] = "schedule.mesh.region"
// OPT:         tessera.cholesky
// OPT:         schedule.yield
// OPT:       return %[[R]]
func.func @chol_step(
    %a: tensor<8x8xf32> {tessera.shard = {axes = ["dp"], dims = [0], sizes = [1]}}
) -> tensor<8x8xf32> {
  %0 = tessera.cholesky %a : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// NOOP-LABEL: func.func @chol_step
// NOOP-NOT:   schedule.mesh.define
