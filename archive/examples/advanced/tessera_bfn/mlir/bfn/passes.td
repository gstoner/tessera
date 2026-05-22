
// Skeleton for BFN cleanup/fusion passes.
def BFNCleanup : Pass<"bfn-cleanup", "func::FuncOp"> {
  let summary = "Fold constants, normalize simplex, outline step regions";
}
def BFNFuseReceiverKL : Pass<"bfn-fuse-recv-kl", "func::FuncOp"> {
  let summary = "Fuse receiver+kl into a single tiled kernel";
}
