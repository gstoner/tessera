
// Sketch of a Tessera BFN dialect (not compiled as-is).
def BFN_Dialect : Dialect {
  let name = "bfn";
  let summary = "Bayesian Flow Networks ops";
}

def BFN_ReceiverOp : Op<"receiver", [Pure]> {
  let summary = "Convolve output distribution with schedule noise";
  let arguments = (ins AnyType:$outParams, F32:$beta);
  let results   = (outs AnyType:$recvParams);
}

def BFN_SenderOp : Op<"sender", [Pure]> {
  let summary = "Noisy sender distribution from data";
  let arguments = (ins AnyType:$x, F32:$beta);
  let results   = (outs AnyType:$sendParams);
}

def BFN_KLOp : Op<"kl", [Pure]> {
  let summary = "KL(receiver || sender)";
  let arguments = (ins AnyType:$recvParams, AnyType:$sendParams);
  let results   = (outs F32:$loss);
}

def BFN_BayesUpdateOp : Op<"bayes_update", []> {
  let summary = "Closed-form update of factorized prior";
  let arguments = (ins AnyType:$prior, AnyType:$sample);
  let results   = (outs AnyType:$posterior);
}
