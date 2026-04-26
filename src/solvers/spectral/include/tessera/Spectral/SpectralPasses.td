
//===- SpectralPasses.td ---------------------------------------*- tablegen -*-===//
// Pass declarations for Tessera Spectral.
// This is a stub header to enable pass registration via TableGen.
//===----------------------------------------------------------------------===//
include "mlir/Pass/PassBase.td"

def LegalizeSpectral : Pass<"tessera-legalize-spectral", "mlir::ModuleOp"> {
  let summary = "Lower tessera_spectral.* to Tile/Schedule IR staging kernels";
}
def SpectralMXP : Pass<"tessera-spectral-mxp", "mlir::ModuleOp"> {
  let summary = "Insert mixed-precision scaling, exponent metadata, and error guards";
}
def SpectralTransposePlan : Pass<"tessera-spectral-transpose-plan", "mlir::ModuleOp"> {
  let summary = "Choose tile sizes, vector widths, and shared-memory padding for transposes";
}
def SpectralAutotune : Pass<"tessera-spectral-autotune", "mlir::ModuleOp"> {
  let summary = "Autotune radix sequences and tile choices; persist best plan";
}
def LowerSpectralToTargetIR : Pass<"lower-spectral-to-target-ir", "mlir::ModuleOp"> {
  let summary = "Lower spectral stages to Target-IR (NVIDIA/AMD/CPU) kernels";
}
def SpectralDistributed : Pass<"tessera-spectral-distributed", "mlir::ModuleOp"> {
  let summary = "Decompose multi-dim FFTs into local-FFT + all-to-all with overlap tokens";
}
