# Tessera Apple Backend

This backend defines the hardware-free Target IR contracts for Apple Silicon.
It intentionally starts as an artifact path:

- `tessera_apple.cpu.*` models Accelerate/vecLib/BNNS CPU calls.
- `tessera_apple.gpu.*` models Metal/MPS-style GPU kernels and dispatch.

Native execution, `metallib` compilation, and runtime launch are follow-up
milestones. The first acceptance gate is verifier/FileCheck-level Target IR.
