// RUN: %tnv --split-input-file --verify-diagnostics %s
//
// `--verify-diagnostics` IS the check: it requires every `expected-error`
// to be produced AND fails on any diagnostic that was not expected, which
// is stricter than grepping stderr. It exits 0 on success, so this RUN line
// must not be wrapped in `not`.
//
// The delegation contract must REJECT, or it is decoration.
//
// `tessera_nvidia.kernel_call` used to be a summary line with no arguments:
// `callee` rode in the inherited attr-dict as a discardable attribute, so an
// emitter could name nothing at all and still verify. Each case below is a
// delegate that would have passed then and is unusable by Decision #28's
// arbiter now -- either it cannot be identified, or its numerical claim is
// missing or self-contradictory.

func.func @empty_callee(%a: f32) -> f32 {
  // expected-error @+1 {{requires a non-empty `callee`}}
  %r = tessera_nvidia.kernel_call %a
      {callee = "", arch = "sm_120", binding = "c_abi",
       provenance = "vendor_library", accuracy = "reference_exact", determinism = "deterministic"}
      : (f32) -> f32
  return %r : f32
}

// -----

func.func @bounded_without_a_bound(%a: f32) -> f32 {
  // A tolerance-bounded claim with no bound is the failure this contract
  // exists to catch: it reads as a numerical guarantee and constrains nothing.
  // expected-error @+1 {{requires `tolerance` and/or}}
  %r = tessera_nvidia.kernel_call %a
      {callee = "cublasLtMatmul", arch = "sm_120", binding = "c_abi",
       provenance = "vendor_library", accuracy = "tolerance_bounded", determinism = "deterministic"}
      : (f32) -> f32
  return %r : f32
}

// -----

func.func @exact_claim_carrying_a_tolerance(%a: f32) -> f32 {
  // Two contradictory claims; a reader cannot tell which one is honoured.
  // expected-error @+1 {{must not carry a tolerance}}
  %r = tessera_nvidia.kernel_call %a
      {callee = "tessera_nvidia_flash", arch = "sm_120",
       binding = "cuda_kernel", provenance = "handwritten_kernel",
       accuracy = "reference_exact", determinism = "deterministic", tolerance = 1.000000e-03 : f64}
      : (f32) -> f32
  return %r : f32
}

// -----

func.func @non_positive_tolerance(%a: f32) -> f32 {
  // expected-error @+1 {{must be finite and greater than zero}}
  %r = tessera_nvidia.kernel_call %a
      {callee = "cublasLtMatmul", arch = "sm_120", binding = "c_abi",
       provenance = "vendor_library", accuracy = "tolerance_bounded", determinism = "deterministic",
       tolerance = 0.000000e+00 : f64}
      : (f32) -> f32
  return %r : f32
}

// -----

func.func @unknown_binding(%a: f32) -> f32 {
  // The legal set is stated by the attribute, not left to a free-form string
  // (Decision #21a: no unvalidated StrAttr where an enum states the set).
  // The diagnostic is attached to the attribute dictionary, not to the op's
  // first line, so it is anchored where the parser actually reports it.
  %r = tessera_nvidia.kernel_call %a
      // expected-error @+1 {{attribute 'binding' failed to satisfy constraint}}
      {callee = "tessera_nvidia_flash", arch = "sm_120",
       binding = "carrier_pigeon", provenance = "handwritten_kernel",
       accuracy = "reference_exact", determinism = "deterministic"}
      : (f32) -> f32
  return %r : f32
}

// -----

func.func @inline_ptx_without_constraints(%a: f32) -> f32 {
  // Unstated operand constraints are how register clobbers become silent
  // miscompiles, so an empty constraint string fails closed.
  // expected-error @+1 {{requires a non-empty `constraints` string}}
  %r = tessera_nvidia.inline_ptx %a
      {ptx = "mul.f32 $0, $1, $1;", constraints = "", arch = "sm_120",
       accuracy = "reference_exact", determinism = "deterministic"}
      : (f32) -> f32
  return %r : f32
}

// -----

func.func @empty_inline_ptx(%a: f32) -> f32 {
  // An empty asm body succeeds silently and computes nothing.
  // expected-error @+1 {{requires non-empty `ptx`}}
  %r = tessera_nvidia.inline_ptx %a
      {ptx = "", constraints = "=f,f", arch = "sm_120",
       accuracy = "reference_exact", determinism = "deterministic"}
      : (f32) -> f32
  return %r : f32
}

// -----

func.func @unknown_provenance(%a: f32) -> f32 {
  // Tiering is what lets the arbiter tell delegated from compiler-generated
  // work; an unrecognised provenance would leave a candidate untierable.
  %r = tessera_nvidia.kernel_call %a
      // expected-error @+1 {{attribute 'provenance' failed to satisfy constraint}}
      {callee = "tessera_nvidia_flash", arch = "sm_120",
       binding = "cuda_kernel", provenance = "rumour",
       accuracy = "reference_exact", determinism = "deterministic"}
      : (f32) -> f32
  return %r : f32
}

// -----

func.func @empty_arch(%a: f32) -> f32 {
  // expected-error @+1 {{requires a non-empty `arch`}}
  %r = tessera_nvidia.kernel_call %a
      {callee = "tessera_nvidia_flash", arch = "",
       binding = "cuda_kernel", provenance = "handwritten_kernel",
       accuracy = "reference_exact", determinism = "deterministic"}
      : (f32) -> f32
  return %r : f32
}

