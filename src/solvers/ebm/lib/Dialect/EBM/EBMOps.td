//===- EBMOps.td --------------------------------------------*- tablegen -*-===//
// Tessera Energy-Based Model dialect (EBM5).
//
// Resurrected from the archived `tessera.ebt` design (see
// `archive/examples/advanced/EBT/Tessera_EBT_Package_v1/`) with the EBM0
// revisions: functional state, explicit RNGKey, broader namespace
// `tessera_ebm` (covers RBM, EBT, score-matching, manifold-aware EBM7).
//
// Five core ops mirror the EBM1 + EBM7 Python surface in `tessera.ebm`:
//
//   ebm.energy          ↔ tessera.ebm.energy
//   ebm.inner_step      ↔ tessera.ebm.inner_step
//   ebm.langevin_step   ↔ tessera.ebm.langevin_step / sphere_langevin_step / bivector_langevin_step
//   ebm.self_verify     ↔ tessera.ebm.self_verify
//   ebm.decode_init     ↔ tessera.ebm.decode_init
//
// Plus partition-function ops from EBM3:
//   ebm.partition_z (with `method = "exact" | "monte_carlo" | "annealed"`)
//
// See `docs/audit/ga_ebm_roadmap.md` § EBM5.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def EBM_Dialect : Dialect {
  let name = "tessera_ebm";
  let cppNamespace = "::tessera::ebm";
  let summary =
    "Tessera Energy-Based Models dialect — energy primitives, samplers, "
    "self-verify, partition function, and manifold-aware Langevin variants.";
  let description = [{
    Operates over functional (state, RNGKey) tuples per the EBM0 scope
    lock. The dialect distinguishes Euclidean, sphere, and bivector
    state via the `manifold` attribute on `langevin_step`. See
    `docs/spec/EBM_SPEC.md`.
  }];
}

//===----------------------------------------------------------------------===//
// Common attribute conventions
//
// Several `StrAttr` operands carry enumerated string values:
//   manifold      : "euclidean" | "sphere" | "bivector"
//   method        : "exact" | "monte_carlo" | "annealed"
//   init_strategy : "noise" | "base_model" | "copy"
//
// MLIR ODS doesn't support `def X : StrAttr` aliases (StrAttr is itself
// a def, not a class), so we use `StrAttr` directly in op operands.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Core EBM ops
//===----------------------------------------------------------------------===//

class EBMOp<string mnemonic, list<Trait> traits = []>
    : Op<EBM_Dialect, mnemonic, traits>;

def EBM_EnergyOp : EBMOp<"energy", [Pure]> {
  let summary = "Evaluate the user-provided energy function on (x, y).";
  let description = [{
    The energy is opaque from the dialect's perspective — `energy_fn`
    is referenced by symbol name. EBM6's energy-grad fusion pass
    inlines and fuses with `inner_step` / `langevin_step` callers.
  }];
  let arguments = (ins
    AnyType:$context_x,
    AnyType:$y,
    FlatSymbolRefAttr:$energy_fn
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $context_x `,` $y `energy_fn` $energy_fn
    attr-dict `:` functional-type(operands, results)
  }];
}

def EBM_InnerStepOp : EBMOp<"inner_step", [Pure]> {
  let summary = "One deterministic inner-loop update y' = y − η · grad.";
  let description = [{
    Pure gradient-descent step.  For stochastic / noise-augmented
    Langevin dynamics use `tessera_ebm.langevin_step` which threads an
    `RNGKey` operand.  This op intentionally drops the noise / RNG
    surface so the downstream code-gen passes can specialize fused
    energy-grad kernels without an RNG-state-handle parameter.
  }];
  let arguments = (ins
    AnyType:$y,
    AnyType:$grad,
    F64Attr:$eta
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $y `,` $grad `eta` $eta
    attr-dict `:` functional-type(operands, results)
  }];
}

def EBM_LangevinStepOp : EBMOp<"langevin_step", []> {
  let summary =
    "One Langevin step on a (possibly non-flat) manifold.";
  let description = [{
    `manifold` selects the integrator:
        "euclidean"  → tessera.ebm.langevin_step (EBM1)
        "sphere"     → tessera.ebm.sphere_langevin_step (EBM7)
        "bivector"   → tessera.ebm.bivector_langevin_step (EBM7)

    For "bivector", `grade` is the restricted grade (default 2 for SO(n)
    sampling).  For "sphere", the state must be unit-norm on entry.

    Returns the (new_state, next_key) tuple as two result values.
  }];
  let arguments = (ins
    AnyType:$state,
    AnyType:$rng_key,
    FlatSymbolRefAttr:$energy_fn,
    F64Attr:$eta,
    F64Attr:$temperature,
    StrAttr:$manifold,
    OptionalAttr<I64Attr>:$grade
  );
  let results = (outs AnyType:$new_state, AnyType:$next_key);
  let assemblyFormat = [{
    $state `,` $rng_key `energy_fn` $energy_fn `eta` $eta
    `temperature` $temperature `manifold` $manifold (`grade` $grade^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def EBM_SelfVerifyOp : EBMOp<"self_verify", [Pure]> {
  let summary = "Reduce K candidates to the lowest-energy one (or soft-min).";
  let arguments = (ins
    AnyType:$energies,
    AnyType:$candidates,
    OptionalAttr<F64Attr>:$beta
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $energies `,` $candidates (`beta` $beta^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def EBM_DecodeInitOp : EBMOp<"decode_init", []> {
  let summary = "Initialize K candidate trajectories.";
  let arguments = (ins
    AnyType:$x,
    AnyType:$rng_key,
    I64Attr:$K,
    StrAttr:$init_strategy,
    OptionalAttr<FlatSymbolRefAttr>:$base_model_fn,
    OptionalAttr<I64ArrayAttr>:$shape
  );
  let results = (outs AnyType:$candidates, AnyType:$next_key);
  let assemblyFormat = [{
    $x `,` $rng_key `K` $K `strategy` $init_strategy
    (`base_model` $base_model_fn^)? (`shape` $shape^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def EBM_PartitionZOp : EBMOp<"partition_z", []> {
  let summary = "Estimate the partition function Z by the chosen method.";
  let description = [{
    `method = "exact"` enumerates a finite support;
    `"monte_carlo"` runs importance sampling;
    `"annealed"` runs AIS with a tempering schedule. Each method
    consumes a different set of optional inputs handled by the
    EBM-pipeline lowering passes.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$energy_fn,
    StrAttr:$method,
    OptionalAttr<I64Attr>:$n_samples,
    OptionalAttr<I64Attr>:$n_chains,
    OptionalAttr<I64Attr>:$n_steps
  );
  let results = (outs AnyType:$Z, AnyType:$diagnostics);
  let assemblyFormat = [{
    `energy_fn` $energy_fn `method` $method
    (`n_samples` $n_samples^)?
    (`n_chains` $n_chains^)?
    (`n_steps` $n_steps^)?
    attr-dict `:` functional-type(operands, results)
  }];
}
