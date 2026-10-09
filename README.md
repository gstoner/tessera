# Tessera

**Pre-alpha. Breaking changes expected. Not production-ready.**

Tessera is a standalone, tile-oriented programming model and compiler for deep
learning and scientific computing. It makes layout, memory ownership, numerical
policy and parallel execution explicit compiler contracts.

The architectural foundation is **MLIR with LLVM native lowering for CPU,
NVIDIA and ROCm, and compiler-owned Metal code generation for Apple GPU**.
Python provides the user interface, tracing, orchestration and
reference implementations. Migration is active: some package families already
consume verified native IR; others still use compatibility IR or source emitters.
A Python result or generated kernel string alone is not native execution proof.

## Quick start

From a source checkout with Python 3.10 or newer:

```bash
python -m pip install -e ".[dev]"
python examples/getting_started/compile_and_explain.py
```

The compiler tour runs without an accelerator and shows the execution report,
IR inspection and support queries. A native backend additionally needs its
compiler, runtime and supported hardware; installing the Python package does
not build those components.

Save this example in a Python file so the frontend can inspect its source:

```python
import numpy as np
import tessera as ts

@ts.jit(target="cpu")
def add_relu(x, y):
    return ts.ops.relu(ts.ops.add(x, y))

x = np.array([-2.0, 1.0], dtype=np.float32)
y = np.array([1.0, 3.0], dtype=np.float32)
print(add_relu(x, y))       # [0. 4.]
print(add_relu.explain())  # Inspect what actually ran: native or reference.
```

See the [frontend guide](docs/guides/Tessera_Developer_Frontend_End_To_End.md)
and [canonical API](docs/CANONICAL_API.md) for annotations, textual input,
constraints and execution policy.

## Architecture

```text
Python / textual frontend
  → typed semantic Graph IR (Tessera MLIR dialect)
  → structured differentiation and optimization (verified MLIR passes)
  → Schedule IR → Tile IR (MLIR dialects and passes)
  → backend Target IR → native lowering (MLIR target dialects, LLVM where supported)
  → native image + checked runtime ABI → execution
```

Tracing builds the semantic program; it is not itself the optimizing compiler.
The final architecture gives MLIR ownership of semantics, differentiation,
optimization, scheduling and target lowering. LLVM supplies CPU code generation
and the NVIDIA NVVM/PTX and ROCm ROCDL/AMDGPU native paths. Apple GPU ends in a
compiler-owned MSL → Metal/metallib path; no general LLVM-to-Metal backend is
assumed. Verified types, effects, layouts, residuals and package identity must
survive these boundaries. Backend-specific Target IR is inspectable without a
GPU but may name hardware instructions. Physical schedules belong to each
architecture.

Python remains the frontend, orchestration layer and numerical oracle. New
compiler enhancements must enter the canonical typed IR and verified native
passes, not add another Python semantic backend or a Graph-derived fast path.
Existing source emitters and delegated libraries remain explicitly bounded
migration candidates until equivalent compiled routes have been proved.

The [integrated compiler plan](docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md)
owns sequencing. The [compiler audit map](docs/audit/compiler/README.md) separates
active plans from references and archived designs. Source emitters remain bounded
compatibility or candidate paths while their replacements are proved; they are
not the destination for new shared lowering infrastructure.

## Development status

Static rank-2/3 ROCm `sqrt`, `exp`, `add`, `div`, `cumsum` and `cummax`
have ordinary `@jit` and portable native package proof on gfx1151 and gfx1201.
Explicit Graph casts admit f16/bf16 input storage with f32 computation/output;
native MLIR folds widening into loads. This is a bounded route, with
[owning-device tests and separate kernel/host timings](benchmarks/baselines/rocm_native_math_20261006/README.md);
general composition, dynamic layouts and AD remain active work.


Support is specific to the operation, shape, dtype, layout, target and execution
route. Consult the linked evidence before treating a whole backend or domain as
supported.

| Backend | Current boundary | Evidence and next work |
|---|---|---|
| x86 CPU | MLIR/LLVM CPU execution and architecture-specific native packages; ISA admission matters. | [x86 queue](docs/audit/backend/x86/todo.md) |
| Apple CPU/GPU | Accelerate/BNNS and Metal/MPS/MSL paths; native coverage varies by family and toolchain. | [Apple queue](docs/audit/backend/apple/todo.md) |
| NVIDIA CUDA | Native scheduled packages and specialized candidates, with RTX 5070 / `sm_120` evidence for admitted families. | [NVIDIA queue](docs/audit/backend/nvidia/todo.md) |
| AMD ROCm | MLIR→ROCDL/LLVM native packages with separate `gfx1151` and RX 9070 XT / `gfx1201` evidence for admitted families; neither architecture proves the other's coverage. | [ROCm queue](docs/audit/backend/rocm/todo.md) and [gfx1201 W8A8 packet](benchmarks/baselines/gfx1201_fp8_blockscale_ragged_20260927/README.md) |

Autodiff includes Python reference rules, compiler-generated forward/reverse
products and bounded native packages. Persistent static f32 tensor tapes now
have CUDA/HIP evidence; isolated Q/K attention JVP programs have CUDA evidence.
General persistent tapes, composed attention AD, native jets and broader backend
promotion remain active work. See the [autodiff plan](docs/audit/compiler/AUTODIFF_EXECUTION_PLAN.md)
and [implementation packets](benchmarks/NATIVE_STORAGE_FOLLOWUP.md).

Distributed APIs and placement/collective IR do not imply completed multi-rank
execution or measured overlap. Reference, mock, compiler-only and exact-device
results must remain distinguishable.

## Mathematical IR surfaces

Tessera includes geometric algebra, energy-based models, matrix/field calculus,
spectral/PDE operations and structured attention. Their maturity differs at
three boundaries: mathematical reference, compiler transformation and native
execution. A specialized GA/EBM update kernel does not establish a device-resident
energy-gradient loop.

The [domain audit](docs/audit/domain/DOMAIN_AUDIT.md) owns that distinction.
The [GA/EBM review](docs/audit/domain/GA_EBM_ARCHITECTURE_REVIEW.md) prioritizes
batched native algebra, traceable energy functions, shared AD/ownership and
measured backend consumers.

## Audit-as-data

Generated dashboards are reproducible projections of registries and evidence,
not a substitute for checking the source and proof scope when a claim conflicts.

| Question | Source |
|---|---|
| What executes on each target? | [Execution matrix](docs/audit/generated/runtime_execution_matrix.md) |
| Which AD stages are connected and proved? | [Autodiff ledger](docs/audit/generated/autodiff_connection_ledger.md) |
| What is implemented across compiler layers? | [Compiler progress](docs/audit/generated/compiler_progress.md) |
| Which domain operations have target evidence? | [Op/target conformance](docs/audit/op_target_conformance.md) |
| What is the runtime ABI? | [Runtime ABI](docs/audit/generated/runtime_abi.md) |

Do not hand-edit derived files. Test-coverage dashboards are generated on demand
or in CI, rather than committed as a merge hotspot:

```bash
python -m tessera.compiler.generated_docs --write
python -m tessera.compiler.generated_docs --check
```

## Build & test

Use a matched LLVM/MLIR toolchain and the relevant [backend guide](docs/backends/README.md).
The setup/build scripts contain the supported configuration switches:

```bash
bash scripts/build.sh
pytest tests/unit -m "not slow" -q
bash scripts/mypy_ratchet.sh
bash scripts/check_generated_docs.sh
```

A broad test command is a validation instruction, not a claim that the current
checkout passed it. Follow [AGENTS.md](AGENTS.md) for this repository's host/WSL
execution requirements. Device validation runs on the owning hardware and must
report native placement, numerical agreement and timing provenance separately.

## Documentation and project layout

- [Documentation index](docs/README.md) — specifications, guides and authority.
- [Compiler reference](docs/spec/COMPILER_REFERENCE.md) — IR and pass contracts.
- [Autodiff specification](docs/spec/AUTODIFF_SPEC.md) — public differentiation semantics.
- [Target IR review](docs/audit/compiler/TARGET_IR_REVIEW.md) — current ownership and gaps.
- [Project structure](PROJECT_STRUCTURE.md) — full source map.
- [Contributing](CONTRIBUTING.md) — development workflow.

`python/tessera/` contains the frontend, reference math and runtime orchestration.
`src/compiler/` and `src/transforms/` contain native dialects and lowering passes;
`src/runtime/` owns runtime implementations. `tests/unit/` contains Python and
contract tests, and `tests/tessera-ir/` contains native MLIR fixtures.
`benchmarks/` holds measurement tools and evidence packets.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

### Resident attention forward AD on NVIDIA SM120

The named static compact FP32 native_jvp attention route accepts resident
CUDA Q/K/V and tangent roots through typed Graph IR, native MLIR AD,
Schedule/Tile, Target and LLVM/PTX packages. Native stream ordering and private
snapshots are numerically proved on RTX5070. Structural frontend certification
performs zero concrete tensor executions; numerical authority is the native
package. This is not full resident attention AD or a general performance claim.
See [the exact-device packet](benchmarks/baselines/sm120_resident_attention_jvp_20261009/README.md).
