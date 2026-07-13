---
status: Normative
classification: Normative
authority: Canonical public Python API naming and documentation routing
scope: Stable public names, namespaces, and terminology; not runtime status or backend capability
last_updated: 2026-07-13
---

# Tessera Canonical API

This is the short, stable entry point for Tessera's public Python API. It
answers two questions only:

1. What is the one canonical public name for a concept?
2. Which document owns that concept's detailed contract or current status?

If another document uses a different public spelling, this document wins on
the spelling. It does **not** override the detailed semantic contract in a
linked specification, nor current execution evidence in a generated audit.

## Authority and Boundaries

Tessera documentation has deliberately separate authorities. Keep those
boundaries intact when adding APIs or examples.

| Need | Authority | Do not put it here |
|---|---|---|
| Canonical public spelling and namespace | this document | exhaustive signatures or implementation inventories |
| Exact Python signatures, exceptions, and call semantics | [`docs/spec/PYTHON_API_SPEC.md`](spec/PYTHON_API_SPEC.md) | target support claims |
| Tensor attributes, dtype canonicalization, and numeric policy | [`docs/reference/tessera_tensor_attributes.md`](reference/tessera_tensor_attributes.md) | duplicate dtype tables |
| Standard operator semantics | [`docs/operations/Tessera_Standard_Operations.md`](operations/Tessera_Standard_Operations.md) | per-backend kernel claims |
| Graph, Schedule, Tile, and Target IR semantics | [`docs/spec/`](spec/) | Python API aliases |
| Target architecture and compiler/runtime model | [`docs/backends/`](backends/) | mutable implementation counts |
| What executes, on which target, and with what proof | [`docs/audit/generated/`](audit/generated/) | hand-maintained status tables |
| Backend decisions, gaps, and plans | [`docs/audit/backend/`](audit/backend/) | claims that a planned feature executes |
| Errors, diagnostics, and debugging workflow | [`docs/guides/`](guides/) | a second error catalog |

The status vocabulary is also intentional:

- **API available** means a public spelling and contract exist.
- **artifact emission** means the compiler can produce a target form.
- **reference execution** means a functional oracle exists.
- **native execution** means a backend runtime launches work.
- **hardware proof** means native execution has target-specific evidence.

These are different claims. Read the generated execution evidence before
choosing a target or describing backend readiness.

## Canonical Spelling Rules

Use these forms in public docs, examples, diagnostics, tests, and migration
guidance.

| Concept | Canonical spelling | Avoid |
|---|---|---|
| Package import | `import tessera` | `import tessera as ts` in normative examples; legacy package aliases |
| Graph/compiler decorator | `@tessera.jit` | `@tessera.function`, `@jit` |
| Tile-kernel decorator | `@tessera.kernel` | `@ts.kernel`, unqualified `@kernel` |
| Public operators | `tessera.ops.<name>` | backend-private op names in portable examples |
| Logical domain | `tessera.domain.Rect(...)` | a distribution object used as a shape |
| Distribution / placement | `tessera.dist.Block(...)`, `tessera.dist.Cyclic(...)`, `tessera.dist.Replicated()` | encoding placement in a domain |
| Distributed-array construction | `tessera.array.from_domain(...)` | direct construction from private storage |
| Mesh launch | `tessera.index_launch(axis=...)` | a backend stream as the portable launch abstraction |
| Structural constraint | `tessera.require(tessera.constraint.<Rule>(...))` | ad-hoc assertion as a compiler constraint |
| Tensor annotation | `tessera.Tensor[...]`, `tessera.Region[...]`, or documented dtype shortcuts | private annotation helpers |
| Python dtype aliases at input | accepted aliases normalize at the boundary | retaining aliases in stored metadata or IR |

The only canonical decorators are `@tessera.jit` and `@tessera.kernel`.
`@tessera.jit` denotes compilation of a Python function through Tessera's
frontend. `@tessera.kernel` denotes a tile-level kernel used with
`index_launch`; it is not a synonym for `jit`.

## Public Namespace Map

This map is intentionally a route map, not an inventory. The linked owner is
where a symbol's full signature, examples, and compatibility rules belong.

| Namespace | Public role | Detailed authority |
|---|---|---|
| `tessera` | root imports, decorators, constraints, tensor annotations, and factories | [`PYTHON_API_SPEC`](spec/PYTHON_API_SPEC.md) |
| `tessera.ops` | portable standard operators | [`Tessera Standard Operator Library`](operations/Tessera_Standard_Operations.md) |
| `tessera.nn` | functional and stateful neural-network building blocks | [`PYTHON_API_SPEC`](spec/PYTHON_API_SPEC.md) |
| `tessera.domain`, `tessera.dist`, `tessera.array` | logical shape, placement, and distributed-array construction | [`PYTHON_API_SPEC`](spec/PYTHON_API_SPEC.md) |
| `tessera.constraint`, `tessera.Region` | shape constraints and privilege annotations | [`PYTHON_API_SPEC`](spec/PYTHON_API_SPEC.md) and [`LANGUAGE_AND_IR_SPEC`](spec/LANGUAGE_AND_IR_SPEC.md) |
| `tessera.autodiff` | Python autodiff API | [`AUTODIFF_SPEC`](spec/AUTODIFF_SPEC.md) |
| `tessera.graph`, `tessera.debug` | inspection, tracing, and debugging | [`Tessera Debugging Tools Guide`](guides/Tessera_Debugging_Tools_Guide.md) |
| `tessera.autotune`, `tessera.profiler` | tuning and measurement interfaces | [`Tessera Profiling and Autotuning Guide`](guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `tessera.distributed` | distributed programming contracts | [`PYTHON_API_SPEC`](spec/PYTHON_API_SPEC.md) |
| `tessera.cache` | KV-cache public API | [`PYTHON_API_SPEC`](spec/PYTHON_API_SPEC.md) |
| `tessera.dflash`, `tessera.models`, `tessera.diffusion_guidance` | workload-specific public surfaces | [`docs/architecture/workloads/`](architecture/workloads/) plus their API-spec entries |

Experimental namespaces must say so in their own reference. Their presence in
the root package does not imply a stable compatibility guarantee or native
support on every target.

## Minimal Canonical Examples

### Compiled function

```python
import tessera

@tessera.jit
def matmul(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.matmul(A, B)
```

`target=` selects a compiler target profile or named target. It is a request
to compile for that target; it is not a claim that every operation will execute
natively. Consult the target's backend page and the generated execution matrix
for the supported execution path and proof.

### `target=` string aliases

String aliases normalize case-insensitively with hyphens converted to
underscores. A canonical target name in the middle column is also accepted
directly. This table owns the public spelling contract; it does not describe
the execution status of any target.

| Input alias | Canonical target |
|---|---|
| `None`, `cpu`, `x86_64` | `cpu` |
| `x86` | `x86` |
| `sm80`, `sm_80` | `nvidia_sm80` |
| `cuda`, `nvidia`, `gpu`, `sm90`, `sm_90`, `sm90a`, `sm_90a`, `hopper` | `nvidia_sm90` |
| `sm100`, `sm_100`, `sm100a`, `sm_100a`, `blackwell` | `nvidia_sm100` |
| `sm120`, `sm_120`, `blackwell_consumer`, `rtx50`, `gb20x` | `nvidia_sm120` |
| `rocm`, `amd`, `hip` | `rocm` |
| `gfx90a`, `mi250`, `mi250x` | `rocm_gfx90a` |
| `gfx940`, `mi300a` | `rocm_gfx940` |
| `gfx942`, `mi300x`, `mi325x` | `rocm_gfx942` |
| `gfx950`, `mi350p`, `mi350x`, `mi355x` | `rocm_gfx950` |
| `gfx1100`, `rdna3`, `rx7900` | `rocm_gfx1100` |
| `gfx1151`, `radeon8060s`, `rdna35`, `ryzenaimax395`, `strixhalo` | `rocm_gfx1151` |
| `gfx1200`, `gfx12`, `rdna4`, `rx9000` | `rocm_gfx1200` |
| `gfx1201`, `r9700`, `radeon_ai_pro_r9700` | `rocm_gfx1201` |
| `gfx1250`, `mi455x` | `rocm_gfx1250` |
| `apple_cpu`, `macos_cpu`, `m_series_cpu` | `apple_cpu` |
| `apple_gpu`, `apple`, `mac`, `macos_gpu`, `m_series_gpu` | `apple_gpu` |

`GPUTargetProfile` remains the typed alternative for NVIDIA profiles. The
authoritative normalization implementation is
`tessera.compiler.capabilities.normalize_target`; update this table and its
tests when adding an alias.

### Logical shape and placement

```python
import tessera

domain = tessera.domain.Rect((4, 128, 256))
placement = tessera.dist.Block(mesh_axes=("dp", "tp"))
x = tessera.array.from_domain(
    domain, dtype="bf16", distribution=placement,
)
```

`domain` describes the logical tensor shape. `placement` describes how it is
partitioned. Do not merge those concepts or infer one from the other.

### Tile-level mesh launch

```python
import tessera

@tessera.kernel
def scale(x):
    return tessera.ops.mul(x, 2.0)

results = tessera.index_launch(axis="tp")(scale)(x.parts("tp"))
```

This is the portable programming form. Its implementation may be reference,
artifact-only, or native on a given target; that distinction belongs in backend
and generated execution documentation.

## Tensor Vocabulary and Dtypes

The canonical tensor attributes are **shape**, **dtype**, **layout**,
**device/target**, **distribution**, and **numeric policy**. Use the complete
vocabulary, canonical dtype spellings, accepted input aliases, and promotion
rules in [`tessera_tensor_attributes.md`](reference/tessera_tensor_attributes.md).

In particular:

- Storage dtype and numeric policy are separate. Do not present TF32 as a
  storage dtype.
- Accepted input aliases normalize at API boundaries; canonical spellings are
  stored in metadata and emitted IR.
- Target-gated or planned dtypes must be identified as such by the tensor
  attribute reference and target evidence, not promoted to universal support
  by a Python example.

## Target and Execution Vocabulary

Use the following terms consistently:

| Term | Meaning |
|---|---|
| **target** | Requested compilation or execution destination, such as x86, Apple, ROCm, or NVIDIA |
| **compiler form** | Graph IR, Schedule IR, Tile IR, Target IR, or an emitted artifact |
| **runtime executor** | component that performs work, such as a reference evaluator, CPU JIT, or hardware runtime |
| **execution unit** | the operation, fused group, compiled subgraph, or program actually launched |
| **placement** | distribution of logical data across mesh axes or devices |
| **proof** | reproducible evidence that a stated execution path occurred |

Avoid using “supported” without qualifying which of these is meant. For
example, artifact emission, native runtime execution, and hardware proof are
not interchangeable.

Start target-specific reading at [`docs/backends/README.md`](backends/README.md).
Use the generated dashboards for the current answer to “what runs?”; use
backend audit documents for gaps and plans.

## Authoring and Migration Rules

When adding or changing a public symbol:

1. Choose one public namespace and one canonical spelling.
2. Define the exact signature and semantics in the owning API/spec document.
3. Add or update focused tests for the public contract.
4. Add a short row here only when the name or namespace itself is new or
   changes; do not copy the whole API catalog.
5. Put target execution evidence in generated audits and backend documentation,
   not in this naming guide.
6. Mark an old public spelling as deprecated or historical where it remains in
   migration material; do not silently leave competing “canonical” names.

For a documentation conflict, resolve it in this order:

1. canonical spelling — this document;
2. exact Python behavior — the owning API/spec document and its tests;
3. IR semantics — the corresponding IR specification and verifier tests;
4. live execution/support — generated audit evidence;
5. rationale and future work — backend or compiler audit documents.

## Related Entry Points

- New users: [`GETTING_STARTED.md`](GETTING_STARTED.md)
- Documentation authority tree: [`docs/README.md`](README.md)
- Python API detail: [`docs/spec/PYTHON_API_SPEC.md`](spec/PYTHON_API_SPEC.md)
- Compiler and IR detail: [`docs/spec/COMPILER_REFERENCE.md`](spec/COMPILER_REFERENCE.md)
- Backend navigation: [`docs/backends/README.md`](backends/README.md)
- Current execution evidence: [`docs/audit/generated/`](audit/generated/)
- Error handling: [`Tessera Error Handling and Diagnostics Guide`](guides/Tessera_Error_Handling_And_Diagnostics_Guide.md)
