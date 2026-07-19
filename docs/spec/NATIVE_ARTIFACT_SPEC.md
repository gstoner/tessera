---
status: Normative
classification: Normative
authority: Compiler native-image and launch schema
last_updated: 2026-07-19
---

# Native image and launch descriptor specification

This specification defines the portable compiler/runtime handoff introduced by
`E2E-SPINE-1` and its orchestration join in `E2E-SPINE-2`. Schema authority is
`python/tessera/compiler/native_artifact.py`; the carriers and generic launch
orchestration are `python/tessera/compiler/driver.py`,
`python/tessera/compiler/canonical_compile.py`, and `python/tessera/runtime.py`.

## 1. Versions and ownership

- Native images use `tessera.native_image.v1`.
- Launch descriptors use `tessera.launch_descriptor.v1`.
- The compiler owns image identity, entry symbols, stable ABI identifiers,
  ordered bindings, launch requirements, and provenance.
- Backends own physical fragments, instructions, work distribution, and
  performance schedules inside Target IR and native payloads.
- Runtime routes and selectors do not become executable merely because these
  schemas exist.

## 2. Native images

A `NativeImageArtifact` records the canonical target, exact architecture,
registered pipeline, compiler and toolchain fingerprints, Target-IR digest,
registered binary format, native bytes, entry points, compile state, and an
optional resource record. It also carries the ordered device-library set used
at the LLVM stage. Each record contains a backend-owned logical name, SHA-256
content digest, and one registered link mode (`llvm_link_only_needed`,
`compiler_driver`, or `embedded`); host installation paths are not serialized.

Three SHA-256 identities are distinct:

1. `payload_digest` hashes the native bytes.
2. `image_digest` hashes immutable image identity plus `payload_digest`.
3. `cache_key` hashes the pre-compilation inputs used to find a compatible
   image. Cold/warm/prepackaged state and measured resources do not affect it.

Device-library records participate in both image and cache identity. A changed
CUDA libdevice or ROCm OCML/OCKL/OCLC input therefore cannot reuse an image
compiled against different device bitcode. The optional empty v1 field keeps
older serialized artifacts readable; reserialization writes the explicit set.

Deserialization recomputes all three and rejects drift. Supported v1 formats
are `ptx`, `cubin`, `hsaco`, `elf`, `object`, `shared_object`, `metallib`, and
`msl_package`. New formats require an explicit registry update.

## 3. Launch descriptors

A `LaunchDescriptor` names one image digest, entry symbol, and ABI identifier.
Its arguments use one contiguous ordinal space across buffers and scalars.
Buffer contracts record name, direction, canonical storage dtype, rank, layout,
and power-of-two alignment. Shape guards support `eq`, `min`, `max`, and
`multiple_of` predicates.

Launch geometry is either fixed three-dimensional grid/workgroup data or one
registered runtime-computed policy. Dynamic local memory, workspace bytes,
alignment, lifetime (`launch` or `session`), initialization (`undefined`,
`zero`, or session-only `preserve`), ordered submission, residency,
synchronization tokens, and JSON-safe provenance are explicit. Session
workspace belongs to a stateful runtime handle and must survive every launch in
that handle until its teardown protocol drains outstanding work.
`ordered_submission` describes host queue/stream submission only. It does not
claim an intra-kernel memory order, atomicity for vector or packed accesses, or
a backend fence/scope. Those properties belong to typed Target IR and the
backend memory model; for PTX in particular, packed/vector accesses decompose
into scalar operations whose element order is unspecified.

`descriptor_digest` fingerprints the full descriptor. `cache_fingerprint`
binds that descriptor to its image digest. Runtime invocation validation checks
the image/symbol/ABI join and exact buffer/scalar contracts before backend
submission.

## 4. Canonical orchestration

`CompileArtifactBundle` carries optional `native_image` and
`launch_descriptor` fields and records the highest honest state:

- `artifact_only`: no typed Target IR was produced;
- `compileable`: Target IR exists, but no native image exists;
- `packaged`: an image is present and joined to the bundle's exact target,
  requested pipeline or target registry's declared producer, and Target-IR
  digest;
- `launchable`: a descriptor additionally joins that image, entry symbol, and
  ABI identifier.

`CompileResult.to_runtime_artifact()` transfers the same typed objects; it does
not reconstruct them from metadata. `RuntimeArtifact` JSON embeds both objects,
rejoins the persisted Target IR to the image's Target-IR digest, checks their
nested hashes and descriptor joins, and includes them in the outer artifact
hash. The AOT compilation-cache identity includes the image cache key and
descriptor cache fingerprint. New persistent entries retain a
`tessera.compilation_cache_entry.v1` manifest joining the lookup key, outer
artifact hash, image cache key, and descriptor fingerprint; mismatched swaps
are rejected. Manifest-free legacy entries remain readable.

`runtime.launch()` gives a present descriptor priority over legacy execution
matrix routing. It binds named buffers and scalars, validates the entire
invocation, then submits only through an exact-target launcher registered with
an explicit accepted binary-format set. A missing launcher is an explicit
`unimplemented` result and never falls back to a legacy candidate. Binding or
staleness failures occur before the backend submission callback.

Launcher callbacks own module loading, device allocation, geometry-policy
evaluation, workspace allocation, stream/queue submission, and cleanup. This
portable layer does not implement those architecture-specific actions.

## 5. Diagnostics

Failures use registered codes:

- `E_NATIVE_IMAGE_SCHEMA`
- `E_NATIVE_IMAGE_DIGEST_MISMATCH`
- `E_LAUNCH_DESCRIPTOR_SCHEMA`
- `E_LAUNCH_BINDING_MISMATCH`
- `E_LAUNCH_STALE_IMAGE`

Wrong JSON types are rejected rather than coerced. Non-finite resource or
provenance values are not serializable contract data.

## 6. Non-goals

The portable schemas contain no CUDA warp map, AMD wave layout, Metal
threadgroup schedule, or x86 vector width. They do not discover symbols from
backend binaries, infer architectures after code generation, select candidates,
or claim exact-device proof. The generic launcher registry is an exact-target
submission hook, not a schedule or selector registry.
