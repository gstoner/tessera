"""Apple GPU target descriptor + tensor ABI (Phase 3 refinement).

Models Apple GPU as an *explicit target descriptor* rather than a bare
``target="apple_gpu"`` string. The descriptor is the contract boundary between
the compiler (what was produced) and the runtime (what can actually execute).

Three DISTINCT execution contracts — never conflate them:

* ``metal_artifact`` — a compile-time artifact only. **No runtime execution
  claim.** Pure compile / containerization produces this.
* ``metal_runtime``  — the classic MPS / MSL / MPSGraph runtime lane (the
  existing, mature Apple GPU execution path).
* ``mtl4_runtime``   — the Metal 4 cooperative-tensor-op runtime (the MTL4
  command model: command queue / allocator / compiler / cooperative ``tensor``
  ops in MSL 4.0).

The lane split is load-bearing (see memory ``apple-metal4-ml-apis``):
MPSGraph/classic-queue (``metal_runtime``) is a *separate surface* from MTL4
cooperative tensor ops (``mtl4_runtime``). MPSGraph does **not** run on the MTL4
command model; a module must not be labeled across the two.

Metal 4 features are *capability-gated requirements*. The capability vocabulary
mirrors the runtime probe bits in
``_apple_gpu_dispatch.apple_gpu_capabilities_snapshot`` (command queue /
allocator / compiler / tensor / MSL 4.0) plus the archive + ML-encoder surfaces.

Compile-time status (``execution_contract``) is kept separate from *observed*
runtime capabilities: a descriptor produced by pure compile carries
``required_capabilities`` (what it would need) but never ``observed_capabilities``
(what a host actually has) — those come only from a real runtime probe.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

# ── Execution contracts ──────────────────────────────────────────────────────

METAL_ARTIFACT = "metal_artifact"
METAL_RUNTIME = "metal_runtime"
MTL4_RUNTIME = "mtl4_runtime"
EXECUTION_CONTRACTS = (METAL_ARTIFACT, METAL_RUNTIME, MTL4_RUNTIME)

# ── Capability vocabulary (Metal 4 feature gates) ────────────────────────────
# Names align with _apple_gpu_dispatch._APPLE_GPU_CAP_BITS + archive/ml-encoder.

CAP_COMMAND_QUEUE = "mtl4_command_queue"
CAP_COMMAND_ALLOCATOR = "mtl4_command_allocator"
CAP_COMPILER = "mtl4_compiler"
CAP_TENSOR = "mtl_tensor"
CAP_MSL_4_0 = "msl_4_0"
CAP_ARCHIVE = "mtl4_archive"
CAP_ML_ENCODER = "mtl4_ml_encoder"
APPLE_CAPABILITIES = (
    CAP_COMMAND_QUEUE,
    CAP_COMMAND_ALLOCATOR,
    CAP_COMPILER,
    CAP_TENSOR,
    CAP_MSL_4_0,
    CAP_ARCHIVE,
    CAP_ML_ENCODER,
)

# The MTL4 command model requires this trio regardless of op.
_MTL4_BASE_CAPS = (CAP_COMMAND_QUEUE, CAP_COMMAND_ALLOCATOR, CAP_COMPILER)
# Cooperative tensor ops (e.g. an MTL4 cooperative matmul) additionally need:
_MTL4_COOPERATIVE_CAPS = (CAP_TENSOR, CAP_MSL_4_0)

# ── Tensor / buffer ABI resource kinds ───────────────────────────────────────

RESOURCE_MTL_BUFFER = "mtl_buffer"          # plain MTLBuffer view
RESOURCE_MTL_TENSOR_VIEW = "mtl_tensor_view"  # MTLTensor over a buffer (MTL4)
RESOURCE_MTLPACKAGE_TENSOR = "mtlpackage_tensor"  # ML-package input/output tensor
RESOURCE_KINDS = (
    RESOURCE_MTL_BUFFER,
    RESOURCE_MTL_TENSOR_VIEW,
    RESOURCE_MTLPACKAGE_TENSOR,
)

# Fixed target identity for Apple GPU.
_VENDOR = "apple"
_API = "metal"
_TRIPLE = "air64-apple-macosx"
_ARCH = "apple-metal"
_MEMORY_MODEL = "unified_64"


class AppleTargetError(ValueError):
    """Raised when an Apple GPU target descriptor / ABI is malformed or violates
    a contract rule."""


# ── Required-capability resolution ───────────────────────────────────────────


def required_capabilities_for(
    execution_contract: str, *, cooperative_tensor: bool = False
) -> list[str]:
    """The capabilities an Apple GPU module *requires* for a given execution
    contract.

    * ``metal_artifact`` / ``metal_runtime`` require none of the MTL4 gates —
      classic MPS/MSL/MPSGraph does not use the MTL4 command model.
    * ``mtl4_runtime`` requires the command-model trio (queue/allocator/compiler);
      a cooperative-tensor op additionally requires ``mtl_tensor`` + ``msl_4_0``.
    """
    if execution_contract not in EXECUTION_CONTRACTS:
        raise AppleTargetError(
            f"unknown execution_contract {execution_contract!r}; "
            f"expected one of {EXECUTION_CONTRACTS}"
        )
    if execution_contract != MTL4_RUNTIME:
        return []
    caps = list(_MTL4_BASE_CAPS)
    if cooperative_tensor:
        caps += list(_MTL4_COOPERATIVE_CAPS)
    return caps


# ── Descriptor ───────────────────────────────────────────────────────────────


def apple_target_descriptor(
    execution_contract: str,
    *,
    required_capabilities: Iterable[str] | None = None,
    cooperative_tensor: bool = False,
) -> dict[str, Any]:
    """Build the Apple GPU target descriptor (for generated Target IR attrs and
    runtime-artifact metadata).

    Vendor / api / triple / arch / memory_model are fixed; ``execution_contract``
    + ``required_capabilities`` carry the variable contract. If
    ``required_capabilities`` is omitted it is derived from the contract via
    :func:`required_capabilities_for`.
    """
    if execution_contract not in EXECUTION_CONTRACTS:
        raise AppleTargetError(
            f"unknown execution_contract {execution_contract!r}; "
            f"expected one of {EXECUTION_CONTRACTS}"
        )
    if required_capabilities is None:
        reqs = required_capabilities_for(
            execution_contract, cooperative_tensor=cooperative_tensor
        )
    else:
        reqs = list(required_capabilities)
    for cap in reqs:
        if cap not in APPLE_CAPABILITIES:
            raise AppleTargetError(
                f"unknown capability {cap!r}; expected one of {APPLE_CAPABILITIES}"
            )
    desc = {
        "vendor": _VENDOR,
        "api": _API,
        "triple": _TRIPLE,
        "arch": _ARCH,
        "memory_model": _MEMORY_MODEL,
        "execution_contract": execution_contract,
        "required_capabilities": reqs,
    }
    return desc


# ── Tensor / buffer ABI ──────────────────────────────────────────────────────


def _row_major_strides(shape: Sequence[int]) -> list[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def apple_tensor_abi(
    dtype: str,
    shape: Sequence[int],
    *,
    strides: Sequence[int] | None = None,
    offset_bytes: int = 0,
    resource_kind: str = RESOURCE_MTL_BUFFER,
    resident: bool = False,
) -> dict[str, Any]:
    """Explicit tensor/buffer ABI for an Apple artifact tensor.

    Captures dtype, rank, shape, strides (row-major by default), offset,
    resource kind (buffer vs tensor-view vs ml-package tensor), and residency.
    ``resident`` is ``True`` only for device-resident paths (TsDeviceTensor);
    classic host→device copy paths are non-resident.
    """
    shape = tuple(int(s) for s in shape)
    if strides is None:
        strides = _row_major_strides(shape)
    else:
        strides = tuple(int(s) for s in strides)
        if len(strides) != len(shape):
            raise AppleTargetError(
                f"strides rank {len(strides)} != shape rank {len(shape)}"
            )
    if resource_kind not in RESOURCE_KINDS:
        raise AppleTargetError(
            f"unknown resource_kind {resource_kind!r}; expected one of {RESOURCE_KINDS}"
        )
    # buffer-vs-tensor view: a tensor-view / ml-package tensor is a "tensor"
    # view; a plain buffer is a "buffer" view.
    view = "buffer" if resource_kind == RESOURCE_MTL_BUFFER else "tensor"
    return {
        "dtype": str(dtype),
        "rank": len(shape),
        "shape": list(shape),
        "strides": list(strides),
        "offset_bytes": int(offset_bytes),
        "resource_kind": resource_kind,
        "resident": bool(resident),
        "view": view,
    }


# ── Contract validation ──────────────────────────────────────────────────────


def validate_descriptor(
    desc: dict[str, Any], *, observed_capabilities: dict[str, bool] | None = None
) -> dict[str, Any]:
    """Validate an Apple GPU target descriptor against the contract rules.

    * ``execution_contract`` must be one of the three.
    * an ``mtl4_runtime`` descriptor's ``required_capabilities`` must include the
      MTL4 command-model trio.
    * if ``observed_capabilities`` is given (a real runtime probe ran), every
      required capability must be observed — otherwise the host cannot actually
      execute this contract.

    Returns ``desc`` on success; raises :class:`AppleTargetError` otherwise.
    """
    ec = desc.get("execution_contract")
    if ec not in EXECUTION_CONTRACTS:
        raise AppleTargetError(f"invalid execution_contract {ec!r}")
    reqs = list(desc.get("required_capabilities", []))
    if ec == MTL4_RUNTIME:
        missing = [c for c in _MTL4_BASE_CAPS if c not in reqs]
        if missing:
            raise AppleTargetError(
                f"mtl4_runtime descriptor must require {_MTL4_BASE_CAPS}; "
                f"missing {missing}"
            )
    else:
        # metal_artifact / metal_runtime are classic lanes; they must not smuggle
        # MTL4 command-model requirements (that would imply the wrong lane).
        smuggled = [c for c in reqs if c in _MTL4_BASE_CAPS]
        if smuggled:
            raise AppleTargetError(
                f"{ec} descriptor must not require MTL4 command-model caps "
                f"{smuggled} (lane split: classic MPS is not the MTL4 model)"
            )
    if observed_capabilities is not None:
        absent = [c for c in reqs if not observed_capabilities.get(c, False)]
        if absent:
            raise AppleTargetError(
                f"{ec} requires {absent} but they are not observed on this host"
            )
    return desc


def assert_not_artifact_claiming_runtime(metadata: dict[str, Any]) -> None:
    """Contract guard: an artifact-only Apple GPU module must not claim a runtime
    execution contract.

    If ``metadata.runtime_status == 'artifact_only'`` then any present
    ``target_descriptor.execution_contract`` must be ``metal_artifact`` — never
    ``metal_runtime`` or ``mtl4_runtime``.
    """
    if metadata.get("runtime_status") != "artifact_only":
        return
    desc = metadata.get("target_descriptor")
    if not isinstance(desc, dict):
        return
    ec = desc.get("execution_contract")
    if ec is not None and ec != METAL_ARTIFACT:
        raise AppleTargetError(
            f"artifact-only module claims execution_contract={ec!r}; "
            f"a pure artifact must be {METAL_ARTIFACT!r}"
        )
