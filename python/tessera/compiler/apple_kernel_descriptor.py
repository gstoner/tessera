"""Apple kernel descriptors — declarative dispatch contract for every
Apple GPU kernel family (P1, 2026-06-02).

APPLE_AUDIT "Still Open": *binding specs are not universal; Target IR
does too much; Apple source strings / fusion recognition / runtime
dispatch decisions need a descriptor-backed backend registry.* Today
the packaged-kernel path has a real binding contract
(``apple_mlpkg.AppleKernelBindingSpec`` — tensor-level reflection with
``buffer_index``), but the other families (MPS, custom MSL, MPSGraph,
conv, encode-session) describe themselves *implicitly* through the
manifest dict + the ``tessera_apple_gpu_<op>_<dtype>`` symbol-naming
convention + the driver envelope sets.

This module promotes a **single declarative descriptor** that every
Apple GPU kernel family carries — ``AppleKernelDescriptor`` — synthesized
(no duplicated truth) from the existing sources:

* ``backend_manifest.manifest_for(op)`` — the apple_gpu ``BackendKernelEntry``
  (status / dtypes / runtime_symbol / shape_envelope / packaged binding spec).
* ``driver._APPLE_GPU_*_OPS`` — the runtime envelope, classifying which
  *family* (mps / msl / mpsgraph / conv / reduction / projection / linalg)
  actually dispatches each op.
* ``apple_gpu_chain.ENCODE_OP_REGISTRY`` — whether the op has a
  one-command-buffer encode-session lane.

The reflection-level ``AppleKernelBindingSpec`` stays the packaged-only
sub-contract (only ``.mtlpackage`` kernels expose buffer-index
reflection); it rides along on the descriptor's ``binding_spec`` field
when present. Everything else gets a uniform, queryable descriptor —
the declarative registry the audit asked for, without ripping out the
runtime dispatch logic yet.

Drift-gated by ``tests/unit/test_apple_kernel_descriptor.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


#: The Apple GPU kernel families a descriptor can belong to. Precedence
#: order (most-specific first) is used to classify an op that appears in
#: more than one driver envelope set.
APPLE_KERNEL_FAMILIES: tuple[str, ...] = (
    "packaged",    # .mtlpackage MTL4 ML pipeline (carries a binding spec)
    "msl",         # hand-written Metal Shading Language kernel
    "mps",         # MetalPerformanceShaders GEMM family
    "conv",        # convolution (MPSGraph convolution2D / im2col+matmul)
    "mpsgraph",    # MetalPerformanceShadersGraph activation/norm/rowop lane
    "reduction",   # MPSGraph reduce/scan lane
    "projection",  # matmul/bmm-backed projection (linear_general/qkv)
    "linalg",      # MPSMatrix decomposition/solve
    "encode_session",  # dispatched only via the one-command-buffer encode lane (e.g. bmm)
    "other",       # has a manifest entry but no envelope family (e.g. kv_cache_read)
)


@dataclass(frozen=True)
class AppleKernelDescriptor:
    """Declarative dispatch contract for one Apple GPU kernel.

    Fields:

    * ``op_name`` — bare op name (manifest key, e.g. ``"matmul"``).
    * ``family`` — one of :data:`APPLE_KERNEL_FAMILIES`.
    * ``status`` — manifest status (``fused`` / ``reference`` /
      ``hardware_verified`` / ``packaged`` / ``artifact_only`` / ``planned``).
    * ``dtypes`` — canonical dtype names the kernel covers.
    * ``runtime_symbol`` — the C ABI dispatch entry point (or ``None``
      for artifact-only / capability-registered rows).
    * ``shape_envelope`` — free-form validated-shape documentation.
    * ``encode_eligible`` — True iff the op has a one-command-buffer
      encode-session lane (``apple_gpu_chain.ENCODE_OP_REGISTRY``).
    * ``binding_spec`` — packaged-only reflection contract
      (``apple_mlpkg.AppleKernelBindingSpec``); ``None`` otherwise.
    """
    op_name: str
    family: str
    status: str
    dtypes: tuple[str, ...]
    runtime_symbol: Optional[str]
    shape_envelope: Optional[str]
    encode_eligible: bool
    binding_spec: Optional[object] = None

    def __post_init__(self) -> None:
        if self.family not in APPLE_KERNEL_FAMILIES:
            raise ValueError(
                f"AppleKernelDescriptor {self.op_name!r}: family "
                f"{self.family!r} not in {APPLE_KERNEL_FAMILIES}")

    @property
    def is_runtime_executable(self) -> bool:
        """True when the descriptor names a real dispatch entry point
        (runtime_symbol) or is a packaged kernel with a binding spec."""
        if self.family == "packaged":
            return self.binding_spec is not None
        return self.runtime_symbol is not None


def _strip_prefix(op: str) -> str:
    return op[len("tessera."):] if op.startswith("tessera.") else op


def _classify_family(op_name: str, status: str,
                     encode_eligible: bool) -> str:
    """Resolve the kernel family for a bare op name, most-specific-first.

    Reads the driver envelope sets so the classification stays in lock-
    step with what actually dispatches at runtime (no second truth). An
    op that is not in any per-op envelope but has an encode-session lane
    (e.g. ``bmm``) classifies as ``encode_session``; a manifest-only op
    with neither (e.g. ``kv_cache_read``) is ``other``."""
    from . import driver as _drv

    if status == "packaged":
        return "packaged"

    def _has(attr: str) -> bool:
        env: frozenset[str] = getattr(_drv, attr, frozenset())
        return f"tessera.{op_name}" in env or op_name in env

    if _has("_APPLE_GPU_MSL_OPS"):
        return "msl"
    if _has("_APPLE_GPU_MPS_OPS"):
        return "mps"
    if _has("_APPLE_GPU_CONV_OPS"):
        return "conv"
    if _has("_APPLE_GPU_MPSGRAPH_OPS"):
        return "mpsgraph"
    if _has("_APPLE_GPU_REDUCTION_OPS"):
        return "reduction"
    if _has("_APPLE_GPU_PROJECTION_OPS"):
        return "projection"
    if _has("_APPLE_GPU_LINALG_OPS"):
        return "linalg"
    if encode_eligible:
        return "encode_session"
    return "other"


def _encode_eligible(op_name: str) -> bool:
    try:
        from ..apple_gpu_chain import ENCODE_OP_REGISTRY
    except Exception:
        return False
    return any(name == op_name for (name, _dtype) in ENCODE_OP_REGISTRY)


def apple_kernel_descriptor(op_name: str) -> Optional[AppleKernelDescriptor]:
    """Synthesize the descriptor for one Apple GPU op, or ``None`` when
    the op has no apple_gpu manifest entry. ``op_name`` may be bare
    (``"matmul"``) or dotted (``"tessera.matmul"``)."""
    from . import backend_manifest as _bm

    bare = _strip_prefix(op_name)
    entry = None
    for e in _bm.manifest_for(bare):
        if e.target == "apple_gpu":
            entry = e
            break
    if entry is None:
        return None
    enc = _encode_eligible(bare)
    return AppleKernelDescriptor(
        op_name=bare,
        family=_classify_family(bare, entry.status, enc),
        status=entry.status,
        dtypes=tuple(entry.dtypes),
        runtime_symbol=entry.runtime_symbol,
        shape_envelope=entry.shape_envelope,
        encode_eligible=enc,
        binding_spec=entry.apple_binding_spec,
    )


def all_apple_kernel_descriptors() -> dict[str, AppleKernelDescriptor]:
    """One descriptor per op in the Apple GPU manifest inventory.
    Keyed by bare op name."""
    from . import backend_manifest as _bm

    out: dict[str, AppleKernelDescriptor] = {}
    for op_name in _bm._APPLE_GPU_KERNELS:
        desc = apple_kernel_descriptor(op_name)
        if desc is not None:
            out[op_name] = desc
    return out


__all__ = [
    "APPLE_KERNEL_FAMILIES",
    "AppleKernelDescriptor",
    "apple_kernel_descriptor",
    "all_apple_kernel_descriptors",
]
