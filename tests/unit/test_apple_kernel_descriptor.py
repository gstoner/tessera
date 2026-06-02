"""P1 (2026-06-02) — Apple kernel descriptor unification.

Pins the declarative ``AppleKernelDescriptor`` surface that promotes
binding/dispatch metadata beyond packaged kernels (APPLE_AUDIT
"binding specs are not universal" / "descriptor-backed backend
registry"). The descriptors are *synthesized* from existing truth
(manifest + driver envelope + encode registry), so these tests double
as a drift gate: if the manifest and the driver envelope disagree about
an Apple op, the family classification or executability check fails.
"""

from __future__ import annotations

import pytest

from tessera.compiler.apple_kernel_descriptor import (
    APPLE_KERNEL_FAMILIES,
    AppleKernelDescriptor,
    all_apple_kernel_descriptors,
    apple_kernel_descriptor,
)
from tessera.compiler import backend_manifest as bm


def test_every_apple_manifest_op_has_a_descriptor():
    descs = all_apple_kernel_descriptors()
    for op in bm._APPLE_GPU_KERNELS:
        assert op in descs, f"{op}: no Apple kernel descriptor synthesized"
        assert isinstance(descs[op], AppleKernelDescriptor)


def test_descriptor_family_is_valid():
    for op, desc in all_apple_kernel_descriptors().items():
        assert desc.family in APPLE_KERNEL_FAMILIES, (op, desc.family)


def test_descriptor_dtypes_match_manifest():
    for op, desc in all_apple_kernel_descriptors().items():
        entry = next(e for e in bm.manifest_for(op) if e.target == "apple_gpu")
        assert desc.dtypes == tuple(entry.dtypes), op
        assert desc.status == entry.status, op
        assert desc.runtime_symbol == entry.runtime_symbol, op


def test_dotted_and_bare_names_resolve_identically():
    a = apple_kernel_descriptor("matmul")
    b = apple_kernel_descriptor("tessera.matmul")
    assert a == b


def test_unknown_op_returns_none():
    assert apple_kernel_descriptor("definitely_not_an_apple_op") is None


def test_msl_family_for_custom_kernels():
    # rope / flash_attn / softmax / gelu ship as hand-written MSL kernels.
    for op in ("rope", "flash_attn", "softmax", "gelu"):
        assert apple_kernel_descriptor(op).family == "msl", op


def test_mpsgraph_family_for_activation_norm_ops():
    for op in ("rmsnorm", "layer_norm", "silu", "relu"):
        assert apple_kernel_descriptor(op).family == "mpsgraph", op


def test_mps_family_for_matmul():
    assert apple_kernel_descriptor("matmul").family == "mps"


def test_conv_family_for_conv2d():
    assert apple_kernel_descriptor("conv2d").family == "conv"


def test_encode_session_family_for_bmm():
    """bmm is not in the per-op runtime envelope — it dispatches only
    via the one-command-buffer encode lane, so it classifies as
    encode_session (not the catch-all 'other')."""
    d = apple_kernel_descriptor("bmm")
    assert d.family == "encode_session"
    assert d.encode_eligible is True


def test_encode_eligible_matches_chain_registry():
    from tessera.apple_gpu_chain import ENCODE_OP_REGISTRY
    eligible = {name for (name, _d) in ENCODE_OP_REGISTRY}
    for op, desc in all_apple_kernel_descriptors().items():
        assert desc.encode_eligible == (op in eligible), op


def test_hardware_verified_descriptors_are_runtime_executable():
    """Every hardware_verified Apple kernel must name a real dispatch
    entry point (the descriptor's executability predicate)."""
    for op, desc in all_apple_kernel_descriptors().items():
        if desc.status == "hardware_verified":
            assert desc.is_runtime_executable, (
                f"{op}: hardware_verified but no runtime_symbol/binding")


def test_packaged_descriptor_carries_binding_spec_when_present():
    """If any manifest entry is a packaged kernel, its descriptor must
    surface the AppleKernelBindingSpec (the packaged-only sub-contract).
    No packaged rows in the default manifest today — assert the
    family/contract invariant holds for any that exist."""
    for op, desc in all_apple_kernel_descriptors().items():
        if desc.family == "packaged":
            assert desc.binding_spec is not None, (
                f"{op}: packaged descriptor missing binding_spec")
