"""Project 3 (2026-06-01) — lock the device_verified_abi promotion.

The 8 encode-eligible ops (softmax, softmax_safe, gelu, rope,
flash_attn, rmsnorm, layer_norm, silu, bmm) carry a real per-op
``_dev_f32_enc`` C ABI symbol AND a numerical-comparison fixture, so
their Apple GPU manifest entries qualify for the top rung of the
readiness ladder: ``status == "device_verified_abi"``.

This file is the single drift gate for the promotion. It locks:

* **Every promoted op carries the contract**: status=device_verified_abi
  AND runtime_symbol points at a real C ABI export AND
  execute_compare_fixture points at a real Python file.
* **runtime_symbol is actually exported by the runtime**: the symbol
  name must be findable in
  ``src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm``.
* **execute_compare_fixture file exists** on disk.
* **shape_envelope is documented** (free-form, just-must-not-be-empty).
* **The conformance matrix accepts device_verified_abi** as a real
  compile/link path: rendering uses the new status seamlessly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.compiler import backend_manifest as bm
from tessera.compiler import conformance_matrix as cm
from tessera.compiler import pipeline_gates as pg


_REPO = Path(__file__).resolve().parents[2]
_RUNTIME_SRC_PATH = (
    _REPO / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend"
    / "runtime" / "apple_gpu_runtime.mm"
)


# 8 encode-eligible ops promoted by Project 3, conv2d promoted by Project 5
# (2026-06-01) once its encode-session lane landed, and the packed-int4
# quantized matmul lane once its runtime symbol + numerical proof landed.
HARDWARE_VERIFIED_OPS = (
    "softmax", "softmax_safe", "gelu", "rope", "flash_attn",
    "rmsnorm", "layer_norm", "silu", "bmm", "conv2d", "quantized_matmul",
)


@pytest.fixture(scope="module")
def runtime_src() -> str:
    return _RUNTIME_SRC_PATH.read_text()


@pytest.mark.parametrize("op", HARDWARE_VERIFIED_OPS)
def test_apple_gpu_entry_is_hardware_verified(op: str) -> None:
    entries = {e.target: e for e in bm.manifest_for(op)}
    assert "apple_gpu" in entries, f"{op}: no apple_gpu manifest entry"
    ag = entries["apple_gpu"]
    assert ag.status == "device_verified_abi", (
        f"{op}: expected status='device_verified_abi', got {ag.status!r}")
    assert ag.is_hardware_verified, (
        f"{op}: is_hardware_verified must be True")


@pytest.mark.parametrize("op", HARDWARE_VERIFIED_OPS)
def test_runtime_symbol_is_real_c_abi_export(
        op: str, runtime_src: str) -> None:
    entries = {e.target: e for e in bm.manifest_for(op)}
    sym = entries["apple_gpu"].runtime_symbol
    assert sym, f"{op}: runtime_symbol must be set on device_verified_abi"
    # The symbol must be a real C ABI export — `extern "C" ... <sym>(`
    # somewhere in the runtime source. We look for the signature
    # pattern instead of plain text to avoid matching comments.
    pattern = re.compile(
        r"extern\s+\"C\"\s+\w+\s+" + re.escape(sym) + r"\s*\(")
    assert pattern.search(runtime_src), (
        f"{op}: runtime_symbol={sym!r} not found as an extern \"C\" "
        f"definition in apple_gpu_runtime.mm")


@pytest.mark.parametrize("op", HARDWARE_VERIFIED_OPS)
def test_execute_compare_fixture_exists(op: str) -> None:
    entries = {e.target: e for e in bm.manifest_for(op)}
    fixture = entries["apple_gpu"].execute_compare_fixture
    assert fixture, (
        f"{op}: execute_compare_fixture must be set on device_verified_abi")
    assert (_REPO / fixture).is_file(), (
        f"{op}: fixture {fixture!r} does not exist on disk")


@pytest.mark.parametrize("op", HARDWARE_VERIFIED_OPS)
def test_shape_envelope_documented(op: str) -> None:
    entries = {e.target: e for e in bm.manifest_for(op)}
    env = entries["apple_gpu"].shape_envelope
    assert env, (
        f"{op}: shape_envelope must be documented on device_verified_abi")
    assert len(env) >= 8, (
        f"{op}: shape_envelope={env!r} looks too short to be meaningful")


def test_promoted_count_is_eight():
    """If a future change accidentally promotes more (or fewer) ops,
    surface it here so the promotion stays explicit."""
    promoted = []
    for op_name, payload in bm._APPLE_GPU_KERNELS.items():
        if payload.get("status") == "device_verified_abi":
            promoted.append(op_name)
    promoted.sort()
    expected = sorted(HARDWARE_VERIFIED_OPS)
    assert promoted == expected, (
        f"Expected exactly these apple_gpu device_verified_abi ops:\n"
        f"  {expected}\n"
        f"Got:\n"
        f"  {promoted}")


def test_conformance_matrix_treats_hardware_verified_as_complete():
    """The conformance matrix backend_compile column must accept
    device_verified_abi as a complete compile path. Pin via a direct
    function call on the row aggregator."""
    # All-device_verified_abi case → complete.
    s = cm._proof_status_from_backend_compile(
        ["device_verified_abi"], "softmax", "apple_gpu"
    )
    assert s == cm.PROOF_COMPLETE
    # Mixed device_verified_abi + fused → still complete.
    s = cm._proof_status_from_backend_compile(
        ["device_verified_abi", "fused"], "softmax", "apple_gpu"
    )
    assert s == cm.PROOF_COMPLETE
    # device_verified_abi + planned → planned (weakest wins).
    s = cm._proof_status_from_backend_compile(
        ["device_verified_abi", "planned"], "softmax", "apple_gpu"
    )
    assert s == cm.PROOF_PLANNED


def test_pipeline_link_gate_treats_hardware_verified_as_pass():
    """The pipeline_gates link gate must accept device_verified_abi as
    linkable — otherwise the conformance dashboard's link column would
    regress to FAIL after the promotion."""
    r = pg._eval_link("apple_gpu", "softmax")
    assert r.status == pg.STATUS_PASS, (
        f"expected PASS, got {r.status!r} reason={r.reason!r}")
