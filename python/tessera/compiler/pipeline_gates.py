"""Named pipeline capability gates — audit recommendation **B**.

Today the runtime reports ``execution_kind: "reference_cpu"`` as post-hoc
metadata: if a backend isn't lit up, the path silently falls through to the
numpy reference and the only signal is a string. The audit's framing: replace
that with a named, ordered gate sequence so the *first un-passed gate* is the
diagnostic, not a single inscrutable kind label.

This module defines the canonical seven gates and a **pure aggregator**
``evaluate(target, op_name=None)`` that resolves each gate's status against
existing truth sources (capabilities + backend_manifest + execution_matrix +
toolchain probes). No new private truth — same rule as ``conformance_matrix``.

    legality → codegen → toolchain → link → runtime_abi → hardware_smoke → numerical

Each gate has one of:

* ``pass``  — concretely satisfied by an upstream truth source
* ``fail``  — concretely unsatisfied (this is what the audit calls "the named
              gate"); short ``detail`` string says why
* ``not_evaluated`` — gate skipped because a prior gate already failed
* ``not_applicable`` — gate doesn't apply (e.g. ``hardware_smoke`` for ``cpu``
                       on a Linux box that lacks AMX is still pass, but
                       ``hardware_smoke`` for ``apple_gpu`` on Darwin is
                       evaluable; on Linux it's NA)

Two consumers in mind:

1. The runtime: when a compile request can't be served, the diagnostic names
   the first failing gate (vs. silently falling through to ``reference_cpu``).
   Wiring is staged — this module is the pure-truth layer first; the
   ``runtime.launch()`` integration is the next sub-step.
2. The conformance matrix: each (op, target) cell in ``op_target_conformance.md``
   now also reports its first-failing-gate name, replacing the implicit
   "weakest column wins" story with an explicit named gate.
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Optional

from tessera.compiler import backend_manifest as _bm
from tessera.compiler import capabilities as _cap
from tessera.compiler import execution_matrix as _em
from tessera.compiler import primitive_coverage as _pc


# --- Gate names + status enum --------------------------------------------

#: Graph IR can lower this op for this target.
GATE_LEGALITY = "legality"
#: Target IR / backend lowering pass produces an artifact.
GATE_CODEGEN = "codegen"
#: Backend compiler binary is present at the pinned version.
GATE_TOOLCHAIN = "toolchain"
#: Backend can produce a linked / loadable runtime artifact.
GATE_LINK = "link"
#: Runtime ABI can launch this artifact (``tsrLaunchKernel`` doesn't return
#: ``UNIMPLEMENTED`` for this target's kernel kind).
GATE_RUNTIME_ABI = "runtime_abi"
#: A tiny kernel actually runs on real hardware on this host.
GATE_HARDWARE_SMOKE = "hardware_smoke"
#: A numerical-comparison test exists and passes.
GATE_NUMERICAL = "numerical"

#: Canonical evaluation order.
GATE_ORDER = (
    GATE_LEGALITY,
    GATE_CODEGEN,
    GATE_TOOLCHAIN,
    GATE_LINK,
    GATE_RUNTIME_ABI,
    GATE_HARDWARE_SMOKE,
    GATE_NUMERICAL,
)

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_NOT_APPLICABLE = "not_applicable"

_ALL_STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_NOT_EVALUATED,
                 STATUS_NOT_APPLICABLE)


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: str
    detail: str = ""


# --- Helpers --------------------------------------------------------------

def _normalize_target(target: str) -> str:
    """Collapse per-SM / per-arch sub-target names to the dashboard family
    used by toolchain / hardware-smoke evaluation.

    The capabilities registry keys per-SM (``nvidia_sm80``..``nvidia_sm120``,
    ``rocm_gfx940``..``rocm_gfx1100``), but the toolchain probe is
    family-level (one nvcc serves every NVIDIA SM, one hipcc serves every
    ROCm arch). The dashboard targets are the families; the runtime can
    legitimately pass either form, so normalize here.
    """
    if target.startswith("nvidia_"):
        return "nvidia"
    if target.startswith("rocm_"):
        return "rocm"
    return target


def _manifest_entries(op: str, target: str) -> list[_bm.BackendKernelEntry]:
    """Resolve the manifest rows that describe how ``target`` compiles ``op``.

    Audit fix (2026-05-31, P2): the manifest keys per-target are
    inconsistent — NVIDIA emits per-SM rows (``nvidia_sm80``..``sm120``)
    while ROCm emits a single family row (``rocm`` / ``rocm_blockfp``),
    and Metalium emits ``metalium`` + ``metalium_blockfp``. The previous
    implementation only mapped family→per-SM for NVIDIA, so passing a
    per-arch ROCm target like ``rocm_gfx942`` matched zero rows and the
    codegen gate spuriously failed. We now map symmetrically: ROCm
    sub-arch targets inherit the family manifest row, and the NVIDIA
    sub-arch case keeps exact per-SM matching.
    """
    if not op:
        return []
    entries = _bm.manifest_for(op)
    out: list[_bm.BackendKernelEntry] = []
    for e in entries:
        t = e.target
        if target == "nvidia" and t.startswith("nvidia_"):
            out.append(e)
        elif target.startswith("nvidia_") and t == target:
            out.append(e)
        elif target == "rocm" and (t == "rocm" or t.startswith("rocm_")):
            out.append(e)
        elif target.startswith("rocm_") and t in ("rocm", target):
            # per-arch target inherits the rocm family manifest row, since
            # the manifest keys ROCm by family today.
            out.append(e)
        elif target == "metalium" and t in ("metalium", "metalium_blockfp"):
            out.append(e)
        elif t == target:
            out.append(e)
    return out


def _execution_row(target: str) -> Optional[_em.ExecutionRow]:
    for row in _em.all_rows():
        if row.target == target:
            return row
    return None


def _platform_is_darwin_arm64() -> bool:
    """True iff this is Apple Silicon (Darwin + arm64).

    Audit fix (2026-05-31): the helper used to return ``sys.platform ==
    "darwin"`` only, which is *also* true on Intel Macs. An Intel Mac would
    have falsely passed the apple_cpu / apple_gpu ``hardware_smoke`` gate
    even though it can't run Metal MPS or AMX. ``platform.machine()``
    returns ``"arm64"`` on Apple Silicon and ``"x86_64"`` on Intel; that's
    the actual check the gate is supposed to perform.
    """
    return sys.platform == "darwin" and platform.machine() == "arm64"


# --- Per-gate evaluators (each returns a single GateResult) --------------

def _safe_get_capability(target: str) -> Optional["_cap.TargetCapability"]:
    """``capabilities.normalize_target`` *raises* ValueError for unknown
    targets; we want a quiet ``None`` so the gate evaluation can return a
    ``FAIL`` with a precise reason instead of crashing."""
    try:
        return _cap.get_target_capability(target)
    except ValueError:
        return None


def _eval_legality(target: str, op_name: Optional[str]) -> GateResult:
    """The graph frontend can lower this op (or any op when not specified)."""
    if op_name is None:
        # Target-level legality is whether the capabilities registry knows it.
        tc = _safe_get_capability(target)
        if tc is None:
            return GateResult(GATE_LEGALITY, STATUS_FAIL,
                              f"target {target!r} not in capabilities registry")
        return GateResult(GATE_LEGALITY, STATUS_PASS)
    cov = _pc.all_primitive_coverages().get(op_name)
    if cov is None:
        return GateResult(GATE_LEGALITY, STATUS_FAIL,
                          f"op {op_name!r} not in primitive_coverage registry")
    lowering = cov.contract_status.get("lowering_rule", "planned")
    if lowering in ("complete", "not_applicable"):
        return GateResult(GATE_LEGALITY, STATUS_PASS)
    return GateResult(GATE_LEGALITY, STATUS_FAIL,
                      f"lowering_rule = {lowering}")


def _eval_codegen(target: str, op_name: Optional[str]) -> GateResult:
    """Backend manifest has at least one entry on this target with a
    non-planned status (= some codegen path produces an artifact)."""
    if op_name is None:
        return GateResult(GATE_CODEGEN, STATUS_NOT_APPLICABLE,
                          "target-level codegen is per-op")
    entries = _manifest_entries(op_name, target)
    if not entries:
        return GateResult(GATE_CODEGEN, STATUS_FAIL,
                          f"no backend_manifest entry for {op_name!r} on {target!r}")
    if all(e.status == "planned" for e in entries):
        return GateResult(GATE_CODEGEN, STATUS_FAIL,
                          f"all manifest entries are planned")
    return GateResult(GATE_CODEGEN, STATUS_PASS)


def _eval_toolchain(target: str, op_name: Optional[str]) -> GateResult:
    """The backend compiler binary exists. We probe the host PATH; pinned
    versions live in ``cmake/TesseraToolchainPins.cmake`` and aren't
    re-asserted here (drift gate on the pins file is its own surface)."""
    if target in ("cpu", "apple_cpu", "apple_gpu"):
        # Host clang/cc + Apple frameworks are part of the Mac base toolchain.
        if shutil.which("c++") or shutil.which("clang++"):
            return GateResult(GATE_TOOLCHAIN, STATUS_PASS, "host C++ compiler present")
        return GateResult(GATE_TOOLCHAIN, STATUS_FAIL, "no host C++ compiler on PATH")
    if target == "nvidia":
        if shutil.which("nvcc"):
            return GateResult(GATE_TOOLCHAIN, STATUS_PASS, "nvcc present")
        return GateResult(GATE_TOOLCHAIN, STATUS_FAIL,
                          "nvcc not on PATH (CUDA Toolkit 13.2.1 not installed)")
    if target == "rocm":
        if shutil.which("hipcc"):
            return GateResult(GATE_TOOLCHAIN, STATUS_PASS, "hipcc present")
        return GateResult(GATE_TOOLCHAIN, STATUS_FAIL,
                          "hipcc not on PATH (ROCm 7.2.3 not installed)")
    if target == "metalium":
        # Metalium toolchain isn't a standard PATH binary; treat as not-evaluated.
        return GateResult(GATE_TOOLCHAIN, STATUS_NOT_EVALUATED,
                          "Metalium SDK probe is a separate surface")
    return GateResult(GATE_TOOLCHAIN, STATUS_FAIL, f"unknown target {target!r}")


def _eval_link(target: str, op_name: Optional[str]) -> GateResult:
    """A kernel artifact for this op can be linked into the runtime ABI
    artifact. fused / reference / compileable count as ``pass``; ``artifact_only``
    is the audit's "IR emits but no link path" state."""
    if op_name is None:
        return GateResult(GATE_LINK, STATUS_NOT_APPLICABLE)
    entries = _manifest_entries(op_name, target)
    if not entries:
        return GateResult(GATE_LINK, STATUS_FAIL,
                          "no manifest entry to link")
    statuses = {e.status for e in entries}
    # ``hardware_verified`` (Project 3, 2026-06-01) is the TOP rung of
    # the readiness ladder — strictly stronger than ``fused`` (it adds
    # a checked-in numerical-comparison proof). ``packaged`` (PK5) is
    # a parallel path: an ``.mtlpackage`` artifact loaded via
    # ``apple_mlpkg.compile_mlpackage`` ships an executable kernel.
    # All four count as linkable.
    if statuses & {"fused", "reference", "compileable",
                   "hardware_verified", "packaged"}:
        return GateResult(GATE_LINK, STATUS_PASS)
    if "artifact_only" in statuses:
        return GateResult(GATE_LINK, STATUS_FAIL,
                          "artifact_only — IR emits but no linked-kernel path today")
    return GateResult(GATE_LINK, STATUS_FAIL,
                      f"no linkable status (saw {sorted(statuses)})")


def _eval_runtime_abi(target: str, op_name: Optional[str]) -> GateResult:
    """The runtime ABI has a launcher for this target. Today: 4 targets in
    ``execution_matrix`` (cpu / apple_cpu / apple_gpu / cpu+jit_cpu_numpy).
    NVIDIA / ROCm / Metalium return UNIMPLEMENTED with a precise reason
    from G6 ("no native C-ABI launch bridge for target=...")."""
    if _execution_row(target) is None:
        return GateResult(GATE_RUNTIME_ABI, STATUS_FAIL,
                          f"no execution_matrix row for {target!r} "
                          f"(tsrLaunchKernel returns UNIMPLEMENTED for this target)")
    return GateResult(GATE_RUNTIME_ABI, STATUS_PASS)


def _eval_hardware_smoke(target: str, op_name: Optional[str]) -> GateResult:
    """A live tiny kernel can run on this host. Honest sniff:

    * cpu — always pass (every host has a CPU).
    * apple_cpu / apple_gpu — pass on Darwin; fail elsewhere.
    * nvidia / rocm — would need a GPU probe; without one, fail with a
      precise reason instead of pretending.
    * metalium — same.
    """
    if target == "cpu":
        return GateResult(GATE_HARDWARE_SMOKE, STATUS_PASS, "host CPU present")
    if target in ("apple_cpu", "apple_gpu"):
        if _platform_is_darwin_arm64():
            return GateResult(GATE_HARDWARE_SMOKE, STATUS_PASS,
                              "Darwin / Apple silicon host")
        return GateResult(GATE_HARDWARE_SMOKE, STATUS_FAIL,
                          "Apple silicon required for native execution")
    if target == "nvidia":
        # We do not run a CUDA driver probe here. The honest answer is
        # "not evaluable on this host" rather than fabricated pass/fail.
        return GateResult(GATE_HARDWARE_SMOKE, STATUS_NOT_EVALUATED,
                          "NVIDIA GPU not probed by this gate (CI lane gap)")
    if target == "rocm":
        return GateResult(GATE_HARDWARE_SMOKE, STATUS_NOT_EVALUATED,
                          "AMD GPU not probed by this gate (CI lane gap)")
    if target == "metalium":
        return GateResult(GATE_HARDWARE_SMOKE, STATUS_NOT_EVALUATED,
                          "Tenstorrent device not probed by this gate (CI lane gap)")
    return GateResult(GATE_HARDWARE_SMOKE, STATUS_FAIL,
                      f"unknown target {target!r}")


def _eval_numerical(target: str, op_name: Optional[str]) -> GateResult:
    """A numerical-comparison test exists. v1 leans on the capabilities
    op-level ``runtime_status``: ``ready`` / ``fused`` means there's a real
    runtime path which is verified by the unit suite (the apple_cpu /
    apple_gpu test files exercise these end-to-end with numpy comparisons)."""
    if op_name is None:
        return GateResult(GATE_NUMERICAL, STATUS_NOT_APPLICABLE)
    tc = _safe_get_capability(target if target != "nvidia"
                                       else "nvidia_sm90")
    if tc is None:
        return GateResult(GATE_NUMERICAL, STATUS_FAIL,
                          f"target {target!r} not in capabilities registry")
    # Resolve op name through the capabilities canonicalization (it accepts
    # both bare ``matmul`` and ``tessera.matmul``).
    canon = _cap.canonical_graph_op_name(op_name) or op_name
    op_cap = tc.supported_ops.get(canon)
    if op_cap is None:
        op_cap = tc.supported_ops.get(f"tessera.{op_name}")
    if op_cap is None:
        return GateResult(GATE_NUMERICAL, STATUS_FAIL,
                          f"no capabilities op-entry for {op_name!r} on {target!r}")
    if op_cap.runtime_status in ("ready", "fused"):
        return GateResult(GATE_NUMERICAL, STATUS_PASS,
                          f"runtime_status = {op_cap.runtime_status}")
    return GateResult(GATE_NUMERICAL, STATUS_FAIL,
                      f"runtime_status = {op_cap.runtime_status}")


_EVALUATORS = {
    GATE_LEGALITY: _eval_legality,
    GATE_CODEGEN: _eval_codegen,
    GATE_TOOLCHAIN: _eval_toolchain,
    GATE_LINK: _eval_link,
    GATE_RUNTIME_ABI: _eval_runtime_abi,
    GATE_HARDWARE_SMOKE: _eval_hardware_smoke,
    GATE_NUMERICAL: _eval_numerical,
}


# --- Public API ----------------------------------------------------------

def evaluate(target: str, op_name: Optional[str] = None) -> tuple[GateResult, ...]:
    """Evaluate all seven gates for ``(target, op_name)`` in canonical order.

    Returns a tuple of seven results, one per gate. Each gate is evaluated
    independently — short-circuiting is done by the *caller* via
    :func:`first_failing_gate`. The reason: the dashboard wants to show every
    gate's status, even after the first failure (so e.g. NVIDIA shows
    ``codegen=pass / toolchain=fail / runtime_abi=fail`` instead of just
    ``toolchain=fail / rest=not_evaluated``).

    Sub-target names like ``nvidia_sm90`` / ``rocm_gfx942`` are normalized to
    their family (``nvidia`` / ``rocm``) so callers can pass either form.
    """
    norm = _normalize_target(target)
    # Codegen / link / numerical inspect the per-arch manifest entries; pass
    # the *original* target to those so e.g. ``nvidia_sm90`` matches one
    # specific manifest row instead of the aggregated family.
    return tuple(
        _EVALUATORS[g](norm if g in _FAMILY_LEVEL_GATES else target, op_name)
        for g in GATE_ORDER
    )


_FAMILY_LEVEL_GATES = frozenset({
    GATE_TOOLCHAIN, GATE_HARDWARE_SMOKE, GATE_LEGALITY,
})


def first_failing_gate(
    target: str, op_name: Optional[str] = None
) -> Optional[GateResult]:
    """Return the first ``status == "fail"`` gate, or ``None`` if every gate
    is ``pass`` / ``not_evaluated`` / ``not_applicable``."""
    for result in evaluate(target, op_name):
        if result.status == STATUS_FAIL:
            return result
    return None


__all__ = [
    "GATE_ORDER",
    "GATE_LEGALITY", "GATE_CODEGEN", "GATE_TOOLCHAIN", "GATE_LINK",
    "GATE_RUNTIME_ABI", "GATE_HARDWARE_SMOKE", "GATE_NUMERICAL",
    "STATUS_PASS", "STATUS_FAIL", "STATUS_NOT_EVALUATED",
    "STATUS_NOT_APPLICABLE",
    "GateResult",
    "evaluate",
    "first_failing_gate",
]
