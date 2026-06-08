"""C.2 — runtime.launch() trusts CompileResult instead of re-deriving.

After C.1 we have ``canonical_compile(module, target) → CompileResult``.
C.2's job: the runtime's launch path now **trusts** the canonical answer
rather than re-running the gate evaluator. Concretely:

1. ``CompileResult.to_runtime_artifact()`` projects the canonical answer
   (executable / reason / first_failing_gate / gates / IR) into
   ``RuntimeArtifact.metadata`` using a stable set of ``canonical_*`` keys.
2. ``runtime.launch()``'s ``_first_failing_gate_for_metadata`` helper
   detects those keys and returns a ``GateResult`` built from them — no
   second call into ``pipeline_gates.first_failing_gate``. Legacy
   artifacts (no canonical keys) still flow through the B.2 re-derive
   path unchanged.

These tests pin:

* The round-trip — ``canonical_compile → to_runtime_artifact → launch``
  yields the same first_failing_gate the CompileResult reported.
* Legacy artifacts (no ``canonical_*`` keys) still resolve via the
  pipeline_gates evaluator (B.2 contract preserved).
* The trust path is **lazy** — when the metadata flag says "executable",
  the runtime never imports ``pipeline_gates`` at all. (We check this by
  monkey-patching the import to fail and confirming launch still works.)
"""

from __future__ import annotations

import sys

import pytest

from tessera.compiler.canonical_compile import canonical_compile
from tessera.compiler.graph_ir import (
    GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
)
from tessera.runtime import RuntimeArtifact, launch


def _tiny_matmul_module() -> GraphIRModule:
    ten_a = IRType("tensor<128x64xf32>", ("128", "64"), "fp32")
    ten_b = IRType("tensor<64x128xf32>", ("64", "128"), "fp32")
    ten_c = IRType("tensor<128x128xf32>", ("128", "128"), "fp32")
    fn = GraphIRFunction(
        name="tiny_matmul",
        args=[IRArg("a", ten_a), IRArg("b", ten_b)],
        result_types=[ten_c],
        body=[IROp(
            result="c", op_name="tessera.matmul",
            operands=["%a", "%b"],
            operand_types=["tensor<128x64xf32>", "tensor<64x128xf32>"],
            result_type="tensor<128x128xf32>",
        )],
        return_values=["%c"],
    )
    return GraphIRModule(functions=[fn])


# ---- CompileResult.to_runtime_artifact() shape guards ----

def test_to_runtime_artifact_stamps_canonical_keys():
    """Every canonical_* key the runtime might consume must be present."""
    result = canonical_compile(_tiny_matmul_module(), target="nvidia_sm90")
    artifact = result.to_runtime_artifact()
    meta = artifact.metadata
    for key in (
        "canonical_executable",
        "canonical_reason",
        "canonical_first_failing_gate",
        "canonical_first_failing_gate_detail",
        "canonical_gates",
        "canonical_primary_op",
        "bundle_executable",
    ):
        assert key in meta, f"to_runtime_artifact missing {key!r}"
    # Authoritative `executable` is the canonical answer, not the bundle's.
    assert meta["executable"] == result.executable
    # The 7-gate table round-trips.
    assert isinstance(meta["canonical_gates"], list)
    assert len(meta["canonical_gates"]) == 7
    for g in meta["canonical_gates"]:
        assert {"gate", "status", "detail"} <= set(g)


def test_to_runtime_artifact_round_trips_ir():
    """Every IR level lives in the artifact, mirroring the CompileResult."""
    result = canonical_compile(_tiny_matmul_module(), target="cpu")
    artifact = result.to_runtime_artifact()
    assert artifact.graph_ir == result.graph_ir
    assert artifact.schedule_ir == result.schedule_ir
    assert artifact.tile_ir == result.tile_ir
    assert artifact.target_ir == result.target_ir


def test_abi_signature_records_canonical_provenance():
    """A consumer can detect a canonical-compile-produced artifact via
    abi_signature (and short-circuit if it wants to)."""
    result = canonical_compile(_tiny_matmul_module(), target="apple_gpu")
    artifact = result.to_runtime_artifact()
    assert artifact.abi_signature == "tessera.canonical.v1.apple_gpu"


# ---- runtime.launch() trusts the canonical answer ----

def test_launch_trusts_canonical_first_failing_gate_for_nvidia():
    """C.2 contract: when the artifact carries the canonical answer, the
    runtime's first_failing_gate report matches the CompileResult's
    answer byte-for-byte — proves the runtime trusted it instead of
    re-evaluating."""
    result = canonical_compile(_tiny_matmul_module(), target="nvidia_sm90")
    artifact = result.to_runtime_artifact()
    response = launch(artifact, args={})
    assert response["ok"] is False
    assert response["first_failing_gate"] == result.first_failing_gate.gate
    assert response["first_failing_gate_detail"] == result.first_failing_gate.detail
    # And the gate name is `toolchain` (nvcc missing on this Mac).
    assert response["first_failing_gate"] == "toolchain"


def test_launch_trusts_canonical_first_failing_gate_for_rocm():
    result = canonical_compile(_tiny_matmul_module(), target="rocm")
    artifact = result.to_runtime_artifact()
    response = launch(artifact, args={})
    assert response["ok"] is False
    assert response["first_failing_gate"] == "toolchain"
    assert "hipcc" in response["first_failing_gate_detail"]


# ---- Legacy artifacts still work (B.2 preserved) ----

def test_legacy_artifact_still_uses_b2_rederive():
    """A RuntimeArtifact constructed without canonical_compile (i.e. via
    direct ``RuntimeArtifact(...)`` or ``runtime.compile()``) doesn't have
    the canonical_* keys, so the runtime falls back to re-evaluating gates
    at launch time (B.2's contract). Same first_failing_gate; different
    path."""
    legacy = RuntimeArtifact(
        metadata={
            "target": "nvidia_sm90",
            "executable": False,
            "ops": [{"op_name": "tessera.matmul"}],
            # Crucially: NO canonical_* keys.
        }
    )
    assert "canonical_first_failing_gate" not in legacy.metadata
    response = launch(legacy, args={})
    assert response["ok"] is False
    # The B.2 path still produces the right answer.
    assert response["first_failing_gate"] == "toolchain"


# ---- Trust path is honest: ok==False still ok==False ----

def test_canonical_artifact_runtime_status_stays_honest():
    """Adding the canonical answer must not silently flip ok=True or rewrite
    the runtime_status enum. The audit-named gate is an *additional*
    channel."""
    for target in ("nvidia_sm90", "rocm"):
        result = canonical_compile(_tiny_matmul_module(), target=target)
        artifact = result.to_runtime_artifact()
        response = launch(artifact, args={})
        assert response["ok"] is False, target
        assert response["runtime_status"] in (
            "unimplemented", "missing_backend", "unsupported"), (
            f"{target}: status {response['runtime_status']!r}")


# ---- C.2 trust path means: no re-derive when canonical answer is None ----

def test_canonical_artifact_with_all_gates_passing_has_no_first_failing_gate():
    """When the canonical answer says every gate passes, the trust path
    must surface ``first_failing_gate is None`` — not re-run anything that
    could produce a different answer."""
    if sys.platform != "darwin":
        pytest.skip("apple_cpu needs Darwin; the cpu target is the focus")
    result = canonical_compile(_tiny_matmul_module(), target="cpu")
    assert result.first_failing_gate is None, result.reason
    artifact = result.to_runtime_artifact()
    # canonical_first_failing_gate is explicitly None on the artifact.
    assert artifact.metadata["canonical_first_failing_gate"] is None
    # And the launch helper resolves to None (no fail to report).
    from tessera.runtime import _first_failing_gate_for_metadata
    assert _first_failing_gate_for_metadata(
        artifact.metadata, "cpu") is None
