r"""B.2 — ``runtime.launch()`` must name the *first failing gate* in its
response when a target can't be served, not fall through to a generic
"unwired" message.

The unsupported response is structured. The audit's framing:

    "If a backend is artifact-only, the pipeline should say exactly which
     gate failed: legality, codegen, toolchain, link, runtime ABI,
     hardware smoke, or numerical validation."

Today on a developer Mac:

* ``nvidia`` artifact → first failing gate = ``toolchain`` (no nvcc).
* ``rocm`` artifact → first failing gate = ``toolchain`` (no hipcc).
* A CPU artifact whose op isn't in the native_cpu / jit_cpu_numpy
  capability set falls through with the CPU-level gate.

These tests pin the contract:

1. Response has a top-level ``first_failing_gate`` key (machine-readable).
2. ``reason`` leads with ``"unsupported: first failing gate \`<gate>\`"``.
3. ``telemetry.metadata`` carries the same gate name + detail.
4. ``ok=False`` and ``runtime_status`` stay honest (``unimplemented`` /
   ``missing_backend`` / ``unsupported``).
"""

from __future__ import annotations

import pytest

from tessera.runtime import RuntimeArtifact, launch


def _unsupported_artifact(target: str, op: str = "matmul") -> RuntimeArtifact:
    """Build a minimal RuntimeArtifact metadata that targets ``target`` with
    one ``op`` in its lowering plan. ``executable=False`` so the launch
    dispatcher falls straight into the unsupported path."""
    return RuntimeArtifact(
        graph_ir="",
        schedule_ir="",
        tile_ir="",
        target_ir="",
        metadata={
            "target": target,
            "compiler_path": "artifact_only",
            "execution_kind": "artifact_only",
            "executable": False,
            "kernel_id": f"{target}_{op}_test",
            "ops": [{"op_name": f"tessera.{op}"}],
        },
    )


# ---- audit-named cases ----

def test_nvidia_unsupported_launch_names_toolchain_gate():
    art = _unsupported_artifact("nvidia_sm90", "matmul")
    result = launch(art, args={})
    assert result["ok"] is False
    # The named gate appears as a structured key…
    assert result["first_failing_gate"] == "toolchain", result
    assert "nvcc" in result["first_failing_gate_detail"]
    # …and leads the reason string.
    assert result["reason"].startswith("unsupported: first failing gate `toolchain`"), result["reason"]
    # And it's carried in telemetry too.
    tel_meta = result["telemetry"]["metadata"]
    assert tel_meta["first_failing_gate"] == "toolchain"
    assert "nvcc" in tel_meta["first_failing_gate_detail"]


def test_rocm_unsupported_launch_names_toolchain_gate():
    # Canonical (toolchain-not-assumed) contract — deterministic so it holds on
    # both the Apple dev box and the Ubuntu ROCm 7.2.4 box (where hipcc is
    # live-present, so the live named blocker is instead `link`).
    from tessera.compiler import pipeline_gates as pg
    art = _unsupported_artifact("rocm", "matmul")
    with pg.deterministic_host_for_dashboard():
        result = launch(art, args={})
    assert result["ok"] is False
    assert result["first_failing_gate"] == "toolchain", result
    assert "hipcc" in result["first_failing_gate_detail"]
    assert result["reason"].startswith("unsupported: first failing gate `toolchain`")


# ---- shape guards ----

def test_unsupported_response_carries_structured_gate_fields():
    """Every unsupported response must carry the two structured keys —
    not just embed them in the reason string. Catches a regression where the
    gate machinery is removed but the reason text accidentally still mentions
    a gate (false-positive grep)."""
    art = _unsupported_artifact("nvidia_sm90", "matmul")
    result = launch(art, args={})
    assert "first_failing_gate" in result
    assert "first_failing_gate_detail" in result
    assert result["first_failing_gate"] is not None
    assert result["first_failing_gate_detail"]  # non-empty
    # Telemetry mirror must also be present.
    assert "first_failing_gate" in result["telemetry"]["metadata"]


def test_unsupported_status_stays_honest():
    """The structured runtime_status is still one of the unsupported variants
    — adding the gate name to the diagnostic shouldn't silently flip ok=True
    or change the status enum."""
    for target in ("nvidia_sm90", "rocm"):
        art = _unsupported_artifact(target, "matmul")
        result = launch(art, args={})
        assert result["ok"] is False
        assert result["runtime_status"] in ("unimplemented", "missing_backend"), (
            f"{target}: status {result['runtime_status']!r}")


def test_reason_string_points_at_conformance_dashboard():
    """The reason should cite the conformance dashboard so a reader can find
    the full per-cell breakdown for follow-up. Pinning this in the test
    catches accidental cross-link rot."""
    art = _unsupported_artifact("nvidia_sm90", "matmul")
    result = launch(art, args={})
    assert "op_target_conformance.md" in result["reason"]


def test_op_name_extracted_from_ops_metadata():
    """The gate evaluator is op-specific when ``metadata["ops"]`` carries an
    op_name. Use a softmax artifact to prove the op flows through; softmax
    has different downstream consequences (e.g. capabilities runtime_status
    is `ready`/`fused` on apple_gpu) than matmul, but on NVIDIA both fail at
    toolchain — so the gate name should be stable across these ops while the
    detail string remains anchored to nvcc."""
    art = _unsupported_artifact("nvidia_sm90", "softmax")
    result = launch(art, args={})
    assert result["first_failing_gate"] == "toolchain"
    assert "nvcc" in result["first_failing_gate_detail"]
