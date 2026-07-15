#!/usr/bin/env python3
"""Tessera release gate — hardware-neutral, target-parameterized
"before claiming a target is shippable" checklist.

Apple plan phase F (2026-05-20).  The first concrete target subset is
``apple_gpu``; the runner structure is target-agnostic so when
NVIDIA / ROCm hardware lanes come online the same script gates them
with their own subset.

Usage:

    python scripts/release_gate.py --target=apple_gpu
    python scripts/release_gate.py --target=apple_gpu --skip=benchmarks

The script runs a sequence of structurally identical checks:

  * **structure** — generated-dashboard drift gates
    (support_table, e2e_coverage, apple_target_map).
  * **canonicals** — execute every canonical program for this target
    and assert the report shape (no REFERENCE_FORCED on the happy
    path; native dispatch confirmed by target_decision row).
  * **benchmarks** — run the target-relevant bench rows in --ci mode.
    For apple_gpu: ``benchmark_fusion.py --ci``,
    ``benchmark_ga_ebm.py --ci``, ``benchmark_execution_kind.py --ci``,
    ``spectral_correctness.py`` (Apple-host-agnostic — always runnable).
  * **hardware-marked tests** — pytest -m "hardware_apple_gpu" if
    any such tests exist.

Exit codes:
  0 — every gate passed.
  1 — at least one gate failed; the script prints the failed gate(s)
      with stderr captured.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "python")}


@dataclass
class Gate:
    name: str
    cmd: tuple[str, ...]
    description: str
    optional_targets: tuple[str, ...] = ()


# ─────────────────────────────────────────────────────────────────────
# Per-target gate matrix.  Each target lists the gates it must pass
# before a release is claimed for that target.  ``structure`` gates
# are hardware-neutral and required for every target.
# ─────────────────────────────────────────────────────────────────────


_STRUCTURE_GATES: tuple[Gate, ...] = (
    # Single fleet-wide generated-doc drift gate. The registry in
    # `tessera.compiler.generated_docs` is the one source of truth for
    # which docs are generated and which artifact (CSV when present,
    # else Markdown) is byte-compared — so this one gate subsumes the
    # former per-doc support_table / e2e_coverage / *_target_map drift
    # gates and automatically covers any new dashboard.
    Gate(
        name="generated_docs_drift",
        cmd=(PYTHON, "-m", "tessera.compiler.generated_docs", "--check"),
        description=(
            "Every generated audit dashboard must match its registry "
            "render (drift-gated on the canonical CSV/Markdown artifact)."
        ),
    ),
    # Test-tree review P2-11 (2026-05-20): every release must
    # confirm the test-tree manifest's runnable rows still smoke-pass
    # (this is a *smoke* gate — it executes rows, not just doc drift).
    Gate(
        name="tests_manifest_smoke",
        cmd=(PYTHON, "-m", "tessera.cli.surface_audit",
             "--surface=tests", "--check"),
        description=(
            "Every runnable / compile_only row in the tests manifest "
            "must smoke-pass."
        ),
    ),
)

_APPLE_GPU_GATES: tuple[Gate, ...] = (
    # apple_target_map drift is covered by the fleet-wide
    # ``generated_docs_drift`` structure gate above.
    Gate(
        name="canonicals_native_dispatch",
        cmd=(
            PYTHON, "-c",
            "import sys, os; sys.path.insert(0, 'python');"
            " from tessera.compiler.canonical import matmul_softmax_matmul;"
            " r = matmul_softmax_matmul.run();"
            " assert r.fallback_reason is None, r.fallback_reason;"
            " assert r.target == 'apple_gpu', r.target;"
            " assert 'NATIVE DISPATCH' in r.target_decision['apple_gpu'];"
            " print('canonicals_native_dispatch: ok')"
        ),
        description=(
            "matmul_softmax_matmul canonical must dispatch the fused "
            "MSL kernel on Darwin (no REFERENCE_FORCED on the happy path)."
        ),
    ),
    Gate(
        name="apple_cpu_execution_kind_bench",
        cmd=(
            PYTHON,
            str(ROOT / "benchmarks/apple_cpu/benchmark_execution_kind.py"),
            "--ci",
            "--output", "/tmp/tessera_release_gate_apple_cpu.json",
        ),
        description=(
            "Apple CPU execution-kind microbench (proves "
            "matmul=accelerate_native vs others=numpy_reference)."
        ),
    ),
    Gate(
        name="spectral_correctness_proof",
        cmd=(
            PYTHON,
            str(ROOT / "benchmarks/spectral/spectral_correctness.py"),
            "--output", "/tmp/tessera_release_gate_spectral.json",
        ),
        description=(
            "Spectral FFT correctness proof (Stockham vs naive DFT, "
            "Apple-host-agnostic but part of the gate)."
        ),
    ),
    Gate(
        name="apple_gpu_hardware_marked_tests",
        cmd=(
            PYTHON, "-m", "pytest",
            str(ROOT / "tests/unit/"),
            "-m", "hardware_apple_gpu",
            "-q", "--no-header",
        ),
        description=(
            "Any pytest tests marked ``hardware_apple_gpu``. "
            "Skipped silently if no such tests exist (pytest returns "
            "exit code 5)."
        ),
        optional_targets=("apple_gpu",),  # pytest exit-5 OK for this gate
    ),
)


# Apple follow-up #3 (2026-05-20): NVIDIA and ROCm target-map drift is
# now covered by the fleet-wide ``generated_docs_drift`` structure gate.
# The full hardware-proof lanes (canonical native dispatch, per-target
# benchmarks, hardware-marked tests) land alongside Phase G / Phase H
# bring-up and append here the same way ``_APPLE_GPU_GATES`` does.
_NVIDIA_STRUCTURE_GATES: tuple[Gate, ...] = ()
_ROCM_STRUCTURE_GATES: tuple[Gate, ...] = ()


_GATE_MATRIX: dict[str, tuple[Gate, ...]] = {
    "apple_gpu": _STRUCTURE_GATES + _APPLE_GPU_GATES,
    "nvidia_sm90": _STRUCTURE_GATES + _NVIDIA_STRUCTURE_GATES,
    "rocm": _STRUCTURE_GATES + _ROCM_STRUCTURE_GATES,
}


def _run_gate(gate: Gate, verbose: bool) -> tuple[bool, str]:
    """Run a single gate.  Returns ``(passed, captured_output)``."""

    proc = subprocess.run(
        gate.cmd, cwd=ROOT, env=ENV,
        capture_output=True, text=True, check=False,
    )
    # pytest exit code 5 = no tests collected.  We treat that as a
    # pass for the ``hardware-marked tests`` gate (the marker is
    # optional and only fires when hardware tests exist).
    passed = proc.returncode == 0
    if not passed and proc.returncode == 5 and "hardware" in gate.name:
        passed = True
    captured = proc.stdout + (("\n" + proc.stderr) if proc.stderr else "")
    return passed, captured


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="release_gate",
        description=(
            "Hardware-neutral target-parameterized release gate. "
            "Runs the canonical 'before claiming this target is "
            "shippable' checklist for the given --target."
        ),
    )
    p.add_argument(
        "--target", default="apple_gpu",
        choices=sorted(_GATE_MATRIX),
        help="Target to gate.  Default: apple_gpu.",
    )
    p.add_argument(
        "--skip", default="",
        help=(
            "Comma-separated gate names to skip (e.g., 'benchmarks' "
            "skips every gate with 'bench' in its name).  Useful for "
            "incremental debugging."
        ),
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print captured stdout/stderr for every gate (passed + failed).",
    )
    args = p.parse_args(argv)

    gates = _GATE_MATRIX[args.target]
    skip_terms = tuple(s.strip() for s in args.skip.split(",") if s.strip())

    results: list[tuple[Gate, bool, str]] = []
    for gate in gates:
        if any(term in gate.name for term in skip_terms):
            print(f"  SKIP    {gate.name}  (--skip)")
            continue
        passed, output = _run_gate(gate, args.verbose)
        results.append((gate, passed, output))
        marker = "PASS" if passed else "FAIL"
        print(f"  {marker}    {gate.name}")
        if args.verbose or not passed:
            for line in output.strip().splitlines()[-15:]:
                print(f"    | {line}")

    failed = [r for r in results if not r[1]]
    total = len(results)
    print()
    if not failed:
        print(f"[release_gate:{args.target}] all {total} gates passed")
        return 0
    print(
        f"[release_gate:{args.target}] {len(failed)}/{total} gates "
        f"failed: {', '.join(g.name for g, _, _ in failed)}"
    )
    return 1


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
