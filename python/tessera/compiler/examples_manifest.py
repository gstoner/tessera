"""Machine-readable manifest of every active ``examples/`` entry point.

This file is the single source of truth for the examples surface audit.
It powers two CI gates:

  * ``tessera.cli.examples_audit --check`` runs every entry whose status
    is ``runnable`` and asserts ``exit == 0``.  ``runnable_optional`` and
    ``compile_only`` entries are run when the declared ``extras_required``
    are importable; otherwise the row is reported as ``skipped`` (not a
    failure).
  * ``tessera.cli.claim_lint --check`` scans the README living in each
    example directory and flags overclaim language (``runnable``,
    ``end-to-end demo``, ``working``) on any row whose status is
    ``scaffold`` or ``broken``.

The 5-element status taxonomy is intentionally narrow:

==================  =========================================================
Status              Meaning
==================  =========================================================
runnable            Runs in the default venv on CPU-only CI.
runnable_optional   Runs *if* declared ``extras_required`` are importable.
compile_only        Emits IR / artifacts but does not execute a workload.
scaffold            Intentionally illustrative; not designed to run today.
broken              Expected to run but currently fails — followup needed.
==================  =========================================================

Per Decision #21 of CLAUDE.md, ``scaffold`` rows must carry a ``reason``
naming the missing piece (typically an upstream research stack like
``tessera.stdlib``) and the directory must ship a ``STATUS.md`` so a
reader at the directory level can see the truth without bouncing to
this manifest.

The manifest is also the input to ``docs/audit/generated/examples_status.md``;
that doc is regenerated and drift-gated against ``render_markdown()``.

This module never imports ``tessera`` at module load time so it stays
cheap to inspect from CI scripts and standalone test runners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tessera.compiler.surface_manifest import (
    ALLOWED_STATUSES,
    SurfaceEntry,
    audit_filesystem as _audit_filesystem_shared,
    render_markdown as _render_markdown_shared,
    status_counts as _status_counts_shared,
)

# Repo root, derived without importing tessera.
_REPO_ROOT = Path(__file__).resolve().parents[3]


# Backwards-compatible alias — pre-2026-05-19 callers referenced
# ``ExampleEntry`` directly; the dataclass moved to ``surface_manifest``
# so every surface (examples / benchmarks / research / tools) shares
# the same row shape.
ExampleEntry = SurfaceEntry


# Per-entry registry. Order is preserved in the generated doc and is
# chosen so the simplest examples render first.
_ENTRIES: tuple[ExampleEntry, ...] = (
    # ── Getting started ────────────────────────────────────────────────
    ExampleEntry(
        directory="examples/getting_started",
        entry_point="examples/getting_started/basic_tensor_ops.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python examples/getting_started/basic_tensor_ops.py"
        ),
        notes="Canonical @tessera.jit walkthrough — add, RMSNorm, SwiGLU on CPU.",
    ),
    ExampleEntry(
        directory="examples/getting_started",
        entry_point=(
            "examples/getting_started/compile_and_explain.py"
        ),
        status="runnable",
        command=(
            "PYTHONPATH=python python examples/getting_started/"
            "compile_and_explain.py"
        ),
        notes=(
            "Canonical compiler tour (P0-4, 2026-05-19). Walks through "
            "@tessera.jit → fn() → fn.explain() → ts.compiler.support() "
            "→ ts.from_text() in ~80 lines.  Linked from README.md."
        ),
    ),
    ExampleEntry(
        directory="examples/getting_started/tessera_flash_attention_demo",
        entry_point=(
            "examples/getting_started/tessera_flash_attention_demo/examples/"
            "flash_attention_demo.py"
        ),
        status="runnable_optional",
        command=(
            "PYTHONPATH=python:examples/getting_started/tessera_flash_attention_demo "
            "python examples/getting_started/tessera_flash_attention_demo/examples/"
            "flash_attention_demo.py --batch 1 --heads 2 --seq 32 --dim 16 "
            "--dtype f32 --device cpu --iters 1"
        ),
        extras_required=("torch",),
        notes=(
            "Runs canonical @tessera.jit flash attention on reference_cpu, "
            "checks it against PyTorch SDPA, and can emit real compiler IR."
        ),
    ),
    # ── Compiler tutorials ─────────────────────────────────────────────
    ExampleEntry(
        directory="examples/compiler/ir_pipeline_tutorial",
        entry_point=(
            "examples/compiler/ir_pipeline_tutorial/"
            "tessera_ir_pipeline_demo.py"
        ),
        status="runnable",
        command=(
            "PYTHONPATH=python python examples/compiler/ir_pipeline_tutorial/"
            "tessera_ir_pipeline_demo.py"
        ),
        notes="Prints Graph IR → Schedule IR → Tile IR → Target IR for a tiny MLP.",
    ),
    ExampleEntry(
        directory="examples/compiler/ir_pipeline_tutorial",
        entry_point=(
            "examples/compiler/ir_pipeline_tutorial/"
            "Tessera_IR_Pipeline_Tutorial.ipynb"
        ),
        status="runnable_optional",
        command=(
            "PYTHONPATH=python python -m jupyter nbconvert --to notebook "
            "--execute --stdout --ExecutePreprocessor.timeout=120 "
            "examples/compiler/ir_pipeline_tutorial/"
            "Tessera_IR_Pipeline_Tutorial.ipynb"
        ),
        extras_required=("jupyter", "nbconvert", "ipykernel"),
        notes=(
            "Notebook form of the compiler tutorial. Executes real CPU "
            "numerics and requires non-empty Graph/Schedule/Tile/Target IR; "
            "no shim or illustrative artifacts."
        ),
    ),
    ExampleEntry(
        directory="examples/compiler/dnas",
        entry_point="examples/compiler/dnas/dnas_schedule_autotune.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python examples/compiler/dnas/"
            "dnas_schedule_autotune.py"
        ),
        notes="Differentiable arch search + schedule search using tessera.arch.",
    ),
    # ── Conformance ────────────────────────────────────────────────────
    ExampleEntry(
        directory="examples/conformance",
        entry_point="examples/conformance/apple_path_ga_ebm_demos.py",
        status="runnable",
        command=(
            "python examples/conformance/apple_path_ga_ebm_demos.py"
        ),
        notes=(
            "Apple-path GA + EBM demos — self-bootstraps sys.path. "
            "Surfaces Apple GPU manifest status for GA kernels."
        ),
    ),
    # ── Advanced — runnable demos ──────────────────────────────────────
    ExampleEntry(
        directory="examples/advanced/kv_cache_serving",
        entry_point="examples/advanced/kv_cache_serving/demo.py",
        status="runnable",
        command=(
            "PYTHONPATH=python:examples/advanced/kv_cache_serving "
            "python examples/advanced/kv_cache_serving/demo.py"
        ),
        notes="Quantized KV-cache serving scheduler walkthrough.",
    ),
    ExampleEntry(
        directory="examples/advanced/long_context_attention",
        entry_point="examples/advanced/long_context_attention/demo.py",
        status="runnable",
        command=(
            "PYTHONPATH=python:examples/advanced/long_context_attention "
            "python examples/advanced/long_context_attention/demo.py"
        ),
        notes="Per-head role classification + cache-bytes estimator.",
    ),
    ExampleEntry(
        directory="examples/attention",
        entry_point="examples/attention/minimax_sparse_attention.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python examples/attention/minimax_sparse_attention.py"
        ),
        notes=(
            "MiniMax Sparse Attention (arXiv:2606.13392) — dense-GQA vs sparse-MSA "
            "comparison (dense-equivalence anchor + selected-block stats) and a "
            "long-context *theoretical* attention-compute reduction sweep. CPU "
            "reference; no wall-clock speedup claimed (native sparse kernel is future)."
        ),
    ),
    ExampleEntry(
        directory="examples/diffusion_guidance",
        entry_point="examples/diffusion_guidance/cgg_benchmark.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python examples/diffusion_guidance/cgg_benchmark.py --smoke"
        ),
        notes=(
            "Contrastive Gradient Guidance (CGG) benchmark harness — gamma grid + "
            "adapter strengths, aggregate JSON, fixed seed. `--smoke` runs a "
            "CI-friendly two-case sweep. CPU reference."
        ),
    ),
    ExampleEntry(
        directory="examples/diffusion_guidance",
        entry_point="examples/diffusion_guidance/cgg_diffusion_gemma.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python examples/diffusion_guidance/cgg_diffusion_gemma.py"
        ),
        notes=(
            "CGG over a DiffusionGemma-shaped denoiser — score-composition demo "
            "exercising the tessera.score_combine Graph IR bridge. CPU reference."
        ),
    ),
    ExampleEntry(
        directory="examples/advanced/gumiho",
        entry_point="examples/advanced/gumiho/demo.py",
        status="runnable",
        command=(
            "PYTHONPATH=python:examples/advanced/gumiho "
            "python examples/advanced/gumiho/demo.py"
        ),
        notes=(
            "Gumiho (ICML'25) hybrid speculative decoding — serial 2-layer "
            "Transformer head + 5 parallel MLP heads + Full Tree Attention, "
            "draft compute on the Apple GPU/CPU backend, verify/advance via "
            "tessera.speculative."
        ),
    ),
    ExampleEntry(
        directory="examples/advanced/rlvr_reasoning_suite",
        entry_point="examples/advanced/rlvr_reasoning_suite/run_demo.py",
        status="runnable",
        command=(
            "PYTHONPATH=python:examples/advanced/rlvr_reasoning_suite "
            "python examples/advanced/rlvr_reasoning_suite/run_demo.py "
            "--steps 1 --group-size 2 "
            "--log examples/advanced/rlvr_reasoning_suite/runs/rewards.jsonl"
        ),
        notes="GRPO accounting + JSONL telemetry; CPU-only toy policy.",
    ),
    # ── Advanced — compile_only smokes ─────────────────────────────────
    ExampleEntry(
        directory="examples/advanced/mla",
        entry_point="examples/advanced/mla/tests/smoke_random.py",
        status="runnable",
        command="python examples/advanced/mla/tests/smoke_random.py",
        notes=(
            "FlashMLA numpy reference + Graph IR build + Apple CPU compile "
            "(self-bootstraps sys.path)."
        ),
    ),
    ExampleEntry(
        directory="examples/advanced/Fast_dLLM_v2",
        entry_point="examples/advanced/Fast_dLLM_v2/tests/smoke_random.py",
        status="runnable",
        command="python examples/advanced/Fast_dLLM_v2/tests/smoke_random.py",
        notes=(
            "Fast dLLM v2 numpy reference + Graph IR build + Apple CPU "
            "compile (self-bootstraps sys.path)."
        ),
    ),
    ExampleEntry(
        directory="examples/advanced/Nemotron_Nano_12B_v2",
        entry_point=(
            "examples/advanced/Nemotron_Nano_12B_v2/tests/smoke_random.py"
        ),
        status="runnable",
        command=(
            "python examples/advanced/Nemotron_Nano_12B_v2/tests/smoke_random.py"
        ),
        notes=(
            "Nemotron-H hybrid pattern numpy reference + Graph IR build + "
            "Apple CPU compile (self-bootstraps sys.path)."
        ),
    ),
    # ── Advanced — maintained slices of larger research examples ──────
    ExampleEntry(
        directory="examples/advanced/Diffusion_LLM",
        entry_point="examples/advanced/Diffusion_LLM/gpu_denoise.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python "
            "examples/advanced/Diffusion_LLM/gpu_denoise.py"
        ),
        notes=(
            "Torch-free MDLM loop with deterministic sampling and a NumPy "
            "oracle; uses Metal kernels when visible and otherwise validates "
            "the portable path. The old removed-API monolith was retired."
        ),
    ),
    ExampleEntry(
        directory="examples/advanced/Jet_nemotron",
        entry_point=(
            "examples/advanced/Jet_nemotron/tests/smoke_linear_attention.py"
        ),
        status="runnable",
        command=(
            "python examples/advanced/Jet_nemotron/tests/"
            "smoke_linear_attention.py"
        ),
        notes=(
            "Canonical D=128 linear-attention numerical + compiler smoke. "
            "The full optional model still requires the external research "
            "stdlib and this host-portable run is not ROCm device evidence."
        ),
    ),
    ExampleEntry(
        directory="examples/advanced/Tessera_Empirical_Software_Agent",
        entry_point=(
            "examples/advanced/Tessera_Empirical_Software_Agent/examples/"
            "kernel_autotuning/benchmark_kernel.py"
        ),
        status="runnable",
        command=(
            "python examples/advanced/Tessera_Empirical_Software_Agent/"
            "examples/kernel_autotuning/benchmark_kernel.py"
        ),
        notes=(
            "Deterministic candidate-config benchmark with a numerical "
            "oracle. The parent LLM/tree-search orchestrator remains an "
            "integration scaffold."
        ),
    ),
    # ── Integration ────────────────────────────────────────────────────
    ExampleEntry(
        directory="examples/integration/HF_transformer",
        entry_point=(
            "examples/integration/HF_transformer/"
            "tessera_huggingface_transformers.py"
        ),
        status="runnable",
        command=(
            "python examples/integration/HF_transformer/"
            "tessera_huggingface_transformers.py"
        ),
        notes=(
            "HF-shaped BERT/GPT-2/Llama configs select canonical "
            "@tessera.jit attention graphs and validate against NumPy."
        ),
    ),
    # ── Optimization ──────────────────────────────────────────────────
    ExampleEntry(
        directory="examples/optimization",
        entry_point="examples/optimization/tessera_schedule_example.py",
        status="runnable",
        command=(
            "python examples/optimization/tessera_schedule_example.py"
        ),
        notes=(
            "Canonical scheduled matmul with a NumPy oracle and Graph, "
            "Schedule, Tile, and Target IR checks. Generic source sketches "
            "remain teaching material rather than backend evidence."
        ),
    ),
)


def all_entries() -> tuple[ExampleEntry, ...]:
    """Return the full registry, in declared order."""

    return _ENTRIES


def entries_by_status(status: str) -> tuple[ExampleEntry, ...]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(
            f"status={status!r} not in {ALLOWED_STATUSES!r}"
        )
    return tuple(e for e in _ENTRIES if e.status == status)


def status_counts() -> dict[str, int]:
    return _status_counts_shared(_ENTRIES)


def find_by_directory(directory: str) -> ExampleEntry | None:
    target = directory.rstrip("/")
    for e in _ENTRIES:
        if e.directory == target:
            return e
    return None


# ─────────────────────────────────────────────────────────────────────────
# Audit helpers — used by ``tessera.cli.examples_audit``.
# ─────────────────────────────────────────────────────────────────────────

def audit_filesystem(
    entries: Iterable[ExampleEntry] | None = None,
) -> list[str]:
    """Return a list of structural issues with the manifest.

    Catches stale rows (declared entry file missing on disk) without
    actually executing anything.  Empty list means clean.
    """

    rows = tuple(entries) if entries is not None else _ENTRIES
    return _audit_filesystem_shared(rows)


# ─────────────────────────────────────────────────────────────────────────
# Doc renderer.
# ─────────────────────────────────────────────────────────────────────────


_SURFACE_INTRO = (
    "This dashboard lists every active ``examples/`` entry point and "
    "its **executable status**.  It is regenerated from "
    "``python/tessera/compiler/examples_manifest.py``.\n\n"
    "CI guards (run as part of ``scripts/validate.sh``):\n\n"
    "* ``python -m tessera.cli.examples_audit --check`` — executes "
    "every ``runnable`` row and ``runnable_optional`` rows whose "
    "extras are available; ``scaffold`` / ``broken`` / ``archived`` "
    "rows are not executed.\n"
    "* ``python -m tessera.cli.claim_lint --check`` — scans each "
    "example README and flags overclaim language on ``scaffold`` / "
    "``broken`` / ``archived`` rows.\n\n"
    "``archive/examples/**`` is out of scope and not tracked here."
)


def render_markdown(entries: Iterable[ExampleEntry] | None = None) -> str:
    """Render the full ``examples_status.md`` from the manifest."""

    rows = tuple(entries) if entries is not None else _ENTRIES
    return _render_markdown_shared(
        surface_title="Tessera Examples — Status Audit",
        surface_intro=_SURFACE_INTRO,
        entries=rows,
        regenerate_command="python -m tessera.cli.examples_audit --render",
    )


__all__ = [
    "ALLOWED_STATUSES",
    "ExampleEntry",
    "all_entries",
    "audit_filesystem",
    "entries_by_status",
    "find_by_directory",
    "render_markdown",
    "status_counts",
]
