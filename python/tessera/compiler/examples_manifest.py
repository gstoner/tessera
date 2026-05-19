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

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Repo root, derived without importing tessera (manifest must stay
# import-cheap for CLI tools that run before tessera is installed).
_REPO_ROOT = Path(__file__).resolve().parents[3]


ALLOWED_STATUSES = (
    "runnable",
    "runnable_optional",
    "compile_only",
    "scaffold",
    "broken",
)


@dataclass(frozen=True)
class ExampleEntry:
    """One row of the examples surface audit.

    Attributes
    ----------
    directory:
        Path to the example directory (relative to repo root).  This is
        the unit the audit tracks — most directories have one canonical
        entry point but the audit row is per-directory.
    entry_point:
        Path to the executable file (relative to repo root).  For
        ``compile_only`` rows this may be a module file that's loaded
        via ``python -c`` rather than executed directly.
    status:
        One of :data:`ALLOWED_STATUSES`.
    command:
        Exact shell command to run the entry from the repo root.  Only
        populated for rows where the audit actually executes something
        (``runnable``, ``runnable_optional``, ``compile_only``).
        ``None`` for ``scaffold``/``broken`` rows.
    extras_required:
        Optional list of importable module names that gate the run.
        For ``runnable_optional`` rows the audit imports each and skips
        the row (not a failure) if any are missing.
    reason:
        Human-readable explanation of why the row is ``scaffold`` or
        ``broken``.  Required for those two statuses; empty otherwise.
    notes:
        Free-text supplementary notes.  Surfaced in the generated doc.
    """

    directory: str
    entry_point: str
    status: str
    command: str | None = None
    extras_required: tuple[str, ...] = ()
    reason: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(
                f"ExampleEntry({self.directory!r}): status={self.status!r} "
                f"is not one of {ALLOWED_STATUSES!r}."
            )
        if self.status in ("runnable", "runnable_optional", "compile_only"):
            if not self.command:
                raise ValueError(
                    f"ExampleEntry({self.directory!r}): status={self.status!r} "
                    f"requires a 'command' field."
                )
        if self.status in ("scaffold", "broken") and not self.reason:
            raise ValueError(
                f"ExampleEntry({self.directory!r}): status={self.status!r} "
                f"requires a non-empty 'reason'."
            )
        if self.status == "runnable_optional" and not self.extras_required:
            raise ValueError(
                f"ExampleEntry({self.directory!r}): status='runnable_optional' "
                f"requires at least one entry in 'extras_required'."
            )

    @property
    def directory_path(self) -> Path:
        return _REPO_ROOT / self.directory

    @property
    def entry_point_path(self) -> Path:
        return _REPO_ROOT / self.entry_point

    def resolve_extras_available(self) -> bool:
        """Return True iff every ``extras_required`` module is importable."""

        import importlib.util

        for mod in self.extras_required:
            if importlib.util.find_spec(mod) is None:
                return False
        return True


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
        directory="examples/getting_started/tessera_flash_attention_demo",
        entry_point=(
            "examples/getting_started/tessera_flash_attention_demo/examples/"
            "flash_attention_demo.py"
        ),
        status="runnable_optional",
        command=(
            "PYTHONPATH=python:examples/getting_started/tessera_flash_attention_demo "
            "python examples/getting_started/tessera_flash_attention_demo/examples/"
            "flash_attention_demo.py"
        ),
        extras_required=("torch",),
        notes="Cross-checks Tessera vs. PyTorch reference attention.",
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
        directory="examples/advanced/speculative_decoding",
        entry_point="examples/advanced/speculative_decoding/demo.py",
        status="runnable",
        command=(
            "PYTHONPATH=python:examples/advanced/speculative_decoding "
            "python examples/advanced/speculative_decoding/demo.py"
        ),
        notes="Tree-shaped speculative-decoding scheduling demo.",
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
    # ── Advanced — scaffolds ───────────────────────────────────────────
    ExampleEntry(
        directory="examples/advanced/Diffusion_LLM",
        entry_point=(
            "examples/advanced/Diffusion_LLM/tessera_diffusion_llm.py"
        ),
        status="scaffold",
        reason=(
            "Research sketch — references non-existent APIs "
            "(``ts.compile(mode='training')``, ``ts.randint``, "
            "``Tensor[]`` syntax) and the package modules require "
            "PyTorch.  Reimplement against the canonical Tessera "
            "surface or mark broken when that work starts."
        ),
        notes=(
            "Submodule ``tessera_diffusion_llm/`` is a separate package "
            "that imports torch; see STATUS.md for the path forward."
        ),
    ),
    ExampleEntry(
        directory="examples/advanced/Jet_nemotron",
        entry_point=(
            "examples/advanced/Jet_nemotron/examples/e2e_infer.py"
        ),
        status="scaffold",
        reason=(
            "Requires the upstream ``tessera.stdlib`` research stack "
            "which is not part of the standalone compiler surface.  "
            "Test ``tests/test_sanity.py`` locks the e2e_infer import "
            "block + skips honestly when stdlib is absent."
        ),
        notes=(
            "Post-2026-05-19 fix: removed bogus "
            "``tessera_jetnemotron`` ghost-package import; example "
            "now bootstraps sys.path against sibling modules."
        ),
    ),
    ExampleEntry(
        directory="examples/advanced/power_retention",
        entry_point=(
            "examples/advanced/power_retention/examples/"
            "minimal_power_attn.py"
        ),
        status="scaffold",
        reason=(
            "Placeholder — entry-point script currently just prints "
            "``'example'``.  Real implementation lives in the "
            "``python/tessera_power/`` subpackage (CUDA scaffolds, "
            "Retention op) which is not wired into the audit yet."
        ),
        notes="See README.md for the v0.2 update plan.",
    ),
    ExampleEntry(
        directory="examples/advanced/Tessera_Empirical_Software_Agent",
        entry_point=(
            "examples/advanced/Tessera_Empirical_Software_Agent/src/"
            "agents/tree_search_runner.py"
        ),
        status="scaffold",
        reason=(
            "End-to-end LLM + tree-search agent — requires a real "
            "LLM client, sandbox executor, and per-task harness.  "
            "DummyLLM only proposes ``print('hello from variant N')`` "
            "stubs; the orchestrator is not runnable as a CI smoke "
            "test."
        ),
        notes=(
            "Kernel-autotuning sub-example "
            "(``examples/kernel_autotuning/``) is closer to runnable "
            "and could graduate independently."
        ),
    ),
    # ── Integration ────────────────────────────────────────────────────
    ExampleEntry(
        directory="examples/integration/HF_transformer",
        entry_point=(
            "examples/integration/HF_transformer/"
            "tessera_huggingface_transformers.py"
        ),
        status="scaffold",
        reason=(
            "References non-existent Tessera APIs "
            "(``from tessera import function, Module``); needs a "
            "rewrite against the canonical surface "
            "(``@tessera.jit`` + ``tessera.nn.Module``)."
        ),
        notes="Hugging Face Transformers compatibility sketch.",
    ),
    # ── Optimization placeholder ───────────────────────────────────────
    ExampleEntry(
        directory="examples/optimization",
        entry_point="examples/optimization/README.md",
        status="scaffold",
        reason=(
            "Top-level placeholder directory with only README.md and "
            "src/ stubs — no entry-point script exists yet."
        ),
        notes="Slated for autotune + roofline tooling examples.",
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
    out = dict.fromkeys(ALLOWED_STATUSES, 0)
    for e in _ENTRIES:
        out[e.status] = out[e.status] + 1
    return out


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

    issues: list[str] = []
    rows = tuple(entries) if entries is not None else _ENTRIES
    for entry in rows:
        if not entry.directory_path.is_dir():
            issues.append(
                f"{entry.directory}: directory does not exist on disk"
            )
            continue
        if not entry.entry_point_path.exists():
            issues.append(
                f"{entry.directory}: entry_point {entry.entry_point!r} "
                f"does not exist on disk"
            )
        if entry.status in ("scaffold", "broken"):
            status_md = entry.directory_path / "STATUS.md"
            if not status_md.exists():
                issues.append(
                    f"{entry.directory}: status={entry.status!r} "
                    f"requires a STATUS.md but none was found"
                )
    return issues


# ─────────────────────────────────────────────────────────────────────────
# Doc renderer.
# ─────────────────────────────────────────────────────────────────────────

_DOC_HEADER = """\
<!-- AUTO-GENERATED by python/tessera/compiler/examples_manifest.py. DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.examples_audit --render -->

# Tessera Examples — Status Audit

This dashboard lists every active ``examples/`` entry point and its
**executable status**.  It is regenerated from
``python/tessera/compiler/examples_manifest.py``.

The 5-element status taxonomy:

| Status              | Meaning                                                       |
|---------------------|---------------------------------------------------------------|
| ``runnable``          | Runs on default venv + CPU-only CI (must pass ``--check``).     |
| ``runnable_optional`` | Runs when declared ``extras_required`` are importable.          |
| ``compile_only``      | Emits IR/artifacts but does not execute the workload.         |
| ``scaffold``          | Intentionally illustrative; not runnable today.               |
| ``broken``            | Expected to run, currently fails — followup needed.           |

CI guards (run as part of ``scripts/validate.sh``):

* ``python -m tessera.cli.examples_audit --check`` — executes every
  ``runnable`` row and ``runnable_optional`` rows whose extras are
  available; ``scaffold`` rows are not executed.
* ``python -m tessera.cli.claim_lint --check`` — scans each example
  README and flags overclaim language (``runnable``, ``working``,
  ``end-to-end``) on ``scaffold``/``broken`` rows.

``examples/archive/**`` is out of scope and not tracked here.
"""


def render_markdown(entries: Iterable[ExampleEntry] | None = None) -> str:
    """Render the full ``examples_status.md`` from the manifest."""

    rows = tuple(entries) if entries is not None else _ENTRIES
    counts = status_counts()
    lines: list[str] = [_DOC_HEADER, ""]
    lines.append("## Counts")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|------:|")
    for status in ALLOWED_STATUSES:
        lines.append(f"| ``{status}`` | {counts[status]} |")
    lines.append(f"| **total** | **{len(rows)}** |")
    lines.append("")
    lines.append("## Entries")
    lines.append("")
    lines.append("| Directory | Status | Entry point | Command / Reason |")
    lines.append("|-----------|--------|-------------|------------------|")
    for entry in rows:
        cell: str
        if entry.status in ("scaffold", "broken"):
            cell = entry.reason
        elif entry.status == "runnable_optional":
            extras = ", ".join(f"``{m}``" for m in entry.extras_required)
            cell = f"``{entry.command}``<br/>extras: {extras}"
        else:
            cell = f"``{entry.command}``"
        lines.append(
            f"| ``{entry.directory}`` | ``{entry.status}`` | "
            f"``{entry.entry_point}`` | {cell} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


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
