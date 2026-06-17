"""Phase B — the Apple GPU "numpy lane" displacement worklist.

A deterministic, host-free classifier that answers *which catalog ops have no
Apple GPU lane today and therefore execute on the numpy reference interpreter*.
It reads the authoritative lane table (``apple_gpu_envelope.runtime_ops`` /
``lane_for``) and the op universe + categories (``op_catalog.OP_SPECS``), so it
runs anywhere (no Darwin, no Metal) and never touches the live dispatch path.

This turns the vague "~87% via numpy" into a ranked worklist grouped by the op's
lowering category, and it surfaces the key empirical fact that guided Phase C:
**every op in the pointwise-DAG fusion vocabulary already has a single-op GPU
lane** — so the displacement win is multi-op *DAG fusion* (fewer dispatches) plus
the genuinely-uncovered non-elementwise tail, not single-op elementwise.

Run as a script for the human-readable report::

    PYTHONPATH=python python -m tessera.compiler.apple_gpu_coverage
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from tessera.compiler import apple_gpu_envelope as _env
from tessera.compiler import fusion as _fusion
from tessera.compiler import op_catalog as _oc


def _graph_name(public_name: str) -> str:
    """Graph-IR op name used for lane lookup (falls back to ``tessera.<name>``)."""
    return _oc.graph_name_for(public_name) or f"tessera.{public_name}"


def has_gpu_lane(public_name: str) -> bool:
    """True iff the op has any Apple GPU dispatch lane today."""
    covered = _env.runtime_ops()
    return _graph_name(public_name) in covered or f"tessera.{public_name}" in covered


# Phase 2 displacement disposition by lowering category (2026-06-17). The naive
# "displace the 124 numpy-only ops" framing is mostly wrong — an investigation
# (matmul→transpose→gelu demotes to artifact_only; optim.adam runs host-side on
# pytrees, never @jit'd) showed most numpy-only ops are NOT a real GPU execution
# gap. Each category is one of:
#   real_gap_structural — layout/indexing ops that, mid-program, demote an
#       otherwise-GPU program off metal_runtime. The genuine Phase-2 target;
#       closing it needs real MPSGraph kernels for data-movers (transpose/concat/
#       slice/gather) or a residency-neutral chain gate for true-view ops
#       (reshape/squeeze/...). The actionable displacement worklist.
#   host_utility — functional optimizer steps; operate on host pytrees of numpy
#       (training-loop utilities, never emitted as a single @jit graph op). No GPU
#       gap — the heavy forward/backward already runs on GPU; the step is tiny.
#   distributed — collectives; multi-rank, run via the collective adapters, not a
#       single-device GPU concern.
#   hard_kernel — genuinely need a dedicated kernel: quantize (packed FP4/6/8 bit
#       layout), sparse (spmm/sddmm), spectral (distributed FFT), stencil, linalg
#       decomposition/solver, complex-number elementwise (2-channel).
_CATEGORY_DISPOSITION: dict[str, str] = {
    "layout_transform": "real_gap_structural",
    "indexing": "real_gap_structural",
    "functional_optimizer_step": "host_utility",
    "collective": "distributed",
    "quantize": "hard_kernel",
    "sparse": "hard_kernel",
    "spectral": "hard_kernel",
    "stencil": "hard_kernel",
    "linalg_decomposition": "hard_kernel",
    "linalg_solver": "hard_kernel",
    "elementwise": "hard_kernel",   # the numpy-only tail is complex-number ops
    "loop_nest": "hard_kernel",     # dequant-GEMM / latent-KV: fused kernels
    "state_update": "real_gap_structural",      # KV-cache append/prune/read on GPU
    "random_mask": "real_gap_structural",       # dropout
    "position_encoding": "real_gap_structural", # alibi / ntk_rope (rowop-shaped)
    "moe_transport": "distributed",             # all-to-all dispatch/combine
    "random_source": "host_utility",            # rng_uniform/normal (host RNG)
    "sort": "hard_kernel",
    "contraction": "hard_kernel",               # einsum
    "segment_reduce": "hard_kernel",
    "stable_reduction": "hard_kernel",
    "ebm": "hard_kernel",
    # Left unclassified on purpose (need per-op judgment, not a category default):
    #   rl_loss (ppo/grpo/cispo — may decompose), attention (specific variants),
    #   fused_epilogue (a fusion artifact). The unclassified bucket honestly flags
    #   "decide per-op" rather than risk a wrong category claim.
}


def disposition_for(category: str) -> str:
    return _CATEGORY_DISPOSITION.get(category, "unclassified")


@dataclass(frozen=True)
class CoverageReport:
    """Apple GPU lane coverage over a set of ops."""

    total: int
    covered: int
    numpy_only: tuple[str, ...]
    by_category: dict[str, tuple[str, ...]] = field(default_factory=dict)
    pointwise_vocab_covered: bool = True

    @property
    def numpy_only_count(self) -> int:
        return len(self.numpy_only)


def numpy_lane_worklist(op_names: Iterable[str] | None = None) -> CoverageReport:
    """Classify ``op_names`` (default: the whole op catalog) into GPU-covered vs
    numpy-only, grouping the numpy-only tail by lowering category. Also reports
    whether the pointwise-DAG fusion vocabulary is fully GPU-covered."""
    names = list(op_names) if op_names is not None else list(_oc.OP_SPECS)
    numpy_only: list[str] = []
    by_cat: dict[str, list[str]] = collections.defaultdict(list)
    for n in names:
        if has_gpu_lane(n):
            continue
        numpy_only.append(n)
        spec = _oc.OP_SPECS.get(n)
        by_cat[spec.lowering if spec else "unknown"].append(n)

    # Invariant Phase C/D care about: every fusable pointwise-vocab op already has
    # a single-op GPU lane (so elementwise single-op displacement is complete).
    pw_uncovered = [k for k in _fusion.POINTWISE_OPS
                    if not has_gpu_lane(k)]

    return CoverageReport(
        total=len(names),
        covered=len(names) - len(numpy_only),
        numpy_only=tuple(sorted(numpy_only)),
        by_category={c: tuple(sorted(v)) for c, v in by_cat.items()},
        pointwise_vocab_covered=not pw_uncovered,
    )


def render_report(report: CoverageReport | None = None) -> str:
    r = report or numpy_lane_worklist()
    lines = [
        "Apple GPU numpy-lane displacement worklist",
        f"  ops classified : {r.total}",
        f"  GPU-covered    : {r.covered}",
        f"  numpy-only     : {r.numpy_only_count}",
        f"  pointwise-vocab fully GPU-covered: {r.pointwise_vocab_covered}",
        "",
        "numpy-only by lowering category (category → displacement disposition):",
    ]
    # Group categories by disposition so the real displacement target stands out.
    order = {"real_gap_structural": 0, "host_utility": 1, "distributed": 2,
             "hard_kernel": 3, "unclassified": 4}
    by_disp: dict[str, int] = {}
    for cat, ops in r.by_category.items():
        by_disp[disposition_for(cat)] = by_disp.get(disposition_for(cat), 0) + len(ops)
    for cat, ops in sorted(
            r.by_category.items(),
            key=lambda kv: (order.get(disposition_for(kv[0]), 9), -len(kv[1]))):
        lines.append(f"  [{disposition_for(cat):20s}] {cat:24s} {len(ops):3d}  "
                     f"{', '.join(ops)}")
    lines.append("")
    lines.append("counts by displacement disposition:")
    for disp in sorted(by_disp, key=lambda d: order.get(d, 9)):
        lines.append(f"  {disp:22s} {by_disp[disp]}")
    return "\n".join(lines)


def fallback_histogram(run_fn: Callable[[], Any]) -> dict[tuple[str, str], int]:
    """Phase B (runtime half) — the complement to the static worklist above.

    Run ``run_fn`` (a closure that invokes a model / decoder layer under
    ``@jit(target="apple_gpu")``) and return a histogram of the *failure-class*
    dispatch fallbacks it triggered, keyed by ``(op_name, reason)`` → count. This
    captures the runtime shape-bail / dtype-bail / Metal-failure reasons the
    static classifier can't see (no-lane ops never reach this path — they raise).

    A silent rot (a kernel that quietly degrades to numpy) shows up here as a
    non-empty histogram; an all-GPU run returns ``{}``. See
    ``runtime.dispatch_fallback_log`` (the purpose-built failure-class log).
    """
    from tessera import runtime as _rt

    _rt.reset_dispatch_fallback_log()
    run_fn()
    hist: dict[tuple[str, str], int] = collections.Counter()
    for op_name, reason in _rt.dispatch_fallback_log():
        hist[(op_name, reason)] += 1
    return dict(hist)


if __name__ == "__main__":  # pragma: no cover - human-facing report
    print(render_report())
