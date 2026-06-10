"""S-series accelerator-proof map — which primitives execute on a real
accelerator, which *can*, and which are host-only or hardware-blocked.

The Tessera audit tracks 12 *contract* axes per primitive (math / shape / vjp /
… / backend_kernel). This module adds an orthogonal, execution-centric lens:
**given the one accelerator this repo can actually prove on — Apple Silicon GPU
(Metal) — where does each primitive stand?**  NVIDIA/ROCm execution is gated on
hardware we don't have (Phase G/H), so "accelerator-proven" here means
`execution_mode == "metal_runtime"` with a numerically-validated result.

Every primitive is classified into exactly one accelerator class:

  * ``proven``       — its Graph IR op is in the Apple GPU runtime envelope
                       (``driver._APPLE_GPU_RUNTIME_OPS``); a ``@jit(target=
                       "apple_gpu")`` call runs it natively (metal_runtime).
  * ``eligible``     — FLOP-bearing numeric work a Metal kernel would accelerate
                       (elementwise / reductions / losses / attention / matmul /
                       norms / optimizers / quant …) but not yet routed. **This
                       is the actionable accelerator-proof gap.**
  * ``host``         — structural / orchestration / shape / I-O primitives
                       (pytrees, control-flow transforms, dataset pipeline,
                       serialization, LR schedules, layout metadata): there is no
                       FLOP to accelerate, so the host (numpy) path *is* the
                       canonical implementation. Accelerator is not-applicable.
  * ``multi_device`` — collectives / sharding: correctness needs real multi-
                       accelerator hardware (NVIDIA NCCL / AMD RCCL). Cannot be
                       proven on a single Apple GPU.
  * ``special``      — needs a dedicated kernel class not yet built on Apple GPU
                       (spectral/FFT, device-side Philox RNG).

The map is data-driven: ``proven`` is computed live from the envelope + the op
catalog, the rest from the primitive's category. It renders
``docs/audit/generated/s_series_accelerator_proof.md`` and is drift-gated by
``tests/unit/test_accelerator_proof_map.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# Accelerator class per primitive *category*. ``proven`` is decided per-primitive
# (envelope membership) and overrides the category default below.
_ACCEL_CLASS_BY_CATEGORY: dict[str, str] = {
    # ── eligible: FLOP-bearing numeric work, route-able to Metal ──────────────
    "elementwise": "eligible",
    "numeric_helper": "eligible",
    "comparison": "eligible",
    "logical": "eligible",
    "reduction": "eligible",
    "stable_reduction": "eligible",
    "segment_reduce": "eligible",
    "loss": "eligible",
    "rl_loss": "eligible",
    "attention": "eligible",
    "normalization": "eligible",
    "pooling": "eligible",
    "position_encoding": "eligible",
    "rotary_embedding": "eligible",
    "model_layer": "eligible",
    "recurrent": "eligible",
    "state_space": "eligible",
    "loop_nest": "eligible",
    "contraction": "eligible",
    "projection": "eligible",
    "fused_epilogue": "eligible",
    "geometric_algebra": "eligible",
    "ebm": "eligible",
    "quantize": "eligible",
    "quantization": "eligible",
    "numerics": "eligible",
    "functional_optimizer_step": "eligible",
    "optimizer": "eligible",
    "grad_transform": "eligible",
    "stencil": "eligible",
    "sparse": "eligible",
    "moe": "eligible",
    "moe_transport": "eligible",
    "random_mask": "eligible",
    "state_update": "eligible",
    "memory": "eligible",
    # ── host: no FLOP to accelerate — the numpy/host path is canonical ────────
    "tensor_algebra": "host",        # reshape / cat / broadcast / squeeze — layout
    "layout_transform": "host",      # cast / transpose / gather metadata
    "indexing": "host",              # slice / index_select — memory addressing
    "sort": "host",                  # ordering — host is adequate at this scale
    "state_tree": "host",            # pytrees
    "control_flow": "host",          # scan/cond/while orchestration (GPU body via tracer)
    "transform": "host",             # vmap / vjp / jvp / remat / autocast
    "extension": "host",             # custom-primitive escape hatches
    "schedule": "host",              # LR schedules (scalar)
    "data": "host",                  # dataset pipeline / IO
    "serialization": "host",         # checkpoint save/load
    "aot": "host",                   # AOT export / compilation cache
    "tokenizer": "host",             # tokenization
    "conformance": "host",           # tiny-model conformance harness
    "sharding": "host",              # partition specs (planning, not execution)
    # ── multi_device: needs real multi-accelerator hardware ──────────────────
    "collective": "multi_device",
    # ── eligible: a Metal kernel class exists / is route-able ─────────────────
    # spectral FFT lane landed 2026-06-10 (MPSGraph FourierTransform) — all 9
    # spectral ops are now in the envelope (per-primitive `proven`); the category
    # default is `eligible` (a future spectral op routes to the FFT lane).
    "spectral": "eligible",
    # ── special: needs a dedicated kernel class not yet on Apple GPU ──────────
    "rng": "special",                # device-side Philox not yet wired
    "random_source": "special",
}

_CLASS_ORDER = ("proven", "eligible", "special", "multi_device", "host")
_CLASS_BLURB = {
    "proven": "executes on Apple GPU today (`metal_runtime`)",
    "eligible": "numeric — route-able to a Metal kernel (the actionable gap)",
    "special": "needs a dedicated Apple-GPU kernel class (device RNG)",
    "multi_device": "needs real multi-accelerator hardware (NVIDIA/AMD)",
    "host": "structural / orchestration / shape — accelerator not-applicable",
}


@dataclass(frozen=True)
class AcceleratorProofRow:
    name: str
    category: str
    accel_class: str
    graph_name: str | None


def _envelope() -> frozenset[str]:
    from tessera.compiler import driver
    return driver._APPLE_GPU_RUNTIME_OPS


def classify(name: str, category: str, graph_name: str | None) -> str:
    """Return the accelerator class for one primitive."""
    if graph_name is not None and graph_name in _envelope():
        return "proven"
    return _ACCEL_CLASS_BY_CATEGORY.get(category, "host")


def all_rows() -> list[AcceleratorProofRow]:
    from tessera.compiler import op_catalog, primitive_coverage
    rows: list[AcceleratorProofRow] = []
    for name, entry in sorted(primitive_coverage.all_primitive_coverages().items()):
        gn = op_catalog.graph_name_for(name)
        rows.append(AcceleratorProofRow(
            name=name, category=entry.category,
            accel_class=classify(name, entry.category, gn), graph_name=gn,
        ))
    return rows


def summary() -> dict[str, int]:
    counts = Counter(r.accel_class for r in all_rows())
    return {cls: counts.get(cls, 0) for cls in _CLASS_ORDER}


def render_markdown() -> str:
    rows = all_rows()
    total = len(rows)
    counts = summary()
    proven, eligible = counts["proven"], counts["eligible"]
    # accelerator-relevant denominator excludes host + multi_device (can't be
    # proven on a single Apple GPU by design).
    relevant = proven + eligible + counts["special"]
    lines: list[str] = [
        "<!-- AUTO-GENERATED by python/tessera/compiler/accelerator_proof.py — DO NOT EDIT BY HAND. -->",
        "<!-- Regenerate via: python -c 'from tessera.compiler.accelerator_proof import write_dashboard; write_dashboard()' -->",
        "",
        "# S-series accelerator-proof map",
        "",
        "Execution-centric lens over the standalone-compiler primitive registry: "
        "given the one accelerator this repo can actually prove on — **Apple "
        "Silicon GPU (Metal)** — where does each primitive stand? "
        "*Accelerator-proven* means a `@jit(target=\"apple_gpu\")` call runs it "
        "with `execution_mode == \"metal_runtime\"` and a numerically-validated "
        "result. NVIDIA/ROCm execution is hardware-gated (Phase G/H) and out of "
        "scope for this map.",
        "",
        f"**{proven}/{total} primitives are accelerator-proven on Apple GPU "
        f"today.** Of the {relevant} accelerator-relevant primitives "
        f"(proven + eligible + special), **{eligible} are *eligible*** — "
        "FLOP-bearing numeric ops that a Metal kernel would accelerate but which "
        "aren't routed through the envelope yet. That is the actionable "
        "accelerator-proof gap; the rest are host-only or hardware-blocked by "
        "design.",
        "",
        "## Classes",
        "",
        "| Class | Count | Meaning |",
        "|-------|------:|---------|",
    ]
    for cls in _CLASS_ORDER:
        lines.append(f"| `{cls}` | {counts[cls]} | {_CLASS_BLURB[cls]} |")
    lines += [
        "",
        "## By category",
        "",
        "| Category | n | proven | eligible | class |",
        "|----------|--:|-------:|---------:|-------|",
    ]
    by_cat: dict[str, list[AcceleratorProofRow]] = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)
    for cat in sorted(by_cat):
        crows = by_cat[cat]
        pv = sum(1 for r in crows if r.accel_class == "proven")
        default = _ACCEL_CLASS_BY_CATEGORY.get(cat, "host")
        el = sum(1 for r in crows if r.accel_class == "eligible")
        lines.append(f"| `{cat}` | {len(crows)} | {pv} | {el} | `{default}` |")
    # The eligible gap, enumerated (the to-route worklist).
    elig = sorted(r.name for r in rows if r.accel_class == "eligible")
    lines += [
        "",
        f"## Eligible worklist ({len(elig)}) — the accelerator-proof gap",
        "",
        "FLOP-bearing numeric primitives with no Apple GPU envelope route yet. "
        "Routing a category here (an MPSGraph/MSL kernel + envelope entry + "
        "dispatcher + a `metal_runtime` test) flips its primitives to `proven`.",
        "",
        "<details><summary>names</summary>",
        "",
        "  " + ", ".join(f"`{n}`" for n in elig),
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def write_dashboard(path: str | None = None) -> str:
    import os
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        path = os.path.join(here, "docs", "audit", "generated",
                            "s_series_accelerator_proof.md")
    text = render_markdown()
    with open(path, "w") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


if __name__ == "__main__":
    print(write_dashboard())
