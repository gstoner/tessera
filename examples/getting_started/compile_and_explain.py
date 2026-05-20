#!/usr/bin/env python3
"""Canonical Tessera compiler tour.

The shortest path from "no Tessera knowledge" to "I understand what
just happened" when a function is JIT-compiled.  Walks through the
five entry points a developer needs:

  1. ``@tessera.jit`` — the decorator.
  2. ``fn(...)`` — call it like a regular function.
  3. ``fn.explain()`` — single front-door diagnostic.
  4. ``tessera.compiler.support(op)`` — per-op readiness query.
  5. ``tessera.from_text(source)`` — notebook-safe factory.

Runs on the CPU reference path with no accelerator required.
"""

import numpy as np

import tessera as ts


# ──────────────────────────────────────────────────────────────────
# 1. Decorate a function — @tessera.jit reads the source, builds
#    Graph IR / Schedule IR / Tile IR / Target IR, and returns a
#    JitFn object you can call like any Python function.
# ──────────────────────────────────────────────────────────────────


@ts.jit
def fused_add_relu(
    x: ts.Tensor["B", "D"],
    y: ts.Tensor["B", "D"],
) -> ts.Tensor["B", "D"]:
    """Elementwise add followed by ReLU."""
    return ts.ops.relu(ts.ops.add(x, y))


def main() -> None:
    print("Tessera Compiler Tour")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # 2. Call the JIT'd function.  Inputs are numpy arrays; the
    #    CPU reference path runs by default when no accelerator
    #    target is set.
    # ──────────────────────────────────────────────────────────
    print("\n[2] Calling fused_add_relu(x, y) on the CPU reference path:")
    x = np.array([[1.0, -2.0, 3.0], [-1.0, 0.0, 4.0]], dtype=np.float32)
    y = np.array([[0.5, 0.5, -1.0], [1.0, 1.0, -5.0]], dtype=np.float32)
    out = fused_add_relu(x, y)
    print(f"  out shape: {out.shape}")
    print(f"  out: {out.tolist()}")

    # ──────────────────────────────────────────────────────────
    # 3. fn.explain() — the single inspection front door.  Prints
    #    a 5-line summary answering: what ran / native vs reference
    #    / why / next action.
    # ──────────────────────────────────────────────────────────
    print("\n[3] fused_add_relu.explain():")
    explain = fused_add_relu.explain()
    print(explain)

    print("\n    Structured fields hang off the same object:")
    print(f"      execution_kind:    {explain.execution_kind}")
    print(f"      is_native:         {explain.is_native}")
    print(f"      is_reference:      {explain.is_reference}")
    print(f"      kernels resolved:  {len(explain.kernels)}")
    for k in explain.kernels:
        print(f"        - {k.op_name:8} runtime={k.runtime_status} ({k.source})")
    print(f"      diagnostics:")
    for d in explain.diagnostics:
        print(f"        [{d.severity}] {d.code_value}: {d.message[:60]}")
    print(f"      next actions:")
    for n in explain.next_actions:
        if n.message:
            print(f"        [{n.code}] {n.message}")
        else:
            print(f"        [{n.code}]")

    # ──────────────────────────────────────────────────────────
    # 4. Per-op readiness query.  Same data the support_table.md
    #    dashboard renders, exposed as a Python function.
    # ──────────────────────────────────────────────────────────
    print("\n[4] Per-op readiness via ts.compiler.support():")
    for op_name in ("matmul", "relu", "add", "mobius", "cross_ratio"):
        info = ts.compiler.support(op_name)
        print(
            f"  {op_name:12} family={info.family:18} "
            f"best_tier={info.best_tier.value}"
        )
        # Show the top-3 target rows so the per-target picture is
        # visible without overwhelming the output.
        for target in info.targets[:3]:
            print(
                f"    {target.target:14} target_ir={target.target_ir:14} "
                f"runtime={target.runtime:10} tier={target.tier.value}"
            )

    # Predicate convenience:
    print("\n    is_native_supported(matmul, target='apple_gpu'):",
          ts.compiler.is_native_supported("matmul", target="apple_gpu"))
    print("    is_native_supported(cross_ratio, target='apple_gpu'):",
          ts.compiler.is_native_supported("cross_ratio", target="apple_gpu"))

    # ──────────────────────────────────────────────────────────
    # 5. Notebook-safe construction via ts.from_text(...).
    #    Useful when @tessera.jit can't read the source (REPL,
    #    Jupyter heredoc, dynamically-generated code).
    # ──────────────────────────────────────────────────────────
    print("\n[5] Notebook-safe construction via ts.from_text(...):")
    fn = ts.from_text("""
        def gelu_then_norm(x):
            return ts.ops.layer_norm(ts.ops.gelu(x))
    """)
    print(fn.explain())

    print("\n" + "=" * 60)
    print("Done. Next: see docs/reference/tessera-api-reference.md.")


if __name__ == "__main__":
    main()
