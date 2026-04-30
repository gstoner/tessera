
#!/usr/bin/env python3
"""Minimal standalone walkthrough of Tessera's compiler artifact flow.

The script uses the current Phase 1 Python surface:

- `@tessera.jit`
- `tessera.ops.*`
- `JitFn.ir_text()`
- `JitFn.schedule_ir`, `tile_ir`, and `target_ir`

Run from the repo root with `PYTHONPATH=python`, or install Tessera in editable
mode first.
"""

from __future__ import annotations

import importlib.util

import numpy as np


HAVE_TESSERA = importlib.util.find_spec("tessera") is not None
if HAVE_TESSERA:
    import tessera
else:
    tessera = None

def main():
    if not HAVE_TESSERA:
        print("Tessera is not importable. Run with `PYTHONPATH=python` from the repo root.")
        return

    @tessera.jit(cpu_tile=(32, 32, 16))
    def mlp_step(x, w):
        h = tessera.ops.matmul(x, w)
        return tessera.ops.relu(h)

    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    w = np.eye(4, dtype=np.float32)
    out = mlp_step(x, w)
    print("ran mlp_step")
    print("output shape:", out.shape)
    print("uses compiled CPU path:", mlp_step.uses_compiled_path)
    print("lowering:")
    print(mlp_step.explain_lowering())

    artifacts = {
        "graph": mlp_step.ir_text(),
        "schedule": mlp_step.schedule_ir,
        "tile": mlp_step.tile_ir,
        "target": mlp_step.target_ir,
    }
    for level, text in artifacts.items():
        print(f"\n==== {level.upper()} IR ====")
        print((text or f"// no {level} artifact emitted")[:1200])

if __name__ == "__main__":
    main()
