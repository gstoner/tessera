---
status: Informative
classification: Guide
authority: Debugging workflows; defers stable diagnostics to docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md
last_updated: 2026-04-28
---

# Tessera Debugging Tools Guide

Tessera debugging is layered because ML failures are layered. A wrong answer may
come from Graph IR shape logic, Schedule IR movement, Tile IR synchronization,
numerics, autodiff, distributed ordering, or runtime device faults. The debug
toolkit gives each layer a narrow job and a shared Python surface.

Use this guide with:

- `docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md` for stable error
  codes and diagnostic fields.
- `docs/spec/shape-system.md` for shape, layout, shard, and witness checks.
- `docs/spec/MEMORY_MODEL_SPEC.md` for barrier, fence, mbarrier, and ordering
  semantics.

---

## 1. Debugging Layers

| Layer | Primary question | Tooling |
|-------|------------------|---------|
| Graph IR | Is the math graph right? | `tessera.graph.trace`, MLIR dumps, GraphViz export |
| Schedule IR | Did fusion, tiling, movement, and layout stay legal? | schedule dumps, schedule artifact hashes |
| Tile IR | Are fragments, shared memory, async copies, and barriers legal? | Tile IR dumps, verifier diagnostics |
| Runtime numerics | Where did values go bad? | `tessera.debug.debug_trace`, tensor summaries, finite checks |
| Autodiff | Are gradients correct? | `tessera.debug.check_grad` |
| Reproducibility | Is execution deterministic? | `tessera.debug.check_determinism`, replay manifests |

---

## 2. Graph Inspection

Graph inspection should be the first step when behavior looks wrong. It answers:

- Which ops are present?
- What are the dependencies?
- Which IR level is being inspected?
- Did lowering preserve the intended structure?

```python
import tessera as ts

@ts.jit
def step(x, w):
    return ts.ops.softmax(ts.ops.matmul(x, w))

ts.graph.trace(step).print()
```

The same API accepts MLIR strings, `@jit` wrappers with `.graph_ir.to_mlir()`,
and lightweight op descriptor lists:

```python
ops = [
    {"op": "tensor", "output": "%0"},
    {"op": "matmul", "inputs": ["%0", "%0.T"], "output": "%1"},
    {"op": "softmax", "inputs": ["%1"], "output": "%2"},
]

trace = ts.debug.trace_graph(ops)
print(trace.format())
print(trace.to_graphviz())
```

Compiler integrations should support dumps at:

- Graph IR: algebraic operators, shapes, state/cache objects, collectives.
- Schedule IR: tiling, fusion, layout casts, movement plans, async copies.
- Tile IR: fragments, memory spaces, barriers/mbarriers, target intrinsics.
- Target IR: LLVM/PTX/ROCm, kernel ABI, launch metadata.

Recommended environment switches:

```text
TESSERA_DEBUG_IR=1
TESSERA_LOG_LEVEL=DEBUG
TESSERA_PROF_TRACE=trace.json
```

---

## 3. Numerical Tracing

Numerical tracing records summaries instead of dumping full tensors by default.
This keeps logs readable and avoids accidentally materializing huge values.

```python
import io
import numpy as np
import tessera as ts

log = io.StringIO()

with ts.debug.debug_trace(samples=4, stream=log) as trace:
    scores = ts.debug.trace_value("%scores", np.array([[0.1, 0.2, 0.3]]))
    probs = ts.debug.trace_value("%probs", np.exp(scores))

print(log.getvalue())
```

Each summary includes:

- Shape and dtype.
- Mean and standard deviation.
- Min and max.
- Finite/non-finite status.
- Optional sample values.

For production-scale traces, prefer summaries plus replay metadata. Full tensor
capture should be opt-in and bounded by tensor name, rank, size, and privacy
policy.

---

## 4. Gradient Checking

Autodiff bugs are best isolated with small scalar functions and finite
differences. Tessera exposes a callable-oriented checker:

```python
import numpy as np
from tessera.debug import check_grad

def loss(x):
    return np.sum(x * x)

x = np.array([1.0, 2.0, 3.0])
analytic = 2.0 * x

result = check_grad(loss, [x], analytic_grads=[analytic], eps=1e-4, atol=1e-5)
print(result.format())
```

The checker intentionally validates scalar-valued functions. For large models,
wrap a focused subgraph and reduce to a scalar loss. Future compiler
integration should add:

- `backward(wrt="weights" | "arch")` agreement checks.
- Custom VJP/JVP validation.
- Distributed gradient all-reduce determinism checks.
- Gradient logging by tensor name and mesh axis.

---

## 5. Determinism Checks

Determinism checks repeatedly run a zero-argument function and compare outputs.
Bitwise mode uses `rtol=0` and `atol=0`.

```python
from tessera.debug import check_determinism

result = check_determinism(lambda: model(batch), runs=5)
print(result.format())
```

Use deterministic checks for:

- Random/dropout debugging.
- Collective ordering regressions.
- Autotuner reproducibility.
- Schedule artifact replay.
- Cross-device reproducibility investigations.

Under deterministic profiles, the compiler/runtime must govern RNG streams,
floating reduction order, collective ordering, dropout masks, and schedule
choice. Violations should surface as `E_NONDETERMINISTIC`.

---

## 6. External Debugger Integration

Tessera should interoperate with existing tools instead of replacing them:

- Python: `pdb`, IPython, pytest failure introspection.
- C++: `gdb`, `lldb`, structured `Status`/`TesseraException` payloads.
- MLIR: `mlir-opt --debug-only=tessera`, pass pipeline dumps.
- GPU: vendor profilers, Chrome traces, NVTX-style ranges, ROCm traces.

Debug symbols and generated artifacts should be retained with:

```text
TESSERA_DEBUG_IR=1
TESSERA_KEEP_PTX=1
TESSERA_DUMP_STATE=1
```

---

## 7. Implementation Contract

The current Python foundation is in `python/tessera/debug.py`:

- `trace_graph(value, ir_level="graph")`
- `export_graphviz(value)`
- `debug_trace(samples=0, stream=None)`
- `trace_value(name, value)`
- `summarize_tensor(value)`
- `check_grad(fn, inputs, analytic_grads=...)`
- `check_determinism(fn, runs=5)`

The convenience namespace `tessera.graph` maps:

- `tessera.graph.trace -> tessera.debug.trace_graph`
- `tessera.graph.debug_trace -> tessera.debug.debug_trace`
- `tessera.graph.export_graphviz -> tessera.debug.export_graphviz`

Future compiler work should add native ODS/debug ops only where they carry
semantic information that cannot be represented by diagnostics or IR dumps:

- `tessera.graph.debug_value` for named graph-level capture points.
- `tessera.schedule.debug_artifact` for schedule hash/movement inspection.
- `tessera.tile.debug_barrier` for barrier/mbarrier verifier traces.
- `tessera.runtime.replay_capture` for deterministic incident replay.

---

## 8. Best Practices

- Start at Graph IR before debugging kernels.
- Prefer tensor summaries over full tensor dumps.
- Keep gradient checks tiny and scalar-valued.
- Run determinism checks before and after enabling autotune.
- Attach graph hash, schedule hash, target, seed, and replay manifest to any
  production bug report.
- When debugging distributed jobs, log per-rank shapes before collectives and
  keep collective trace windows short.
