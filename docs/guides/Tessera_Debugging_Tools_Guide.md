---
status: Informative
classification: Guide
authority: Debugging workflows; defers stable diagnostics to docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md
last_updated: 2026-05-06
---

# Tessera Debugging Tools Guide

Tessera debugging is layered because ML failures are layered. A wrong answer may
come from Graph IR shape logic, Schedule IR movement, Tile IR synchronization,
numerics, autodiff, distributed ordering, or runtime device faults. The debug
toolkit gives each layer a narrow job and a shared Python surface.

Use this guide with:

- `docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md` for stable error
  codes and diagnostic fields.
- `docs/spec/SHAPE_SYSTEM.md` for shape, layout, shard, and witness checks.
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

Current support status:

| Capability | Status | Notes |
|------------|--------|-------|
| Python graph tracing | Implemented | Accepts `@jit` wrappers, MLIR strings, object models, and op descriptor lists. |
| Structured debug trace JSON | Implemented | `DebugTrace.to_dict()` and `DebugTrace.to_json()` emit bounded summaries. |
| Replay manifests | Implemented | `tessera.debug.replay_manifest(...)` captures hashes, metadata, and debug environment switches. |
| Compiler artifact hashes | Implemented | JIT/runtime artifacts include Graph, Schedule, Tile, and Target hashes where available. |
| `tessera-mlir` static inspection | Implemented | Does not execute model files. |
| `tessera-mlir` compile artifact mode | Implemented, opt-in | Requires `--mode=compile_artifact --symbol=name`; imports the source module and reads a JIT artifact without launching tensors. |
| Native runtime tensor capture | Planned | Full tensor capture must remain opt-in and bounded. |

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
print(trace.to_json())
```

Compiler integrations should support dumps at:

- Graph IR: algebraic operators, shapes, state/cache objects, collectives.
- Schedule IR: tiling, fusion, layout casts, movement plans, async copies.
- Tile IR: fragments, memory spaces, barriers/mbarriers, target intrinsics.
- Target IR: LLVM/PTX/ROCm, kernel ABI, launch metadata.

Recommended environment switches:

```text
TESSERA_DEBUG_IR=1
TESSERA_DUMP_DIR=/tmp/tessera_debug
TESSERA_DUMP_STATE=1
TESSERA_LOG_LEVEL=DEBUG
TESSERA_PROF_TRACE=trace.json
```

Command-line IR dumps use the `tessera-mlir` developer command:

```bash
tessera-mlir my_model.py --emit=graph-ir --debug
tessera-mlir my_model.py --emit=schedule-ir -o schedule.mlir
tessera-mlir my_model.py --emit=tile-ir --debug
tessera-mlir my_model.py --emit=metadata --target=apple_cpu
tessera-mlir my_model.py --emit=all --artifacts-dir debug_artifacts
tessera-mlir my_model.py --mode=compile_artifact --symbol=step --emit=all --artifacts-dir compiled_debug
```

The current command performs static source inspection and does not execute the
model file. It emits stable debug artifacts for Graph IR, Schedule IR, Tile IR,
Target IR, metadata, diagnostics, Chrome trace JSON, and GraphViz entry points.
`--mode=compile_artifact --symbol=name` is explicitly opt-in because it imports
the Python module. It does not launch tensors, but it does read the selected
JIT wrapper's verified runtime artifact and lowering trace.

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
print(trace.to_json())
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

Named capture points use the same summary behavior:

```python
with ts.debug.debug_trace(samples=2, metadata={"graph_hash": "..."}) as trace:
    value = ts.graph.debug_value("%scores", scores)
```

The Python marker returns `value` unchanged. Compiler-native lowering preserves
the marker as a debug artifact where the object-model pipeline can represent it.

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
TESSERA_DUMP_DIR=/tmp/tessera_debug
TESSERA_KEEP_PTX=1
TESSERA_DUMP_STATE=1
```

Environment switch behavior:

| Variable | Behavior |
|----------|----------|
| `TESSERA_DEBUG_IR=1` | JIT/compiler bundle paths write `graph.mlir`, `schedule.mlir`, `tile.mlir`, `target.mlir`, and backend artifacts when present. |
| `TESSERA_DUMP_STATE=1` | Writes `metadata.json` and Chrome trace `trace.json` for the compiler bundle. |
| `TESSERA_DUMP_DIR=...` | Selects the artifact root. If omitted, the compiler uses a `tessera_debug` directory under the system temp directory. |
| `TESSERA_KEEP_PTX=1` | Reserved for native NVIDIA codegen artifacts. Current Target IR paths mark CUDA artifacts as inspectable contracts. |
| `TESSERA_PROF_TRACE=trace.json` | Used by profiling surfaces for Chrome trace export; see the profiling and autotuning guide. |

Dump directories are named from function, target, and the graph hash prefix so
multiple compiler invocations can coexist without overwriting one another.

---

## 7. Implementation Contract

The current Python foundation is in `python/tessera/debug.py`:

- `trace_graph(value, ir_level="graph")`
- `export_graphviz(value)`
- `debug_trace(samples=0, stream=None, metadata=None)`
- `trace_value(name, value)`
- `summarize_tensor(value)`
- `TensorSummary.to_dict()`
- `DebugTrace.to_dict()` and `DebugTrace.to_json()`
- `GraphTrace.to_dict()` and `GraphTrace.to_json()`
- `debug_value(name, value, metadata=None)`
- `debug_artifact(name, artifact=None, metadata=None)`
- `debug_barrier(name, queue_id=None, scope="block", metadata=None)`
- `replay_manifest(value=None, **metadata)`
- `save_replay_manifest(path, value=None, **metadata)`
- `replay_capture(value=None, **metadata)`
- `check_grad(fn, inputs, analytic_grads=...)`
- `check_determinism(fn, runs=5)`

The command foundation is in `python/tessera/cli/mlir.py` and is installed as:

- `tessera-mlir`

The convenience namespace `tessera.graph` maps:

- `tessera.graph.trace -> tessera.debug.trace_graph`
- `tessera.graph.debug_trace -> tessera.debug.debug_trace`
- `tessera.graph.debug_value -> tessera.debug.debug_value`
- `tessera.graph.export_graphviz -> tessera.debug.export_graphviz`
- `tessera.graph.replay_capture -> tessera.debug.replay_capture`

Compiler object-model hooks currently preserve lightweight debug markers:

- `tessera.graph.debug_value` for named graph-level capture points.
- `schedule.debug_artifact` for schedule hash/movement inspection.
- `tile.debug_artifact` and `tile.debug_barrier` for barrier/mbarrier verifier
  traces.
- `tessera.runtime.replay_capture`-style replay manifests through
  `tessera.debug.replay_capture`.

Native ODS/debug ops should only be added where they carry semantic information
that cannot be represented by diagnostics, artifact metadata, or IR dumps.

Replay manifest example:

```python
artifact = step.runtime_artifact()
manifest = ts.debug.replay_manifest(
    artifact,
    seed=1234,
    profiler_trace="trace.json",
)
ts.debug.save_replay_manifest("replay.json", artifact, seed=1234)
```

Manifests include the debug environment switches and artifact summaries. They do
not include full tensor values unless the caller separately attaches bounded
summaries.

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
