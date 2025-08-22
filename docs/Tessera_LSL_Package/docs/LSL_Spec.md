# Tessera Learning Specification Language (LSL)

**Goal.** Let users state *what to learn* plus *constraints/capabilities*, while Tessera decides *how* (architecture, schedules, precision, kernels).

---

## 1. Concepts

- **Spec**: Declarative task with IO schemas, inductive bias hints, constraints (latency, memory, energy, accuracy), and capabilities (few-shot, uncertainty, interpretability, robustness).
- **Spaces**: Architecture space (e.g., ResNet/ConvNeXt/equivariant), ScheduleIR space (tile sizes, pipeline stages), Precision space (AMP/INT8).
- **Solve**: Multi-objective search with cost models + on-device measurements, producing Graph IR → Schedule IR → Tile IR and a report of constraint satisfaction.

---

## 2. Python API (Prototype)

```python
import tessera.lsl as lsl

spec = lsl.learning_objective(
    task="image_classification",
    description="classify images with rotation invariance",
    inputs={"x": "image[b,c,h,w]"},
    outputs={"y": "class[b,k]"},
    inductive_bias=["cnn","group-equivariance:so2"]
)
spec += lsl.constraint.latency("<50ms@A100:bs=32")
spec += lsl.constraint.accuracy(">0.95@CIFAR10/val")
spec += lsl.capability.few_shot(True, k=16)
spec += lsl.capability.uncertainty(outputs=("mean","std","epistemic","aleatoric"))

artifact = lsl.solve(spec, target="nvidia:a100:sm80")
print(artifact["report"])
```

**Spec fields**
- `task`, `description`
- `inputs`, `outputs` (string shapes)
- `inductive_bias`: optional hints (e.g., `"cnn"`, `"group-equivariance:so2"`)
- constraints: `latency`, `accuracy`, `memory`, `energy`, `size`
- capabilities: `few_shot`, `uncertainty`, `interpretability`, `robustness`

---

## 3. Semantics

- **Constraints**
  - *Hard* constraints (latency/memory/size) prune candidates prior to training.
  - *Soft* constraints (accuracy/energy) enter the scalarized search objective.
- **Determinism**
  - Seeds recorded; reductions order fixed when uncertainty/metrics require it.
- **Outputs**
  - Bundle: `{graph_ir, schedule_ir, tile_ir, binaries?, report}`.
  - Cached schedules per `(op, shape, arch, dtype)`.

---

## 4. Mapping to Tessera IRs

- **Graph IR**: Parametric operator graphs; heads for uncertainty & interpretability taps.
- **Schedule IR**: Legal tiling/pipeline sets + autotuner hooks.
- **Tile IR**: Kernel realizations (mma/wgmma/mfma, cp.async, mbarrier).
- **Target IR**: NVVM/PTX, ROCDL, CPU LLVM (AMP/INT8 policies applied).

---

## 5. Examples included

- `vision_rotation_invariance.py` — builds a CIFAR-10 spec with rotation invariance + few-shot + latency cap.
- `llm_uncertainty_energy_cap.py` — language modeling spec with uncertainty outputs and an energy budget.

Both examples run with the **prototype** solver in this package (no external Tessera needed).

---

## 6. Roadmap (sketch)

- Connect spaces to real Tessera Graph/Schedule/Tile IR builders.
- Add cost models (latency/energy) and on-device profilers.
- Support distillation warm-start and differentiable NAS.
- Export full artifacts (IR dumps + target binaries + JSON report).
