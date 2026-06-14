---
name: add-moe-model
description: >
  Port a new MoE (or dense) architecture into tessera.train end-to-end as a
  single self-contained model file, with a shape + forward smoke test. Use when
  the task is "add model X", "integrate <paper> into Tessera training", "port
  MoBA/DynMoE/Diff-Transformer", or similar new-architecture requests.
triggers:
  - "add a new model"
  - "port <architecture> into tessera.train"
  - "integrate this paper's model"
  - "implement model X end-to-end"
---

# Skill: add-moe-model

Port a new architecture into `tessera.train` **agent-natively**: one
self-contained file, directly instantiated, no registry indirection, verified
by a runnable test. This encodes the procedure so you act on a fixed plan
instead of re-deriving one.

## Specific scope

In scope: adding ONE model as `python/tessera/train/models/<name>.py`, built by
direct `nn.Module` instantiation, plus a smoke test under `tests/unit/`.
Out of scope: new compiler primitives/kernels (that's a `@tessera.jit` /
`ops.*` change — do that first, separately, if the model needs an op Tessera
lacks), and multi-node/throughput work (hardware-gated).

## Prerequisites (check before editing — fail fast if missing)

1. A reference implementation or paper for the architecture.
2. The required ops already exist in `tessera.nn` / `tessera.ops`. Grep
   `python/tessera/nn/__init__.py` `__all__` and `python/tessera/ops.pyi`. If a
   needed op is missing, STOP — adding the op is a separate task with its own
   VJP/JVP + catalog/coverage updates (Decision #24). Do not fake it in the
   model file.
3. `python/tessera/train/models/qwen3_moe.py` exists (the template to copy).

## Procedure

1. **Copy the template.** `cp python/tessera/train/models/qwen3_moe.py
   python/tessera/train/models/<name>.py`. Everything you change stays in this
   one file — no shared skeleton to thread.
2. **Edit in place.** Rename `Qwen3MoEConfig/Block/Model` → `<Name>...`. Adjust
   the config dataclass, then the block's `forward` to match the reference
   (e.g. swap attention for block attention, change the router, add a gate).
   Reuse `tessera.train.engine.moe.MoEFeedForward` / `MoERouter` where the MoE
   semantics are standard; inline a new variant in THIS file when they differ.
3. **Register the export** in `python/tessera/train/models/__init__.py` and (if
   public) `python/tessera/train/__init__.py`. This is a 1-line edit, not a
   registry — direct name binding only.
4. **Write the smoke test** `tests/unit/test_train_<name>.py`: instantiate at a
   tiny config, run `forward` on a random batch, assert output shape and that
   aux losses are finite. Copy `scripts/smoke_template.py` in this skill folder
   as a starting point.
5. **Verify.** Run the verifiable check below. It must print `PASS`.

## Verifiable success

```bash
PYTHONPATH=python python3 python/tessera/train/skills/add-moe-model/scripts/verify.py <name>
```

Exit 0 / `PASS` ⇔ the model imports, instantiates at a tiny config, forward
produces correctly-shaped logits, and all aux losses are finite. Exit 1 / `FAIL`
otherwise. Do not report success on self-assessment — run this script.

## Anti-patterns (the things PithTrain measures as costly)

- ❌ Resolving submodules through a spec/registry/string key in another file.
- ❌ Splitting one model across multiple files for "reuse".
- ❌ Importing `primitive_coverage` / `op_catalog` / `backend_manifest` or any
  C++/MLIR from the model file (agent-native firewall — see `train/__init__.py`).
- ❌ Claiming the port works without running `verify.py`.
