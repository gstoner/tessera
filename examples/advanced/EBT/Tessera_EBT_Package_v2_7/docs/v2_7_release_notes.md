<!-- MERGE_START -->
# EBT v2.7 — Dialect ODS, Autodiff MLP Example, CPU JVP Unit Test

**Adds**
- `tessera.ebt` dialect in ODS (`.td`) defining core ops:
  - `ebt.energy`, `ebt.energy_bilinear`, `ebt.energy_mlp`, `ebt.energy_bilinear_jvp`
  - `ebt.grad_y`, `ebt.inner_step`, `ebt.decode_init`, `ebt.self_verify`
- Autodiff MLP energy example showing `tessera.autodiff.grad_y` on the MLP head.
- CPU unit test that checks **JVP == directional derivative**:
  - For scalar energy `E(y)`, verify `(E(y+εv)-E(y))/ε ≈ grad(y)·v` and matches custom JVP.
<!-- MERGE_END -->
