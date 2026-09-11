# Native benchmark adapters — 2026-09-10

RTX 5070 (SM120, Super-Bear) and Radeon 8060S (gfx1151, Princess-Luna),
both WSL. Compiler and recorder identities are embedded in packets.

- `*-dlop.json`: ReLU, abs and square two-affine ANN programs, 16×8,
  three measured calls per original/transformed artifact after warmup.
  Every call reports one accepted driver launch followed by successful
  synchronous completion/copyback. There is no measured dispatch reduction.
- `*-public-ssd.json`: public `tessera.control.vjp`, three shapes and two
  cotangents each, all five input gradients. Independent float64 NumPy
  recurrence/central differences; largest absolute gradient error <1.2e-8.
- `*-superbench.json`: standalone native SuperBench adapter smoke, two
  repetitions per variant. Uses the same checked execution boundary.

Commands: `benchmark_native_dispatch.py --backend <backend> --compiler <tool>
--repeat 3 --output <file>`; `benchmark_public_ssd.py --backend <backend>
--compiler <tool> --output <file>`; `ann_native.py --backend <backend>
--compiler <tool> --repeat 2`.

These are bounded correctness and instrumentation diagnostics. Host-wall package
latency includes H2D, validation, driver launch, synchronization and D2H. Receipt
instrumentation adds overhead. Profiler kernel counts remain null; no device
clock/counter or bare-metal performance eligibility is claimed. Neither family
stands in for the whole DLOP catalog or general automatic frontend AD.
