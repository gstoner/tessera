# gfx1201 scheduled integration and fleet validation

Owner: ROCM-2 / F0 / F2 / EVIDENCE-PACKET-1.
Sync: `GFX1201-INTEGRATION-2026-09-13`.
Baseline: `ca0e0c7263a443ea62974dffb7a33e6f786cff66` plus the uncommitted
increment recorded in `source-identity.json`. The previous foundation packet
is historical and remains unchanged.

## Correctness and compiler checks

Tajasarus: RX 9070 XT gfx1201, ROCm 10.0, Ubuntu 26.04 **WSL2**.
LLVM/MLIR 23.1.1 assertions are enabled (foundation packet records the assertion
probe). The current opt and ROCm opt are rebuilt from this increment.

- `gfx1201-final-tests.txt`: 266 passed, 18 skipped. Native scheduled f32
  softmax and interior-axis mean execute with replay-derived descriptors;
  tampered pointer order, metadata and cached-launcher envelope are refused.
  Skips: 13 Darwin-only, two unavailable native x86, three exact-gfx1151 gates.
- `gfx1201-backward-tests.txt`: ten standalone f16/bf16 backward cases pass,
  comparing dQ/dK/dV for causal/ragged shapes and head dimensions 16/64.
  Forward O comes from the oracle; this does **not** establish resident LSE,
  paired public AD, general masks or scheduled attention packaging.
- `gfx1201-staged-tests.txt`: ten shipped LDS/pipelined GEMM checks pass.
- `contract-audit-tests.txt`: 196 audit, dtype, architecture and fragment checks
  pass. `package-regressions.txt`: 83 passed, 26 device-gated skips.
  Ruff and the zero-error mypy ratchet pass.

Five GPU generators now set LLVM 23's inherent kernel property, eliminating
legacy duplicate `gpu.kernel` attributes under assertions. No matrix/attention
scheduled plugin is admitted for gfx1201 by this increment.

## Fleet separation

Princess-Luna and Super-Bear main checkouts remain clean at the baseline above.
Their existing main compilers were rebuilt: 57 focused tests pass on Luna;
253 on Bear. Bear's assertions-enabled compiler also passes all 34 SM120
arithmetic execution cases with CUDA SDK **13.4.1**, recorded in
`nvidia-arithmetic.json`; this is correctness, not performance promotion.

The **candidate** compiler is separate at `/tmp/tessera-opt-gfx1201-wave` on Luna,
using its existing layout-algebra shared library via an explicit loader path.
Ten gfx1151 backward and six typed-GEMM tests pass (the two `gfx1151-*.txt` files).
This is not a relocated-install smoke or a replacement of Luna's main compiler.
No Metal execution is claimed for this wave. Historical worktrees are preserved.

## Performance decision: retain the register baseline

`gfx1201-staged-comparison.json` contains five fresh processes, identical runtime
hashes, three shapes and all individual samples. Each process validates every
variant against the same f32 oracle before measuring 100 launches. Maximum
absolute error stays below 0.01; no tolerance is changed per candidate.

Median per-launch milliseconds, using the runtime's HIP-event timer checked
against wall time:

| M × N × K | Register | LDS | Pipelined |
|---|---:|---:|---:|
| 128 × 128 × 128 | 0.00943 | 0.08873 | 0.08998 |
| 511 × 513 × 509 | 0.13927 | 0.47643 | 0.51354 |
| 1024 × 1024 × 1024 | 0.40923 | 1.99023 | 1.99385 |

All runs, including the register timing spread, remain in the packet. The
first API-call duration is recorded separately; it is not guaranteed cold-disk
compilation or kernel-only latency. No outlier removal or cross-clock mixing.
These are diagnostic WSL measurements: exact HSACO/profiler attribution,
independent clock validation, native-Linux execution and production candidate
binding remain missing. `promotion_eligible` is explicitly false.

## ISA-guided next experiment

Princess-Luna's `/home/gstoner/AMD_GPU_ISA_DOCS/rdna4-instruction-set-architecture.pdf`
has SHA256 `96dc97df3468a4e63a13095e2540ba13aaa75cf4635a29516b59760695e25e0c`,
identical to the [checked-in archive metadata](../../../docs/reference/isa/rdna/rdna4/meta.json).
Use its §7.12.2 fragment layout, §7.12.1 data hazards, §5.6 barriers and
§11.6.2 load-transpose constraints. Full-wave transpose loads require full EXEC:
first separate interior tiles from guarded tails, inspect emitted waits/LDS
traffic/register occupancy, then measure a new variant. Do not infer overlap
from double buffering or performance from ISA availability.

## Reproduce on the owning hosts

After sourcing Tajasarus's `~/.config/tessera/env.sh`:

```sh
TESSERA_GFX1201_DEVICE_PROOF=1 python -m pytest tests/unit/test_rocm_gfx1201_scheduled.py -q
TESSERA_ROCM_CHIP=gfx1201 python -m pytest tests/unit/test_rocm_flash_attn_bwd_compiled.py -k matches_numpy -q
python -m pytest tests/unit/test_rocm_wmma_runtime_symbol.py -k 'lds or pipe' -q
python -m benchmarks.rocm.record_gfx1201_staged --output /tmp/comparison.json
```

Mainline hardware checks and candidate tests must keep their compiler paths,
architecture and source identities explicit. General packaging and automatic
AD integration remain open; no Graph constructor was deleted.
