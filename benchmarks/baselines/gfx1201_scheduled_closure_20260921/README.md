# GFX1201 scheduled-suite closure

Owner: `ROCM-2`. Sync: `GFX1201-SCHEDULED-SKIP-CLOSURE-2026-09-21`.

The 90 rows that ordinary non-ROCm CI reports as hardware/compiler-dependent
are closed on their owning device. Tajasarus is an AMD Radeon RX 9070 XT
(`gfx1201`) running ROCm 10.0. The compiler was rebuilt from the
provenance-hardening follow-up to PR #802 against assertions-enabled LLVM/MLIR
23.1.1 before the recorded run. The recorder validated ROCm 10.0.0, HIP
7.15.26333, the selected HIP device's reported model, and the exact toolkit
compiler and loaded runtime library before assigning the proof-build label.

## Result

The complete `tests/unit/test_rocm_gfx1201_scheduled.py` suite passes with
**95 passed, 0 failed, 0 errors, and 0 skipped**. The recorder accounts for:

- **90 exact-device/compiler-dependent rows**: public exact-target dispatch,
  unary packages, dense/fused/BF16/FP8/integer/mixed-FP8 matmul, forward and
  backward attention, saved/recomputed ownership, dynamic image reuse,
  external-reader retirement, paged KV, selected panels, K32 int4, transpose
  loads, and macro-K traversal.
- **5 adjacent host-contract rows**: stale projection rejection, cached-launch
  architecture/ABI refusal, and unknown-architecture rejection.

The first run found a stale `~/.local/bin/tessera-opt` and was deliberately not
accepted as evidence even though all tests passed. The hardened rerun was built
from `9b7099100ebdae14564a17105143201f2e045478`; the recorder required a clean
tested checkout, the same clean compiler-source revision,
`stale_generator_sources == 0`, `/opt/rocm/core-10.0/bin/hipcc`, and the loaded
`/opt/rocm/core-10.0/lib/libamdhip64.so.7.15.26333-0000000`. The committed
[`evidence.json`](evidence.json) binds the fixture and recorder hashes, exact
device/target, compiler binary hash, toolkit paths, HIP runtime hash, family
counts, and JUnit summary. The unit drift gate recomputes both content hashes,
so edits to either the fixture or recorder invalidate this packet.

## Policy boundary

The explicit skip gates stay in ordinary CI: a Mac, CUDA host, gfx1151 host,
or compiler-free job cannot provide gfx1201 proof. Closure means the skip set
has a reproducible, zero-skip owning-host gate and a machine-readable proof
registry entry; it does not mean pretending every CI runner has the hardware.
No result transfers to `gfx1200` or to a Radeon AI PRO R9700 performance claim.

Reproduce on Tajasarus after rebuilding `tessera-opt` from the checked-out
revision:

```sh
ROCM_PATH=/opt/rocm/core \
TESSERA_ROCM_CHIP=gfx1201 \
TESSERA_GFX1201_DEVICE_PROOF=1 \
TESSERA_OPT="$PWD/build-rocm/tools/tessera-opt/tessera-opt" \
PYTHONPATH=python \
python -m benchmarks.rocm.record_gfx1201_scheduled_closure \
  --output /tmp/gfx1201-scheduled-closure.json
```
