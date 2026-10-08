# Native requested attention gradient activity

Owner AD-RESIDUAL-EVAL-1; sibling FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-VJP-ACTIVITY-2026-10-05.

## Compiler integration

Public isolated reverse attention enables native paired-AD
prune-checkpoint-gradients. The native pass validates nonempty unique in-range
wrt indices and resolves requested Q/K/V/bias roles through actual forward SSA
BlockArguments. Reordered frontend arguments retain their physical roles.
Paired reverse AD currently emits all cotangents; pruning does not assume
those returns were already narrowed by wrt.

The native checkpoint export carries activity. Graph-to-Schedule hashes activity
and zero_fill_v1 semantics. Schedule-to-Tile replay checks the retained contract.
Tile verification and SM120 Target lowering validate binary result activity,
omit inactive gradient arithmetic and emit deterministic zero stores.
Public runtime checks native activity against requested gradient ordering.
The default complete checkpoint export remains unchanged. Python carries typed
requests and reads checked native metadata; no Python GPU body is introduced.

Flow: Python frontend → native paired AD → verified Graph → hashed Schedule
→ Tile launch carrier → NVIDIA Target/LLVM/NVVM → PTX plus checked resident ABI.
The complete buffer ABI, output allocations, launch range, private saved O/LSE
and synchronization remain unchanged. Compact output allocation/launch remains
a follow-on obligation.

## Exact-device evidence

Device: NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 12.0, 610.88. Matching tessera-opt and tessera-nvidia-opt, all lowering
sources and the recorder are fingerprinted in packet.json.

- 431 focused tests pass: native request validation, reordered arguments,
  changed/malformed activity refusal, default full-export compatibility,
  existing bias/broadcast device execution, pass metadata and diagnostics.
- 24 matched finite cases cover Sk=5/129, causal/noncausal, grouped heads,
  unequal Q/K lengths, Q-only/K-only/V-only, reordered mixed and full requests.
  Float64 forward/backward oracle checks precede and follow timed launches,
  at unchanged 4e-5 absolute/relative tolerance.
- Eight V-only cases prove finite dV for Inf/NaN primal V and repeated
  cotangent multipliers 1/2. Inactive Q/K outputs are exactly zero.
- Four V-only cases with reordered K/bias/V/Q arguments prove full and physical
  broadcast bias inputs. Inactive Q/K/physical dBias outputs are exactly zero.
- Maximum absolute error across all 36 cases: 5.83429264e-08.
- Native V-only Target snapshots have no pointer addressing from primal V or
  saved O; integrity.json records matching source/tool hashes.
- Actual block is 128 threads, local/shared bytes zero. V-only registers are
  40/48 noncausal/causal versus 48/56 for all gradients. Occupancy is queried
  using the actual block and shared size.

## Balanced dispatch comparison

Five alternating-order trials, 32 launches per CUDA-event window,
after correctness and warmup. Both packages share input values, primal policy
and complete allocation/launch geometry. Baseline requests all Q/K/V gradients;
candidate requests the listed subset. Preloaded dispatch windows include host
dispatch gaps; these are not isolated kernel or application speedups.
Checked allocating/synchronizing backward wall samples remain separate.

| V-only envelope | All gradients, ms | Requested gradients, ms | Dispatch ratio |
| --- | ---: | ---: | ---: |
| Sk=5, causal=False | 0.011409 | 0.009812 | 1.16x |
| Sk=5, causal=True | 0.011499 | 0.010471 | 1.10x |
| Sk=129, causal=False | 0.071635 | 0.017171 | 4.17x |
| Sk=129, causal=True | 0.070190 | 0.013845 | 5.07x |

Q/K/mixed/full controls remain in packet.json. No global format or schedule
strategy is promoted. FP8/MXFP8/MXFP4 remain independent quality/numerical/
performance gates.

## Sibling assessment and remaining work

Matching RX 9070 XT / gfx1201 compiler rebuild completed. Thirty-nine shared
native AD/request/argument-role tests pass, three CUDA-only tests skip.
Eighteen existing gfx1201 norm/epilogue exact-device regressions pass.
Initial shared failures were missing imported benchmark fixtures; the matched
successful retry is retained separately. HIP physical pruning is not implemented:
its existing backward carrier rejects this SM120 saved-output ABI.

Apple and x86 consumers are outside this native SM120 physical contract.
General paired AD remains unchanged outside the opt-in export. No Metal,
AVX-512 or HIP pruning/performance proof is claimed.

General composed/dynamic/higher AD, compact gradient allocation and sibling
consumers remain open. Full five-slice closure also requires general W1.1
producer reconstruction and broader ROCm cache/layout/movement/performance.
Graphify CLI is unavailable in the authoritative scratch checkout; no refreshed
graph is claimed.

Evidence: build-final.txt, unit-final.txt, packet.json, device.txt, artifacts/,
integrity.json, vjp-activity-build-20261005.txt,
vjp-activity-shared-initial-20261005.txt, vjp-activity-shared-20261005.txt,
vjp-activity-gfx1201-20261005.txt.
