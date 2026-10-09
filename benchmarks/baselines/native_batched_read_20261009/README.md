# Native batched program readback — gfx1201

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization ROCM-BATCHED-PROGRAM-READBACK-20261009.
Dependent on PR909's continuous mapped reverse integration.

## Contract

The additive `tessera_rocm_program_read_many` ABI validates the whole returned
destination frame before enqueue, copies into retained native staging, completes
the owner stream once, and publishes independent caller outputs only after
success. It rejects overlapping destinations, duplicate/private slots, wrong
byte counts, stale generations and device/context mismatches. Copy/completion
failure quarantines the owner without publishing partial caller results.
The legacy single-read symbol remains supported; Python only marshals outputs.

Native preparation reserves aggregate returned-output staging. Automatic pinned
selection bounds aggregate output bytes at 8 MiB. Idle-cache accounting includes
pageable input snapshots and returned-output capacity as well as pinned slabs;
this can change retention near the existing 128 MiB cache limit. No GPU kernel,
physical schedule, image identity or numerical policy changes.

## Proof

Source-built full HIP provider on RX 9070 XT / gfx1201: 598 public f32/FP8/MXFP8
map/reverse checks, 16 forced-pinned mixed-map checks, and 28 NVFP4 ingest/MXFP4
producer-consumer regressions pass. Host WSL: 18 compiled production-body tests
prove serial versus batched completion counts, pre-enqueue refusal and failure
quarantine; 149 binding/ABI/artifact integration tests pass.
Delivery gates: 325 drift/lifecycle and 14 recorder census checks pass; CI-scope
Ruff and zero-error mypy pass. Full CI unit selection: 19,919 passed, 9,559
skipped, two existing generic scaled_matmul batching/transpose closure failures
in 350.22 seconds. No closure state or assertion is weakened.
The host shim is not GPU evidence. Owning pytest emits an existing unknown
timeout configuration warning.

## Measurement

`benchmarks/rocm/record_native_batched_read.py` performs nine rounds alternating
arm order with identical provider, package and images. Read-only latency,
completed public native_backward calls, and native program event timing are
separate domains. Independent numerical checks run before and after each arm.
Warm public calls forbid compiler subprocesses. Packets record live inventory
and hashes of providers, compilers, source and oracles.

| Packet | Case | Serial read ms | Batched read ms | Read ratio | Public ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| gfx1201.json | f32_four_gradients | 0.525320 | 0.367227 | 0.699 | 0.857 |
| gfx1201.json | fp8_two_gradients | 0.295837 | 0.231250 | 0.782 | 0.972 |
| gfx1201.json | fp8_single_gradient_control | 0.146786 | 0.145360 | 0.990 | 0.999 |
| gfx1201.json | fp8_primal_control | 0.158475 | 0.159488 | 1.006 | n/a |
| gfx1201.json | mxfp8_primal_control | 0.158484 | 0.161042 | 1.016 | n/a |
| gfx1201_pinned.json | f32_four_gradients | 0.373746 | 0.106124 | 0.284 | 0.729 |
| gfx1201_pinned.json | fp8_two_gradients | 0.195642 | 0.102817 | 0.526 | 0.938 |
| gfx1201_pinned.json | fp8_single_gradient_control | 0.098342 | 0.098204 | 0.999 | 1.005 |
| gfx1201_pinned.json | fp8_primal_control | 0.140376 | 0.105261 | 0.750 | n/a |
| gfx1201_pinned.json | mxfp8_primal_control | 0.105789 | 0.107081 | 1.012 | n/a |

Ratios are batched/serial, not kernel speedups. Automatic-mode multioutput read
latency improves 22–30%, with public-call improvement 3–14% in these cases.
Single-output controls use the same old native read path in both arms; they
cannot establish a batched-ABI speedup. Forced-pinned FP8 primal control in the
first packet varies by 25%; preserve this packet and verify with a repeat
before interpreting pinned performance. The retained repeat packet
`gfx1201_pinned_repeat.json` restores FP8 primal ratio to 0.997 and reproduces
multioutput read ratios 0.284 (four gradients) and 0.518 (two gradients),
with public-call ratios 0.709 and 0.947. The single-gradient read control varies
by 10.5% in that repeat despite using identical transport, so timings are
observations with host noise, not a universal guarantee. No default pinned-policy promotion.

## Remaining work

gfx1151 needs its own consumer and exact-device evidence. SM120 uses CUDA
resident ownership, Apple uses Metal completion, and x86 uses CPU output
ownership; this HIP symbol supplies no parity claim for those backends.
Generic scaled_matmul batching/transpose closure remains open, including the
two existing full-unit closure failures. This transport slice does not close
the original five-slice program.
