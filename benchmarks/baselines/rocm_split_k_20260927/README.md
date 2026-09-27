# ROCM-SPLIT-K-1 follow-on: gfx1201 split-K slice sweep on the device clock, 2026-09-27

Sync key `GFX1201-LANES-2026-09-27`. Supersedes the **selection rule** measured in
[`../rocm_split_k_20260926/`](../rocm_split_k_20260926/README.md). That packet stays
as history: it is host-wall timing of one shape, and its numbers are not
comparable with these (different clock, different window protocol).

## Host, trees, source

- **Host:** Tajasarus, AMD Radeon RX 9070 XT (gfx1201), queried at record time
  (`hipGetDevicePropertiesR0600`, stored in every run as `identity`). WSL2
  `/dev/dxg`, no `/dev/kfd`, ROCm 10.0 / HIP 7.15.
- **Source:** both sweeps at commit `ca2653ec` with a clean tree (`git_dirty:
  false` in `sweep.json`; every packet's `source.worktree_dirty` is false). The
  rule change (`077eabe7`) came **after** these measurements and does not
  change any image in them. Every timed image was compiled by the Tile IR route
  at that commit, so the evidence is valid for the new rule's selections.
- **Tree:** a private worktree (`~/programming/tessera-w-splitk`) with its own
  Release `build/` (NDEBUG LLVM/MLIR 23.1.1). `tessera_opt` and its sha256 are
  in `sweep.json`; `TESSERA_OPT`, `PYTHONPATH` and `PATH` were re-pointed at that
  worktree, not the shared checkout.
- **Recorder:** `benchmarks/rocm/record_split_k_sweep.py` (see its docstring).
  Commands:

  ```bash
  # sweep_20ms/ : 16 shapes x {fp16, bf16} x S in {1,2,4,8,16,32}
  flock /tmp/tessera-timing.lock python benchmarks/rocm/record_split_k_sweep.py \
      --output-dir /tmp/w-splitk-sweep
  # admission_50ms/ : the 7 shapes with an unadmitted row, longer windows
  flock /tmp/tessera-timing.lock python benchmarks/rocm/record_split_k_sweep.py \
      --output-dir /tmp/w-splitk-sweep50 --window-ms 50 --slices 1,2,4,8,16 \
      --shapes 16x64x2048,16x128x2048,16x256x7168,32x256x7168,16x2048x768,32x128x512,64x128x4096
  ```

## Timing source and admission

- **Clock:** `device_wall_clock_ns` from the compiler-built
  `--tessera-device-clock-span` marker (`llvm.readsteadycounter`, ticks at
  `hipDeviceAttributeWallClockRate`). Two marker launches bracket N launches of
  the unmodified image, and a HIP event pair witnesses each window.
- **Protocol:** 3 fresh processes x 9 interleaved rounds. Each round has one
  plain window and one bracketed window per variant, in alternating order. N is
  sized from a host-wall estimate of the fastest variant toward the target
  (20 ms / 50 ms). The estimate undershoots, so the shortest measured windows
  are 12.2 ms and 20.8 ms. Both clear the 5 ms ROCm per-window minimum, and no
  packet carries a window refusal.
  - A split iteration is **both** launches (partial + ordered reduce).
  - The fp32 workspace is allocated once, outside timing. `runtime.launch`
    allocates it per call, and that cost is excluded, as in the 09-26 packet.
- **Admission:** each variant's windows become a
  `tessera.profiler_rocm_packet.v1` built by `build_rocm_profiler_packet`
  (`packets.jsonl.gz`, one per run x variant). That builder re-derives the
  route (`device_clock_witness`) and the per-window refusals itself. A row is
  **admitted** when that packet is `eligible_for_promotion`.
  - Two reasons refuse rows here, both from the two-sided marker-overhead gate
    (bracketed vs plain event within 5%): `INSTRUMENTATION_CHANGED_THE_KERNEL`
    and `INSTRUMENTATION_OVERHEAD_EXCEEDED`. No row was refused for clock/event
    disagreement or a short window.
  - The refusals are sporadic across runs and sweeps, not tied to a variant.
    Every selection below has at least one sweep in which both it and its
    unsplit baseline are admitted in 3/3 runs, **except 16x2048x768**. That row
    is admitted in 1-2 of 3 runs in both sweeps (see below).
- **Correctness first:** every variant is checked against an f64 reference
  (max relative error <= 1.3e-5 over all rows) and against the unsplit kernel.
  Two launches of every variant are compared **bit for bit** with the output
  poisoned in between (`bit_identical_reruns`: true on every row).

## Result: the rule, old vs new

Old rule (09-26): split when tiles < 32 WGPs, `S = ceil(32/tiles)`.

New rule: split when `2 x tiles <= 256`, and take the largest power-of-two
`S <= min(32, 256 / tiles)` whose slices are whole 32-wide macro K blocks of at
least 256.
- It was chosen because **every selection it makes was measured positive in
  both storages**, not because it hits each shape's peak.
- It is within ~10% of the best measured S on most shapes. It deliberately
  leaves S=16/S=32 gains on some shapes: 16x8x4096 S=32 is 13.2x vs 10.2x, and
  48x96x1536 S=16 is 2.85x vs 2.60x.
- On 48x96x1536 the axis is not monotone (S=8 is 2.18x, between S=4 and S=16),
  which is exactly why a peak-chasing rule was not adopted.

Speedup = median over runs of each run's median paired per-round ratio
(unsplit device-clock time / split device-clock time).

| shape (MxNxK) | tiles | old S | new S | dtype | 20 ms sweep: new S vs unsplit (runs admitted: unsplit, new) | 50 ms re-run | best measured (20 ms) |
|---|---|---|---|---|---|---|---|
| 16x8x4096 | 1 | 16 (10.23x) | 16 | bf16 | 10.23x, 27/27 rounds (3/3, 3/3) | — | S=32 13.24x |
| 16x8x4096 | 1 | 16 (10.22x) | 16 | fp16 | 10.22x, 27/27 rounds (3/3, 3/3) | — | S=32 13.22x |
| 16x64x2048 | 4 | 8 (3.88x) | 8 | bf16 | 3.88x, 27/27 rounds (0/3, 3/3) | 4.44x, 27/27 rounds (3/3, 3/3) | S=16 4.02x |
| 16x64x2048 | 4 | 8 (3.63x) | 8 | fp16 | 3.63x, 27/27 rounds (0/3, 3/3) | 4.62x, 27/27 rounds (3/3, 3/3) | S=16 3.81x |
| 16x128x2048 | 8 | 4 (3.16x) | 8 | bf16 | 3.65x, 27/27 rounds (3/3, 3/3) | 4.50x, 27/27 rounds (3/3, 3/3) | S=16 3.79x |
| 16x128x2048 | 8 | 4 (3.15x) | 8 | fp16 | 3.65x, 27/27 rounds (2/3, 3/3) | 4.52x, 27/27 rounds (3/3, 3/3) | S=16 3.84x |
| 16x256x256 | 16 | 1 | 1 | bf16 | unsplit (3/3) | — | none (all lose) |
| 16x256x256 | 16 | 1 | 1 | fp16 | unsplit (3/3) | — | none (all lose) |
| 16x256x2048 | 16 | 2 (2.17x) | 8 | bf16 | 3.37x, 27/27 rounds (3/3, 3/3) | — | S=16 3.49x |
| 16x256x2048 | 16 | 2 (2.18x) | 8 | fp16 | 3.40x, 27/27 rounds (3/3, 3/3) | — | S=16 3.51x |
| 16x256x7168 | 16 | 2 (2.77x) | 16 | bf16 | 7.43x, 27/27 rounds (0/3, 3/3) | 9.17x, 27/27 rounds (3/3, 3/3) | S=16 7.43x |
| 16x256x7168 | 16 | 2 (2.78x) | 16 | fp16 | 7.45x, 27/27 rounds (0/3, 3/3) | 9.14x, 27/27 rounds (3/3, 3/3) | S=16 7.45x |
| 16x768x2048 | 48 | 1 | 4 | bf16 | 2.44x, 27/27 rounds (3/3, 3/3) | — | S=8 2.48x |
| 16x768x2048 | 48 | 1 | 4 | fp16 | 2.43x, 27/27 rounds (3/3, 3/3) | — | S=8 2.46x |
| 16x2048x768 | 128 | 1 | 2 | bf16 | 1.24x, 27/27 rounds (3/3, 1/3) | 1.12x, 27/27 rounds (3/3, 1/3) | S=2 1.24x |
| 16x2048x768 | 128 | 1 | 2 | fp16 | 1.20x, 27/27 rounds (3/3, 1/3) | 1.12x, 25/27 rounds (3/3, 2/3) | S=2 1.20x |
| 16x2048x7168 | 128 | 1 | 2 | bf16 | 1.30x, 27/27 rounds (3/3, 3/3) | — | S=32 1.33x |
| 16x2048x7168 | 128 | 1 | 2 | fp16 | 1.36x, 27/27 rounds (3/3, 3/3) | — | S=2 1.36x |
| 32x128x512 | 16 | 2 (1.16x) | 2 | bf16 | 1.16x, 26/27 rounds (2/3, 3/3) | 1.23x, 25/27 rounds (3/3, 3/3) | S=16 1.22x |
| 32x128x512 | 16 | 2 (1.18x) | 2 | fp16 | 1.18x, 26/27 rounds (2/3, 3/3) | 1.30x, 24/27 rounds (3/3, 2/3) | S=4 1.24x |
| 32x256x7168 | 32 | 1 | 8 | bf16 | 4.85x, 27/27 rounds (1/3, 3/3) | 6.17x, 27/27 rounds (3/3, 3/3) | S=8 4.85x |
| 32x256x7168 | 32 | 1 | 8 | fp16 | 4.80x, 27/27 rounds (0/3, 3/3) | 6.11x, 27/27 rounds (3/3, 3/3) | S=8 4.80x |
| 32x1536x4096 | 192 | 1 | 1 | bf16 | unsplit (3/3) | — | S=4 1.00x |
| 32x1536x4096 | 192 | 1 | 1 | fp16 | unsplit (3/3) | — | S=2 1.01x |
| 48x96x1536 | 18 | 2 (2.15x) | 4 | bf16 | 2.60x, 27/27 rounds (3/3, 3/3) | — | S=16 2.85x |
| 48x96x1536 | 18 | 2 (2.14x) | 4 | fp16 | 2.60x, 27/27 rounds (3/3, 3/3) | — | S=16 2.85x |
| 64x64x1024 | 16 | 2 (1.75x) | 4 | bf16 | 1.96x, 27/27 rounds (3/3, 3/3) | — | S=16 2.12x |
| 64x64x1024 | 16 | 2 (1.75x) | 4 | fp16 | 1.97x, 27/27 rounds (3/3, 3/3) | — | S=16 2.10x |
| 64x128x4096 | 32 | 1 | 8 | bf16 | 3.92x, 27/27 rounds (3/3, 3/3) | 5.02x, 27/27 rounds (0/3, 3/3) | S=16 3.98x |
| 64x128x4096 | 32 | 1 | 8 | fp16 | 3.85x, 27/27 rounds (3/3, 3/3) | 4.99x, 27/27 rounds (0/3, 3/3) | S=16 3.99x |
| 64x512x2048 | 128 | 1 | 2 | bf16 | 1.80x, 27/27 rounds (3/3, 3/3) | — | S=2 1.80x |
| 64x512x2048 | 128 | 1 | 2 | fp16 | 1.82x, 27/27 rounds (3/3, 3/3) | — | S=2 1.82x |

What bounds the rule (the negative side, all in `sweep_20ms/sweep.json`):

- **Past the workgroup target.**
  - 32x1536x4096 (192 tiles): S=2 is neutral (0.985x / 1.009x, 9-16 of 27
    rounds), and S >= 8 loses (0.87-0.90x).
  - 16x2048x768 (128 tiles): S=8 loses (0.85-0.86x, tiles x S = 1024).
  - 64x512x2048 (128 tiles): S=32 loses (0.90x, tiles x S = 4096).
  - The rule never selects any of these.
- **The 256-per-slice guard, now measured at its boundary.**
  - 16x256x256 loses at every S (0.89-0.92x, 0/27 rounds): 128-wide slices.
  - 32x128x512 at S=2 (two 256-wide slices) gains 1.16-1.18x.
  - Narrower slices were positive at larger K (64x64x1024 S=16, 16x64x2048
    S=32). The guard is therefore conservative there. It was kept rather than
    generalized from two small-K shapes.
- **The S=32 cap** is the largest count measured, and not a demonstrated
  optimum: 16x8x4096 still gains from S=16 to S=32.

## What these numbers are and are not

- **Absolute times depend on window length.** In the 50 ms re-run, the long
  unsplit kernels ran 25-35% slower per launch than in the 20 ms sweep: the
  unsplit 16x256x7168 took 61.2 us vs 46.3 us, and 64x128x4096 took 34.2 us vs
  27.3 us. The split variants barely moved, so the ratios grew.
  - One untested hypothesis: a low-occupancy kernel held for longer lets the
    GPU settle into a lower clock state. This WSL2 host has no counters to
    confirm it.
  - **Read ratios within one sweep; never compare absolutes across the two.**
  - The 20 ms sweep is the conservative one, and the table's decision column
    uses it.
- **Why splitting helps past one workgroup per WGP is not measured.** Several
  single-wave workgroups co-residing on a WGP's four SIMDs is one hypothesis;
  the 09-26 packet recorded the same one.
- **16x2048x768 S=2 is the weakest selection.**
  - It gains 1.12-1.24x and wins in 25-27 of 27 rounds.
  - Its packets pass the marker-overhead gate in only 1-2 of 3 runs, in both
    sweeps.
  - The ratio is positive on the device clock in every run. Admission of that
    row is partial, and it is recorded as such.
- **Not proven.**
  - gfx1151: the rule never splits there. There is no Princess-Luna run, and
    gfx1201 evidence does not transfer.
  - fp8/int8 storages: never split.
  - Dynamic shapes and the LDS body.
  - The per-call workspace allocation cost of `runtime.launch`.
  - M > 64.
- **No production promotion is claimed.** These rows are admissible
  device-clock evidence for the slice rule's selections. They are not a
  promotion packet for a route.
