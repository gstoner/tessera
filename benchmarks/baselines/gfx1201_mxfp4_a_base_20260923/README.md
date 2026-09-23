# GFX1201 packed MXFP4 A-stage address ablation

Owner `ROCM-MXFP4-W4A8-1 / IKF-1`; sync
`GFX1201-PACKED-A-BASE-2026-09-23`. This opt-in compiler variant separates
the CTA-uniform A tile base from its lane-local clamped row offset. The packed
weight ABI, K32 numerical policy, output layout, and default selector are
unchanged. A separate standalone vector-pair rerun closes two PR #822 review
findings: its v4 packet includes the immediate packed-permute control and its
candidate row carries the vector-pair sync key.

On Tajasarus RX 9070 XT/gfx1201, five targeted A-base cases passed for
ragged N48/N80, lossless and lossy folds, all E2M1 codes, exponent deltas,
and reserved E8M0 zero blocks. Before timing, the benchmark required an
independent sampled exact reference and bitwise BF16 agreement across the
exact K32 route, packed variants, and pinned Radiance comparator. Eleven
interleaved HIP-event trials of twelve iterations each gave kernel medians:

| M×N×K | Packed permute | Hoisted A base | Radiance |
| --- | ---: | ---: | ---: |
| 256×5120×8704 | 184.5 µs | 180.7 µs | 137.9 µs |
| 1024×17408×5120 | 1163.5 µs | 1178.5 µs | 892.3 µs |

The smaller shape improves 2.1%, while the wide shape regresses 1.3%.
Selected-symbol ISA digests differ and static instruction count falls by
five, but both variants use 110 VGPRs, 54 SGPRs, 25,600 LDS bytes, no spills,
and 75 static `s_wait_loadcnt` instructions. Thus the source-level base hoist
does not reduce register pressure or waits. Static counts do not establish
dynamic address cost or stall attribution. The candidate remains manual and
cannot be promoted.

The standalone [vector-pair review packet](vector_pair_review_evidence.json)
shows both shapes include `tessera_packed_batched_b_permute` and that the
vector-pair rows now carry `GFX1201-PACKED-VECTOR-PAIR-2026-09-23`. It still
improves only the smaller shape; the wide-shape regression remains. The
[A-base packet](evidence.json) and review packet bind timed Tessera payload
SHA-256 to selected-symbol ISA and record source/benchmark hashes. Their
`source_revision` is the merged-main base `fa44b468`; hashes identify the
modified sources in the isolated device worktree. The prior v4 packet remains
historical, including its incorrect row-level sync key.

Next isolate 32-bit lane-local A offsets from the now-measured base hoist,
under an explicit dimension bound, then compare address ISA and occupancy
without relaxing exact K32 correctness or timing methodology.
