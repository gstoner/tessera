# GFX1201 packed MXFP4 bounded A-offset ablation

Owner `ROCM-MXFP4-W4A8-1 / IKF-1`; sync
`GFX1201-PACKED-A-OFFSET32-2026-09-23`. This opt-in producer keeps the
CTA-uniform A base wide but computes the clamped lane-local byte offset in
signed 32-bit arithmetic. The maximum supported K is
`floor((INT32_MAX - 48) / 255)`; larger K is rejected before compilation.
The versioned packed-weight ABI, exact per-K32 oracle, and default selector
are unchanged.

On Tajasarus RX 9070 XT/gfx1201, five focused ragged/lossy/all-code cases
passed. The [v6 packet](evidence.json) records an independent sampled exact
FP32-dequantized reference and bitwise BF16 agreement among exact K32,
packed controls, the candidate, and pinned Radiance before timing. Eleven
interleaved HIP-event trials of twelve iterations each gave kernel medians:

| M×N×K | Packed permute | 64-bit A base | 32-bit local offset | Radiance |
| --- | ---: | ---: | ---: | ---: |
| 256×5120×8704 | 186.9 µs | 188.8 µs | 184.3 µs | 140.1 µs |
| 1024×17408×5120 | 1163.3 µs | 1184.4 µs | 1186.4 µs | 892.4 µs |

The candidate is 2.4% faster than the A-base control on the smaller shape,
but 0.2% slower than that control and 2.0% slower than packed permute on
wide N. Radiance remains about 1.32× faster on both shapes. The selected
symbol shrinks from 4,392 to 4,358 static instructions; `v_mul_lo_u32`
falls 138→130 and `v_add_co_u32` 157→153. Both variants retain 110 VGPRs,
54 SGPRs, 25,600 LDS bytes, no spills, and 75 static `s_wait_loadcnt`.
These selected-symbol counts bind to the timed HSACO payloads, but are not
dynamic A-stage counters or stall attribution. Reduced integer ISA has not
translated into a reproducible wide-shape performance gain, so selection
remains closed.

Review correction: this historical v6 packet's `integer_alu_isa` field also
contains floating-point mnemonics such as `v_mul_f32_e32` and `v_mul_f64_e32`.
Its aggregate is not an integer-instruction count. The benchmark parser now
excludes floating-point operand types; this immutable packet is not rewritten.
The named integer mnemonic comparisons above remain valid.

The packet's `source_revision` is PR #823's base `cd758719`; source hashes
bind the modified files in the isolated device worktree. This is a stacked
experiment while #823 remains open. A next experiment should target the
remaining wide-N producer traffic or K-step scheduling, measuring one
change at a time against the same exact K32 oracle and matched Radiance.
