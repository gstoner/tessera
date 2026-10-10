# gfx1201 MXFP8 K64 staging candidate

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-MXFP8-K64-2026-10-06.

The explicit lds_k64 frontend intent selects native MLIR Graph → Schedule →
Tile → ROCm lowering. Python authors semantic IR and binds packages; it does
not construct the kernel. K64 LDS slabs contain two independent zero-initialized
K32 WMMA partials, joined with their own E8M0 scales in ascending group order.
K must be a positive multiple of 64. The checked runtime rejects partial slabs.
Image keys retain the K64 recipe while reusing runtime K across whole slabs.

## Engineering decision

Initial candidate stage: automatic selection was unchanged. The first 128x128 long-K candidate spilled
to scratch and was rejected. three_formats.json is an initial screening run
with overlapping regression activity, not promotion evidence.
The retained explicit candidate uses 128x64. narrow_panel.json was run after
the regression process completed, with rotated/reversed five-window HIP graph
replays, bracketed device-clock/event checks, and separate host end-to-end samples.

| M,N,K | K32 control device µs | K64 candidate device µs |
|---|---:|---:|
| 200,1024,128 | 8.666 | 8.401 |
| 200,2048,1536 | 37.331 | 30.847 |
| 256,4096,5120 | 161.312 | 160.896 |

These initial candidate observations preceded the independent repetitions below.
The long-K difference is effectively a tie. FP8 K128/N128, FP8 K32/N1,
standard MXFP8 and folded MXFP4 controls use matched source operands and their
own stated numerical contracts; folded MXFP4 remains explicitly approximate.
All numerical checks passed before and after timing. They establish no AITER
or Radiance performance comparison and no isolated LDS cost attribution.

## Validation and remaining work

39 initial focused checks and 354 device/diagnostic/pass-metadata regressions
passed. The final narrow-panel rerun passes 40 checks, including the partial-K guard.
The measured long-K K32 and narrower K64 images both have zero static scratch
load/store instructions.
ISA, compiler/source digests, raw windows, launch geometry and numerical bounds
are retained in JSON and .s files. Graphify is unavailable on the WSL host.

Repeat independent process orders and broaden shapes before choosing a bounded
automatic rule. Dynamic capacity/residency, persistence, deep fragment buffering,
and general prologue/epilogue integration remain separate open work.
gfx1151 has no RDNA4 FP8 WMMA; Apple/x86/NVIDIA require their own schedules
and exact-device evidence. No physical schedule or proof transfers.


## Independent repetitions and retained native selection

repeat_forward.json and repeat_reverse.json contain separate processes with
reversed construction/calibration, arm and shape order. Seven shapes and six
format/recipe arms passed numerical checks in each process (84 timed arms).
Measured K64 device-time reductions relative to forced K32 LDS:

| M,N,K | Forward order | Reverse order |
|---|---:|---:|
| 128,4096,1024 | 16.65% | 17.27% |
| 200,2049,1536 | 10.37% | 10.38% |
| 200,2048,2048 | 17.72% | 20.77% |

128x128 control panels did not gain: measured differences ranged from 0.12%
to 4.17% slower. The long-K case also did not gain. The native selector retains
their K32 recipe. The low-K/global seed remains unchanged.

The retained automatic rule applies only when the pre-existing gfx1201 NK LDS
selector chooses 128x64 and K lies in [1024,2048] with whole K64 slabs.
The existing M>=128, K>=1024 and >=64-workgroup LDS gate still applies.
This is a bounded MXFP8 schedule decision, not an FP8/MXFP4 dtype or format
promotion. Explicit lds continues to expose the original K32 recipe.

125 exact-device/contract checks pass, including automatic K1024/1088/1536/1984/
2048, fixed old-profile image reuse, isolated groups, extreme scales/NaNs,
ragged shapes, BF16 output, distinct keys and pre-HIP refusal of partial slabs.
automatic_route.json is the final numerical/timing check of automatic dispatch
against explicit K64 and K32 controls. FP8 and folded MXFP4 numerical controls
remain included. This increment closes the named multi-group staging gate;
general persistence, deep buffering, native prologue/epilogue, cache/layout
families and the M256 MXFP4/Radiance attribution requirement remain open.
