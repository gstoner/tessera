# Native folded-scale common gate

Owner: ROCM-MXFP4-W4A8-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-FOLDED-COMMON-GATE-20261019.
Base: merged PR989 / a66ebd679. Delivery remains pending a clean full CPU rerun, remaining audit gates and public CI/review.

## Implementation and numerical contract

The ROCm native Tile consumer computes the same per-element FP32 combined
scale and normal result. The vector-scale path tests whether every element
of a fragment has a finite, nonzero combined scale. If true it returns those
same normal results through one common branch. Otherwise it uses every
original per-element predicate and FP64 exceptional-product recovery.
Regular elements of a mixed fragment retain their FP32 rounding points.

The scalar lowering and original scale/store order are preserved. There is no
new dtype, Graph/Tile operation, diagnostic, pass, ABI, physical schedule key or
Python backend body. Graph -> Schedule -> Tile -> ROCm Target -> ROCDL/LLVM ->
HSACO and the checked runtime ABI remain the execution route.

## Exact-device numerical and compiler proof

Tajasaurus: AMD Radeon RX 9070 XT / gfx1201, WSL2; assertions-enabled
LLVM/MLIR 23.1.1 six-backend compiler.
Candidate compiler: 8e5c7f678529f1ab89fe1b0edc5101ecfd066d4d5068b4a7a3b16256ef225a29.
Frozen control: d475845a3ea58ff3623bbc56193b19612553db7cf4ba3367091280ae9ebfd71e.
The source-content manifest binds the tested private prototype to the clean
delivery source; private experiment history is not the PR implementation.

- 315 delivery audit/diagnostic/pass drift tests pass.
- 411 native contract/diagnostic/pass tests pass.
- 165 owning device tests pass across native packages, full/partial wide-scale
  recovery, masked-but-observable result lanes, aligned/misaligned activation
  scales, signed zeros/subnormals, NaN/Inf classification, packed vector-band
  and pitched layouts. NaN classification is checked; NaN payload bits are
  not certified. A forty-case refinement passes both control and candidate,
  including rounded-subnormal mixed fragments whose result differs from
  uniform FP64 widening; 363 combined numerical/ownership/drift cases pass.
- 574 separate composed/public primal/JVP/reverse regressions pass.
- Twelve FP8/MXFP8 KN/NK shape-format controls retain identical Tile, Target,
  image, entry and ABI. This is unchanged-format proof, not format promotion.
- gfx1151 FP16 and gfx1201 FP8 scalar Target IR remain byte-identical.
  gfx1151 execution is not inferred from that artifact equality.

## Paired timing

Each unpacked order uses eleven alternating trials, three shared resident
frames, independent sampled oracle checks, full bitwise comparisons, and
poisoned bracketed/plain graph replay validation. Compiler roles and case
order are reversed together. All physical schedule fields match.
Ratios below are candidate/control device graph latency: below one is faster.
They include graph dispatch and captured kernels; they are not isolated
profiler kernel times. Public complete-call wall samples remain separate.

| Shape M×N×K | Forward order | Reverse order |
|---|---:|---:|
| 200×2048×2048 | 0.9744 | 0.9732 |
| 200×8192×1024 | 0.9803 | 0.9872 |
| 256×1024×1024 | 0.9621 | 0.9621 |
| 256×4096×2048 | 0.9817 | 0.9846 |
| 256×8192×1536 | 0.9798 | 0.9969 |
| 256×8192×5120 | 0.9936 | 0.9946 |
| 256×16384×1536 | 0.9824 | 0.9830 |
| 256×16384×5120 | 1.0039 | 1.0047 |
| 256×24576×1536 | 0.9814 | 0.9803 |
| 512×8192×1536 | 0.9895 | 0.9813 |

The widest long-K row is about 0.4% slower. Other named rows improve by varying
amounts; no universal speedup or Radiance closure is claimed. The raw packets
retain the independent Radiance comparison and its exact module hash/tuning
environment rather than conflating it with the HIP control ratio.

Eight packed-native shape/order cases also pass the public checked launcher,
full bitwise and sampled independent oracle, and poisoned three-frame graph
checks. Candidate/control graph ratios range from 0.9469 to 1.0002.
These use the native packed expanded-memref ABI, not the old hand-emitted HIP
packed benchmark. Shared frame leases close before their allocations retire.

Selected disassembly is bound to the exact timed HSACO and instruction-stream
hashes. Full profiles retain 177 VGPRs, 25,600 bytes LDS and no scratch/spills;
SGPRs rise 29 -> 33. The masked M200 profile retains 163 VGPRs and rises
56 -> 61 SGPRs. All four inspected profile pairs retain their original WMMA,
LDS-wait and barrier counts. Total static branch instructions increase because
the per-element fallback remains. Common-path control-flow reduction is not
hardware-counter stall attribution.

## Rejected experiments and retained evidence

- K64 read grouping: numerically valid, no useful gain; removed.
- Long-K CU mode: within same-image control variation; removed.
- Staging all fragment scale SSA before stores: lower registers, but the
  wider N4096/K2048 profile independently regresses 3.4-3.8%; removed.
- Explicit LLVM IEEE-class intrinsic: all twenty measured arms are
  byte-identical to the original predicate; redundant source removed.

Compressed raw packets and decisions retain these negative results. The
experiments/ compressed patches reproduce only historical C++/ODS experiments against
a66ebd679; they are evidence, not active production configuration.
No unused selector or experimental policy is installed. Read a compressed
packet with gzip -dc FILE.json.gz; reconstruct an experiment in an isolated
base checkout with gzip -dc experiments/NAME.patch.gz | git apply.
Normalized historical decisions are compressed to avoid repeating full
receipts in the review diff; their decoded contents are unchanged.

## Reproduction

From the project root in owning gfx1201 WSL, provide immutable candidate and
control compilers and the pinned LLVM tools. Run the canonical
benchmarks/rocm/record_gfx1201_folded_native_package.py with
--tessera-opt, --native-reference-opt, --share-native-resident-frames,
--hip-graph-windows, --llvm-bin and --trials 11; take shapes from the table.
Use no schedule-difference flags. Reverse both compiler roles and shape order.
An optional Radiance comparison requires the same fingerprinted external
module and captured RADIANCE_MXFP4_WPERM=1 environment.

For the packed evidence:

```sh
python benchmarks/baselines/rocm_folded_common_gate_20261019/record_packed_native_common.py \
  --candidate-opt CANDIDATE --control-opt CONTROL --llvm-bin LLVM_BIN --output OUTPUT
```
The script queries the actual selected GPU name/architecture and records
source, recorder and compiler hashes. It reuses canonical ABI-aware package,
clock, graph and frame owners;
it changes no compiler policy or generated kernel body.

## Backend assessment and remaining scope

ROCm gfx1201: named parity and paired measurements above. gfx1151: scalar
artifact parity only; RDNA3.5 cannot inherit RDNA4 FP8 WMMA execution.
Apple, NVIDIA and x86 do not consume this ROCm conversion pattern; no runtime,
shared IR or physical admission is changed for them.
All five aggregate slices remain open, including wider model quality,
producer/AD/layout integration and ROCm physical performance obligations.
