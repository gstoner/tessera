# Native static leading-map integration — 2026-10-07

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync key DEEP-LEADING-MAPS-2026-10-07.

Public gfx1201 maps now retain arbitrary positive static leading depth with
matching matrix/scale policies. Graph types preserve the entire prefix;
native verification, Schedule projection and flattened checked ABI own execution.
The obsolete rank-four native scale-transpose ceiling is removed. The target
capability represents a minimum rank of two explicitly. Mixed/nonleading axes,
dynamic/composed graphs, packed/NVIDIA nested maps and encoded-scale AD remain
outside this extension; generic closure states remain unchanged.

Tests found 48 device failures at the old rank capability/native reverse gates.
After repair, 60 owning gfx1201 cases pass FP8/MXFP8 primal, scale JVP, serial
and wave scale VJP at three/four maps, both matrix orientations, three policies,
and compiler-free changed-input replay with unchanged numerical bounds.

Host evidence: initial 106 frontend tests; 83 capability gates; 66 reverse and
deep-bound tests; 474 shared dtype/op/diagnostic/pass/NVIDIA policy gates;
89 native transpose export/image/semantic regressions. Reverse now checks source
constraints before capture. Fifteen controls cover invalid and valid bounds,
argument binding and stale receipt clearing across Apple, x86, gfx1151,
gfx1201 and SM120. This ordering proof is host-only.

Sibling owning RTX 5070 regression: 72 public multi-result attention/VJP cases
and 12 host NVFP4 projection tests pass together; 11 owning public NVFP4 map/
orientation/geometry/bound cases pass separately. gfx1151, Apple and x86
deeper scale-map physical execution remain follow-up required.

Eighteen controlled paired timing rows use the same six logical planes under
prefixes [2,3], [2,1,3], [1,2,1,3]. Independent float64 numerics precede/follow
timing and changed-cotangent replay; compiler subprocesses are forbidden warm.
Public serial medians span 3.65-10.77 ms; wave medians 1.18-1.53 ms.
Native event windows are separate multi-kernel submissions: wave medians span
0.00188-0.00911 ms. These are distinct measurement domains, not isolated-ISA
or full-model speedups. Serial remains default; no selector promotion.

Owning gfx1201 is RX 9070 XT, UUID GPU-28d9e7efbf2ef716 on Tajasaurus.
Compiler built on Super-Bear with LLVM/MLIR 23.1.1 and configured CUDA 13.4.59;
immutable tools replay on Tajasaurus with the isolated native HIP owner.
Raw device probe, source/tool hashes, initial failures and passing logs remain
in this packet. Generic AD/layout/dynamic/performance/full-unit and PR delivery
remain open.
