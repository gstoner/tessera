# Native gfx1201 K16 and odd scale-group defaults

Owner ROCM-FP8-BLOCKSCALE-1 / ROCM-NVFP4-INGEST-1.
Synchronization key: ROCM-K16-SCALE-GROUPS-20261015.

## Native compiler repair

Graph/Schedule already admit positive multiples of the 16-wide FP8 WMMA K
instruction. The native register generator nevertheless required its implicit
two-panel default to divide every group. K16, K48 and K80 therefore failed
before packaging even though explicit one-panel requests were valid.
Three host compiler regressions fail before the fix; five explicit/legacy
controls pass (compiler-red.txt).

The native generator now chooses one panel for odd groups when the option is
implicit. Its measured two-panel default remains for even groups. Explicit
requests retain their divisibility validation, including explicit two-panel
refusal for odd groups. The MLIR option hasValue flag preserves that distinction
through pass cloning. No Python shape selector or backend fast path is added.
No operation, dtype, diagnostic, ABI or pass order is introduced.

The rebuilt core/ROCm tools pass 431 focused compiler/contract/diagnostic/pass
checks. All 102 existing gfx1201 device controls pass. The subsequent filtered
run passes 16 cases: 12 new K16/K48/K80 cases across KN/NK layouts and exact/
random inputs, plus four overlapping existing controls. These check FP64
numerics, isolated partial accumulators, scale joins and actual FP8 WMMA ISA.
Independent cold-process K32/K128 legacy packages have byte-identical Tile,
Target and HSACO identities against the preserved reference compiler.

## Source-preserving diagnostic control

probe.py uses all E2M1 values (including signed zero), per-K16 E4M3 scales and
two independent binary-exact globals. Host-side expansion prepares ordinary
FP8 values/scales; each matmul then follows the existing native semantic Graph
-> Schedule -> Tile -> ROCm Target -> LLVM -> HSACO and checked runtime ABI.
The three synthetic controls have zero elementwise f32-accumulation/BF16-store
bound violations on RX9070XT/gfx1201.

| M,N,K | HIP graph execution/dispatch median | Completed host median |
| --- | ---: | ---: |
| 7,19,32 | 4.64 us | 0.849 ms |
| 128,128,128 | 15.12 us | 0.699 ms |
| 129,144,128 | 15.87 us | 0.746 ms |

All three selected bodies are register staged. HIP graph dispatch includes
dispatch overhead; these are not pure kernel or optimization A/B figures.
Raw samples retain their spread. ISA is stored once per image, compressed and
hashed. The measurement Python source is pinned by source-k16-control.json
at source epoch 92a1fcc02; native C++ arithmetic was built from 8873a4b64.

This proves a required K16 consumer foundation. It does not implement a native
NVFP4 expansion producer, preserve packed four-bit transport, test arbitrary
model globals, establish model-quality acceptance, or promote a selector.
The earlier captured-input quality gap remains open. FP8, MXFP8 and MXFP4
evaluation requirements and broader W8A8/MXFP4 performance obligations remain.

## Reproduce

On the owning gfx1201 host, source the matching compiler/runtime environment
and run probe.py with an output JSON path. It requires TESSERA_OPT and LLVM_BIN,
uses seed 151091, verifies numerics before timing, and records compiler/source/
image/ISA identities. Run tests/unit/test_rocm_scale_group_default.py and the
owning tests/device/rocm/test_fp8_blockscale_w8a8.py suite for permanent guards.
