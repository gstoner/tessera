# gfx1201 native KN W8A8 LDS integration

Owner: ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync: GFX1201-KN-LDS-20261017.

## Compiler boundary

Python/textual Graph -> native verified Schedule -> typed Tile views/fragments
-> ROCm Target -> GPU/ROCDL/LLVM -> HSACO -> checked expanded-memref ABI
executes on Tajasaurus, RX 9070 XT/gfx1201. Python does not generate this
kernel or choose its production schedule.

Native selection chooses a 128x64 tile, eight waves, one K128 buffer and
16-byte LDS padding for compact KN E4M3 W8A8: 128<=M<=256,
1024<=N<=8192 with N divisible by 128, 1024<=K<=4096 with K divisible by
128, and FP32 scale blocks 128x128. FP32 accumulation and FP32/BF16 stores
retain their existing semantics. Existing transposed-A normalization and
out-of-envelope profiles retain their previous route. NK/MX policies are
unchanged. No public dtype, operation or ABI version is added.

B reads contiguous physical N vectors, then scatters into the existing
[Ntile,Kstage] LDS consumer. Whole-N loads use a checked alignment proof;
ragged diagnostics use safe scalar tails. Barriers remain outside lane masks.
Static/projected images, scale groups, padding, double buffering and changed
inputs are checked. There is no host-side weight transpose.

## Numerical and integration evidence

Selections overlap and are not summed:

- 132 W8A8 tests passed before the aligned specialization; 24 targeted cases
  passed after it.
- 210 native Schedule/public/sampler cases pass after production selection.
  Nine out-of-envelope cases retain registers; ordinary JIT changed-input
  replay runs with compilation forbidden.
- 612 post-selection registry/identity/W8A8/MXFP8/MXFP4 cases pass.
- 15 public scale-JVP cases pass, including the newly selected
  128x1024x1024 profile, changed tangents and independent finite differences.
- All 20 production-selected images, entries, ABIs and 256-thread workgroups
  match the measured diagnostic candidates exactly. NK, MXFP8 and M256 folded
  MXFP4 controls remain identical against the frozen baseline compiler.
- Mypy is clean at baseline zero.

The native verifier needed KN W8A8 admission alongside its existing NK
contract. Separate MX restrictions remain. Elided global staging defaults
are read through the canonical Schedule parser.

The complete production compiler-route lane passes 1,600 tests with zero
skips; its execution certificate confirms every selected proof ran. All 33
generated documents, 37 focused publication tests, five public surface
audits and both claim linters pass. The broad CPU lane and public CI remain
pending.

The first compiler-route run had 153 skips because llc was absent from
PATH; the certificate rejected it. The exact existing LLVM 23.1.1
assertions-enabled tool is now exposed through a scratch-owned symlink.
The rejected run and restoration record are retained.

## Timing and rejected alternatives

Ten boundary/interior shapes cover both output types. Initial matched
dispatch gains are 1.37-7.25x against KN registers; corrected paired runs retain
1.33-7.28x. Device-clock and HIP-event windows agree. These are dispatch
windows, not isolated kernel profiler counters.

Counterbalanced completed public calls include staging/transfers/completion
and exclude compile and oracle checks. Twenty-ms windows show ratios
0.959-1.082x; longer 50-ms reversed-order controls on the three non-improving
profiles show 0.998-1.032x. Public cost is roughly flat within observed
variation. No universal end-to-end speedup is claimed.

Whole-N specialization reduces KN VGPRs from 244 to 191 without spills;
the compared NK body uses 192. WMMA and split-barrier counts are unchanged.
These are static disassembly counts, not dynamic conflict attribution.

A row-major LDS prototype passes 24 numerical cases but loses 27-33% in
three paired shapes. It removes 32 LDS byte-store instructions but adds 128 LDS byte-load instructions.
It is rejected; its patch, image hash, resources and raw timings are retained.
Hardware B fragments remain column-major regardless of backing-view order;
the first prototype incorrectly changed that register contract.

The first new host sampler did not balance window order correctly. Its
results are quarantined in kn-envelope-paired-host-invalid-order.json.xz.
A deterministic red/green test proves actual adjacent-pair reversal and
exclusion of oracle time. Only kn-envelope-paired-host.json.xz and
kn-host-reverse-controls.json.xz support paired-host conclusions.

## Reproduction

Use matching assertions-enabled LLVM/MLIR 23.1.1 core/ROCm tools and the
owning gfx1201 runtime. Set TESSERA_OPT/TESSERA_ROCM_OPT and
TESSERA_GFX1201_DEVICE_PROOF=1. Run the named test selections in validation.json.

For a matched register/LDS physical comparison:

    python -m benchmarks.rocm.benchmark_gfx1201_fp8_blockscale \
    --compiler "$TESSERA_OPT" --shape 200,8192,1536 \
    --sweep 16x32:1:-1:kn+bf16 \
    --sweep lds:128x64:8:1:128:16:0:kn+bf16 \
    --windows 9 --min-window-ms 10 \
    --paired-host-windows 10 --paired-host-min-window-ms 20 \
    --sync-key GFX1201-KN-LDS-20261017 --output kn-lds.json

Large raw JSON packets are losslessly compressed as .json.xz; raw-packet-archives.json records both hashes. Read them with Python lzma and json. Raw packets retain source/compiler hashes. source-identity.json binds the
final implementation, tools and runtime libraries; production-image parity
binds earlier measurements to the promoted native kernel. Frozen compiler
warnings are expected A/B controls, never current-source build evidence.

## Remaining work

KN remains 21-37% slower than matched AITER in the three recorded M200 rows.
Broader KN shapes/layouts, MXFP4 M256 attribution, W8A8 performance closure,
general compiled-route/AD integration and the original five-slice goal remain
open. No gfx1151, gfx1200, NVIDIA, Apple or x86 physical proof transfers.
