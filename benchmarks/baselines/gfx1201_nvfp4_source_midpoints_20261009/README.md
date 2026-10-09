# Native source-unit midpoint comparisons on gfx1201

Owner ROCM-NVFP4-INGEST-1; sync ROCM-NVFP4-SOURCE-MIDPOINTS-20261009.
Recorder: benchmarks/rocm/record_nvfp4_winner_codes.py.
Control: PR914 a9bec72ad0f48f8a7522b77c238ff789ebc6457c.
Candidate parent: PR915 339c31e41a4f566a7604e633ffe888f8b0687bbd.
The candidate changes native MLIR emission, not the Python numerical backend.

## Exact comparison law

For bounded exponent e in [-126,127], each midpoint theta times 2^e is an
exact normal f64 dyadic number. In the neighborhood of each midpoint,
abs(value) times 2^-e is also exact. Underflow/overflow can only occur outside
all midpoint decisions and cannot invert their strict ordered comparison.
Thus abs(value)*2^-e > theta iff abs(value) > theta*2^e. NaN remains unordered;
infinity remains above every finite threshold. The sign comparison, candidate
SSE arithmetic/reduction order, nine trials, exponent ties and zero-code guard
are unchanged. The midpoint threshold is shared by 32 values per candidate.

## Oracle correction and numerical gates

The new lowest-exponent fixture initially violated the existing source-scale
range before launch; its 4/1 scale pair now preserves that boundary while
keeping every K32 block admissible. The upper case exposed premature fp32
reconstruction in the reference energy/SQNR calculation. An explicit fp64
reference decode now retains valid E2M1/E8M0 values beyond fp32 range. The
existing default fp32 decode, physical storage, numeric policy and runtime ABI
are unchanged. This is oracle precision, not target fp64 matmul support.
No tolerance is relaxed and neither endpoint was removed.

Matching full LLVM/MLIR23.1.1 tools: 20 native Graph/target checks pass.
Reference/default-compatibility gates: 24 pass. Image/storage/diagnostic/pass
regressions: 322 pass, seven hardware skips. Owning RX 9070 XT/gfx1201:
60 converter/public/bounded/resident checks pass, including below/at/above all
seven strict midpoints at exponents -126,-64,0,64,120. Another 153 FP8/MXFP8/
MXFP4 owning-device checks pass. Device pytest warns that its timeout plugin is
unavailable; no enforced-timeout claim is made.

Both matched sweeps compile the ordinary frontend three-stage converter ->
folded storage -> FP8 activation matmul through native Graph/Schedule/Tile/
Target/LLVM/HSACO. The same operands/provider and identical storage/consumer
images are used. Codes, exponents, f64 statistics, stored buffers and output
are bitwise equal between arms, with independent oracles around all windows.
The later public calls change activation rows/scales and forbid compiler and
eager execution. Public samples are compiler-warm; the first may prepare a
native frame. They include input checks/uploads, native execution, synchronization
and readback and are retained as characterization, not a wall speedup claim.

## Measured domains

Each native stage has seven AB/BA rounds, 128 repeats. Direct/captured samples
are separate; the captured window has one host submission. Ratios are control
median / candidate median, so below 1 means candidate is slower.

| Packet | M,N,K | Converter captured ratio | Native program captured ratio | Public wall ratio |
| --- | --- | --- | --- | --- |
| nvfp4-threshold-comparison.json | [256, 64, 1024] | 1.0810 | 1.0646 | not measured |
| nvfp4-threshold-comparison.json | [256, 512, 1024] | 1.1084 | 1.0868 | not measured |
| nvfp4-threshold-comparison.json | [256, 1024, 4096] | 0.7367 | 1.0778 | not measured |
| nvfp4-threshold-isa-public.json | [256, 64, 1024] | 1.0815 | 1.0646 | 1.0368 |
| nvfp4-threshold-isa-public.json | [256, 512, 1024] | 1.0884 | 1.0733 | 0.9207 |
| nvfp4-threshold-isa-public.json | [256, 1024, 4096] | 1.0541 | 1.0801 | 1.0241 |

Full native program gains repeat at roughly 6–9% across the measured profiles.
The first large standalone-converter sweep regressed; the second improved.
Public wall results are mixed. Different stage windows are not simultaneous
clock states and cannot be summed. No hardware counters or fixed-frequency
certificate exists, and no universal performance or model-quality claim is made.

Static disassembly counts: FP64 multiplies 1065 -> 831; VGPRs stay 256;
private-segment metadata 180 -> 128 bytes; image size 135704 -> 123800 bytes.
These are compiled metadata/static opcode counts, not dynamic issue counts,
measured occupancy or unique bottleneck attribution. Multiplication, scratch
and code-size changes are correlated by this one native comparison rewrite.
V_LDEXP_F64 exists in the RDNA4 JSON instruction archive; the original already
emits it and the candidate changes its static count.

Tool, image, input, recorder, runtime and live HIP device identities are in the
packets and source witness. The retained-control source-age warning is expected;
no file timestamps or old packet hashes were patched. Rebuild both tools from
the pinned control and candidate, use the same leaf-ready native provider, and
run the named recorder with --control, --candidate and --output build paths.

Broader source layouts, dynamics/AD, short/long strategy selection, W8A8 and
MXFP4 A-fetch/LDS/Radiance attribution remain open. This pattern preserves the
existing package contract and selector policy; it does not close those programs
or establish sibling-device physical parity.

## Fresh delivery gate

Full CI-equivalent WSL unit lane: 2 failed, 20444 passed, 9583 skipped, 14
warnings in 356.54 seconds. Only the existing scaled_matmul batching and
transpose closure assertions fail. They remain unchanged and genuinely open.
Mypy errors=0, baseline=0; 32 generated documents in sync; 15 audit/citation/
recorder gates pass. These results do not make the draft stack merge-ready.
