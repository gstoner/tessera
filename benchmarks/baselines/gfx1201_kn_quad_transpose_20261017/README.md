# gfx1201 KN quad transpose into the existing LDS view

Owner: ROCM-FP8-BLOCKSCALE-1 / LAYOUT-ALG-1 / E2E-REAL-6.
Sync: GFX1201-KN-QUAD-TRANSPOSE-20261017.

## Compiler and mathematical contract

Python/textual Graph -> verified native Schedule -> Tile views/fragments ->
ROCm Target -> GPU/LLVM -> HSACO -> checked HIP ABI executes on Tajasaurus,
RX 9070 XT/gfx1201. The implementation is C++ MLIR code generation;
Python supplies inputs, package binding, numerical oracles and measurement.

The KN byte loader retains its physical [Ntile,Kstage] LDS view. Four K rows
inside a wave exchange four packed i32 words using two XOR butterfly stages,
then eight V_PERM byte operations transpose each 4x4 byte group. Each lane
writes four aligned K-contiguous words. Existing scale application, ascending
partial accumulation, A staging, B fragment layout, phases, allocation and
workgroup barriers remain unchanged. Loads rejoin before exchange; active
copy rounds are closed under both XOR peers. Unsupported geometry retains
the scalar copy. No dtype, operation, ABI or production Schedule admission is
added. No gfx1151/gfx1200 or sibling device proof transfers.

Eligibility: gfx1201 KN, N-vector count a power of two <=8, Kstage divisible
by four, thread count divisible by four times N-vector count, and aligned LDS
stride. For old byte destination d and local quad coordinate q=(Krow mod4),
new destination is d+q*(4*stride-1); four columns remain spaced by stride.

## Numerical and instruction evidence

Normal CMake core/ROCm targets rebuild successfully. The 42 owning tests
comprise 40 numerical cases and two retained 64 KiB capacity refusals:
K16/32/64/128 stages, N-vector groups1/2/4/8, non-power-of-two and over-wide
scalar fallback, tiny/ragged columns, FP32/BF16 outputs, phases/padding,
prefetch0/1/2 and static/projected image reuse. The initial new test omitted
capacity expectations for two oversized configurations; its red log is
retained, and the existing compiler capacity gate was not changed.

A production 200x8192x1024, 128x64/eight-wave profile changes 24 static
ds_store_b8 sites to four ds_store_2addr_b32 sites, with eight DPP exchanges
and16 V_PERM sites. Its32 WMMA sites,16 paired LDS fragment reads, two barrier
signals/two waits,191 registers, zero scratch,27648 LDS bytes and live occupancy
of two blocks at256 threads are unchanged. These are static instruction sites
and live HIP resource queries, not dynamic counter attribution.
Two unsupported geometry pairs have identical complete image hashes.

The earlier isolated mechanism diagnostics cover128000 host-model vectors,
155648 active/90112 inactive wave words, and768 masked multi-wave LDS
configurations/25067520 bytes. They support mechanism attribution, not a
standalone production performance claim.

## Paired timings

Source e0cae0944 is clean and unchanged across the measured runs. Both arm and
shape order are reversed between trials. Each arm has10 admitted >=10ms
device-clock windows, cross-checked by HIP events. Separate completed-public
windows include staging/transfers/completion with equal calls per paired
window; numerical checks are outside those windows.

| M,N,K | Candidate/control kernel ratio, two orders |
|---|---|
| 128,4096,128 | 1.031-1.039 |
| 200,8192,1024 | 0.843-0.843 |
| 200,2048,1536 | 0.902-0.907 |
| 200,2048,2048 | 0.891-0.891 |
| 256,4096,4096 | 0.814-0.814 |

The four K1024-4096 profiles improve9-19percent in this forced matched
physical schedule. The forced K128 LDS profile regresses3-4percent; it is
outside the existing production KN LDS Schedule envelope. Schedule defaults
remain unchanged. Completed-public costs have no consistent gain, so no
end-to-end speedup is claimed.

A separate matched BF16 AITER packet records candidate/AITER ratios
1.108,1.218,1.112 for the three M200 rows; control/AITER is1.342,1.378,1.293.
The remaining KN gap is11-22percent. AITER consumes its native transposed
weight layout; Tessera KN consumes compact [K,N]. The separate NK FP32 arm
is retained in raw data and is not a matched BF16 denominator.

Twelve independent FP8 K128, FP8 K32, MXFP8 and folded MXFP4 control pairs
across three shapes, including M256, pass numerical checks and retain identical
images. Folded MXFP4 keeps its explicit approximate numeric policy. These are
unchanged controls, not an MXFP4 improvement or model-quality acceptance.

## Reproduction and remaining gates

Use matching assertions-enabled LLVM/MLIR23.1.1 core/ROCm tools on exact
gfx1201. Keep the merged-main control compiler and current compiler distinct.
The raw packets record complete benchmark arguments through schedule fields,
compiler/source hashes, input/ABI lineage, numerical checks and windows.
Run the existing benchmark_gfx1201_fp8_blockscale with two matched
lds:128x64:8:1:128:16:0:kn+bf16 arms, selecting the control with @control and
--alt-compiler control=PATH; reverse arm/shape order. Use
record_gfx1201_interleaved_compiler_formats --include-mxfp4 for format controls.
inspect-native.py.xz and inspect-production.py.xz preserve exact resource probes.
The initial six-geometry matrix uses separately linked, source-identical
scratch tools; the matching normal CMake build revalidates all42 tests and
supplies the production-profile resource query.

Large raw files are losslessly compressed. raw-packet-archives.json records
raw and compressed SHA256 values. source-identity.json binds native source
and both compiler tools. No tool binaries or duplicate transport bundles are
checked in.

Broad frozen CPU, production compiler-route, generated-doc/lint and public
CI/review gates remain pending. General KN/W8A8 closure, tiny-LDS strategy,
MXFP4 M256 cost attribution, model quality and all five aggregate compiler
slices remain open.
