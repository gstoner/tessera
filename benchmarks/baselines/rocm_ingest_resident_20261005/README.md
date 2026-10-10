# Resident native NVFP4 conversion, storage bridge and packed matmul

Owner ROCM-NVFP4-INGEST-1; sibling ROCM-MXFP4-W4A8-1.
Sync ROCM-INGEST-RESIDENT-2026-10-05.

## Compiler and runtime ownership

Three separately verified Graph/Schedule/Tile/Target packages lower through
MLIR/LLVM into gfx1201 HSACO images. Conversion owns joint-SSE E2M1/E8M0
requantization, the integer storage bridge permutes bytes and appends column
reference exponents, and the packed consumer owns integer E2M1-to-E4M3 decoding,
WMMA, FP32 accumulation and BF16 output. Python validates the package contract,
snapshots inputs, owns buffers/stream and submits the checked expanded-memref
ABIs. GPU arithmetic is implemented by the native C++ materializers.

Shape-only consumer packaging carries dimensions, physical layouts, explicit
approximate policy and IR/image hashes. It creates no fictitious weight content
hash or fold-loss certificate. Host-payload packages retain their content hash
checks and ancestry. The legacy HIP probe ancestry test now checks each
producer function, since native and probe producers coexist in one module.

The fixed envelope is M>64, N16 and K64, with explicit ordered projection
boundaries, finite input scales and exact gfx1201 ownership. Eleven distinct
private allocations and one private HIP stream carry converter outputs into
storage and matmul. Converted-weight readback is used only for numerical
diagnostics; those bytes are never reuploaded to the consumer. Activation
updates reuse the resident weights, images and allocation addresses.

Input staging survives asynchronous uploads. Failed activation upload disables
matmul until a successful update. Close synchronizes before destroying graph
executables, images or allocations. Failed completion retains the owners for
retry; partial close prevents further launches. Host-input mutation cannot
retarget the live session.

## Exact-device evidence

Tajasaurus WSL, verified AMD Radeon RX 9070 XT / gfx1201.
[Checkpoint packet](checkpoint.json) uses pinned Qwen3-8B layer-0 gate/up:
NVFP4 revision ccd10a893cbca613259517c3efe08e151ddf2b8e and BF16 revision
b968826d9c46dd6066d109eabc6255188de91218. N=24576, K=4096, M=256.
Tensor range/index hashes, projection globals, source hashes, compiler hash,
three image digests and descriptors are recorded. No model shard is stored.

Native converted codes/exponents and the lossless storage bytes match their
independent oracles bitwise. F64 signal/SSE statistics agree. Matmul passes an
elementwise FP32 accumulation bound plus BF16 store rounding before and after
timing; maximum absolute error versus its declared folded oracle is
0.031090, with zero bound violations.

### Timings

| Stage | Resident HIP graph window per iteration ms |
| --- | ---: |
| Conversion | 40.724 |
| Lossless storage bridge | 0.340 |
| Packed matmul | 0.344 |
| Combined three-stage chain | 41.475 |

Compile/package wall: 4021.858 ms.
Checked full-session wall median: 184.219 ms.
Resident-weight reuse wall median: 5.561 ms.

Full-session wall includes validation, private allocation, three module loads,
all input uploads, three native stages, result readback and cleanup.
Reuse wall includes activation snapshot/upload, matmul and result readback;
conversion/storage/modules/weight upload are amortized. These are distinct
workloads, so their ratio is not a kernel speedup.

Graph capture/instantiation are outside timing. One graph submission per
window removes between-node host submission gaps. Graph node census matches
the stage count, and stages rotate/reverse between trials. Graph times include
GPU graph dispatch; they are not isolated kernel/instruction measurements.
Separate host-submission event windows and raw samples remain in the packet.

### Independent format gates on matched sources

The activation SHA is identical across the resident row and controls; all
controls use the same pinned BF16 source matrices and seeded FP32 activations.
Each arm is checked against its own quantization/arithmetic oracle.

| Arm | Device graph execution/dispatch us | Checked wall ms | Output relative RMS vs original source |
| --- | ---: | ---: | ---: |
| fp8_k128_n128 | 547.791 | 18.056 | 3.686% |
| fp8_k32_n1_control | 805.233 | 22.006 | 3.388% |
| mxfp8_k32_n1 | 824.729 | 21.152 | 3.751% |
| mxfp4_folded | 386.909 | 98.129 | 12.049% |
| mxfp4_folded_native_packed | 385.677 | 55.426 | 12.049% |
| NVFP4-ingested resident packed MXFP4 consumer | 344.406 | 5.561 | 15.177% |

The resident consumer wall reuses its resident weights; other arm walls are
checked package launches with host staging. They do not measure identical
runtime ownership. The ingested folded weight error is
14.980% relative RMS.
Seeded activations are not captured model activations. Arithmetic correctness
does not establish model-quality acceptance. FP8, MXFP8 and MXFP4 remain
mandatory independent gates, and no selector/dtype/default is promoted.

## Validation and remaining scope

- 60 focused gfx1201 resident/payload/converter/storage tests pass, including
  complete/ragged tiles, zero scales, input snapshots, allocation/graph
  retirement, failed upload recovery and failed-completion owner retention.
- 473 shared ABI/ancestry/operator/dtype/diagnostic/pass/manifest regressions
  pass on Super-Bear; nine exact gfx1201 cases skip there.
- [Synthetic packet](synthetic.json) proves three smaller/ragged envelopes.
- Fifteen final audit/legacy-evidence tests pass; all 32 generated documents
  are in sync. Compiler-plan, scoped Ruff and diff checks pass.
- All ten archived packed HIP flag sets have byte-identical source emissions
  from the sealed relabel-era generator and the current emitter.
  [Source identity](legacy_packed_emission_identity.json) preserves the old
  timing packets and makes no fresh legacy-kernel timing claim.
- Matching gfx1201 compiler build was retained from the storage bridge slice;
  this increment changes packaging/runtime orchestration, not GPU materializers.

Ordinary composed three-operation JIT dispatch, portable program serialization,
general frontend/AD integration, dynamic packing/layout envelopes, generalized
image keys and whole-model quality acceptance remain open. Individual converter
and storage JIT/replay are proved by the previous packets. The five-slice goal
remains active.

gfx1151 lacks the RDNA4 FP8 WMMA required by this consumer and needs a separate
physical route. NVIDIA, Apple and x86 need architecture-owned layouts, lifetime
contracts and exact-device proof; ROCm evidence does not establish sibling parity.
All four queues are assessed. Graphify is unavailable in WSL; no refreshed graph
is claimed. This increment has not been committed or published.
