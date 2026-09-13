# GFX1201 reusable attention, dynamic matrix and sparse packing proof

Owner: E2E-REAL-6 / ROCM-2. Sync: `GFX1201-RESIDENT-SPARSE-2026-09-13`.
Host: Tajasarus, RX 9070 XT gfx1201, ROCm 10.0, assertions-enabled LLVM/MLIR
23.1.1, Ubuntu 26.04 WSL2. Source/compiler identities accompany the packet.

## Implemented and checked

- `ResidentROCmAttentionTape` keeps module, input snapshots and workspace across
  backward calls. Its single owning worker serializes HIP calls; futures retain
  the frame through submission and retirement. Results are separate host copies.
  Device tests cover fp16/bf16, repeated calls, caller mutation, closed admission,
  one allocation/module set and balanced cleanup. Host interleaving tests cover
  delayed completion, a failed call cancelling subsequent execution and invalid
  cotangents preserving a healthy owner. Uncertain teardown retains resources
  in quarantine; recovery and raw external-device readers are not implemented.
- Dynamic f16 matmul reuses one exact image for (M,N,K) = (37,29,35), (1,7,5),
  (64,64,48), checking fp32 results and refusal beyond declared bounds. The
  existing row-major runtime ABI remains unchanged. Epilogues/other dtypes stay gated.
- `pack_sparse_wmma_inputs` produces immutable wave32/OPSEL=0 A/B register bytes
  and ordered 2:4 compression indices for fp16/bf16 16x32 by 32x16. It refuses
  pruning and unsupported storage. Six native LLVM probe comparisons exercise
  all index pairs; disassembly requires each precise SWMMAC opcode. These are
  executable packing probes, not production Tessera sparse Schedule/Tile packages.

## Measurements and limits

Five alternating pairs in one process: resident call median host time 1.123854 ms;
per-call allocation/module setup median 47.607267 ms. Resident capture and
retirement are excluded. HIP event samples are retained separately; this is
uncalibrated diagnostic timing, not an eligible comparison or kernel speedup.

rocprofv3 successfully traces the sparse probe after compiler subprocesses are
excluded from profiler injection (injected ld.lld crashed). The retained database
has 98 API events but zero code objects, kernel symbols, kernel dispatches and
counter events. Runtime kernel attribution and performance promotion remain closed.

Saved-LSE admission remains closed: this owner retains the recompute program's
workspace but does not establish reuse of forward LSE. GPU-stream overlap,
asynchronous device allocation/free, external resident readers, isolated recovery,
production sparse IR/index producers and the remaining SWMMAC datatype signatures
are follow-ons. No gfx1151, NVIDIA, Apple or x86 device proof transfers.

Sparse layout authority: repository RDNA4 ISA sections 7.12.2/7.12.3 and
[AMD's matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator).
