# SM120 row-major RHS native Schedule package

Owner: W1.1. Synchronization: NVIDIA-ROW-MAJOR-B-SCHEDULE-2026-10-03.

Native Graph selection is hashed and replayed through Schedule, typed Tile views/fragments, NVIDIA Target IR and PTX. Static unfused fp16/BF16 consumers with fp32 output have distinct checked row-major RHS ABIs. Macro CTA selection remains restricted to its existing column-major contract.

## Exact-device proof

RTX 5070, SM120, driver 610.88; identity and compiler/source hashes are in packet.json. All 20 matched-input cases passed through checked host and resident execution: complete and ragged shapes, multi-K and both physical RHS orders. Maximum absolute error: 9.059906005859375e-06 against stored-value fp32 matmul. Padded host views are explicitly rejected by the compact static ABI; compaction is performed explicitly by the recorder before launch.

87 focused compiler/producer tests passed (contracts.txt). The first recorder run correctly rejected a padded host view; run.txt records the corrected completed run.

## Timing and decision limits

packet.json separates host staging wall time from resident CUDA-event dispatch windows. These short windows are integration measurements, not isolated kernel attribution or evidence for default promotion. FP8, MXFP8 and MXFP4 remain mandatory separate correctness and performance gates before selecting a strategy. No low-precision layout support is inferred from half-storage execution.

## Remaining work

The named static RMSNorm RHS producer is integrated: rhs-packet.json records eight fp16/BF16 complete/ragged multi-K cases, independent normalization and matmul checks, same-stream resident execution, owned allocations and closed-handle rejection. Producer and consumer CUDA-event dispatch windows are separate; these remain dispatch-window measurements, not isolated kernel-only attribution. Descriptor identity, shape and layout mutation tests reject drift before allocation. General producer graphs, dynamic capacities and fused RHS consumers remain open. Dynamic/fused row-major RHS contracts remain outside this slice. HIP, Metal and x86 need their own physical producer contracts and exact-device evidence. Graphify is unavailable on this host; graph freshness is not claimed.

Final integration validation: 411 focused producer/fragment/descriptor/diagnostic/pass/audit tests passed; 261 final metadata/audit checks passed after documentation updates. Ruff passed. The owning generated-document workflow completed; graphify returned 127 (unavailable).

## Frontend integration

The explicit JIT compile_native_rhs_matmul API traces one semantic matmul(lhs, rmsnorm(source)) function. It verifies the complete Graph, checks structured CFG identity, then partitions existing operations into native packages with freshly recovered CFG metadata. It copies the caller Graph and preserves public argument order. The compiled program supports positional and named inputs and returns the owned resident result.

jit-packet.json proves eight public API cases on RTX 5070, including both frontend argument orders, fp16/BF16 and complete/ragged multi-K shapes. Normalization and stored-edge matmul checks precede separate producer/consumer dispatch windows. Duplicate argument binding and malformed/stale graph contracts are rejected. This explicit compiled program does not claim automatic native dispatch from ordinary JIT calls, autodiff for the composed graph or general producer closure.

Frontend final validation: 452 focused tests passed with three skips; all four focused JIT partition/binding/CFG tests passed. The public API device rerun passed eight cases. Ruff, scoped diff checks and compiler plan checks passed. Owning generators refreshed checked-in documentation; graphify returned 127.

The final generated-document registry, CI drift script and audit suite passed 84 tests (jit-final-docs.txt).
