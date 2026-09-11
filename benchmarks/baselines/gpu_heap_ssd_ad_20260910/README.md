# GPU payload frames, checkpoint AD and exact-artifact admission

Uncommitted continuation after PR #741. The CUDA packet is from Super-Bear
RTX 5070 SM120; the ROCm packet is from Princess-Luna gfx1151. Each uses the
assertions-enabled compiler identified in the packet.

`record_gpu_heap_ssd_ad.py` checks native transactional numeric payload allocation:
valid and empty allocations, negative lengths, capacity overflow and signed-int64
extremes, untouched payload/records on failure, a subsequent successful request,
and host decoder rejection of stale generations. Storage is preallocated,
single-writer and synchronous; generations are caller-supplied. This is not
concurrent object allocation/collection or automatic throw-site fusion.

The same recorder executes cooperative SSD forward and passes its actual chunk
checkpoints into the native GPU VJP. All five gradients match independent finite
differences with nonzero cotangents for Y, final carry and checkpoints. Inputs stay
unchanged. GPU-resident automatic AD and backward performance are not claimed.

The admission JSON files replay the actual measured serial/cooperative packages
and call `bind_measured_ssd`. Both retain the incumbent. CUDA needs its typed
native calibration adapter; ROCm needs eligible per-process native calibration
and its policy requires bare metal. Edited eligibility flags cannot override the
recomputed gates. No production promotion or new speedup claim is made.

Reproduce the device proof with `benchmarks/record_gpu_heap_ssd_ad.py --backend
nvidia|rocm --compiler /absolute/path/to/tessera-opt --output packet.json`.
Reproduce selection with `benchmarks/check_ssd_admission.py --comparison paired.json
--compiler /absolute/path/to/tessera-opt --output decision.json`.

Validation: 502 focused compiler/AD/registry/audit tests passed, with five
owning-device/toolkit skips. Separate CUDA attention execution passed 15 tests
(one x86 skip), including two shape buckets at scales 1.0 and 0.5. Ruff and the
zero-error mypy ratchet passed. No full unit-suite run is claimed.
