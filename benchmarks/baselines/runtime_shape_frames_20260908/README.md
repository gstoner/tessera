# Runtime-shaped AD and scoped frame retirement — 2026-09-08

The recorder exports native AD products, compiles each target independently and
checks numerical output on SM120 and gfx1151. Packets bind the compiler, recorder
and implementation source hashes.

- Dynamic GPU backward input/output lengths: 0, 2 and 4.
- Negative, oversized and incompatible input shape sidecars refuse exposure.
- Dynamic matrix gradients: 2x2, 1x4 and 0x4, packed into flat capacity storage.
- Two independent matrix results preserve their logical shapes and values.
- Eight scoped captures retire complete frame and generation storage after
  registered readers. Instrumentation rejects explicit context waits in the
  retirement region; every iteration releases all frame buffers.

Capacity is physical storage, not a promise that every logical shape fits.
Per-axis and total-volume checks run in the native product before access/copy.
Asynchronous public-result submission queries completion before small status and
shape readbacks; it exposes no result before successful validation.

Scoped persistent frames are static-shaped; integrating runtime-shaped public
results into their ownership protocol remains open. Capture, exceptional cleanup
and closing unrestricted public views remain
synchronous. Driver module unload has no bounded-host-latency claim. These are
correctness and ownership packets, not overlap or performance measurements.
Arbitrary Python CFG capture and general saved heterogeneous products remain open.

Validation for this increment: 18,502 full non-slow unit tests passed on
Princess-Luna (2,231 skipped); 299 focused compiler/audit checks passed with
Super-Bear's assertions-enabled Tessera build. Four isolated descriptor checks
also passed with the compiler deliberately unavailable. Ruff, the zero-error
mypy ratchet and the 30-document generated drift check passed.

Recorded by `benchmarks/record_runtime_shape_frames.py` (`nvidia.json`, `rocm.json`).
