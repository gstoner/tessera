# Probed admission and health-checked replacement of isolated heap workers

Owner: W4-PRODUCT-1 (sync `HEAP-REPLACEMENT-HEALTH-2026-09-15`). Continues
`gated_heap_20260911`. `record_gated_heap.py` executes independently on CUDA
SM120 (The-Super-Bear, RTX 5070, CUDA 13.4 / driver 610.88) and ROCm gfx1151
(Princess-Luna, ROCm 10.0); `source-hashes.json` binds the implementation used
by both owning-device runs. Packets are correctness evidence; nothing here is
a performance promotion, and neither packet transfers to the other
architecture.

What is new since 2026-09-11:

- **Probed admission.** A spawned `IsolatedHeapPool` worker is admitted only
  after its own in-process device probe (`native_isolated_heap._probe_health`)
  verifies the admitted producers on the ordinal it will own: an int8
  allocation with a known pattern publishes exactly one live slot of the
  payload width, reads back bitwise through a generation-checked pin, and
  disappears after an empty graph is published, marked and reclaimed. The
  ready message carries the probe tag (`health` in each packet); a bare or
  foreign ready message is a failed admission and the worker is torn down,
  never an owner. Admission has its own bound (`startup_seconds`, default
  180 s) because it compiles the metadata kernels in-process — measured 21 s
  on the RTX 5070 alone — while `timeout_seconds` keeps modelling one device
  command that never returns.
- **Health-checked replacement.** `replacement()` refuses until the
  predecessor's death is confirmed (closed, failed, lease reusable) and then
  admits a fresh worker whose own startup probe is the health evidence. The
  recorder proves the refusal before death, the admission after it, and that
  the replacement serves allocate/inspect/close and is itself torn down with
  confirmed death.
- **Replay pipeline repair (prerequisite).** On `origin/main` every
  `ResidentObjectPool`, SSD, ANN, exception-heap, public-result and
  gradient-sum package was refused on both boxes with "disagrees with native
  replay": the packager had gained `--tessera-expand-lowp-conversions`
  (100a2980) and six validators still replayed a hand-copied four-pass
  pipeline. `native_gpu_storage.ARENA_PIPELINE` is now the one spelling and
  `tests/unit/test_arena_replay_pipeline.py` fails a copy. These packets are
  the first device runs of the heap family since that regression.

Limits, unchanged from 2026-09-11: the stopped-worker fault is injected with
SIGSTOP, not a reproduced driver hang; death proves resource-owner
termination, and the probe proves the selected workload on this device ordinal
now — not global driver health or a device reset. Legacy snapshot/import
callers are still not migrated to a gated producer. No measured overlap,
arbitrary command transport or cross-architecture transfer is claimed.

Reproduce on an owning host (after `source scripts/_nvidia_env.sh` or
`scripts/_rocm_env.sh`):

```bash
PYTHONPATH=python:. python benchmarks/record_gated_heap.py --backend nvidia \
  --compiler build/tools/tessera-opt/tessera-opt --output <dir>/nvidia.json
```
