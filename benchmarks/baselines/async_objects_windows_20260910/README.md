# Stream-owned object graphs, SSD AD and calibration — 2026-09-10

These packets are uncommitted development evidence following PR #741. No route
is promoted. CUDA uses Super-Bear WSL/RTX 5070 SM120; ROCm uses Princess-Luna
WSL/AMD 8060S gfx1151. Each execution packet records its own compiler and package
identities; the hosts have different compiler builds. Both are assertions-enabled
LLVM 23 builds. Source hashes identify this accompanying implementation snapshot.

`record_async_pool_ad.py` proves:

- Four fixed 64-byte records with four reference edges each; a cycle reachable
  through the fourth edge survives partial collection. Opaque JSON bytes survive
  a reader copy; unrooted cycles collect and slots reuse incremented generations.
- Three streams coordinate read scopes, collection and reuse. Collection refuses
  an active reader lease, then waits on its recorded completion after scope exit.
- Two SSD VJPs compose via the first gradient on different streams, with an
  external reader on a third stream. Both derivative generations retire through
  asynchronous frees. All five copied gradients match the independent synchronous
  VJP composition with zero observed difference on each host.
- Context synchronization is forbidden through the frame API during asynchronous
  backward/composition. Capture, benchmark readback and whole-frame close remain
  synchronous. This is first-order family composition, not a general public tape.

Host-free regressions additionally exercise event-record failure and explicit
completion before retry. Collection remains exclusive with mutations; it is not
a concurrent marking algorithm or automatic traversal of arbitrary Python objects.

`cuda-calibration.json` records a 512x2x32x8 cooperative SSD, chunk 32. It binds
actual run nonces, PID, CUDA UUID, image/binding identity and 701 Nsight intervals.
Maximum event/window disagreement is 0.500014%; profiler overhead is 1.002378x.
Both pass the fixed 5% limits. Dirty source and WSL still refuse promotion.
`rocm-large-ssd.json` proves the corresponding larger forward numerics independently;
HIP event windows are not a replacement for missing native profiler calibration.

Raw CUDA capture remains on Super-Bear at
`/tmp/ssd-calibration-large-current.nsys-rep` and `.sqlite`. The SQLite digest is
in the packet. `record_ssd_calibrated_pairs.py` collects nine alternating independent
process pairs with 18 separate calibrations on an eligible clean bare-metal host.
Its diagnostic flag permits investigation without bypassing admission. This
snapshot supplies one diagnostic calibration, not 18 eligible process records.

The larger workload became possible by giving SSD a checked 64 MiB per-buffer
storage envelope instead of importing the persistent-tape pilot's 1024-element
shape parser. That bound is not a performance guarantee for all admitted shapes.
Windowed GQA CUDA proof is in `test_attention_loop_idiom.py`; mandatory Presburger
nonempty-row constraints survive caller-supplied constraints. Other backend
window contracts and arbitrary/additive/padding masks remain open.
