# Public SM120 saved-LSE reverse dispatch

Owner **AD-RESIDUAL-EVAL-1**; siblings E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Synchronization key **NVIDIA-PUBLIC-ATTENTION-VJP-2026-10-06**.

The canonical @jit(target="nvidia_sm120", autodiff="reverse", wrt=...)
native_backward(..., out_cotangents=...) entry now selects the registered
attention-family consumer. The certified tracer Graph supplies the native
paired AD pass; forward and backward lower through Schedule, Tile and NVIDIA
Target IR into checked native images. Forward privately owns saved output and
LSE until requested backward gradients have completed and been copied out.

## Owning-device proof

- Super-Bear, RTX 5070 / SM120, driver 610.88, CUDA 13.3.
- 88 independent FP64 gradient cases pass: six Q/K/V argument permutations,
  requested gradient subsets/order, grouped-query heads, causal K=129, full
  and broadcast score bias, including reordered bias inputs and bias gradients.
- Maximum absolute error: 1.990847742217028e-8.
- 407 focused WSL host tests pass; the device/native-program fixture batch passes
  61 tests, including two CUDA public reverse cases.
- Changed cotangents and retained prior host outputs pass. Every warm call
  forbids compiler subprocesses.
- Three externally pinned fresh-process common-runtime replays pass with
  TESSERA_OPT=/missing/native/compiler and all subprocesses forbidden after
  device/runtime discovery.
- Portable products check image/descriptor identity, native gradient activity,
  frontend permutation and Tile/Target lineage before CUDA allocation.
  Invalid storage, corruption, repinned incompatible activity and planner
  use after fork have focused host gates.

packet.json contains every row, physical attestation and source fingerprints.
artifacts/ contains pinned native runtime products and independent inputs/
expected gradients; replay-*.json retain fresh-process receipts.
compiler-identity.json pins the matching compiler binary.

## Timing and next ownership boundary

The median of the 88 warm synchronous public host-call medians is **11.3600 ms**.
These values include frontend certification, product validation/restoration,
context/image/storage work, transfers and execution. They are not isolated
device kernel timings, speedup claims or strategy promotion.

warm-profile.json attributes 25 instrumented Q-only calls. Product restoration
is about 9.14 ms cumulative per profiled call, against 17.09 ms total; nested
cumulative values overlap and profiling adds overhead. Repeated contract
decoding/validation is a concrete next native prepared-service boundary.
Launch control remains Python in this increment; no native C++ reverse owner
or retirement of every Python layer is claimed. GPU math comes from native
compiler packages.

## Reproduction

On owning WSL with the matching validation environment:

~~~sh
source .build-sm120-w1-1/validation-env.sh
export PYTHONPATH="$PWD/python:$PWD"
.venv/bin/python -m benchmarks.nvidia.benchmark_public_attention_vjp --output benchmarks/baselines/nvidia_public_attention_vjp_20261006/packet.json
.venv/bin/python -m pytest -q tests/device/nvidia/test_public_attention_vjp.py
~~~

Replay one adjacent artifact with benchmarks.nvidia.replay_public_attention_vjp
and its external artifact_hash from the packet.

## Remaining scope

Static f32 isolated canonical flash_attn is proved. NVIDIA gqa_attention and
mqa_attention aliases are not admitted by this target-specific declaration.
General composition, dynamic/layout/dropout/higher AD, native prepared reverse
ownership and sibling physical consumers remain open. Apple/x86/ROCm share the
registry/result schema but gain no SM120 physical execution claim.
FP8, MXFP8 and MXFP4 remain independent evaluation gates.
The complete five-slice objective remains open.

Generated documentation and compiler-plan checks are retained in adjacent gate files.
Graphify update remains blocked by the missing owning-WSL CLI (exit 127).
