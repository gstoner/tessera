# Native source-attention reverse candidate

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-ATTENTION-SOURCE-REVERSE-20261018.
Base: merged PR987 (9e63d8c71). Source hashes bind the unmerged candidate.

The native compiler clones actual original paired AD into four typed call
members. All four pass Graph/Schedule/Tile/NVIDIA/NVVM/PTX and execute in
one retained C++ private frame on RTX5070 / SM120. Python reconstructs no
derivatives or Tile operations. The Graph plan alone remains artifact_only.

## Evidence

- 128 compiler transport/adjacent contracts passed before runtime integration.
- 326 outline/transport/diagnostic/pass registry tests pass.
- 20 new source VJP and 35 existing source-forward cases pass, no skips.
  Deterministic pending writes cover separate, legacy-default and per-thread
  producer streams. Rejected byte extents preserve output sentinels; changed
  inputs and restored context remain executable.
- FP16/BF16, root and O/LSE order, requested gradient subsets, zero/tiny Q,
  D257 and 136 logical rows have independent numerical comparisons.
- Four correctness-gated timing profiles use 21 alternating host/resident
  samples of eight completed calls. Inputs and images are matched.
  sm120.json separates export/compilation/native preparation, four stage
  event windows, program window, completed native host and end-to-end time.
  Event windows include possible dispatch gaps, exclude copies, and do not
  measure isolated instruction time. Cold compilation is outside warm timing.
- Float64 attention/RMSNorm oracles respect half storage edges and FP32
  normalization statistics. This is the declared AD arithmetic contract,
  not differentiation of discrete rounding. Warm calls forbid compilation;
  output generations are independent and read-only roots remain unchanged.

Small profile end-to-end medians are approximately 0.34–0.54 ms; D257 profiles
are approximately 5.2–5.3 ms. These characterize the current implementation;
no speedup or selector promotion is claimed. Requested subsets still compute
all checkpoint gradients.

Ordinary JIT native_backward, sealed portable RuntimeArtifact restoration,
public certificates, broader validation and PR delivery remain open.
None of the five aggregate programs closes from this packet. No physical
evidence transfers to Apple, x86 or ROCm.

## Reproduction

Use matching LLVM/MLIR 23.1.1 assertions compiler and CUDA 13.3 provider built
from the manifest-bound source. On the owning host, set TESSERA_OPT,
TESSERA_NVIDIA_OPT and TESSERA_NVIDIA_PTX_LAUNCH_LIB to those builds.

python -m pytest tests/device/nvidia/test_source_attention_reverse.py tests/device/nvidia/test_source_attention_prefix.py -q
python benchmarks/nvidia/benchmark_source_attention_reverse.py --output /scratch/source-reverse-timings.json

The recorder depends on the repository's independent test oracle and typed
frontend fixture. The runtime owner does not import tests or run tensor math.
