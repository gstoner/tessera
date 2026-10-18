# SM120 mixed source pullback

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync W11-MIXED-ADJOINT-CONTRACTION-20261018.

## Executed boundary

Original frontend RMSNorm → ReLU → plain FP32-output matmul, with one requested
source gradient, FP16/BF16 primal storage and an FP32 output cotangent. Native
paired AD outlines a mixed contraction and one-result row member; both cross
Schedule/Tile/NVIDIA/NVVM into one checked C++ CUDA owner. Public host/resident
native_backward and compiler-free portable replay preserve argument order,
private intermediate lifetime and producer ordering. This is the named static
source pullback, not generic reverse/batching/encoded-operand AD closure.

Measured on RTX 5070, UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, SM120,
driver 610.88, CUDA 13.3 and assertions-enabled LLVM/MLIR 23.1.1 in Super-Bear
WSL. The active CUDA context's PCI ID is resolved against nvidia-smi.
The Git revision is the candidate's base; modified-source hashes and the frozen
source snapshot identify the actual uncommitted code, not stock base behavior.

## Correctness and timings

Twelve profiles: M/N/K 17/19/35, 64/64/128 and 128/128/512, both half-storage
types and both original argument orders. Every profile passes an independent
FP64 pullback oracle with explicit half-store boundaries before and after
timing. FP32 contraction error and half-store uncertainty propagate into the
normalization bound. This is a finite random-input envelope; it does not
certify every range, rounding or nondifferentiable point.

Seven samples per standalone profile. Completed public medians are
0.955–1.319 ms; captured composed CUDA medians are 0.00959–0.24228 ms.
The two native stages, composed enqueue/capture and completed host wall
intervals remain separate. First-call timing includes certification/packaging
but makes no cold compiler-cache claim. There is no kernel-speedup claim.

The diagnostic A/B retains the previous repeated Graph copy/portable validation
in its control. Both arms borrow the identical native owner, images, roots and
execution certificate. Two reversed orders, seven samples per round and twenty
calls per sample produce control/candidate ratios 1.75–2.01 (geometric mean
1.908). The improvement concerns public metadata/projection overhead. It does
not change the kernel, numeric policy, storage format or default strategy.

## Artifacts and validation

- timings.json: final standalone recorder, device identity, source/tool/provider
  hashes, per-profile raw intervals, numerical errors and execution certificates.
- programs/: compressed compiler-produced RuntimeArtifacts for all twelve rows.
- metadata-ab.json.gz: paired public overhead windows and identity checks.
- source-snapshot.json.gz: final recorder's modified production source.
- controls/: frozen diagnostic controls, control source snapshot and exact A/B
  recorder source. These are evidence, not alternate shipping backends.
- compression-manifest.json: raw/compressed byte counts and SHA256 for every
  compressed artifact.

The final matching-build owning suite passes 274 tests with zero skips;
existing public/resident JVP compatibility passes 442. The full WSL CPU lane
passes 26,588 tests with 5,719 skips and zero failures. The compiler-route lane
passes 1,754 tests with zero skips and its executed-proof certificate. Audit
lifecycle passes 11; Ruff and all 33 owning documentation generators pass.
The twelve standalone rows were rerun after those workloads with stable source,
compiler and provider identities. The diagnostic A/B retains its separately
frozen historical control/candidate source scope; the final standalone packet
does not relabel that earlier A/B as a fresh comparison. Refreshed packet tests
pass in the final staged gate (1,391 passed; five gfx1201-only skips on this
NVIDIA host). Publication remains pending. The aggregate five-slice goal remains
open.

## Reproduce

Use a matching current compiler/provider and the owning SM120 WSL environment.
TESSERA_OPT, TESSERA_NVIDIA_OPT and TESSERA_NVIDIA_PTX_LAUNCH_LIB select the
recorded tool/runtime interfaces. From the Tessera repository root:

    python -m benchmarks.nvidia.benchmark_source_reverse \
      --output /path/to/scratch/timings.json \
      --artifact-dir /path/to/scratch/programs

The exact diagnostic A/B recorder is frozen under controls/record_ab.py.gz.
It expects a scratch parent containing worktree/; extract the controls beside
that checkout as source_reverse_runtime_control.py and
public-reverse-control-source.json.gz, then run the extracted recorder from
worktree/. It creates its composite-control module and outputs in that scratch
parent. It temporarily selects metadata functions inside its own process and
restores the candidate before exit; it never substitutes numerical kernels.
