# Native public source-attention reverse

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-ATTENTION-SOURCE-REVERSE-20261018.
Base: PR987 / 9e63d8c71; private prepared-owner checkpoint a9d5295ab.

Ordinary native_backward uses the actual original RMSNorm(Q)/saved-LSE
attention source, native paired AD and four typed member calls. Every member
passes Graph/Schedule/Tile/Target/NVVM/PTX and executes under a retained native
C++ owner. Python reconstructs neither numerical derivatives nor Tile code.

## Proof

- Owning RTX5070 / SM120 passes 67 cases: 12 public/cache/portable, 20 native
  reverse and 35 existing source-forward cases. Both half formats, requested
  subsets/order, original argument/result ordering, changed values, static
  shape revisits, independent outputs and read-only roots are covered.
- Fresh-process portable launch succeeds with compiler paths unavailable and
  all external subprocess calls forbidden. Four member receipts and exact
  SM120 attestation follow native completion.
  portable_f16_program.json.gz and portable_bf16_program.json.gz preserve the
  actual programs from those fresh-process tests, with roots/outputs in the
  corresponding NPZ files. Arrays use FP32 transport; reconstruct root storage
  from the sealed plan before launch. These files contain no live handles.
- 410 focused tests cover offline integrity/physical contracts, rehashed
  source/numeric/lifetime/call/image/ABI mutations, composite declarations,
  runtime matrix totality, diagnostics, pass metadata, benchmark naming and
  audit lifecycle. Review regressions reject mismatched normalization
  accumulation and typed numeric attributes; component ABI receipts cannot
  mutate a cached sealed manifest. Unsupported nonzero/NaN dropout is refused
  before concrete frontend evaluation. The 12 public/replay cases are rerun after
  these review fixes; numerical kernel images remain unchanged.
- Shared mock dispatch tests cover Apple/x86/gfx1151/gfx1201 native owner
  success and provider refusal without generic C-library discovery. This is
  dispatch contract evidence, not sibling exact-device numerical proof.

## Matched-input/image dispatch A/B

public-ab.json retains 21 alternating samples of eight calls per arm for
FP16/BF16 small and D257 profiles. Both arms use the same inputs, images and
native owner. The serialized_control.py source is preserved and hashed.
Before/after each sample, outputs match an independent float64 attention and
RMSNorm oracle with explicit half storage boundaries and FP32 statistics.

Retained-owner warm public medians are about 0.75–0.90 ms for small profiles,
versus 3.09–3.13 ms in the serialized runtime control (3.46–4.10x). D257 public
medians are about 5.79–5.81 ms versus 7.90 ms (1.36–1.37x). D257 device
program windows stay about 4.76–4.79 ms. Tiny-case event windows vary from
about 0.11–0.21 ms and include dispatch noise. This is host dispatch improvement;
no kernel speedup or physical selector promotion is claimed.

Compilation/native preparation are outside the warm comparison. Event windows
include possible dispatch gaps and exclude root copies; native completed host
and public end-to-end timing remain distinct. The current backward member
computes all three checkpoint gradients even when returning a subset.

## Reproduction and limits

Use the matching LLVM/MLIR 23.1.1 assertions compiler and CUDA 13.3 provider
identified in the packet. Set TESSERA_OPT, TESSERA_NVIDIA_OPT and
TESSERA_NVIDIA_PTX_LAUNCH_LIB. On owning RTX5070:

python -m pytest tests/device/nvidia/test_public_source_attention_reverse.py tests/device/nvidia/test_source_attention_reverse.py tests/device/nvidia/test_source_attention_prefix.py -q
python benchmarks/nvidia/benchmark_public_source_attention_reverse.py --control benchmarks/baselines/native_public_source_attention_reverse_20261018/serialized_control.py --output /scratch/public-source-ab.json

The native plan remains artifact_only until compiled and executed. Offline
content hashes detect corruption, not authenticity of arbitrary supplied PTX.
Final WSL CPU validation passes 30,752 tests with 3,721 declared skips.
The production compiler-route lane passes 2,079 tests with every selected
proof executed. All 33 generated documents, claim lint, five surface audits,
type checking and plan ownership gates pass. These receipts bind code
checkpoint e8aafdc65ccbb90778b50aea9f10c9c7fdc34548; subsequent changes only record evidence.

Exact-head review/CI and public delivery remain open. Dynamic/bias/dropout/layout and higher-AD envelopes and
all five aggregate programs remain open; no sibling physical proof transfers.
