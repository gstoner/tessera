# Forward-only native attention JVP checkpoint

Owner AD-RESIDUAL-EVAL-1; siblings FRONTEND-IR-MEDIUM-1 / W1.1.
Synchronization key NVIDIA-FORWARD-ATTENTION-JVP-2026-10-06.

## Compiler change

Public native_jvp now packages a native saved-LSE forward checkpoint plus the native tangent image. It no longer lowers or compiles the unused reverse executable. A distinct AttentionForwardCheckpoint product and pinned v2 serialization contain no backward image. Historical v1 paired products still round-trip and replay; VJP compilation retains the complete forward/backward pair.

The forward descriptor supplies the compiler-owned frontend permutation. The original traced body still goes through native AD, Graph-to-Schedule, Schedule-to-Tile, NVIDIA Target IR and NVVM/LLVM. No Python kernel body or derivative recipe is introduced. Native paired AD may still construct reverse intermediate IR while extracting the forward checkpoint; eliminating that intermediate is a separate compiler optimization.

Resident capture loads one checkpoint module and preserves private Q/K/V/O/LSE ownership. Forward-only frames refuse backward calls before reading buffers. The prepared common-runtime owner already loads only forward/tangent images and consumes the new portable form without a compiler.

## Exact RTX 5070 evidence

Super-Bear WSL, sm120, driver 610.88.

- 72 public oracle cases across six argument permutations, two causal/shape profiles and six tangent selections. Maximum absolute tangent error 1.10436e-8.
- Four matched compile cases, three alternating paired/forward-only rounds per case. Median per-case forward/paired compile ratio 0.828158: about 17.2% less compile wall time. Warm subprocess counts are 18 versus 28; the first paired round has 38 calls because it also populates cached compiler setup. Raw counts and samples remain in compile-packet.json.
- Packages occupy 87,625–101,205 bytes forward-only versus 175,044–187,106 bytes paired. Forward and tangent image bytes are equal between matched arms.
- Three externally pinned fresh common-runtime replays plus three fresh resident-capture replays pass with compiler subprocesses forbidden. The resident replays prove repeated/scaled directions and retained outputs.
- 422 shared/AD/runtime host tests plus 110 checkpoint/reverse guards pass (532 total). Seven exact-device tests pass, including three existing compact requested-gradient backward cases.
- Current source fingerprints in both benchmark packets match. Ruff and diff whitespace gates pass. Graphify update is unavailable in WSL (CLI absent).

Compile wall measurements include native compiler subprocess and package construction; they are not device kernel measurements. First public compile/launch and warm synchronous host samples are retained separately in public-packet.json. No GPU algorithm or performance-default change is claimed.

## Remaining work

Generic native VJP public dispatch, composed/dynamic/bias/dropout/higher attention AD, further native compiler orchestration and asynchronous ownership remain open. Native reverse intermediate construction is still present in the forward checkpoint extraction. Sibling native tangent consumers require architecture-owned implementation and proof. CUDA execution does not establish Metal/HIP/x86 physical parity. Full five-slice closure remains open.

Evidence: compile-packet.json, public-packet.json, artifacts/, *-replay.json,
*-capture-replay.json, contracts.txt, reverse-guards.txt, device-tests.txt.
