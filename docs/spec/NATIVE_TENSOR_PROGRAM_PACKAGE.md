# Original-source native SM120 tensor packages

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-PROGRAM-PIPELINE-20261019.

## Native admission and original-source replay

The Graph-to-Schedule tensor-program-query option checks admission on an
isolated clone and returns a boolean module attribute. It preserves the caller's
semantic operations and policies; a decline does not rewrite a fused or
unsupported program into another computation. Actual compilation stays strict.

The tensor-program-replay option consumes the complete typed NVIDIA Target
program and its retained original Graph. It re-runs the source through native
Graph/Schedule/Tile/Target passes and compares the entire reconstructed program,
including arithmetic, semantic attributes, symbols, buffer SSA and launch
bindings. The source witness canonicalizes only outer printer whitespace,
whose final newline differs between top-level and nested modules. Its content
seal binds the original canonical Graph.

Replaying requires a nested pass execution. The source is temporarily attached
as a separately owned isolated module, then detached before comparison. The
actual controller/members are not edited during replay. Typed but changed
policies, native arithmetic, entry symbols and source seals are rejected.

## Native package API

The frontend can explicitly compile its original Graph:

    program = fn.compile_native_tensor_program(a, b)
    with program.prepare() as owner:
        output = owner((a, b))

fn is an exact nvidia_sm120 JIT function. The current named profile uses static
compact rank-two f16/bf16 inputs, admitted unary producers and a plain matmul.
Explicit matching FP32 normalization policy survives unchanged. Reversed
frontend argument order and two-sided producer dependencies remain native SSA.

NativeTensorProgram stores immutable original Graph, complete Target, projected
members and native image bytes. Construction validates native source replay,
binds the external original Graph seal, reprojects each member from the actual
Target and compares each image to native code generation. Replacing a frozen
package field runs those checks again. Frame inspection returns a copy.

Preparation and retained host calls do not invoke the compiler. The existing
C++ native-frame owner retains images, CUDA stream and arena, validates physical
buffers and owns completion. This increment creates no numerical Graph/Tile
bodies in Python and no Python operation-family backend dispatcher.

## Scope and evidence

Evidence: benchmarks/baselines/native_tensor_source_package_20261019/.
Eight owning RTX5070 cases exercise the explicit frontend compile API for
FP16/BF16 ReLU, explicit-policy RMSNorm, two-sided ReLU/softmax and reversed
arguments. Compiler tools are forbidden during binding and warm replay.
Independent float64 oracles include half-storage producer rounding, changed
inputs, independent completed outputs, intact guards and stable arena capacity.

Ordinary fn(a, b) dispatch and compiler-free portable deserialization remain
open. This API does not silently substitute for fused/dynamic legacy packages.
There is no dictionary loader that admits images using self-consistent hashes
alone. General source/layout/dynamic/AD and five-slice closure remain open.
Compilation, binding, completed host calls and device windows are measured
separately; no performance or dtype promotion follows.
