# Apple GPU test fixtures

This directory holds compiled Metal package (`.mtlpackage`) artifacts the
packaged-kernel sprint (PK1–PK7) tests load through
`tessera.apple_mlpkg.compile_mlpackage`. Without these fixtures the
real-artifact tests in `tests/unit/test_apple_mlpkg_pk{1,2,3,4}.py`
skip cleanly — drop a new package in here to unlock them.

## Bundled fixtures

### `matrix-multiplication.mtlpackage/` (16 KB)

Compiled Core ML → Metal matrix-multiplication model from Apple's
sample project [Running a machine learning model on the GPU
timeline](https://developer.apple.com/documentation/metal/running-a-machine-learning-model-on-the-gpu-timeline).
The package's compiled artifact (a `MPSGraphPackage` wrapped in a
Metal package manifest) carries three tensor bindings:

| Binding   | Rank | Shape (innermost-first) | Dtype |
|-----------|-----:|-------------------------|-------|
| `inputA`  | 2    | `(K, M)`                | `fp32` |
| `inputB`  | 2    | `(N, K)`                | `fp32` |
| `output`  | 2    | `(N, M)`                | `fp32` |

The model computes `output = inputA @ inputB` (with the
innermost-first storage layout — what numpy sees as `A[m, k]` is the
package's `inputA[k, m]`).

PK4's numerical-correctness test
(`test_dispatch_produces_correct_matrix_multiply`) generates random
`A`/`B` with numpy, dispatches through the packaged path, and
validates against `numpy @ numpy` at fp32 rtol=1e-3.

**Provenance:** copied verbatim from
`Packages/matrix-multiplication.mtlpackage/` in Apple's sample on
2026-05-31. Source repo / sample code:
`https://developer.apple.com/documentation/metal/running-a-machine-learning-model-on-the-gpu-timeline`.

**License:** see [`APPLE_SAMPLE_LICENSE.txt`](APPLE_SAMPLE_LICENSE.txt)
in this directory — Apple's standard sample-code license. The copy
preserves the original notice as the license requires.

## Adding a new fixture

To unlock more `mlpkg` test paths, drop a `.mtlpackage` directory in
here and:

1. Append a row to the "Bundled fixtures" table above describing its
   bindings + shape contract.
2. Note the provenance + license in the README so a future maintainer
   knows where it came from.
3. If the license is not MIT-compatible (Apple sample / public
   research / etc.), copy the original license text alongside.
4. The package must be `[device newLibraryWithURL:]`-loadable on
   macOS 26+ — `.mlpackage` (Core ML source format) does NOT work;
   it must be the **compiled** `.mtlpackage` form (Xcode / Core ML
   Tools output).
