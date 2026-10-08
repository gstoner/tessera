# SM120 asynchronous saved-LSE owner

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-ASYNC-OWNER-2026-10-08.

Public compiled VJP programs accept capture(..., asynchronous=True). Native CUDA stream-ordered allocations, private D2D snapshots, saved O/LSE and compiler-generated gradients remain owned by one frame. Forward/backward images, checkpoint digest and arithmetic are identical to synchronous capture; synchronous remains default.

Producer dependencies precede snapshots. A reverse dependency after each snapshot prevents later writes on its declared producer stream from overtaking private reads. Source references remain until synchronize/close. Async seeds free on the private stream after backward. Escaped views advertise the private stream. wait_on validates/registers a consumer; close completes registered consumer work before freeing any output.

156 affected owning/host tests pass. Eight strengthened RTX 5070 cases restore a serialized program and forbid compiler calls, hold producer seeds pending, prove enqueue returns before release, mutate captured sources, replay gradients, retain prior outputs and close while a registered consumer reads an output. Three new host checks cover snapshot handoff and consumer-before-free ordering. The initial 73-pass/one-failure fake capture signature is preserved and repaired without changing an admission gate.

## Separate timing characterization

Each mode has five windows of eleven backward calls. Submission duration is not completed execution. Completed host includes the explicit private-stream wait. CUDA events encompass allocation, D2D snapshots, kernel, stream-ordered frees and host enqueue gaps; they do not isolate the kernel. Synchronous then asynchronous order is not counterbalanced A/B, so no speedup or automatic policy claim is made.

| Shape | Bias | Async | Submission median ms | Completed host median ms | Owned-stream event median ms |
| --- | --- | --- | ---: | ---: | ---: |
| [1, 2, 1, 3, 5, 4, 3] | False | False | 0.190556 | 0.195776 | 0.191276 |
| [1, 2, 1, 3, 5, 4, 3] | False | True | 0.162007 | 0.165559 | 0.162275 |
| [1, 2, 1, 3, 5, 4, 3] | True | False | 0.214754 | 0.219856 | 0.215727 |
| [1, 2, 1, 3, 5, 4, 3] | True | True | 0.173609 | 0.201257 | 0.197684 |
| [2, 4, 2, 16, 19, 8, 6] | False | False | 0.190976 | 0.195714 | 0.192311 |
| [2, 4, 2, 16, 19, 8, 6] | False | True | 0.158496 | 0.163354 | 0.159633 |
| [2, 4, 2, 16, 19, 8, 6] | True | False | 0.304203 | 0.308871 | 0.305542 |
| [2, 4, 2, 16, 19, 8, 6] | True | True | 0.179695 | 0.191698 | 0.188300 |

Independent float64 O/LSE and gradient oracles pass before/after each window; maximum gradient absolute error is 1.6764358856669048e-07. Actual GPU identity: NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0. Compiler and source snapshots, image identities and raw samples are bound in the packet.

## Lifetime and remaining scope

Inputs must declare producer streams. Producer and registered consumer streams must remain alive until frame close. Enqueue all external reads after wait_on and before close; later reads are invalid. synchronize completes producer work and releases retained source references, not an implicit global consumer fence. Dynamic/composed attention, automatic external-consumer discovery, general asynchronous JVP and higher-order AD remain open. This result does not transfer to ROCm, Apple or x86. Full unit generic scaled-matmul closure remains open.
