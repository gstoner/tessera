# Row ANN and strided while owning-device evidence

CUDA: RTX 5070 / SM120. ROCm: Radeon 8060S / gfx1151. Each packet records
its own compiler and source hashes. These are separate architecture proofs.

`row-ann-*.json` validate ReLU, absolute value and rank-changing row-sum
original/rewrite execution and budget exclusion through the scoped arbiter.
`strided-tape-*.json` validate four early exits and two predicate paths with
12 asynchronous derivative submissions per backend. Generation release still
uses the existing context barrier; this is not fully asynchronous reclamation.

Nine fresh processes per backend (`nvidia-1.json` through `nvidia-9.json`, and
corresponding ROCm files) compare original and rewritten ANN under the fused,
row-parallel schedule. Each run verifies numerical output before and after
31 paired trials. `summary.json` uses exact order-statistic bounds on run
medians, each with at least 95% one-sided coverage. No samples were discarded.
The lower bounds do not clear 1.02x, so performance promotion remains disabled.
These are warm package host-wall times including transfers. They do not compare
row versus serial scheduling and do not measure kernel clocks or overlap.

Native host shape-varying while proof is in
`tests/unit/test_native_loop_next.py::test_shape_varying_while_native_host_products`:
widths 3/4/5/8/16, zero through three iterations, repeated backward evaluation.
No dynamic GPU residual storage or Apple device proof is claimed.
