# Thread-local native JIT review follow-up

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-JIT-20261020. PR #993.
This packet supersedes the parent packet's wrapper source and public latency
measurements. Parent raw data remains unchanged.

## Correctness and lifetime

The original shared owner reproduces four failing worker-thread cases.
Compiler products remain shared, while each thread now retains at most 24
owners, its own completion receipt and native inspection artifact.
Cleanup and eviction release only frames created by the calling thread;
worker code closes in a finally block before leaving the thread.
A retired product cannot silently reuse a different product's owner.
Native affinity/context/process guards remain enforced.

All 175 owning RTX5070 cases pass: 57 new JIT cases including four worker-thread
cases, plus 118 predecessor/adjacent cases. Main-thread cleanup while workers
retain live owners, warm reuse and close/rebind with tools/tracing forbidden
are numerically checked for FP16/BF16 ReLU and explicit-FP32 RMSNorm programs.
This is host-root thread proof; it is not cross-thread resident-buffer proof
or concurrent GPU throughput. Native dispatch retains its serialization lock.
Host contracts pass 21; focused compiler/runtime/diagnostic/pass gates pass 445.
Ruff/Mypy remain clean, with a zero-error baseline.

## Paired measurements

The same RTX5070 UUID, matching ff8bad compiler and 72aaf provider are bound in
all four arms. Four predecessor files match merged main fac21f5bc exactly;
candidate restoration is verified. All four IR stages and executed image
hashes match the parent's native packet and both new native orders.

Six comparable cases give legacy/native completed-public ratios
1.7488-2.4490x across both ABBA orders. Cold calls remain
1967.24-3031.77ms.
Each arm uses 21 samples of 16 completed calls and numerical checks before and
after timing. Unsupported legacy norm rows have no timing denominator.

Two-sided captured windows are mixed: one native row is faster and three are
slower (approximately 1-7%). These small windows changed across runs despite
unchanged native IR/images. Neither a physical-kernel gain nor resolution of
capture/device attribution follows. Individual stage and program windows
exclude host copies, overlap in interpretation and are not additive.

summary.json and the four jit-abba JSON files are authoritative.
Raw failure logs are losslessly compressed with uncompressed hashes.
Code, source-control and compiler identities are in source-identity.json.
Exact-head delivery CI/review remains required before merge.
