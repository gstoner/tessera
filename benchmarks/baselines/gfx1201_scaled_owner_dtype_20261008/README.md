# gfx1201 cached checked storage dtype binding

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync SCALED-OWNER-DTYPE-2026-10-08.

Native scaled ABI marshalling now resolves each supported storage token to its NumPy binding dtype once, retaining it with immutable shape and byte-capacity metadata. Per-call input binding checks actual dtype, shape, contiguity and byte extent against that checked metadata. FP8 values are bit-viewed as bytes; no numeric conversion, dtype promotion, Python numerical backend or native kernel change is introduced.

168 matching-compiler host checks pass, including canonical dtype/attribute gates and input/output diagnostic-mutation guards. 80 owning RX 9070 XT/gfx1201 checks pass for FP8/MXFP8 primal, scale JVP/VJP and owner/cache isolation. One existing timeout-plugin configuration warning remains.

Recorder: benchmarks/rocm/benchmark_scaled_owner_metadata_ab.py.
Historical control: benchmarks/baselines/gfx1201_scaled_owner_dtype_20261008/dtype_control.py.

Two counterbalanced rounds, 48 profile executions and 24 matched pairs. Nine public warm-wall samples per profile. Native program/image/frontend Graph hashes match. Overall control/candidate median ratio 1.031385388; primal/JVP/VJP medians 1.038577 / 1.027876 / 1.031385. The candidate is about 3.1% faster relative to the prior lazy-metadata version in this named wall-time envelope. Native event windows remain separate; this does not establish a kernel gain, model-scale gain, or recovery relative to earlier versions measured in other runs.

GPU name/UUID/ordinal, compiler/runtime and control/candidate/recorder hashes are in matched-ab.json. Candidate source hash matches the sealed coordinated source. Existing generic batching/transpose and backend parity states remain open.
