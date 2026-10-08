# SM120 grouped attention producer dependencies

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync ATTENTION-GROUPED-STREAM-2026-10-08.

The owner validates all tensor allocations in each capture, reverse-seed or JVP-direction group, then records one event per distinct declared producer stream. No dependency is cached across calls. Different producer streams retain separate dependencies. Input context, shape, pitch and capacity checks remain; Q/K/V and optional bias still use separate private allocations.

146 affected unit/public/device tests passed on Super-Bear RTX 5070, including pending producer writes and actual one-event wait assertions, reverse and JVP numerical checks, context/lifetime guards. Four existing fork deprecation warnings remain. An initial run exposed a biased Q/K/V copy-loop arity bug; it was repaired and the full selection rerun.

Recorder: benchmarks/nvidia/benchmark_attention_owned_stream_ab.py.
Historical control: benchmarks/baselines/nvidia_attention_grouped_stream_20261008/per_buffer_control.py.
Counterbalanced identical-package A/B completed: 96 profile executions and 48 paired comparisons, with numerical checks passing. Median per-buffer/grouped ratios are capture 1.011891, backward 1.043463 and pair 1.026943, or about 1.2%, 4.3% and 2.7% higher control wall time. This is wall-time characterization in the named small-shape envelope, not isolated kernel gain or global selector promotion. The 12 audit/recorder gates pass. The earlier context-wide comparison in nvidia_attention_owned_stream_20261008 remains a separate source snapshot.

Synchronous allocation/free and synchronous returns remain. Dynamic/composed attention, asynchronous ownership and generic AD closure are open. This is CUDA-only execution evidence.
