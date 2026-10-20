# PR998 scalar-width and oracle review closure

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-BINARY-PRODUCERS-20261020.
Initial published head: b56976250547f78c2715335d20f2dc76f275d4ea.
The current source is that head plus the review fix; source and build hashes
below bind the actual compiled dirty snapshot. Exact-head CI is separate.

## Correct native extent semantics

The compiler already emits i64 add element counts, including N=2^31, while
Python/native owners incorrectly imposed INT32_MAX on the existing i64 wire
fields. Both now preserve positive signed-i64 values. Shape axes, capacities,
selected kernel semantics and launch bounds remain independently checked.
The ABI layout and all 102 shipped provider exports are unchanged.

Production prepare and a test-only observer share one pure C++ metadata planner.
The observer TU lives under tests and includes the production implementation;
the ABI inventory has no scan exemption and a regression pins observer absence.
Prepare remains eager. The retired no-allocation preparation diagnostic is
explained in diagnostic-correction.json; it is not a numerical failure.

Three compiler/PTX/native metadata cases prove N=2^31 for FP16/BF16/FP32 and
reject zero/negative scalars with zeroed receipts. They allocate no CUDA arena
and do not claim numerical execution of multi-gigabyte buffers. Normal-sized
ordinary/resident/rounding proof remains separate.

The source-bound fresh-provider cohort passes 143 cases: 60 compiler, 45 device
(including the three metadata cases), 18 recorder and 20 transport. Canonical
production provider SHA57374626 is byte-identical to that tested binary. The
relocated observer is independently rechecked in the same three metadata cases.
Production/observer source identities, actual compile commands and exports are
included. They bind CUDA13.3, the host compiler and the exact current sources.

## Valid correctness oracle and fresh timings

The recorder rejects K>=2^24 before hardware; its FP32 gamma-K bound requires
K*u<1. The oracle itself also guards the condition. Hardware-independent tests
cover the boundary and invalid event timings. This limits the measurement
oracle, not backend matmul admission.

All six M256/K4096/N256 rows validate before and after timing. All 36 captured
windows exceed 10ms. Fresh packet: binary-i64-recorded-long-k.json. The comparison
proves every program IR/frame/member image remains byte-identical to the initial
packet; no kernel or performance-selector change is introduced. Each stage and
whole-program event region is independent. Public wall time includes binding,
transfers and readback; there is no baseline speedup claim.

| Storage | Placement | Producer events (us) | Consumer event (us) | Public call (ms) |
|---|---|---|---|---|
| f16 | lhs | 5.018 | 81.498 | 1.3616 |
| f16 | rhs | 5.055 | 81.406 | 1.3560 |
| f16 | both | 5.070, 5.112 | 81.370 | 2.0058 |
| bf16 | lhs | 5.108 | 81.438 | 1.2975 |
| bf16 | rhs | 5.117 | 81.782 | 1.3013 |
| bf16 | both | 5.107, 5.116 | 81.927 | 2.0181 |

## Final local gates

Full CPU: 30,854 passed / 3,721 declared skips, zero
failures. Full compiler: 2,755 passed, zero skips, executed-case gate passed.
Runtime ABI/diagnostic/pass cohort: 327 passed / 7 existing core-artifact skips;
public ABI observer exclusion: 22 passed. All four backend queues assess the
NVIDIA-owned wire fix. All five surfaces, claim lint, 33 generated docs, zero-error typing, Ruff,
audit ownership/lifecycle and Graphify pass. Exact-head CI/review is required
before merge.

Reproduce ordinary execution with the matching production provider. To execute
the three wide metadata cases, build the test-only TU using observer-command.json
and set TESSERA_NATIVE_FRAME_OBSERVER_LIB to that library. Production build uses
production-command.json. Adjust source/output paths for a fresh build directory;
never overwrite a binary used by a live process. Both artifacts must match the
source identity receipts. The recorder command remains:

```sh
python -m benchmarks.nvidia.record_native_tensor_binary --output /tmp/native-binary-i64.json
```

All five original aggregate programs remain open. This fixes the declared static
native route; general dynamic/layout/AD reconstruction is a separate obligation.

Negative-control pytest receipts are compressed without changing their raw bytes;
compressed-raw-receipts.json records both filenames and raw SHA-256 values.
