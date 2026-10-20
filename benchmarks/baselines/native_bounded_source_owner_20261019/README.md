# Retained native bounded source reverse owner

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.

This increment executes the compiler-exported bounded source-VJP members in a
retained C++ owner. Maximum device/staging capacities are allocated once;
actual Sq/Sk dimensions, spans and grids are validated as a temporary value
before committing runtime geometry. Existing static signatures are retained.
This is native C ABI execution, not ordinary public JIT/portable closure.

## Validation

- 48 distinct production-provider cases pass across the initial 36-case cohort
  and 12 added cooperative-width cases. The earlier 24-case run is a subset.
  Both FP16/BF16, four/five-member programs, permutations, requested gradients,
  readonly/zero inputs, five shrink/grow frames, guarded outputs, invalid/null
  dimensions, wrong spans and legacy/bounded invoke mismatch are covered.
  Pending writes on separate, legacy and per-thread streams and context recovery
  pass. Independent source-VJP comparisons and completed receipt clearing pass.
- The final 12-case test-only observer cohort proves unchanged allocation
  addresses/capacities/offsets and rejects private-root aliases. A different valid
  candidate geometry with incompatible spans leaves the old active geometry
  unchanged. Earlier observer runs are subsets, not additional coverage.
- 40 existing static source reverse cases pass against the production provider.
- Runtime ABI checks pass 25 cases initially; three generic artifact lifecycle
  cases then pass after building/exposing the matching runtime archive and link
  sidecar. Only the Apple hardware bridge remains skipped on WSL.
- Compiler, production/observer providers, SDK header, source and frozen test
  fixture identities and exact build commands are in SOURCE.json.

The original36-case fixture is preserved before diagnostic timer fields were
added. The wide cooperative cases and current recorder use the final helper.
The first timing attempt exposed a serial-only assumption in that helper: width257
selects cooperative_128. The native provider refused the wrong registration.
The helper now preserves the compiler-selected schedule; production code and
timed images were not changed. The failed attempt is retained.

## Measurements

Both complete orders pass 18 paired profiles (36 arms) using 21 alternating trials,
eight calls per arm per trial, reversed construction/profile/frame/initial-arm
orders, and identical source inputs and images. Independent numerical/root/guard
checks precede and follow timing. The normalization, saved-forward and
source-pullback images are byte-identical between inline and row arms; only the
backward decomposition differs.

The recorder separates frontend/native compilation, native owner preparation,
each native stage/program CUDA-event window, and completed ctypes-host calls.
Output allocation and timing-query overhead are outside the completed invocation
window. Event windows include enqueue effects and are not isolated GPU kernel
time. Native preparation occurs once per capacity/policy, not per runtime frame.

At Sq17/Sk19 and width257, rows/inline program-event ratios are 0.170-0.177.
Completed-host medians are 0.97-1.04ms for rows versus 4.94-5.41ms inline.
Tiny Sq=Sk=1 row arms are slower in both orders. Every sample and profile is
retained; no universal row-policy promotion or public latency claim follows.
Actual-shape native policy and ordinary public/portable route timing remain open.

## Reproduce on owning SM120 in host WSL

Use the matching compiler and build the production/observer providers with the
commands in SOURCE.json. Point TESSERA_NVIDIA_PTX_LAUNCH_LIB to the appropriate
provider; observer symbols are test-only.

```sh
python -m pytest -q tests/device/nvidia/test_bounded_source_reverse_owner.py -o junit_family=legacy
TESSERA_TEST_SOURCE_OWNER_OBSERVER=1 python -m pytest -q tests/device/nvidia/test_bounded_source_owner_storage.py
python -m benchmarks.nvidia.benchmark_bounded_source_owner --output owner-forward.json
python -m benchmarks.nvidia.benchmark_bounded_source_owner --reverse --output owner-reverse.json
```

Retained bounded public/portable adapters, full delivery gates and the original
five aggregate slices remain open.
