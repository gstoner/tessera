# Native bounded query-row prerequisite

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.
Base: merged PR990, d0579fdb1546fda5278b700dbee8c77a251d339b.

## Established scope

This packet proves one native prerequisite for bounded original-source
RMSNorm(Q) -> saved-LSE attention reverse. It does not prove that composed
bounded program, its public JIT or portable replay. Those integrations remain
open. Python contains test/reference/recorder code; native C++ MLIR owns the
lowering and the bounds proof.

Schema 8 retains dynamic query tensor types and fixed B/H/D in Graph.
Schedule seals the sequence capacity and physical row count. Tile/Target
retain two readonly pointers, one output pointer and actual row/column i64
scalars. Positive/capacity/prefix/column guards precede access; per-thread
logical and physical bounds dominate the scalar row SSA. Static schemas
1/6/7 keep their original meanings and recipes.

The shared Tile scalar-view proof recognizes an actual enclosing conjunction.
Disjunction, negation, select, a different bound and else-region conditions
retain access masking. There is no trusted in-bounds metadata flag.

## Tests and identities

- 410 focused compiler/view/diagnostic/pass/audit cases pass. Four separate
  compiler cases additionally lower actual native source-attention AD adjoints.
- RTX5070/SM120, driver610.88: four bounded profiles and eight static
  regressions pass. Each bounded profile reuses one image across query
  shrink/grow; random/tiny/zero oracles, inactive tails, null-input invalid
  geometry, column mismatch and extreme i64 rows are checked. Reruns of
  these same twelve tests are not additional coverage.
- Four additional RTX5070 cases execute the actual pure normalization adjoint
  emitted by native source-attention paired AD. Both formats and widths retain
  dynamic-query replay, independent random/tiny/zero mathematics and inactive
  storage checks. These are standalone members, not a composed VJP.
- RX9070XT/gfx1201: 23 owning quantizer/parent-view/source-pullback cases
  and 44 adjacent unit/compiler cases pass. This is separate AMD evidence.
- Sixteen static SM120/ROCm profiles retain identical Schedule, Tile,
  Target and native text against the immutable merged compiler.
- Final compiler SHA256:
  27e3cd12649947cf33ccd0d44f6e6da293ef452d9b1a17dc5c2723ff6d1104e9.
  Final timing source and compiler hashes are checked against actual files.
  Resource records bind the very same four wide-row PTX payloads that were timed.

No broad CPU/production-compiler publication certificate is asserted here.
Generated documents were regenerated; final comprehensive gates and delivery
remain part of the composed integration.

## Engineering loop and measurements

The timed body is the older native RMSNorm/ReLU paired-AD row fixture.
The actual pure attention-source adjoint has separate numerical tests; these
timings are not its performance certificate.

Each final order has twelve storage/query/width profiles, twenty-one
alternating trials and thirty-two launches per arm. The second reverses
profile order, query order and the initial A/B arm. Both arms use identical
inputs; independent numerical checks and bounded/static bit equality pass
before and after timing using raw uint8 storage views, including signed zeros.
These orders repeat the same twelve profiles. Earlier numerical-only repeats
are retained separately and do not add coverage.

| Variant | Width257 result |
| --- | --- |
| Initial dynamic view extents | Consistent roughly1.5-1.7x bounded/static penalty |
| Static capacity views alone | Penalty remains; reject as a performance solution |
| Explicit dominating physical-row guard | Penalty reduced to roughly1.14-1.31x |
| Dominating guard plus native conjunction proof | Final byte-checked forward0.96472-1.00193; reverse0.97596-1.00328 bounded/static |

Width35 remains mixed: final byte-checked orders span roughly0.799-1.033.
The earlier numerical-only repeat spans0.848-1.075 and is retained separately. No universal
speedup or default promotion is claimed.

Actual PTX/ptxas attribution for both half formats: six static loads throughout;
initial thirteen branch mnemonics become six with the proof. Registers rise
from32 to39, matching the final static control; all report zero stack, spill
loads/stores and barriers. The final bounded kernel retains three extra
entry predicates relative to static. These are emitted-code/resource facts,
not hardware profiler stall counters.

The packet separates compilation, resident raw-member CUDA-event dispatch
windows and completed raw launch/synchronize host windows. Events may include
enqueue gaps. Neither domain is public end-to-end VJP or isolated profiler
kernel time.

## Reproduction and historical experiments

On owning Super-Bear with a matching native compiler/session runtime:

    python -m benchmarks.nvidia.benchmark_bounded_source_row --samples 21 --reps 32 --output forward.json
    python -m benchmarks.nvidia.benchmark_bounded_source_row --reverse --samples 21 --reps 32 --output reverse.json

Both commands require actual SM120 detection. The recorder queries GPU name,
compute capability, driver and UUID and records compiler, provider, source and
per-stage hashes.

Raw packets/logs are compressed losslessly. DECISION.json indexes outcomes.
experiments/ preserves the initial native patch and successive capacity,
dominating-guard and conjunction-proof source deltas. Initial fixture syntax
debugging logs stay private; the two committed red logs are actual compiler
contract/proof counterexamples. No binaries are committed.

## Remaining original objective

Bounded normalization member projection, original paired-source outlining,
capacity-owned composed execution, public/portable replay, wider AD/layout
and model-quality/performance requirements remain open. This is not closure
of any complete five-slice program or sibling physical admission.
