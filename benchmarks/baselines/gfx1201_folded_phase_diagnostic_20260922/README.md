# gfx1201 folded-prefill phase diagnostic: attribution refused

Tajasarus (RX 9070 XT, gfx1201) ran the opt-in same-CTA phase probe from
`4368c825b866c3eb7b40761c5c17bbcb817e2665`. The instrumented kernel's
BF16 output matched the uninstrumented folded kernel before timing. Each CTA
recorded aggregate copy/barrier and WMMA/barrier ticks across K steps. All waves
now rendezvous before and after thread 0 records each phase boundary; the
host validated all slots and monotonic same-CTA intervals. Alternating HIP
events timed instrumented and uninstrumented launches separately.

| M×N×K | uninstrumented median | traced median | median overhead | CTA slots |
|---|---:|---:|---:|---:|
| 256×5120×8704 | 0.1729 ms | 0.1959 ms | +13.30% | 80 |
| 1024×17408×5120 | 1.1404 ms | 1.1827 ms | +3.71% | 1,088 |

This probe **fails structural perturbation admission** on both shapes: the
production HSACO has 32 FP8 WMMA instructions and four workgroup barriers,
while the traced HSACO has 64 and 24. Its phase fractions are therefore
properties of a different compiled schedule, **not evidence** that the
production kernel spends any particular fraction in copying or compute. The
packet marks `phase_attribution_admissible: false` and is not selector or
cost-model training data. The diagnostic reports only same-CTA clock deltas;
cross-CU clock consistency, read-cost distribution, and full IKF-P0 remain
unproved. No L3 stall or critical-path claim follows.

The useful result is a concrete gate for the next measurement design: preserve
the production loop's ISA structure and validate clock/perturbation before
interpreting phase time. Until then, prioritize uninstrumented HIP-event
timing, code-object ISA/resources, and controlled staging/weight-traffic
variants. See [small.json](small.json) and [large.json](large.json) for source
and image hashes, slots, samples, and explicit refusal reasons.
