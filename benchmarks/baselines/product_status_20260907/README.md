# Checked persistent products and scoped consumer proof

Owner: W2.4a / AD-RESIDUAL-EVAL-1; sync key IR-NATIVE-FOUNDATION-1.

`nvidia.json` is RTX 5070 / SM120 evidence; `rocm.json` is Radeon 8060S /
gfx1151 evidence. Both were measured in their owning WSL environment with
`benchmarks/record_product_status.py`. Packets bind the compiler binary,
implementation sources and recorder fingerprints.

Each packet checks three switch branches with repeated backward calls, one
imported multi-block function's forward/reverse execution, refusal of a CFG
exhausted at one dispatch step, and scoped reverse composition followed by a
native forward consumer. The latter crosses three streams and retires the
parent derivative before the child result is consumed. Its reference is
(2 * 2 * 2 * 3)^2 = 576 after the final square consumer.

Checked-status frames are synchronous. Scoped asynchronous composition uses
ordinary static products, not the checked-status route. These packets establish
correctness and lifetime ordering, not measured overlap or performance
promotion. Exported runtime-sized result shapes, nested assertions, arbitrary
Python CFG recovery and asynchronous checked-output exposure remain open.
