# Bounded native result and reader proof — 2026-09-08

Reproduce each packet with `benchmarks/record_deep_native_contracts.py` on its
owning target and the recorded compiler build. Source/recorder/compiler hashes
are carried in each packet. Both SM120 and gfx1151 passed:

- Device-computed logical lengths 0, 3 and 8 with capacity 8.
- Missing or oversized logical length refuses public exposure.
- Nested guard failure leaves output sentinels unchanged, keeps shape at its
  initialized invalid sentinel, and sets failure status.
- A dependent backward reads a successful parent's derivative after an event
  wait, without a parent host-status wait before enqueue.
- Explicitly injected upstream failure prevents successful child exposure.

The public producer is an explicit rank-one buffer program, not automatic
shape-varying AD lowering. Reader allocation/release still has synchronous
context barriers. No asynchronous reclamation, measured overlap or performance
promotion is claimed. Apple and x86 execution proof is separate.
