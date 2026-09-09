# Completion state and asynchronous status fan-in

Owners: W4-PRODUCT-1 and W2.4a. Synchronization key:
`COMPLETION-STATE-FANIN-2026-09-08`. Uncommitted continuation after PR #736.

## Native source completion

Native LLVM CPU differential tests on Princess-Luna compare ordinary Python
execution against recovered source for nested loops, tuple returns, zero-trip
loops, handled raises crossing loop boundaries, try/except/else/finally, caught
assertions and finally overrides. Explicit state tests compare both returned
values and mutated arrays for aliased inputs, including writes before/after
handlers and loop returns. Capture must leave inputs unchanged; changed alias
topology, partial overlaps, readonly aliases, mutable return aliases and use
after close refuse.

This is opt-in bounded source capture, not arbitrary Python or general AD closure.
Expanded loops retain the 16-iteration/256-continuation limits. Only explicit,
statically handled ValueError, RuntimeError and AssertionError edges are admitted.
Uncaught errors have no exported Python exception ABI; native exhaustion/assertion
failure retains the host abort boundary.

The native CPU state adapter serializes exact full-tensor aliases and emits state
as SSA results. It snapshots inputs, invokes native code and copies declared state
back after completion. It requires exclusive ownership during invocation.
Partial/strided mutation, returned mutable aliases, concurrent external observers,
GPU mutation and general JIT integration remain outside this proof.

## GPU status fan-in

`fanin-nvidia.json` records RTX 5070 / SM120; `fanin-rocm.json` records
Princess-Luna / gfx1151. Each independently exercises all sixteen combinations of
one capture status and three upstream derivative statuses. Only all-success
exposes the derivative (24 in every element); any failure produces the named
product-guard failure. Unexpected runtime errors fail recording.

Failures are stream-ordered device-to-device status writes, not hardware faults.
Every prerequisite retains a scoped reader until the consumer is submitted.
Status-only prerequisites do not provide additional cotangent operands.
The recorder forbids host capture checks and explicit context synchronization
inside composition; final result inspection is an explicit host boundary.
All frames retire with no remaining owned buffers.

The compiler/package admits one through eight incoming statuses, with native
contract tests at counts three, four and eight. These packets physically validate
four, not eight. They include compiler, recorder, source and package fingerprints.
No timing, overlap, performance promotion, Apple, RDNA4 or CDNA claim transfers.

## Validation

The assertions-enabled LLVM23 host passed 255 compiler/registry tests.
Audit and governance checks passed 25 tests. Ruff and the zero-error mypy ratchet
passed. The frozen-tree WSL unit suite passed **18,569 tests**, with 2,231
skipped and 870 slow tests deselected, in 635 seconds. A subsequent docstring-only
clarification does not change execution; all 59 focused tests passed again
afterward. `validation.json` records those tests and native compiler/library/source
fingerprints.
