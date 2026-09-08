# Dynamic temporary storage and tracked reader retirement

CUDA evidence belongs to RTX 5070 / SM120; ROCm evidence belongs to Radeon 8060S /
gfx1151. Each packet binds its own compiler, source and recorder hashes.

- `dynamic-storage-*.json`: actual logical widths 0/1/2 and 1/2/3 copied between
  capacity-bounded dynamic allocations. Three iteration paths own distinct slots.
  Unused output capacity remains untouched. This is internal storage lowering;
  it is not an exported dynamic-shape persistent tape ABI.
- `readers-*.json`: four native tape exit paths, two readers per derivative
  generation, and frees queued on a third stream. The tracked region forbids a
  context barrier. Scope exit invalidates views; all copied derivative values
  are checked after retirement. Frame closure outside that region retains its
  existing barrier for unrestricted primal/residual exports.

No overlap, latency-bound, throughput or performance promotion is claimed.
Fault-injection tests independently cover partial frees and failed event records.
A failed free quarantines its frame for device teardown; it must not be retried
by normal close. Event failures retain explicit stream-completion recovery.

Native x86 multiway CFG forward/reverse proof lives in
`tests/unit/test_native_dynamic_cfg_storage.py`. Its exported GPU product remains
blocked by the retained bound-exhaustion assertion. Dynamic-storage GPU evidence
must not be cited as device proof for this CFG product.
