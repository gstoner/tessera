# Automatic AD results and checked pool retirement — 2026-09-08

`record_checked_retirement.py` validates each target independently:

- The compiler exports an AD forward product and generates its capacity copy,
  logical-length sidecar and status checks. Device-computed lengths 0, 2 and 4
  agree with the reference; undersized capacity refuses result exposure.
- Eight successful checked derivative chains allocate data and status through
  the stream-ordered pool. Parent retirement waits for the registered child
  reader; child retirement waits for its numerical-oracle reader.
- One explicitly injected upstream failure refuses a generic child reader,
  while both generations still retire safely.
- Every iteration returns to the original frame buffer count. Healthy
  generation submission/retirement is instrumented to reject a context barrier.

An explicit child status wait/read is used for the numerical oracle. Retirement
itself needs no host success readback; failed generations may also be reclaimed.
The packets do not claim asynchronous frame capture/close, arbitrary external
reader tracking, dynamic GPU backward inputs, multi-result/multidimensional
AD output closure or measured overlap. The result generator currently admits
one rank-one result and static external inputs.
