#pragma once

namespace tprof {

// Attach the Tessera runtime C-ABI profiling callback to the current tprof
// process.  The Tessera runtime remains independent of tprof; this helper lives
// entirely in tools/profiler and maps runtime callback events into tprof event
// categories when both libraries are linked by a profiler harness.
bool attach_tessera_runtime_trace(bool enable_tessera_profiling = true);

// Clear the Tessera runtime profiling callback after a successful attach.
void detach_tessera_runtime_trace();

} // namespace tprof
