# 9.5. Alignment

> RDNA4 ISA — pages 123–123

9.5. Alignment
Formatted ops such as BUFFER_LOAD_FORMAT_* must be aligned as follows:
  • 1-byte formats require 1-byte alignment
  • 2-byte formats require 2-byte alignment
  • 4-byte and larger formats require 4-byte alignment

Memory alignment enforcement for non-formatted ops is controlled by a configuration register:
SH_MEM_CONFIG.alignment_mode.

Atomics must be aligned to the data size, or they trigger a MEMVIOL.

Options are:
 0. : DWORD - hardware automatically aligns request to the smaller of: element-size or DWORD.
