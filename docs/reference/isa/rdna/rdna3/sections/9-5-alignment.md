# 9.5. Alignment

> RDNA3 ISA — pages 100–100

                                         Example of Buffer Swizzling

9.5. Alignment
Formatted ops such as BUFFER_LOAD_FORMAT_* must be aligned as follows:
  • 1-byte formats require 1-byte alignment
  • 2-byte formats require 2-byte alignment
  • 4-byte and larger formats require 4-byte alignment

Atomics must be aligned to the data size, or triggers a MEMVIOL.

Memory alignment enforcement for non-formatted ops is controlled by a configuration register:
SH_MEM_CONFIG.alignment_mode.

Options are:
