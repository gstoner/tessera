# 8.4.1. Address and GPR Range Checking

> RDNA3 ISA — pages 88–88

8.4.1. Address and GPR Range Checking
The hardware checks for both the address being out of range (BUFFER instructions only), and for the source or
destination SGPRs being out of range.

  Address Out-of-Range if              offset >= ( (stride==0 ? 1 : stride) * num_records).
                                       where "offset" is: inst_offset + {M0 or sgpr-offset}
                                       Any DWORDs that are out of range in memory from a buffer_load
                                       return zero. If a multi-DWORD request (e.g. S_BUFFER_LOAD_B256) is
                                       partially out of range, the DWORDs that are in range return data as
                                       normal, and the out-of-range DWORDs return zero.

  Source SGPR out of range             If any source data is out of the range of SGPRs (either partially or
                                       completely), the value 'zero' is used instead.

  Destination SGPR out of range        If the destination SGPR is partially or fully out of range, no data is
                                       written back to SGPRs for this instruction.
