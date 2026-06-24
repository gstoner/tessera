# 8.3. Scalar Memory Clauses and Groups

> RDNA4 ISA — pages 110–110

8.3. Scalar Memory Clauses and Groups
A clause is a sequence of instructions starting with S_CLAUSE and continuing for 2-63 instructions. Clauses
lock the instruction arbiter onto this wave until the clause completes.

A group is a set of the same type of instruction that happen to occur in the code but are not necessarily
executed as a clause. A group ends when a non-SMEM instruction is encountered. Scalar memory instructions
are issued in groups. The hardware does not enforce that a single wave executes an entire group before issuing
instructions from another wave.

Group restrictions:
  • INV must be in a group by itself and may not be in a clause

8.4. Alignment and Bounds Checking
SDST
   The value of SDST must be even for fetches of two DWORDs, or a multiple of four for larger fetches. If this
   rule is not followed, invalid data can result.

SBASE
   The value of SBASE must be even for S_BUFFER_LOAD (specifying the address of an SGPR that is a multiple
   of four). If SBASE is out-of-range, the value from SGPR0 is used.

OFFSET
   The value of OFFSET has no alignment restrictions.

8.4.1. Address and GPR Range Checking
The hardware checks for both the address being out of range (BUFFER instructions only), and for the source or
destination SGPRs being out of range.

Memory addresses are forced into alignment:
The base address is forced to DWORD alignment; DWORD or larger loads force memory address to DWORD
alignment; loads of 16-bit data force the address to 2-byte alignment, and byte loads have no forced alignment.

  Address Out-of-Range if               offset >= ( (stride==0 ? 1 : stride) * num_records).
                                        where "offset" is: IOFFSET + {M0 or sgpr-offset}
                                        Any DWORDs that are out of range in memory from a buffer_load
                                        return zero. If a multi-DWORD request (e.g. S_BUFFER_LOAD_B256) is
                                        partially out of range, the DWORDs that are in range return data as
                                        normal, and the out-of-range DWORDs return zero.

  Source SGPR out of range              If any source data is out of the range of SGPRs (either partially or
                                        completely), the value 'zero' is used instead.

  Destination SGPR out of range         If the destination SGPR is partially or fully out of range, no data is
                                        written back to SGPRs for this instruction.
