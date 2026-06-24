# 8.3. Scalar Memory Clauses and Groups

> RDNA3 ISA — pages 87–87

S_GL1_INV and S_DCACHE_INV do not have any address or data arguments.

8.2. Dependency Checking
Scalar memory loads can return data out-of-order from how they were issued; they can return partial results at
different times when the load crosses two cache lines. The shader program uses the LGKMcnt counter to
determine when the data has been returned to the SDST SGPRs. This is done as follows.

  • LGKMcnt is incremented by 1 for every fetch of a single DWORD, or cache invalidates.
  • LGKMcnt is incremented by 2 for every fetch of two or more DWORDs.
  • LGKMcnt is decremented by an equal amount when each instruction completes.

Because the instructions can return out-of-order, the only sensible way to use this counter is to implement
"S_WAITCNT LGKMcnt 0"; this imposes a wait for all data to return from previous SMEMs before continuing.

Cache invalidate instructions are not known to have completed until the shader waits for LGKMcnt==0.

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
   The value of SBASE must be even for S_BUFFER_LOAD (specifying the address of an SGPR which is a
   multiple of four). If SBASE is out-of-range, the value from SGPR0 is used.

OFFSET
   The value of OFFSET has no alignment restrictions.
