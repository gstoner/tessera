# 7.9.1. WMMA Scheduling

> RDNA3.5 ISA — pages 86–86

7.9.1. WMMA Scheduling
Back-to-back dependent WMMA instructions require one V_NOP (or independent VALU op) between them if
the first instruction’s matrix D is the same or overlaps with the second instruction’s matrices A or B. Matrix A/B
can overlap C as long as C is distinct from D. The typical case is that C and D are the same.

In the table below "WMMA" is either WMMA or SWMMAC.

First Instruction   Second Instruction                              Requirement between First and Second Inst
                                   The cases below are required for correct function
WMMA                First WMMA’s matrix-D overlaps second           1 V_NOP or unrelated VALU instruction in between
Instruction         WMMA’s matrix A or B.                           two WMMA instructions is needed.
                    The cases below are only to avoid stalls and are not required for correct function
WMMA                WMMA instruction with same VGPR of previous Stall if the first and second instruction are not the
instruction         WMMA instruction’s Matrix D as Matrix C     same type of WMMA or use IMOD on SRC2 of the
                                                                second instruction.
WMMA                WMMA instruction with overlapped VGPR of        Hardware may Stall
instruction         previous WMMA instruction’s Matrix D as
                    Matrix C
WMMA                VALU instruction would read the previous        Hardware may stall VALU instruction
instruction         WMMA instruction’s Matrix D
