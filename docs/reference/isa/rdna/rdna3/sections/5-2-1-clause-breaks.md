# 5.2.1. Clause Breaks

> RDNA3 ISA — pages 49–49

  • SMEM
  • VALU

May also be in a clause ("clause internal instructions"):
  • S_DELAY_ALU is legal inside a clause (internal) but is pointless.
      ◦ S_DELAY_ALU must not occur within a VALU clause.
  • S_NOP and S_SLEEP may be used inside a clause, but the first instruction of the clause must be the clause-
    type instruction (ALU, memory).

Cannot be in a clause:
  • Instructions of a different type those of the clause type are illegal
  • S_CLAUSE
  • S_ENDPGM
  • SALU, Export, branch, message, GDS, lds_param_load, lds_direct_load
  • S_WAITCNT, S_WAIT_IDLE, S_WAIT_DEPCTR

S_CLAUSE defines both the total length of the clause, and how often it should be broken to allow other waves a
chance to go. For instance, it could say: clause of 16 instructions, but break after every 4th to allow a higher
priority wave to get access to the execution unit. "clause internal instructions" count against this clause size.

If a clause defines regular clause breaks (e.g. a clause of 16 instructions, but break every 4th), the first
instruction of each sub-clause (every 4 instructions) must be of the clause-type, not a "clause internal
instruction". Each group of instructions must have at least two of the clause-type of instructions. E.g. a clause of
12 VALU instructions broken up into 4 groups of 3 instructions - each group of 3 instructions must have at least two
VALU instructions. Clause groups with only 1 VALU instruction per group make no sense - they are no longer a clause.

If the first instruction in a VALU clause has EXEC==0, then the clause is ignored and instructions are issued as
if there were no clause. If the VALU clause starts with EXEC!=0 but EXEC becomes zero in the middle of the
clause, the clause continues until the last instruction of the specified clause.

If an S_DELAY_ALU is needed before starting a clause, the order must be:

       S_DELAY_ALU // must not come immediately after S_CLAUSE - that inst declares clause type
       S_CLAUSE
       <first instruction in clause>

If the first instruction after S_CLAUSE is skipped (e.g. due to EXEC==0, or VMEM-load skipped due to EXEC==0
and VMcnt==0) then then a clause is not started. Subsequent instructions within what would have been the
clause that are not skipped and are still executed but individually, not as part of a clause.

5.2.1. Clause Breaks
The following conditions can break a clause:

 1. VALU exception (trap) breaks a VALU clause
 2. Host commands to wave (halt, resume, single step, etc) breaks all active clauses.
    Context-save breaks clauses of affected waves.
    This allows the host to read and write SGPRs & VGPRs while debugging. If clauses were not broken by host
