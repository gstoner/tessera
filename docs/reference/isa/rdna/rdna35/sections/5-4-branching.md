# 5.4. Branching

> RDNA3.5 ISA — pages 52–52

5.4. Branching
Branching is done using one of the following scalar ALU instructions. "SIMM16" is a sign-extended 16 bit
integer constant, treated as a DWORD offset for branches.

                                            Table 17. Branch Instructions
Instructions                  Description
S_BRANCH                      Unconditional branch. PC = PC + (SIMM16 * 4) + 4
S_CBRANCH_<test>              Conditional branch. Branch only if <condition> is true.
                              if (cond) PC = PC + (SIMM16 *4) +4; else NOP;
                              If SIMM16=0, the branch goes to the next instruction).
                              <cond> : SCC1, SCC0, VCCZ, VCCNZ, EXECZ, EXECNZ (SCC==1, SCC==0, VCC==0, VCC!=0,
                              EXEC==0, EXEC!=0)
S_CBRANCH_CDBGSYS             Conditional branch, taken if the COND_DBG_SYS status bit is set.
                              if (cond) PC = PC + (SIMM16 *4) +4; else NOP;
                              <cond> = SYS, USER, SYS_AND_USER, SYS_OR_USER.
S_CBRANCH_CDBGUSER            Conditional branch, taken if the COND_DBG_USER status bit is set.
S_CBRANCH_CDBGSYS_AND Conditional branch, taken only if both COND_DBG_SYS and COND_DBG_USER are set.
_USER
S_CBRANCH_CDBGSYS_OR_U Conditional branch, taken if either COND_DBG_SYS or COND_DBG_USER is set.
SER
S_SETPC_B64                   Directly set the PC from an SGPR pair: PC = SGPR-pair
S_SWAPPC_B64                  Swap the current PC with an address in an SGPR pair. SWAP (PC+4, SGPR-pair).
                              (result is: PC of this instruction + 4, zero extended)
S_GETPC_B64                   Retrieve the current PC value (does not cause a branch). (SGPR-pair = PC of this instruction
                              + 4, zero extended)
S_CALL_B64                    Jump to a subroutine, and save return address. SGPR_pair = PC+4; PC = PC+4+SIMM16*4.

For conditional branches, the branch condition can be determined by either scalar or vector operations. A
scalar compare operation sets the Scalar Condition Code (SCC) which then can be used as a conditional branch
condition. Vector compare operations set the VCC mask, and VCCZ or VCCNZ then can be used to determine
branching.

5.5. Work-groups and Barriers
Work-groups are collections of waves running on the same work-group processor that can synchronize and
share data. Up to 1024 work-items (16 wave64’s or 32 wave32’s) can be combined into a work-group. When
multiple waves are in a work-group, the S_BARRIER instruction can be used to force each wave to wait until all
other waves reach the same instruction; then, all waves continue. Work-groups of a single wave treat all
barrier instructions as S_NOP.

If a wave executes an S_BARRIER before all of the waves of the work-group have been created, the wave waits
until the work-group is complete.

Any wave may terminate early using S_ENDPGM, and the barrier is considered satisfied when the remaining
live waves reach their barrier instruction.
