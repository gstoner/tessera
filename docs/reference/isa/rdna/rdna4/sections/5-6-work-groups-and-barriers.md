# 5.6. Work-groups and Barriers

> RDNA4 ISA — pages 58–58

branches.

PC value: S_BRANCH, S_CBRANCH, S_SWAPPC, S_CALL, and S_GETPC operate on the PC when it is pointing
to the instruction after this one. So "S_BRANCH 0" is like a NOP, not an infinite loop.

                                             Table 23. Branch Instructions
Instructions        Description
S_BRANCH            Unconditional branch. PC = PC + (SIMM16 * 4)
S_CBRANCH_          Conditional branch. Branch only if <condition> is true.
<cond>              if (cond) PC = PC + (SIMM16 *4); else NOP;
                    If SIMM16=0, the branch goes to the next instruction).
                    <cond> : SCC1, SCC0, VCCZ, VCCNZ, EXECZ, EXECNZ
                    (SCC==1, SCC==0, VCC==0, VCC!=0, EXEC==0, EXEC!=0)
S_SETPC_B64         Directly set the PC from an SGPR pair: PC = SGPR-pair
S_SWAPPC_B64        Swap the current PC (pointing to instruction after this) with an address in an SGPR pair. SWAP (PC,
                    SGPR-pair).
S_GETPC_B64         Retrieve the current PC value (zero-extended). This does not cause a branch. (SGPR-pair = PC of next
                    instruction)
S_CALL_B64          Jump to a subroutine, and save return address. SGPR_pair = PC; PC = PC+SIMM16*4.

For conditional branches, the branch condition can be determined by either scalar or vector operations. A
scalar compare operation sets the Scalar Condition Code (SCC) which then can be used as a conditional branch
condition. Vector compare operations set the VCC mask, and VCCZ or VCCNZ then can be used to determine
branching.

5.6. Work-groups and Barriers
Work-groups are collections of waves running on the same WGP that can synchronize and share data. Up to
1024 work-items (16 wave64’s or 32 wave32’s) can be combined into a work-group. When multiple waves are in
a work-group, the barrier instructions can be used to cause each wave to wait until all other waves reach the
same instruction; then, all waves continue.

The barrier operation is divided into two parts: Signal (arrive) and Wait. Each wave first issues an
S_BARRIER_SIGNAL to indicate that other waves may proceed. Later the wave issues S_BARRIER_WAIT that
causes the wave to wait until every wave in the work-group has also issued an S_BARRIER_SIGNAL. When all
member waves have signaled, the barrier resets and the waves may proceed through their S_BARRIER_WAITs.

Barrier Valid
   A barrier is valid once all of the waves in the work-group have been created. Until then, the number of waves
   in the work-group is unknown.

Barrier Complete
   A barrier is complete when each of the waves in the work-group have signaled (or terminated).
   A barrier cannot complete until it is valid, but waves may signal before the barrier is valid.

It is allowable for a wave to terminate early instead of signaling, and the remaining waves can still use the
barrier and when the remaining waves have all signaled, they can pass through an S_BARRIER_WAIT. Waves
should not signal then exit - this could leave the barrier in an unusable state.

Work-groups consisting of a single wave, and waves not in a work-group, treat all barrier instructions as
