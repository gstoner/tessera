# 5.2. Program Control Instructions

> RDNA4 ISA — pages 53–54

Chapter 5. Program Flow Control
Program flow control is programmed using scalar ALU instructions. This includes loops, branches, subroutine
calls, and traps. The program uses SGPRs to store branch conditions and loop counters. Constants can be
fetched from the scalar constant cache directly into SGPRs.

The instructions in the this chapter control the priority and termination of a shader program, as well as
provide support for trap handlers.

5.1. Program Flow Control Instruction Formats
These instructions are encoded in one of these microcode formats, shown below:

Name           Size       Function
SOPP           32 bit     SALU program control op with a 16-bit immediate
                          constant
SOP1           32 bit     SALU op with 1 input

Each of these instruction formats uses some of these fields:

Field                    Description
OP                       Opcode: instruction to be executed.
SDST                     Destination SGPR, M0, NULL or EXEC.
SSRC0                    First source operand.
SIMM16                   Signed immediate 16-bit integer constant.

5.2. Program Control Instructions
                                        Table 17. Wave Termination and Traps
Instructions                  Description
S_ENDPGM                      Terminates the wave. It can appear anywhere in the shader program and can appear
                              multiple times.
S_ENDPGM_SAVED                Terminates the wave due to context save. Intended for use only within the trap handler.

Instructions                  Description
S_TRAP                        Jump to the trap handler and pass in 4-bit TRAP id from SIMM[3:0].

                                    <wait for outstanding instructions to finish>
                                    { TTMP1, TTMP0 } = {TrapID[3:0], zeros, PC[47:0]} // TrapID=0 unless the exception is
                                    "S_TRAP"
                                    PC = TBA (trap base address)
                                    PRIV = 1

                              Host traps cause the shader hardware to generate an S_TRAP instruction. Note: the save-PC
                              points to the S_TRAP instruction. TRAPID 0 is reserved and should not be used.
S_RFE_B64                     Return from exception (trap handler) and continue.
                              Start executing at PC stored in SGPRs or TTMPs.
                              MOVE PC, <src> ; STATUS.PRIV = 0.
                              This instruction may only be used within a trap handler.
S_SETHALT                     Set the HALT or FATAL_HALT bit to the value of SIMM16[0].
                              SIMM16[0]: 1 = halt, 0 = resume. Applies to HALT or FATAL_HALT (see SIMM16[2])
                              SIMM16[2]: 1 = set FATAL_HALT bit; 0 = set HALT bit.

                              The user shader can set the HALT bit, but not the FATAL_HALT bit;
                              the trap handler (priv=1) is allowed to set either bit.

                               HALT == 1               halts a wave when PRIV==0, but not when PRIV==1
                               FATAL_HALT == 1         halts a wave when PRIV==0 and when PRIV==1

                              The HALT==1 bit causes shaders to halt when PRIV==0, but is ignored when PRIV==1. If
                              HALT is set to 1 while PRIV==1, it only takes effect and halts the shader after returning to
                              the user shader (S_RFE, which makes PRIV=0).
                              The FATAL_HALT==1 bit causes shaders to halt regardless of PRIV value. Once
                              FATAL_HALTed, a shader cannot un-halt itself – only host-commands can do that.

                               Table 18. Dependency, Delay and Scheduling Instructions
Instructions                  Description
S_NOP                         NOP. Repeat SIMM16[6:0] times. [1..128]
                              Like a short version of S_SLEEP
S_SLEEP                       Cause a wave to sleep for approximately 64*SIMM16[6:0] clocks, or can be woken up early
                              by S_WAKEUP instructions.
                              "s_sleep 0" sleeps the wave for 0 cycles. If SIMM16[15]==1, sleep forever (until wakeup, trap or
                              kill).
S_SLEEP_VAR                   Cause a wave to sleep for approximately SGPR_value[6:0]*64 cycles. This is also woken up
                              by S_WAKEUP instructions.
S_WAKEUP                      Causes one wave in a work-group to signal all other waves in the same work-group to wake
                              up from S_SLEEP/S_SLEEP_VAR early. If waves are not sleeping, they are not affected by
                              this instruction.
S_SETPRIO                     Set 2-bits of USER_PRIO: user-settable wave priority. 0 = low, 3 = high.
                              Overall wave priority is: {MIN(3,(SysPrio[1:0] + UserPrio[1:0])), WaveAge[3:0]}
S_CLAUSE                      Begin a clause consisting of instructions matching the type of instruction after the s_clause.
                              The clause length is: (SIMM16[5:0] + 1), and clauses must be between 2 and 63 instructions.
                              SIMM16[5:0] must be 1-32, not 0 or 63.
S_BARRIER_SIGNAL         Signal that a wave has as arrived at a barrier. Synchronize waves within a work-group.
S_BARRIER_SIGNAL_ISFIRST Waves not in a work-group (or work-group size = 1 wave), treat this as S_NOP. ISFIRST
                         variant returns to SCC if this is the first wave to signal.
