# Chapter 5. Program Flow Control

> RDNA3 ISA — pages 47–47

Chapter 5. Program Flow Control
Program flow control is programmed using scalar ALU instructions. This includes loops, branches, subroutine
calls, and traps. The program uses SGPRs to store branch conditions and loop counters. Constants can be
fetched from the scalar constant cache directly into SGPRs.

5.1. Program Control
The instructions in the table below control the priority and termination of a shader program, as well as provide
support for trap handlers.

                                         Table 12. Wave Termination and Traps
Instructions                  Description
S_ENDPGM                      Terminates the wave. It can appear anywhere in the shader program and can appear
                              multiple times.
S_ENDPGM_SAVED                Terminates the wave due to context save. Intended for use only within the trap handler.
S_TRAP                        Jump to the trap handler and pass in 8-bit TRAP id from SIMM[7:0].
                              It does not affect SCCZ.

                                 <wait for outstanding instructions to finish>
                                 {TTMP1,TTMP0} = {7'h0,HT[0],trapID[7:0],PC[47:0]}
                                 PC = TBA (trap base address)
                                 PRIV = 1

                              "HT" : 1 = this is a host-initiated trap, 0 = user (s_trap). Host traps cause the shader
                              hardware to generate an S_TRAP instruction. Note: the save-PC points to the S_TRAP
                              instruction. TRAPID 0 is reserved for hardware use.
S_RFE_B64                     Return from exception (trap handler) and continue.
                              Start executing at PC (trap handler must increment PC past the faulting instruction).
                              MOVE PC, <src> ; STATUS.PRIV = 0.
                              This instruction may only be used within a trap handler.
S_SETKILL                     Set the KILL bit to 1, causing the shader to s_endpgm immediately. Used primarily for
                              debugging 'kill' wave-command behavior.
S_SETHALT                     Set the HALT bit to the value of SIMM16[0].
                              Setting to 1 halts the shader when PRIV=0 (not in trap handler);
                              setting to 0 resumes the shader (can only occur in trap handler).
                              Fatal Halt control: SIMM16[2] 1 : set fatal halt; 0 : clear fatal halt.

                              Table 13. Dependency, Delay and Scheduling Instructions
Instructions                  Description
S_NOP                         NOP. Repeat SIMM16[3:0] times. (1..16)
                              Like a short version of S_SLEEP
S_SLEEP                       Cause a wave to sleep for approx. 64*SIMM16[6:0] clocks.
                              "s_sleep 0" sleeps the wave for 0 cycles.
S_WAKEUP                      Causes one wave in a work-group to signal all other waves in the same work-group to wake
                              up from S_SLEEP early. If waves are not sleeping, they are not affected by this instruction.
S_SETPRIO                     Set 2-bits of USER_PRIO: user-settable wave priority. 0 = low, 3 = high.
                              Overall wave priority is: {MIN(3,(SysPrio[1:0] + UserPrio[1:0])), WaveAge[3:0]}
