# 5.3. Instruction Clauses

> RDNA4 ISA — pages 55–55

Instructions                  Description
S_BARRIER_WAIT                Wait for all waves in the work-group to signal the barrier before proceeding. Synchronize
                              waves within a work-group. If not all waves in group have been created yet, waits for entire
                              group before proceeding. Waves that have ended do not prevent barriers from being
                              satisfied. Waves not in a work-group (or work-group size = 1 wave), treat this as S_NOP.
S_GET_BARRIER_STATE           Get the current barrier state, typically for context switching.

                                              Table 19. Control Instructions
Instructions                  Description
S_VERSION                     Does nothing (treated as S_NOP), but can be used as a code comment to indicate the
                              hardware version the shader is compiled for (using the SIMM16 field).
S_CODE_END                    Treated as an illegal instruction. Used to pad past the end of shaders.
S_SENDMSG                     Send a message upstream to the Interrupt handler or dedicated hardware. SIMM[9:0] is an
                              immediate value holding the message type. There is no "S_WAIT_*CNT" enforced before this.
S_SENDMSG_RTN_B32             Send a message upstream to that requests that some data be returned to an SGPR. Uses
S_SENDMSG_RTN_B64             KMcnt to track when data is returned.
                              SDST = SGPR to return to (or an aligned SGPR-pair for "_B64").
                              SSRC0 = enum, not an SGPR with the code for what data is requested. (see the message table
                              below).
                              If this is used to write VCC, then VCCZ is undefined.
S_SENDMSGHALT                 S_SENDMSG and then HALT.
S_ICACHE_INV                  Invalidate first-level shader instruction cache for the WGP associated with this wave.

                                       Table 20. Float Arithmetic State Instructions
Instructions                  Description
S_ROUND_MODE                  Set the round mode from an immediate: SIMM16[3:0]
S_DENORM_MODE                 Set the denorm mode from an immediate: SIMM16[3:0]

5.3. Instruction Clauses
An instruction clause is a series of instructions of the same type that are to be executed in an uninterrupted
sequence. Normally hardware may interleave instructions from different waves, but a clause can be used to
override that behavior and force the hardware to service only one wave for a given instruction type for the
duration of the clause, even if that leaves the execution hardware idle.

Clauses are defined and started using the S_CLAUSE instruction, and must contain only a single type of
instruction. The clause-type is implicitly defined by the type of instruction immediately following the s_clause.

Clause Types are:
  • Non-Flat Loads: Image, buffer, global, scratch, BVH and sample/gather
  • Non-Flat Stores: Image, buffer, global, scratch
  • Non-Flat Atomics: Image, buffer, global, scratch
  • Flat Load
  • Flat Store
  • Flat Atomic
  • LDS Indexed Load, Store, Atomic , BVH_stack (all one type)
  • SMEM
  • VALU
