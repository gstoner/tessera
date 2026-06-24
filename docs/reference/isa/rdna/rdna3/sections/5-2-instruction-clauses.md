# 5.2. Instruction Clauses

> RDNA3 ISA — pages 48–48

Instructions                  Description
S_CLAUSE                      Begin a clause consisting of instructions matching the instruction after the s_clause. The
                              clause length is: (SIMM16[5:0] + 1), and clauses must be between 2 and 63 instructions.
                              SIMM16[5:0] must be 1-62, not 0 or 63. The clause breaks after every N instructions, N =
                              simm[11:8] (0 - 15; 0 = no breaks)
S_BARRIER                     Synchronize waves within a work-group. If not all waves in group have been created yet,
                              waits for entire group before proceeding. Waves that have ended do not prevent barriers
                              from being satisfied. Waves not in a work-group (or work-group size = 1 wave), treat this as
                              S_NOP.

                                             Table 14. Control Instructions
Instructions                  Description
S_VERSION                     Does nothing (treated as S_NOP), but can be used as a code comment to indicate the
                              hardware version the shader is compiled for (using the SIMM16 field).
S_CODE_END                    Treated as an illegal instruction. Used to pad past the end of shaders.
S_SENDMSG                     Send a message upstream to the Interrupt handler or dedicated hardware. SIMM[9:0] is an
                              immediate value holding the message type. There is no "s_waitcnt" enforced before this.
S_SENDMSG_RTN_B32             Send a message upstream to that requests that some data be returned to an SGPR. Uses
S_SENDMSG_RTN_B64             LGKMcnt to track when data is returned. (or an aligned SGPR-pair for "_B64").
                              SDST = SGPR to return to.
                              SSRC0 = enum, not an SGPR with the code for what data is requested. (see the message table
                              below).
                              If this is used to write VCC, then VCCZ is undefined.
S_SENDMSGHALT                 S_SENDMSG and then HALT.
S_ICACHE_INV                  Invalidate first-level shader instruction cache for the WGP associated with this wave.

5.2. Instruction Clauses
An instruction clause is a group of instructions of the same type that are to be executed in an uninterrupted
sequence. Normally hardware may interleave instructions from different waves, but a clause can be used to
override that behavior and force the hardware to service only one wave for a given instruction type for the
duration of the clause, even if that leaves the execution hardware idle.

Clauses are defined and started using the S_CLAUSE instruction, and must contain only a single type of
instruction. The clause-type is implicitly defined by the type of instruction immediately following the clause.

Clause Types are:
  • Image (no sampler) load
  • Image store
  • Image atomic
  • Image sample
  • Buffer / Global / Scratch load
  • Buffer / Global / Scratch store
  • Buffer / Global / Scratch atomic
  • Flat load
  • Flat store
  • Flat atomic
  • LDS load / store / atomic / bvh_stack
  • IMAGE_BVH
