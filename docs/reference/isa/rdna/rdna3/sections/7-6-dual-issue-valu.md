# 7.6. Dual Issue VALU

> RDNA3 ISA — pages 77–77

Opcode                         inline                                                  OPSEL
DOT8_U32_U4                    use 32bit inline src0/1 (ignore OPSEL)                  OPSEL/OPSEL_HI on src0/1
DOT2_F32_F16                   use FP32 inline, supports OPSEL                         OPSEL/OPSEL_HI on src0/1
DOT2_F32_BF16                  upper16(FP32)/same as replicate (src0/1) ignore OPSEL   OPSEL/OPSEL_HI on src0/1
DOT2ACC_F32_F16                Duplicate lo to hi, ignore OPSEL                        none
DOT2ACC_F32_BF16               Duplicate lo to hi, ignore OPSEL                        none

7.6. Dual Issue VALU
The VOPD instruction encoding allows a single shader instruction to encode two separate VALU operations
that are executed in parallel. The two operations must be independent of each other. This instruction has
certain restrictions that must be met - hardware does not function correctly if they are not. This instruction
format is legal only for wave32. It must not be used by wave64’s. It is skipped for wave64.

The instruction defines 2 operations, named "X" and "Y", each with their own sources and destination VGPRs.
The two instructions packed into this one ISA are referred to as OpcodeX and OpcodeY.

  • OpcodeX sources data from SRC0X (a VGPR, SGPR or constant), and SRC1X (a VGPR);
  • OpcodeY sources data from SRC0Y (a VGPR, SGPR or constant), and SRC1Y (a VGPR).

The two instructions in the VOPD are executed at the same time, so there are no races between them if one
reads a VGPR and the other writes the same VGPR. The 'read' gets the old value.

Restrictions:
  • Each of the two instructions may use up to 2 VGPRs
  • Each instruction in the pair may use at most 1 SGPR or they may share a single literal
     ◦ Legal combinations for the dual-op: at most 2 SGPRs, or 1 SGPR + 1 literal, or share a literal.
  • SRC0 can be either a VGPR or SGPR (or constant)
  • VSRC1 can only be a VGPR
  • Instructions must not exceed the VGPR source-cache port limits
      ◦ There are 4 VGPR banks (indexed by SRC[1:0]), and each bank has a cache
      ◦ Each cache has 3 read ports: one dedicated to SRC0, one dedicated to SRC1 and one for SRC2
         ▪ A cache can read all 3 of them at once, but it can’t read two SRC0’s at once (or SRC1/2).
      ◦ SRCX0 and SRCY0 must use different VGPR banks;
      ◦ VSRCX1 and VSRCY1 must use different banks.
         ▪ FMAMK is an exception : V = S0 + K * S1 ("S1" uses the SRC2 read port)
      ◦ If both operations use the SRC2 input, then one SRC2 input must be even and the other SRC2 input
        must be odd. The following operations use SRC2: FMAMK_F32 (second input operand);
        DOT2ACC_F32_F16, DOT2ACC_F32_BF16, FMAC_F32 (destination operand).
      ◦ These are hard rules - the instruction does not function if these rules are broken
  • The pair of instructions combined have the following restrictions:
     ◦ At most one literal constant, or they may share the same literal
      ◦ Dest VGPRs: one must be even and the other odd
      ◦ The instructions must be independent of each other
  • Must not use DPP
  • Must be wave32.
