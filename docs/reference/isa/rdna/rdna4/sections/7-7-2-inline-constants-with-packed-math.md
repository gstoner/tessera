# 7.7.2. Inline Constants with Packed Math

> RDNA4 ISA — pages 92–93

7.7.1. Scalar Constants with Packed Math
This section defines the rules for scalar constants in packed math operations and how much data is read per
operand.

Input Type       PK_F16                                PK_F32
                 All cases use OPSEL to select high
                 or low 16 bits per op.
VGPR             1 vgpr                                2 vgprs. opsel selects high/low
                                                       dword
SGPR             1 SGPR                                1 SGPR
Literal32        like SGPR                             replicates literal to 64 bits. no
M0                                                     opsel (redundant)
{31’h0, SCC}
EXEC             invalid                               invalid
Inline Const     {16’h0, const}                        32-bit const. no opsel

7.7.2. Inline Constants with Packed Math
Inline constants may be used with packed math, but they require the use of OPSEL. Inline constants produce a
value in only the low 16-bits of the 32-bit constant value. Inline constants used with float 16-bit sources produce
an F16 constant value. Without using OPSEL, only the lower half of the source would contain the constant. To
use the inline constant in both halves, use OPSEL to select the lower input for both low and high sources.

BF16 uses 32-bit float constants and then the BF16 operand selects the upper 16 bits of the FP32 constant
(matches the definition of BF16).

For WMMA and DOT instructions sourcing 16-bit data, the LSBs of the inline constant value are used.

Any packed math instructions (excluding WMMA) that have source float data sizes less than 16 bits do not work
with inline constants. 8-bit and 4-bit integer inlines constants work as expected.

Opcode                            inline                                                   OPSEL
DOT4_I32_IU8                      use 32bit inline src0/1 (ignore OPSEL)                   OPSEL/OPSEL_HI on src0/1
DOT8_I32_IU4                      use 32bit inline src0/1 (ignore OPSEL)                   OPSEL/OPSEL_HI on src0/1
DOT4_U32_U8                       use 32bit inline src0/1 (ignore OPSEL)                   OPSEL/OPSEL_HI on src0/1
DOT8_U32_U4                       use 32bit inline src0/1 (ignore OPSEL)                   OPSEL/OPSEL_HI on src0/1
DOT2_F32_F16                      use FP32 inline, supports OPSEL                          OPSEL/OPSEL_HI on src0/1
DOT2_F32_BF16                     upper16(FP32)/same as replicate (src0/1) ignore OPSEL    OPSEL/OPSEL_HI on src0/1
DOT2ACC_F32_F16                   Duplicate lo to hi, ignore OPSEL                         none
DOT2ACC_F32_BF16                  Duplicate lo to hi, ignore OPSEL                         none

7.8. Dual Issue VALU (VOPD)
The VOPD instruction encoding allows a single shader instruction to encode two separate VALU operations
that are executed in parallel. The two operations must be independent of each other. This instruction has
certain restrictions that must be met - hardware does not function correctly if they are not. This instruction
format is legal only for wave32. It must not be used by wave64’s.

The instruction defines 2 operations, named "X" and "Y", each with their own source and destination VGPRs.
The two instructions packed into this one ISA are referred to as OpcodeX and OpcodeY.

  • OpcodeX sources data from SRC0X (a VGPR, SGPR or constant), and SRC1X (a VGPR);
  • OpcodeY sources data from SRC0Y (a VGPR, SGPR or constant), and SRC1Y (a VGPR).

Restrictions:
  • Each of the two instructions may use up to 2 VGPRs
  • Each instruction in the pair may use at most 1 SGPR or they may share a single literal
     ◦ Legal combinations for the dual-op: at most 2 SGPRs, or 1 SGPR + 1 literal, or share a literal.
           ▪ SCC and EXEC as data count as one SGPR
  • SRC0 can be either a VGPR or SGPR (or constant)
  • VSRC1 can only be a VGPR
  • Instructions must not exceed the VGPR source-cache port limits
      ◦ SRC0x and SRC0y must come from different VGPR banks (bank# = SRC % 4), or be the same VGPR and
        same sized operand. Same for SRC1.
      ◦ Each cache has 3 read ports: one dedicated to SRC0, one dedicated to SRC1 and one for SRC2
           ▪ A cache can read all 3 of them at once, but it can’t read two SRC0’s at once (or SRC1/2).
        ◦ SRCX0 and SRCY0 must use different VGPR banks;
        ◦ VSRCX1 and VSRCY1 must use different banks.
           ▪ FMAMK is an exception : V = S0 + K * S1 ("S1" uses the SRC2 read port)
           ▪ V_MOV_B32 when in OPY uses SRC2 (not SRC0) if OPX is also a V_MOV_B32.
        ◦ If both operations use the SRC2 input, then one SRC2 input must be even and the other SRC2 input
          must be odd. The following operations use SRC2: FMAMK_F32 (second input operand);
          DOT2ACC_F32_F16, DOT2ACC_F32_BF16, FMAC_F32 (destination operand).
        ◦ These are hard rules - the instruction does not function if these rules are broken
  • The pair of instructions combined have the following restrictions:
     ◦ At most one literal constant, or they may share the same literal
        ◦ Dest VGPRs: one must be even and the other odd
        ◦ The two instructions must be independent of each other: The OPX instruction must not overwrite
          sources of the OPY instruction but the OPY instruction may overwrite the sources of the OPX
          instruction without creating a hazard.
  • Must not use DPP
  • Must be wave32.

VOPD Instruction Fields

Field           Size   Description
opX             4      instruction opcode for the X operation
opY             5      instruction opcode for the Y operation
src0X           9      Source 0 for X operation. May be a VGPR, SGPR, exec, inline or literal constant
src0Y           9      Source 0 for Y operation. May be a VGPR, SGPR, exec, inline or literal constant
vsrc1X          8      Source 1 for X operation. Must be a VGPR. Ignored for V_MOV_B32
vsrc1Y          8      Source 1 for Y operation. Must be a VGPR. Ignored for V_MOV_B32
