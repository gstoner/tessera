# 6.1. SALU Instruction Formats

> RDNA3 ISA — pages 55–57

Chapter 6. Scalar ALU Operations
Scalar ALU (SALU) instructions operate on values that are common to all work-items in the wave. These
operations consist of 32-bit integer or float arithmetic, and 32- or 64-bit bit-wise operations. The SALU also can
perform operations directly on the Program Counter, allowing the program to create a call stack in SGPRs.
Many operations also set the Scalar Condition Code bit (SCC) to indicate the result of a comparison, a carry-out,
or whether the instruction result was zero.

6.1. SALU Instruction Formats
SALU instructions are encoded in one of five microcode formats, shown below:

Name          Size        Function
SOP1          32 bit      SALU op with 1 input
SOP2          32 bit      SALU op with 2 inputs
SOPK          32 bit      SALU op with 1 constant signed 16-bit integer input
SOPC          32 bit      SALU compare op
SOPP          32 bit      SALU program control op

Each of these instruction formats uses some of these fields:

Field                    Description
OP                       Opcode: instruction to be executed.
SDST                     Destination SGPR, M0, NULL or EXEC.
SSRC0                    First source operand.
SSRC1                    Second source operand.
SIMM16                   Signed immediate 16-bit integer constant.

The lists of similar instructions sometimes use a condensed form using curly braces { } to express a list of
possible names. For example, S_AND_{B32, B64} defines two legal instructions: S_AND_B32 and S_AND_B64.

6.2. Scalar ALU Operands
Valid operands of SALU instructions are:

  • SGPRs, including trap temporary SGPRs
  • Mode register
  • Status register (read-only)
  • M0 register
  • EXEC mask
  • VCC mask
  • SCC
  • Inline constants: integers from -16 to 64, and select floating point values
  • Hardware registers (at most 1 of: EXEC, M0, SCC)
  • One 32-bit literal constant
  • If the destination is NULL, the instruction does not execute: nothing is written and SCC is not modified

In the table below, 0-127 can be used as scalar sources or destinations; 128-255 can only be used as sources.

                                             Table 20. Scalar Operands

                       Code            Meaning
Scalar      Scalar Dest 0-105          SGPR 0 .. 105         SGPRs. One DWORD each.
Source (8   (7 bits)    106            VCC_LO                VCC[31:0]
bits)                  107             VCC_HI                VCC[63:32]
                       108-123         ttmp0 .. ttmp15       Trap handler temporary SGPRs (privileged)
                       124             NULL                  Reads return zero, writes are ignored. When used as a
                                                             destination, nullifies the instruction.
                       125             M0                    Temporary register, use for a variety of functions
                       126             EXEC_LO               EXEC[31:0]
                       127             EXEC_HI               EXEC[63:32]
            Integer   128              0                     Inline constant zero
            Inline    129-192          int 1 .. 64           Integer inline constants
            Constants 193-208          int -1 .. -16
                       209-232         Reserved              Reserved
                       233             DPP8                  8-lane DPP (only valid as SRC0)
                       234             DPP8FI                8-lane DPP with Fetch-Invalid (only valid as SRC0)
                       235             SHARED_BASE           Memory Aperture Definition
                       236             SHARED_LIMIT
                       237             PRIVATE_BASE
                       238             PRIVATE_LIMIT
                       239             Reserved              Reserved
            Float     240              0.5                   Inline floating point constants. Can be used in 16, 32 and
            Inline    241              -0.5                  64 bit floating point math. They may be used with non-
            Constants 242              1.0                   float instructions but the value remains a float.

                       243             -1.0
                                                             1/(2*PI) is 0.15915494. The hex values are:
                       244             2.0                   half: 0x3118
                       245             -2.0                  single: 0x3e22f983
                       246             4.0                   double: 0x3fc45f306dc9c882
                       247             -4.0
                       248             1.0 / (2 * PI)
                       249             Reserved              Reserved
                       250             DPP16                 data parallel primitive
                       251             Reserved              Reserved
                       252             Reserved              Reserved
                       253             SCC                   { 31’b0, SCC }
                       254             Reserved              Reserved
                       255             Literal constant      32 bit constant from instruction stream

SALU destinations are in the range 0-127.

SALU instructions can use a 32-bit literal constant. This constant is part of the instruction stream and is
available to all SALU microcode formats except SOPP and SOPK (except literal is allowed in
S_SETREG_IMM32_B32). Literal constants are used by setting the source instruction field to "literal" (255), and
then the following instruction DWORD is used as the source value.

If the destination SGPR is out-of-range, no SGPR is written with the result and SCC is not updated.

If an instruction uses 64-bit data in SGPRs, the SGPR pair must be aligned to an even boundary. For example, it
is legal to use SGPRs 2 and 3 or 8 and 9 (but not 11 and 12) to represent 64-bit data.
