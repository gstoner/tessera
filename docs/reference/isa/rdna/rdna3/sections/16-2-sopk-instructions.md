# 16.2. SOPK Instructions

> RDNA3 ISA — pages 209–216

16.2. SOPK Instructions

Instructions in this format may not use a 32-bit literal constant that occurs immediately after the instruction.

S_MOVK_I32                                                                                                         0

Sign extend a literal 16-bit constant and store the result into a scalar register.

  D0.i = 32'I(signext(SIMM16.i16))

S_VERSION                                                                                                          1

Do nothing. This opcode is used to specify the microcode version for tools that interpret shader microcode.

Argument is ignored by hardware. This opcode is not designed for inserting wait states as the next instruction
may issue in the same cycle. Do not use this opcode to resolve wait state hazards, use S_NOP instead.

This opcode may also be used to validate microcode is running with the correct compatibility settings in
drivers and functional models that support multiple generations. We strongly encourage this opcode be
included at the top of every shader block to simplify debug and catch configuration errors.

This opcode must appear in the first 16 bytes of a block of shader code in order to be recognized by external
tools and functional models. Avoid placing opcodes > 32 bits or encodings that are not available in all versions
of the microcode before the S_VERSION opcode. If this opcode is absent then tools are allowed to make an
educated guess of the microcode version using cues from the environment; the guess may be incorrect and
lead to an invalid decode. It is highly recommended that this be the first opcode of a shader block except for
trap handlers, where it should be the second opcode (allowing the first opcode to be a 32-bit branch to
accommodate context switch).

SIMM16[7:0] specifies the microcode version.
SIMM16[15:8] must be set to zero.

  nop();
  // Do nothing - for use by tools only

S_CMOVK_I32                                                                                                        2

Move the sign extension of a literal 16-bit constant into a scalar register iff SCC is nonzero.

  if SCC then

       D0.i = 32'I(signext(SIMM16.i16))
  endif

S_CMPK_EQ_I32                                                                                                         3

Set SCC to 1 iff scalar input is equal to the sign extension of a literal 16-bit constant.

  SCC = 64'I(S0.i) == signext(SIMM16.i16)

S_CMPK_LG_I32                                                                                                         4

Set SCC to 1 iff scalar input is less than or greater than the sign extension of a literal 16-bit constant.

  SCC = 64'I(S0.i) != signext(SIMM16.i16)

S_CMPK_GT_I32                                                                                                         5

Set SCC to 1 iff scalar input is greater than the sign extension of a literal 16-bit constant.

  SCC = 64'I(S0.i) > signext(SIMM16.i16)

S_CMPK_GE_I32                                                                                                         6

Set SCC to 1 iff scalar input is greater than or equal to the sign extension of a literal 16-bit constant.

  SCC = 64'I(S0.i) >= signext(SIMM16.i16)

S_CMPK_LT_I32                                                                                                         7

Set SCC to 1 iff scalar input is less than the sign extension of a literal 16-bit constant.

  SCC = 64'I(S0.i) < signext(SIMM16.i16)

S_CMPK_LE_I32                                                                                                         8

Set SCC to 1 iff scalar input is less than or equal to the sign extension of a literal 16-bit constant.

  SCC = 64'I(S0.i) <= signext(SIMM16.i16)

S_CMPK_EQ_U32                                                                                                         9

Set SCC to 1 iff scalar input is equal to the zero extension of a literal 16-bit constant.

  SCC = S0.u == 32'U(SIMM16.u16)

S_CMPK_LG_U32                                                                                                        10

Set SCC to 1 iff scalar input is less than or greater than the zero extension of a literal 16-bit constant.

  SCC = S0.u != 32'U(SIMM16.u16)

S_CMPK_GT_U32                                                                                                        11

Set SCC to 1 iff scalar input is greater than the zero extension of a literal 16-bit constant.

  SCC = S0.u > 32'U(SIMM16.u16)

S_CMPK_GE_U32                                                                                                        12

Set SCC to 1 iff scalar input is greater than or equal to the zero extension of a literal 16-bit constant.

  SCC = S0.u >= 32'U(SIMM16.u16)

S_CMPK_LT_U32                                                                                                        13

Set SCC to 1 iff scalar input is less than the zero extension of a literal 16-bit constant.

  SCC = S0.u < 32'U(SIMM16.u16)

S_CMPK_LE_U32                                                                                                         14

Set SCC to 1 iff scalar input is less than or equal to the zero extension of a literal 16-bit constant.

  SCC = S0.u <= 32'U(SIMM16.u16)

S_ADDK_I32                                                                                                            15

Add a scalar input and the sign extension of a literal 16-bit constant, store the result into a scalar register and
store the carry-out bit into SCC.

  tmp = D0.i;
  // save value so we can check sign bits for overflow later.
  D0.i = 32'I(64'I(D0.i) + signext(SIMM16.i16));
  SCC = ((tmp[31] == SIMM16.i16[15]) && (tmp[31] != D0.i[31]));
  // signed overflow.

S_MULK_I32                                                                                                            16

Multiply a scalar input with the sign extension of a literal 16-bit constant and store the result into a scalar
register.

  D0.i = 32'I(64'I(D0.i) * signext(SIMM16.i16))

S_GETREG_B32                                                                                                          17

Read some or all of a hardware register into the LSBs of destination.

The SIMM16 argument is encoded as follows:

ID = SIMM16[5:0]
   ID of hardware register to access.

OFFSET = SIMM16[10:6]
   LSB offset of register bits to access.

SIZE = SIMM16[15:11]
   Size of register bits to access, minus 1. Set this field to 31 to read/write all bits of the hardware register.

  hwRegId = SIMM16.u16[5 : 0];
  offset = SIMM16.u16[10 : 6];
  size = SIMM16.u16[15 : 11].u + 1U;
  // logical size is in range 1:32
  value = HW_REGISTERS[hwRegId];
  D0.u = 32'U(32'I(value >> offset.u) & ((1 << size) - 1))

S_SETREG_B32                                                                                                         18

Write some or all of the LSBs of source argument into a hardware register.

The SIMM16 argument is encoded as follows:

ID = SIMM16[5:0]
   ID of hardware register to access.

OFFSET = SIMM16[10:6]
   LSB offset of register bits to access.

SIZE = SIMM16[15:11]
   Size of register bits to access, minus 1. Set this field to 31 to read/write all bits of the hardware register.

  hwRegId = SIMM16.u16[5 : 0];
  offset = SIMM16.u16[10 : 6];
  size = SIMM16.u16[15 : 11].u + 1U;
  // logical size is in range 1:32
  mask = (1 << size) - 1;
  mask = (mask & 32'I(writeableBitMask(hwRegId.u, WAVE_STATUS.PRIV)));
  // Mask of bits we are allowed to modify
  value = ((S0.u << offset.u) & mask.u);
  value = (value | 32'U(HW_REGISTERS[hwRegId].i & ~mask));
  HW_REGISTERS[hwRegId] = value.b;
  // Side-effects may trigger here if certain bits are modified

S_SETREG_IMM32_B32                                                                                                   19

Write some or all of the LSBs of a 32-bit literal constant into a hardware register; this instruction requires a 32-
bit literal constant.

The SIMM16 argument is encoded as follows:

ID = SIMM16[5:0]
   ID of hardware register to access.

OFFSET = SIMM16[10:6]
   LSB offset of register bits to access.

SIZE = SIMM16[15:11]
   Size of register bits to access, minus 1. Set this field to 31 to read/write all bits of the hardware register.

  hwRegId = SIMM16.u16[5 : 0];
  offset = SIMM16.u16[10 : 6];
  size = SIMM16.u16[15 : 11].u + 1U;
  // logical size is in range 1:32
  mask = (1 << size) - 1;
  mask = (mask & 32'I(writeableBitMask(hwRegId.u, WAVE_STATUS.PRIV)));
  // Mask of bits we are allowed to modify
  value = ((SIMM32.u << offset.u) & mask.u);
  value = (value | 32'U(HW_REGISTERS[hwRegId].i & ~mask));
  HW_REGISTERS[hwRegId] = value.b;
  // Side-effects may trigger here if certain bits are modified

S_CALL_B64                                                                                                           20

Store the address of the next instruction to a scalar register and then jump to a constant offset relative to the
current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction. The byte address of
the instruction immediately following this instruction is saved to the destination.

  D0.i64 = PC + 4LL;
  PC = PC + signext(SIMM16.i16 * 16'4) + 4LL

Notes

This implements a short subroutine call where the return address (the next instruction after the S_CALL_B64)
is saved to D. Long calls should consider S_SWAPPC_B64 instead.

This instruction must be 4 bytes.

S_WAITCNT_VSCNT                                                                                                      24

Wait for the counts of outstanding vector store events -- vector memory stores and atomics that DO NOT return
data -- to be at or below the specified level. This counter is not used in 'all-in-order' mode.

Waits for the following condition to hold before continuing:

        vscnt <= S0.u[5:0] + S1.u[5:0].
        // Comparison is 6 bits, no clamping is applied for add overflow

To wait on a literal constant only, write 'null' for the GPR argument.

This opcode may only appear inside a clause if the SGPR operand is set to NULL.

See also S_WAITCNT.

S_WAITCNT_VMCNT                                                                                                25

Wait for the counts of outstanding vector memory events -- everything except for memory stores and atomics-
without-return -- to be at or below the specified level. When in 'all-in-order' mode, wait for all vector memory
events.

Waits for the following condition to hold before continuing:

      vmcnt <= S0.u[5:0] + S1.u[5:0].
      // Comparison is 6 bits, no clamping is applied for add overflow

To wait on a literal constant only, write 'null' for the GPR argument or use S_WAITCNT.

This opcode may only appear inside a clause if the SGPR operand is set to NULL.

See also S_WAITCNT.

S_WAITCNT_EXPCNT                                                                                               26

Wait for the counts of outstanding export events to be at or below the specified level.

Waits for the following condition to hold before continuing:

      expcnt <= S0.u[2:0] + S1.u[2:0].
      // Comparison is 3 bits, no clamping is applied for add overflow

To wait on a literal constant only, write 'null' for the GPR argument or use S_WAITCNT.

This opcode may only appear inside a clause if the SGPR operand is set to NULL.

See also S_WAITCNT.

S_WAITCNT_LGKMCNT                                                                                              27

Wait for the counts of outstanding DS (LG), scalar memory (K) and message (M) events to be at or below the
specified level.

Waits for the following condition to hold before continuing:

      lgkmcnt <= S0.u[5:0] + S1.u[5:0].
      // Comparison is 6 bits, no clamping is applied for add overflow

To wait on a literal constant only, write 'null' for the GPR argument or use S_WAITCNT.

This opcode may only appear inside a clause if the SGPR operand is set to NULL.

See also S_WAITCNT.
