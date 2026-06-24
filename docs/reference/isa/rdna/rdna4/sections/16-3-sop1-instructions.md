# 16.3. SOP1 Instructions

> RDNA4 ISA — pages 240–266

16.3. SOP1 Instructions

Instructions in this format may use a 32-bit literal constant that occurs immediately after the instruction.

S_MOV_B32                                                                                                        0

Move scalar input into a scalar register.

  D0.b32 = S0.b32

S_MOV_B64                                                                                                        1

Move scalar input into a scalar register.

  D0.b64 = S0.b64

S_CMOV_B32                                                                                                       2

Move scalar input into a scalar register iff SCC is nonzero.

  if SCC then
      D0.b32 = S0.b32
  endif

S_CMOV_B64                                                                                                       3

Move scalar input into a scalar register iff SCC is nonzero.

  if SCC then
      D0.b64 = S0.b64
  endif

S_BREV_B32                                                                                                       4

Reverse the order of bits in a scalar input and store the result into a scalar register.

  D0.u32[31 : 0] = S0.u32[0 : 31]

S_BREV_B64                                                                                                           5

Reverse the order of bits in a scalar input and store the result into a scalar register.

  D0.u64[63 : 0] = S0.u64[0 : 63]

S_CTZ_I32_B32                                                                                                        8

Count the number of trailing "0" bits before the first "1" in a scalar input and store the result into a scalar
register. Store -1 if there are no "1" bits in the input.

  tmp = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from LSB
        if S0.u32[i] == 1'1U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp

Notes

Functional examples:

  S_CTZ_I32_B32(0xaaaaaaaa) => 1
  S_CTZ_I32_B32(0x55555555) => 0
  S_CTZ_I32_B32(0x00000000) => 0xffffffff
  S_CTZ_I32_B32(0xffffffff) => 0
  S_CTZ_I32_B32(0x00010000) => 16

Compare with V_CTZ_I32_B32, which performs the equivalent operation in the vector ALU.

S_CTZ_I32_B64                                                                                                        9

Count the number of trailing "0" bits before the first "1" in a scalar input and store the result into a scalar
register. Store -1 if there are no "1" bits in the input.

  tmp = -1;
  // Set if no ones are found
  for i in 0 : 63 do
        // Search from LSB
        if S0.u64[i] == 1'1U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp

S_CLZ_I32_U32                                                                                                      10

Count the number of leading "0" bits before the first "1" in a scalar input and store the result into a scalar
register. Store -1 if there are no "1" bits.

  tmp = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from MSB
        if S0.u32[31 - i] == 1'1U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp

Notes

Functional examples:

  S_CLZ_I32_U32(0x00000000) => 0xffffffff
  S_CLZ_I32_U32(0x0000cccc) => 16
  S_CLZ_I32_U32(0xffff3333) => 0
  S_CLZ_I32_U32(0x7fffffff) => 1
  S_CLZ_I32_U32(0x80000000) => 0
  S_CLZ_I32_U32(0xffffffff) => 0

Compare with V_CLZ_I32_U32, which performs the equivalent operation in the vector ALU.

S_CLZ_I32_U64                                                                                                      11

Count the number of leading "0" bits before the first "1" in a scalar input and store the result into a scalar
register. Store -1 if there are no "1" bits.

  tmp = -1;

  // Set if no ones are found
  for i in 0 : 63 do
        // Search from MSB
        if S0.u64[63 - i] == 1'1U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp

S_CLS_I32                                                                                                          12

Count the number of leading bits that are the same as the sign bit of a scalar input and store the result into a
scalar register. Store -1 if all input bits are the same.

  tmp = -1;
  // Set if all bits are the same
  for i in 1 : 31 do
        // Search from MSB
        if S0.u32[31 - i] != S0.u32[31] then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp

Notes

Functional examples:

  S_CLS_I32(0x00000000) => 0xffffffff
  S_CLS_I32(0x0000cccc) => 16
  S_CLS_I32(0xffff3333) => 16
  S_CLS_I32(0x7fffffff) => 1
  S_CLS_I32(0x80000000) => 1
  S_CLS_I32(0xffffffff) => 0xffffffff

Compare with V_CLS_I32, which performs the equivalent operation in the vector ALU.

S_CLS_I32_I64                                                                                                      13

Count the number of leading bits that are the same as the sign bit of a scalar input and store the result into a
scalar register. Store -1 if all input bits are the same.

  tmp = -1;
  // Set if all bits are the same
  for i in 1 : 63 do

       // Search from MSB
       if S0.u64[63 - i] != S0.u64[63] then
            tmp = i;
            break
       endif
  endfor;
  D0.i32 = tmp

S_SEXT_I32_I8                                                                                                 14

Sign extend a signed 8 bit scalar input to 32 bits and store the result into a scalar register.

  D0.i32 = 32'I(signext(S0.i8))

S_SEXT_I32_I16                                                                                                15

Sign extend a signed 16 bit scalar input to 32 bits and store the result into a scalar register.

  D0.i32 = 32'I(signext(S0.i16))

S_BITSET0_B32                                                                                                 16

Given a bit offset in a scalar input, set the indicated bit in the destination scalar register to 0.

  D0.u32[S0.u32[4 : 0]] = 1'0U

S_BITSET0_B64                                                                                                 17

Given a bit offset in a scalar input, set the indicated bit in the destination scalar register to 0.

  D0.u64[S0.u32[5 : 0]] = 1'0U

S_BITSET1_B32                                                                                                 18

Given a bit offset in a scalar input, set the indicated bit in the destination scalar register to 1.

  D0.u32[S0.u32[4 : 0]] = 1'1U

S_BITSET1_B64                                                                                                         19

Given a bit offset in a scalar input, set the indicated bit in the destination scalar register to 1.

  D0.u64[S0.u32[5 : 0]] = 1'1U

S_BITREPLICATE_B64_B32                                                                                                20

Substitute each bit of a 32 bit scalar input with two instances of itself and store the result into a 64 bit scalar
register.

  tmp = S0.u32;
  for i in 0 : 31 do
        D0.u64[i * 2] = tmp[i];
        D0.u64[i * 2 + 1] = tmp[i]
  endfor

Notes

This opcode can be used to convert a quad mask into a pixel mask; given quad mask in s0, the following
sequence produces a pixel mask in s2:

        s_bitreplicate_b64 s2, s0
        s_bitreplicate_b64 s2, s2

To perform the inverse operation see S_QUADMASK_B64.

S_ABS_I32                                                                                                             21

Compute the absolute value of a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.i32 = S0.i32 < 0 ? -S0.i32 : S0.i32;
  SCC = D0.i32 != 0

Notes

Functional examples:

  S_ABS_I32(0x00000001) => 0x00000001
  S_ABS_I32(0x7fffffff) => 0x7fffffff
  S_ABS_I32(0x80000000) => 0x80000000        // Note this is negative!
  S_ABS_I32(0x80000001) => 0x7fffffff
  S_ABS_I32(0x80000002) => 0x7ffffffe
  S_ABS_I32(0xffffffff) => 0x00000001

S_BCNT0_I32_B32                                                                                                     22

Count the number of "0" bits in a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

  tmp = 0;
  for i in 0 : 31 do
        tmp += S0.u32[i] == 1'0U ? 1 : 0
  endfor;
  D0.i32 = tmp;
  SCC = D0.u32 != 0U

Notes

Functional examples:

  S_BCNT0_I32_B32(0x00000000) => 32
  S_BCNT0_I32_B32(0xcccccccc) => 16
  S_BCNT0_I32_B32(0xffffffff) => 0

S_BCNT0_I32_B64                                                                                                     23

Count the number of "0" bits in a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

  tmp = 0;
  for i in 0 : 63 do
        tmp += S0.u64[i] == 1'0U ? 1 : 0
  endfor;
  D0.i32 = tmp;
  SCC = D0.u64 != 0ULL

S_BCNT1_I32_B32                                                                                                     24

Count the number of "1" bits in a scalar input, store the result into a scalar register and set SCC iff the result is

nonzero.

  tmp = 0;
  for i in 0 : 31 do
        tmp += S0.u32[i] == 1'1U ? 1 : 0
  endfor;
  D0.i32 = tmp;
  SCC = D0.u32 != 0U

Notes

Functional examples:

  S_BCNT1_I32_B32(0x00000000) => 0
  S_BCNT1_I32_B32(0xcccccccc) => 16
  S_BCNT1_I32_B32(0xffffffff) => 32

S_BCNT1_I32_B64                                                                                                     25

Count the number of "1" bits in a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

  tmp = 0;
  for i in 0 : 63 do
        tmp += S0.u64[i] == 1'1U ? 1 : 0
  endfor;
  D0.i32 = tmp;
  SCC = D0.u64 != 0ULL

S_QUADMASK_B32                                                                                                      26

Reduce a pixel mask from the scalar input into a quad mask, store the result in a scalar register and set SCC iff
the result is nonzero.

  tmp = 0U;
  for i in 0 : 7 do
        tmp[i] = S0.u32[i * 4 +: 4] != 0U
  endfor;
  D0.u32 = tmp;
  SCC = D0.u32 != 0U

Notes

To perform the inverse operation see S_BITREPLICATE_B64_B32.

S_QUADMASK_B64                                                                                                  27

Reduce a pixel mask from the scalar input into a quad mask, store the result in a scalar register and set SCC iff
the result is nonzero.

  tmp = 0ULL;
  for i in 0 : 15 do
        tmp[i] = S0.u64[i * 4 +: 4] != 0ULL
  endfor;
  D0.u64 = tmp;
  SCC = D0.u64 != 0ULL

Notes

To perform the inverse operation see S_BITREPLICATE_B64_B32.

S_WQM_B32                                                                                                       28

Given an active pixel mask in a scalar input, calculate whole quad mode mask for that input, store the result
into a scalar register and set SCC iff the result is nonzero.

In whole quad mode, if any pixel in a quad is active then all pixels of the quad are marked active.

  tmp = 0U;
  declare i : 6'U;
  for i in 6'0U : 6'31U do
        tmp[i] = S0.u32[i & 6'60U +: 6'4U] != 0U
  endfor;
  D0.u32 = tmp;
  SCC = D0.u32 != 0U

S_WQM_B64                                                                                                       29

Given an active pixel mask in a scalar input, calculate whole quad mode mask for that input, store the result
into a scalar register and set SCC iff the result is nonzero.

In whole quad mode, if any pixel in a quad is active then all pixels of the quad are marked active.

  tmp = 0ULL;
  declare i : 6'U;
  for i in 6'0U : 6'63U do
        tmp[i] = S0.u64[i & 6'60U +: 6'4U] != 0ULL
  endfor;
  D0.u64 = tmp;

  SCC = D0.u64 != 0ULL

S_NOT_B32                                                                                                             30

Calculate bitwise negation on a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u32 = ~S0.u32;
  SCC = D0.u32 != 0U

S_NOT_B64                                                                                                             31

Calculate bitwise negation on a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u64 = ~S0.u64;
  SCC = D0.u64 != 0ULL

S_AND_SAVEEXEC_B32                                                                                                    32

Calculate bitwise AND on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u32;
  EXEC.u32 = (S0.u32 & EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_AND_SAVEEXEC_B64                                                                                                    33

Calculate bitwise AND on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 & EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_OR_SAVEEXEC_B32                                                                                              34

Calculate bitwise OR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask, set
SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar destination
register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u32;
  EXEC.u32 = (S0.u32 | EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_OR_SAVEEXEC_B64                                                                                              35

Calculate bitwise OR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask, set
SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar destination
register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 | EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_XOR_SAVEEXEC_B32                                                                                             36

Calculate bitwise XOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u32;
  EXEC.u32 = (S0.u32 ^ EXEC.u32);
  D0.u32 = saveexec.u32;

  SCC = EXEC.u32 != 0U

S_XOR_SAVEEXEC_B64                                                                                           37

Calculate bitwise XOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 ^ EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_NAND_SAVEEXEC_B32                                                                                          38

Calculate bitwise NAND on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

  saveexec = EXEC.u32;
  EXEC.u32 = ~(S0.u32 & EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_NAND_SAVEEXEC_B64                                                                                          39

Calculate bitwise NAND on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

  saveexec = EXEC.u64;
  EXEC.u64 = ~(S0.u64 & EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_NOR_SAVEEXEC_B32                                                                                           40

Calculate bitwise NOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,

set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

  saveexec = EXEC.u32;
  EXEC.u32 = ~(S0.u32 | EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_NOR_SAVEEXEC_B64                                                                                              41

Calculate bitwise NOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

  saveexec = EXEC.u64;
  EXEC.u64 = ~(S0.u64 | EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_XNOR_SAVEEXEC_B32                                                                                             42

Calculate bitwise XNOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

  saveexec = EXEC.u32;
  EXEC.u32 = ~(S0.u32 ^ EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_XNOR_SAVEEXEC_B64                                                                                             43

Calculate bitwise XNOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

  saveexec = EXEC.u64;
  EXEC.u64 = ~(S0.u64 ^ EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_AND_NOT0_SAVEEXEC_B32                                                                                          44

Calculate bitwise AND on the EXEC mask and the negation of the scalar input, store the calculated result into
the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into
the scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u32;
  EXEC.u32 = (~S0.u32 & EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_AND_NOT0_SAVEEXEC_B64                                                                                          45

Calculate bitwise AND on the EXEC mask and the negation of the scalar input, store the calculated result into
the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into
the scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u64;
  EXEC.u64 = (~S0.u64 & EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_OR_NOT0_SAVEEXEC_B32                                                                                           46

Calculate bitwise OR on the EXEC mask and the negation of the scalar input, store the calculated result into the
EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the
scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u32;
  EXEC.u32 = (~S0.u32 | EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_OR_NOT0_SAVEEXEC_B64                                                                                           47

Calculate bitwise OR on the EXEC mask and the negation of the scalar input, store the calculated result into the
EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the

scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u64;
  EXEC.u64 = (~S0.u64 | EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_AND_NOT1_SAVEEXEC_B32                                                                                          48

Calculate bitwise AND on the scalar input and the negation of the EXEC mask, store the calculated result into
the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into
the scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u32;
  EXEC.u32 = (S0.u32 & ~EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_AND_NOT1_SAVEEXEC_B64                                                                                          49

Calculate bitwise AND on the scalar input and the negation of the EXEC mask, store the calculated result into
the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into
the scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 & ~EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_OR_NOT1_SAVEEXEC_B32                                                                                           50

Calculate bitwise OR on the scalar input and the negation of the EXEC mask, store the calculated result into the
EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the
scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u32;
  EXEC.u32 = (S0.u32 | ~EXEC.u32);
  D0.u32 = saveexec.u32;
  SCC = EXEC.u32 != 0U

S_OR_NOT1_SAVEEXEC_B64                                                                                           51

Calculate bitwise OR on the scalar input and the negation of the EXEC mask, store the calculated result into the
EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the
scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 | ~EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL

S_AND_NOT0_WREXEC_B32                                                                                            52

Calculate bitwise AND on the EXEC mask and the negation of the scalar input, store the calculated result into
the EXEC mask and also into the scalar destination register, and set SCC iff the calculated result is nonzero.

Unlike the SAVEEXEC series of opcodes, the value written to destination SGPRs is the result of the bitwise-op
result. EXEC and the destination SGPRs have the same value at the end of this instruction. This instruction is
intended to help accelerate waterfalling.

  EXEC.u32 = (~S0.u32 & EXEC.u32);
  D0.u32 = EXEC.u32;
  SCC = EXEC.u32 != 0U

S_AND_NOT0_WREXEC_B64                                                                                            53

Calculate bitwise AND on the EXEC mask and the negation of the scalar input, store the calculated result into
the EXEC mask and also into the scalar destination register, and set SCC iff the calculated result is nonzero.

Unlike the SAVEEXEC series of opcodes, the value written to destination SGPRs is the result of the bitwise-op
result. EXEC and the destination SGPRs have the same value at the end of this instruction. This instruction is
intended to help accelerate waterfalling.

  EXEC.u64 = (~S0.u64 & EXEC.u64);
  D0.u64 = EXEC.u64;

  SCC = EXEC.u64 != 0ULL

S_AND_NOT1_WREXEC_B32                                                                                            54

Calculate bitwise AND on the scalar input and the negation of the EXEC mask, store the calculated result into
the EXEC mask and also into the scalar destination register, and set SCC iff the calculated result is nonzero.

Unlike the SAVEEXEC series of opcodes, the value written to destination SGPRs is the result of the bitwise-op
result. EXEC and the destination SGPRs have the same value at the end of this instruction. This instruction is
intended to help accelerate waterfalling.

  EXEC.u32 = (S0.u32 & ~EXEC.u32);
  D0.u32 = EXEC.u32;
  SCC = EXEC.u32 != 0U

Notes

See S_AND_NOT1_WREXEC_B64 for example code.

S_AND_NOT1_WREXEC_B64                                                                                            55

Calculate bitwise AND on the scalar input and the negation of the EXEC mask, store the calculated result into
the EXEC mask and also into the scalar destination register, and set SCC iff the calculated result is nonzero.

Unlike the SAVEEXEC series of opcodes, the value written to destination SGPRs is the result of the bitwise-op
result. EXEC and the destination SGPRs have the same value at the end of this instruction. This instruction is
intended to help accelerate waterfalling.

  EXEC.u64 = (S0.u64 & ~EXEC.u64);
  D0.u64 = EXEC.u64;
  SCC = EXEC.u64 != 0ULL

Notes

In particular, the following sequence of waterfall code is optimized by using a WREXEC instead of two separate
scalar ops:

  // V0 holds the index value per lane
  // save exec mask for restore at the end
  s_mov_b64 s2, exec
  // exec mask of remaining (unprocessed) threads
  s_mov_b64 s4, exec
  loop:
  // get the index value for the first active lane
  v_readfirstlane_b32     s0, v0
  // find all other lanes with same index value

  v_cmpx_eq s0, v0
  <OP>          // do the operation using the current EXEC mask. S0 holds the index.
  // mask out thread that was just executed
  // s_andn2_b64    s4, s4, exec
  // s_mov_b64      exec, s4
  s_andn2_wrexec_b64 s4, s4        // replaces above 2 ops
  // repeat until EXEC==0
  s_cbranch_scc1    loop
  s_mov_b64      exec, s2

S_MOVRELS_B32                                                                                 64

Move data from a relatively-indexed scalar register into another scalar register.

  addr = SRC0.u32;
  // Raw value from instruction
  addr += M0.u32[31 : 0];
  D0.b32 = SGPR[addr].b32

Notes

Example: The following instruction sequence performs the move s5 <= s17:

        s_mov_b32 m0, 10
        s_movrels_b32 s5, s7

S_MOVRELS_B64                                                                                 65

Move data from a relatively-indexed scalar register into another scalar register.

The index in M0.u and the operand address in SRC0.u must be even for this operation.

  addr = SRC0.u32;
  // Raw value from instruction
  addr += M0.u32[31 : 0];
  D0.b64 = SGPR[addr].b64

S_MOVRELD_B32                                                                                 66

Move data from a scalar input into a relatively-indexed scalar register.

  addr = DST.u32;
  // Raw value from instruction

  addr += M0.u32[31 : 0];
  SGPR[addr].b32 = S0.b32

Notes

Example: The following instruction sequence performs the move s15 <= s7:

        s_mov_b32 m0, 10
        s_movreld_b32 s5, s7

S_MOVRELD_B64                                                                                                  67

Move data from a scalar input into a relatively-indexed scalar register.

The index in M0.u and the operand address in DST.u must be even for this operation.

  addr = DST.u32;
  // Raw value from instruction
  addr += M0.u32[31 : 0];
  SGPR[addr].b64 = S0.b64

S_MOVRELSD_2_B32                                                                                               68

Move data from a relatively-indexed scalar register into another relatively-indexed scalar register, using
different offsets for each index.

  addrs = SRC0.u32;
  // Raw value from instruction
  addrd = DST.u32;
  // Raw value from instruction
  addrs += M0.u32[9 : 0].u32;
  addrd += M0.u32[25 : 16].u32;
  SGPR[addrd].b32 = SGPR[addrs].b32

Notes

Example: The following instruction sequence performs the move s25 <= s17:

        s_mov_b32 m0, ((20 << 16) | 10)
        s_movrelsd_2_b32 s5, s7

S_GETPC_B64                                                                                                      71

Store the address of the next instruction to a scalar register.

The byte address of the instruction immediately following this instruction is saved to the destination.

  D0.i64 = PC + 4LL

Notes

This instruction must be 4 bytes.

S_SETPC_B64                                                                                                      72

Jump to an address specified in a scalar register.

The argument is a byte address of the instruction to jump to.

  PC = S0.i64

S_SWAPPC_B64                                                                                                     73

Store the address of the next instruction to a scalar register and then jump to an address specified in the scalar
input.

The argument is a byte address of the instruction to jump to. The byte address of the instruction immediately
following this instruction is saved to the destination.

  jump_addr = S0.i64;
  D0.i64 = PC + 4LL;
  PC = jump_addr.i64

Notes

This instruction must be 4 bytes.

S_RFE_B64                                                                                                        74

Return from the exception handler. Clear the wave's PRIV bit and then jump to an address specified by the
scalar input.

The argument is a byte address of the instruction to jump to; this address is likely derived from the state passed
into the trap handler.

This instruction may only be used within a trap handler.

  WAVE_STATUS.PRIV = 1'0U;
  PC = S0.i64

S_SENDMSG_RTN_B32                                                                                            76

Send a message to upstream control hardware.

SSRC[7:0] contains the message type encoded in the instruction directly (this instruction does not read an
SGPR). The message is expected to return a response from the upstream control hardware and the result is
written to SDST. Use S_WAIT_KMCNT to wait for the response on the dependent instruction.

S_SENDMSG_RTN* instructions return data in-order among themselves but out-of-order with other
instructions that manipulate lgkmcnt (including S_SENDMSG and S_SENDMSGHALT).

If the message returns a 64 bit value then only the lower 32 bits are written to SDST.

If SDST is VCC then VCCZ is undefined.

S_SENDMSG_RTN_B64                                                                                            77

Send a message to upstream control hardware.

SSRC[7:0] contains the message type encoded in the instruction directly (this instruction does not read an
SGPR). The message is expected to return a response from the upstream control hardware and the result is
written to SDST. Use S_WAIT_KMCNT to wait for the response on the dependent instruction.

S_SENDMSG_RTN* instructions return data in-order among themselves but out-of-order with other
instructions that manipulate lgkmcnt (including S_SENDMSG and S_SENDMSGHALT).

If the message returns a 32 bit value then this instruction fills the upper bits of SDST with zero.

If SDST is VCC then VCCZ is undefined.

S_BARRIER_SIGNAL                                                                                             78

Signal that a wave has arrived at a barrier. The argument specifies which barrier to signal.

Support for M0 as an operand is reserved for other architectures.

  ;

  // M0 cannot reference the negative barrier numbers.
  barrierNumber = IsM0(SRC0.u32) ? 32'I(M0[4 : 0].u32) : SRC0.i32;

  if !InWorkgroup() then
      // Must be in a workgroup to signal a barrier.
      s_nop(16'0U)
  elsif ((barrierNumber == -2) && !WAVE_STATUS.PRIV) then
      // Barrier #-2 is privileged (for traps only).
      s_nop(16'0U)
  elsif barrierNumber == 0 then
      // Barrier #0 is a NOP.
      s_nop(16'0U)
  else
      BARRIER_STATE[barrierNumber & 63].signalCnt += 7'1U
  endif;
  // Check for barrier completion.
  CheckBarrierComplete(barrierNumber)

S_BARRIER_SIGNAL_ISFIRST                                                                                               79

Signal that a wave has arrived at a barrier and set SCC to indicate if this is the first wave to signal the barrier.
The argument specifies which barrier to signal.

Support for M0 as an operand is reserved for other architectures.

  ;

  // M0 cannot reference the negative barrier numbers.
  barrierNumber = IsM0(SRC0.u32) ? 32'I(M0[4 : 0].u32) : SRC0.i32;
  if !InWorkgroup() then
      // Must be in a workgroup to signal a barrier.
      SCC = 1'0U
  elsif ((barrierNumber == -2) && !WAVE_STATUS.PRIV) then
      // Barrier #-2 is privileged (for traps only).
      SCC = 1'0U
  elsif barrierNumber == 0 then
      // Barrier #0 is a NOP.
      SCC = 1'0U
  else
      // Set SCC if this is the first signaling event for this barrier.
      SCC = BARRIER_STATE[barrierNumber & 63].signalCnt.u32 == 0U;
      BARRIER_STATE[barrierNumber & 63].signalCnt += 7'1U
  endif;
  CheckBarrierComplete(barrierNumber)

S_GET_BARRIER_STATE                                                                                                    80

Read out current barrier state. Increments KMCNT on issue and decrements KMCNT when operation
completes.

  barrierNumber = IsM0(SRC0.u32) ? 32'I(M0[4 : 0].u32) : SRC0.i32;
  D0.u32 = 32'U({ 9'0, BARRIER_STATE[barrierNumber & 63].signalCnt.u7, 5'0, BARRIER_STATE[barrierNumber &

  63].memberCnt.u7, 3'0, BARRIER_STATE[barrierNumber & 63].valid.u1 })

Notes

Use S_WAIT_KMCNT to determine when D0 can be read.

S_ALLOC_VGPR                                                                                                      83

Attempt to set the wave's VGPR allocation to the specified number of VGPRs. The desired VGPR count may be
specified as a constant or in an SGPR. The request is rounded up to the next block size so a successful allocation
may include more than the requested number of VGPRs.

Depending on the current allocation, executing this instruction may cause additional logical VGPRs to be
allocated or it may cause logical VGPRs to be released. This operation may also return a failure code if it is
unable to allocate the requested number of VGPRs.

The 1-bit success or failure status of the request is returned to the SCC register. Success is indicated by setting
SCC to 1. A failure can occur if there are not enough free registers. The shader must check the status and is
expected to generally implement a retry loop.

VGPRs allocated by SPI cannot be fully deallocated. Only registers allocated by a prior S_ALLOC_VGPR call can
be deallocated.

  WaitIdleExceptStoreCnt();
  n = ReallocVgprs(32'I(S0[8 : 0].u32));
  // ReallocVgprs returns the actual number of VGPRs allocated rounded to segment size.
  // ReallocVgprs returns a negative value if reallocation fails.
  if n < 0 then
        SCC = 1'0U
  else
        NUM_VGPRS = n;
        SCC = 1'1U
  endif

Notes

By default forward progress is ensured for one wave on each SIMD. Software may implement its own scheme
for ensuring forward progress if wave allocation requests exceed the available VGPRs.

Allocation is atomic --- either the full request is allocated or no registers are allocated.

The instruction buffer waits for idle both before and after changing the allocation of VGPRs, with the exception
that STORECNT may be nonzero. This allows for a reduction in VGPRs while memory writes are pending as
long as they have read all of their inputs. No following instruction can issue until the allocation operation is
complete (or allocation has failed).

MSG_DEALLOC_VGPRS is incompatible with Dynamic VGPR mode.

This instruction is illegal when in wave64 mode . This instruction is also illegal when the shader is not a CS

shader or when dynamic VGPR mode is disabled (DVGPR_EN == 0).

S_SLEEP_VAR                                                                                                      88

Cause a wave to sleep for up to ~8000 clocks, or to sleep until an external event wakes the wave up.

S0[6:0] determines the sleep duration. The wave sleeps for (64*(S0[6:0]-1) … 64*S0[6:0]) clocks. The exact
amount of delay is approximate. Compare with S_NOP. When S0[6:0] is zero then no sleep occurs.

This instruction does not support "Sleep Forever" mode. To enable that mode the shader must use S_SLEEP.

See also S_SLEEP.

S_CEIL_F32                                                                                                       96

Round the single-precision float input up to next integer and store the result in floating point format into a
scalar register.

  D0.f32 = trunc(S0.f32);
  if ((S0.f32 > 0.0F) && (S0.f32 != D0.f32)) then
      D0.f32 += 1.0F
  endif

S_FLOOR_F32                                                                                                      97

Round the single-precision float input down to previous integer and store the result in floating point format
into a scalar register.

  D0.f32 = trunc(S0.f32);
  if ((S0.f32 < 0.0F) && (S0.f32 != D0.f32)) then
      D0.f32 += -1.0F
  endif

S_TRUNC_F32                                                                                                      98

Compute the integer part of a single-precision float input using round toward zero semantics and store the
result in floating point format into a scalar register.

  D0.f32 = trunc(S0.f32)

S_RNDNE_F32                                                                                                        99

Round the single-precision float input to the nearest even integer and store the result in floating point format
into a scalar register.

  D0.f32 = floor(S0.f32 + 0.5F);
  if (isEven(64'F(floor(S0.f32))) && (fract(S0.f32) == 0.5F)) then
      D0.f32 -= 1.0F
  endif

S_CVT_F32_I32                                                                                                     100

Convert from a signed 32-bit integer input to a single-precision float value and store the result into a scalar
register.

  D0.f32 = i32_to_f32(S0.i32)

S_CVT_F32_U32                                                                                                     101

Convert from an unsigned 32-bit integer input to a single-precision float value and store the result into a scalar
register.

  D0.f32 = u32_to_f32(S0.u32)

S_CVT_I32_F32                                                                                                     102

Convert from a single-precision float input to a signed 32-bit integer value and store the result into a scalar
register.

  D0.i32 = f32_to_i32(S0.f32)

S_CVT_U32_F32                                                                                                     103

Convert from a single-precision float input to an unsigned 32-bit integer value and store the result into a scalar
register.

  D0.u32 = f32_to_u32(S0.f32)

S_CVT_F16_F32                                                                                                       104

Convert from a single-precision float input to a half-precision float value and store the result into a scalar
register.

  D0.f16 = f32_to_f16(S0.f32)

S_CVT_F32_F16                                                                                                       105

Convert from a half-precision float input to a single-precision float value and store the result into a scalar
register.

  D0.f32 = f16_to_f32(S0.f16)

S_CVT_HI_F32_F16                                                                                                    106

Convert from a half-precision float value in the high 16 bits of a scalar input to a single-precision float value
and store the result into a scalar register.

  D0.f32 = f16_to_f32(S0[31 : 16].f16)

S_CEIL_F16                                                                                                          107

Round the half-precision float input up to next integer and store the result in floating point format into a scalar
register.

  D0.f16 = trunc(S0.f16);
  if ((S0.f16 > 16'0.0) && (S0.f16 != D0.f16)) then
      D0.f16 += 16'1.0
  endif

S_FLOOR_F16                                                                                                         108

Round the half-precision float input down to previous integer and store the result in floating point format into
a scalar register.

  D0.f16 = trunc(S0.f16);
  if ((S0.f16 < 16'0.0) && (S0.f16 != D0.f16)) then
      D0.f16 += -16'1.0
  endif

S_TRUNC_F16                                                                                                      109

Compute the integer part of a half-precision float input using round toward zero semantics and store the result
in floating point format into a scalar register.

  D0.f16 = trunc(S0.f16)

S_RNDNE_F16                                                                                                      110

Round the half-precision float input to the nearest even integer and store the result in floating point format
into a scalar register.

  D0.f16 = floor(S0.f16 + 16'0.5);
  if (isEven(64'F(floor(S0.f16))) && (fract(S0.f16) == 16'0.5)) then
      D0.f16 -= 16'1.0
  endif
