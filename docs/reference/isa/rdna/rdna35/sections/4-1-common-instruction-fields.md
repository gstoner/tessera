# 4.1. Common Instruction Fields

> RDNA3.5 ISA — pages 44–44

Chapter 4. Shader Instruction Set
This chapter describes the shader instruction set. Instructions are divided into the following groups:
  • Program Flow
  • Scalar ALU
  • Scalar memory read from constant cache
  • Vector ALU & Parameter-Interpolate
  • Vector Memory read/write :
     ◦ buffers
      ◦ Flat, Global and Scratch
      ◦ LDS
  • GDS
  • Misc: wait on counter, barrier, send message

Instructions are encoded in various microcode formats. The formats are defined by a set of "encoding" bits (in
red) that define the family of instructions and the meaning of the rest of the bits in the instruction. Not every
instruction uses every field in its encoding. Fields which can specify an SGPR as a source or dest are typically
set to NULL when unused; other fields are typically set to zero.

4.1. Common Instruction Fields
"inline constant" - a constant specified in place of a source argument, # 128-248. E.g 1.0, -0.5, 32 etc.

   Float constants work with single, double and 16bit float instructions, and when used in non-float
   instructions, the data is not converted (remains a float).

   Float constants are encoded according to the size of the source operand. For 16-bit operations (both
   packed and non-packed), a float constant is treated as zero-extended 32-bit data, i.e. with the 16-bit
   floating point in the low bits and zeros in the high bits.

   Integer constants used with 32-bit or smaller operands are treated as 32-bit signed integers. Integer
   constants are signed extended for 64-bit sources.

"literal constant" - a 32-bit constant in the instruction stream immediately after a 32- or 64-bit instruction.

   When used in a 64-bit signed integer operation, it is sign-extended to 64 bits. For unsigned 64-bit integer
   ops (and 64-bit binary ops) it is zero extended. When used in a double-float operation, the 32-bit literal is
   the most-significant bits, and the LSBs are zero. Other operations (32 bits or less, or packed math) treat it
   as 32-bit data.
