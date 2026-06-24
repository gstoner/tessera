# 1.2. Hardware Overview

> RDNA4 ISA — pages 15–15

Format                   Meaning
I16                      signed 16-bit integer
I32                      signed 32-bit integer
I64                      signed 64-bit integer
U8                       unsigned 8-bit integer
U16                      unsigned 16-bit integer
U32                      unsigned 32-bit integer
U64                      unsigned 64-bit integer
D.i                      Destination that is a signed integer
D.u                      Destination that is an unsigned integer
D.f                      Destination that is a float
S*.i                     Source that is a signed integer. E.g. S2.i means "source 2 is a signed integer"
S*.u                     Source that is an unsigned integer
S*.f                     Source that is a float

If an instruction has two suffixes (for example, _I32_F32), the first suffix indicates the destination type, the
second the source type.

The following abbreviations are used in instruction definitions:
  • D = destination
  • U = unsigned integer
  • S = source
  • SCC = scalar condition code
  • I = signed integer
  • B = bitfield

Note: .u or .i specifies to interpret the argument as an unsigned or signed integer.

1.2. Hardware Overview
The figure below shows a block diagram of the AMD RDNA4 Generation series processors:
