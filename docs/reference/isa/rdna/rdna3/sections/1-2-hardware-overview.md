# 1.2. Hardware Overview

> RDNA3 ISA — pages 14–14

Format                   Meaning
U64                      unsigned 64-bit integer
D.i                      Destination which is a signed integer
D.u                      Destination which is an unsigned integer
D.f                      Destination which is a float
S*.i                     Source which is a signed integer
S*.u                     Source which is an unsigned integer
S*.f                     Source which is a float

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
The figure below shows a block diagram of the AMD RDNA3 Generation series processors:

                                Figure 1. AMD RDNA3 Generation Series Block Diagram
