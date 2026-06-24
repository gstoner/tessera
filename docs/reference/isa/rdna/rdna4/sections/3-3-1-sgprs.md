# 3.3.1. SGPRs

> RDNA4 ISA — pages 26–26

3.3.1. SGPRs

3.3.1.1. SGPR Allocation and storage
Every wave is allocated a fixed number of SGPRs:
  • 106 normal SGPRs
  • VCC_HI and VCC_LO (stored in SGPRs 106 and 107)
  • 16 Trap-temporary SGPRs, meant for use by the trap handler

3.3.1.1.1. VCC

The Vector Condition Code (VCC) is a named SGPR-pair that can be written by V_CMP and integer vector
ADD/SUB instructions. VCC is implicitly read by V_ADD_CI, V_SUB_CI, V_CNDMASK and V_DIV_FMAS. VCC
is subject to the same dependency checks as any other SGPR.

3.3.1.2. SGPR Alignment
There are a few cases where even-aligned SGPRs are required:
 1. any time 64-bit data is used
     a. this includes moves to/from 64-bit registers, including PC
 2. Scalar memory loads when the address-base comes from an SGPR-pair

When a 64-bit SGPR data value is used as a source to a VALU op, it must be even aligned regardless of size. In
contrast, when a 32-bit SGPR data value is used as a source to a VALU op, it can be arbitrarily aligned
regardless of wave size.

Quad-alignment of SGPRs is required for operations on more than 64-bits, and for the data SGPR when a scalar
memory operation (read, write or atomic) operates on more than 2 DWORDs.

When a 64-bit quantity is stored in SGPRs, the LSB’s are in SGPR[n], and the MSB’s are in SGPR[n+1].

When an SGPR is used as a carry-in, carry-out or mask value to a VALU op, it must be even-aligned in wave64
shaders but may be arbitrarily aligned in wave32 shaders.

It is illegal to use mis-aligned source or destination SGPRs for data larger than 32 bits and results are
unpredictable.

3.3.1.3. SGPR Out of Range Behavior
Scalar sources and dests use a 7-bit encoding:

   Scalar 0-105=SGPR; 106,107=VCC, 108-123=TTMP0-15, and 124-127={NULL, M0, EXEC_LO, EXEC_HI}.

It is illegal to use GPR indexing or a multi-DWORD operand to cross SGPR regions. The regions are:
  • SGPRs 0 - 107 (includes VCC)
  • Trap Temp SGPRs
