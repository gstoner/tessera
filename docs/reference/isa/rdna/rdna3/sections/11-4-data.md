# 11.4. Data

> RDNA3 ISA — pages 126–126

  • Misaligned data (scratch accesses may be misaligned)
  • Out-of-range address:
     ◦ LDS access with an address outside the range: [ 0, LDS_SIZE-1 ]

The policy for threads with bad addresses is: stores outside this range do not write a value, and reads return
zero. The aperture check for invalid address occurs before adding any address offsets - it is based only on the
base address; the other checks are performed after adding the offsets.

Addressing errors from either LDS or TA are returned on their respective "instruction done" busses as
MEMVIOL. This sets the wave’s MEMVIOL TrapStatus bit, and also causes an exception (trap).

11.4. Data
FLAT instructions can use from zero to four consecutive DWORDs of data in VGPRs and/or memory. The DATA
field determines which VGPR(s) supply source data (if any) and the VDST VGPRs hold return data (if any).
There is no data-format conversion performed.

"D16" instructions use only 16-bit of the VGPR instead of the full 32bits. "D16_HI" instructions read or write
only the high 16-bits, while "D16" use the low 16-bits.
