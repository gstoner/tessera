# 3.3.1. SGPRs

> RDNA3 ISA — pages 24–25

3.2.3. Instruction Skipping: EXEC==0
The shader hardware may skip vector instructions when EXEC==0. Instructions which may be skipped are:

  • VALU - skip if EXEC == 0
     ◦ Not skipped if the instruction writes SGPRs/VCC
      ◦ Does not skip WMMA or SWMMA
      ◦ This skipping is opportunistic and may not occur depending on timing after a V_CMPX.
  • These are not skipped regardless of EXEC mask value, and are issued only once in wave64
     ◦ V_NOP, V_PIPEFLUSH, V_READLANE, V_READFIRSTLANE, V_WRITELANE
      ◦ BUFFER_GL1_INV, BUFFER_GL0_INV
  • These are not skipped and are issued twice regardless of EXEC mask value in wave64 mode
     ◦ V_CMP which writes SGPR or VCC (not V_CMPX - may skip one pass but not both)
      ◦ Any VALU which writes an SGPR
  • Export Request - skip unless: Done==1 or if export target is POS0
     ◦ Skipped if the wave was created with SKIP_EXPORT=1
  • LDS_param_load / LDS-direct: are skipped when EXEC==0 and EXP_cnt==0
  • LDS, Memory, GDS - do not skip
     ◦ VMEM can be skipped only if: VMcnt/VScnt==0 and EXEC==0
         ▪ otherwise for wave64 one pass can be skipped if EXEC==0 for that half, but not both halves.
      ◦ LDS can be skipped only if: LGKMcnt==0 and EXEC==0
      ◦ Does not skip GDS or GWS

3.3. Storage State: SGPR, VGPR, LDS

3.3.1. SGPRs

3.3.1.1. SGPR Allocation and storage
Every wave is allocated a fixed number of SGPRs:

  • 106 normal SGPRs
  • VCC_HI and VCC_LO (stored in SGPRs 106 and 107)
  • 16 Trap-temporary SGPRs, meant for use by the trap handler

3.3.1.2. VCC
The Vector Condition Code (VCC) can be written by V_CMP and integer vector ADD/SUB instructions. VCC is
implicitly read by V_ADD_CI, V_SUB_CI, V_CNDMASK and V_DIV_FMAS. VCC is a named SGPR-pair and is
subject to the same dependency checks as any other SGPR.

3.3.1.3. SGPR Alignment
There are a few cases where even-aligned SGPRs are required:
 1. any time 64-bit data is used
     a. this includes moves to/from 64-bit registers, including PC
 2. Scalar memory reads when the address-base comes from an SGPR-pair

Quad-alignment of SGPRs is required for operation on more than 64-bits, and for the data GPR when a scalar
memory operation (read, write or atomic) operates on more than 2 DWORDs. Similarly, when a 64-bit SGPR
data value is used as a source to a VALU op, it must be even aligned regardless of size. In contrast, when a 32-
bit SGPR data value is used as a source to a VALU op, it can be arbitrarily aligned regardless of wave size.

When a 64-bit quantity is stored in SGPRs, the LSB’s are in SGPR[n], and the MSB’s are in SGPR[n+1].

It is illegal to use mis-aligned source or destination SGPRs for data larger than 32 bits and results are
unpredictable.

As an example, VALU ops with carry-in or carry-out:
  • When used with wave32, these are 32 bit values and may have any arbitrary alignment
  • When used with wave64, these are 64 bit values and must be aligned to an even SGPR address

Hardware enforces SGPR alignment by ignoring LSB’s as necessary and treating them as zero. For
*MOVREL*_B64, the LSB of the index is also ignored and treated as zero.

3.3.1.4. SGPR Out of Range Behavior
Scalar sources and dests use a 7-bit encoding:

   Scalar 0-105=SGPR; 106,107=VCC, 108-123=TTMP0-15, and 124-127={NULL, M0, EXEC_LO, EXEC_HI}.

It is illegal to use GPR indexing or a multi-DWORD operand to cross SGPR regions. The regions are:
  • SGPRs 0 - 107 (includes VCC)
  • Trap Temp SGPRs
  • All other SGPR & Scalar-source addresses must not be indexed and no single operand can reference
    multiple register ranges.

General Rules:
  • Out of range source SGPRs return zero (using a TTMP when STATUS.PRIV=0, NULL, M0 or EXEC where not
    allowed)
  • Writes to an out of range SGPR are ignored

TTMP0-15 can only be written while in the trap handler (STATUS.PRIV=1) and cannot be read by the user’s
shader (returns zero when STATUS.PRIV=0). Writes to TTMPs while outside the trap handler are ignored. SALU
instructions which try but fail to write a TTMP also do not update SCC.

  • SALU: Above rules apply.
      ◦ WREXEC and SAVEEXEC write the EXEC mask even when the SDST is out-of-range
  • VALU: Above rules apply.
  • VMEM: S#, T#, V# must be contained within one region.
