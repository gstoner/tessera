# 3.3.2. VGPRs

> RDNA4 ISA — pages 27–27

  • All other SGPR & Scalar-source addresses must not be indexed and no single operand can reference
    multiple register ranges.

General Rules:
  • Out of range source SGPRs return zero (NULL, M0 or EXEC where not allowed)
  • Writes to an out of range SGPR are ignored

TTMP0-15 can only be written while in the trap handler (STATUS.PRIV=1) but can be read by the user’s shader
(STATUS.PRIV=0). Writes to TTMPs while outside the trap handler are ignored. SALU instructions that try but
fail to write a TTMP also do not update SCC.

  • SALU: Above rules apply.
      ◦ WREXEC and SAVEEXEC write the EXEC mask even when the SDST is out-of-range
  • VALU: Above rules apply.
  • VMEM: S#, T#, V# must be contained within one region.
     ◦ T# (128b), V# or S#: no possible range violation exists (forced alignment puts all in one range).
      ◦ T# (256b) starting at 104 and extending into TTMPs; or starting at TTMP12 and going past TTMP15 is a
        violation. If this occurs, force to use S0.
  • SMEM ops return data starting in SGPRs/VCC and extending into TTMPs, or starting in TTMPs and
    extending outside TTMPs becomes out of range.
      ◦ No data gets written to dest-SGPRs that are out-of-range
      ◦ Addr and write-data are aligned and so cannot go out of range, except:
         ▪ Referencing M0, NULL, or EXEC* returns zero, and SMEM loads cannot load into these registers.
  • S_MOVREL:
      ◦ Indexing is allowed only within SGPRs and TTMPs, and must not cross between the two. Indexing must
        stay within the "base" range (the operand type where index==0).
        The ranges are: [ SGPRs 0-105 and VCC_LO, VCC_HI ], [ Trap Temps 0-15 ], [ all other values ]
      ◦ Indexing must not reach M0, exec or inline constants, the rule is:
         ▪ Base is SGPR: addr > VCC_HI (or if 64-bit operand, addr > VCC_LO)
         ▪ Base is TTMP: addr > TTMP15 (or if B64 if addr > TTMP14)
      ◦ If the source is out of range, S0 is used.
        If the dest is out of range, nothing is written.

3.3.2. VGPRs

3.3.2.1. VGPR Allocation and Alignment
VGPRs are allocated in blocks of 16 for wave32 or 8 for wave64, and a shader may have up to 256 VGPRs. In
other words, VGPRs are allocated in units of (16*32 or 8*64 = 512 DWORDs). A wave may not be created with zero
VGPRs. Devices that have 1536 VGPRs per SIMD allocate in blocks of 24 for wave32 and 12 for wave64.

A wave may voluntarily deallocate all of its VGPRs via S_SENDMSG. Once this is done, the wave may not
reallocate them and the only valid action is to terminate the wave. This can be useful if a wave has issued stores
to memory and is waiting for the write-confirms before terminating. Releasing the VGPRs while waiting may
allow a new wave to allocate them and start earlier.
