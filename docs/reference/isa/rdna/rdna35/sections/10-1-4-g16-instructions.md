# 10.1.4. G16 Instructions

> RDNA3.5 ISA — pages 108–108

that thread.

The data returned is: TEXEL_FAIL | (LOD_WARNING << 1) | (LOD << 16)
  • TEXEL_FAIL : 1 bit indicating that 1 or more texels for this pixel produced a NACK.
    "failure" means accessing an unmapped page.
      ◦ TFE == 0
          ▪ TD writes the data for threads that didn’t NACK to VGPR DST
          ▪ TD writes zeros or the result of blend using zeros for samples that NACKed to VGPR DST
      ◦ TFE == 1
          ▪ VGPR DST is written similar to above
          ▪ TD writes to VGPR DST+1 with a status where the bits corresponding to threads that NACKed are
            set to 1
  • LOD_WARNING : 1 bit indicating a that a pixel attempted to access a texel at too small a LOD:
    warn = ( LOD < T#.min_lod_warning)
  • LOD : indicates which LOD was attempted to be accessed that caused the NACK. Returns the floor of the
    requested LOD.

A pixel cannot receive both TEXEL_FAIL and LOD_WARNING: TEXEL_FAIL takes precedence.

10.1.2. D16 Instructions
Load-format and store-format instructions also come in a "d16" variant. For stores, each 32-bit VGPR holds two
16-bit data elements that are passed to the texture unit. The texture unit converts them to the texture format
before writing to memory. For loads, data returned from the texture unit is converted to 16 bits, and a pair of
data are stored in each 32- bit VGPR (LSBs first, then MSBs). The DMASK bit represents individual 16- bit
elements; so, when DMASK=0011 for an image-load, two 16-bit components are loaded into a single 32-bit
VGPR.

10.1.3. A16 Instructions
The A16 instruction bit indicates that the address components are 16 bits instead of the usual 32 bits.
Components are packed such that the first address component goes into the low 16 bits ([15:0]), and the next
into the high 16 bits ([31:16]).

10.1.4. G16 Instructions
The instructions with "G16" in the name mean the user provided derivatives are 16 bits instead of the usual 32
bits. Derivatives are packed such that the first derivative goes into the low 16 bits ([15:0]), and the next into the
high 16 bits ([31:16]).

10.1.5. Image Non-Sequential Address (NSA)
To avoid having many V_MOV instructions to pack image address VGPRs together, MIMG supports a "Non
Sequential Address" version of the instruction where the VGPR of every address component is uniquely
defined. Data components are still packed. This format creates a larger instruction word, which can be up to 3
