# 10.1.2. D16 Instructions

> RDNA4 ISA — pages 129–129

Instruction Fields
VADDR0-4       5x8         Five 8-bit VGPR address fields. Each one provides one or a consecutive group of address VGPRs.
                                             Fields Available only in VSAMPLE
VADDR0-3       4x8         Four 8-bit VGPR address fields. Each one provides one or a consecutive group of address VGPRs.
SAMP           9           Specifies which SGPR supplies S# (sampler constant) in four consecutive SGPRs. Must be a
                           multiple of 4, in the range 0-120.
UNRM           1           Force address to be un-normalized. Must be set to 1 for Image stores & atomics.
                           0: for image ops with samplers, S,T,R from [0.0, 1.0] span the entire texture map;
                           1: for image ops with samplers, S,T,R from [0.0 to N] span the texture map, where N is width,
                           height or depth. Array/cube slice, lod, bias etc. are not affected. Image ops without sampler are
                           not affected. UINT inputs are "unnormalized".
                           This bit is logically OR’d with the S#.force_unnormalized bit.
LWE            1           LOD Warning Enable. When set to 1, a texture fetch may return "LOD_CLAMPED = 1", and causes
                           a VGPR write into DST+1 (first GPR after all fetch-dest gprs). LWE only works for sampler ops;
                           LWE is ignored for non-sampler ops.

10.1.1. Texture Fault Enable (TFE) and LOD Warning Enable (LWE)
This is related to "Partially Resident Textures".

When either of these bits are set in the instruction, any texture fetch may return one extra VGPR after all of the
data-return VGPRs. This data is returned uniquely to each thread and indicates the error / warning status of
that thread and nothing is returned if no thread experiences a texture fault or LOD warning.

The data returned is: TEXEL_FAIL | (LOD_WARNING << 1) | (LOD << 16)
  • TEXEL_FAIL : 1 bit indicating that 1 or more texels for this pixel produced a NACK.
    "failure" means accessing an unmapped page.
      ◦ TFE == 0
          ▪ TEX writes the data for threads that didn’t NACK to VGPR DST
          ▪ TEX writes zeros or the result of blend using zeros for samples that NACKed to VGPR DST
       ◦ TFE == 1
          ▪ VGPR DST is written similar to above
          ▪ TEX writes to VGPR DST+1 with a status where the bits corresponding to threads that NACKed are
            set to 1
  • LOD_WARNING : 1 bit indicating a that a pixel attempted to access a texel at too small a LOD:
    warn = ( LOD < T#.min_lod_warning)
  • LOD : indicates which LOD was attempted to be accessed that caused the NACK. Returns the floor of the
    requested LOD.

A pixel cannot receive both TEXEL_FAIL and LOD_WARNING: TEXEL_FAIL takes precedence.

10.1.2. D16 Instructions
Load-format and store-format instructions also come in a "D16" variant. For stores, each 32-bit VGPR holds two
16-bit data elements that are passed to the texture unit. The texture unit converts them to the texture format
before writing to memory. For loads, data returned from the texture unit is converted to 16 bits, and a pair of
data are stored in each 32- bit VGPR (LSBs first, then MSBs). The DMASK bit represents individual 16- bit
elements; so, when DMASK=0011 for an image-load, two 16-bit components are loaded into a single 32-bit
