# 10.10. Partially Resident Textures

> RDNA4 ISA — pages 144–144

10.10. Partially Resident Textures
"Partially Resident Textures" provides support for texture maps in which not all levels of detail are resident in
memory. The shader compiler declares the texture map as being P.R.T. in the resource, but the shader
program must also be aware of this because if a texture fetch accesses a MIP level that is not present, the
texture unit returns an extra DWORD of status into VGPRs indicating the fetch failure. If any of the texels are
not present in memory, the texture cache returns NACK that causes a non-zero value to be written into
DST_VGPR+1 for each failing thread. The value may represent the LOD requested. The shader program must
allocate this extra VGPR for all PRT texture fetches and check that it is zero after the fetch. The user should
initialize this PRT VGPR to zero prior to issuing a texture fetch which may return a PRT result.

PRT is enabled when the texture resource MIN_LOD_WARN value is non-zero. Normal textures cannot NACK,
so only PRT’s can get a NACK, and a NACK causes a write to DST_VGPR+Num_VGPRS. E.g. if a SAMPLE loads 4
values into 4 VGPRs: 4,5,6,7 then PRT may return NACK status into VGPR_8.
