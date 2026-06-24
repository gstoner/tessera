# 2.4. Shader Padding Requirement

> RDNA4 ISA — pages 22–22

WGP mode
   In this mode, the LDS is one large contiguous memory that waves on the WGP allocate from, up to the same
   maximum allocation size. In WGP mode, waves of a work-group may be distributed across both CU’s (all 4
   SIMD32s) in the WGP. DS_PARAM_LOAD and DS_DIRECT_LOAD are not supported in WGP mode.

The WGP (and LDS) can simultaneously have some waves running in WGP mode and other waves in CU mode
running. LDS performance may degrade when wave reference data on the "opposite side" from the SIMD
they’re on.

2.4. Shader Padding Requirement
Due to aggressive instruction prefetching used in RDNA4 devices, the user must pad all shaders with 64 extra
DWORDs (256 bytes) of data past the end of the shader. It is recommended to use the S_CODE_END instruction
as padding to make it easier to identify the end of a shader when debugging. This ensures that if the instruction
prefetch hardware goes beyond the end of the shader, it may not reach into uninitialized memory (or
unmapped memory pages).

2.5. Whole Quad Mode
Whole Quad Mode (WQM) is a method of operating where if any thread in a set of 4 threads is enabled, then all
4 act as if they were enabled. This mode is useful for operations which rely on all threads in a quad to be
enabled to calculate data which relies on knowing the values in neighboring pixels. SAMPLE level-of-detail
calculation is a common example of this.

Whole Quad Mode can be applied explicitly using S_WQM_B32 or S_WQM_B64 to modify the EXEC mask to
enable groups of 4 lanes.

Whole Quad Mode is temporarily applied for these cases:
  • V_INTERP_* instructions: uses data from 3 lanes in a quad
  • DS_PARAM_LOAD, DS_DIRECT_LOAD: loads data into 3 lanes per quad
  • SAMPLE: operations which use neighboring pixel data to calculate a level-of-detail
