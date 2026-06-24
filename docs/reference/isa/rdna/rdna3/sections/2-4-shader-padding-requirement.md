# 2.4. Shader Padding Requirement

> RDNA3 ISA — pages 21–21

WGP mode
   In this mode, the LDS is one large contiguous memory that all waves on the WGP can access. In WGP mode,
   waves of a work-group may be distributed across both CU’s (all 4 SIMD32’s) in the WGP.
   LDS_PARAM_LOAD and LDS_DIRECT_LOAD are not supported in WGP mode.

The WGP (and LDS) can simultaneously have some waves running in WGP mode and other waves in CU mode
running.

A barrier is a synchronization primitive which makes each wave reach a given point in the shader before any
wave proceeds.

2.4. Shader Padding Requirement
Due to aggressive instruction prefetching used in some graphics devices, the user must pad all shaders with 64
extra DWORDs (256 bytes) of data past the end of the shader. It is recommended to use the S_CODE_END
instruction as padding. This ensures that if the instruction prefetch hardware goes beyond the end of the
shader, it may not reach into uninitialized memory (or unmapped memory pages).

The amount of shader padding required is related to how far the shader may prefetch ahead. The shader can be
set to prefetch 1, 2 or 3 cachelines (64 bytes) ahead of the current program counter. This is controlled via a
wave-launch state register, or by the shader program itself with S_SET_INST_PREFETCH_DISTANCE.
