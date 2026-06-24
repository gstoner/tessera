# 12.3.1. 16-bit Parameter Interpolation

> RDNA3 ISA — pages 133–133

Instructions Restrictions and Limitations:
  • V_INTERP instructions do not detect or report exceptions
  • V_INTERP instructions do not support data forwarding into inputs that would normally come from LDS
    data (sources A and C for V_INTERP_P10_* and source A for V_INTERP_P2_*).

VGPRs are preloaded with some or all of:
  • I_persp_sample, J_persp_sample, I_persp_center, J_persp_center,
  • I_persp_centroid, J_persp_centroid,
  • I/W, J/W, 1.0/W,
  • I_linear_sample, J_linear_sample,
  • I_linear_center, J_linear_center,
  • I_linear_centroid, J_linear_centroid

These instructions consume data that was supplied by LDS_PARAM_LOAD. These instructions contain a built-
in "s_waitcnt EXPcnt <= N" capability to allow for efficient software pipelining.

  lds_param_load V0,    attr0
  lds_param_load V10, attr1
  lds_param_load V20, attr2
  lds_param_load V30, attr3
  v_interp_p0     V1,   V0[1],   Vi, V0[0]   s_waitcnt EXPcnt<=3 //Wait V0
  v_interp_p0     V11, V10[1], Vi, V10[0]    s_waitcnt EXPcnt<=2
  v_interp_p0     V21, V20[1], Vi, V20[0]    s_waitcnt EXPcnt<=1
  v_interp_p0     V31, V30[1], Vi, V30[0]    s_waitcnt EXPcnt<=0 //Wait V30
  v_interp_p2     V2,   V0[2],   Vj, V1
  v_interp_p2     V12, V10[2], Vj, V11
  v_interp_p2     V22, V20[2], Vj, V21
  v_interp_p2     V32, V30[2], Vj, V31

12.3.1. 16-bit Parameter Interpolation
16-bit interpolation operates on pairs of attribute values packed into a 16-bit VGPR. These use the same I and J
values during interpolation. OPSEL is used to select the upper or lower portion of the data.

There are variants of the 16-bit interpolation instructions that override the round mode to "round toward zero".

V_INTERP_P10_F16_F32 dst.f32 = vgpr_hi/lo.f16 * vgpr.f32 + vgpr_hi/lo.f16 // tmp = P10 * I + P0
  • allows OPSEL; Src0 uses DPP8=1,1,1,1,5,5,5,5; Src2 uses DPP8=0,0,0,0,4,4,4,4

V_INTERP_P2_F16_F32 dst.f16 = vgpr_hi/lo.f16 * vgpr.f32 + vgpr.f32 // dst = P2 * J + tmp
  • allows OPSEL; Src0 uses DPP8=2,2,2,2,6,6,6,6

12.4. LDS Direct Load
Direct loads are only available in LDS, not in GDS. Direct access is allowed only in CU mode, not WGP mode.

The LDS_DIRECT_LOAD instruction reads a single DWORD from LDS and returns it to a VGPR, broadcasting it
