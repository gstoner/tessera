# 12.3.1. 16-bit Parameter Interpolation

> RDNA4 ISA — pages 159–159

The VINTERP instructions include a builtin "S_WAIT_EXPCNT" to easily allow data hazard resolution for data
produced by DS_PARAM_LOAD.

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

These instructions consume data that was supplied by DS_PARAM_LOAD. These instructions contain a built-in
"S_WAIT_EXPCNT <= N" capability to allow for efficient software pipelining.

  ds_param_load V0,     attr0
  ds_param_load V10, attr1
  ds_param_load V20, attr2
  ds_param_load V30, attr3
  v_interp_p0     V1,    V0[1],   Vi, V0[0]   S_WAIT_EXPCNT <=3 //Wait V0
  v_interp_p0     V11, V10[1], Vi, V10[0]     S_WAIT_EXPCNT <=2
  v_interp_p0     V21, V20[1], Vi, V20[0]     S_WAIT_EXPCNT <=1
  v_interp_p0     V31, V30[1], Vi, V30[0]     S_WAIT_EXPCNT <=0 //Wait V30
  v_interp_p2     V2,    V0[2],   Vj, V1
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
