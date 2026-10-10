
/tmp/tmpneacxi8w.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_5bd481a230a95469>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b128 s[20:23], s[0:1], 0xc8                         // 000000001b04: f4004500 f80000c8
	s_load_b64 s[4:5], s[0:1], 0xd8                            // 000000001b0c: f4002100 f80000d8
	s_mov_b32 s8, ttmp7                                        // 000000001b14: be880073
	s_ashr_i32 s9, ttmp7, 31                                   // 000000001b18: 86099f73
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001b1c: 320a0081
	s_lshl_b64 s[12:13], s[8:9], 7                             // 000000001b20: 848c8708
	v_lshlrev_b32_e32 v10, 4, v0                               // 000000001b24: 30140084
	s_clause 0x3                                               // 000000001b28: bf850003
	s_load_b64 s[14:15], s[0:1], 0x8                           // 000000001b2c: f4002380 f8000008
	s_load_b64 s[16:17], s[0:1], 0x30                          // 000000001b34: f4002400 f8000030
	s_load_b64 s[10:11], s[0:1], 0x58                          // 000000001b3c: f4002280 f8000058
	s_load_b64 s[6:7], s[0:1], 0x80                            // 000000001b44: f4002180 f8000080
	s_mov_b32 s2, ttmp9                                        // 000000001b4c: be820075
	v_or_b32_e32 v1, s12, v5                                   // 000000001b50: 38020a0c
	v_mul_u32_u24_e32 v13, 48, v5                              // 000000001b54: 161a0ab0
	v_and_b32_e32 v10, 16, v10                                 // 000000001b58: 36141490
	v_mov_b32_e32 v2, s13                                      // 000000001b5c: 7e04020d
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b60: 86039f75
	v_dual_mov_b32 v18, 0 :: v_dual_and_b32 v23, 32, v0        // 000000001b64: ca240080 121600a0
	s_delay_alu instid0(valu_dep_3)                            // 000000001b6c: bf870003
	v_add_nc_u32_e32 v19, v13, v10                             // 000000001b70: 4a26150d
	s_lshl_b64 s[18:19], s[2:3], 6                             // 000000001b74: 84928602
	v_dual_mov_b32 v13, s13 :: v_dual_and_b32 v6, 0x60, v5     // 000000001b78: ca24000d 0d060aff 00000060
	s_wait_kmcnt 0x0                                           // 000000001b84: bfc70000
	s_add_nc_u64 s[8:9], s[20:21], -1                          // 000000001b88: a988c114
	v_add_co_u32 v4, s2, s18, v5                               // 000000001b8c: d7000204 02020a12
	v_cmp_gt_u64_e32 vcc_lo, s[8:9], v[1:2]                    // 000000001b94: 7cb80208
	v_and_b32_e32 v22, 15, v0                                  // 000000001b98: 362c008f
	v_add_co_ci_u32_e64 v9, null, s19, 0, s2                   // 000000001b9c: d5207c09 00090013
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_4)// 000000001ba4: bf870234
	v_mul_lo_u32 v14, s5, v4                                   // 000000001ba8: d72c000e 02020805
	v_dual_cndmask_b32 v3, s9, v2 :: v_dual_and_b32 v16, 8, v5 // 000000001bb0: ca640409 03100a88
	v_cndmask_b32_e32 v1, s8, v1, vcc_lo                       // 000000001bb8: 02020208
	v_mul_lo_u32 v9, s4, v9                                    // 000000001bbc: d72c0009 02021204
	v_or_b32_e32 v8, v22, v23                                  // 000000001bc4: 38102f16
	v_or_b32_e32 v15, s12, v6                                  // 000000001bc8: 381e0c0c
	v_mul_lo_u32 v12, v3, s4                                   // 000000001bcc: d72c000c 02000903
	v_mul_lo_u32 v11, v1, s5                                   // 000000001bd4: d72c000b 02000b01
	v_mad_co_u64_u32 v[1:2], null, v1, s4, s[14:15]            // 000000001bdc: d6fe7c01 00380901
	v_mad_co_u64_u32 v[3:4], null, s4, v4, s[16:17]            // 000000001be4: d6fe7c03 00420804
	v_or_b32_e32 v7, 16, v6                                    // 000000001bec: 380e0c90
	v_cmp_gt_u32_e64 s3, 0x80, v0                              // 000000001bf0: d44c0003 020200ff 00000080
	v_and_b32_e32 v0, 47, v0                                   // 000000001bfc: 360000af
	v_or_b32_e32 v17, 16, v8                                   // 000000001c00: 38221090
	v_or_b32_e32 v24, 1, v16                                   // 000000001c04: 38302081
	v_or_b32_e32 v25, 2, v16                                   // 000000001c08: 38322082
	v_add3_u32 v2, v12, v2, v11                                // 000000001c0c: d6550002 042e050c
	v_add_co_u32 v20, vcc_lo, v1, v10                          // 000000001c14: d7006a14 02021501
	v_add3_u32 v1, v14, v4, v9                                 // 000000001c1c: d6550001 0426090e
	v_or_b32_e32 v12, v15, v16                                 // 000000001c24: 3818210f
	s_wait_alu depctr_va_vcc(0)                                // 000000001c28: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, 0, v2, vcc_lo               // 000000001c2c: d5207c15 01aa0480
	v_or_b32_e32 v2, v6, v22                                   // 000000001c34: 38042d06
	v_add_co_u32 v42, vcc_lo, v3, v10                          // 000000001c38: d7006a2a 02021503
	s_wait_alu depctr_va_vcc(0)                                // 000000001c40: bf88ff9d
	v_add_co_ci_u32_e64 v43, null, 0, v1, vcc_lo               // 000000001c44: d5207c2b 01aa0280
	s_delay_alu instid0(valu_dep_3)                            // 000000001c4c: bf870003
	v_mul_u32_u24_e32 v1, 48, v2                               // 000000001c50: 160204b0
	v_or_b32_e32 v4, v7, v22                                   // 000000001c54: 38082d07
	v_mov_b32_e32 v9, s19                                      // 000000001c58: 7e120213
	v_mul_u32_u24_e32 v0, 48, v0                               // 000000001c5c: 160000b0
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[12:13]                // 000000001c60: 7ca81814
	v_or_b32_e32 v44, v1, v16                                  // 000000001c64: 38582101
	v_mul_u32_u24_e32 v1, 48, v17                              // 000000001c68: 160222b0
	v_mul_u32_u24_e32 v2, 48, v4                               // 000000001c6c: 160408b0
	v_or_b32_e32 v27, v16, v0                                  // 000000001c70: 38360110
	v_or_b32_e32 v0, v24, v15                                  // 000000001c74: 38001f18
	s_wait_alu depctr_va_vcc(0)                                // 000000001c78: bf88ff9d
	v_cndmask_b32_e32 v3, 0, v12, vcc_lo                       // 000000001c7c: 02061880
	v_or_b32_e32 v28, v1, v16                                  // 000000001c80: 38382101
	v_mov_b32_e32 v1, s13                                      // 000000001c84: 7e02020d
	v_or_b32_e32 v45, v2, v16                                  // 000000001c88: 385a2102
	v_cndmask_b32_e32 v2, 0, v13, vcc_lo                       // 000000001c8c: 02041a80
	v_or_b32_e32 v26, s12, v7                                  // 000000001c90: 38340e0c
	s_lshr_b64 s[8:9], s[4:5], 5                               // 000000001c94: 85888504
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000001c98: 7ca80014
	s_lshr_b32 s12, s5, 5                                      // 000000001c9c: 850c8505
	v_or_b32_e32 v30, 3, v16                                   // 000000001ca0: 383c2083
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ca4: bf88ff9e
	v_mul_lo_u32 v4, s12, v3                                   // 000000001ca8: d72c0004 0202060c
	v_or_b32_e32 v36, 7, v16                                   // 000000001cb0: 38482087
	v_or_b32_e32 v8, s18, v8                                   // 000000001cb4: 38101012
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb8: bf88ff9d
	v_cndmask_b32_e32 v7, 0, v0, vcc_lo                        // 000000001cbc: 020e0080
	v_or_b32_e32 v0, v25, v15                                  // 000000001cc0: 38001f19
	v_mul_lo_u32 v10, s8, v2                                   // 000000001cc4: d72c000a 02020408
	v_mad_co_u64_u32 v[2:3], null, s8, v3, 0                   // 000000001ccc: d6fe7c02 02020608
	v_cndmask_b32_e32 v6, 0, v1, vcc_lo                        // 000000001cd4: 020c0280
	v_mul_lo_u32 v14, s12, v7                                  // 000000001cd8: d72c000e 02020e0c
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000001ce0: 7ca80014
	v_mov_b32_e32 v11, s13                                     // 000000001ce4: 7e16020d
	v_cmp_gt_i64_e64 s2, s[22:23], v[8:9]                      // 000000001ce8: d4540002 02021016
	v_add_nc_u32_e32 v91, 0x1800, v27                          // 000000001cf0: 4ab636ff 00001800
	v_mov_b32_e32 v75, 0                                       // 000000001cf8: 7e960280
	v_add3_u32 v3, v3, v10, v4                                 // 000000001cfc: d6550003 04121503
	s_wait_alu depctr_va_vcc(0)                                // 000000001d04: bf88ff9d
	v_cndmask_b32_e32 v31, 0, v1, vcc_lo                       // 000000001d08: 023e0280
	v_mul_lo_u32 v29, s8, v6                                   // 000000001d0c: d72c001d 02020c08
	v_mad_co_u64_u32 v[6:7], null, s8, v7, 0                   // 000000001d14: d6fe7c06 02020e08
	v_or_b32_e32 v10, v30, v15                                 // 000000001d1c: 38141f1e
	v_cndmask_b32_e32 v32, 0, v0, vcc_lo                       // 000000001d20: 02400080
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001d24: 3e000482
	s_wait_alu depctr_va_sdst(0)                               // 000000001d28: bf88f19f
	v_cndmask_b32_e64 v5, 0, v9, s2                            // 000000001d2c: d5010005 000a1280
	v_cndmask_b32_e64 v4, 0, v8, s2                            // 000000001d34: d5010004 000a1080
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001d3c: 7ca81414
	v_mad_co_u64_u32 v[2:3], null, s8, v32, 0                  // 000000001d40: d6fe7c02 02024008
	v_add3_u32 v7, v7, v29, v14                                // 000000001d48: d6550007 043a3b07
	v_mul_lo_u32 v29, s8, v31                                  // 000000001d50: d72c001d 02023e08
	v_or_b32_e32 v31, 4, v16                                   // 000000001d58: 383e2084
	v_mul_lo_u32 v14, s12, v32                                 // 000000001d5c: d72c000e 0202400c
	s_wait_alu depctr_va_vcc(0)                                // 000000001d64: bf88ff9d
	v_dual_cndmask_b32 v32, 0, v11 :: v_dual_cndmask_b32 v33, 0, v10// 000000001d68: ca521680 20201480
	v_add_co_u32 v51, vcc_lo, s10, v0                          // 000000001d70: d7006a33 0202000a
	v_or_b32_e32 v10, v31, v15                                 // 000000001d78: 38141f1f
	s_wait_alu depctr_va_vcc(0)                                // 000000001d7c: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, s11, v1, vcc_lo             // 000000001d80: d5207c34 01aa020b
	v_add3_u32 v3, v3, v29, v14                                // 000000001d88: d6550003 043a3b03
	v_mul_lo_u32 v29, s8, v32                                  // 000000001d90: d72c001d 02024008
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001d98: 7ca81414
	v_or_b32_e32 v32, 5, v16                                   // 000000001d9c: 38402085
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001da0: 3e000c82
	v_mul_lo_u32 v14, s12, v33                                 // 000000001da4: d72c000e 0202420c
	v_mad_co_u64_u32 v[6:7], null, s8, v33, 0                  // 000000001dac: d6fe7c06 02024208
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v50, 0             // 000000001db4: ca100080 45320080
	s_wait_alu depctr_va_vcc(0)                                // 000000001dbc: bf88ff9d
	v_dual_cndmask_b32 v33, 0, v11 :: v_dual_cndmask_b32 v34, 0, v10// 000000001dc0: ca521680 21221480
	v_or_b32_e32 v10, v32, v15                                 // 000000001dc8: 38141f20
	v_add_co_u32 v54, vcc_lo, s10, v0                          // 000000001dcc: d7006a36 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001dd4: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s11, v1, vcc_lo             // 000000001dd8: d5207c37 01aa020b
	s_delay_alu instid0(valu_dep_3)                            // 000000001de0: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001de4: 7ca81414
	v_add3_u32 v7, v7, v29, v14                                // 000000001de8: d6550007 043a3b07
	v_mul_lo_u32 v29, s8, v33                                  // 000000001df0: d72c001d 02024208
	v_or_b32_e32 v33, 6, v16                                   // 000000001df8: 38422086
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001dfc: 3e000482
	v_mul_lo_u32 v14, s12, v34                                 // 000000001e00: d72c000e 0202440c
	v_mad_co_u64_u32 v[2:3], null, s8, v34, 0                  // 000000001e08: d6fe7c02 02024408
	s_wait_alu depctr_va_vcc(0)                                // 000000001e10: bf88ff9d
	v_dual_cndmask_b32 v34, 0, v11 :: v_dual_cndmask_b32 v35, 0, v10// 000000001e14: ca521680 22221480
	v_or_b32_e32 v10, v33, v15                                 // 000000001e1c: 38141f21
	v_add_co_u32 v56, vcc_lo, s10, v0                          // 000000001e20: d7006a38 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e28: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s11, v1, vcc_lo             // 000000001e2c: d5207c39 01aa020b
	s_delay_alu instid0(valu_dep_3)                            // 000000001e34: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001e38: 7ca81414
	v_add3_u32 v3, v3, v29, v14                                // 000000001e3c: d6550003 043a3b03
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001e44: 3e000c82
	v_or_b32_e32 v6, v36, v15                                  // 000000001e48: 380c1f24
	v_mov_b32_e32 v7, s13                                      // 000000001e4c: 7e0e020d
	v_mul_lo_u32 v29, s12, v35                                 // 000000001e50: d72c001d 0202460c
	s_wait_alu depctr_va_vcc(0)                                // 000000001e58: bf88ff9d
	v_dual_cndmask_b32 v14, 0, v11 :: v_dual_cndmask_b32 v37, 0, v10// 000000001e5c: ca521680 0e241480
	v_mad_co_u64_u32 v[10:11], null, s8, v35, 0                // 000000001e64: d6fe7c0a 02024608
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[6:7]                  // 000000001e6c: 7ca80c14
	v_mul_lo_u32 v34, s8, v34                                  // 000000001e70: d72c0022 02024408
	s_delay_alu instid0(valu_dep_4)                            // 000000001e78: bf870004
	v_mul_lo_u32 v38, s8, v14                                  // 000000001e7c: d72c0026 02021c08
	v_mul_lo_u32 v35, s12, v37                                 // 000000001e84: d72c0023 02024a0c
	v_mad_co_u64_u32 v[14:15], null, s8, v37, 0                // 000000001e8c: d6fe7c0e 02024a08
	v_add_co_u32 v59, s4, s10, v0                              // 000000001e94: d700043b 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e9c: bf88ff9d
	v_dual_cndmask_b32 v6, 0, v6 :: v_dual_cndmask_b32 v7, 0, v7// 000000001ea0: ca520c80 06060e80
	s_wait_alu depctr_va_sdst(0)                               // 000000001ea8: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s11, v1, s4                 // 000000001eac: d5207c3c 0012020b
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001eb4: 3e000482
	v_add3_u32 v15, v15, v38, v35                              // 000000001eb8: d655000f 048e4d0f
	v_add3_u32 v11, v11, v34, v29                              // 000000001ec0: d655000b 0476450b
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v61, 0             // 000000001ec8: ca100080 4e3c0080
	v_mov_b32_e32 v48, 0                                       // 000000001ed0: 7e600280
	s_delay_alu instid0(valu_dep_4)                            // 000000001ed4: bf870004
	v_lshlrev_b64_e32 v[2:3], 2, v[14:15]                      // 000000001ed8: 3e041c82
	v_mul_lo_u32 v14, s12, v6                                  // 000000001edc: d72c000e 02020c0c
	v_mul_lo_u32 v15, s8, v7                                   // 000000001ee4: d72c000f 02020e08
	v_mad_co_u64_u32 v[6:7], null, s8, v6, 0                   // 000000001eec: d6fe7c06 02020c08
	v_add_co_u32 v62, vcc_lo, s10, v0                          // 000000001ef4: d7006a3e 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001efc: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, s11, v1, vcc_lo             // 000000001f00: d5207c3f 01aa020b
	v_lshlrev_b64_e32 v[0:1], 2, v[10:11]                      // 000000001f08: 3e001482
	v_add_co_u32 v67, s4, s10, v2                              // 000000001f0c: d7000443 0202040a
	v_add3_u32 v7, v7, v15, v14                                // 000000001f14: d6550007 043a1f07
	v_or_b32_e32 v10, v26, v16                                 // 000000001f1c: 3814211a
	s_wait_alu depctr_va_sdst(0)                               // 000000001f20: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s11, v3, s4                 // 000000001f24: d5207c44 0012060b
	v_add_co_u32 v64, vcc_lo, s10, v0                          // 000000001f2c: d7006a40 0202000a
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000001f34: 3e040c82
	v_or_b32_e32 v6, s18, v17                                  // 000000001f38: 380c2212
	v_mov_b32_e32 v11, s13                                     // 000000001f3c: 7e16020d
	s_wait_alu depctr_va_vcc(0)                                // 000000001f40: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s11, v1, vcc_lo             // 000000001f44: d5207c41 01aa020b
	v_or_b32_e32 v0, v26, v24                                  // 000000001f4c: 3800311a
	v_add_nc_u32_e32 v92, 0x1800, v28                          // 000000001f50: 4ab838ff 00001800
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001f58: 7ca81414
	v_dual_mov_b32 v1, s13 :: v_dual_mov_b32 v72, 0            // 000000001f5c: ca10000d 01480080
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v7, s19            // 000000001f64: ca100080 3a060013
	v_mov_b32_e32 v38, 0                                       // 000000001f6c: 7e4c0280
	s_wait_alu depctr_va_vcc(0)                                // 000000001f70: bf88ff9d
	v_dual_cndmask_b32 v14, 0, v11 :: v_dual_cndmask_b32 v15, 0, v10// 000000001f74: ca521680 0e0e1480
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000001f7c: 7ca80014
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v39, 0             // 000000001f80: ca100080 42260080
	v_mov_b32_e32 v46, 0                                       // 000000001f88: 7e5c0280
	s_delay_alu instid0(valu_dep_4)                            // 000000001f8c: bf870004
	v_mul_lo_u32 v16, s12, v15                                 // 000000001f90: d72c0010 02021e0c
	v_mul_lo_u32 v17, s8, v14                                  // 000000001f98: d72c0011 02021c08
	v_mad_co_u64_u32 v[14:15], null, s8, v15, 0                // 000000001fa0: d6fe7c0e 02021e08
	s_wait_alu depctr_va_vcc(0)                                // 000000001fa8: bf88ff9d
	v_cndmask_b32_e32 v29, 0, v0, vcc_lo                       // 000000001fac: 023a0080
	v_or_b32_e32 v0, v26, v25                                  // 000000001fb0: 3800331a
	v_cndmask_b32_e32 v24, 0, v1, vcc_lo                       // 000000001fb4: 02300280
	v_add_co_u32 v70, vcc_lo, s10, v2                          // 000000001fb8: d7006a46 0202040a
	s_wait_alu depctr_va_vcc(0)                                // 000000001fc0: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, s11, v3, vcc_lo             // 000000001fc4: d5207c47 01aa060b
	v_cmp_gt_i64_e64 s4, s[20:21], v[0:1]                      // 000000001fcc: d4540004 02020014
	v_add3_u32 v15, v15, v17, v16                              // 000000001fd4: d655000f 0442230f
	v_mul_lo_u32 v16, s12, v29                                 // 000000001fdc: d72c0010 02023a0c
	v_mul_lo_u32 v17, s8, v24                                  // 000000001fe4: d72c0011 02023008
	v_mad_co_u64_u32 v[2:3], null, s8, v29, 0                  // 000000001fec: d6fe7c02 02023a08
	v_cmp_gt_i64_e32 vcc_lo, s[22:23], v[6:7]                  // 000000001ff4: 7ca80c16
	s_wait_alu depctr_va_sdst(0)                               // 000000001ff8: bf88f19f
	v_cndmask_b32_e64 v24, 0, v1, s4                           // 000000001ffc: d5010018 00120280
	v_cndmask_b32_e64 v25, 0, v0, s4                           // 000000002004: d5010019 00120080
	v_lshlrev_b64_e32 v[0:1], 2, v[14:15]                      // 00000000200c: 3e001c82
	v_mov_b32_e32 v15, s13                                     // 000000002010: 7e1e020d
	v_or_b32_e32 v14, v26, v30                                 // 000000002014: 381c3d1a
	v_mul_lo_u32 v24, s8, v24                                  // 000000002018: d72c0018 02023008
	v_add3_u32 v3, v3, v17, v16                                // 000000002020: d6550003 04422303
	v_mul_lo_u32 v29, s12, v25                                 // 000000002028: d72c001d 0202320c
	v_add_co_u32 v73, s5, s10, v0                              // 000000002030: d7000549 0202000a
	v_cmp_gt_i64_e64 s4, s[20:21], v[14:15]                    // 000000002038: d4540004 02021c14
	s_wait_alu depctr_va_sdst(0)                               // 000000002040: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s11, v1, s5                 // 000000002044: d5207c4a 0016020b
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 00000000204c: 3e000482
	v_mov_b32_e32 v3, s13                                      // 000000002050: 7e06020d
	v_or_b32_e32 v2, v26, v31                                  // 000000002054: 38043f1a
	v_mad_co_u64_u32 v[16:17], null, s8, v25, 0                // 000000002058: d6fe7c10 02023208
	v_cndmask_b32_e64 v15, 0, v15, s4                          // 000000002060: d501000f 00121e80
	v_cndmask_b32_e64 v14, 0, v14, s4                          // 000000002068: d501000e 00121c80
	v_add_co_u32 v76, s5, s10, v0                              // 000000002070: d700054c 0202000a
	v_cmp_gt_i64_e64 s4, s[20:21], v[2:3]                      // 000000002078: d4540004 02020414
	s_delay_alu instid0(valu_dep_4)                            // 000000002080: bf870004
	v_mul_lo_u32 v25, s8, v15                                  // 000000002084: d72c0019 02021e08
	s_wait_alu depctr_va_sdst(0)                               // 00000000208c: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s11, v1, s5                 // 000000002090: d5207c4d 0016020b
	v_add3_u32 v17, v17, v24, v29                              // 000000002098: d6550011 04763111
	v_mul_lo_u32 v24, s12, v14                                 // 0000000020a0: d72c0018 02021c0c
	v_mad_co_u64_u32 v[14:15], null, s8, v14, 0                // 0000000020a8: d6fe7c0e 02021c08
	v_cndmask_b32_e64 v30, 0, v2, s4                           // 0000000020b0: d501001e 00120480
	v_or_b32_e32 v2, v26, v32                                  // 0000000020b8: 3804411a
	v_lshlrev_b64_e32 v[0:1], 2, v[16:17]                      // 0000000020bc: 3e002082
	v_cndmask_b32_e64 v29, 0, v3, s4                           // 0000000020c0: d501001d 00120680
	s_wait_alu depctr_va_vcc(0)                                // 0000000020c8: bf88ff9d
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo                        // 0000000020cc: 020e0e80
	v_mul_lo_u32 v31, s12, v30                                 // 0000000020d0: d72c001f 02023c0c
	v_cmp_gt_i64_e64 s4, s[20:21], v[2:3]                      // 0000000020d8: d4540004 02020414
	v_add3_u32 v15, v15, v25, v24                              // 0000000020e0: d655000f 0462330f
	v_mov_b32_e32 v25, s13                                     // 0000000020e8: 7e32020d
	v_or_b32_e32 v24, v26, v33                                 // 0000000020ec: 3830431a
	v_add_co_u32 v79, s5, s10, v0                              // 0000000020f0: d700054f 0202000a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020f8: bf88f19f
	v_add_co_ci_u32_e64 v80, null, s11, v1, s5                 // 0000000020fc: d5207c50 0016020b
	v_lshlrev_b64_e32 v[0:1], 2, v[14:15]                      // 000000002104: 3e001c82
	v_cndmask_b32_e64 v15, 0, v2, s4                           // 000000002108: d501000f 00120480
	v_or_b32_e32 v2, v26, v36                                  // 000000002110: 3804491a
	v_mul_lo_u32 v29, s8, v29                                  // 000000002114: d72c001d 02023a08
	v_mad_co_u64_u32 v[16:17], null, s8, v30, 0                // 00000000211c: d6fe7c10 02023c08
	v_mov_b32_e32 v36, 0                                       // 000000002124: 7e480280
	v_cmp_gt_i64_e64 s5, s[20:21], v[24:25]                    // 000000002128: d4540005 02023014
	v_cndmask_b32_e64 v14, 0, v3, s4                           // 000000002130: d501000e 00120680
	v_cmp_gt_i64_e64 s4, s[20:21], v[2:3]                      // 000000002138: d4540004 02020414
	v_mul_lo_u32 v26, s12, v15                                 // 000000002140: d72c001a 02021e0c
	v_dual_cndmask_b32 v6, 0, v6 :: v_dual_mov_b32 v37, 0      // 000000002148: ca500c80 06240080
	v_mov_b32_e32 v40, 0                                       // 000000002150: 7e500280
	s_wait_alu depctr_va_sdst(0)                               // 000000002154: bf88f19f
	v_cndmask_b32_e64 v25, 0, v25, s5                          // 000000002158: d5010019 00163280
	v_cndmask_b32_e64 v24, 0, v24, s5                          // 000000002160: d5010018 00163080
	v_add3_u32 v17, v17, v29, v31                              // 000000002168: d6550011 047e3b11
	v_cndmask_b32_e64 v3, 0, v3, s4                            // 000000002170: d5010003 00120680
	v_cndmask_b32_e64 v2, 0, v2, s4                            // 000000002178: d5010002 00120480
	v_mul_lo_u32 v29, s8, v14                                  // 000000002180: d72c001d 02021c08
	v_mad_co_u64_u32 v[14:15], null, s8, v15, 0                // 000000002188: d6fe7c0e 02021e08
	v_mul_lo_u32 v30, s12, v24                                 // 000000002190: d72c001e 0202300c
	v_mul_lo_u32 v31, s8, v25                                  // 000000002198: d72c001f 02023208
	v_mad_co_u64_u32 v[24:25], null, s8, v24, 0                // 0000000021a0: d6fe7c18 02023008
	v_add_co_u32 v81, s4, s10, v0                              // 0000000021a8: d7000451 0202000a
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b0: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s11, v1, s4                 // 0000000021b4: d5207c52 0012020b
	v_lshlrev_b64_e32 v[0:1], 2, v[16:17]                      // 0000000021bc: 3e002082
	v_mul_lo_u32 v16, s12, v2                                  // 0000000021c0: d72c0010 0202040c
	v_mul_lo_u32 v17, s8, v3                                   // 0000000021c8: d72c0011 02020608
	v_mad_co_u64_u32 v[2:3], null, s8, v2, 0                   // 0000000021d0: d6fe7c02 02020408
	v_add3_u32 v15, v15, v29, v26                              // 0000000021d8: d655000f 046a3b0f
	v_add3_u32 v25, v25, v31, v30                              // 0000000021e0: d6550019 047a3f19
	v_add_co_u32 v83, s4, s10, v0                              // 0000000021e8: d7000453 0202000a
	s_wait_alu depctr_va_sdst(0)                               // 0000000021f0: bf88f19f
	v_add_co_ci_u32_e64 v84, null, s11, v1, s4                 // 0000000021f4: d5207c54 0012020b
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]                    // 0000000021fc: 3e1c1c82
	v_add3_u32 v3, v3, v17, v16                                // 000000002200: d6550003 04422303
	v_lshlrev_b64_e32 v[0:1], 2, v[24:25]                      // 000000002208: 3e003082
	v_lshlrev_b64_e32 v[16:17], 2, v[6:7]                      // 00000000220c: 3e200c82
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v30, 0             // 000000002210: ca100080 231e0080
	s_delay_alu instid0(valu_dep_4)                            // 000000002218: bf870004
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 00000000221c: 3e040482
	v_add_co_u32 v85, s4, s10, v14                             // 000000002220: d7000455 02021c0a
	s_wait_alu depctr_va_sdst(0)                               // 000000002228: bf88f19f
	v_add_co_ci_u32_e64 v86, null, s11, v15, s4                // 00000000222c: d5207c56 00121e0b
	v_add_co_u32 v87, s4, s10, v0                              // 000000002234: d7000457 0202000a
	s_wait_alu depctr_va_sdst(0)                               // 00000000223c: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s11, v1, s4                 // 000000002240: d5207c58 0012020b
	v_add_co_u32 v89, s4, s10, v2                              // 000000002248: d7000459 0202040a
	v_lshlrev_b64_e32 v[14:15], 2, v[4:5]                      // 000000002250: 3e1c0882
	s_wait_alu depctr_va_sdst(0)                               // 000000002254: bf88f19f
	v_add_co_ci_u32_e64 v90, null, s11, v3, s4                 // 000000002258: d5207c5a 0012060b
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0             // 000000002260: ca100080 22200080
	v_mov_b32_e32 v28, 0                                       // 000000002268: 7e380280
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v53, 0             // 00000000226c: ca100080 20340080
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v49, 0             // 000000002274: ca100080 1a300080
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v47, 0             // 00000000227c: ca100080 182e0080
	v_mov_b32_e32 v41, 0                                       // 000000002284: 7e520280
	v_mov_b32_e32 v31, 0                                       // 000000002288: 7e3e0280
	v_mov_b32_e32 v29, 0                                       // 00000000228c: 7e3a0280
	v_mov_b32_e32 v27, 0                                       // 000000002290: 7e360280
	v_mov_b32_e32 v25, 0                                       // 000000002294: 7e320280
	s_mov_b64 s[10:11], 0                                      // 000000002298: be8a0180
	s_branch 302                                               // 00000000229c: bfa0012e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0xc58>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022a0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000022a4: 8c7e047e
	s_mul_u64 s[14:15], s[10:11], s[22:23]                     // 0000000022a8: aa8e160a
	s_lshl_b64 s[12:13], s[10:11], 2                           // 0000000022ac: 848c820a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022b0: bf88ff9e
	s_lshl_b64 s[14:15], s[14:15], 2                           // 0000000022b4: 848e820e
	v_add_co_u32 v0, s4, v51, s12                              // 0000000022b8: d7000400 02001933
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022c0: bf88ff9e
	s_add_nc_u64 s[14:15], s[6:7], s[14:15]                    // 0000000022c4: a98e0e06
	v_add_co_ci_u32_e64 v1, null, s13, v52, s4                 // 0000000022c8: d5207c01 0012680d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022d0: bf88ff9e
	v_add_co_u32 v2, s4, s14, v14                              // 0000000022d4: d7000402 02021c0e
	s_wait_alu depctr_va_sdst(0)                               // 0000000022dc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s15, v15, s4                 // 0000000022e0: d5207c03 00121e0f
	v_add_co_u32 v4, s4, v54, s12                              // 0000000022e8: d7000404 02001936
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v55, s4                 // 0000000022f4: d5207c05 00126e0d
	s_wait_dscnt 0x0                                           // 0000000022fc: bfc60000
	s_barrier_signal -1                                        // 000000002300: be804ec1
	s_barrier_wait 0xffff                                      // 000000002304: bf94ffff
	global_load_b32 v131, v[0:1], off                          // 000000002308: ee05007c 00000083 00000000
	v_add_co_u32 v0, s4, v56, s12                              // 000000002314: d7000400 02001938
	global_load_b32 v132, v[2:3], off                          // 00000000231c: ee05007c 00000084 00000002
	s_wait_alu depctr_va_sdst(0)                               // 000000002328: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v57, s4                 // 00000000232c: d5207c01 0012720d
	v_add_co_u32 v2, s4, v59, s12                              // 000000002334: d7000402 0200193b
	global_load_b32 v133, v[4:5], off                          // 00000000233c: ee05007c 00000085 00000004
	s_wait_alu depctr_va_sdst(0)                               // 000000002348: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s13, v60, s4                 // 00000000234c: d5207c03 0012780d
	v_add_co_u32 v4, s4, v62, s12                              // 000000002354: d7000404 0200193e
	s_wait_alu depctr_va_sdst(0)                               // 00000000235c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v63, s4                 // 000000002360: d5207c05 00127e0d
	v_add_co_u32 v6, s4, v64, s12                              // 000000002368: d7000406 02001940
	s_wait_alu depctr_va_sdst(0)                               // 000000002370: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v65, s4                 // 000000002374: d5207c07 0012820d
	v_add_co_u32 v93, s4, v67, s12                             // 00000000237c: d700045d 02001943
	s_wait_alu depctr_va_sdst(0)                               // 000000002384: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s13, v68, s4                // 000000002388: d5207c5e 0012880d
	global_load_b32 v134, v[0:1], off                          // 000000002390: ee05007c 00000086 00000000
	v_add_co_u32 v0, s4, v70, s12                              // 00000000239c: d7000400 02001946
	global_load_b32 v135, v[2:3], off                          // 0000000023a4: ee05007c 00000087 00000002
	s_wait_alu depctr_va_sdst(0)                               // 0000000023b0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v71, s4                 // 0000000023b4: d5207c01 00128e0d
	v_add_co_u32 v2, s4, s14, v16                              // 0000000023bc: d7000402 0202200e
	global_load_b32 v136, v[4:5], off                          // 0000000023c4: ee05007c 00000088 00000004
	s_wait_alu depctr_va_sdst(0)                               // 0000000023d0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s15, v17, s4                 // 0000000023d4: d5207c03 0012220f
	v_add_co_u32 v4, s4, v73, s12                              // 0000000023dc: d7000404 02001949
	global_load_b32 v137, v[6:7], off                          // 0000000023e4: ee05007c 00000089 00000006
	s_wait_alu depctr_va_sdst(0)                               // 0000000023f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v74, s4                 // 0000000023f4: d5207c05 0012940d
	v_add_co_u32 v6, s4, v76, s12                              // 0000000023fc: d7000406 0200194c
	global_load_b32 v138, v[93:94], off                        // 000000002404: ee05007c 0000008a 0000005d
	s_wait_alu depctr_va_sdst(0)                               // 000000002410: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v77, s4                 // 000000002414: d5207c07 00129a0d
	v_add_co_u32 v93, s4, v79, s12                             // 00000000241c: d700045d 0200194f
	s_wait_alu depctr_va_sdst(0)                               // 000000002424: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s13, v80, s4                // 000000002428: d5207c5e 0012a00d
	global_load_b32 v139, v[0:1], off                          // 000000002430: ee05007c 0000008b 00000000
	v_add_co_u32 v0, s4, v81, s12                              // 00000000243c: d7000400 02001951
	global_load_b32 v140, v[2:3], off                          // 000000002444: ee05007c 0000008c 00000002
	s_wait_alu depctr_va_sdst(0)                               // 000000002450: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v82, s4                 // 000000002454: d5207c01 0012a40d
	v_add_co_u32 v2, s4, v83, s12                              // 00000000245c: d7000402 02001953
	global_load_b32 v141, v[4:5], off                          // 000000002464: ee05007c 0000008d 00000004
	s_wait_alu depctr_va_sdst(0)                               // 000000002470: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s13, v84, s4                 // 000000002474: d5207c03 0012a80d
	v_add_co_u32 v4, s4, v85, s12                              // 00000000247c: d7000404 02001955
	global_load_b32 v142, v[6:7], off                          // 000000002484: ee05007c 0000008e 00000006
	s_wait_alu depctr_va_sdst(0)                               // 000000002490: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v86, s4                 // 000000002494: d5207c05 0012ac0d
	v_add_co_u32 v6, s4, v87, s12                              // 00000000249c: d7000406 02001957
	global_load_b32 v143, v[93:94], off                        // 0000000024a4: ee05007c 0000008f 0000005d
	s_wait_alu depctr_va_sdst(0)                               // 0000000024b0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v88, s4                 // 0000000024b4: d5207c07 0012b00d
	v_add_co_u32 v93, s4, v89, s12                             // 0000000024bc: d700045d 02001959
	s_wait_alu depctr_va_sdst(0)                               // 0000000024c4: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s13, v90, s4                // 0000000024c8: d5207c5e 0012b40d
	s_clause 0x4                                               // 0000000024d0: bf850004
	global_load_b32 v144, v[0:1], off                          // 0000000024d4: ee05007c 00000090 00000000
	global_load_b32 v145, v[2:3], off                          // 0000000024e0: ee05007c 00000091 00000002
	global_load_b32 v146, v[4:5], off                          // 0000000024ec: ee05007c 00000092 00000004
	global_load_b32 v147, v[6:7], off                          // 0000000024f8: ee05007c 00000093 00000006
	global_load_b32 v148, v[93:94], off                        // 000000002504: ee05007c 00000094 0000005d
	ds_load_2addr_b64 v[115:118], v44 offset1:2                // 000000002510: d9dc0200 7300002c
	ds_load_2addr_b64 v[119:122], v91 offset1:2                // 000000002518: d9dc0200 7700005b
	ds_load_2addr_b64 v[123:126], v92 offset1:2                // 000000002520: d9dc0200 7b00005c
	ds_load_2addr_b64 v[127:130], v45 offset1:2                // 000000002528: d9dc0200 7f00002d
	s_add_nc_u64 s[10:11], s[10:11], 1                         // 000000002530: a98a810a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002534: bf88ff9e
	s_cmp_lg_u64 s[10:11], s[8:9]                              // 000000002538: bf11080a
	s_wait_dscnt 0x2                                           // 00000000253c: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[115:116], v[119:120], 0// 000000002540: cc464000 1a02ef73
	s_wait_dscnt 0x1                                           // 000000002548: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[115:116], v[123:124], 0// 00000000254c: cc46405d 1a02f773
	s_wait_dscnt 0x0                                           // 000000002554: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[127:128], v[119:120], 0// 000000002558: cc464065 1a02ef7f
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[127:128], v[123:124], 0// 000000002560: cc46406d 1a02f77f
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[117:118], v[121:122], v[0:7]// 000000002568: cc464000 1c02f375
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[117:118], v[125:126], v[93:100]// 000000002570: cc46405d 1d76fb75
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002578: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[129:130], v[121:122], v[101:108]// 00000000257c: cc464065 1d96f381
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[129:130], v[125:126], v[109:116]// 000000002584: cc46406d 1db6fb81
	s_wait_loadcnt 0xf                                         // 00000000258c: bfc0000f
	v_dual_mul_f32 v117, v131, v132 :: v_dual_mul_f32 v118, v132, v133// 000000002590: c8c70983 75770b84
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002598: bf870091
	v_dual_mul_f32 v0, v0, v117 :: v_dual_mul_f32 v1, v1, v118 // 00000000259c: c8c6eb00 0000ed01
	v_add_f32_e32 v18, v18, v0                                 // 0000000025a4: 06240112
	s_wait_loadcnt 0xe                                         // 0000000025a8: bfc0000e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_2)// 0000000025ac: bf870142
	v_dual_add_f32 v78, v78, v1 :: v_dual_mul_f32 v119, v132, v134// 0000000025b0: c906034e 4e770d84
	s_wait_loadcnt 0xd                                         // 0000000025b8: bfc0000d
	v_mul_f32_e32 v120, v132, v135                             // 0000000025bc: 10f10f84
	s_wait_loadcnt 0xc                                         // 0000000025c0: bfc0000c
	v_dual_mul_f32 v2, v2, v119 :: v_dual_mul_f32 v121, v132, v136// 0000000025c4: c8c6ef02 02791184
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000025cc: bf870122
	v_mul_f32_e32 v3, v3, v120                                 // 0000000025d0: 1006f103
	s_wait_loadcnt 0xb                                         // 0000000025d4: bfc0000b
	v_dual_add_f32 v75, v75, v2 :: v_dual_mul_f32 v122, v132, v137// 0000000025d8: c906054b 4b7b1384
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000025e0: bf870193
	v_mul_f32_e32 v4, v4, v121                                 // 0000000025e4: 1008f304
	v_add_f32_e32 v72, v72, v3                                 // 0000000025e8: 06900748
	s_wait_loadcnt 0xa                                         // 0000000025ec: bfc0000a
	v_mul_f32_e32 v123, v132, v138                             // 0000000025f0: 10f71584
	v_mul_f32_e32 v5, v5, v122                                 // 0000000025f4: 100af505
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000025f8: bf870112
	v_dual_add_f32 v69, v69, v4 :: v_dual_mul_f32 v6, v6, v123 // 0000000025fc: c9060945 4506f706
	v_add_f32_e32 v66, v66, v5                                 // 000000002604: 06840b42
	s_wait_loadcnt 0x8                                         // 000000002608: bfc00008
	v_dual_mul_f32 v124, v132, v139 :: v_dual_mul_f32 v125, v131, v140// 00000000260c: c8c71784 7c7d1983
	v_dual_mul_f32 v126, v133, v140 :: v_dual_mul_f32 v127, v134, v140// 000000002614: c8c71985 7e7f1986
	v_dual_mul_f32 v128, v135, v140 :: v_dual_mul_f32 v129, v136, v140// 00000000261c: c8c71987 80811988
	v_dual_mul_f32 v130, v137, v140 :: v_dual_mul_f32 v131, v138, v140// 000000002624: c8c71989 8283198a
	s_wait_loadcnt 0x7                                         // 00000000262c: bfc00007
	v_dual_mul_f32 v133, v139, v140 :: v_dual_mul_f32 v134, v132, v141// 000000002630: c8c7198b 85871b84
	v_mul_f32_e32 v141, v140, v141                             // 000000002638: 111b1b8c
	s_wait_loadcnt 0x6                                         // 00000000263c: bfc00006
	v_mul_f32_e32 v135, v132, v142                             // 000000002640: 110f1d84
	v_dual_mul_f32 v142, v140, v142 :: v_dual_mul_f32 v7, v7, v124// 000000002644: c8c71d8c 8e06f907
	v_dual_mul_f32 v93, v93, v125 :: v_dual_mul_f32 v94, v94, v126// 00000000264c: c8c6fb5d 5d5efd5e
	s_wait_loadcnt 0x5                                         // 000000002654: bfc00005
	v_mul_f32_e32 v136, v132, v143                             // 000000002658: 11111f84
	v_mul_f32_e32 v143, v140, v143                             // 00000000265c: 111f1f8c
	v_dual_mul_f32 v95, v95, v127 :: v_dual_mul_f32 v96, v96, v128// 000000002660: c8c6ff5f 5f610160
	v_dual_mul_f32 v97, v97, v129 :: v_dual_mul_f32 v98, v98, v130// 000000002668: c8c70361 61630562
	v_mul_f32_e32 v99, v99, v131                               // 000000002670: 10c70763
	s_wait_loadcnt 0x3                                         // 000000002674: bfc00003
	v_dual_mul_f32 v137, v132, v144 :: v_dual_mul_f32 v138, v132, v145// 000000002678: c8c72184 898b2384
	s_wait_loadcnt 0x2                                         // 000000002680: bfc00002
	v_mul_f32_e32 v139, v132, v146                             // 000000002684: 11172584
	s_wait_loadcnt 0x0                                         // 000000002688: bfc00000
	v_dual_mul_f32 v149, v132, v147 :: v_dual_mul_f32 v132, v132, v148// 00000000268c: c8c72784 95852984
	v_dual_mul_f32 v144, v140, v144 :: v_dual_mul_f32 v145, v140, v145// 000000002694: c8c7218c 9091238c
	v_dual_mul_f32 v146, v140, v146 :: v_dual_mul_f32 v147, v140, v147// 00000000269c: c8c7258c 9293278c
	v_mul_f32_e32 v140, v140, v148                             // 0000000026a4: 1119298c
	v_dual_mul_f32 v100, v100, v133 :: v_dual_mul_f32 v101, v101, v134// 0000000026a8: c8c70b64 64650d65
	v_dual_mul_f32 v102, v102, v135 :: v_dual_mul_f32 v103, v103, v136// 0000000026b0: c8c70f66 66671167
	v_dual_mul_f32 v104, v104, v137 :: v_dual_mul_f32 v105, v105, v138// 0000000026b8: c8c71368 68691569
	v_dual_mul_f32 v106, v106, v139 :: v_dual_mul_f32 v107, v107, v149// 0000000026c0: c8c7176a 6a6b2b6b
	v_dual_mul_f32 v108, v108, v132 :: v_dual_mul_f32 v109, v109, v141// 0000000026c8: c8c7096c 6c6d1b6d
	v_dual_mul_f32 v110, v110, v142 :: v_dual_mul_f32 v111, v111, v143// 0000000026d0: c8c71d6e 6e6f1f6f
	v_dual_mul_f32 v112, v112, v144 :: v_dual_mul_f32 v113, v113, v145// 0000000026d8: c8c72170 70712371
	v_dual_mul_f32 v114, v114, v146 :: v_dual_mul_f32 v115, v115, v147// 0000000026e0: c8c72572 72732773
	v_dual_mul_f32 v116, v116, v140 :: v_dual_add_f32 v61, v61, v6// 0000000026e8: c8c91974 743c0d3d
	v_dual_add_f32 v58, v58, v7 :: v_dual_add_f32 v39, v39, v93// 0000000026f0: c9080f3a 3a26bb27
	v_dual_add_f32 v38, v38, v94 :: v_dual_add_f32 v37, v37, v95// 0000000026f8: c908bd26 2624bf25
	v_dual_add_f32 v36, v36, v96 :: v_dual_add_f32 v35, v35, v97// 000000002700: c908c124 2422c323
	v_dual_add_f32 v34, v34, v98 :: v_dual_add_f32 v33, v33, v99// 000000002708: c908c522 2220c721
	v_dual_add_f32 v32, v32, v100 :: v_dual_add_f32 v53, v53, v101// 000000002710: c908c920 2034cb35
	v_dual_add_f32 v50, v50, v102 :: v_dual_add_f32 v49, v49, v103// 000000002718: c908cd32 3230cf31
	v_dual_add_f32 v48, v48, v104 :: v_dual_add_f32 v47, v47, v105// 000000002720: c908d130 302ed32f
	v_dual_add_f32 v46, v46, v106 :: v_dual_add_f32 v41, v41, v107// 000000002728: c908d52e 2e28d729
	v_dual_add_f32 v40, v40, v108 :: v_dual_add_f32 v31, v31, v109// 000000002730: c908d928 281edb1f
	v_dual_add_f32 v30, v30, v110 :: v_dual_add_f32 v29, v29, v111// 000000002738: c908dd1e 1e1cdf1d
	v_dual_add_f32 v28, v28, v112 :: v_dual_add_f32 v27, v27, v113// 000000002740: c908e11c 1c1ae31b
	v_dual_add_f32 v26, v26, v114 :: v_dual_add_f32 v25, v25, v115// 000000002748: c908e51a 1a18e719
	v_add_f32_e32 v24, v24, v116                               // 000000002750: 0630e918
	s_cbranch_scc0 37                                          // 000000002754: bfa10025 <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0xcec>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002758: bf88ff9e
	s_lshl_b64 s[12:13], s[10:11], 5                           // 00000000275c: 848c850a
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0               // 000000002760: ca100080 02020080
	s_wait_alu depctr_sa_sdst(0)                               // 000000002768: bf88ff9e
	v_add_co_u32 v0, s4, v20, s12                              // 00000000276c: d7000400 02001914
	s_wait_alu depctr_va_sdst(0)                               // 000000002774: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v21, s4                 // 000000002778: d5207c01 00122a0d
	global_load_b128 v[4:7], v[0:1], off                       // 000000002780: ee05c07c 00000004 00000000
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 00000000278c: ca100080 00000080
	s_and_saveexec_b32 s5, s3                                  // 000000002794: be852003
	s_cbranch_execz 8                                          // 000000002798: bfa50008 <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0xcbc>
	v_add_co_u32 v0, s4, v42, s12                              // 00000000279c: d7000400 0200192a
	s_wait_alu depctr_va_sdst(0)                               // 0000000027a4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v43, s4                 // 0000000027a8: d5207c01 0012560d
	global_load_b128 v[0:3], v[0:1], off                       // 0000000027b0: ee05c07c 00000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000027c0: 8c7e057e
	s_barrier_signal -1                                        // 0000000027c4: be804ec1
	s_barrier_wait 0xffff                                      // 0000000027c8: bf94ffff
	s_wait_loadcnt 0x0                                         // 0000000027cc: bfc00000
	ds_store_b128 v19, v[4:7]                                  // 0000000027d0: db7c0000 00000413
	s_and_saveexec_b32 s4, s3                                  // 0000000027d8: be842003
	s_cbranch_execz 65200                                      // 0000000027dc: bfa5feb0 <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x7a0>
	ds_store_b128 v19, v[0:3] offset:6144                      // 0000000027e0: db7c1800 00000013
	s_branch 65197                                             // 0000000027e8: bfa0fead <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x7a0>
	s_load_b64 s[24:25], s[0:1], 0xa8                          // 0000000027ec: f4002600 f80000a8
	v_mul_lo_u32 v4, s23, v12                                  // 0000000027f4: d72c0004 02021817
	v_mul_lo_u32 v5, s22, v13                                  // 0000000027fc: d72c0005 02021a16
	v_mad_co_u64_u32 v[2:3], null, s22, v12, 0                 // 000000002804: d6fe7c02 02021816
	v_sub_co_u32 v0, s0, s20, v12                              // 00000000280c: d7010000 02021814
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002814: bf870191
	v_sub_co_ci_u32_e64 v1, null, s21, v13, s0                 // 000000002818: d5217c01 00021a15
	v_add3_u32 v3, v3, v5, v4                                  // 000000002820: d6550003 04120b03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002828: bf8701a2
	v_cmp_lt_i64_e64 s15, 0, v[0:1]                            // 00000000282c: d451000f 02020080
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 000000002834: 3e081081
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000002838: 3e040481
	s_and_b32 s0, s15, s2                                      // 00000000283c: 8b00020f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002840: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002844: be812000
	s_cbranch_execz 28                                         // 000000002848: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0xdbc>
	v_bfe_u32 v6, v18, 16, 1                                   // 00000000284c: d6100006 02052112
	s_wait_kmcnt 0x0                                           // 000000002854: bfc70000
	v_add_co_u32 v7, s0, s24, v2                               // 000000002858: d7000007 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002860: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s25, v3, s0                 // 000000002864: d5207c0c 00020619
	v_add3_u32 v13, v6, v18, 0x7fff                            // 00000000286c: d655000d 03fe2506 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002878: bf870003
	v_add_co_u32 v6, s0, v7, v4                                // 00000000287c: d7000006 02020907
	v_or_b32_e32 v14, 0x400000, v18                            // 000000002884: 381c24ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000288c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v12, v5, s0                  // 000000002890: d5207c07 00020b0c
	v_cmp_u_f32_e64 s0, v18, v18                               // 000000002898: d4180000 02022512
	s_wait_alu depctr_va_sdst(0)                               // 0000000028a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000028a4: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 0000000028a8: d501000c 00021d0d
	global_store_d16_hi_b16 v[6:7], v12, off                   // 0000000028b0: ee09407c 06000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000028c0: 8c7e017e
	v_add_co_u32 v6, s0, s22, v8                               // 0000000028c4: d7000006 02021016
	s_wait_alu depctr_va_sdst(0)                               // 0000000028cc: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s23, v9, s0                  // 0000000028d0: d5207c07 00021217
	v_cmp_lt_i64_e64 s16, 1, v[0:1]                            // 0000000028d8: d4510010 02020081
	s_delay_alu instid0(valu_dep_2)                            // 0000000028e0: bf870002
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 0000000028e4: 3e0c0c81
	s_and_b32 s0, s16, s2                                      // 0000000028e8: 8b000210
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028ec: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000028f0: be812000
	s_cbranch_execz 28                                         // 0000000028f4: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0xe68>
	v_bfe_u32 v12, v78, 16, 1                                  // 0000000028f8: d610000c 0205214e
	s_wait_kmcnt 0x0                                           // 000000002900: bfc70000
	v_add_co_u32 v13, s0, s24, v2                              // 000000002904: d700000d 02020418
	s_wait_alu depctr_va_sdst(0)                               // 00000000290c: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s25, v3, s0                 // 000000002910: d5207c0e 00020619
	v_add3_u32 v15, v12, v78, 0x7fff                           // 000000002918: d655000f 03fe9d0c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002924: bf870003
	v_add_co_u32 v12, s0, v13, v6                              // 000000002928: d700000c 02020d0d
	v_or_b32_e32 v16, 0x400000, v78                            // 000000002930: 38209cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002938: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v7, s0                 // 00000000293c: d5207c0d 00020f0e
	v_cmp_u_f32_e64 s0, v78, v78                               // 000000002944: d4180000 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 00000000294c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002950: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 000000002954: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 00000000295c: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002968: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000296c: 8c7e017e
	s_lshl_b64 s[38:39], s[22:23], 1                           // 000000002970: 84a68116
	v_cmp_lt_i64_e64 s14, 2, v[0:1]                            // 000000002974: d451000e 02020082
	v_add_co_u32 v12, s0, s38, v8                              // 00000000297c: d700000c 02021026
	s_wait_alu depctr_va_sdst(0)                               // 000000002984: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s39, v9, s0                 // 000000002988: d5207c0d 00021227
	s_and_b32 s0, s14, s2                                      // 000000002990: 8b00020e
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000002994: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002998: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000299c: be812000
	s_cbranch_execz 28                                         // 0000000029a0: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0xf14>
	v_bfe_u32 v14, v75, 16, 1                                  // 0000000029a4: d610000e 0205214b
	s_wait_kmcnt 0x0                                           // 0000000029ac: bfc70000
	v_add_co_u32 v15, s0, s24, v2                              // 0000000029b0: d700000f 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000029b8: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s25, v3, s0                 // 0000000029bc: d5207c10 00020619
	v_add3_u32 v17, v14, v75, 0x7fff                           // 0000000029c4: d6550011 03fe970e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000029d0: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 0000000029d4: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v75                            // 0000000029dc: 382496ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000029e4: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 0000000029e8: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v75, v75                               // 0000000029f0: d4180000 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 0000000029f8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000029fc: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 000000002a00: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 000000002a08: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002a18: 8c7e017e
	s_mul_u64 s[36:37], s[22:23], 3                            // 000000002a1c: aaa48316
	v_cmp_lt_i64_e64 s13, 3, v[0:1]                            // 000000002a20: d451000d 02020083
	v_add_co_u32 v14, s0, s36, v8                              // 000000002a28: d700000e 02021024
	s_wait_alu depctr_va_sdst(0)                               // 000000002a30: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s37, v9, s0                 // 000000002a34: d5207c0f 00021225
	s_and_b32 s0, s13, s2                                      // 000000002a3c: 8b00020d
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000002a40: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a44: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002a48: be812000
	s_cbranch_execz 28                                         // 000000002a4c: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0xfc0>
	v_bfe_u32 v16, v72, 16, 1                                  // 000000002a50: d6100010 02052148
	s_wait_kmcnt 0x0                                           // 000000002a58: bfc70000
	v_add_co_u32 v17, s0, s24, v2                              // 000000002a5c: d7000011 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002a64: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s25, v3, s0                 // 000000002a68: d5207c12 00020619
	v_add3_u32 v19, v16, v72, 0x7fff                           // 000000002a70: d6550013 03fe9110 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002a7c: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 000000002a80: d7000010 02021d11
	v_or_b32_e32 v20, 0x400000, v72                            // 000000002a88: 382890ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002a90: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 000000002a94: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v72, v72                               // 000000002a9c: d4180000 02029148
	s_wait_alu depctr_va_sdst(0)                               // 000000002aa4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002aa8: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 000000002aac: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000002ab4: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ac0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ac4: 8c7e017e
	s_lshl_b64 s[34:35], s[22:23], 2                           // 000000002ac8: 84a28216
	v_cmp_lt_i64_e64 s12, 4, v[0:1]                            // 000000002acc: d451000c 02020084
	v_add_co_u32 v16, s0, s34, v8                              // 000000002ad4: d7000010 02021022
	s_wait_alu depctr_va_sdst(0)                               // 000000002adc: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s35, v9, s0                 // 000000002ae0: d5207c11 00021223
	s_and_b32 s0, s12, s2                                      // 000000002ae8: 8b00020c
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000002aec: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002af0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002af4: be812000
	s_cbranch_execz 28                                         // 000000002af8: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x106c>
	v_bfe_u32 v18, v69, 16, 1                                  // 000000002afc: d6100012 02052145
	s_wait_kmcnt 0x0                                           // 000000002b04: bfc70000
	v_add_co_u32 v19, s0, s24, v2                              // 000000002b08: d7000013 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002b10: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s25, v3, s0                 // 000000002b14: d5207c14 00020619
	v_add3_u32 v21, v18, v69, 0x7fff                           // 000000002b1c: d6550015 03fe8b12 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002b28: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 000000002b2c: d7000012 02022113
	v_or_b32_e32 v42, 0x400000, v69                            // 000000002b34: 38548aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002b3c: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v17, s0                // 000000002b40: d5207c13 00022314
	v_cmp_u_f32_e64 s0, v69, v69                               // 000000002b48: d4180000 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 000000002b50: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002b54: bf870001
	v_cndmask_b32_e64 v20, v21, v42, s0                        // 000000002b58: d5010014 00025515
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000002b60: ee09407c 0a000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002b70: 8c7e017e
	s_mul_u64 s[30:31], s[22:23], 5                            // 000000002b74: aa9e8516
	v_cmp_lt_i64_e64 s11, 5, v[0:1]                            // 000000002b78: d451000b 02020085
	v_add_co_u32 v18, s0, s30, v8                              // 000000002b80: d7000012 0202101e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b88: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s31, v9, s0                 // 000000002b8c: d5207c13 0002121f
	s_and_b32 s0, s11, s2                                      // 000000002b94: 8b00020b
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 000000002b98: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b9c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002ba0: be812000
	s_cbranch_execz 28                                         // 000000002ba4: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1118>
	v_bfe_u32 v20, v66, 16, 1                                  // 000000002ba8: d6100014 02052142
	s_wait_kmcnt 0x0                                           // 000000002bb0: bfc70000
	v_add_co_u32 v21, s0, s24, v2                              // 000000002bb4: d7000015 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002bbc: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s25, v3, s0                 // 000000002bc0: d5207c2a 00020619
	v_add3_u32 v43, v20, v66, 0x7fff                           // 000000002bc8: d655002b 03fe8514 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002bd4: bf870003
	v_add_co_u32 v20, s0, v21, v18                             // 000000002bd8: d7000014 02022515
	v_or_b32_e32 v44, 0x400000, v66                            // 000000002be0: 385884ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002be8: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v42, v19, s0                // 000000002bec: d5207c15 0002272a
	v_cmp_u_f32_e64 s0, v66, v66                               // 000000002bf4: d4180000 02028542
	s_wait_alu depctr_va_sdst(0)                               // 000000002bfc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002c00: bf870001
	v_cndmask_b32_e64 v42, v43, v44, s0                        // 000000002c04: d501002a 0002592b
	global_store_d16_hi_b16 v[20:21], v42, off                 // 000000002c0c: ee09407c 15000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c18: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002c1c: 8c7e017e
	s_mul_u64 s[28:29], s[22:23], 6                            // 000000002c20: aa9c8616
	v_cmp_lt_i64_e64 s9, 6, v[0:1]                             // 000000002c24: d4510009 02020086
	v_add_co_u32 v20, s0, s28, v8                              // 000000002c2c: d7000014 0202101c
	s_wait_alu depctr_va_sdst(0)                               // 000000002c34: bf88f19f
	v_add_co_ci_u32_e64 v21, null, s29, v9, s0                 // 000000002c38: d5207c15 0002121d
	s_and_b32 s0, s9, s2                                       // 000000002c40: 8b000209
	v_lshlrev_b64_e32 v[20:21], 1, v[20:21]                    // 000000002c44: 3e282881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c48: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002c4c: be812000
	s_cbranch_execz 28                                         // 000000002c50: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x11c4>
	v_bfe_u32 v42, v61, 16, 1                                  // 000000002c54: d610002a 0205213d
	s_wait_kmcnt 0x0                                           // 000000002c5c: bfc70000
	v_add_co_u32 v43, s0, s24, v2                              // 000000002c60: d700002b 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002c68: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s25, v3, s0                 // 000000002c6c: d5207c2c 00020619
	v_add3_u32 v45, v42, v61, 0x7fff                           // 000000002c74: d655002d 03fe7b2a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c80: bf870003
	v_add_co_u32 v42, s0, v43, v20                             // 000000002c84: d700002a 0202292b
	v_or_b32_e32 v51, 0x400000, v61                            // 000000002c8c: 38667aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002c94: bf88f19f
	v_add_co_ci_u32_e64 v43, null, v44, v21, s0                // 000000002c98: d5207c2b 00022b2c
	v_cmp_u_f32_e64 s0, v61, v61                               // 000000002ca0: d4180000 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000002ca8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002cac: bf870001
	v_cndmask_b32_e64 v44, v45, v51, s0                        // 000000002cb0: d501002c 0002672d
	global_store_d16_hi_b16 v[42:43], v44, off                 // 000000002cb8: ee09407c 16000000 0000002a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cc4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002cc8: 8c7e017e
	s_mul_u64 s[26:27], s[22:23], 7                            // 000000002ccc: aa9a8716
	v_cmp_lt_i64_e64 s8, 7, v[0:1]                             // 000000002cd0: d4510008 02020087
	v_add_co_u32 v8, s0, s26, v8                               // 000000002cd8: d7000008 0202101a
	s_wait_alu depctr_va_sdst(0)                               // 000000002ce0: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s27, v9, s0                  // 000000002ce4: d5207c09 0002121b
	s_and_b32 s0, s8, s2                                       // 000000002cec: 8b000208
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000002cf0: 3e101081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cf4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002cf8: be812000
	s_cbranch_execz 28                                         // 000000002cfc: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1270>
	v_bfe_u32 v0, v58, 16, 1                                   // 000000002d00: d6100000 0205213a
	s_wait_kmcnt 0x0                                           // 000000002d08: bfc70000
	v_add_co_u32 v1, s0, s24, v2                               // 000000002d0c: d7000001 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002d14: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s25, v3, s0                 // 000000002d18: d5207c2a 00020619
	v_add3_u32 v43, v0, v58, 0x7fff                            // 000000002d20: d655002b 03fe7500 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d2c: bf870003
	v_add_co_u32 v0, s0, v1, v8                                // 000000002d30: d7000000 02021101
	v_or_b32_e32 v44, 0x400000, v58                            // 000000002d38: 385874ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d40: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v42, v9, s0                  // 000000002d44: d5207c01 0002132a
	v_cmp_u_f32_e64 s0, v58, v58                               // 000000002d4c: d4180000 0202753a
	s_wait_alu depctr_va_sdst(0)                               // 000000002d54: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d58: bf870001
	v_cndmask_b32_e64 v42, v43, v44, s0                        // 000000002d5c: d501002a 0002592b
	global_store_d16_hi_b16 v[0:1], v42, off                   // 000000002d64: ee09407c 15000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002d74: 8c7e017e
	v_mul_lo_u32 v42, s23, v10                                 // 000000002d78: d72c002a 02021417
	v_mul_lo_u32 v43, s22, v11                                 // 000000002d80: d72c002b 02021616
	v_mad_co_u64_u32 v[0:1], null, s22, v10, 0                 // 000000002d88: d6fe7c00 02021416
	v_sub_co_u32 v10, s0, s20, v10                             // 000000002d90: d701000a 02021414
	s_wait_alu depctr_va_sdst(0)                               // 000000002d98: bf88f19f
	v_sub_co_ci_u32_e64 v11, null, s21, v11, s0                // 000000002d9c: d5217c0b 00021615
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000002da4: bf870211
	v_cmp_lt_i64_e64 s10, 0, v[10:11]                          // 000000002da8: d451000a 02021480
	v_add3_u32 v1, v1, v43, v42                                // 000000002db0: d6550001 04aa5701
	s_delay_alu instid0(valu_dep_1)                            // 000000002db8: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000002dbc: 3e000081
	s_and_b32 s0, s10, s2                                      // 000000002dc0: 8b00020a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dc4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002dc8: be812000
	s_cbranch_execz 28                                         // 000000002dcc: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1340>
	s_wait_kmcnt 0x0                                           // 000000002dd0: bfc70000
	v_add_co_u32 v43, s0, s24, v0                              // 000000002dd4: d700002b 02020018
	v_bfe_u32 v42, v53, 16, 1                                  // 000000002ddc: d610002a 02052135
	s_wait_alu depctr_va_sdst(0)                               // 000000002de4: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s25, v1, s0                 // 000000002de8: d5207c2c 00020219
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002df0: bf870193
	v_add_co_u32 v4, s0, v43, v4                               // 000000002df4: d7000004 0202092b
	v_add3_u32 v42, v42, v53, 0x7fff                           // 000000002dfc: d655002a 03fe6b2a 00007fff
	v_or_b32_e32 v45, 0x400000, v53                            // 000000002e08: 385a6aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e10: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v44, v5, s0                  // 000000002e14: d5207c05 00020b2c
	v_cmp_u_f32_e64 s0, v53, v53                               // 000000002e1c: d4180000 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 000000002e24: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e28: bf870001
	v_cndmask_b32_e64 v42, v42, v45, s0                        // 000000002e2c: d501002a 00025b2a
	global_store_d16_hi_b16 v[4:5], v42, off                   // 000000002e34: ee09407c 15000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e40: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002e44: 8c7e017e
	v_cmp_lt_i64_e64 s7, 1, v[10:11]                           // 000000002e48: d4510007 02021481
	s_and_b32 s0, s7, s2                                       // 000000002e50: 8b000207
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e54: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002e58: be812000
	s_cbranch_execz 28                                         // 000000002e5c: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x13d0>
	v_bfe_u32 v4, v50, 16, 1                                   // 000000002e60: d6100004 02052132
	s_wait_kmcnt 0x0                                           // 000000002e68: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 000000002e6c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000002e74: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s25, v1, s0                 // 000000002e78: d5207c2a 00020219
	v_add3_u32 v43, v4, v50, 0x7fff                            // 000000002e80: d655002b 03fe6504 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e8c: bf870003
	v_add_co_u32 v4, s0, v5, v6                                // 000000002e90: d7000004 02020d05
	v_or_b32_e32 v44, 0x400000, v50                            // 000000002e98: 385864ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002ea0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v42, v7, s0                  // 000000002ea4: d5207c05 00020f2a
	v_cmp_u_f32_e64 s0, v50, v50                               // 000000002eac: d4180000 02026532
	s_wait_alu depctr_va_sdst(0)                               // 000000002eb4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002eb8: bf870001
	v_cndmask_b32_e64 v6, v43, v44, s0                         // 000000002ebc: d5010006 0002592b
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000002ec4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ed0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ed4: 8c7e017e
	v_cmp_lt_i64_e64 s6, 2, v[10:11]                           // 000000002ed8: d4510006 02021482
	s_and_b32 s0, s6, s2                                       // 000000002ee0: 8b000206
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ee4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002ee8: be812000
	s_cbranch_execz 28                                         // 000000002eec: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1460>
	v_bfe_u32 v4, v49, 16, 1                                   // 000000002ef0: d6100004 02052131
	s_wait_kmcnt 0x0                                           // 000000002ef8: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 000000002efc: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000002f04: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 000000002f08: d5207c06 00020219
	v_add3_u32 v7, v4, v49, 0x7fff                             // 000000002f10: d6550007 03fe6304 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f1c: bf870003
	v_add_co_u32 v4, s0, v5, v12                               // 000000002f20: d7000004 02021905
	v_or_b32_e32 v42, 0x400000, v49                            // 000000002f28: 385462ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f30: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v13, s0                  // 000000002f34: d5207c05 00021b06
	v_cmp_u_f32_e64 s0, v49, v49                               // 000000002f3c: d4180000 02026331
	s_wait_alu depctr_va_sdst(0)                               // 000000002f44: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f48: bf870001
	v_cndmask_b32_e64 v6, v7, v42, s0                          // 000000002f4c: d5010006 00025507
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000002f54: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f60: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002f64: 8c7e017e
	v_cmp_lt_i64_e64 s5, 3, v[10:11]                           // 000000002f68: d4510005 02021483
	s_and_b32 s0, s5, s2                                       // 000000002f70: 8b000205
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f74: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f78: be812000
	s_cbranch_execz 28                                         // 000000002f7c: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x14f0>
	v_bfe_u32 v4, v48, 16, 1                                   // 000000002f80: d6100004 02052130
	s_wait_kmcnt 0x0                                           // 000000002f88: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 000000002f8c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000002f94: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 000000002f98: d5207c06 00020219
	v_add3_u32 v7, v4, v48, 0x7fff                             // 000000002fa0: d6550007 03fe6104 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002fac: bf870003
	v_add_co_u32 v4, s0, v5, v14                               // 000000002fb0: d7000004 02021d05
	v_or_b32_e32 v12, 0x400000, v48                            // 000000002fb8: 381860ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002fc0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v15, s0                  // 000000002fc4: d5207c05 00021f06
	v_cmp_u_f32_e64 s0, v48, v48                               // 000000002fcc: d4180000 02026130
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002fd8: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 000000002fdc: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000002fe4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ff0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ff4: 8c7e017e
	v_cmp_lt_i64_e64 s4, 4, v[10:11]                           // 000000002ff8: d4510004 02021484
	s_and_b32 s0, s4, s2                                       // 000000003000: 8b000204
	s_wait_alu depctr_sa_sdst(0)                               // 000000003004: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003008: be812000
	s_cbranch_execz 28                                         // 00000000300c: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1580>
	v_bfe_u32 v4, v47, 16, 1                                   // 000000003010: d6100004 0205212f
	s_wait_kmcnt 0x0                                           // 000000003018: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 00000000301c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003024: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 000000003028: d5207c06 00020219
	v_add3_u32 v7, v4, v47, 0x7fff                             // 000000003030: d6550007 03fe5f04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000303c: bf870003
	v_add_co_u32 v4, s0, v5, v16                               // 000000003040: d7000004 02022105
	v_or_b32_e32 v12, 0x400000, v47                            // 000000003048: 38185eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003050: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v17, s0                  // 000000003054: d5207c05 00022306
	v_cmp_u_f32_e64 s0, v47, v47                               // 00000000305c: d4180000 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003064: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003068: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 00000000306c: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003074: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003080: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003084: 8c7e017e
	v_cmp_lt_i64_e64 s3, 5, v[10:11]                           // 000000003088: d4510003 02021485
	s_and_b32 s0, s3, s2                                       // 000000003090: 8b000203
	s_wait_alu depctr_sa_sdst(0)                               // 000000003094: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003098: be812000
	s_cbranch_execz 28                                         // 00000000309c: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1610>
	v_bfe_u32 v4, v46, 16, 1                                   // 0000000030a0: d6100004 0205212e
	s_wait_kmcnt 0x0                                           // 0000000030a8: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 0000000030ac: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000030b4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 0000000030b8: d5207c06 00020219
	v_add3_u32 v7, v4, v46, 0x7fff                             // 0000000030c0: d6550007 03fe5d04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000030cc: bf870003
	v_add_co_u32 v4, s0, v5, v18                               // 0000000030d0: d7000004 02022505
	v_or_b32_e32 v12, 0x400000, v46                            // 0000000030d8: 38185cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030e0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v19, s0                  // 0000000030e4: d5207c05 00022706
	v_cmp_u_f32_e64 s0, v46, v46                               // 0000000030ec: d4180000 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000030f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000030f8: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 0000000030fc: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003104: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003110: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003114: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[10:11]                           // 000000003118: d4510001 02021486
	s_and_b32 s0, s1, s2                                       // 000000003120: 8b000201
	s_wait_alu depctr_sa_sdst(0)                               // 000000003124: bf88ff9e
	s_and_saveexec_b32 s17, s0                                 // 000000003128: be912000
	s_cbranch_execz 28                                         // 00000000312c: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x16a0>
	v_bfe_u32 v4, v41, 16, 1                                   // 000000003130: d6100004 02052129
	s_wait_kmcnt 0x0                                           // 000000003138: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 00000000313c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003144: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 000000003148: d5207c06 00020219
	v_add3_u32 v7, v4, v41, 0x7fff                             // 000000003150: d6550007 03fe5304 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000315c: bf870003
	v_add_co_u32 v4, s0, v5, v20                               // 000000003160: d7000004 02022905
	v_or_b32_e32 v12, 0x400000, v41                            // 000000003168: 381852ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003170: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v21, s0                  // 000000003174: d5207c05 00022b06
	v_cmp_u_f32_e64 s0, v41, v41                               // 00000000317c: d4180000 02025329
	s_wait_alu depctr_va_sdst(0)                               // 000000003184: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003188: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 00000000318c: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003194: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031a0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 0000000031a4: 8c7e117e
	v_cmp_lt_i64_e64 s0, 7, v[10:11]                           // 0000000031a8: d4510000 02021487
	s_and_b32 s2, s0, s2                                       // 0000000031b0: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031b4: bf88ff9e
	s_and_saveexec_b32 s17, s2                                 // 0000000031b8: be912002
	s_cbranch_execz 28                                         // 0000000031bc: bfa5001c <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1730>
	v_bfe_u32 v4, v40, 16, 1                                   // 0000000031c0: d6100004 02052128
	s_wait_kmcnt 0x0                                           // 0000000031c8: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 0000000031cc: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000031d4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 0000000031d8: d5207c06 000a0219
	v_add3_u32 v7, v4, v40, 0x7fff                             // 0000000031e0: d6550007 03fe5104 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000031ec: bf870003
	v_add_co_u32 v4, s2, v5, v8                                // 0000000031f0: d7000204 02021105
	v_or_b32_e32 v10, 0x400000, v40                            // 0000000031f8: 381450ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003200: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v9, s2                   // 000000003204: d5207c05 000a1306
	v_cmp_u_f32_e64 s2, v40, v40                               // 00000000320c: d4180002 02025128
	s_wait_alu depctr_va_sdst(0)                               // 000000003214: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003218: bf870001
	v_cndmask_b32_e64 v6, v7, v10, s2                          // 00000000321c: d5010006 000a1507
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003224: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003230: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003234: 8c7e117e
	s_and_b32 s2, s15, vcc_lo                                  // 000000003238: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000323c: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003240: be8f2002
	s_cbranch_execz 40                                         // 000000003244: bfa50028 <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x17e8>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003248: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003250: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 000000003254: d5207c05 00082680
	v_bfe_u32 v6, v39, 16, 1                                   // 00000000325c: d6100006 02052127
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003264: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003268: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003270: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003274: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 00000000327c: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003280: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003288: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 00000000328c: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003294: 3e080881
	v_add3_u32 v6, v6, v39, 0x7fff                             // 000000003298: d6550006 03fe4f06 00007fff
	v_or_b32_e32 v9, 0x400000, v39                             // 0000000032a4: 38124eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000032ac: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 0000000032b0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000032b8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000032bc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v39, v39                               // 0000000032c4: d4180002 02024f27
	s_wait_alu depctr_va_sdst(0)                               // 0000000032cc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000032d0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000032d4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000032dc: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032e8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000032ec: 8c7e0f7e
	s_and_b32 s2, s16, vcc_lo                                  // 0000000032f0: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032f4: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 0000000032f8: be8f2002
	s_cbranch_execz 46                                         // 0000000032fc: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x18b8>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003300: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003308: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000330c: d5207c05 00082680
	v_bfe_u32 v6, v38, 16, 1                                   // 000000003314: d6100006 02052126
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000331c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003320: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003328: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000332c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v38                             // 000000003334: 38124cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000333c: bf8701a3
	v_add_co_u32 v4, s2, s22, v4                               // 000000003340: d7000204 02020816
	s_wait_alu depctr_va_sdst(0)                               // 000000003348: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s2                  // 00000000334c: d5207c05 000a0a17
	s_wait_kmcnt 0x0                                           // 000000003354: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003358: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003360: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003364: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000336c: 3e080881
	v_add3_u32 v6, v6, v38, 0x7fff                             // 000000003370: d6550006 03fe4d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000337c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003380: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003388: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000338c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v38, v38                               // 000000003394: d4180002 02024d26
	s_wait_alu depctr_va_sdst(0)                               // 00000000339c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033a0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000033a4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000033ac: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000033bc: 8c7e0f7e
	s_and_b32 s2, s14, vcc_lo                                  // 0000000033c0: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033c4: bf88ff9e
	s_and_saveexec_b32 s14, s2                                 // 0000000033c8: be8e2002
	s_cbranch_execz 46                                         // 0000000033cc: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1988>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000033d0: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000033d8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 0000000033dc: d5207c05 00082680
	v_bfe_u32 v6, v37, 16, 1                                   // 0000000033e4: d6100006 02052125
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000033ec: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 0000000033f0: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000033f8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000033fc: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v37                             // 000000003404: 38124aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000340c: bf8701a3
	v_add_co_u32 v4, s2, s38, v4                               // 000000003410: d7000204 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000003418: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s2                  // 00000000341c: d5207c05 000a0a27
	s_wait_kmcnt 0x0                                           // 000000003424: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003428: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003430: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003434: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000343c: 3e080881
	v_add3_u32 v6, v6, v37, 0x7fff                             // 000000003440: d6550006 03fe4b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000344c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003450: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003458: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000345c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v37, v37                               // 000000003464: d4180002 02024b25
	s_wait_alu depctr_va_sdst(0)                               // 00000000346c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003470: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003474: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000347c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003488: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 00000000348c: 8c7e0e7e
	s_and_b32 s2, s13, vcc_lo                                  // 000000003490: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000003494: bf88ff9e
	s_and_saveexec_b32 s13, s2                                 // 000000003498: be8d2002
	s_cbranch_execz 46                                         // 00000000349c: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1a58>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000034a0: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000034a8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 0000000034ac: d5207c05 00082680
	v_bfe_u32 v6, v36, 16, 1                                   // 0000000034b4: d6100006 02052124
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000034bc: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 0000000034c0: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000034c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000034cc: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v36                             // 0000000034d4: 381248ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000034dc: bf8701a3
	v_add_co_u32 v4, s2, s36, v4                               // 0000000034e0: d7000204 02020824
	s_wait_alu depctr_va_sdst(0)                               // 0000000034e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s2                  // 0000000034ec: d5207c05 000a0a25
	s_wait_kmcnt 0x0                                           // 0000000034f4: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 0000000034f8: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003500: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003504: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000350c: 3e080881
	v_add3_u32 v6, v6, v36, 0x7fff                             // 000000003510: d6550006 03fe4906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000351c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003520: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003528: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000352c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v36, v36                               // 000000003534: d4180002 02024924
	s_wait_alu depctr_va_sdst(0)                               // 00000000353c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003540: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003544: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000354c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003558: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 00000000355c: 8c7e0d7e
	s_and_b32 s2, s12, vcc_lo                                  // 000000003560: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003564: bf88ff9e
	s_and_saveexec_b32 s12, s2                                 // 000000003568: be8c2002
	s_cbranch_execz 46                                         // 00000000356c: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1b28>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003570: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003578: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000357c: d5207c05 00082680
	v_bfe_u32 v6, v35, 16, 1                                   // 000000003584: d6100006 02052123
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000358c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003590: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003598: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000359c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v35                             // 0000000035a4: 381246ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035ac: bf8701a3
	v_add_co_u32 v4, s2, s34, v4                               // 0000000035b0: d7000204 02020822
	s_wait_alu depctr_va_sdst(0)                               // 0000000035b8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s2                  // 0000000035bc: d5207c05 000a0a23
	s_wait_kmcnt 0x0                                           // 0000000035c4: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 0000000035c8: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000035d0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 0000000035d4: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000035dc: 3e080881
	v_add3_u32 v6, v6, v35, 0x7fff                             // 0000000035e0: d6550006 03fe4706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035ec: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000035f0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000035f8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000035fc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v35, v35                               // 000000003604: d4180002 02024723
	s_wait_alu depctr_va_sdst(0)                               // 00000000360c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003610: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003614: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000361c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003628: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 00000000362c: 8c7e0c7e
	s_and_b32 s2, s11, vcc_lo                                  // 000000003630: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003634: bf88ff9e
	s_and_saveexec_b32 s11, s2                                 // 000000003638: be8b2002
	s_cbranch_execz 46                                         // 00000000363c: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1bf8>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003640: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003648: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000364c: d5207c05 00082680
	v_bfe_u32 v6, v34, 16, 1                                   // 000000003654: d6100006 02052122
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000365c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003660: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003668: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000366c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v34                             // 000000003674: 381244ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000367c: bf8701a3
	v_add_co_u32 v4, s2, s30, v4                               // 000000003680: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000003688: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 00000000368c: d5207c05 000a0a1f
	s_wait_kmcnt 0x0                                           // 000000003694: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003698: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000036a0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 0000000036a4: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000036ac: 3e080881
	v_add3_u32 v6, v6, v34, 0x7fff                             // 0000000036b0: d6550006 03fe4506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000036bc: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000036c0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000036c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000036cc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v34, v34                               // 0000000036d4: d4180002 02024522
	s_wait_alu depctr_va_sdst(0)                               // 0000000036dc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000036e0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000036e4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000036ec: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 0000000036fc: 8c7e0b7e
	s_and_b32 s2, s9, vcc_lo                                   // 000000003700: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000003704: bf88ff9e
	s_and_saveexec_b32 s9, s2                                  // 000000003708: be892002
	s_cbranch_execz 46                                         // 00000000370c: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1cc8>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003710: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003718: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000371c: d5207c05 00082680
	v_bfe_u32 v6, v33, 16, 1                                   // 000000003724: d6100006 02052121
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000372c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003730: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003738: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000373c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v33                             // 000000003744: 381242ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000374c: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 000000003750: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000003758: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 00000000375c: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 000000003764: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003768: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003770: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003774: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000377c: 3e080881
	v_add3_u32 v6, v6, v33, 0x7fff                             // 000000003780: d6550006 03fe4306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000378c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003790: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003798: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000379c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v33, v33                               // 0000000037a4: d4180002 02024321
	s_wait_alu depctr_va_sdst(0)                               // 0000000037ac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000037b0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000037b4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000037bc: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000037cc: 8c7e097e
	s_and_b32 s2, s8, vcc_lo                                   // 0000000037d0: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037d4: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 0000000037d8: be882002
	s_cbranch_execz 46                                         // 0000000037dc: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1d98>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000037e0: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000037e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 0000000037ec: d5207c05 00082680
	v_bfe_u32 v6, v32, 16, 1                                   // 0000000037f4: d6100006 02052120
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000037fc: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003800: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003808: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000380c: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v32                             // 000000003814: 380e40ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000381c: bf8701a3
	v_add_co_u32 v4, s2, s26, v4                               // 000000003820: d7000204 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000003828: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s2                  // 00000000382c: d5207c05 000a0a1b
	s_wait_kmcnt 0x0                                           // 000000003834: bfc70000
	v_add_co_u32 v2, s2, s24, v2                               // 000000003838: d7000202 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003840: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s25, v3, s2                  // 000000003844: d5207c03 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000384c: 3e080881
	v_add3_u32 v6, v6, v32, 0x7fff                             // 000000003850: d6550006 03fe4106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000385c: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 000000003860: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000003868: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 00000000386c: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v32, v32                               // 000000003874: d4180002 02024120
	s_wait_alu depctr_va_sdst(0)                               // 00000000387c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003880: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 000000003884: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 00000000388c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003898: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 00000000389c: 8c7e087e
	s_and_b32 s2, s10, vcc_lo                                  // 0000000038a0: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038a4: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 0000000038a8: be882002
	s_cbranch_execz 40                                         // 0000000038ac: bfa50028 <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1e50>
	v_add_co_u32 v2, s2, v23, s18                              // 0000000038b0: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000038b8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 0000000038bc: d5207c03 00082680
	v_bfe_u32 v4, v31, 16, 1                                   // 0000000038c4: d6100004 0205211f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000038cc: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 0000000038d0: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000038d8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000038dc: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 0000000038e4: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 0000000038e8: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000038f0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 0000000038f4: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000038fc: 3e040481
	v_add3_u32 v4, v4, v31, 0x7fff                             // 000000003900: d6550004 03fe3f04 00007fff
	v_or_b32_e32 v7, 0x400000, v31                             // 00000000390c: 380e3eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003914: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 000000003918: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003920: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003924: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v31, v31                               // 00000000392c: d4180002 02023f1f
	s_wait_alu depctr_va_sdst(0)                               // 000000003934: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003938: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 00000000393c: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003944: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003950: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003954: 8c7e087e
	s_and_b32 s2, s7, vcc_lo                                   // 000000003958: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 00000000395c: bf88ff9e
	s_and_saveexec_b32 s7, s2                                  // 000000003960: be872002
	s_cbranch_execz 46                                         // 000000003964: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1f20>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003968: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003970: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003974: d5207c03 00082680
	v_bfe_u32 v4, v30, 16, 1                                   // 00000000397c: d6100004 0205211e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003984: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003988: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003990: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003994: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v30                             // 00000000399c: 380e3cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039a4: bf8701a3
	v_add_co_u32 v2, s2, s22, v2                               // 0000000039a8: d7000202 02020416
	s_wait_alu depctr_va_sdst(0)                               // 0000000039b0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s2                  // 0000000039b4: d5207c03 000a0617
	s_wait_kmcnt 0x0                                           // 0000000039bc: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 0000000039c0: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000039c8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 0000000039cc: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000039d4: 3e040481
	v_add3_u32 v4, v4, v30, 0x7fff                             // 0000000039d8: d6550004 03fe3d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039e4: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 0000000039e8: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 0000000039f0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 0000000039f4: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v30, v30                               // 0000000039fc: d4180002 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003a04: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a08: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003a0c: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003a14: ee09407c 02000000 00002002
	s_or_b32 exec_lo, exec_lo, s7                              // 000000003a20: 8c7e077e
	s_and_b32 s2, s6, vcc_lo                                   // 000000003a24: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a28: bf88ff9e
	s_and_saveexec_b32 s6, s2                                  // 000000003a2c: be862002
	s_cbranch_execz 46                                         // 000000003a30: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x1fec>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003a34: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003a3c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003a40: d5207c03 00082680
	v_bfe_u32 v4, v29, 16, 1                                   // 000000003a48: d6100004 0205211d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a50: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003a54: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003a5c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003a60: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v29                             // 000000003a68: 380e3aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a70: bf8701a3
	v_add_co_u32 v2, s2, s38, v2                               // 000000003a74: d7000202 02020426
	s_wait_alu depctr_va_sdst(0)                               // 000000003a7c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s39, v3, s2                  // 000000003a80: d5207c03 000a0627
	s_wait_kmcnt 0x0                                           // 000000003a88: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003a8c: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003a94: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003a98: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003aa0: 3e040481
	v_add3_u32 v4, v4, v29, 0x7fff                             // 000000003aa4: d6550004 03fe3b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ab0: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003ab4: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003abc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003ac0: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v29, v29                               // 000000003ac8: d4180002 02023b1d
	s_wait_alu depctr_va_sdst(0)                               // 000000003ad0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ad4: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003ad8: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ae0: ee09407c 02000000 00002002
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003aec: 8c7e067e
	s_and_b32 s2, s5, vcc_lo                                   // 000000003af0: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003af4: bf88ff9e
	s_and_saveexec_b32 s5, s2                                  // 000000003af8: be852002
	s_cbranch_execz 46                                         // 000000003afc: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x20b8>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003b00: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003b08: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003b0c: d5207c03 00082680
	v_bfe_u32 v4, v28, 16, 1                                   // 000000003b14: d6100004 0205211c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b1c: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003b20: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003b28: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003b2c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v28                             // 000000003b34: 380e38ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b3c: bf8701a3
	v_add_co_u32 v2, s2, s36, v2                               // 000000003b40: d7000202 02020424
	s_wait_alu depctr_va_sdst(0)                               // 000000003b48: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, v3, s2                  // 000000003b4c: d5207c03 000a0625
	s_wait_kmcnt 0x0                                           // 000000003b54: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003b58: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003b60: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003b64: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003b6c: 3e040481
	v_add3_u32 v4, v4, v28, 0x7fff                             // 000000003b70: d6550004 03fe3904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b7c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003b80: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003b88: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003b8c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v28, v28                               // 000000003b94: d4180002 0202391c
	s_wait_alu depctr_va_sdst(0)                               // 000000003b9c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ba0: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003ba4: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003bac: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bb8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003bbc: 8c7e057e
	s_and_b32 s2, s4, vcc_lo                                   // 000000003bc0: 8b026a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bc4: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003bc8: be842002
	s_cbranch_execz 46                                         // 000000003bcc: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x2188>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003bd0: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003bdc: d5207c03 00082680
	v_bfe_u32 v4, v27, 16, 1                                   // 000000003be4: d6100004 0205211b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003bec: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003bf0: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003bf8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003bfc: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v27                             // 000000003c04: 380e36ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c0c: bf8701a3
	v_add_co_u32 v2, s2, s34, v2                               // 000000003c10: d7000202 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000003c18: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s2                  // 000000003c1c: d5207c03 000a0623
	s_wait_kmcnt 0x0                                           // 000000003c24: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003c28: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003c30: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003c34: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003c3c: 3e040481
	v_add3_u32 v4, v4, v27, 0x7fff                             // 000000003c40: d6550004 03fe3704 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c4c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003c50: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003c58: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003c5c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v27, v27                               // 000000003c64: d4180002 0202371b
	s_wait_alu depctr_va_sdst(0)                               // 000000003c6c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c70: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003c74: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003c7c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c88: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003c8c: 8c7e047e
	s_and_b32 s2, s3, vcc_lo                                   // 000000003c90: 8b026a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c94: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003c98: be832002
	s_cbranch_execz 46                                         // 000000003c9c: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x2258>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003ca0: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003ca8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003cac: d5207c03 00082680
	v_bfe_u32 v4, v26, 16, 1                                   // 000000003cb4: d6100004 0205211a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cbc: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003cc0: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003cc8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003ccc: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v26                             // 000000003cd4: 380e34ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cdc: bf8701a3
	v_add_co_u32 v2, s2, s30, v2                               // 000000003ce0: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000003ce8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000003cec: d5207c03 000a061f
	s_wait_kmcnt 0x0                                           // 000000003cf4: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003cf8: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003d00: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003d04: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003d0c: 3e040481
	v_add3_u32 v4, v4, v26, 0x7fff                             // 000000003d10: d6550004 03fe3504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d1c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003d20: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003d28: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003d2c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v26, v26                               // 000000003d34: d4180002 0202351a
	s_wait_alu depctr_va_sdst(0)                               // 000000003d3c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003d40: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003d44: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003d4c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d58: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000003d5c: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 000000003d60: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d64: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000003d68: be822001
	s_cbranch_execz 46                                         // 000000003d6c: bfa5002e <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x2328>
	v_add_co_u32 v2, s1, v23, s18                              // 000000003d70: d7000102 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003d78: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s1                   // 000000003d7c: d5207c03 00042680
	v_bfe_u32 v4, v25, 16, 1                                   // 000000003d84: d6100004 02052119
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d8c: bf8701a3
	v_add_co_u32 v2, s1, v2, v22                               // 000000003d90: d7000102 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003d98: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 000000003d9c: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v25                             // 000000003da4: 380e32ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003dac: bf8701a3
	v_add_co_u32 v2, s1, s28, v2                               // 000000003db0: d7000102 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 000000003db8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s1                  // 000000003dbc: d5207c03 0006061d
	s_wait_kmcnt 0x0                                           // 000000003dc4: bfc70000
	v_add_co_u32 v5, s1, s24, v0                               // 000000003dc8: d7000105 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003dd0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s1                  // 000000003dd4: d5207c06 00060219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003ddc: 3e040481
	v_add3_u32 v4, v4, v25, 0x7fff                             // 000000003de0: d6550004 03fe3304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003dec: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 000000003df0: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003df8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 000000003dfc: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v25, v25                               // 000000003e04: d4180001 02023319
	s_wait_alu depctr_va_sdst(0)                               // 000000003e0c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003e10: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 000000003e14: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003e1c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e28: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003e2c: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 000000003e30: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e34: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003e38: be812000
	s_cbranch_execz 43                                         // 000000003e3c: bfa5002b <tessera_rocm_scaled_matmul_lds_5bd481a230a95469+0x23ec>
	v_add_co_u32 v2, s0, v23, s18                              // 000000003e40: d7000002 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003e48: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s0                   // 000000003e4c: d5207c03 00002680
	v_bfe_u32 v4, v24, 16, 1                                   // 000000003e54: d6100004 02052118
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e5c: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v22                           // 000000003e60: d7006a02 02022d02
	s_wait_alu depctr_va_vcc(0)                                // 000000003e68: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000003e6c: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v24                             // 000000003e74: 380a30ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e7c: bf8701a3
	v_add_co_u32 v2, vcc_lo, s26, v2                           // 000000003e80: d7006a02 0202041a
	s_wait_alu depctr_va_vcc(0)                                // 000000003e88: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s27, v3, vcc_lo              // 000000003e8c: d5207c03 01aa061b
	s_wait_kmcnt 0x0                                           // 000000003e94: bfc70000
	v_add_co_u32 v0, vcc_lo, s24, v0                           // 000000003e98: d7006a00 02020018
	s_wait_alu depctr_va_vcc(0)                                // 000000003ea0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s25, v1, vcc_lo              // 000000003ea4: d5207c01 01aa0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003eac: 3e040481
	v_add3_u32 v4, v4, v24, 0x7fff                             // 000000003eb0: d6550004 03fe3104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ebc: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 000000003ec0: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 000000003ec8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 000000003ecc: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 000000003ed4: 7c303118
	s_wait_alu depctr_va_vcc(0)                                // 000000003ed8: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000003edc: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000003ee0: ee09407c 01000000 00002000
	s_nop 0                                                    // 000000003eec: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000003ef0: bfb60003
	s_endpgm                                                   // 000000003ef4: bfb00000
	s_code_end                                                 // 000000003ef8: bf9f0000
	s_code_end                                                 // 000000003efc: bf9f0000
	s_code_end                                                 // 000000003f00: bf9f0000
	s_code_end                                                 // 000000003f04: bf9f0000
	s_code_end                                                 // 000000003f08: bf9f0000
	s_code_end                                                 // 000000003f0c: bf9f0000
	s_code_end                                                 // 000000003f10: bf9f0000
	s_code_end                                                 // 000000003f14: bf9f0000
	s_code_end                                                 // 000000003f18: bf9f0000
	s_code_end                                                 // 000000003f1c: bf9f0000
	s_code_end                                                 // 000000003f20: bf9f0000
	s_code_end                                                 // 000000003f24: bf9f0000
	s_code_end                                                 // 000000003f28: bf9f0000
	s_code_end                                                 // 000000003f2c: bf9f0000
	s_code_end                                                 // 000000003f30: bf9f0000
	s_code_end                                                 // 000000003f34: bf9f0000
	s_code_end                                                 // 000000003f38: bf9f0000
	s_code_end                                                 // 000000003f3c: bf9f0000
	s_code_end                                                 // 000000003f40: bf9f0000
	s_code_end                                                 // 000000003f44: bf9f0000
	s_code_end                                                 // 000000003f48: bf9f0000
	s_code_end                                                 // 000000003f4c: bf9f0000
	s_code_end                                                 // 000000003f50: bf9f0000
	s_code_end                                                 // 000000003f54: bf9f0000
	s_code_end                                                 // 000000003f58: bf9f0000
	s_code_end                                                 // 000000003f5c: bf9f0000
	s_code_end                                                 // 000000003f60: bf9f0000
	s_code_end                                                 // 000000003f64: bf9f0000
	s_code_end                                                 // 000000003f68: bf9f0000
	s_code_end                                                 // 000000003f6c: bf9f0000
	s_code_end                                                 // 000000003f70: bf9f0000
	s_code_end                                                 // 000000003f74: bf9f0000
	s_code_end                                                 // 000000003f78: bf9f0000
	s_code_end                                                 // 000000003f7c: bf9f0000
	s_code_end                                                 // 000000003f80: bf9f0000
	s_code_end                                                 // 000000003f84: bf9f0000
	s_code_end                                                 // 000000003f88: bf9f0000
	s_code_end                                                 // 000000003f8c: bf9f0000
	s_code_end                                                 // 000000003f90: bf9f0000
	s_code_end                                                 // 000000003f94: bf9f0000
	s_code_end                                                 // 000000003f98: bf9f0000
	s_code_end                                                 // 000000003f9c: bf9f0000
	s_code_end                                                 // 000000003fa0: bf9f0000
	s_code_end                                                 // 000000003fa4: bf9f0000
	s_code_end                                                 // 000000003fa8: bf9f0000
	s_code_end                                                 // 000000003fac: bf9f0000
	s_code_end                                                 // 000000003fb0: bf9f0000
	s_code_end                                                 // 000000003fb4: bf9f0000
	s_code_end                                                 // 000000003fb8: bf9f0000
	s_code_end                                                 // 000000003fbc: bf9f0000
	s_code_end                                                 // 000000003fc0: bf9f0000
	s_code_end                                                 // 000000003fc4: bf9f0000
	s_code_end                                                 // 000000003fc8: bf9f0000
	s_code_end                                                 // 000000003fcc: bf9f0000
	s_code_end                                                 // 000000003fd0: bf9f0000
	s_code_end                                                 // 000000003fd4: bf9f0000
	s_code_end                                                 // 000000003fd8: bf9f0000
	s_code_end                                                 // 000000003fdc: bf9f0000
	s_code_end                                                 // 000000003fe0: bf9f0000
	s_code_end                                                 // 000000003fe4: bf9f0000
	s_code_end                                                 // 000000003fe8: bf9f0000
	s_code_end                                                 // 000000003fec: bf9f0000
	s_code_end                                                 // 000000003ff0: bf9f0000
	s_code_end                                                 // 000000003ff4: bf9f0000
	s_code_end                                                 // 000000003ff8: bf9f0000
	s_code_end                                                 // 000000003ffc: bf9f0000
	s_code_end                                                 // 000000004000: bf9f0000
	s_code_end                                                 // 000000004004: bf9f0000
	s_code_end                                                 // 000000004008: bf9f0000
	s_code_end                                                 // 00000000400c: bf9f0000
	s_code_end                                                 // 000000004010: bf9f0000
	s_code_end                                                 // 000000004014: bf9f0000
	s_code_end                                                 // 000000004018: bf9f0000
	s_code_end                                                 // 00000000401c: bf9f0000
	s_code_end                                                 // 000000004020: bf9f0000
	s_code_end                                                 // 000000004024: bf9f0000
	s_code_end                                                 // 000000004028: bf9f0000
	s_code_end                                                 // 00000000402c: bf9f0000
	s_code_end                                                 // 000000004030: bf9f0000
	s_code_end                                                 // 000000004034: bf9f0000
	s_code_end                                                 // 000000004038: bf9f0000
	s_code_end                                                 // 00000000403c: bf9f0000
	s_code_end                                                 // 000000004040: bf9f0000
	s_code_end                                                 // 000000004044: bf9f0000
	s_code_end                                                 // 000000004048: bf9f0000
	s_code_end                                                 // 00000000404c: bf9f0000
	s_code_end                                                 // 000000004050: bf9f0000
	s_code_end                                                 // 000000004054: bf9f0000
	s_code_end                                                 // 000000004058: bf9f0000
	s_code_end                                                 // 00000000405c: bf9f0000
	s_code_end                                                 // 000000004060: bf9f0000
	s_code_end                                                 // 000000004064: bf9f0000
	s_code_end                                                 // 000000004068: bf9f0000
	s_code_end                                                 // 00000000406c: bf9f0000
	s_code_end                                                 // 000000004070: bf9f0000
	s_code_end                                                 // 000000004074: bf9f0000
	s_code_end                                                 // 000000004078: bf9f0000
	s_code_end                                                 // 00000000407c: bf9f0000
