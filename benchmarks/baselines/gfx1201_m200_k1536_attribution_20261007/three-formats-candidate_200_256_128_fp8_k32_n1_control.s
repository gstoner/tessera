
/tmp/tmpnsfhwsky.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703>:
	s_clause 0x2                                               // 000000001b00: bf850002
	s_load_b128 s[12:15], s[0:1], 0xc8                         // 000000001b04: f4004300 f80000c8
	s_load_b64 s[16:17], s[0:1], 0xa8                          // 000000001b0c: f4002400 f80000a8
	s_load_b64 s[18:19], s[0:1], 0xd8                          // 000000001b14: f4002480 f80000d8
	s_mov_b32 s2, ttmp9                                        // 000000001b1c: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b20: 86039f75
	s_clause 0x3                                               // 000000001b24: bf850003
	s_load_b64 s[22:23], s[0:1], 0x8                           // 000000001b28: f4002580 f8000008
	s_load_b64 s[26:27], s[0:1], 0x30                          // 000000001b30: f4002680 f8000030
	s_load_b64 s[20:21], s[0:1], 0x58                          // 000000001b38: f4002500 f8000058
	s_load_b64 s[28:29], s[0:1], 0x80                          // 000000001b40: f4002700 f8000080
	s_lshl_b64 s[34:35], s[2:3], 5                             // 000000001b48: 84a28502
	s_delay_alu instid0(salu_cycle_1)                          // 000000001b4c: bf870009
	v_dual_mov_b32 v21, s35 :: v_dual_and_b32 v28, 15, v0      // 000000001b50: ca240023 151c008f
	s_mov_b32 s4, ttmp7                                        // 000000001b58: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b5c: 86059f73
	s_add_nc_u64 s[2:3], s[34:35], 32                          // 000000001b60: a982a022
	s_lshl_b64 s[30:31], s[4:5], 4                             // 000000001b64: 849e8404
	v_or_b32_e32 v16, s34, v28                                 // 000000001b68: 38203822
	s_add_nc_u64 s[0:1], s[30:31], 16                          // 000000001b6c: a980901e
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v17, s35           // 000000001b70: ca100080 22100023
	v_bfe_u32 v0, v0, 4, 1                                     // 000000001b78: d6100000 02050900
	s_delay_alu instid0(valu_dep_3)                            // 000000001b80: bf870003
	v_or_b32_e32 v20, 16, v16                                  // 000000001b84: 38282090
	s_wait_kmcnt 0x0                                           // 000000001b88: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[12:13]                      // 000000001b8c: d4540000 02001800
	v_cmp_gt_i64_e64 s2, s[2:3], s[14:15]                      // 000000001b94: d4540002 02001c02
	v_cmp_gt_i64_e64 s1, s[14:15], v[16:17]                    // 000000001b9c: d4540001 0202200e
	v_lshlrev_b32_e32 v18, 3, v0                               // 000000001ba4: 30240083
	s_lshr_b64 s[24:25], s[18:19], 5                           // 000000001ba8: 85988512
	s_or_b32 s2, s0, s2                                        // 000000001bac: 8c020200
	v_cmp_gt_i64_e64 s0, s[14:15], v[20:21]                    // 000000001bb0: d4540000 0202280e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb8: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001bbc: 8b6a027e
	s_mov_b32 s2, -1                                           // 000000001bc0: be8200c1
	s_cbranch_vccnz 4                                          // 000000001bc4: bfa40004 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0xd8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc8: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001bcc: 8b6a027e
	s_cbranch_vccnz 2214                                       // 000000001bd0: bfa408a6 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x236c>
	s_endpgm                                                   // 000000001bd4: bfb00000
	v_or_b32_e32 v0, s30, v28                                  // 000000001bd8: 3800381e
	v_mov_b32_e32 v1, s31                                      // 000000001bdc: 7e02021f
	v_or_b32_e32 v22, s30, v18                                 // 000000001be0: 382c241e
	s_mul_i32 s2, s18, s31                                     // 000000001be4: 96021f12
	v_mul_lo_u32 v8, s19, v16                                  // 000000001be8: d72c0008 02022013
	v_mul_lo_u32 v9, s19, v0                                   // 000000001bf0: d72c0009 02020013
	v_mad_co_u64_u32 v[2:3], null, s18, v0, 0                  // 000000001bf8: d6fe7c02 02020012
	v_mul_lo_u32 v10, s18, v17                                 // 000000001c00: d72c000a 02022212
	v_mad_co_u64_u32 v[4:5], null, s18, v16, 0                 // 000000001c08: d6fe7c04 02022012
	v_mul_lo_u32 v11, s19, v20                                 // 000000001c10: d72c000b 02022813
	v_mul_lo_u32 v12, s18, v21                                 // 000000001c18: d72c000c 02022a12
	v_mad_co_u64_u32 v[6:7], null, s18, v20, 0                 // 000000001c20: d6fe7c06 02022812
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v46, 0             // 000000001c28: ca100080 352e0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c30: bf88ff9e
	v_add3_u32 v39, v3, s2, v9                                 // 000000001c34: d6550027 04240503
	v_or_b32_e32 v40, v2, v18                                  // 000000001c3c: 38502502
	v_mov_b32_e32 v2, s31                                      // 000000001c40: 7e04021f
	v_cmp_gt_i64_e64 s2, s[12:13], v[0:1]                      // 000000001c44: d4540002 0202000c
	v_or_b32_e32 v0, 1, v22                                    // 000000001c4c: 38002c81
	v_mov_b32_e32 v23, s31                                     // 000000001c50: 7e2e021f
	v_add3_u32 v41, v5, v10, v8                                // 000000001c54: d6550029 04221505
	v_or_b32_e32 v43, v4, v18                                  // 000000001c5c: 38562504
	v_add3_u32 v44, v7, v12, v11                               // 000000001c60: d655002c 042e1907
	v_or_b32_e32 v45, v6, v18                                  // 000000001c68: 385a2506
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[22:23]                // 000000001c6c: 7ca82c0c
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v42, 0             // 000000001c70: ca100080 2f2a0080
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0             // 000000001c78: ca100080 26240080
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v35, 0             // 000000001c80: ca100080 24220080
	v_cndmask_b32_e32 v3, 0, v22, vcc_lo                       // 000000001c88: 02062c80
	v_cndmask_b32_e64 v5, 0, s31, vcc_lo                       // 000000001c8c: d5010005 01a83e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001c94: 7ca8000c
	v_or_b32_e32 v1, 2, v22                                    // 000000001c98: 38022c82
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0             // 000000001c9c: ca100080 21200080
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0             // 000000001ca4: ca100080 1f1e0080
	s_wait_alu depctr_va_vcc(0)                                // 000000001cac: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo                        // 000000001cb0: 02000080
	v_cndmask_b32_e64 v8, 0, s31, vcc_lo                       // 000000001cb4: d5010008 01a83e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[1:2]                  // 000000001cbc: 7ca8020c
	v_mul_lo_u32 v2, s24, v5                                   // 000000001cc0: d72c0002 02020a18
	v_mov_b32_e32 v29, 0                                       // 000000001cc8: 7e3a0280
	v_mul_lo_u32 v9, s25, v0                                   // 000000001ccc: d72c0009 02020019
	v_mad_co_u64_u32 v[5:6], null, s24, v0, 0                  // 000000001cd4: d6fe7c05 02020018
	v_mul_lo_u32 v10, s24, v8                                  // 000000001cdc: d72c000a 02021018
	s_wait_alu depctr_va_vcc(0)                                // 000000001ce4: bf88ff9d
	v_cndmask_b32_e32 v8, 0, v1, vcc_lo                        // 000000001ce8: 02100280
	v_mul_lo_u32 v7, s25, v3                                   // 000000001cec: d72c0007 02020619
	v_mad_co_u64_u32 v[3:4], null, s24, v3, 0                  // 000000001cf4: d6fe7c03 02020618
	v_or_b32_e32 v0, 3, v22                                    // 000000001cfc: 38002c83
	v_mov_b32_e32 v1, s31                                      // 000000001d00: 7e02021f
	v_cndmask_b32_e64 v11, 0, s31, vcc_lo                      // 000000001d04: d501000b 01a83e80
	v_mul_lo_u32 v12, s25, v8                                  // 000000001d0c: d72c000c 02021019
	v_add3_u32 v6, v6, v10, v9                                 // 000000001d14: d6550006 04261506
	v_mov_b32_e32 v19, 0                                       // 000000001d1c: 7e260280
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001d20: 7ca8000c
	v_add3_u32 v4, v4, v2, v7                                  // 000000001d24: d6550004 041e0504
	v_mad_co_u64_u32 v[7:8], null, s24, v8, 0                  // 000000001d2c: d6fe7c07 02021018
	v_mul_lo_u32 v11, s24, v11                                 // 000000001d34: d72c000b 02021618
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]                        // 000000001d3c: 3e0a0a82
	s_wait_alu depctr_va_vcc(0)                                // 000000001d40: bf88ff9d
	v_dual_mov_b32 v57, 0 :: v_dual_cndmask_b32 v0, 0, v0      // 000000001d44: ca120080 39000080
	v_cndmask_b32_e64 v13, 0, s31, vcc_lo                      // 000000001d4c: d501000d 01a83e80
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]                        // 000000001d54: 3e060682
	v_cndmask_b32_e64 v2, 0, v17, s1                           // 000000001d58: d5010002 00062280
	v_cndmask_b32_e64 v1, 0, v16, s1                           // 000000001d60: d5010001 00062080
	v_add3_u32 v8, v8, v11, v12                                // 000000001d68: d6550008 04321708
	v_mul_lo_u32 v11, s25, v0                                  // 000000001d70: d72c000b 02020019
	v_mad_co_u64_u32 v[9:10], null, s24, v0, 0                 // 000000001d78: d6fe7c09 02020018
	v_mul_lo_u32 v0, s24, v13                                  // 000000001d80: d72c0000 02021a18
	v_add_co_u32 v48, vcc_lo, s20, v3                          // 000000001d88: d7006a30 02020614
	s_wait_alu depctr_va_vcc(0)                                // 000000001d90: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s21, v4, vcc_lo             // 000000001d94: d5207c31 01aa0815
	v_lshlrev_b64_e32 v[3:4], 2, v[7:8]                        // 000000001d9c: 3e060e82
	v_add_co_u32 v50, vcc_lo, s20, v5                          // 000000001da0: d7006a32 02020a14
	s_wait_alu depctr_va_vcc(0)                                // 000000001da8: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s21, v6, vcc_lo             // 000000001dac: d5207c33 01aa0c15
	v_add3_u32 v10, v10, v0, v11                               // 000000001db4: d655000a 042e010a
	v_or_b32_e32 v5, 4, v22                                    // 000000001dbc: 380a2c84
	v_mov_b32_e32 v6, s31                                      // 000000001dc0: 7e0c021f
	v_add_co_u32 v52, vcc_lo, s20, v3                          // 000000001dc4: d7006a34 02020614
	s_wait_alu depctr_va_vcc(0)                                // 000000001dcc: bf88ff9d
	v_add_co_ci_u32_e64 v54, null, s21, v4, vcc_lo             // 000000001dd0: d5207c36 01aa0815
	v_lshlrev_b64_e32 v[3:4], 2, v[9:10]                       // 000000001dd8: 3e061282
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000001ddc: 7ca80a0c
	v_mov_b32_e32 v8, s31                                      // 000000001de0: 7e10021f
	v_lshlrev_b64_e32 v[24:25], 2, v[1:2]                      // 000000001de4: 3e300282
	s_add_nc_u64 s[36:37], s[18:19], -1                        // 000000001de8: a9a4c112
	s_mov_b64 s[38:39], 0                                      // 000000001dec: bea60180
	v_add_co_u32 v55, s3, s20, v3                              // 000000001df0: d7000337 02020614
	v_or_b32_e32 v3, 6, v22                                    // 000000001df8: 38062c86
	s_wait_alu depctr_va_vcc(0)                                // 000000001dfc: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v5, vcc_lo                        // 000000001e00: 02000a80
	v_or_b32_e32 v5, 5, v22                                    // 000000001e04: 380a2c85
	v_cndmask_b32_e64 v7, 0, s31, vcc_lo                       // 000000001e08: d5010007 01a83e80
	s_wait_alu depctr_va_sdst(0)                               // 000000001e10: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s21, v4, s3                 // 000000001e14: d5207c38 000e0815
	v_mov_b32_e32 v4, s31                                      // 000000001e1c: 7e08021f
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000001e20: 7ca80a0c
	s_wait_alu depctr_va_vcc(0)                                // 000000001e24: bf88ff9d
	v_cndmask_b32_e32 v9, 0, v5, vcc_lo                        // 000000001e28: 02120a80
	v_cndmask_b32_e64 v12, 0, s31, vcc_lo                      // 000000001e2c: d501000c 01a83e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[3:4]                  // 000000001e34: 7ca8060c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000001e38: bf870223
	v_mul_lo_u32 v13, s25, v9                                  // 000000001e3c: d72c000d 02021219
	v_mad_co_u64_u32 v[9:10], null, s24, v9, 0                 // 000000001e44: d6fe7c09 02021218
	v_mul_lo_u32 v12, s24, v12                                 // 000000001e4c: d72c000c 02021818
	s_wait_alu depctr_va_vcc(0)                                // 000000001e54: bf88ff9d
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo                        // 000000001e58: 02060680
	v_mul_lo_u32 v11, s25, v0                                  // 000000001e5c: d72c000b 02020019
	v_mad_co_u64_u32 v[5:6], null, s24, v0, 0                  // 000000001e64: d6fe7c05 02020018
	v_mul_lo_u32 v0, s24, v7                                   // 000000001e6c: d72c0000 02020e18
	v_or_b32_e32 v7, 7, v22                                    // 000000001e74: 380e2c87
	v_cndmask_b32_e64 v14, 0, s31, vcc_lo                      // 000000001e78: d501000e 01a83e80
	v_add3_u32 v10, v10, v12, v13                              // 000000001e80: d655000a 0436190a
	s_delay_alu instid0(valu_dep_3)                            // 000000001e88: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[7:8]                  // 000000001e8c: 7ca80e0c
	v_add3_u32 v6, v6, v0, v11                                 // 000000001e90: d6550006 042e0106
	v_mul_lo_u32 v0, s25, v3                                   // 000000001e98: d72c0000 02020619
	v_mad_co_u64_u32 v[3:4], null, s24, v3, 0                  // 000000001ea0: d6fe7c03 02020618
	v_mul_lo_u32 v11, s24, v14                                 // 000000001ea8: d72c000b 02021c18
	s_wait_alu depctr_va_vcc(0)                                // 000000001eb0: bf88ff9d
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo                        // 000000001eb4: 020e0e80
	v_cndmask_b32_e64 v14, 0, s31, vcc_lo                      // 000000001eb8: d501000e 01a83e80
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]                        // 000000001ec0: 3e0a0a82
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]                      // 000000001ec4: 3e121282
	s_delay_alu instid0(valu_dep_4)                            // 000000001ec8: bf870004
	v_mul_lo_u32 v12, s25, v7                                  // 000000001ecc: d72c000c 02020e19
	v_mad_co_u64_u32 v[7:8], null, s24, v7, 0                  // 000000001ed4: d6fe7c07 02020e18
	v_mul_lo_u32 v13, s24, v14                                 // 000000001edc: d72c000d 02021c18
	v_add3_u32 v4, v4, v11, v0                                 // 000000001ee4: d6550004 04021704
	v_add_co_u32 v58, vcc_lo, s20, v5                          // 000000001eec: d7006a3a 02020a14
	s_wait_alu depctr_va_vcc(0)                                // 000000001ef4: bf88ff9d
	v_add_co_ci_u32_e64 v59, null, s21, v6, vcc_lo             // 000000001ef8: d5207c3b 01aa0c15
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000001f00: bf870253
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]                        // 000000001f04: 3e060682
	v_add_co_u32 v60, vcc_lo, s20, v9                          // 000000001f08: d7006a3c 02021214
	v_add3_u32 v8, v8, v13, v12                                // 000000001f10: d6550008 04321b08
	s_wait_alu depctr_va_vcc(0)                                // 000000001f18: bf88ff9d
	v_add_co_ci_u32_e64 v61, null, s21, v10, vcc_lo            // 000000001f1c: d5207c3d 01aa1415
	v_add_co_u32 v62, vcc_lo, s20, v3                          // 000000001f24: d7006a3e 02020614
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000001f2c: bf870253
	v_lshlrev_b64_e32 v[5:6], 2, v[7:8]                        // 000000001f30: 3e0a0e82
	s_wait_alu depctr_va_vcc(0)                                // 000000001f34: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, s21, v4, vcc_lo             // 000000001f38: d5207c3f 01aa0815
	v_cndmask_b32_e64 v4, 0, v21, s0                           // 000000001f40: d5010004 00022a80
	v_cndmask_b32_e64 v3, 0, v20, s0                           // 000000001f48: d5010003 00022880
	v_add_co_u32 v64, vcc_lo, s20, v5                          // 000000001f50: d7006a40 02020a14
	s_wait_alu depctr_va_vcc(0)                                // 000000001f58: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s21, v6, vcc_lo             // 000000001f5c: d5207c41 01aa0c15
	s_delay_alu instid0(valu_dep_3)                            // 000000001f64: bf870003
	v_lshlrev_b64_e32 v[26:27], 2, v[3:4]                      // 000000001f68: 3e340682
	v_mov_b32_e32 v5, s39                                      // 000000001f6c: 7e0a0227
	v_or_b32_e32 v4, s38, v18                                  // 000000001f70: 38082426
	v_add_co_u32 v8, vcc_lo, s38, v40                          // 000000001f74: d7006a08 02025026
	s_wait_alu depctr_va_vcc(0)                                // 000000001f7c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s39, v39, vcc_lo             // 000000001f80: d5207c09 01aa4e27
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000001f88: bf870193
	v_cmp_gt_u64_e32 vcc_lo, s[18:19], v[4:5]                  // 000000001f8c: 7cb80812
	v_or_b32_e32 v3, 2, v8                                     // 000000001f90: 38061082
	v_or_b32_e32 v6, 3, v8                                     // 000000001f94: 380c1083
	v_mov_b32_e32 v7, s39                                      // 000000001f98: 7e0e0227
	s_or_b32 s33, s38, 16                                      // 000000001f9c: 8c219026
	v_mov_b32_e32 v71, s39                                     // 000000001fa0: 7e8e0227
	s_and_b32 s3, s2, vcc_lo                                   // 000000001fa4: 8b036a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fa8: bf88ff9e
	v_or_b32_e32 v70, s33, v18                                 // 000000001fac: 388c2421
	v_cndmask_b32_e64 v0, 0, v8, s3                            // 000000001fb0: d5010000 000e1080
	v_cndmask_b32_e64 v1, 0, v9, s3                            // 000000001fb8: d5010001 000e1280
	v_mov_b32_e32 v73, s39                                     // 000000001fc0: 7e920227
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000001fc4: bf8701a3
	v_add_co_u32 v0, s4, s22, v0                               // 000000001fc8: d7000400 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000001fd0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s23, v1, s4                  // 000000001fd4: d5207c01 00120217
	global_load_d16_u8 v0, v[0:1], off                         // 000000001fdc: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v8                                     // 000000001fe8: 38021081
	s_wait_loadcnt 0x0                                         // 000000001fec: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s3                            // 000000001ff0: d65d0000 000e0080
	v_cmp_gt_u64_e64 s3, s[36:37], v[4:5]                      // 000000001ff8: d45c0003 02020824
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002000: bf870152
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002004: d7620000 020200ff 000000ff
	s_and_b32 s4, s2, s3                                       // 000000002010: 8b040302
	s_wait_alu depctr_sa_sdst(0)                               // 000000002014: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s4                            // 000000002018: d5010001 00120280
	v_cndmask_b32_e64 v2, 0, v9, s4                            // 000000002020: d5010002 00121280
	v_add_co_u32 v1, s5, s22, v1                               // 000000002028: d7000501 02020216
	s_wait_alu depctr_va_sdst(0)                               // 000000002030: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002034: bf870002
	v_add_co_ci_u32_e64 v2, null, s23, v2, s5                  // 000000002038: d5207c02 00160417
	global_load_d16_hi_u8 v0, v[1:2], off                      // 000000002040: ee08407c 00000000 00000001
	v_or_b32_e32 v1, 2, v4                                     // 00000000204c: 38020882
	v_mov_b32_e32 v2, s39                                      // 000000002050: 7e040227
	s_wait_loadcnt 0x0                                         // 000000002054: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, s4                            // 000000002058: d65d5000 00120080
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002060: bf870112
	v_cmp_gt_u64_e64 s4, s[18:19], v[1:2]                      // 000000002064: d45c0004 02020212
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 00000000206c: d7385000 02020088
	s_and_b32 s5, s2, s4                                       // 000000002074: 8b050402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002078: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v3, s5                            // 00000000207c: d5010001 00160680
	v_cndmask_b32_e64 v2, 0, v9, s5                            // 000000002084: d5010002 00161280
	v_mov_b32_e32 v3, s39                                      // 00000000208c: 7e060227
	v_or_b16 v66.l, v0.l, v0.h op_sel:[0,1,0]                  // 000000002090: d7631042 02020100
	s_delay_alu instid0(valu_dep_4)                            // 000000002098: bf870004
	v_add_co_u32 v1, s6, s22, v1                               // 00000000209c: d7000601 02020216
	s_wait_alu depctr_va_sdst(0)                               // 0000000020a4: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s23, v2, s6                  // 0000000020a8: d5207c02 001a0417
	global_load_d16_u8 v1, v[1:2], off                         // 0000000020b0: ee07807c 00000001 00000001
	v_or_b32_e32 v2, 3, v4                                     // 0000000020bc: 38040883
	s_wait_loadcnt 0x0                                         // 0000000020c0: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, s5                            // 0000000020c4: d65d0001 00160280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000020cc: bf870112
	v_cmp_gt_u64_e64 s5, s[18:19], v[2:3]                      // 0000000020d0: d45c0005 02020412
	v_and_b16 v1.l, 0xff, v1.l                                 // 0000000020d8: d7620001 020202ff 000000ff
	s_and_b32 s6, s2, s5                                       // 0000000020e4: 8b060502
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020e8: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v6, s6                            // 0000000020ec: d5010002 001a0c80
	v_cndmask_b32_e64 v3, 0, v9, s6                            // 0000000020f4: d5010003 001a1280
	v_or_b32_e32 v6, 4, v8                                     // 0000000020fc: 380c1084
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002100: bf8701a3
	v_add_co_u32 v2, s7, s22, v2                               // 000000002104: d7000702 02020416
	s_wait_alu depctr_va_sdst(0)                               // 00000000210c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s7                  // 000000002110: d5207c03 001e0617
	global_load_d16_hi_u8 v1, v[2:3], off                      // 000000002118: ee08407c 00000001 00000002
	v_or_b32_e32 v2, 4, v4                                     // 000000002124: 38040884
	v_mov_b32_e32 v3, s39                                      // 000000002128: 7e060227
	s_wait_loadcnt 0x0                                         // 00000000212c: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, s6                            // 000000002130: d65d5001 001a0280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002138: bf870112
	v_cmp_gt_u64_e64 s6, s[18:19], v[2:3]                      // 00000000213c: d45c0006 02020412
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 000000002144: d7385001 02020288
	s_and_b32 s7, s2, s6                                       // 00000000214c: 8b070602
	s_wait_alu depctr_sa_sdst(0)                               // 000000002150: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v6, s7                            // 000000002154: d5010002 001e0c80
	v_cndmask_b32_e64 v3, 0, v9, s7                            // 00000000215c: d5010003 001e1280
	v_or_b32_e32 v6, 5, v4                                     // 000000002164: 380c0885
	v_or_b16 v66.h, v1.l, v1.h op_sel:[0,1,1]                  // 000000002168: d7635042 02020301
	s_delay_alu instid0(valu_dep_4)                            // 000000002170: bf870004
	v_add_co_u32 v2, s8, s22, v2                               // 000000002174: d7000802 02020416
	s_wait_alu depctr_va_sdst(0)                               // 00000000217c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s8                  // 000000002180: d5207c03 00220617
	global_load_d16_u8 v2, v[2:3], off                         // 000000002188: ee07807c 00000002 00000002
	v_or_b32_e32 v3, 5, v8                                     // 000000002194: 38061085
	s_wait_loadcnt 0x0                                         // 000000002198: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s7                            // 00000000219c: d65d0002 001e0480
	v_cmp_gt_u64_e64 s7, s[18:19], v[6:7]                      // 0000000021a4: d45c0007 02020c12
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 0000000021ac: bf870152
	v_and_b16 v2.l, 0xff, v2.l                                 // 0000000021b0: d7620002 020204ff 000000ff
	s_and_b32 s8, s2, s7                                       // 0000000021bc: 8b080702
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021c0: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s8                            // 0000000021c4: d5010003 00220680
	v_cndmask_b32_e64 v7, 0, v9, s8                            // 0000000021cc: d5010007 00221280
	v_add_co_u32 v6, s9, s22, v3                               // 0000000021d4: d7000906 02020616
	s_wait_alu depctr_va_sdst(0)                               // 0000000021dc: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000021e0: bf870002
	v_add_co_ci_u32_e64 v7, null, s23, v7, s9                  // 0000000021e4: d5207c07 00260e17
	v_or_b32_e32 v3, 6, v8                                     // 0000000021ec: 38061086
	global_load_d16_hi_u8 v2, v[6:7], off                      // 0000000021f0: ee08407c 00000002 00000006
	v_or_b32_e32 v6, 6, v4                                     // 0000000021fc: 380c0886
	v_mov_b32_e32 v7, s39                                      // 000000002200: 7e0e0227
	v_or_b32_e32 v4, 7, v4                                     // 000000002204: 38080887
	s_wait_loadcnt 0x0                                         // 000000002208: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s8                            // 00000000220c: d65d5002 00220480
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002214: bf870113
	v_cmp_gt_u64_e64 s8, s[18:19], v[6:7]                      // 000000002218: d45c0008 02020c12
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002220: d7385002 02020488
	s_and_b32 s9, s2, s8                                       // 000000002228: 8b090802
	s_wait_alu depctr_sa_sdst(0)                               // 00000000222c: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s9                            // 000000002230: d5010003 00260680
	v_cndmask_b32_e64 v7, 0, v9, s9                            // 000000002238: d5010007 00261280
	v_or_b16 v67.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002240: d7631043 02020502
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002248: bf8701a3
	v_add_co_u32 v6, s10, s22, v3                              // 00000000224c: d7000a06 02020616
	s_wait_alu depctr_va_sdst(0)                               // 000000002254: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s23, v7, s10                 // 000000002258: d5207c07 002a0e17
	global_load_d16_u8 v3, v[6:7], off                         // 000000002260: ee07807c 00000003 00000006
	v_or_b32_e32 v6, 7, v8                                     // 00000000226c: 380c1087
	s_wait_loadcnt 0x0                                         // 000000002270: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s9                            // 000000002274: d65d0003 00260680
	v_cmp_gt_u64_e64 s9, s[18:19], v[4:5]                      // 00000000227c: d45c0009 02020812
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002284: bf870152
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002288: d7620003 020206ff 000000ff
	s_and_b32 s10, s2, s9                                      // 000000002294: 8b0a0902
	s_wait_alu depctr_sa_sdst(0)                               // 000000002298: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v6, s10                           // 00000000229c: d5010004 002a0c80
	v_cndmask_b32_e64 v5, 0, v9, s10                           // 0000000022a4: d5010005 002a1280
	v_add_co_u32 v4, s11, s22, v4                              // 0000000022ac: d7000b04 02020816
	s_wait_alu depctr_va_sdst(0)                               // 0000000022b4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000022b8: bf870002
	v_add_co_ci_u32_e64 v5, null, s23, v5, s11                 // 0000000022bc: d5207c05 002e0a17
	global_load_d16_hi_u8 v3, v[4:5], off                      // 0000000022c4: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 0000000022d0: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s10                           // 0000000022d4: d65d5003 002a0680
	v_add_co_u32 v5, s10, s38, v43                             // 0000000022dc: d7000a05 02025626
	s_wait_alu depctr_va_sdst(0)                               // 0000000022e4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s39, v41, s10                // 0000000022e8: d5207c06 002a5227
	s_and_b32 s10, s1, vcc_lo                                  // 0000000022f0: 8b0a6a01
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000022f4: d7385003 02020688
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022fc: bf88ff9e
	v_cndmask_b32_e64 v0, 0, v5, s10                           // 000000002300: d5010000 002a0a80
	v_cndmask_b32_e64 v1, 0, v6, s10                           // 000000002308: d5010001 002a0c80
	s_and_b32 vcc_lo, s0, vcc_lo                               // 000000002310: 8b6a6a00
	v_or_b16 v67.h, v3.l, v3.h op_sel:[0,1,1]                  // 000000002314: d7635043 02020703
	s_delay_alu instid0(valu_dep_3)                            // 00000000231c: bf870003
	v_add_co_u32 v0, s11, s26, v0                              // 000000002320: d7000b00 0202001a
	s_wait_alu depctr_va_sdst(0)                               // 000000002328: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s27, v1, s11                 // 00000000232c: d5207c01 002e021b
	global_load_d16_u8 v0, v[0:1], off                         // 000000002334: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v5                                     // 000000002340: 38020a81
	s_wait_loadcnt 0x0                                         // 000000002344: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s10                           // 000000002348: d65d0000 002a0080
	s_and_b32 s10, s1, s3                                      // 000000002350: 8b0a0301
	s_wait_alu depctr_sa_sdst(0)                               // 000000002354: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s10                           // 000000002358: d5010001 002a0280
	v_cndmask_b32_e64 v2, 0, v6, s10                           // 000000002360: d5010002 002a0c80
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002368: d7620000 020200ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002374: bf8701a3
	v_add_co_u32 v1, s11, s26, v1                              // 000000002378: d7000b01 0202021a
	s_wait_alu depctr_va_sdst(0)                               // 000000002380: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s27, v2, s11                 // 000000002384: d5207c02 002e041b
	global_load_d16_hi_u8 v0, v[1:2], off                      // 00000000238c: ee08407c 00000000 00000001
	v_or_b32_e32 v1, 2, v5                                     // 000000002398: 38020a82
	s_wait_loadcnt 0x0                                         // 00000000239c: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, s10                           // 0000000023a0: d65d5000 002a0080
	s_and_b32 s10, s1, s4                                      // 0000000023a8: 8b0a0401
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023ac: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s10                           // 0000000023b0: d5010001 002a0280
	v_cndmask_b32_e64 v2, 0, v6, s10                           // 0000000023b8: d5010002 002a0c80
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 0000000023c0: d7385000 02020088
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000023c8: bf8701a3
	v_add_co_u32 v1, s11, s26, v1                              // 0000000023cc: d7000b01 0202021a
	s_wait_alu depctr_va_sdst(0)                               // 0000000023d4: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s27, v2, s11                 // 0000000023d8: d5207c02 002e041b
	global_load_d16_u8 v1, v[1:2], off                         // 0000000023e0: ee07807c 00000001 00000001
	v_or_b32_e32 v2, 3, v5                                     // 0000000023ec: 38040a83
	s_wait_loadcnt 0x0                                         // 0000000023f0: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, s10                           // 0000000023f4: d65d0001 002a0280
	s_and_b32 s10, s1, s5                                      // 0000000023fc: 8b0a0501
	s_wait_alu depctr_sa_sdst(0)                               // 000000002400: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v2, s10                           // 000000002404: d5010002 002a0480
	v_cndmask_b32_e64 v3, 0, v6, s10                           // 00000000240c: d5010003 002a0c80
	v_and_b16 v1.l, 0xff, v1.l                                 // 000000002414: d7620001 020202ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002420: bf8701a3
	v_add_co_u32 v2, s11, s26, v2                              // 000000002424: d7000b02 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 00000000242c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s27, v3, s11                 // 000000002430: d5207c03 002e061b
	global_load_d16_hi_u8 v1, v[2:3], off                      // 000000002438: ee08407c 00000001 00000002
	v_or_b32_e32 v2, 4, v5                                     // 000000002444: 38040a84
	s_wait_loadcnt 0x0                                         // 000000002448: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, s10                           // 00000000244c: d65d5001 002a0280
	s_and_b32 s10, s1, s6                                      // 000000002454: 8b0a0601
	s_wait_alu depctr_sa_sdst(0)                               // 000000002458: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v2, s10                           // 00000000245c: d5010002 002a0480
	v_cndmask_b32_e64 v3, 0, v6, s10                           // 000000002464: d5010003 002a0c80
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 00000000246c: d7385001 02020288
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002474: bf8701a3
	v_add_co_u32 v2, s11, s26, v2                              // 000000002478: d7000b02 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 000000002480: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s27, v3, s11                 // 000000002484: d5207c03 002e061b
	global_load_d16_u8 v2, v[2:3], off                         // 00000000248c: ee07807c 00000002 00000002
	v_or_b32_e32 v3, 5, v5                                     // 000000002498: 38060a85
	s_wait_loadcnt 0x0                                         // 00000000249c: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s10                           // 0000000024a0: d65d0002 002a0480
	s_and_b32 s10, s1, s7                                      // 0000000024a8: 8b0a0701
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024ac: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s10                           // 0000000024b0: d5010003 002a0680
	v_cndmask_b32_e64 v4, 0, v6, s10                           // 0000000024b8: d5010004 002a0c80
	v_and_b16 v2.l, 0xff, v2.l                                 // 0000000024c0: d7620002 020204ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000024cc: bf8701a3
	v_add_co_u32 v3, s11, s26, v3                              // 0000000024d0: d7000b03 0202061a
	s_wait_alu depctr_va_sdst(0)                               // 0000000024d8: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s27, v4, s11                 // 0000000024dc: d5207c04 002e081b
	global_load_d16_hi_u8 v2, v[3:4], off                      // 0000000024e4: ee08407c 00000002 00000003
	v_or_b32_e32 v3, 6, v5                                     // 0000000024f0: 38060a86
	s_wait_loadcnt 0x0                                         // 0000000024f4: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s10                           // 0000000024f8: d65d5002 002a0480
	s_and_b32 s10, s1, s8                                      // 000000002500: 8b0a0801
	s_wait_alu depctr_sa_sdst(0)                               // 000000002504: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s10                           // 000000002508: d5010003 002a0680
	v_cndmask_b32_e64 v4, 0, v6, s10                           // 000000002510: d5010004 002a0c80
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002518: d7385002 02020488
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002520: bf8701a3
	v_add_co_u32 v3, s11, s26, v3                              // 000000002524: d7000b03 0202061a
	s_wait_alu depctr_va_sdst(0)                               // 00000000252c: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s27, v4, s11                 // 000000002530: d5207c04 002e081b
	global_load_d16_u8 v3, v[3:4], off                         // 000000002538: ee07807c 00000003 00000003
	v_or_b32_e32 v4, 7, v5                                     // 000000002544: 38080a87
	s_wait_loadcnt 0x0                                         // 000000002548: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s10                           // 00000000254c: d65d0003 002a0680
	s_and_b32 s10, s1, s9                                      // 000000002554: 8b0a0901
	s_wait_alu depctr_sa_sdst(0)                               // 000000002558: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s10                           // 00000000255c: d5010004 002a0880
	v_cndmask_b32_e64 v5, 0, v6, s10                           // 000000002564: d5010005 002a0c80
	v_and_b16 v3.l, 0xff, v3.l                                 // 00000000256c: d7620003 020206ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002578: bf8701a3
	v_add_co_u32 v4, s11, s26, v4                              // 00000000257c: d7000b04 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000002584: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s11                 // 000000002588: d5207c05 002e0a1b
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002590: ee08407c 00000003 00000004
	v_or_b16 v4.l, v0.l, v0.h op_sel:[0,1,0]                   // 00000000259c: d7631004 02020100
	v_or_b16 v4.h, v1.l, v1.h op_sel:[0,1,1]                   // 0000000025a4: d7635004 02020301
	v_or_b16 v5.l, v2.l, v2.h op_sel:[0,1,0]                   // 0000000025ac: d7631005 02020502
	s_wait_loadcnt 0x0                                         // 0000000025b4: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s10                           // 0000000025b8: d65d5003 002a0680
	v_add_co_u32 v8, s10, s38, v45                             // 0000000025c0: d7000a08 02025a26
	s_wait_alu depctr_va_sdst(0)                               // 0000000025c8: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s39, v44, s10                // 0000000025cc: d5207c09 002a5827
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000025d4: bf870113
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000025d8: d7385003 02020688
	v_dual_cndmask_b32 v0, 0, v8 :: v_dual_cndmask_b32 v1, 0, v9// 0000000025e0: ca521080 00001280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000025e8: bf870112
	v_or_b16 v5.h, v3.l, v3.h op_sel:[0,1,1]                   // 0000000025ec: d7635005 02020703
	v_add_co_u32 v0, s10, s26, v0                              // 0000000025f4: d7000a00 0202001a
	s_wait_alu depctr_va_sdst(0)                               // 0000000025fc: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002600: bf870003
	v_add_co_ci_u32_e64 v1, null, s27, v1, s10                 // 000000002604: d5207c01 002a021b
	global_load_d16_u8 v0, v[0:1], off                         // 00000000260c: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v8                                     // 000000002618: 38021081
	s_wait_loadcnt 0x0                                         // 00000000261c: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, vcc_lo                        // 000000002620: d65d0000 01aa0080
	s_and_b32 vcc_lo, s0, s3                                   // 000000002628: 8b6a0300
	s_wait_alu depctr_sa_sdst(0)                               // 00000000262c: bf88ff9e
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000002630: 02020280
	v_cndmask_b32_e32 v2, 0, v9, vcc_lo                        // 000000002634: 02041280
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002638: d7620000 020200ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002644: bf8701a3
	v_add_co_u32 v1, s3, s26, v1                               // 000000002648: d7000301 0202021a
	s_wait_alu depctr_va_sdst(0)                               // 000000002650: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s27, v2, s3                  // 000000002654: d5207c02 000e041b
	global_load_d16_hi_u8 v0, v[1:2], off                      // 00000000265c: ee08407c 00000000 00000001
	v_or_b32_e32 v1, 2, v8                                     // 000000002668: 38021082
	s_wait_loadcnt 0x0                                         // 00000000266c: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, vcc_lo                        // 000000002670: d65d5000 01aa0080
	s_and_b32 vcc_lo, s0, s4                                   // 000000002678: 8b6a0400
	s_wait_alu depctr_sa_sdst(0)                               // 00000000267c: bf88ff9e
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000002680: 02020280
	v_cndmask_b32_e32 v2, 0, v9, vcc_lo                        // 000000002684: 02041280
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 000000002688: d7385000 02020088
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002690: bf8701a3
	v_add_co_u32 v1, s3, s26, v1                               // 000000002694: d7000301 0202021a
	s_wait_alu depctr_va_sdst(0)                               // 00000000269c: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s27, v2, s3                  // 0000000026a0: d5207c02 000e041b
	s_delay_alu instid0(valu_dep_3)                            // 0000000026a8: bf870003
	v_or_b16 v68.l, v0.l, v0.h op_sel:[0,1,0]                  // 0000000026ac: d7631044 02020100
	global_load_d16_u8 v1, v[1:2], off                         // 0000000026b4: ee07807c 00000001 00000001
	v_or_b32_e32 v2, 3, v8                                     // 0000000026c0: 38041083
	s_wait_loadcnt 0x0                                         // 0000000026c4: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, vcc_lo                        // 0000000026c8: d65d0001 01aa0280
	s_and_b32 vcc_lo, s0, s5                                   // 0000000026d0: 8b6a0500
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026d4: bf88ff9e
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_cndmask_b32 v3, 0, v9// 0000000026d8: ca520480 02021280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000026e0: bf870112
	v_and_b16 v1.l, 0xff, v1.l                                 // 0000000026e4: d7620001 020202ff 000000ff
	v_add_co_u32 v2, s3, s26, v2                               // 0000000026f0: d7000302 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 0000000026f8: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000026fc: bf870003
	v_add_co_ci_u32_e64 v3, null, s27, v3, s3                  // 000000002700: d5207c03 000e061b
	global_load_d16_hi_u8 v1, v[2:3], off                      // 000000002708: ee08407c 00000001 00000002
	v_or_b32_e32 v2, 4, v8                                     // 000000002714: 38041084
	s_wait_loadcnt 0x0                                         // 000000002718: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, vcc_lo                        // 00000000271c: d65d5001 01aa0280
	s_and_b32 vcc_lo, s0, s6                                   // 000000002724: 8b6a0600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002728: bf88ff9e
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_cndmask_b32 v3, 0, v9// 00000000272c: ca520480 02021280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002734: bf870112
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 000000002738: d7385001 02020288
	v_add_co_u32 v2, s3, s26, v2                               // 000000002740: d7000302 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 000000002748: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000274c: bf870193
	v_add_co_ci_u32_e64 v3, null, s27, v3, s3                  // 000000002750: d5207c03 000e061b
	v_or_b16 v68.h, v1.l, v1.h op_sel:[0,1,1]                  // 000000002758: d7635044 02020301
	global_load_d16_u8 v2, v[2:3], off                         // 000000002760: ee07807c 00000002 00000002
	v_or_b32_e32 v3, 5, v8                                     // 00000000276c: 38061085
	s_wait_loadcnt 0x0                                         // 000000002770: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, vcc_lo                        // 000000002774: d65d0002 01aa0480
	s_and_b32 vcc_lo, s0, s7                                   // 00000000277c: 8b6a0700
	s_wait_alu depctr_sa_sdst(0)                               // 000000002780: bf88ff9e
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo                        // 000000002784: 02060680
	v_cndmask_b32_e32 v7, 0, v9, vcc_lo                        // 000000002788: 020e1280
	v_and_b16 v2.l, 0xff, v2.l                                 // 00000000278c: d7620002 020204ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002798: bf8701a3
	v_add_co_u32 v6, s3, s26, v3                               // 00000000279c: d7000306 0202061a
	s_wait_alu depctr_va_sdst(0)                               // 0000000027a4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v7, s3                  // 0000000027a8: d5207c07 000e0e1b
	v_or_b32_e32 v3, 6, v8                                     // 0000000027b0: 38061086
	global_load_d16_hi_u8 v2, v[6:7], off                      // 0000000027b4: ee08407c 00000002 00000006
	s_wait_loadcnt 0x0                                         // 0000000027c0: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, vcc_lo                        // 0000000027c4: d65d5002 01aa0480
	s_and_b32 vcc_lo, s0, s8                                   // 0000000027cc: 8b6a0800
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027d0: bf88ff9e
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo                        // 0000000027d4: 02060680
	v_cndmask_b32_e32 v7, 0, v9, vcc_lo                        // 0000000027d8: 020e1280
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 0000000027dc: d7385002 02020488
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000027e4: bf8701a3
	v_add_co_u32 v6, s3, s26, v3                               // 0000000027e8: d7000306 0202061a
	s_wait_alu depctr_va_sdst(0)                               // 0000000027f0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v7, s3                  // 0000000027f4: d5207c07 000e0e1b
	s_delay_alu instid0(valu_dep_3)                            // 0000000027fc: bf870003
	v_or_b16 v69.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002800: d7631045 02020502
	global_load_d16_u8 v3, v[6:7], off                         // 000000002808: ee07807c 00000003 00000006
	v_or_b32_e32 v6, 7, v8                                     // 000000002814: 380c1087
	s_wait_loadcnt 0x0                                         // 000000002818: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, vcc_lo                        // 00000000281c: d65d0003 01aa0680
	s_and_b32 vcc_lo, s0, s9                                   // 000000002824: 8b6a0900
	s_wait_alu depctr_sa_sdst(0)                               // 000000002828: bf88ff9e
	v_dual_cndmask_b32 v6, 0, v6 :: v_dual_cndmask_b32 v7, 0, v9// 00000000282c: ca520c80 06061280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002834: bf8701a2
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002838: d7620003 020206ff 000000ff
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[66:67], v[4:5], 0   // 000000002844: cc464008 1a020942
	v_add_co_u32 v6, s3, s26, v6                               // 00000000284c: d7000306 02020c1a
	s_wait_alu depctr_va_sdst(0)                               // 000000002854: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v7, s3                  // 000000002858: d5207c07 000e0e1b
	global_load_d16_hi_u8 v3, v[6:7], off                      // 000000002860: ee08407c 00000003 00000006
	s_wait_loadcnt 0x0                                         // 00000000286c: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, vcc_lo                        // 000000002870: d65d5003 01aa0680
	v_add_co_u32 v74, vcc_lo, s33, v40                         // 000000002878: d7006a4a 02025021
	s_wait_alu depctr_va_vcc(0)                                // 000000002880: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, s39, v39, vcc_lo            // 000000002884: d5207c4b 01aa4e27
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_3)// 00000000288c: bf8701b3
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002890: d7385003 02020688
	v_cmp_gt_u64_e32 vcc_lo, s[18:19], v[70:71]                // 000000002898: 7cb88c12
	v_or_b32_e32 v72, 3, v74                                   // 00000000289c: 38909483
	v_or_b16 v69.h, v3.l, v3.h op_sel:[0,1,1]                  // 0000000028a0: d7635045 02020703
	s_and_b32 s3, s2, vcc_lo                                   // 0000000028a8: 8b036a02
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 0000000028ac: bf8701d1
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[66:67], v[68:69], 0  // 0000000028b0: cc464000 1a028942
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028b8: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v74, s3                          // 0000000028bc: d5010042 000e9480
	v_cndmask_b32_e64 v67, 0, v75, s3                          // 0000000028c4: d5010043 000e9680
	v_or_b32_e32 v69, 2, v74                                   // 0000000028cc: 388a9482
	v_add_co_u32 v66, s4, s22, v66                             // 0000000028d0: d7000442 02028416
	s_wait_alu depctr_va_sdst(0)                               // 0000000028d8: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000028dc: bf870003
	v_add_co_ci_u32_e64 v67, null, s23, v67, s4                // 0000000028e0: d5207c43 00128617
	global_load_d16_u8 v66, v[66:67], off                      // 0000000028e8: ee07807c 00000042 00000042
	v_or_b32_e32 v67, 1, v74                                   // 0000000028f4: 38869481
	s_wait_loadcnt 0x0                                         // 0000000028f8: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, s3                          // 0000000028fc: d65d0042 000e8480
	v_cmp_gt_u64_e64 s3, s[36:37], v[70:71]                    // 000000002904: d45c0003 02028c24
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 00000000290c: bf870152
	v_and_b16 v66.l, 0xff, v66.l                               // 000000002910: d7620042 020284ff 000000ff
	s_and_b32 s4, s2, s3                                       // 00000000291c: 8b040302
	s_wait_alu depctr_sa_sdst(0)                               // 000000002920: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v67, s4                          // 000000002924: d5010043 00128680
	v_cndmask_b32_e64 v68, 0, v75, s4                          // 00000000292c: d5010044 00129680
	v_add_co_u32 v67, s5, s22, v67                             // 000000002934: d7000543 02028616
	s_wait_alu depctr_va_sdst(0)                               // 00000000293c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002940: bf870002
	v_add_co_ci_u32_e64 v68, null, s23, v68, s5                // 000000002944: d5207c44 00168817
	global_load_d16_hi_u8 v66, v[67:68], off                   // 00000000294c: ee08407c 00000042 00000043
	v_or_b32_e32 v67, 2, v70                                   // 000000002958: 38868c82
	v_mov_b32_e32 v68, s39                                     // 00000000295c: 7e880227
	s_wait_loadcnt 0x0                                         // 000000002960: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, s4                          // 000000002964: d65d5042 00128480
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000296c: bf870112
	v_cmp_gt_u64_e64 s4, s[18:19], v[67:68]                    // 000000002970: d45c0004 02028612
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000002978: d7385042 02028488
	s_and_b32 s5, s2, s4                                       // 000000002980: 8b050402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002984: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v69, s5                          // 000000002988: d5010043 00168a80
	v_cndmask_b32_e64 v68, 0, v75, s5                          // 000000002990: d5010044 00169680
	v_mov_b32_e32 v69, s39                                     // 000000002998: 7e8a0227
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000299c: bf8701a3
	v_add_co_u32 v67, s6, s22, v67                             // 0000000029a0: d7000643 02028616
	s_wait_alu depctr_va_sdst(0)                               // 0000000029a8: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s23, v68, s6                // 0000000029ac: d5207c44 001a8817
	global_load_d16_u8 v67, v[67:68], off                      // 0000000029b4: ee07807c 00000043 00000043
	v_or_b32_e32 v68, 3, v70                                   // 0000000029c0: 38888c83
	s_wait_loadcnt 0x0                                         // 0000000029c4: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, s5                          // 0000000029c8: d65d0043 00168680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000029d0: bf870112
	v_cmp_gt_u64_e64 s5, s[18:19], v[68:69]                    // 0000000029d4: d45c0005 02028812
	v_and_b16 v67.l, 0xff, v67.l                               // 0000000029dc: d7620043 020286ff 000000ff
	s_and_b32 s6, s2, s5                                       // 0000000029e8: 8b060502
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029ec: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v72, s6                          // 0000000029f0: d5010044 001a9080
	v_cndmask_b32_e64 v69, 0, v75, s6                          // 0000000029f8: d5010045 001a9680
	v_or_b32_e32 v72, 4, v74                                   // 000000002a00: 38909484
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002a04: bf8701a3
	v_add_co_u32 v68, s7, s22, v68                             // 000000002a08: d7000744 02028816
	s_wait_alu depctr_va_sdst(0)                               // 000000002a10: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s23, v69, s7                // 000000002a14: d5207c45 001e8a17
	global_load_d16_hi_u8 v67, v[68:69], off                   // 000000002a1c: ee08407c 00000043 00000044
	v_or_b32_e32 v68, 4, v70                                   // 000000002a28: 38888c84
	v_mov_b32_e32 v69, s39                                     // 000000002a2c: 7e8a0227
	s_wait_loadcnt 0x0                                         // 000000002a30: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, s6                          // 000000002a34: d65d5043 001a8680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002a3c: bf870112
	v_cmp_gt_u64_e64 s6, s[18:19], v[68:69]                    // 000000002a40: d45c0006 02028812
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 000000002a48: d7385043 02028688
	s_and_b32 s7, s2, s6                                       // 000000002a50: 8b070602
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a54: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v72, s7                          // 000000002a58: d5010044 001e9080
	v_cndmask_b32_e64 v69, 0, v75, s7                          // 000000002a60: d5010045 001e9680
	v_or_b32_e32 v72, 5, v70                                   // 000000002a68: 38908c85
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002a6c: bf8701a3
	v_add_co_u32 v68, s8, s22, v68                             // 000000002a70: d7000844 02028816
	s_wait_alu depctr_va_sdst(0)                               // 000000002a78: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s23, v69, s8                // 000000002a7c: d5207c45 00228a17
	global_load_d16_u8 v68, v[68:69], off                      // 000000002a84: ee07807c 00000044 00000044
	v_or_b32_e32 v69, 5, v74                                   // 000000002a90: 388a9485
	s_wait_loadcnt 0x0                                         // 000000002a94: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, s7                          // 000000002a98: d65d0044 001e8880
	v_cmp_gt_u64_e64 s7, s[18:19], v[72:73]                    // 000000002aa0: d45c0007 02029012
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002aa8: bf870152
	v_and_b16 v68.l, 0xff, v68.l                               // 000000002aac: d7620044 020288ff 000000ff
	s_and_b32 s8, s2, s7                                       // 000000002ab8: 8b080702
	s_wait_alu depctr_sa_sdst(0)                               // 000000002abc: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s8                          // 000000002ac0: d5010045 00228a80
	v_cndmask_b32_e64 v73, 0, v75, s8                          // 000000002ac8: d5010049 00229680
	v_add_co_u32 v72, s9, s22, v69                             // 000000002ad0: d7000948 02028a16
	s_wait_alu depctr_va_sdst(0)                               // 000000002ad8: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002adc: bf870002
	v_add_co_ci_u32_e64 v73, null, s23, v73, s9                // 000000002ae0: d5207c49 00269217
	v_or_b32_e32 v69, 6, v74                                   // 000000002ae8: 388a9486
	global_load_d16_hi_u8 v68, v[72:73], off                   // 000000002aec: ee08407c 00000044 00000048
	v_or_b32_e32 v72, 6, v70                                   // 000000002af8: 38908c86
	v_mov_b32_e32 v73, s39                                     // 000000002afc: 7e920227
	v_or_b32_e32 v70, 7, v70                                   // 000000002b00: 388c8c87
	s_wait_loadcnt 0x0                                         // 000000002b04: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, s8                          // 000000002b08: d65d5044 00228880
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002b10: bf870113
	v_cmp_gt_u64_e64 s8, s[18:19], v[72:73]                    // 000000002b14: d45c0008 02029012
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 000000002b1c: d7385044 02028888
	s_and_b32 s9, s2, s8                                       // 000000002b24: 8b090802
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b28: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s9                          // 000000002b2c: d5010045 00268a80
	v_cndmask_b32_e64 v73, 0, v75, s9                          // 000000002b34: d5010049 00269680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b3c: bf870122
	v_add_co_u32 v72, s10, s22, v69                            // 000000002b40: d7000a48 02028a16
	s_wait_alu depctr_va_sdst(0)                               // 000000002b48: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s23, v73, s10               // 000000002b4c: d5207c49 002a9217
	global_load_d16_u8 v69, v[72:73], off                      // 000000002b54: ee07807c 00000045 00000048
	v_or_b32_e32 v72, 7, v74                                   // 000000002b60: 38909487
	s_wait_loadcnt 0x0                                         // 000000002b64: bfc00000
	v_cndmask_b16 v69.l, 0, v69.l, s9                          // 000000002b68: d65d0045 00268a80
	v_cmp_gt_u64_e64 s9, s[18:19], v[70:71]                    // 000000002b70: d45c0009 02028c12
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002b78: bf870152
	v_and_b16 v69.l, 0xff, v69.l                               // 000000002b7c: d7620045 02028aff 000000ff
	s_and_b32 s10, s2, s9                                      // 000000002b88: 8b0a0902
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b8c: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v72, s10                         // 000000002b90: d5010046 002a9080
	v_cndmask_b32_e64 v71, 0, v75, s10                         // 000000002b98: d5010047 002a9680
	v_add_co_u32 v70, s11, s22, v70                            // 000000002ba0: d7000b46 02028c16
	s_wait_alu depctr_va_sdst(0)                               // 000000002ba8: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002bac: bf870002
	v_add_co_ci_u32_e64 v71, null, s23, v71, s11               // 000000002bb0: d5207c47 002e8e17
	global_load_d16_hi_u8 v69, v[70:71], off                   // 000000002bb8: ee08407c 00000045 00000046
	v_or_b16 v70.l, v66.l, v66.h op_sel:[0,1,0]                // 000000002bc4: d7631046 02028542
	v_or_b16 v70.h, v67.l, v67.h op_sel:[0,1,1]                // 000000002bcc: d7635046 02028743
	v_or_b16 v71.l, v68.l, v68.h op_sel:[0,1,0]                // 000000002bd4: d7631047 02028944
	s_wait_loadcnt 0x0                                         // 000000002bdc: bfc00000
	v_cndmask_b16 v69.h, 0, v69.h, s10                         // 000000002be0: d65d5045 002a8a80
	v_add_co_u32 v74, s10, s33, v43                            // 000000002be8: d7000a4a 02025621
	s_wait_alu depctr_va_sdst(0)                               // 000000002bf0: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s39, v41, s10               // 000000002bf4: d5207c4b 002a5227
	s_and_b32 s10, s1, vcc_lo                                  // 000000002bfc: 8b0a6a01
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 000000002c00: d7385045 02028a88
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c08: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v74, s10                         // 000000002c0c: d5010042 002a9480
	v_cndmask_b32_e64 v67, 0, v75, s10                         // 000000002c14: d5010043 002a9680
	s_and_b32 vcc_lo, s0, vcc_lo                               // 000000002c1c: 8b6a6a00
	v_or_b16 v71.h, v69.l, v69.h op_sel:[0,1,1]                // 000000002c20: d7635047 02028b45
	s_delay_alu instid0(valu_dep_3)                            // 000000002c28: bf870003
	v_add_co_u32 v66, s11, s26, v66                            // 000000002c2c: d7000b42 0202841a
	s_wait_alu depctr_va_sdst(0)                               // 000000002c34: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s27, v67, s11               // 000000002c38: d5207c43 002e861b
	global_load_d16_u8 v66, v[66:67], off                      // 000000002c40: ee07807c 00000042 00000042
	v_or_b32_e32 v67, 1, v74                                   // 000000002c4c: 38869481
	s_wait_loadcnt 0x0                                         // 000000002c50: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, s10                         // 000000002c54: d65d0042 002a8480
	s_and_b32 s10, s1, s3                                      // 000000002c5c: 8b0a0301
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c60: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v67, s10                         // 000000002c64: d5010043 002a8680
	v_cndmask_b32_e64 v68, 0, v75, s10                         // 000000002c6c: d5010044 002a9680
	v_and_b16 v66.l, 0xff, v66.l                               // 000000002c74: d7620042 020284ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002c80: bf8701a3
	v_add_co_u32 v67, s11, s26, v67                            // 000000002c84: d7000b43 0202861a
	s_wait_alu depctr_va_sdst(0)                               // 000000002c8c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s27, v68, s11               // 000000002c90: d5207c44 002e881b
	global_load_d16_hi_u8 v66, v[67:68], off                   // 000000002c98: ee08407c 00000042 00000043
	v_or_b32_e32 v67, 2, v74                                   // 000000002ca4: 38869482
	s_wait_loadcnt 0x0                                         // 000000002ca8: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, s10                         // 000000002cac: d65d5042 002a8480
	s_and_b32 s10, s1, s4                                      // 000000002cb4: 8b0a0401
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cb8: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v67, s10                         // 000000002cbc: d5010043 002a8680
	v_cndmask_b32_e64 v68, 0, v75, s10                         // 000000002cc4: d5010044 002a9680
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000002ccc: d7385042 02028488
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002cd4: bf8701a3
	v_add_co_u32 v67, s11, s26, v67                            // 000000002cd8: d7000b43 0202861a
	s_wait_alu depctr_va_sdst(0)                               // 000000002ce0: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s27, v68, s11               // 000000002ce4: d5207c44 002e881b
	global_load_d16_u8 v67, v[67:68], off                      // 000000002cec: ee07807c 00000043 00000043
	v_or_b32_e32 v68, 3, v74                                   // 000000002cf8: 38889483
	s_wait_loadcnt 0x0                                         // 000000002cfc: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, s10                         // 000000002d00: d65d0043 002a8680
	s_and_b32 s10, s1, s5                                      // 000000002d08: 8b0a0501
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d0c: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v68, s10                         // 000000002d10: d5010044 002a8880
	v_cndmask_b32_e64 v69, 0, v75, s10                         // 000000002d18: d5010045 002a9680
	v_and_b16 v67.l, 0xff, v67.l                               // 000000002d20: d7620043 020286ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002d2c: bf8701a3
	v_add_co_u32 v68, s11, s26, v68                            // 000000002d30: d7000b44 0202881a
	s_wait_alu depctr_va_sdst(0)                               // 000000002d38: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s27, v69, s11               // 000000002d3c: d5207c45 002e8a1b
	global_load_d16_hi_u8 v67, v[68:69], off                   // 000000002d44: ee08407c 00000043 00000044
	v_or_b32_e32 v68, 4, v74                                   // 000000002d50: 38889484
	s_wait_loadcnt 0x0                                         // 000000002d54: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, s10                         // 000000002d58: d65d5043 002a8680
	s_and_b32 s10, s1, s6                                      // 000000002d60: 8b0a0601
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d64: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v68, s10                         // 000000002d68: d5010044 002a8880
	v_cndmask_b32_e64 v69, 0, v75, s10                         // 000000002d70: d5010045 002a9680
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 000000002d78: d7385043 02028688
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002d80: bf8701a3
	v_add_co_u32 v68, s11, s26, v68                            // 000000002d84: d7000b44 0202881a
	s_wait_alu depctr_va_sdst(0)                               // 000000002d8c: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s27, v69, s11               // 000000002d90: d5207c45 002e8a1b
	global_load_d16_u8 v68, v[68:69], off                      // 000000002d98: ee07807c 00000044 00000044
	v_or_b32_e32 v69, 5, v74                                   // 000000002da4: 388a9485
	s_wait_loadcnt 0x0                                         // 000000002da8: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, s10                         // 000000002dac: d65d0044 002a8880
	s_and_b32 s10, s1, s7                                      // 000000002db4: 8b0a0701
	s_wait_alu depctr_sa_sdst(0)                               // 000000002db8: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002dbc: d5010045 002a8a80
	v_cndmask_b32_e64 v73, 0, v75, s10                         // 000000002dc4: d5010049 002a9680
	v_and_b16 v68.l, 0xff, v68.l                               // 000000002dcc: d7620044 020288ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002dd8: bf8701a3
	v_add_co_u32 v72, s11, s26, v69                            // 000000002ddc: d7000b48 02028a1a
	s_wait_alu depctr_va_sdst(0)                               // 000000002de4: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s27, v73, s11               // 000000002de8: d5207c49 002e921b
	v_or_b32_e32 v69, 6, v74                                   // 000000002df0: 388a9486
	global_load_d16_hi_u8 v68, v[72:73], off                   // 000000002df4: ee08407c 00000044 00000048
	s_wait_loadcnt 0x0                                         // 000000002e00: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, s10                         // 000000002e04: d65d5044 002a8880
	s_and_b32 s10, s1, s8                                      // 000000002e0c: 8b0a0801
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e10: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002e14: d5010045 002a8a80
	v_cndmask_b32_e64 v73, 0, v75, s10                         // 000000002e1c: d5010049 002a9680
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 000000002e24: d7385044 02028888
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002e2c: bf8701a3
	v_add_co_u32 v72, s11, s26, v69                            // 000000002e30: d7000b48 02028a1a
	s_wait_alu depctr_va_sdst(0)                               // 000000002e38: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s27, v73, s11               // 000000002e3c: d5207c49 002e921b
	global_load_d16_u8 v69, v[72:73], off                      // 000000002e44: ee07807c 00000045 00000048
	v_or_b32_e32 v72, 7, v74                                   // 000000002e50: 38909487
	s_wait_loadcnt 0x0                                         // 000000002e54: bfc00000
	v_cndmask_b16 v69.l, 0, v69.l, s10                         // 000000002e58: d65d0045 002a8a80
	s_and_b32 s10, s1, s9                                      // 000000002e60: 8b0a0901
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e64: bf88ff9e
	v_cndmask_b32_e64 v72, 0, v72, s10                         // 000000002e68: d5010048 002a9080
	v_cndmask_b32_e64 v73, 0, v75, s10                         // 000000002e70: d5010049 002a9680
	v_and_b16 v69.l, 0xff, v69.l                               // 000000002e78: d7620045 02028aff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002e84: bf8701a3
	v_add_co_u32 v72, s11, s26, v72                            // 000000002e88: d7000b48 0202901a
	s_wait_alu depctr_va_sdst(0)                               // 000000002e90: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s27, v73, s11               // 000000002e94: d5207c49 002e921b
	global_load_d16_hi_u8 v69, v[72:73], off                   // 000000002e9c: ee08407c 00000045 00000048
	v_or_b16 v72.l, v66.l, v66.h op_sel:[0,1,0]                // 000000002ea8: d7631048 02028542
	v_or_b16 v72.h, v67.l, v67.h op_sel:[0,1,1]                // 000000002eb0: d7635048 02028743
	v_or_b16 v73.l, v68.l, v68.h op_sel:[0,1,0]                // 000000002eb8: d7631049 02028944
	s_wait_loadcnt 0x0                                         // 000000002ec0: bfc00000
	v_cndmask_b16 v69.h, 0, v69.h, s10                         // 000000002ec4: d65d5045 002a8a80
	v_add_co_u32 v76, s10, s33, v45                            // 000000002ecc: d7000a4c 02025a21
	s_wait_alu depctr_va_sdst(0)                               // 000000002ed4: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s39, v44, s10               // 000000002ed8: d5207c4d 002a5827
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002ee0: bf870113
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 000000002ee4: d7385045 02028a88
	v_dual_cndmask_b32 v66, 0, v76 :: v_dual_cndmask_b32 v67, 0, v77// 000000002eec: ca529880 42429a80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002ef4: bf870112
	v_or_b16 v73.h, v69.l, v69.h op_sel:[0,1,1]                // 000000002ef8: d7635049 02028b45
	v_add_co_u32 v66, s10, s26, v66                            // 000000002f00: d7000a42 0202841a
	s_wait_alu depctr_va_sdst(0)                               // 000000002f08: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002f0c: bf870193
	v_add_co_ci_u32_e64 v67, null, s27, v67, s10               // 000000002f10: d5207c43 002a861b
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[70:71], v[72:73], v[8:15]// 000000002f18: cc464008 1c229146
	global_load_d16_u8 v66, v[66:67], off                      // 000000002f20: ee07807c 00000042 00000042
	v_or_b32_e32 v67, 1, v76                                   // 000000002f2c: 38869881
	s_wait_loadcnt 0x0                                         // 000000002f30: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, vcc_lo                      // 000000002f34: d65d0042 01aa8480
	s_and_b32 vcc_lo, s0, s3                                   // 000000002f3c: 8b6a0300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f40: bf88ff9e
	v_dual_cndmask_b32 v67, 0, v67 :: v_dual_cndmask_b32 v68, 0, v77// 000000002f44: ca528680 43449a80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002f4c: bf870112
	v_and_b16 v66.l, 0xff, v66.l                               // 000000002f50: d7620042 020284ff 000000ff
	v_add_co_u32 v67, s3, s26, v67                             // 000000002f5c: d7000343 0202861a
	s_wait_alu depctr_va_sdst(0)                               // 000000002f64: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002f68: bf870003
	v_add_co_ci_u32_e64 v68, null, s27, v68, s3                // 000000002f6c: d5207c44 000e881b
	global_load_d16_hi_u8 v66, v[67:68], off                   // 000000002f74: ee08407c 00000042 00000043
	v_or_b32_e32 v67, 2, v76                                   // 000000002f80: 38869882
	s_wait_loadcnt 0x0                                         // 000000002f84: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, vcc_lo                      // 000000002f88: d65d5042 01aa8480
	s_and_b32 vcc_lo, s0, s4                                   // 000000002f90: 8b6a0400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f94: bf88ff9e
	v_dual_cndmask_b32 v67, 0, v67 :: v_dual_cndmask_b32 v68, 0, v77// 000000002f98: ca528680 43449a80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002fa0: bf870112
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000002fa4: d7385042 02028488
	v_add_co_u32 v67, s3, s26, v67                             // 000000002fac: d7000343 0202861a
	s_wait_alu depctr_va_sdst(0)                               // 000000002fb4: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002fb8: bf870003
	v_add_co_ci_u32_e64 v68, null, s27, v68, s3                // 000000002fbc: d5207c44 000e881b
	global_load_d16_u8 v67, v[67:68], off                      // 000000002fc4: ee07807c 00000043 00000043
	v_or_b32_e32 v68, 3, v76                                   // 000000002fd0: 38889883
	s_wait_loadcnt 0x0                                         // 000000002fd4: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, vcc_lo                      // 000000002fd8: d65d0043 01aa8680
	s_and_b32 vcc_lo, s0, s5                                   // 000000002fe0: 8b6a0500
	s_lshr_b64 s[4:5], s[38:39], 5                             // 000000002fe4: 85848526
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fe8: bf88ff9e
	v_dual_cndmask_b32 v68, 0, v68 :: v_dual_cndmask_b32 v69, 0, v77// 000000002fec: ca528880 44449a80
	v_and_b16 v67.l, 0xff, v67.l                               // 000000002ff4: d7620043 020286ff 000000ff
	s_mul_u64 s[4:5], s[4:5], s[14:15]                         // 000000003000: aa840e04
	s_delay_alu instid0(valu_dep_2)                            // 000000003004: bf870002
	v_add_co_u32 v68, s3, s26, v68                             // 000000003008: d7000344 0202881a
	s_wait_alu depctr_va_sdst(0)                               // 000000003010: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s27, v69, s3                // 000000003014: d5207c45 000e8a1b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000301c: bf88ff9e
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000003020: 84848204
	s_wait_alu depctr_sa_sdst(0)                               // 000000003024: bf88ff9e
	s_add_nc_u64 s[4:5], s[28:29], s[4:5]                      // 000000003028: a984041c
	global_load_d16_hi_u8 v67, v[68:69], off                   // 00000000302c: ee08407c 00000043 00000044
	v_or_b32_e32 v68, 4, v76                                   // 000000003038: 38889884
	s_wait_loadcnt 0x0                                         // 00000000303c: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, vcc_lo                      // 000000003040: d65d5043 01aa8680
	s_and_b32 vcc_lo, s0, s6                                   // 000000003048: 8b6a0600
	s_wait_alu depctr_sa_sdst(0)                               // 00000000304c: bf88ff9e
	v_dual_cndmask_b32 v68, 0, v68 :: v_dual_cndmask_b32 v69, 0, v77// 000000003050: ca528880 44449a80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003058: bf870112
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 00000000305c: d7385043 02028688
	v_add_co_u32 v68, s3, s26, v68                             // 000000003064: d7000344 0202881a
	s_wait_alu depctr_va_sdst(0)                               // 00000000306c: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003070: bf870003
	v_add_co_ci_u32_e64 v69, null, s27, v69, s3                // 000000003074: d5207c45 000e8a1b
	global_load_d16_u8 v68, v[68:69], off                      // 00000000307c: ee07807c 00000044 00000044
	v_or_b32_e32 v69, 5, v76                                   // 000000003088: 388a9885
	s_wait_loadcnt 0x0                                         // 00000000308c: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, vcc_lo                      // 000000003090: d65d0044 01aa8880
	s_and_b32 vcc_lo, s0, s7                                   // 000000003098: 8b6a0700
	s_lshr_b64 s[6:7], s[38:39], 3                             // 00000000309c: 85868326
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030a0: bf88ff9e
	v_cndmask_b32_e32 v69, 0, v69, vcc_lo                      // 0000000030a4: 028a8a80
	v_cndmask_b32_e32 v75, 0, v77, vcc_lo                      // 0000000030a8: 02969a80
	v_and_b16 v68.l, 0xff, v68.l                               // 0000000030ac: d7620044 020288ff 000000ff
	s_add_nc_u64 s[38:39], s[38:39], 32                        // 0000000030b8: a9a6a026
	s_delay_alu instid0(valu_dep_3)                            // 0000000030bc: bf870003
	v_add_co_u32 v74, s3, s26, v69                             // 0000000030c0: d700034a 02028a1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c8: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s27, v75, s3                // 0000000030cc: d5207c4b 000e961b
	v_or_b32_e32 v69, 6, v76                                   // 0000000030d4: 388a9886
	global_load_d16_hi_u8 v68, v[74:75], off                   // 0000000030d8: ee08407c 00000044 0000004a
	s_wait_loadcnt 0x0                                         // 0000000030e4: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, vcc_lo                      // 0000000030e8: d65d5044 01aa8880
	s_and_b32 vcc_lo, s0, s8                                   // 0000000030f0: 8b6a0800
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030f4: bf88ff9e
	v_cndmask_b32_e32 v69, 0, v69, vcc_lo                      // 0000000030f8: 028a8a80
	v_cndmask_b32_e32 v75, 0, v77, vcc_lo                      // 0000000030fc: 02969a80
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 000000003100: d7385044 02028888
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003108: bf8701a3
	v_add_co_u32 v74, s3, s26, v69                             // 00000000310c: d700034a 02028a1a
	s_wait_alu depctr_va_sdst(0)                               // 000000003114: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s27, v75, s3                // 000000003118: d5207c4b 000e961b
	global_load_d16_u8 v69, v[74:75], off                      // 000000003120: ee07807c 00000045 0000004a
	v_or_b32_e32 v74, 7, v76                                   // 00000000312c: 38949887
	s_wait_loadcnt 0x0                                         // 000000003130: bfc00000
	v_cndmask_b16 v69.l, 0, v69.l, vcc_lo                      // 000000003134: d65d0045 01aa8a80
	s_and_b32 vcc_lo, s0, s9                                   // 00000000313c: 8b6a0900
	s_wait_alu depctr_sa_sdst(0)                               // 000000003140: bf88ff9e
	v_dual_cndmask_b32 v74, 0, v74 :: v_dual_cndmask_b32 v75, 0, v77// 000000003144: ca529480 4a4a9a80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000314c: bf870112
	v_and_b16 v69.l, 0xff, v69.l                               // 000000003150: d7620045 02028aff 000000ff
	v_add_co_u32 v74, s3, s26, v74                             // 00000000315c: d700034a 0202941a
	s_wait_alu depctr_va_sdst(0)                               // 000000003164: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_1)// 000000003168: bf8700d3
	v_add_co_ci_u32_e64 v75, null, s27, v75, s3                // 00000000316c: d5207c4b 000e961b
	v_cmp_lt_u64_e64 s3, s[38:39], s[18:19]                    // 000000003174: d4590003 02002426
	global_load_d16_hi_u8 v69, v[74:75], off                   // 00000000317c: ee08407c 00000045 0000004a
	s_wait_loadcnt 0x0                                         // 000000003188: bfc00000
	v_cndmask_b16 v69.h, 0, v69.h, vcc_lo                      // 00000000318c: d65d5045 01aa8a80
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 000000003194: d7385045 02028a88
	s_delay_alu instid0(valu_dep_1)                            // 00000000319c: bf870001
	v_or_b16 v69.h, v69.l, v69.h op_sel:[0,1,1]                // 0000000031a0: d7635045 02028b45
	v_or_b16 v69.l, v68.l, v68.h op_sel:[0,1,0]                // 0000000031a8: d7631045 02028944
	v_or_b16 v68.l, v66.l, v66.h op_sel:[0,1,0]                // 0000000031b0: d7631044 02028542
	v_add_co_u32 v66, vcc_lo, v48, s6                          // 0000000031b8: d7006a42 02000d30
	v_or_b16 v68.h, v67.l, v67.h op_sel:[0,1,1]                // 0000000031c0: d7635044 02028743
	s_wait_alu depctr_va_vcc(0)                                // 0000000031c8: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s7, v49, vcc_lo             // 0000000031cc: d5207c43 01aa6207
	s_delay_alu instid0(valu_dep_2)                            // 0000000031d4: bf870002
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[70:71], v[68:69], v[0:7]// 0000000031d8: cc464000 1c028946
	global_load_b32 v66, v[66:67], off                         // 0000000031e0: ee05007c 00000042 00000042
	v_add_co_u32 v67, vcc_lo, s4, v24                          // 0000000031ec: d7006a43 02023004
	s_wait_alu depctr_va_vcc(0)                                // 0000000031f4: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s5, v25, vcc_lo             // 0000000031f8: d5207c44 01aa3205
	global_load_b32 v69, v[67:68], off                         // 000000003200: ee05007c 00000045 00000043
	s_wait_loadcnt 0x0                                         // 00000000320c: bfc00000
	v_mul_f32_e32 v67, v66, v69                                // 000000003210: 10868b42
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003214: bf8701c1
	v_mul_f32_e32 v8, v8, v67                                  // 000000003218: 10108708
	v_add_co_u32 v67, vcc_lo, v50, s6                          // 00000000321c: d7006a43 02000d32
	s_wait_alu depctr_va_vcc(0)                                // 000000003224: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s7, v51, vcc_lo             // 000000003228: d5207c44 01aa6607
	v_add_f32_e32 v34, v34, v8                                 // 000000003230: 06441122
	global_load_b32 v8, v[67:68], off                          // 000000003234: ee05007c 00000008 00000043
	s_wait_loadcnt 0x0                                         // 000000003240: bfc00000
	v_mul_f32_e32 v67, v69, v8                                 // 000000003244: 10861145
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003248: bf8701c1
	v_mul_f32_e32 v9, v9, v67                                  // 00000000324c: 10128709
	v_add_co_u32 v67, vcc_lo, v52, s6                          // 000000003250: d7006a43 02000d34
	s_wait_alu depctr_va_vcc(0)                                // 000000003258: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s7, v54, vcc_lo             // 00000000325c: d5207c44 01aa6c07
	v_add_f32_e32 v57, v57, v9                                 // 000000003264: 06721339
	global_load_b32 v9, v[67:68], off                          // 000000003268: ee05007c 00000009 00000043
	s_wait_loadcnt 0x0                                         // 000000003274: bfc00000
	v_mul_f32_e32 v67, v69, v9                                 // 000000003278: 10861345
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 00000000327c: bf8701c1
	v_mul_f32_e32 v10, v10, v67                                // 000000003280: 1014870a
	v_add_co_u32 v67, vcc_lo, v55, s6                          // 000000003284: d7006a43 02000d37
	s_wait_alu depctr_va_vcc(0)                                // 00000000328c: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s7, v56, vcc_lo             // 000000003290: d5207c44 01aa7007
	v_add_f32_e32 v53, v53, v10                                // 000000003298: 066a1535
	global_load_b32 v10, v[67:68], off                         // 00000000329c: ee05007c 0000000a 00000043
	s_wait_loadcnt 0x0                                         // 0000000032a8: bfc00000
	v_mul_f32_e32 v67, v69, v10                                // 0000000032ac: 10861545
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 0000000032b0: bf8701c1
	v_mul_f32_e32 v11, v11, v67                                // 0000000032b4: 1016870b
	v_add_co_u32 v67, vcc_lo, v58, s6                          // 0000000032b8: d7006a43 02000d3a
	s_wait_alu depctr_va_vcc(0)                                // 0000000032c0: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s7, v59, vcc_lo             // 0000000032c4: d5207c44 01aa7607
	v_add_f32_e32 v47, v47, v11                                // 0000000032cc: 065e172f
	global_load_b32 v11, v[67:68], off                         // 0000000032d0: ee05007c 0000000b 00000043
	s_wait_loadcnt 0x0                                         // 0000000032dc: bfc00000
	v_mul_f32_e32 v67, v69, v11                                // 0000000032e0: 10861745
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 0000000032e4: bf8701c1
	v_mul_f32_e32 v12, v12, v67                                // 0000000032e8: 1018870c
	v_add_co_u32 v67, vcc_lo, v60, s6                          // 0000000032ec: d7006a43 02000d3c
	s_wait_alu depctr_va_vcc(0)                                // 0000000032f4: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s7, v61, vcc_lo             // 0000000032f8: d5207c44 01aa7a07
	v_add_f32_e32 v46, v46, v12                                // 000000003300: 065c192e
	global_load_b32 v67, v[67:68], off                         // 000000003304: ee05007c 00000043 00000043
	s_wait_loadcnt 0x0                                         // 000000003310: bfc00000
	v_mul_f32_e32 v12, v69, v67                                // 000000003314: 10188745
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003318: bf870091
	v_mul_f32_e32 v12, v13, v12                                // 00000000331c: 1018190d
	v_add_f32_e32 v42, v42, v12                                // 000000003320: 0654192a
	v_add_co_u32 v12, vcc_lo, v62, s6                          // 000000003324: d7006a0c 02000d3e
	s_wait_alu depctr_va_vcc(0)                                // 00000000332c: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s7, v63, vcc_lo             // 000000003330: d5207c0d 01aa7e07
	global_load_b32 v68, v[12:13], off                         // 000000003338: ee05007c 00000044 0000000c
	s_wait_loadcnt 0x0                                         // 000000003344: bfc00000
	v_mul_f32_e32 v12, v69, v68                                // 000000003348: 10188945
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000334c: bf870091
	v_mul_f32_e32 v12, v14, v12                                // 000000003350: 1018190e
	v_add_f32_e32 v38, v38, v12                                // 000000003354: 064c1926
	v_add_co_u32 v12, vcc_lo, v64, s6                          // 000000003358: d7006a0c 02000d40
	s_wait_alu depctr_va_vcc(0)                                // 000000003360: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s7, v65, vcc_lo             // 000000003364: d5207c0d 01aa8207
	global_load_b32 v14, v[12:13], off                         // 00000000336c: ee05007c 0000000e 0000000c
	s_wait_loadcnt 0x0                                         // 000000003378: bfc00000
	v_mul_f32_e32 v12, v69, v14                                // 00000000337c: 10181d45
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003380: bf870091
	v_mul_f32_e32 v12, v15, v12                                // 000000003384: 1018190f
	v_add_f32_e32 v37, v37, v12                                // 000000003388: 064a1925
	v_add_co_u32 v12, vcc_lo, s4, v26                          // 00000000338c: d7006a0c 02023404
	s_wait_alu depctr_va_vcc(0)                                // 000000003394: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s5, v27, vcc_lo             // 000000003398: d5207c0d 01aa3605
	s_and_b32 vcc_lo, exec_lo, s3                              // 0000000033a0: 8b6a037e
	global_load_b32 v12, v[12:13], off                         // 0000000033a4: ee05007c 0000000c 0000000c
	s_wait_loadcnt 0x0                                         // 0000000033b0: bfc00000
	v_mul_f32_e32 v13, v66, v12                                // 0000000033b4: 101a1942
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000033b8: bf870091
	v_mul_f32_e32 v0, v0, v13                                  // 0000000033bc: 10001b00
	v_add_f32_e32 v36, v36, v0                                 // 0000000033c0: 06480124
	v_mul_f32_e32 v0, v8, v12                                  // 0000000033c4: 10001908
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000033c8: bf870091
	v_mul_f32_e32 v0, v1, v0                                   // 0000000033cc: 10000101
	v_add_f32_e32 v35, v35, v0                                 // 0000000033d0: 06460123
	v_mul_f32_e32 v0, v9, v12                                  // 0000000033d4: 10001909
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000033d8: bf870091
	v_mul_f32_e32 v0, v2, v0                                   // 0000000033dc: 10000102
	v_add_f32_e32 v33, v33, v0                                 // 0000000033e0: 06420121
	v_mul_f32_e32 v0, v10, v12                                 // 0000000033e4: 1000190a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000033e8: bf870091
	v_mul_f32_e32 v0, v3, v0                                   // 0000000033ec: 10000103
	v_add_f32_e32 v32, v32, v0                                 // 0000000033f0: 06400120
	v_mul_f32_e32 v0, v11, v12                                 // 0000000033f4: 1000190b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000033f8: bf870091
	v_mul_f32_e32 v0, v4, v0                                   // 0000000033fc: 10000104
	v_add_f32_e32 v31, v31, v0                                 // 000000003400: 063e011f
	v_mul_f32_e32 v0, v67, v12                                 // 000000003404: 10001943
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003408: bf870091
	v_mul_f32_e32 v0, v5, v0                                   // 00000000340c: 10000105
	v_add_f32_e32 v30, v30, v0                                 // 000000003410: 063c011e
	v_mul_f32_e32 v0, v68, v12                                 // 000000003414: 10001944
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003418: bf870091
	v_mul_f32_e32 v0, v6, v0                                   // 00000000341c: 10000106
	v_add_f32_e32 v29, v29, v0                                 // 000000003420: 063a011d
	v_mul_f32_e32 v0, v14, v12                                 // 000000003424: 1000190e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003428: bf870091
	v_mul_f32_e32 v0, v7, v0                                   // 00000000342c: 10000107
	v_add_f32_e32 v19, v19, v0                                 // 000000003430: 06260113
	s_wait_alu depctr_sa_sdst(0)                               // 000000003434: bf88ff9e
	s_cbranch_vccnz 64204                                      // 000000003438: bfa4facc <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x46c>
	v_mul_lo_u32 v4, s15, v22                                  // 00000000343c: d72c0004 02022c0f
	v_mul_lo_u32 v5, s14, v23                                  // 000000003444: d72c0005 02022e0e
	v_mad_co_u64_u32 v[0:1], null, s14, v22, 0                 // 00000000344c: d6fe7c00 02022c0e
	v_sub_co_u32 v2, vcc_lo, s12, v22                          // 000000003454: d7016a02 02022c0c
	s_wait_alu depctr_va_vcc(0)                                // 00000000345c: bf88ff9d
	v_sub_co_ci_u32_e64 v3, null, s13, v23, vcc_lo             // 000000003460: d5217c03 01aa2e0d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003468: bf870211
	v_cmp_lt_i64_e32 vcc_lo, 0, v[2:3]                         // 00000000346c: 7ca20480
	v_add3_u32 v1, v1, v5, v4                                  // 000000003470: d6550001 04120b01
	s_delay_alu instid0(valu_dep_1)                            // 000000003478: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 00000000347c: 3e000081
	s_and_b32 s2, vcc_lo, s1                                   // 000000003480: 8b02016a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003484: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003488: be832002
	s_cbranch_execz 28                                         // 00000000348c: bfa5001c <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1a00>
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003490: 3e082081
	v_add_co_u32 v7, s2, s16, v0                               // 000000003494: d7000207 02020010
	v_bfe_u32 v6, v34, 16, 1                                   // 00000000349c: d6100006 02052122
	s_wait_alu depctr_va_sdst(0)                               // 0000000034a4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, v1, s2                  // 0000000034a8: d5207c08 000a0211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000034b0: bf870193
	v_add_co_u32 v4, s2, v7, v4                                // 0000000034b4: d7000204 02020907
	v_add3_u32 v6, v6, v34, 0x7fff                             // 0000000034bc: d6550006 03fe4506 00007fff
	v_or_b32_e32 v9, 0x400000, v34                             // 0000000034c8: 381244ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000034d0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000034d4: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v34, v34                               // 0000000034dc: d4180002 02024522
	s_wait_alu depctr_va_sdst(0)                               // 0000000034e4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000034e8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000034ec: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000034f4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003500: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000003504: 8c7e037e
	v_cmp_lt_i64_e64 s2, 1, v[2:3]                             // 000000003508: d4510002 02020481
	s_and_b32 s3, s2, s1                                       // 000000003510: 8b030102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003514: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000003518: be842003
	s_cbranch_execz 35                                         // 00000000351c: bfa50023 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1aac>
	v_add_co_u32 v6, s3, s16, v0                               // 000000003520: d7000306 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003528: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s3                  // 00000000352c: d5207c07 000e0211
	s_lshl_b64 s[6:7], s[14:15], 1                             // 000000003534: 8486810e
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003538: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000353c: bf88ff9e
	v_add_co_u32 v6, s3, v6, s6                                // 000000003540: d7000306 02000d06
	v_bfe_u32 v8, v57, 16, 1                                   // 000000003548: d6100008 02052139
	s_wait_alu depctr_va_sdst(0)                               // 000000003550: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s7, v7, s3                   // 000000003554: d5207c07 000e0e07
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000355c: bf870193
	v_add_co_u32 v4, s3, v6, v4                                // 000000003560: d7000304 02020906
	v_add3_u32 v8, v8, v57, 0x7fff                             // 000000003568: d6550008 03fe7308 00007fff
	v_or_b32_e32 v9, 0x400000, v57                             // 000000003574: 381272ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000357c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s3                   // 000000003580: d5207c05 000e0b07
	v_cmp_u_f32_e64 s3, v57, v57                               // 000000003588: d4180003 02027339
	s_wait_alu depctr_va_sdst(0)                               // 000000003590: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003594: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s3                           // 000000003598: d5010006 000e1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000035a0: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000035b0: 8c7e047e
	v_cmp_lt_i64_e64 s3, 2, v[2:3]                             // 0000000035b4: d4510003 02020482
	s_lshl_b64 s[10:11], s[14:15], 1                           // 0000000035bc: 848a810e
	s_and_b32 s4, s3, s1                                       // 0000000035c0: 8b040103
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035c4: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 0000000035c8: be852004
	s_cbranch_execz 35                                         // 0000000035cc: bfa50023 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1b5c>
	v_add_co_u32 v6, s4, s16, v0                               // 0000000035d0: d7000406 02020010
	s_wait_alu depctr_va_sdst(0)                               // 0000000035d8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s4                  // 0000000035dc: d5207c07 00120211
	s_lshl_b64 s[6:7], s[10:11], 1                             // 0000000035e4: 8486810a
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 0000000035e8: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035ec: bf88ff9e
	v_add_co_u32 v6, s4, v6, s6                                // 0000000035f0: d7000406 02000d06
	v_bfe_u32 v8, v53, 16, 1                                   // 0000000035f8: d6100008 02052135
	s_wait_alu depctr_va_sdst(0)                               // 000000003600: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s7, v7, s4                   // 000000003604: d5207c07 00120e07
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000360c: bf870193
	v_add_co_u32 v4, s4, v6, v4                                // 000000003610: d7000404 02020906
	v_add3_u32 v8, v8, v53, 0x7fff                             // 000000003618: d6550008 03fe6b08 00007fff
	v_or_b32_e32 v9, 0x400000, v53                             // 000000003624: 38126aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000362c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s4                   // 000000003630: d5207c05 00120b07
	v_cmp_u_f32_e64 s4, v53, v53                               // 000000003638: d4180004 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 000000003640: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003644: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s4                           // 000000003648: d5010006 00121308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003650: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000365c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003660: 8c7e057e
	v_cmp_lt_i64_e64 s4, 3, v[2:3]                             // 000000003664: d4510004 02020483
	s_mul_u64 s[36:37], s[14:15], 3                            // 00000000366c: aaa4830e
	s_and_b32 s5, s4, s1                                       // 000000003670: 8b050104
	s_wait_alu depctr_sa_sdst(0)                               // 000000003674: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 000000003678: be862005
	s_cbranch_execz 35                                         // 00000000367c: bfa50023 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1c0c>
	v_add_co_u32 v6, s5, s16, v0                               // 000000003680: d7000506 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003688: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s5                  // 00000000368c: d5207c07 00160211
	s_lshl_b64 s[8:9], s[36:37], 1                             // 000000003694: 84888124
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003698: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000369c: bf88ff9e
	v_add_co_u32 v6, s5, v6, s8                                // 0000000036a0: d7000506 02001106
	v_bfe_u32 v8, v47, 16, 1                                   // 0000000036a8: d6100008 0205212f
	s_wait_alu depctr_va_sdst(0)                               // 0000000036b0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s9, v7, s5                   // 0000000036b4: d5207c07 00160e09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000036bc: bf870193
	v_add_co_u32 v4, s5, v6, v4                                // 0000000036c0: d7000504 02020906
	v_add3_u32 v8, v8, v47, 0x7fff                             // 0000000036c8: d6550008 03fe5f08 00007fff
	v_or_b32_e32 v9, 0x400000, v47                             // 0000000036d4: 38125eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000036dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s5                   // 0000000036e0: d5207c05 00160b07
	v_cmp_u_f32_e64 s5, v47, v47                               // 0000000036e8: d4180005 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000036f0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000036f4: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s5                           // 0000000036f8: d5010006 00161308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003700: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000370c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003710: 8c7e067e
	v_cmp_lt_i64_e64 s5, 4, v[2:3]                             // 000000003714: d4510005 02020484
	s_lshl_b64 s[38:39], s[14:15], 2                           // 00000000371c: 84a6820e
	s_and_b32 s6, s5, s1                                       // 000000003720: 8b060105
	s_wait_alu depctr_sa_sdst(0)                               // 000000003724: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 000000003728: be872006
	s_cbranch_execz 35                                         // 00000000372c: bfa50023 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1cbc>
	v_add_co_u32 v6, s6, s16, v0                               // 000000003730: d7000606 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003738: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s6                  // 00000000373c: d5207c07 001a0211
	s_lshl_b64 s[8:9], s[38:39], 1                             // 000000003744: 84888126
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003748: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000374c: bf88ff9e
	v_add_co_u32 v6, s6, v6, s8                                // 000000003750: d7000606 02001106
	v_bfe_u32 v8, v46, 16, 1                                   // 000000003758: d6100008 0205212e
	s_wait_alu depctr_va_sdst(0)                               // 000000003760: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s9, v7, s6                   // 000000003764: d5207c07 001a0e09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000376c: bf870193
	v_add_co_u32 v4, s6, v6, v4                                // 000000003770: d7000604 02020906
	v_add3_u32 v8, v8, v46, 0x7fff                             // 000000003778: d6550008 03fe5d08 00007fff
	v_or_b32_e32 v9, 0x400000, v46                             // 000000003784: 38125cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000378c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s6                   // 000000003790: d5207c05 001a0b07
	v_cmp_u_f32_e64 s6, v46, v46                               // 000000003798: d4180006 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000037a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000037a4: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s6                           // 0000000037a8: d5010006 001a1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000037b0: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 0000000037c0: 8c7e077e
	v_cmp_lt_i64_e64 s6, 5, v[2:3]                             // 0000000037c4: d4510006 02020485
	s_mul_u64 s[40:41], s[14:15], 5                            // 0000000037cc: aaa8850e
	s_and_b32 s7, s6, s1                                       // 0000000037d0: 8b070106
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037d4: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 0000000037d8: be882007
	s_cbranch_execz 34                                         // 0000000037dc: bfa50022 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1d68>
	v_add_co_u32 v6, s7, s16, v0                               // 0000000037e0: d7000706 02020010
	s_wait_alu depctr_va_sdst(0)                               // 0000000037e8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s7                  // 0000000037ec: d5207c07 001e0211
	s_lshl_b64 s[42:43], s[40:41], 1                           // 0000000037f4: 84aa8128
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 0000000037f8: 3e082081
	v_add_co_u32 v6, s7, v6, s42                               // 0000000037fc: d7000706 02005506
	v_bfe_u32 v8, v42, 16, 1                                   // 000000003804: d6100008 0205212a
	s_wait_alu depctr_va_sdst(0)                               // 00000000380c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s43, v7, s7                  // 000000003810: d5207c07 001e0e2b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003818: bf870193
	v_add_co_u32 v4, s7, v6, v4                                // 00000000381c: d7000704 02020906
	v_add3_u32 v8, v8, v42, 0x7fff                             // 000000003824: d6550008 03fe5508 00007fff
	v_or_b32_e32 v9, 0x400000, v42                             // 000000003830: 381254ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003838: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s7                   // 00000000383c: d5207c05 001e0b07
	v_cmp_u_f32_e64 s7, v42, v42                               // 000000003844: d4180007 0202552a
	s_wait_alu depctr_va_sdst(0)                               // 00000000384c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003850: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s7                           // 000000003854: d5010006 001e1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000385c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003868: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 00000000386c: 8c7e087e
	v_cmp_lt_i64_e64 s7, 6, v[2:3]                             // 000000003870: d4510007 02020486
	s_mul_u64 s[42:43], s[14:15], 6                            // 000000003878: aaaa860e
	s_and_b32 s8, s7, s1                                       // 00000000387c: 8b080107
	s_wait_alu depctr_sa_sdst(0)                               // 000000003880: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003884: be892008
	s_cbranch_execz 34                                         // 000000003888: bfa50022 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1e14>
	v_add_co_u32 v6, s8, s16, v0                               // 00000000388c: d7000806 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003894: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s8                  // 000000003898: d5207c07 00220211
	s_lshl_b64 s[44:45], s[42:43], 1                           // 0000000038a0: 84ac812a
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 0000000038a4: 3e082081
	v_add_co_u32 v6, s8, v6, s44                               // 0000000038a8: d7000806 02005906
	v_bfe_u32 v8, v38, 16, 1                                   // 0000000038b0: d6100008 02052126
	s_wait_alu depctr_va_sdst(0)                               // 0000000038b8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s45, v7, s8                  // 0000000038bc: d5207c07 00220e2d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000038c4: bf870193
	v_add_co_u32 v4, s8, v6, v4                                // 0000000038c8: d7000804 02020906
	v_add3_u32 v8, v8, v38, 0x7fff                             // 0000000038d0: d6550008 03fe4d08 00007fff
	v_or_b32_e32 v9, 0x400000, v38                             // 0000000038dc: 38124cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000038e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s8                   // 0000000038e8: d5207c05 00220b07
	v_cmp_u_f32_e64 s8, v38, v38                               // 0000000038f0: d4180008 02024d26
	s_wait_alu depctr_va_sdst(0)                               // 0000000038f8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038fc: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s8                           // 000000003900: d5010006 00221308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003908: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003914: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003918: 8c7e097e
	v_cmp_lt_i64_e64 s8, 7, v[2:3]                             // 00000000391c: d4510008 02020487
	s_mul_u64 s[44:45], s[14:15], 7                            // 000000003924: aaac870e
	s_and_b32 s1, s8, s1                                       // 000000003928: 8b010108
	s_wait_alu depctr_sa_sdst(0)                               // 00000000392c: bf88ff9e
	s_and_saveexec_b32 s9, s1                                  // 000000003930: be892001
	s_cbranch_execz 34                                         // 000000003934: bfa50022 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1ec0>
	v_add_co_u32 v4, s1, s16, v0                               // 000000003938: d7000104 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003940: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v1, s1                  // 000000003944: d5207c05 00060211
	s_lshl_b64 s[46:47], s[44:45], 1                           // 00000000394c: 84ae812c
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003950: 3e042081
	v_add_co_u32 v4, s1, v4, s46                               // 000000003954: d7000104 02005d04
	v_bfe_u32 v6, v37, 16, 1                                   // 00000000395c: d6100006 02052125
	s_wait_alu depctr_va_sdst(0)                               // 000000003964: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s47, v5, s1                  // 000000003968: d5207c05 00060a2f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003970: bf870193
	v_add_co_u32 v2, s1, v4, v2                                // 000000003974: d7000102 02020504
	v_add3_u32 v6, v6, v37, 0x7fff                             // 00000000397c: d6550006 03fe4b06 00007fff
	v_or_b32_e32 v7, 0x400000, v37                             // 000000003988: 380e4aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003990: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v5, v3, s1                   // 000000003994: d5207c03 00060705
	v_cmp_u_f32_e64 s1, v37, v37                               // 00000000399c: d4180001 02024b25
	s_wait_alu depctr_va_sdst(0)                               // 0000000039a4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039a8: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s1                           // 0000000039ac: d5010004 00060f06
	global_store_d16_hi_b16 v[2:3], v4, off                    // 0000000039b4: ee09407c 02000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000039c4: 8c7e097e
	s_and_b32 s9, vcc_lo, s0                                   // 0000000039c8: 8b09006a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039cc: bf88ff9e
	s_and_saveexec_b32 s1, s9                                  // 0000000039d0: be812009
	s_cbranch_execz 25                                         // 0000000039d4: bfa50019 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1f3c>
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 0000000039d8: 3e042081
	v_add_co_u32 v5, vcc_lo, s16, v0                           // 0000000039dc: d7006a05 02020010
	v_bfe_u32 v4, v36, 16, 1                                   // 0000000039e4: d6100004 02052124
	s_wait_alu depctr_va_vcc(0)                                // 0000000039ec: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s17, v1, vcc_lo              // 0000000039f0: d5207c06 01aa0211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000039f8: bf870193
	v_add_co_u32 v2, vcc_lo, v5, v2                            // 0000000039fc: d7006a02 02020505
	v_add3_u32 v4, v4, v36, 0x7fff                             // 000000003a04: d6550004 03fe4904 00007fff
	v_or_b32_e32 v7, 0x400000, v36                             // 000000003a10: 380e48ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003a18: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v6, v3, vcc_lo               // 000000003a1c: d5207c03 01aa0706
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 000000003a24: 7c304924
	s_wait_alu depctr_va_vcc(0)                                // 000000003a28: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo                       // 000000003a2c: 02080f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003a30: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003a40: 8c7e017e
	s_and_b32 s2, s2, s0                                       // 000000003a44: 8b020002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a48: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003a4c: be812002
	s_cbranch_execz 31                                         // 000000003a50: bfa5001f <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1fd0>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003a54: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003a5c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003a60: d5207c05 01aa0211
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003a68: 3e042081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000003a6c: bf8701c3
	v_add_co_u32 v4, vcc_lo, v4, s10                           // 000000003a70: d7006a04 02001504
	v_bfe_u32 v6, v35, 16, 1                                   // 000000003a78: d6100006 02052123
	s_wait_alu depctr_va_vcc(0)                                // 000000003a80: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s11, v5, vcc_lo              // 000000003a84: d5207c05 01aa0a0b
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003a8c: d7006a02 02020504
	s_delay_alu instid0(valu_dep_3)                            // 000000003a94: bf870003
	v_add3_u32 v6, v6, v35, 0x7fff                             // 000000003a98: d6550006 03fe4706 00007fff
	v_or_b32_e32 v7, 0x400000, v35                             // 000000003aa4: 380e46ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003aac: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003ab0: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 000000003ab8: 7c304723
	s_wait_alu depctr_va_vcc(0)                                // 000000003abc: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003ac0: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ac4: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ad0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003ad4: 8c7e017e
	s_and_b32 s2, s3, s0                                       // 000000003ad8: 8b020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000003adc: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003ae0: be812002
	s_cbranch_execz 32                                         // 000000003ae4: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2068>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003ae8: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003af0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003af4: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[10:11], 1                             // 000000003afc: 8482810a
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003b00: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b04: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003b08: d7006a04 02000504
	v_bfe_u32 v6, v33, 16, 1                                   // 000000003b10: d6100006 02052121
	s_wait_alu depctr_va_vcc(0)                                // 000000003b18: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003b1c: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003b24: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003b28: d7006a02 02020504
	v_add3_u32 v6, v6, v33, 0x7fff                             // 000000003b30: d6550006 03fe4306 00007fff
	v_or_b32_e32 v7, 0x400000, v33                             // 000000003b3c: 380e42ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003b44: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003b48: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 000000003b50: 7c304321
	s_wait_alu depctr_va_vcc(0)                                // 000000003b54: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003b58: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003b5c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b68: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003b6c: 8c7e017e
	s_and_b32 s2, s4, s0                                       // 000000003b70: 8b020004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b74: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003b78: be812002
	s_cbranch_execz 32                                         // 000000003b7c: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2100>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003b80: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003b88: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003b8c: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[36:37], 1                             // 000000003b94: 84828124
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003b98: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b9c: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003ba0: d7006a04 02000504
	v_bfe_u32 v6, v32, 16, 1                                   // 000000003ba8: d6100006 02052120
	s_wait_alu depctr_va_vcc(0)                                // 000000003bb0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003bb4: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003bbc: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003bc0: d7006a02 02020504
	v_add3_u32 v6, v6, v32, 0x7fff                             // 000000003bc8: d6550006 03fe4106 00007fff
	v_or_b32_e32 v7, 0x400000, v32                             // 000000003bd4: 380e40ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003bdc: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003be0: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 000000003be8: 7c304120
	s_wait_alu depctr_va_vcc(0)                                // 000000003bec: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003bf0: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003bf4: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c00: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003c04: 8c7e017e
	s_and_b32 s2, s5, s0                                       // 000000003c08: 8b020005
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c0c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003c10: be812002
	s_cbranch_execz 32                                         // 000000003c14: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2198>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003c18: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003c20: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003c24: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[38:39], 1                             // 000000003c2c: 84828126
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003c30: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c34: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003c38: d7006a04 02000504
	v_bfe_u32 v6, v31, 16, 1                                   // 000000003c40: d6100006 0205211f
	s_wait_alu depctr_va_vcc(0)                                // 000000003c48: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003c4c: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003c54: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003c58: d7006a02 02020504
	v_add3_u32 v6, v6, v31, 0x7fff                             // 000000003c60: d6550006 03fe3f06 00007fff
	v_or_b32_e32 v7, 0x400000, v31                             // 000000003c6c: 380e3eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003c74: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003c78: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v31, v31                           // 000000003c80: 7c303f1f
	s_wait_alu depctr_va_vcc(0)                                // 000000003c84: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003c88: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003c8c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c98: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003c9c: 8c7e017e
	s_and_b32 s2, s6, s0                                       // 000000003ca0: 8b020006
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ca4: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003ca8: be812002
	s_cbranch_execz 32                                         // 000000003cac: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2230>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003cb0: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003cb8: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003cbc: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[40:41], 1                             // 000000003cc4: 84828128
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003cc8: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ccc: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003cd0: d7006a04 02000504
	v_bfe_u32 v6, v30, 16, 1                                   // 000000003cd8: d6100006 0205211e
	s_wait_alu depctr_va_vcc(0)                                // 000000003ce0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003ce4: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003cec: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003cf0: d7006a02 02020504
	v_add3_u32 v6, v6, v30, 0x7fff                             // 000000003cf8: d6550006 03fe3d06 00007fff
	v_or_b32_e32 v7, 0x400000, v30                             // 000000003d04: 380e3cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003d0c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003d10: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 000000003d18: 7c303d1e
	s_wait_alu depctr_va_vcc(0)                                // 000000003d1c: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003d20: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003d24: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d30: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003d34: 8c7e017e
	s_and_b32 s2, s7, s0                                       // 000000003d38: 8b020007
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d3c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003d40: be812002
	s_cbranch_execz 32                                         // 000000003d44: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x22c8>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003d48: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003d50: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003d54: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[42:43], 1                             // 000000003d5c: 8482812a
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003d60: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d64: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003d68: d7006a04 02000504
	v_bfe_u32 v6, v29, 16, 1                                   // 000000003d70: d6100006 0205211d
	s_wait_alu depctr_va_vcc(0)                                // 000000003d78: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003d7c: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003d84: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003d88: d7006a02 02020504
	v_add3_u32 v6, v6, v29, 0x7fff                             // 000000003d90: d6550006 03fe3b06 00007fff
	v_or_b32_e32 v7, 0x400000, v29                             // 000000003d9c: 380e3aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003da4: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003da8: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 000000003db0: 7c303b1d
	s_wait_alu depctr_va_vcc(0)                                // 000000003db4: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003db8: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003dbc: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dc8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003dcc: 8c7e017e
	s_and_b32 s1, s8, s0                                       // 000000003dd0: 8b010008
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dd4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003dd8: be802001
	s_cbranch_execz 32                                         // 000000003ddc: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2360>
	v_add_co_u32 v2, vcc_lo, s16, v0                           // 000000003de0: d7006a02 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003de8: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s17, v1, vcc_lo              // 000000003dec: d5207c03 01aa0211
	s_lshl_b64 s[2:3], s[44:45], 1                             // 000000003df4: 8482812c
	v_lshlrev_b64_e32 v[0:1], 1, v[16:17]                      // 000000003df8: 3e002081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dfc: bf88ff9e
	v_add_co_u32 v2, vcc_lo, v2, s2                            // 000000003e00: d7006a02 02000502
	v_bfe_u32 v4, v19, 16, 1                                   // 000000003e08: d6100004 02052113
	s_wait_alu depctr_va_vcc(0)                                // 000000003e10: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo               // 000000003e14: d5207c03 01aa0603
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003e1c: bf870193
	v_add_co_u32 v0, vcc_lo, v2, v0                            // 000000003e20: d7006a00 02020102
	v_add3_u32 v4, v4, v19, 0x7fff                             // 000000003e28: d6550004 03fe2704 00007fff
	v_or_b32_e32 v5, 0x400000, v19                             // 000000003e34: 380a26ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003e3c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo               // 000000003e40: d5207c01 01aa0303
	v_cmp_u_f32_e32 vcc_lo, v19, v19                           // 000000003e48: 7c302713
	s_wait_alu depctr_va_vcc(0)                                // 000000003e4c: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000003e50: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000003e54: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e60: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003e64: 8c7e007e
	s_branch 63322                                             // 000000003e68: bfa0f75a <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0xd4>
	v_or_b32_e32 v0, s30, v18                                  // 000000003e6c: 3800241e
	v_cmp_gt_i64_e32 vcc_lo, s[14:15], v[16:17]                // 000000003e70: 7ca8200e
	v_dual_mov_b32 v1, s31 :: v_dual_mov_b32 v6, s31           // 000000003e74: ca10001f 0106001f
	v_mov_b32_e32 v19, 0                                       // 000000003e7c: 7e260280
	s_delay_alu instid0(valu_dep_4)                            // 000000003e80: bf870004
	v_or_b32_e32 v2, 1, v0                                     // 000000003e84: 38040081
	v_or_b32_e32 v5, 2, v0                                     // 000000003e88: 380a0082
	v_or_b32_e32 v10, 5, v0                                    // 000000003e8c: 38140085
	v_dual_mov_b32 v3, s31 :: v_dual_cndmask_b32 v4, 0, v17    // 000000003e90: ca12001f 03042280
	v_or_b32_e32 v8, 4, v0                                     // 000000003e98: 38100084
	v_cmp_gt_i64_e64 s0, s[12:13], v[0:1]                      // 000000003e9c: d4540000 0202000c
	v_mov_b32_e32 v7, s31                                      // 000000003ea4: 7e0e021f
	s_delay_alu instid0(valu_dep_4)                            // 000000003ea8: bf870004
	v_cmp_gt_i64_e64 s1, s[12:13], v[2:3]                      // 000000003eac: d4540001 0202040c
	v_cndmask_b32_e32 v3, 0, v16, vcc_lo                       // 000000003eb4: 02062080
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000003eb8: 7ca80a0c
	v_or_b32_e32 v6, 3, v0                                     // 000000003ebc: 380c0083
	v_dual_mov_b32 v11, s31 :: v_dual_mov_b32 v44, v19         // 000000003ec0: ca10001f 0b2c0113
	s_wait_alu depctr_va_sdst(0)                               // 000000003ec8: bf88f19f
	v_cndmask_b32_e64 v22, 0, v2, s1                           // 000000003ecc: d5010016 00060480
	v_cndmask_b32_e64 v24, 0, s31, s1                          // 000000003ed4: d5010018 00043e80
	s_wait_alu depctr_va_vcc(0)                                // 000000003edc: bf88ff9d
	v_dual_cndmask_b32 v25, 0, v5 :: v_dual_mov_b32 v42, v19   // 000000003ee0: ca500a80 192a0113
	v_or_b32_e32 v5, 6, v0                                     // 000000003ee8: 380a0086
	v_mov_b32_e32 v9, s31                                      // 000000003eec: 7e12021f
	v_cndmask_b32_e64 v14, 0, v0, s0                           // 000000003ef0: d501000e 00020080
	v_cndmask_b32_e64 v23, 0, s31, s0                          // 000000003ef8: d5010017 00003e80
	v_cndmask_b32_e64 v26, 0, s31, vcc_lo                      // 000000003f00: d501001a 01a83e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[10:11]                // 000000003f08: 7ca8140c
	v_cmp_gt_i64_e64 s1, s[12:13], v[8:9]                      // 000000003f0c: d4540001 0202100c
	v_mul_lo_u32 v36, v14, s25                                 // 000000003f14: d72c0024 0200330e
	v_mul_lo_u32 v23, v23, s24                                 // 000000003f1c: d72c0017 02003117
	v_mul_lo_u32 v37, v22, s25                                 // 000000003f24: d72c0025 02003316
	s_wait_alu depctr_va_vcc(0)                                // 000000003f2c: bf88ff9d
	v_dual_mov_b32 v45, v19 :: v_dual_cndmask_b32 v32, 0, v10  // 000000003f30: ca120113 2d201480
	s_wait_alu depctr_va_sdst(0)                               // 000000003f38: bf88f19f
	v_cndmask_b32_e64 v30, 0, v8, s1                           // 000000003f3c: d501001e 00061080
	v_mov_b32_e32 v8, s31                                      // 000000003f44: 7e10021f
	v_cmp_gt_i64_e64 s0, s[12:13], v[6:7]                      // 000000003f48: d4540000 02020c0c
	v_cndmask_b32_e64 v33, 0, s31, vcc_lo                      // 000000003f50: d5010021 01a83e80
	v_or_b32_e32 v7, 7, v0                                     // 000000003f58: 380e0087
	v_cndmask_b32_e64 v31, 0, s31, s1                          // 000000003f5c: d501001f 00043e80
	v_cmp_gt_i64_e64 s1, s[14:15], v[20:21]                    // 000000003f64: d4540001 0202280e
	v_mul_lo_u32 v39, v30, s25                                 // 000000003f6c: d72c0027 0200331e
	s_wait_alu depctr_va_sdst(0)                               // 000000003f74: bf88f19f
	v_cndmask_b32_e64 v27, 0, v6, s0                           // 000000003f78: d501001b 00020c80
	v_cndmask_b32_e64 v29, 0, s31, s0                          // 000000003f80: d501001d 00003e80
	v_add_co_u32 v10, s0, s34, v28                             // 000000003f88: d700000a 02023822
	v_mov_b32_e32 v6, s31                                      // 000000003f90: 7e0c021f
	s_wait_alu depctr_va_sdst(0)                               // 000000003f94: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s35, 0, s0                  // 000000003f98: d5207c0b 00010023
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_3)// 000000003fa0: bf8701b3
	v_add_co_u32 v2, vcc_lo, v10, 16                           // 000000003fa4: d7006a02 0201210a
	v_cmp_gt_i64_e64 s0, s[12:13], v[7:8]                      // 000000003fac: d4540000 02020e0c
	s_wait_alu depctr_va_vcc(0)                                // 000000003fb4: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, 0, v11, vcc_lo               // 000000003fb8: d5207c09 01aa1680
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000003fc0: 7ca80a0c
	v_mul_lo_u32 v12, s19, v2                                  // 000000003fc4: d72c000c 02020413
	v_cndmask_b32_e64 v8, 0, v21, s1                           // 000000003fcc: d5010008 00062a80
	s_delay_alu instid0(valu_dep_4)                            // 000000003fd4: bf870004
	v_mul_lo_u32 v9, s18, v9                                   // 000000003fd8: d72c0009 02021212
	s_wait_alu depctr_va_sdst(0)                               // 000000003fe0: bf88f19f
	v_cndmask_b32_e64 v13, 0, v7, s0                           // 000000003fe4: d501000d 00020e80
	v_cndmask_b32_e64 v15, 0, s31, s0                          // 000000003fec: d501000f 00003e80
	s_wait_alu depctr_va_vcc(0)                                // 000000003ff4: bf88ff9d
	v_cndmask_b32_e32 v34, 0, v5, vcc_lo                       // 000000003ff8: 02440a80
	v_mad_co_u64_u32 v[5:6], null, s18, v2, v[18:19]           // 000000003ffc: d6fe7c05 044a0412
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]                        // 000000004004: 3e040682
	v_cndmask_b32_e64 v7, 0, v20, s1                           // 000000004008: d5010007 00062880
	v_cndmask_b32_e64 v35, 0, s31, vcc_lo                      // 000000004010: d5010023 01a83e80
	v_mul_lo_u32 v20, s19, v10                                 // 000000004018: d72c0014 02021413
	v_mul_lo_u32 v21, v13, s25                                 // 000000004020: d72c0015 0200330d
	v_mul_lo_u32 v38, v29, s24                                 // 000000004028: d72c0026 0200311d
	v_add_co_u32 v2, vcc_lo, s28, v2                           // 000000004030: d7006a02 0202041c
	v_add3_u32 v6, v12, v6, v9                                 // 000000004038: d6550006 04260d0c
	s_wait_alu depctr_va_vcc(0)                                // 000000004040: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s29, v3, vcc_lo              // 000000004044: d5207c03 01aa061d
	v_add_co_u32 v4, vcc_lo, s26, v5                           // 00000000404c: d7006a04 02020a1a
	s_wait_alu depctr_va_vcc(0)                                // 000000004054: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s27, v6, vcc_lo              // 000000004058: d5207c05 01aa0c1b
	v_lshlrev_b64_e32 v[6:7], 2, v[7:8]                        // 000000004060: 3e0c0e82
	v_mad_co_u64_u32 v[8:9], null, s18, v10, v[18:19]          // 000000004064: d6fe7c08 044a1412
	v_mul_lo_u32 v12, s18, v11                                 // 00000000406c: d72c000c 02021612
	v_mad_co_u64_u32 v[10:11], null, v13, s24, 0               // 000000004074: d6fe7c0a 0200310d
	v_mul_lo_u32 v13, v15, s24                                 // 00000000407c: d72c000d 0200310f
	v_add_co_u32 v15, s0, s30, v28                             // 000000004084: d700000f 0202381e
	s_wait_alu depctr_va_sdst(0)                               // 00000000408c: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s31, 0, s0                  // 000000004090: d5207c1c 0001001f
	v_mul_lo_u32 v40, v31, s24                                 // 000000004098: d72c0028 0200311f
	v_add3_u32 v9, v20, v9, v12                                // 0000000040a0: d6550009 04321314
	v_mul_lo_u32 v41, v32, s25                                 // 0000000040a8: d72c0029 02003320
	v_add3_u32 v11, v11, v21, v13                              // 0000000040b0: d655000b 04362b0b
	v_mad_co_u64_u32 v[12:13], null, s18, v15, v[18:19]        // 0000000040b8: d6fe7c0c 044a1e12
	v_mul_lo_u32 v18, s18, v28                                 // 0000000040c0: d72c0012 02023812
	v_mul_lo_u32 v28, s19, v15                                 // 0000000040c8: d72c001c 02021e13
	v_mad_co_u64_u32 v[14:15], null, v14, s24, 0               // 0000000040d0: d6fe7c0e 0200310e
	v_mad_co_u64_u32 v[20:21], null, v22, s24, 0               // 0000000040d8: d6fe7c14 02003116
	v_mul_lo_u32 v22, v24, s24                                 // 0000000040e0: d72c0016 02003118
	v_add_co_u32 v6, vcc_lo, s28, v6                           // 0000000040e8: d7006a06 02020c1c
	s_wait_alu depctr_va_vcc(0)                                // 0000000040f0: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s29, v7, vcc_lo              // 0000000040f4: d5207c07 01aa0e1d
	v_add3_u32 v13, v28, v13, v18                              // 0000000040fc: d655000d 044a1b1c
	v_add3_u32 v15, v15, v36, v23                              // 000000004104: d655000f 045e490f
	v_mul_lo_u32 v18, v25, s25                                 // 00000000410c: d72c0012 02003319
	v_add3_u32 v21, v21, v37, v22                              // 000000004114: d6550015 045a4b15
	v_mad_co_u64_u32 v[22:23], null, v25, s24, 0               // 00000000411c: d6fe7c16 02003119
	v_mul_lo_u32 v36, v26, s24                                 // 000000004124: d72c0024 0200311a
	v_mul_lo_u32 v37, v27, s25                                 // 00000000412c: d72c0025 0200331b
	v_mad_co_u64_u32 v[24:25], null, v27, s24, 0               // 000000004134: d6fe7c18 0200311b
	v_mad_co_u64_u32 v[26:27], null, v30, s24, 0               // 00000000413c: d6fe7c1a 0200311e
	v_mad_co_u64_u32 v[28:29], null, v32, s24, 0               // 000000004144: d6fe7c1c 02003120
	v_mul_lo_u32 v32, v33, s24                                 // 00000000414c: d72c0020 02003121
	v_mul_lo_u32 v33, v34, s25                                 // 000000004154: d72c0021 02003322
	v_mad_co_u64_u32 v[30:31], null, v34, s24, 0               // 00000000415c: d6fe7c1e 02003122
	v_mul_lo_u32 v34, v35, s24                                 // 000000004164: d72c0022 02003123
	v_add3_u32 v23, v23, v18, v36                              // 00000000416c: d6550017 04922517
	v_add3_u32 v25, v25, v37, v38                              // 000000004174: d6550019 049a4b19
	v_add3_u32 v27, v27, v39, v40                              // 00000000417c: d655001b 04a24f1b
	v_add_co_u32 v8, vcc_lo, s26, v8                           // 000000004184: d7006a08 0202101a
	v_add3_u32 v29, v29, v41, v32                              // 00000000418c: d655001d 0482531d
	s_wait_alu depctr_va_vcc(0)                                // 000000004194: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s27, v9, vcc_lo              // 000000004198: d5207c09 01aa121b
	v_add3_u32 v31, v31, v33, v34                              // 0000000041a0: d655001f 048a431f
	v_add_co_u32 v12, vcc_lo, s22, v12                         // 0000000041a8: d7006a0c 02021816
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]                    // 0000000041b0: 3e141482
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]                    // 0000000041b4: 3e1c1c82
	v_lshlrev_b64_e32 v[20:21], 2, v[20:21]                    // 0000000041b8: 3e282882
	v_lshlrev_b64_e32 v[22:23], 2, v[22:23]                    // 0000000041bc: 3e2c2c82
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]                    // 0000000041c0: 3e303082
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]                    // 0000000041c4: 3e343482
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]                    // 0000000041c8: 3e383882
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]                    // 0000000041cc: 3e3c3c82
	s_wait_alu depctr_va_vcc(0)                                // 0000000041d0: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s23, v13, vcc_lo            // 0000000041d4: d5207c0d 01aa1a17
	v_dual_mov_b32 v43, v19 :: v_dual_mov_b32 v38, v19         // 0000000041dc: ca100113 2b260113
	v_dual_mov_b32 v41, v19 :: v_dual_mov_b32 v36, v19         // 0000000041e4: ca100113 29240113
	v_dual_mov_b32 v40, v19 :: v_dual_mov_b32 v39, v19         // 0000000041ec: ca100113 28260113
	v_dual_mov_b32 v34, v19 :: v_dual_mov_b32 v37, v19         // 0000000041f4: ca100113 22240113
	v_dual_mov_b32 v32, v19 :: v_dual_mov_b32 v35, v19         // 0000000041fc: ca100113 20220113
	v_dual_mov_b32 v18, v19 :: v_dual_mov_b32 v33, v19         // 000000004204: ca100113 12200113
	s_lshl_b64 s[0:1], s[14:15], 2                             // 00000000420c: 8480820e
	s_mov_b64 s[2:3], 0                                        // 000000004210: be820180
	v_add_co_u32 v46, vcc_lo, s20, v14                         // 000000004214: d7006a2e 02021c14
	s_wait_alu depctr_va_vcc(0)                                // 00000000421c: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s21, v15, vcc_lo            // 000000004220: d5207c2f 01aa1e15
	v_add_co_u32 v48, vcc_lo, s20, v20                         // 000000004228: d7006a30 02022814
	s_wait_alu depctr_va_vcc(0)                                // 000000004230: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s21, v21, vcc_lo            // 000000004234: d5207c31 01aa2a15
	v_add_co_u32 v50, vcc_lo, s20, v22                         // 00000000423c: d7006a32 02022c14
	s_wait_alu depctr_va_vcc(0)                                // 000000004244: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s21, v23, vcc_lo            // 000000004248: d5207c33 01aa2e15
	v_add_co_u32 v52, vcc_lo, s20, v24                         // 000000004250: d7006a34 02023014
	s_wait_alu depctr_va_vcc(0)                                // 000000004258: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s21, v25, vcc_lo            // 00000000425c: d5207c35 01aa3215
	v_add_co_u32 v56, vcc_lo, s20, v26                         // 000000004264: d7006a38 02023414
	s_wait_alu depctr_va_vcc(0)                                // 00000000426c: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s21, v27, vcc_lo            // 000000004270: d5207c39 01aa3615
	v_add_co_u32 v58, vcc_lo, s20, v28                         // 000000004278: d7006a3a 02023814
	s_clause 0x1                                               // 000000004280: bf850001
	global_load_b64 v[62:63], v[12:13], off                    // 000000004284: ee05407c 0000003e 0000000c
	global_load_b64 v[64:65], v[12:13], off offset:16          // 000000004290: ee05407c 00000040 0000100c
	s_clause 0x1                                               // 00000000429c: bf850001
	global_load_b64 v[54:55], v[8:9], off                      // 0000000042a0: ee05407c 00000036 00000008
	global_load_b64 v[66:67], v[8:9], off offset:16            // 0000000042ac: ee05407c 00000042 00001008
	s_clause 0x1                                               // 0000000042b8: bf850001
	global_load_b64 v[68:69], v[4:5], off                      // 0000000042bc: ee05407c 00000044 00000004
	global_load_b64 v[70:71], v[4:5], off offset:16            // 0000000042c8: ee05407c 00000046 00001004
	s_wait_alu depctr_va_vcc(0)                                // 0000000042d4: bf88ff9d
	v_add_co_ci_u32_e64 v59, null, s21, v29, vcc_lo            // 0000000042d8: d5207c3b 01aa3a15
	v_add_co_u32 v60, vcc_lo, s20, v30                         // 0000000042e0: d7006a3c 02023c14
	s_wait_alu depctr_va_vcc(0)                                // 0000000042e8: bf88ff9d
	v_add_co_ci_u32_e64 v61, null, s21, v31, vcc_lo            // 0000000042ec: d5207c3d 01aa3e15
	v_add_co_u32 v72, vcc_lo, s20, v10                         // 0000000042f4: d7006a48 02021414
	global_load_b32 v74, v[2:3], off                           // 0000000042fc: ee05007c 0000004a 00000002
	s_wait_alu depctr_va_vcc(0)                                // 000000004308: bf88ff9d
	v_add_co_ci_u32_e64 v73, null, s21, v11, vcc_lo            // 00000000430c: d5207c49 01aa1615
	global_load_b32 v75, v[6:7], off                           // 000000004314: ee05007c 0000004b 00000006
	s_clause 0x7                                               // 000000004320: bf850007
	global_load_b32 v76, v[46:47], off                         // 000000004324: ee05007c 0000004c 0000002e
	global_load_b32 v77, v[48:49], off                         // 000000004330: ee05007c 0000004d 00000030
	global_load_b32 v78, v[50:51], off                         // 00000000433c: ee05007c 0000004e 00000032
	global_load_b32 v79, v[52:53], off                         // 000000004348: ee05007c 0000004f 00000034
	global_load_b32 v80, v[56:57], off                         // 000000004354: ee05007c 00000050 00000038
	global_load_b32 v81, v[58:59], off                         // 000000004360: ee05007c 00000051 0000003a
	global_load_b32 v82, v[60:61], off                         // 00000000436c: ee05007c 00000052 0000003c
	global_load_b32 v72, v[72:73], off                         // 000000004378: ee05007c 00000048 00000048
	s_wait_alu depctr_sa_sdst(0)                               // 000000004384: bf88ff9e
	v_add_co_u32 v2, vcc_lo, v2, s0                            // 000000004388: d7006a02 02000102
	s_wait_alu depctr_va_vcc(0)                                // 000000004390: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo               // 000000004394: d5207c03 01aa0601
	v_add_co_u32 v4, vcc_lo, v4, 32                            // 00000000439c: d7006a04 02014104
	s_wait_alu depctr_va_vcc(0)                                // 0000000043a4: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo                // 0000000043a8: d5207c05 01aa0a80
	v_add_co_u32 v6, vcc_lo, v6, s0                            // 0000000043b0: d7006a06 02000106
	s_add_nc_u64 s[2:3], s[2:3], 32                            // 0000000043b8: a982a002
	s_wait_alu depctr_va_vcc(0)                                // 0000000043bc: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo               // 0000000043c0: d5207c07 01aa0e01
	v_add_co_u32 v8, vcc_lo, v8, 32                            // 0000000043c8: d7006a08 02014108
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043d0: bf88ff9e
	v_cmp_lt_u64_e64 s4, s[2:3], s[18:19]                      // 0000000043d4: d4590004 02002402
	s_wait_alu depctr_va_vcc(0)                                // 0000000043dc: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, 0, v9, vcc_lo                // 0000000043e0: d5207c09 01aa1280
	v_add_co_u32 v12, vcc_lo, v12, 32                          // 0000000043e8: d7006a0c 0201410c
	s_wait_alu depctr_va_vcc(0)                                // 0000000043f0: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, 0, v13, vcc_lo              // 0000000043f4: d5207c0d 01aa1a80
	s_and_b32 vcc_lo, exec_lo, s4                              // 0000000043fc: 8b6a047e
	s_add_nc_u64 s[20:21], s[20:21], 4                         // 000000004400: a9948414
	s_wait_loadcnt 0xd                                         // 000000004404: bfc0000d
	v_wmma_f32_16x16x16_fp8_fp8 v[46:53], v[62:63], v[54:55], 0// 000000004408: cc46402e 1a026d3e
	s_wait_loadcnt 0xb                                         // 000000004410: bfc0000b
	v_wmma_f32_16x16x16_fp8_fp8 v[54:61], v[62:63], v[68:69], 0// 000000004414: cc464036 1a02893e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000441c: bf870122
	v_wmma_f32_16x16x16_fp8_fp8 v[46:53], v[64:65], v[66:67], v[46:53]// 000000004420: cc46402e 1cba8540
	s_wait_loadcnt 0xa                                         // 000000004428: bfc0000a
	v_wmma_f32_16x16x16_fp8_fp8 v[54:61], v[64:65], v[70:71], v[54:61]// 00000000442c: cc464036 1cda8d40
	s_wait_loadcnt 0x6                                         // 000000004434: bfc00006
	v_dual_mul_f32 v70, v76, v75 :: v_dual_mul_f32 v71, v77, v75// 000000004438: c8c6974c 4646974d
	s_wait_loadcnt 0x5                                         // 000000004440: bfc00005
	v_dual_mul_f32 v73, v78, v75 :: v_dual_mul_f32 v62, v76, v74// 000000004444: c8c6974e 493e954c
	v_dual_mul_f32 v63, v74, v77 :: v_dual_mul_f32 v64, v74, v78// 00000000444c: c8c69b4a 3f409d4a
	s_wait_loadcnt 0x3                                         // 000000004454: bfc00003
	v_dual_mul_f32 v65, v74, v79 :: v_dual_mul_f32 v66, v74, v80// 000000004458: c8c69f4a 4142a14a
	s_wait_loadcnt 0x1                                         // 000000004460: bfc00001
	v_dual_mul_f32 v67, v74, v81 :: v_dual_mul_f32 v68, v74, v82// 000000004464: c8c6a34a 4344a54a
	s_wait_loadcnt 0x0                                         // 00000000446c: bfc00000
	v_dual_mul_f32 v69, v74, v72 :: v_dual_mul_f32 v74, v79, v75// 000000004470: c8c6914a 454a974f
	v_dual_mul_f32 v76, v80, v75 :: v_dual_mul_f32 v77, v81, v75// 000000004478: c8c69750 4c4c9751
	v_dual_mul_f32 v78, v82, v75 :: v_dual_mul_f32 v49, v49, v65// 000000004480: c8c69752 4e308331
	s_delay_alu instid0(valu_dep_3)                            // 000000004488: bf870003
	v_dual_mul_f32 v72, v72, v75 :: v_dual_mul_f32 v53, v53, v69// 00000000448c: c8c69748 48348b35
	v_dual_mul_f32 v46, v46, v62 :: v_dual_mul_f32 v47, v47, v63// 000000004494: c8c67d2e 2e2e7f2f
	v_dual_mul_f32 v48, v48, v64 :: v_dual_mul_f32 v51, v51, v67// 00000000449c: c8c68130 30328733
	v_dual_mul_f32 v50, v50, v66 :: v_dual_mul_f32 v55, v55, v71// 0000000044a4: c8c68532 32368f37
	v_dual_mul_f32 v52, v52, v68 :: v_dual_mul_f32 v57, v57, v74// 0000000044ac: c8c68934 34389539
	v_dual_mul_f32 v54, v54, v70 :: v_dual_mul_f32 v59, v59, v77// 0000000044b4: c8c68d36 363a9b3b
	v_dual_mul_f32 v56, v56, v73 :: v_dual_mul_f32 v61, v61, v72// 0000000044bc: c8c69338 383c913d
	v_dual_mul_f32 v58, v58, v76 :: v_dual_add_f32 v19, v19, v46// 0000000044c4: c8c8993a 3a125d13
	v_dual_mul_f32 v60, v60, v78 :: v_dual_add_f32 v45, v45, v47// 0000000044cc: c8c89d3c 3c2c5f2d
	v_dual_add_f32 v44, v44, v48 :: v_dual_add_f32 v43, v43, v49// 0000000044d4: c908612c 2c2a632b
	v_dual_add_f32 v42, v42, v50 :: v_dual_add_f32 v41, v41, v51// 0000000044dc: c908652a 2a286729
	v_dual_add_f32 v40, v40, v52 :: v_dual_add_f32 v39, v39, v53// 0000000044e4: c9086928 28266b27
	v_dual_add_f32 v38, v38, v54 :: v_dual_add_f32 v37, v37, v55// 0000000044ec: c9086d26 26246f25
	v_dual_add_f32 v36, v36, v56 :: v_dual_add_f32 v35, v35, v57// 0000000044f4: c9087124 24227323
	v_dual_add_f32 v34, v34, v58 :: v_dual_add_f32 v33, v33, v59// 0000000044fc: c9087522 22207721
	v_add_f32_e32 v32, v32, v60                                // 000000004504: 06407920
	v_add_f32_e32 v18, v18, v61                                // 000000004508: 06247b12
	s_wait_alu depctr_sa_sdst(0)                               // 00000000450c: bf88ff9e
	s_cbranch_vccnz 65344                                      // 000000004510: bfa4ff40 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2714>
	v_mul_lo_u32 v2, s15, v0                                   // 000000004514: d72c0002 0202000f
	v_mul_lo_u32 v3, s14, v1                                   // 00000000451c: d72c0003 0202020e
	v_mad_co_u64_u32 v[0:1], null, s14, v0, 0                  // 000000004524: d6fe7c00 0202000e
	v_bfe_u32 v4, v19, 16, 1                                   // 00000000452c: d6100004 02052113
	v_or_b32_e32 v5, 0x400000, v19                             // 000000004534: 380a26ff 00400000
	v_bfe_u32 v6, v45, 16, 1                                   // 00000000453c: d6100006 0205212d
	v_or_b32_e32 v7, 0x400000, v45                             // 000000004544: 380e5aff 00400000
	s_lshl_b64 s[0:1], s[14:15], 1                             // 00000000454c: 8480810e
	v_add3_u32 v4, v4, v19, 0x7fff                             // 000000004550: d6550004 03fe2704 00007fff
	v_or_b32_e32 v13, 0x400000, v43                            // 00000000455c: 381a56ff 00400000
	v_add3_u32 v1, v1, v3, v2                                  // 000000004564: d6550001 040a0701
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 00000000456c: 3e042081
	v_add3_u32 v6, v6, v45, 0x7fff                             // 000000004570: d6550006 03fe5b06 00007fff
	v_bfe_u32 v15, v42, 16, 1                                  // 00000000457c: d610000f 0205212a
	v_or_b32_e32 v16, 0x400000, v42                            // 000000004584: 382054ff 00400000
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 00000000458c: 3e000081
	v_or_b32_e32 v20, 0x400000, v40                            // 000000004590: 382850ff 00400000
	v_bfe_u32 v22, v39, 16, 1                                  // 000000004598: d6100016 02052127
	v_add3_u32 v15, v15, v42, 0x7fff                           // 0000000045a0: d655000f 03fe550f 00007fff
	v_or_b32_e32 v23, 0x400000, v39                            // 0000000045ac: 382e4eff 00400000
	v_add_co_u32 v8, vcc_lo, s16, v0                           // 0000000045b4: d7006a08 02020010
	s_wait_alu depctr_va_vcc(0)                                // 0000000045bc: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s17, v1, vcc_lo              // 0000000045c0: d5207c09 01aa0211
	v_cmp_u_f32_e32 vcc_lo, v19, v19                           // 0000000045c8: 7c302713
	v_add3_u32 v22, v22, v39, 0x7fff                           // 0000000045cc: d6550016 03fe4f16 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000045d8: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 0000000045dc: 02080b04
	v_add_co_u32 v0, vcc_lo, v8, v2                            // 0000000045e0: d7006a00 02020508
	s_wait_alu depctr_va_vcc(0)                                // 0000000045e8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v9, v3, vcc_lo               // 0000000045ec: d5207c01 01aa0709
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 0000000045f4: 7c305b2d
	v_bfe_u32 v5, v44, 16, 1                                   // 0000000045f8: d6100005 0205212c
	global_store_d16_hi_b16 v[0:1], v4, off                    // 000000004600: ee09407c 02000000 00000000
	s_wait_alu depctr_va_vcc(0)                                // 00000000460c: bf88ff9d
	v_cndmask_b32_e32 v10, v6, v7, vcc_lo                      // 000000004610: 02140f06
	s_wait_alu depctr_sa_sdst(0)                               // 000000004614: bf88ff9e
	v_add_co_u32 v6, vcc_lo, v8, s0                            // 000000004618: d7006a06 02000108
	s_wait_alu depctr_va_vcc(0)                                // 000000004620: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s1, v9, vcc_lo               // 000000004624: d5207c07 01aa1201
	v_add3_u32 v8, v5, v44, 0x7fff                             // 00000000462c: d6550008 03fe5905 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004638: bf870003
	v_add_co_u32 v4, vcc_lo, v6, v2                            // 00000000463c: d7006a04 02020506
	v_or_b32_e32 v9, 0x400000, v44                             // 000000004644: 381258ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000464c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v7, v3, vcc_lo               // 000000004650: d5207c05 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v44, v44                           // 000000004658: 7c30592c
	s_wait_alu depctr_va_vcc(0)                                // 00000000465c: bf88ff9d
	v_cndmask_b32_e32 v11, v8, v9, vcc_lo                      // 000000004660: 02161308
	v_add_co_u32 v9, vcc_lo, v6, s0                            // 000000004664: d7006a09 02000106
	v_bfe_u32 v8, v43, 16, 1                                   // 00000000466c: d6100008 0205212b
	s_wait_alu depctr_va_vcc(0)                                // 000000004674: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v7, vcc_lo              // 000000004678: d5207c0c 01aa0e01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004680: bf870193
	v_add_co_u32 v6, vcc_lo, v9, v2                            // 000000004684: d7006a06 02020509
	v_add3_u32 v8, v8, v43, 0x7fff                             // 00000000468c: d6550008 03fe5708 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004698: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 00000000469c: bf870003
	v_add_co_ci_u32_e64 v7, null, v12, v3, vcc_lo              // 0000000046a0: d5207c07 01aa070c
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 0000000046a8: 7c30572b
	s_wait_alu depctr_va_vcc(0)                                // 0000000046ac: bf88ff9d
	v_cndmask_b32_e32 v13, v8, v13, vcc_lo                     // 0000000046b0: 021a1b08
	v_add_co_u32 v14, vcc_lo, v9, s0                           // 0000000046b4: d7006a0e 02000109
	s_wait_alu depctr_va_vcc(0)                                // 0000000046bc: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo             // 0000000046c0: d5207c0c 01aa1801
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000046c8: bf870122
	v_add_co_u32 v8, vcc_lo, v14, v2                           // 0000000046cc: d7006a08 0202050e
	s_wait_alu depctr_va_vcc(0)                                // 0000000046d4: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v12, v3, vcc_lo              // 0000000046d8: d5207c09 01aa070c
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 0000000046e0: 7c30552a
	s_wait_alu depctr_va_vcc(0)                                // 0000000046e4: bf88ff9d
	v_cndmask_b32_e32 v16, v15, v16, vcc_lo                    // 0000000046e8: 0220210f
	s_clause 0x2                                               // 0000000046ec: bf850002
	global_store_d16_hi_b16 v[4:5], v10, off                   // 0000000046f0: ee09407c 05000000 00000004
	global_store_d16_hi_b16 v[6:7], v11, off                   // 0000000046fc: ee09407c 05800000 00000006
	global_store_d16_hi_b16 v[8:9], v13, off                   // 000000004708: ee09407c 06800000 00000008
	v_bfe_u32 v10, v41, 16, 1                                  // 000000004714: d610000a 02052129
	v_add_co_u32 v13, vcc_lo, v14, s0                          // 00000000471c: d7006a0d 0200010e
	s_wait_alu depctr_va_vcc(0)                                // 000000004724: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo             // 000000004728: d5207c0c 01aa1801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004730: bf870193
	v_add3_u32 v14, v10, v41, 0x7fff                           // 000000004734: d655000e 03fe530a 00007fff
	v_add_co_u32 v10, vcc_lo, v13, v2                          // 000000004740: d7006a0a 0202050d
	v_or_b32_e32 v15, 0x400000, v41                            // 000000004748: 381e52ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004750: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v12, v3, vcc_lo             // 000000004754: d5207c0b 01aa070c
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 00000000475c: 7c305329
	s_wait_alu depctr_va_vcc(0)                                // 000000004760: bf88ff9d
	v_cndmask_b32_e32 v17, v14, v15, vcc_lo                    // 000000004764: 02221f0e
	v_add_co_u32 v15, vcc_lo, v13, s0                          // 000000004768: d7006a0f 0200010d
	v_bfe_u32 v14, v40, 16, 1                                  // 000000004770: d610000e 02052128
	s_wait_alu depctr_va_vcc(0)                                // 000000004778: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s1, v12, vcc_lo             // 00000000477c: d5207c13 01aa1801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004784: bf870193
	v_add_co_u32 v12, vcc_lo, v15, v2                          // 000000004788: d7006a0c 0202050f
	v_add3_u32 v14, v14, v40, 0x7fff                           // 000000004790: d655000e 03fe510e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000479c: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000047a0: bf870003
	v_add_co_ci_u32_e64 v13, null, v19, v3, vcc_lo             // 0000000047a4: d5207c0d 01aa0713
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 0000000047ac: 7c305128
	s_wait_alu depctr_va_vcc(0)                                // 0000000047b0: bf88ff9d
	v_cndmask_b32_e32 v20, v14, v20, vcc_lo                    // 0000000047b4: 0228290e
	v_add_co_u32 v21, vcc_lo, v15, s0                          // 0000000047b8: d7006a15 0200010f
	s_wait_alu depctr_va_vcc(0)                                // 0000000047c0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s1, v19, vcc_lo             // 0000000047c4: d5207c13 01aa2601
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000047cc: bf870122
	v_add_co_u32 v14, vcc_lo, v21, v2                          // 0000000047d0: d7006a0e 02020515
	s_wait_alu depctr_va_vcc(0)                                // 0000000047d8: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v19, v3, vcc_lo             // 0000000047dc: d5207c0f 01aa0713
	v_cmp_u_f32_e32 vcc_lo, v39, v39                           // 0000000047e4: 7c304f27
	s_clause 0x2                                               // 0000000047e8: bf850002
	global_store_d16_hi_b16 v[10:11], v16, off                 // 0000000047ec: ee09407c 08000000 0000000a
	global_store_d16_hi_b16 v[12:13], v17, off                 // 0000000047f8: ee09407c 08800000 0000000c
	global_store_d16_hi_b16 v[14:15], v20, off                 // 000000004804: ee09407c 0a000000 0000000e
	v_bfe_u32 v17, v38, 16, 1                                  // 000000004810: d6100011 02052126
	s_wait_alu depctr_va_vcc(0)                                // 000000004818: bf88ff9d
	v_cndmask_b32_e32 v16, v22, v23, vcc_lo                    // 00000000481c: 02202f16
	v_add_co_u32 v20, vcc_lo, v21, s0                          // 000000004820: d7006a14 02000115
	s_wait_alu depctr_va_vcc(0)                                // 000000004828: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s1, v19, vcc_lo             // 00000000482c: d5207c13 01aa2601
	v_add3_u32 v17, v17, v38, 0x7fff                           // 000000004834: d6550011 03fe4d11 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004840: bf870003
	v_add_co_u32 v2, vcc_lo, v20, v2                           // 000000004844: d7006a02 02020514
	v_or_b32_e32 v21, 0x400000, v38                            // 00000000484c: 382a4cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004854: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v19, v3, vcc_lo              // 000000004858: d5207c03 01aa0713
	v_bfe_u32 v19, v37, 16, 1                                  // 000000004860: d6100013 02052125
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 000000004868: 7c304d26
	v_bfe_u32 v20, v36, 16, 1                                  // 00000000486c: d6100014 02052124
	global_store_d16_hi_b16 v[2:3], v16, off                   // 000000004874: ee09407c 08000000 00000002
	v_add3_u32 v16, v19, v37, 0x7fff                           // 000000004880: d6550010 03fe4b13 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000488c: bf88ff9d
	v_cndmask_b32_e32 v17, v17, v21, vcc_lo                    // 000000004890: 02222b11
	v_or_b32_e32 v19, 0x400000, v37                            // 000000004894: 38264aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 00000000489c: 7c304b25
	global_store_d16_hi_b16 v[0:1], v17, off offset:32         // 0000000048a0: ee09407c 08800000 00002000
	v_add3_u32 v0, v20, v36, 0x7fff                            // 0000000048ac: d6550000 03fe4914 00007fff
	v_or_b32_e32 v1, 0x400000, v36                             // 0000000048b8: 380248ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000048c0: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v19, vcc_lo                    // 0000000048c4: 02202710
	v_bfe_u32 v17, v35, 16, 1                                  // 0000000048c8: d6100011 02052123
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 0000000048d0: 7c304924
	global_store_d16_hi_b16 v[4:5], v16, off offset:32         // 0000000048d4: ee09407c 08000000 00002004
	v_add3_u32 v4, v17, v35, 0x7fff                            // 0000000048e0: d6550004 03fe4711 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000048ec: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000048f0: 02000300
	v_bfe_u32 v1, v34, 16, 1                                   // 0000000048f4: d6100001 02052122
	v_or_b32_e32 v5, 0x400000, v35                             // 0000000048fc: 380a46ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 000000004904: 7c304723
	v_or_b32_e32 v16, 0x400000, v32                            // 000000004908: 382040ff 00400000
	global_store_d16_hi_b16 v[6:7], v0, off offset:32          // 000000004910: ee09407c 00000000 00002006
	v_add3_u32 v0, v1, v34, 0x7fff                             // 00000000491c: d6550000 03fe4501 00007fff
	v_or_b32_e32 v1, 0x400000, v34                             // 000000004928: 380244ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004930: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000004934: 02080b04
	v_bfe_u32 v5, v33, 16, 1                                   // 000000004938: d6100005 02052121
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 000000004940: 7c304522
	v_bfe_u32 v6, v32, 16, 1                                   // 000000004944: d6100006 02052120
	v_or_b32_e32 v7, 0x400000, v33                             // 00000000494c: 380e42ff 00400000
	v_or_b32_e32 v17, 0x400000, v18                            // 000000004954: 382224ff 00400000
	v_add3_u32 v5, v5, v33, 0x7fff                             // 00000000495c: d6550005 03fe4305 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004968: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000496c: 02000300
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 000000004970: 7c304321
	v_bfe_u32 v1, v18, 16, 1                                   // 000000004974: d6100001 02052112
	v_add3_u32 v6, v6, v32, 0x7fff                             // 00000000497c: d6550006 03fe4106 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004988: bf88ff9d
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo                       // 00000000498c: 020a0f05
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 000000004990: 7c304120
	v_add3_u32 v1, v1, v18, 0x7fff                             // 000000004994: d6550001 03fe2501 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000049a0: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v16, vcc_lo                      // 0000000049a4: 020c2106
	v_cmp_u_f32_e32 vcc_lo, v18, v18                           // 0000000049a8: 7c302512
	s_wait_alu depctr_va_vcc(0)                                // 0000000049ac: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v17, vcc_lo                      // 0000000049b0: 02022301
	s_clause 0x3                                               // 0000000049b4: bf850003
	global_store_d16_hi_b16 v[8:9], v4, off offset:32          // 0000000049b8: ee09407c 02000000 00002008
	global_store_d16_hi_b16 v[10:11], v0, off offset:32        // 0000000049c4: ee09407c 00000000 0000200a
	global_store_d16_hi_b16 v[12:13], v5, off offset:32        // 0000000049d0: ee09407c 02800000 0000200c
	global_store_d16_hi_b16 v[14:15], v6, off offset:32        // 0000000049dc: ee09407c 03000000 0000200e
	global_store_d16_hi_b16 v[2:3], v1, off offset:32          // 0000000049e8: ee09407c 00800000 00002002
	s_endpgm                                                   // 0000000049f4: bfb00000
	s_code_end                                                 // 0000000049f8: bf9f0000
	s_code_end                                                 // 0000000049fc: bf9f0000
	s_code_end                                                 // 000000004a00: bf9f0000
	s_code_end                                                 // 000000004a04: bf9f0000
	s_code_end                                                 // 000000004a08: bf9f0000
	s_code_end                                                 // 000000004a0c: bf9f0000
	s_code_end                                                 // 000000004a10: bf9f0000
	s_code_end                                                 // 000000004a14: bf9f0000
	s_code_end                                                 // 000000004a18: bf9f0000
	s_code_end                                                 // 000000004a1c: bf9f0000
	s_code_end                                                 // 000000004a20: bf9f0000
	s_code_end                                                 // 000000004a24: bf9f0000
	s_code_end                                                 // 000000004a28: bf9f0000
	s_code_end                                                 // 000000004a2c: bf9f0000
	s_code_end                                                 // 000000004a30: bf9f0000
	s_code_end                                                 // 000000004a34: bf9f0000
	s_code_end                                                 // 000000004a38: bf9f0000
	s_code_end                                                 // 000000004a3c: bf9f0000
	s_code_end                                                 // 000000004a40: bf9f0000
	s_code_end                                                 // 000000004a44: bf9f0000
	s_code_end                                                 // 000000004a48: bf9f0000
	s_code_end                                                 // 000000004a4c: bf9f0000
	s_code_end                                                 // 000000004a50: bf9f0000
	s_code_end                                                 // 000000004a54: bf9f0000
	s_code_end                                                 // 000000004a58: bf9f0000
	s_code_end                                                 // 000000004a5c: bf9f0000
	s_code_end                                                 // 000000004a60: bf9f0000
	s_code_end                                                 // 000000004a64: bf9f0000
	s_code_end                                                 // 000000004a68: bf9f0000
	s_code_end                                                 // 000000004a6c: bf9f0000
	s_code_end                                                 // 000000004a70: bf9f0000
	s_code_end                                                 // 000000004a74: bf9f0000
	s_code_end                                                 // 000000004a78: bf9f0000
	s_code_end                                                 // 000000004a7c: bf9f0000
	s_code_end                                                 // 000000004a80: bf9f0000
	s_code_end                                                 // 000000004a84: bf9f0000
	s_code_end                                                 // 000000004a88: bf9f0000
	s_code_end                                                 // 000000004a8c: bf9f0000
	s_code_end                                                 // 000000004a90: bf9f0000
	s_code_end                                                 // 000000004a94: bf9f0000
	s_code_end                                                 // 000000004a98: bf9f0000
	s_code_end                                                 // 000000004a9c: bf9f0000
	s_code_end                                                 // 000000004aa0: bf9f0000
	s_code_end                                                 // 000000004aa4: bf9f0000
	s_code_end                                                 // 000000004aa8: bf9f0000
	s_code_end                                                 // 000000004aac: bf9f0000
	s_code_end                                                 // 000000004ab0: bf9f0000
	s_code_end                                                 // 000000004ab4: bf9f0000
	s_code_end                                                 // 000000004ab8: bf9f0000
	s_code_end                                                 // 000000004abc: bf9f0000
	s_code_end                                                 // 000000004ac0: bf9f0000
	s_code_end                                                 // 000000004ac4: bf9f0000
	s_code_end                                                 // 000000004ac8: bf9f0000
	s_code_end                                                 // 000000004acc: bf9f0000
	s_code_end                                                 // 000000004ad0: bf9f0000
	s_code_end                                                 // 000000004ad4: bf9f0000
	s_code_end                                                 // 000000004ad8: bf9f0000
	s_code_end                                                 // 000000004adc: bf9f0000
	s_code_end                                                 // 000000004ae0: bf9f0000
	s_code_end                                                 // 000000004ae4: bf9f0000
	s_code_end                                                 // 000000004ae8: bf9f0000
	s_code_end                                                 // 000000004aec: bf9f0000
	s_code_end                                                 // 000000004af0: bf9f0000
	s_code_end                                                 // 000000004af4: bf9f0000
	s_code_end                                                 // 000000004af8: bf9f0000
	s_code_end                                                 // 000000004afc: bf9f0000
	s_code_end                                                 // 000000004b00: bf9f0000
	s_code_end                                                 // 000000004b04: bf9f0000
	s_code_end                                                 // 000000004b08: bf9f0000
	s_code_end                                                 // 000000004b0c: bf9f0000
	s_code_end                                                 // 000000004b10: bf9f0000
	s_code_end                                                 // 000000004b14: bf9f0000
	s_code_end                                                 // 000000004b18: bf9f0000
	s_code_end                                                 // 000000004b1c: bf9f0000
	s_code_end                                                 // 000000004b20: bf9f0000
	s_code_end                                                 // 000000004b24: bf9f0000
	s_code_end                                                 // 000000004b28: bf9f0000
	s_code_end                                                 // 000000004b2c: bf9f0000
	s_code_end                                                 // 000000004b30: bf9f0000
	s_code_end                                                 // 000000004b34: bf9f0000
	s_code_end                                                 // 000000004b38: bf9f0000
	s_code_end                                                 // 000000004b3c: bf9f0000
	s_code_end                                                 // 000000004b40: bf9f0000
	s_code_end                                                 // 000000004b44: bf9f0000
	s_code_end                                                 // 000000004b48: bf9f0000
	s_code_end                                                 // 000000004b4c: bf9f0000
	s_code_end                                                 // 000000004b50: bf9f0000
	s_code_end                                                 // 000000004b54: bf9f0000
	s_code_end                                                 // 000000004b58: bf9f0000
	s_code_end                                                 // 000000004b5c: bf9f0000
	s_code_end                                                 // 000000004b60: bf9f0000
	s_code_end                                                 // 000000004b64: bf9f0000
	s_code_end                                                 // 000000004b68: bf9f0000
	s_code_end                                                 // 000000004b6c: bf9f0000
	s_code_end                                                 // 000000004b70: bf9f0000
	s_code_end                                                 // 000000004b74: bf9f0000
	s_code_end                                                 // 000000004b78: bf9f0000
	s_code_end                                                 // 000000004b7c: bf9f0000
