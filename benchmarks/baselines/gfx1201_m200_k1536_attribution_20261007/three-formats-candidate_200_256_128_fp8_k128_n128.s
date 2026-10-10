
/tmp/tmpnery042o.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_9160add00a4b6aae>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b128 s[12:15], s[0:1], 0xc8                         // 000000001b04: f4004300 f80000c8
	s_load_b64 s[16:17], s[0:1], 0xa8                          // 000000001b0c: f4002400 f80000a8
	v_dual_mov_b32 v21, 0 :: v_dual_and_b32 v26, 15, v0        // 000000001b14: ca240080 151a008f
	s_mov_b32 s2, ttmp9                                        // 000000001b1c: be820075
	s_mov_b32 s4, ttmp7                                        // 000000001b20: be840073
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b24: 86039f75
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b28: 86059f73
	s_clause 0x4                                               // 000000001b2c: bf850004
	s_load_b64 s[18:19], s[0:1], 0xd8                          // 000000001b30: f4002480 f80000d8
	s_load_b64 s[26:27], s[0:1], 0x8                           // 000000001b38: f4002680 f8000008
	s_load_b64 s[24:25], s[0:1], 0x30                          // 000000001b40: f4002600 f8000030
	s_load_b64 s[20:21], s[0:1], 0x58                          // 000000001b48: f4002500 f8000058
	s_load_b64 s[22:23], s[0:1], 0x80                          // 000000001b50: f4002580 f8000080
	s_lshl_b64 s[28:29], s[4:5], 4                             // 000000001b58: 849c8404
	s_lshl_b64 s[34:35], s[2:3], 5                             // 000000001b5c: 84a28502
	s_add_nc_u64 s[0:1], s[28:29], 16                          // 000000001b60: a980901c
	s_add_nc_u64 s[2:3], s[34:35], 32                          // 000000001b64: a982a022
	v_or_b32_e32 v16, s34, v26                                 // 000000001b68: 38203422
	v_lshrrev_b32_e32 v0, 1, v0                                // 000000001b6c: 32000081
	v_mov_b32_e32 v17, s35                                     // 000000001b70: 7e220223
	v_mov_b32_e32 v23, s35                                     // 000000001b74: 7e2e0223
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001b78: bf870214
	v_or_b32_e32 v22, 16, v16                                  // 000000001b7c: 382c2090
	v_and_b32_e32 v20, 8, v0                                   // 000000001b80: 36280088
	s_wait_kmcnt 0x0                                           // 000000001b84: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[12:13]                      // 000000001b88: d4540000 02001800
	v_cmp_gt_i64_e64 s1, s[2:3], s[14:15]                      // 000000001b90: d4540001 02001c02
	s_mov_b32 s2, -1                                           // 000000001b98: be8200c1
	s_add_nc_u64 s[36:37], s[14:15], 0x7f                      // 000000001b9c: a9a4ff0e 0000007f
	v_or_b32_e32 v18, s28, v20                                 // 000000001ba4: 3824281c
	s_lshr_b64 s[30:31], s[18:19], 7                           // 000000001ba8: 859e8712
	s_or_b32 s0, s0, s1                                        // 000000001bac: 8c000100
	v_cmp_gt_i64_e64 s1, s[14:15], v[16:17]                    // 000000001bb0: d4540001 0202200e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb8: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bbc: 8b6a007e
	v_cmp_gt_i64_e64 s0, s[14:15], v[22:23]                    // 000000001bc0: d4540000 02022c0e
	s_cbranch_vccnz 3                                          // 000000001bc8: bfa40003 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0xd8>
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001bcc: 8b6a027e
	s_cbranch_vccnz 2192                                       // 000000001bd0: bfa40890 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2314>
	s_endpgm                                                   // 000000001bd4: bfb00000
	v_or_b32_e32 v0, s28, v26                                  // 000000001bd8: 3800341c
	v_mov_b32_e32 v19, s29                                     // 000000001bdc: 7e26021d
	s_mul_i32 s3, s18, s29                                     // 000000001be0: 96031d12
	v_mul_lo_u32 v5, s19, v16                                  // 000000001be4: d72c0005 02022013
	v_mul_lo_u32 v6, s19, v22                                  // 000000001bec: d72c0006 02022c13
	v_mul_lo_u32 v4, s19, v0                                   // 000000001bf4: d72c0004 02020013
	v_mad_co_u64_u32 v[2:3], null, s18, v0, 0                  // 000000001bfc: d6fe7c02 02020012
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[18:19]                // 000000001c04: 7ca8240c
	v_mov_b32_e32 v1, s29                                      // 000000001c08: 7e02021d
	v_mul_lo_u32 v7, s18, v23                                  // 000000001c0c: d72c0007 02022e12
	s_lshr_b64 s[52:53], s[36:37], 7                           // 000000001c14: 85b48724
	s_lshr_b64 s[6:7], s[34:35], 7                             // 000000001c18: 85868722
	s_add_nc_u64 s[4:5], s[52:53], -1                          // 000000001c1c: a984c134
	v_cmp_gt_i64_e64 s2, s[12:13], v[0:1]                      // 000000001c20: d4540002 0202000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c28: bf88ff9e
	v_add3_u32 v37, v3, s3, v4                                 // 000000001c2c: d6550025 04100703
	v_mul_lo_u32 v4, s18, v17                                  // 000000001c34: d72c0004 02022212
	v_mad_co_u64_u32 v[0:1], null, s18, v16, 0                 // 000000001c3c: d6fe7c00 02022012
	v_or_b32_e32 v38, v2, v20                                  // 000000001c44: 384c2902
	v_mad_co_u64_u32 v[2:3], null, s18, v22, 0                 // 000000001c48: d6fe7c02 02022c12
	v_cmp_lt_u64_e64 s3, s[6:7], s[4:5]                        // 000000001c50: d4590003 02000806
	v_dual_mov_b32 v51, v21 :: v_dual_mov_b32 v36, v21         // 000000001c58: ca100115 33240115
	v_dual_mov_b32 v45, v21 :: v_dual_mov_b32 v34, v21         // 000000001c60: ca100115 2d220115
	v_add3_u32 v40, v1, v4, v5                                 // 000000001c68: d6550028 04160901
	v_or_b32_e32 v41, v0, v20                                  // 000000001c70: 38522900
	v_or_b32_e32 v0, 1, v18                                    // 000000001c74: 38002481
	v_dual_mov_b32 v1, s29 :: v_dual_cndmask_b32 v4, 0, v19    // 000000001c78: ca12001d 01042680
	v_cndmask_b32_e32 v5, 0, v18, vcc_lo                       // 000000001c80: 020a2480
	v_add3_u32 v42, v3, v7, v6                                 // 000000001c84: d655002a 041a0f03
	v_or_b32_e32 v44, v2, v20                                  // 000000001c8c: 38582902
	s_delay_alu instid0(valu_dep_4)                            // 000000001c90: bf870004
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001c94: 7ca8000c
	v_mul_lo_u32 v7, s30, v4                                   // 000000001c98: d72c0007 0202081e
	v_mul_lo_u32 v6, s31, v5                                   // 000000001ca0: d72c0006 02020a1f
	v_mad_co_u64_u32 v[3:4], null, s30, v5, 0                  // 000000001ca8: d6fe7c03 02020a1e
	s_and_b32 s3, s3, exec_lo                                  // 000000001cb0: 8b037e03
	v_dual_mov_b32 v43, v21 :: v_dual_mov_b32 v32, v21         // 000000001cb4: ca100115 2b200115
	s_wait_alu depctr_va_vcc(0)                                // 000000001cbc: bf88ff9d
	v_dual_cndmask_b32 v8, 0, v0 :: v_dual_cndmask_b32 v5, 0, v19// 000000001cc0: ca520080 08042680
	v_or_b32_e32 v0, 2, v18                                    // 000000001cc8: 38002482
	v_dual_mov_b32 v39, v21 :: v_dual_mov_b32 v30, v21         // 000000001ccc: ca100115 271e0115
	v_add3_u32 v4, v4, v7, v6                                  // 000000001cd4: d6550004 041a0f04
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001cdc: bf870214
	v_mul_lo_u32 v7, s31, v8                                   // 000000001ce0: d72c0007 0202101f
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001ce8: 7ca8000c
	v_mad_co_u64_u32 v[1:2], null, s30, v8, 0                  // 000000001cec: d6fe7c01 0202101e
	v_mov_b32_e32 v6, s29                                      // 000000001cf4: 7e0c021d
	v_mul_lo_u32 v9, s30, v5                                   // 000000001cf8: d72c0009 02020a1e
	v_or_b32_e32 v5, 3, v18                                    // 000000001d00: 380a2483
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]                        // 000000001d04: 3e060682
	s_wait_alu depctr_va_vcc(0)                                // 000000001d08: bf88ff9d
	v_cndmask_b32_e32 v8, 0, v19, vcc_lo                       // 000000001d0c: 02102680
	v_dual_cndmask_b32 v0, 0, v0 :: v_dual_mov_b32 v35, v21    // 000000001d10: ca500080 00220115
	v_mov_b32_e32 v28, v21                                     // 000000001d18: 7e380315
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000001d1c: 7ca80a0c
	v_add3_u32 v2, v2, v9, v7                                  // 000000001d20: d6550002 041e1302
	s_delay_alu instid0(valu_dep_4)                            // 000000001d28: bf870004
	v_mul_lo_u32 v9, s31, v0                                   // 000000001d2c: d72c0009 0202001f
	v_mul_lo_u32 v10, s30, v8                                  // 000000001d34: d72c000a 0202101e
	v_mad_co_u64_u32 v[7:8], null, s30, v0, 0                  // 000000001d3c: d6fe7c07 0202001e
	v_add_co_u32 v46, s3, s20, v3                              // 000000001d44: d700032e 02020614
	v_lshlrev_b64_e32 v[0:1], 2, v[1:2]                        // 000000001d4c: 3e000282
	v_or_b32_e32 v2, 4, v18                                    // 000000001d50: 38042484
	v_mov_b32_e32 v3, s29                                      // 000000001d54: 7e06021d
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)             // 000000001d58: bf88f19e
	v_add_co_ci_u32_e64 v47, null, s21, v4, s3                 // 000000001d5c: d5207c2f 000e0815
	s_wait_alu depctr_va_vcc(0)                                // 000000001d64: bf88ff9d
	v_dual_cndmask_b32 v4, 0, v19 :: v_dual_cndmask_b32 v5, 0, v5// 000000001d68: ca522680 04040a80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001d70: 7ca8040c
	v_add3_u32 v8, v8, v10, v9                                 // 000000001d74: d6550008 04261508
	v_add_co_u32 v49, s3, s20, v0                              // 000000001d7c: d7000331 02020014
	s_delay_alu instid0(valu_dep_4)                            // 000000001d84: bf870004
	v_mul_lo_u32 v6, s31, v5                                   // 000000001d88: d72c0006 02020a1f
	v_mul_lo_u32 v9, s30, v4                                   // 000000001d90: d72c0009 0202081e
	v_mad_co_u64_u32 v[4:5], null, s30, v5, 0                  // 000000001d98: d6fe7c04 02020a1e
	s_wait_alu depctr_va_vcc(0)                                // 000000001da0: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v2, vcc_lo                       // 000000001da4: 02160480
	v_or_b32_e32 v2, 5, v18                                    // 000000001da8: 38042485
	s_wait_alu depctr_va_sdst(0)                               // 000000001dac: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s21, v1, s3                 // 000000001db0: d5207c32 000e0215
	v_lshlrev_b64_e32 v[0:1], 2, v[7:8]                        // 000000001db8: 3e000e82
	v_cndmask_b32_e32 v10, 0, v19, vcc_lo                      // 000000001dbc: 02142680
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001dc0: 7ca8040c
	v_add3_u32 v5, v5, v9, v6                                  // 000000001dc4: d6550005 041a1305
	v_or_b32_e32 v8, 6, v18                                    // 000000001dcc: 38102486
	v_mov_b32_e32 v54, v21                                     // 000000001dd0: 7e6c0315
	v_add_co_u32 v52, s3, s20, v0                              // 000000001dd4: d7000334 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000001ddc: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s21, v1, s3                 // 000000001de0: d5207c35 000e0215
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001de8: 3e000882
	s_wait_alu depctr_va_vcc(0)                                // 000000001dec: bf88ff9d
	v_cndmask_b32_e32 v5, 0, v2, vcc_lo                        // 000000001df0: 020a0480
	v_or_b32_e32 v2, 7, v18                                    // 000000001df4: 38042487
	v_cndmask_b32_e32 v4, 0, v19, vcc_lo                       // 000000001df8: 02082680
	v_mul_lo_u32 v12, s31, v11                                 // 000000001dfc: d72c000c 0202161f
	v_mul_lo_u32 v10, s30, v10                                 // 000000001e04: d72c000a 0202141e
	v_mad_co_u64_u32 v[6:7], null, s30, v11, 0                 // 000000001e0c: d6fe7c06 0202161e
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001e14: 7ca8040c
	v_mov_b32_e32 v9, s29                                      // 000000001e18: 7e12021d
	v_mul_lo_u32 v11, s30, v4                                  // 000000001e1c: d72c000b 0202081e
	v_mad_co_u64_u32 v[3:4], null, s30, v5, 0                  // 000000001e24: d6fe7c03 02020a1e
	v_dual_mov_b32 v48, v21 :: v_dual_mov_b32 v33, v21         // 000000001e2c: ca100115 30200115
	s_wait_alu depctr_va_vcc(0)                                // 000000001e34: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo                        // 000000001e38: 02040480
	v_cmp_gt_i64_e64 s3, s[12:13], v[8:9]                      // 000000001e3c: d4540003 0202100c
	v_add3_u32 v7, v7, v10, v12                                // 000000001e44: d6550007 04321507
	v_mul_lo_u32 v10, s31, v5                                  // 000000001e4c: d72c000a 02020a1f
	v_cndmask_b32_e32 v5, 0, v19, vcc_lo                       // 000000001e54: 020a2680
	v_add_co_u32 v55, vcc_lo, s20, v0                          // 000000001e58: d7006a37 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000001e60: bf88f19f
	v_cndmask_b32_e64 v9, 0, v19, s3                           // 000000001e64: d5010009 000e2680
	v_cndmask_b32_e64 v8, 0, v8, s3                            // 000000001e6c: d5010008 000e1080
	s_wait_alu depctr_va_vcc(0)                                // 000000001e74: bf88ff9d
	v_add_co_ci_u32_e64 v56, null, s21, v1, vcc_lo             // 000000001e78: d5207c38 01aa0215
	v_add3_u32 v4, v4, v11, v10                                // 000000001e80: d6550004 042a1704
	v_mul_lo_u32 v13, s30, v9                                  // 000000001e88: d72c000d 0202121e
	v_mul_lo_u32 v12, s31, v8                                  // 000000001e90: d72c000c 0202101f
	v_mad_co_u64_u32 v[8:9], null, s30, v8, 0                  // 000000001e98: d6fe7c08 0202101e
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001ea0: 3e000c82
	v_mul_lo_u32 v7, s31, v2                                   // 000000001ea4: d72c0007 0202041f
	v_mul_lo_u32 v10, s30, v5                                  // 000000001eac: d72c000a 02020a1e
	v_mad_co_u64_u32 v[5:6], null, s30, v2, 0                  // 000000001eb4: d6fe7c05 0202041e
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]                        // 000000001ebc: 3e040682
	v_mov_b32_e32 v31, v21                                     // 000000001ec0: 7e3e0315
	v_add_co_u32 v57, vcc_lo, s20, v0                          // 000000001ec4: d7006a39 02020014
	v_add3_u32 v9, v9, v13, v12                                // 000000001ecc: d6550009 04321b09
	s_wait_alu depctr_va_vcc(0)                                // 000000001ed4: bf88ff9d
	v_add_co_ci_u32_e64 v58, null, s21, v1, vcc_lo             // 000000001ed8: d5207c3a 01aa0215
	v_add3_u32 v6, v6, v10, v7                                 // 000000001ee0: d6550006 041e1506
	v_add_co_u32 v59, vcc_lo, s20, v2                          // 000000001ee8: d7006a3b 02020414
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]                        // 000000001ef0: 3e001082
	s_wait_alu depctr_va_vcc(0)                                // 000000001ef4: bf88ff9d
	v_add_co_ci_u32_e64 v60, null, s21, v3, vcc_lo             // 000000001ef8: d5207c3c 01aa0615
	v_lshlrev_b64_e32 v[2:3], 2, v[5:6]                        // 000000001f00: 3e040a82
	v_mov_b32_e32 v29, v21                                     // 000000001f04: 7e3a0315
	v_mov_b32_e32 v27, v21                                     // 000000001f08: 7e360315
	v_add_co_u32 v61, vcc_lo, s20, v0                          // 000000001f0c: d7006a3d 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000001f14: bf88ff9d
	v_add_co_ci_u32_e64 v62, null, s21, v1, vcc_lo             // 000000001f18: d5207c3e 01aa0215
	v_add_co_u32 v63, vcc_lo, s20, v2                          // 000000001f20: d7006a3f 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000001f28: bf88ff9d
	v_add_co_ci_u32_e64 v64, null, s21, v3, vcc_lo             // 000000001f2c: d5207c40 01aa0615
	s_cselect_b32 s5, s7, s5                                   // 000000001f34: 98050507
	s_cselect_b32 s4, s6, s4                                   // 000000001f38: 98040406
	s_add_nc_u64 s[38:39], s[18:19], -1                        // 000000001f3c: a9a6c112
	s_add_nc_u64 s[40:41], s[18:19], -2                        // 000000001f40: a9a8c212
	s_add_nc_u64 s[42:43], s[18:19], -3                        // 000000001f44: a9aac312
	s_add_nc_u64 s[44:45], s[18:19], -4                        // 000000001f48: a9acc412
	s_add_nc_u64 s[46:47], s[18:19], -5                        // 000000001f4c: a9aec512
	s_add_nc_u64 s[48:49], s[18:19], -6                        // 000000001f50: a9b0c612
	s_add_nc_u64 s[50:51], s[18:19], -7                        // 000000001f54: a9b2c712
	s_mov_b64 s[58:59], 0                                      // 000000001f58: beba0180
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f5c: bf88ff9e
	s_lshl_b64 s[54:55], s[4:5], 2                             // 000000001f60: 84b68204
	v_mov_b32_e32 v0, 0                                        // 000000001f64: 7e000280
	s_add_nc_u64 s[56:57], s[58:59], 0x80                      // 000000001f68: a9b8ff3a 00000080
	s_mov_b64 s[60:61], s[58:59]                               // 000000001f70: bebc013a
	s_delay_alu instid0(valu_dep_1)                            // 000000001f74: bf870001
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0             // 000000001f78: ca100100 01020100
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0             // 000000001f80: ca100100 03040100
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0             // 000000001f88: ca100100 05060100
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v8, v0             // 000000001f90: ca100100 07080100
	v_dual_mov_b32 v9, v0 :: v_dual_mov_b32 v10, v0            // 000000001f98: ca100100 090a0100
	v_dual_mov_b32 v11, v0 :: v_dual_mov_b32 v12, v0           // 000000001fa0: ca100100 0b0c0100
	v_dual_mov_b32 v13, v0 :: v_dual_mov_b32 v14, v0           // 000000001fa8: ca100100 0d0e0100
	v_mov_b32_e32 v15, v0                                      // 000000001fb0: 7e1e0300
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fb4: bf88ff9e
	v_mov_b32_e32 v68, s61                                     // 000000001fb8: 7e88023d
	v_or_b32_e32 v67, s60, v20                                 // 000000001fbc: 3886283c
	v_add_co_u32 v71, vcc_lo, s60, v38                         // 000000001fc0: d7006a47 02024c3c
	s_wait_alu depctr_va_vcc(0)                                // 000000001fc8: bf88ff9d
	v_add_co_ci_u32_e64 v72, null, s61, v37, vcc_lo            // 000000001fcc: d5207c48 01aa4a3d
	s_delay_alu instid0(valu_dep_3)                            // 000000001fd4: bf870003
	v_cmp_gt_u64_e32 vcc_lo, s[18:19], v[67:68]                // 000000001fd8: 7cb88612
	s_or_b32 s33, s60, 16                                      // 000000001fdc: 8c21903c
	s_and_b32 s3, s2, vcc_lo                                   // 000000001fe0: 8b036a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fe4: bf88ff9e
	v_cndmask_b32_e64 v24, 0, v71, s3                          // 000000001fe8: d5010018 000e8e80
	v_cndmask_b32_e64 v25, 0, v72, s3                          // 000000001ff0: d5010019 000e9080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000001ff8: bf870122
	v_add_co_u32 v24, s4, s26, v24                             // 000000001ffc: d7000418 0202301a
	s_wait_alu depctr_va_sdst(0)                               // 000000002004: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s27, v25, s4                // 000000002008: d5207c19 0012321b
	global_load_d16_u8 v24, v[24:25], off                      // 000000002010: ee07807c 00000018 00000018
	v_or_b32_e32 v25, 1, v71                                   // 00000000201c: 38328e81
	s_wait_loadcnt 0x0                                         // 000000002020: bfc00000
	v_cndmask_b16 v24.l, 0, v24.l, s3                          // 000000002024: d65d0018 000e3080
	v_cmp_gt_i64_e64 s3, s[38:39], v[67:68]                    // 00000000202c: d4540003 02028626
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002034: bf870152
	v_and_b16 v24.l, 0xff, v24.l                               // 000000002038: d7620018 020230ff 000000ff
	s_and_b32 s4, s2, s3                                       // 000000002044: 8b040302
	s_wait_alu depctr_sa_sdst(0)                               // 000000002048: bf88ff9e
	v_cndmask_b32_e64 v25, 0, v25, s4                          // 00000000204c: d5010019 00123280
	v_cndmask_b32_e64 v66, 0, v72, s4                          // 000000002054: d5010042 00129080
	v_add_co_u32 v65, s5, s26, v25                             // 00000000205c: d7000541 0202321a
	s_wait_alu depctr_va_sdst(0)                               // 000000002064: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002068: bf870002
	v_add_co_ci_u32_e64 v66, null, s27, v66, s5                // 00000000206c: d5207c42 0016841b
	v_or_b32_e32 v25, 2, v71                                   // 000000002074: 38328e82
	global_load_d16_hi_u8 v24, v[65:66], off                   // 000000002078: ee08407c 00000018 00000041
	s_wait_loadcnt 0x0                                         // 000000002084: bfc00000
	v_cndmask_b16 v24.h, 0, v24.h, s4                          // 000000002088: d65d5018 00123080
	v_cmp_gt_i64_e64 s4, s[40:41], v[67:68]                    // 000000002090: d4540004 02028628
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002098: bf870152
	v_lshlrev_b16 v24.h, 8, v24.h op_sel:[0,1,1]               // 00000000209c: d7385018 02023088
	s_and_b32 s5, s2, s4                                       // 0000000020a4: 8b050402
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020a8: bf88ff9e
	v_cndmask_b32_e64 v25, 0, v25, s5                          // 0000000020ac: d5010019 00163280
	v_cndmask_b32_e64 v66, 0, v72, s5                          // 0000000020b4: d5010042 00169080
	v_add_co_u32 v65, s6, s26, v25                             // 0000000020bc: d7000641 0202321a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020c4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000020c8: bf870002
	v_add_co_ci_u32_e64 v66, null, s27, v66, s6                // 0000000020cc: d5207c42 001a841b
	global_load_d16_u8 v25, v[65:66], off                      // 0000000020d4: ee07807c 00000019 00000041
	v_or_b32_e32 v65, 3, v71                                   // 0000000020e0: 38828e83
	s_wait_loadcnt 0x0                                         // 0000000020e4: bfc00000
	v_cndmask_b16 v25.l, 0, v25.l, s5                          // 0000000020e8: d65d0019 00163280
	v_cmp_gt_i64_e64 s5, s[42:43], v[67:68]                    // 0000000020f0: d4540005 0202862a
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 0000000020f8: bf870152
	v_and_b16 v25.l, 0xff, v25.l                               // 0000000020fc: d7620019 020232ff 000000ff
	s_and_b32 s6, s2, s5                                       // 000000002108: 8b060502
	s_wait_alu depctr_sa_sdst(0)                               // 00000000210c: bf88ff9e
	v_cndmask_b32_e64 v65, 0, v65, s6                          // 000000002110: d5010041 001a8280
	v_cndmask_b32_e64 v66, 0, v72, s6                          // 000000002118: d5010042 001a9080
	v_add_co_u32 v65, s7, s26, v65                             // 000000002120: d7000741 0202821a
	s_wait_alu depctr_va_sdst(0)                               // 000000002128: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 00000000212c: bf870002
	v_add_co_ci_u32_e64 v66, null, s27, v66, s7                // 000000002130: d5207c42 001e841b
	global_load_d16_hi_u8 v25, v[65:66], off                   // 000000002138: ee08407c 00000019 00000041
	v_or_b32_e32 v65, 4, v71                                   // 000000002144: 38828e84
	s_wait_loadcnt 0x0                                         // 000000002148: bfc00000
	v_cndmask_b16 v25.h, 0, v25.h, s6                          // 00000000214c: d65d5019 001a3280
	v_cmp_gt_i64_e64 s6, s[44:45], v[67:68]                    // 000000002154: d4540006 0202862c
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 00000000215c: bf870152
	v_lshlrev_b16 v25.h, 8, v25.h op_sel:[0,1,1]               // 000000002160: d7385019 02023288
	s_and_b32 s7, s2, s6                                       // 000000002168: 8b070602
	s_wait_alu depctr_sa_sdst(0)                               // 00000000216c: bf88ff9e
	v_cndmask_b32_e64 v65, 0, v65, s7                          // 000000002170: d5010041 001e8280
	v_cndmask_b32_e64 v66, 0, v72, s7                          // 000000002178: d5010042 001e9080
	v_add_co_u32 v65, s8, s26, v65                             // 000000002180: d7000841 0202821a
	s_wait_alu depctr_va_sdst(0)                               // 000000002188: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 00000000218c: bf870002
	v_add_co_ci_u32_e64 v66, null, s27, v66, s8                // 000000002190: d5207c42 0022841b
	global_load_d16_u8 v65, v[65:66], off                      // 000000002198: ee07807c 00000041 00000041
	v_or_b32_e32 v66, 5, v71                                   // 0000000021a4: 38848e85
	s_wait_loadcnt 0x0                                         // 0000000021a8: bfc00000
	v_cndmask_b16 v65.l, 0, v65.l, s7                          // 0000000021ac: d65d0041 001e8280
	v_cmp_gt_i64_e64 s7, s[46:47], v[67:68]                    // 0000000021b4: d4540007 0202862e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 0000000021bc: bf870152
	v_and_b16 v65.l, 0xff, v65.l                               // 0000000021c0: d7620041 020282ff 000000ff
	s_and_b32 s8, s2, s7                                       // 0000000021cc: 8b080702
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021d0: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v66, s8                          // 0000000021d4: d5010042 00228480
	v_cndmask_b32_e64 v70, 0, v72, s8                          // 0000000021dc: d5010046 00229080
	v_add_co_u32 v69, s9, s26, v66                             // 0000000021e4: d7000945 0202841a
	s_wait_alu depctr_va_sdst(0)                               // 0000000021ec: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000021f0: bf870002
	v_add_co_ci_u32_e64 v70, null, s27, v70, s9                // 0000000021f4: d5207c46 00268c1b
	v_or_b32_e32 v66, 6, v71                                   // 0000000021fc: 38848e86
	global_load_d16_hi_u8 v65, v[69:70], off                   // 000000002200: ee08407c 00000041 00000045
	s_wait_loadcnt 0x0                                         // 00000000220c: bfc00000
	v_cndmask_b16 v65.h, 0, v65.h, s8                          // 000000002210: d65d5041 00228280
	v_cmp_gt_i64_e64 s8, s[48:49], v[67:68]                    // 000000002218: d4540008 02028630
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002220: bf870152
	v_lshlrev_b16 v65.h, 8, v65.h op_sel:[0,1,1]               // 000000002224: d7385041 02028288
	s_and_b32 s9, s2, s8                                       // 00000000222c: 8b090802
	s_wait_alu depctr_sa_sdst(0)                               // 000000002230: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v66, s9                          // 000000002234: d5010042 00268480
	v_cndmask_b32_e64 v70, 0, v72, s9                          // 00000000223c: d5010046 00269080
	v_add_co_u32 v69, s10, s26, v66                            // 000000002244: d7000a45 0202841a
	s_wait_alu depctr_va_sdst(0)                               // 00000000224c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002250: bf870002
	v_add_co_ci_u32_e64 v70, null, s27, v70, s10               // 000000002254: d5207c46 002a8c1b
	global_load_d16_u8 v66, v[69:70], off                      // 00000000225c: ee07807c 00000042 00000045
	v_or_b32_e32 v69, 7, v71                                   // 000000002268: 388a8e87
	s_wait_loadcnt 0x0                                         // 00000000226c: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, s9                          // 000000002270: d65d0042 00268480
	v_cmp_gt_i64_e64 s9, s[50:51], v[67:68]                    // 000000002278: d4540009 02028632
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002280: bf870152
	v_and_b16 v66.l, 0xff, v66.l                               // 000000002284: d7620042 020284ff 000000ff
	s_and_b32 s10, s2, s9                                      // 000000002290: 8b0a0902
	s_wait_alu depctr_sa_sdst(0)                               // 000000002294: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v69, s10                         // 000000002298: d5010043 002a8a80
	v_cndmask_b32_e64 v68, 0, v72, s10                         // 0000000022a0: d5010044 002a9080
	v_add_co_u32 v67, s11, s26, v67                            // 0000000022a8: d7000b43 0202861a
	s_wait_alu depctr_va_sdst(0)                               // 0000000022b0: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000022b4: bf870002
	v_add_co_ci_u32_e64 v68, null, s27, v68, s11               // 0000000022b8: d5207c44 002e881b
	global_load_d16_hi_u8 v66, v[67:68], off                   // 0000000022c0: ee08407c 00000042 00000043
	v_or_b16 v67.l, v24.l, v24.h op_sel:[0,1,0]                // 0000000022cc: d7631043 02023118
	v_or_b16 v67.h, v25.l, v25.h op_sel:[0,1,1]                // 0000000022d4: d7635043 02023319
	v_or_b16 v68.l, v65.l, v65.h op_sel:[0,1,0]                // 0000000022dc: d7631044 02028341
	s_wait_loadcnt 0x0                                         // 0000000022e4: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, s10                         // 0000000022e8: d65d5042 002a8480
	v_add_co_u32 v71, s10, s60, v41                            // 0000000022f0: d7000a47 0202523c
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s61, v40, s10               // 0000000022fc: d5207c48 002a503d
	s_and_b32 s10, s1, vcc_lo                                  // 000000002304: 8b0a6a01
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000002308: d7385042 02028488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002310: bf88ff9e
	v_cndmask_b32_e64 v24, 0, v71, s10                         // 000000002314: d5010018 002a8e80
	v_cndmask_b32_e64 v25, 0, v72, s10                         // 00000000231c: d5010019 002a9080
	s_and_b32 vcc_lo, s0, vcc_lo                               // 000000002324: 8b6a6a00
	v_or_b16 v68.h, v66.l, v66.h op_sel:[0,1,1]                // 000000002328: d7635044 02028542
	s_delay_alu instid0(valu_dep_3)                            // 000000002330: bf870003
	v_add_co_u32 v24, s11, s24, v24                            // 000000002334: d7000b18 02023018
	s_wait_alu depctr_va_sdst(0)                               // 00000000233c: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s25, v25, s11               // 000000002340: d5207c19 002e3219
	global_load_d16_u8 v24, v[24:25], off                      // 000000002348: ee07807c 00000018 00000018
	v_or_b32_e32 v25, 1, v71                                   // 000000002354: 38328e81
	s_wait_loadcnt 0x0                                         // 000000002358: bfc00000
	v_cndmask_b16 v24.l, 0, v24.l, s10                         // 00000000235c: d65d0018 002a3080
	s_and_b32 s10, s1, s3                                      // 000000002364: 8b0a0301
	s_wait_alu depctr_sa_sdst(0)                               // 000000002368: bf88ff9e
	v_cndmask_b32_e64 v25, 0, v25, s10                         // 00000000236c: d5010019 002a3280
	v_cndmask_b32_e64 v66, 0, v72, s10                         // 000000002374: d5010042 002a9080
	v_and_b16 v24.l, 0xff, v24.l                               // 00000000237c: d7620018 020230ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002388: bf8701a3
	v_add_co_u32 v65, s11, s24, v25                            // 00000000238c: d7000b41 02023218
	s_wait_alu depctr_va_sdst(0)                               // 000000002394: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s25, v66, s11               // 000000002398: d5207c42 002e8419
	v_or_b32_e32 v25, 2, v71                                   // 0000000023a0: 38328e82
	global_load_d16_hi_u8 v24, v[65:66], off                   // 0000000023a4: ee08407c 00000018 00000041
	s_wait_loadcnt 0x0                                         // 0000000023b0: bfc00000
	v_cndmask_b16 v24.h, 0, v24.h, s10                         // 0000000023b4: d65d5018 002a3080
	s_and_b32 s10, s1, s4                                      // 0000000023bc: 8b0a0401
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023c0: bf88ff9e
	v_cndmask_b32_e64 v25, 0, v25, s10                         // 0000000023c4: d5010019 002a3280
	v_cndmask_b32_e64 v66, 0, v72, s10                         // 0000000023cc: d5010042 002a9080
	v_lshlrev_b16 v24.h, 8, v24.h op_sel:[0,1,1]               // 0000000023d4: d7385018 02023088
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000023dc: bf8701a3
	v_add_co_u32 v65, s11, s24, v25                            // 0000000023e0: d7000b41 02023218
	s_wait_alu depctr_va_sdst(0)                               // 0000000023e8: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s25, v66, s11               // 0000000023ec: d5207c42 002e8419
	global_load_d16_u8 v25, v[65:66], off                      // 0000000023f4: ee07807c 00000019 00000041
	v_or_b32_e32 v65, 3, v71                                   // 000000002400: 38828e83
	s_wait_loadcnt 0x0                                         // 000000002404: bfc00000
	v_cndmask_b16 v25.l, 0, v25.l, s10                         // 000000002408: d65d0019 002a3280
	s_and_b32 s10, s1, s5                                      // 000000002410: 8b0a0501
	s_wait_alu depctr_sa_sdst(0)                               // 000000002414: bf88ff9e
	v_cndmask_b32_e64 v65, 0, v65, s10                         // 000000002418: d5010041 002a8280
	v_cndmask_b32_e64 v66, 0, v72, s10                         // 000000002420: d5010042 002a9080
	v_and_b16 v25.l, 0xff, v25.l                               // 000000002428: d7620019 020232ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002434: bf8701a3
	v_add_co_u32 v65, s11, s24, v65                            // 000000002438: d7000b41 02028218
	s_wait_alu depctr_va_sdst(0)                               // 000000002440: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s25, v66, s11               // 000000002444: d5207c42 002e8419
	global_load_d16_hi_u8 v25, v[65:66], off                   // 00000000244c: ee08407c 00000019 00000041
	v_or_b32_e32 v65, 4, v71                                   // 000000002458: 38828e84
	s_wait_loadcnt 0x0                                         // 00000000245c: bfc00000
	v_cndmask_b16 v25.h, 0, v25.h, s10                         // 000000002460: d65d5019 002a3280
	s_and_b32 s10, s1, s6                                      // 000000002468: 8b0a0601
	s_wait_alu depctr_sa_sdst(0)                               // 00000000246c: bf88ff9e
	v_cndmask_b32_e64 v65, 0, v65, s10                         // 000000002470: d5010041 002a8280
	v_cndmask_b32_e64 v66, 0, v72, s10                         // 000000002478: d5010042 002a9080
	v_lshlrev_b16 v25.h, 8, v25.h op_sel:[0,1,1]               // 000000002480: d7385019 02023288
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002488: bf8701a3
	v_add_co_u32 v65, s11, s24, v65                            // 00000000248c: d7000b41 02028218
	s_wait_alu depctr_va_sdst(0)                               // 000000002494: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s25, v66, s11               // 000000002498: d5207c42 002e8419
	global_load_d16_u8 v65, v[65:66], off                      // 0000000024a0: ee07807c 00000041 00000041
	v_or_b32_e32 v66, 5, v71                                   // 0000000024ac: 38848e85
	s_wait_loadcnt 0x0                                         // 0000000024b0: bfc00000
	v_cndmask_b16 v65.l, 0, v65.l, s10                         // 0000000024b4: d65d0041 002a8280
	s_and_b32 s10, s1, s7                                      // 0000000024bc: 8b0a0701
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024c0: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v66, s10                         // 0000000024c4: d5010042 002a8480
	v_cndmask_b32_e64 v70, 0, v72, s10                         // 0000000024cc: d5010046 002a9080
	v_and_b16 v65.l, 0xff, v65.l                               // 0000000024d4: d7620041 020282ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000024e0: bf8701a3
	v_add_co_u32 v69, s11, s24, v66                            // 0000000024e4: d7000b45 02028418
	s_wait_alu depctr_va_sdst(0)                               // 0000000024ec: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s25, v70, s11               // 0000000024f0: d5207c46 002e8c19
	v_or_b32_e32 v66, 6, v71                                   // 0000000024f8: 38848e86
	global_load_d16_hi_u8 v65, v[69:70], off                   // 0000000024fc: ee08407c 00000041 00000045
	s_wait_loadcnt 0x0                                         // 000000002508: bfc00000
	v_cndmask_b16 v65.h, 0, v65.h, s10                         // 00000000250c: d65d5041 002a8280
	s_and_b32 s10, s1, s8                                      // 000000002514: 8b0a0801
	s_wait_alu depctr_sa_sdst(0)                               // 000000002518: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v66, s10                         // 00000000251c: d5010042 002a8480
	v_cndmask_b32_e64 v70, 0, v72, s10                         // 000000002524: d5010046 002a9080
	v_lshlrev_b16 v65.h, 8, v65.h op_sel:[0,1,1]               // 00000000252c: d7385041 02028288
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002534: bf8701a3
	v_add_co_u32 v69, s11, s24, v66                            // 000000002538: d7000b45 02028418
	s_wait_alu depctr_va_sdst(0)                               // 000000002540: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s25, v70, s11               // 000000002544: d5207c46 002e8c19
	global_load_d16_u8 v66, v[69:70], off                      // 00000000254c: ee07807c 00000042 00000045
	v_or_b32_e32 v69, 7, v71                                   // 000000002558: 388a8e87
	s_wait_loadcnt 0x0                                         // 00000000255c: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, s10                         // 000000002560: d65d0042 002a8480
	s_and_b32 s10, s1, s9                                      // 000000002568: 8b0a0901
	s_wait_alu depctr_sa_sdst(0)                               // 00000000256c: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002570: d5010045 002a8a80
	v_cndmask_b32_e64 v70, 0, v72, s10                         // 000000002578: d5010046 002a9080
	v_and_b16 v66.l, 0xff, v66.l                               // 000000002580: d7620042 020284ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000258c: bf8701a3
	v_add_co_u32 v69, s11, s24, v69                            // 000000002590: d7000b45 02028a18
	s_wait_alu depctr_va_sdst(0)                               // 000000002598: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s25, v70, s11               // 00000000259c: d5207c46 002e8c19
	global_load_d16_hi_u8 v66, v[69:70], off                   // 0000000025a4: ee08407c 00000042 00000045
	v_or_b16 v69.l, v24.l, v24.h op_sel:[0,1,0]                // 0000000025b0: d7631045 02023118
	v_or_b16 v69.h, v25.l, v25.h op_sel:[0,1,1]                // 0000000025b8: d7635045 02023319
	v_or_b16 v70.l, v65.l, v65.h op_sel:[0,1,0]                // 0000000025c0: d7631046 02028341
	s_wait_loadcnt 0x0                                         // 0000000025c8: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, s10                         // 0000000025cc: d65d5042 002a8480
	v_add_co_u32 v73, s10, s60, v44                            // 0000000025d4: d7000a49 0202583c
	s_wait_alu depctr_va_sdst(0)                               // 0000000025dc: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s61, v42, s10               // 0000000025e0: d5207c4a 002a543d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000025e8: bf870113
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 0000000025ec: d7385042 02028488
	v_dual_cndmask_b32 v24, 0, v73 :: v_dual_cndmask_b32 v25, 0, v74// 0000000025f4: ca529280 18189480
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000025fc: bf870112
	v_or_b16 v70.h, v66.l, v66.h op_sel:[0,1,1]                // 000000002600: d7635046 02028542
	v_add_co_u32 v24, s10, s24, v24                            // 000000002608: d7000a18 02023018
	s_wait_alu depctr_va_sdst(0)                               // 000000002610: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002614: bf870193
	v_add_co_ci_u32_e64 v25, null, s25, v25, s10               // 000000002618: d5207c19 002a3219
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[67:68], v[69:70], v[0:7]// 000000002620: cc464000 1c028b43
	global_load_d16_u8 v24, v[24:25], off                      // 000000002628: ee07807c 00000018 00000018
	v_or_b32_e32 v25, 1, v73                                   // 000000002634: 38329281
	s_wait_loadcnt 0x0                                         // 000000002638: bfc00000
	v_cndmask_b16 v24.l, 0, v24.l, vcc_lo                      // 00000000263c: d65d0018 01aa3080
	s_and_b32 vcc_lo, s0, s3                                   // 000000002644: 8b6a0300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002648: bf88ff9e
	v_dual_cndmask_b32 v66, 0, v74 :: v_dual_cndmask_b32 v25, 0, v25// 00000000264c: ca529480 42183280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002654: bf870112
	v_and_b16 v24.l, 0xff, v24.l                               // 000000002658: d7620018 020230ff 000000ff
	v_add_co_u32 v65, s3, s24, v25                             // 000000002664: d7000341 02023218
	s_wait_alu depctr_va_sdst(0)                               // 00000000266c: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002670: bf870003
	v_add_co_ci_u32_e64 v66, null, s25, v66, s3                // 000000002674: d5207c42 000e8419
	v_or_b32_e32 v25, 2, v73                                   // 00000000267c: 38329282
	global_load_d16_hi_u8 v24, v[65:66], off                   // 000000002680: ee08407c 00000018 00000041
	s_wait_loadcnt 0x0                                         // 00000000268c: bfc00000
	v_cndmask_b16 v24.h, 0, v24.h, vcc_lo                      // 000000002690: d65d5018 01aa3080
	s_and_b32 vcc_lo, s0, s4                                   // 000000002698: 8b6a0400
	s_wait_alu depctr_sa_sdst(0)                               // 00000000269c: bf88ff9e
	v_dual_cndmask_b32 v66, 0, v74 :: v_dual_cndmask_b32 v25, 0, v25// 0000000026a0: ca529480 42183280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000026a8: bf870112
	v_lshlrev_b16 v24.h, 8, v24.h op_sel:[0,1,1]               // 0000000026ac: d7385018 02023088
	v_add_co_u32 v65, s3, s24, v25                             // 0000000026b4: d7000341 02023218
	s_wait_alu depctr_va_sdst(0)                               // 0000000026bc: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000026c0: bf870003
	v_add_co_ci_u32_e64 v66, null, s25, v66, s3                // 0000000026c4: d5207c42 000e8419
	global_load_d16_u8 v25, v[65:66], off                      // 0000000026cc: ee07807c 00000019 00000041
	v_or_b32_e32 v65, 3, v73                                   // 0000000026d8: 38829283
	s_wait_loadcnt 0x0                                         // 0000000026dc: bfc00000
	v_cndmask_b16 v25.l, 0, v25.l, vcc_lo                      // 0000000026e0: d65d0019 01aa3280
	s_and_b32 vcc_lo, s0, s5                                   // 0000000026e8: 8b6a0500
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026ec: bf88ff9e
	v_dual_cndmask_b32 v66, 0, v74 :: v_dual_cndmask_b32 v65, 0, v65// 0000000026f0: ca529480 42408280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000026f8: bf870112
	v_and_b16 v25.l, 0xff, v25.l                               // 0000000026fc: d7620019 020232ff 000000ff
	v_add_co_u32 v65, s3, s24, v65                             // 000000002708: d7000341 02028218
	s_wait_alu depctr_va_sdst(0)                               // 000000002710: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002714: bf870003
	v_add_co_ci_u32_e64 v66, null, s25, v66, s3                // 000000002718: d5207c42 000e8419
	global_load_d16_hi_u8 v25, v[65:66], off                   // 000000002720: ee08407c 00000019 00000041
	v_or_b32_e32 v65, 4, v73                                   // 00000000272c: 38829284
	s_wait_loadcnt 0x0                                         // 000000002730: bfc00000
	v_cndmask_b16 v25.h, 0, v25.h, vcc_lo                      // 000000002734: d65d5019 01aa3280
	s_and_b32 vcc_lo, s0, s6                                   // 00000000273c: 8b6a0600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002740: bf88ff9e
	v_dual_cndmask_b32 v66, 0, v74 :: v_dual_cndmask_b32 v65, 0, v65// 000000002744: ca529480 42408280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000274c: bf870112
	v_lshlrev_b16 v25.h, 8, v25.h op_sel:[0,1,1]               // 000000002750: d7385019 02023288
	v_add_co_u32 v65, s3, s24, v65                             // 000000002758: d7000341 02028218
	s_wait_alu depctr_va_sdst(0)                               // 000000002760: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002764: bf870003
	v_add_co_ci_u32_e64 v66, null, s25, v66, s3                // 000000002768: d5207c42 000e8419
	global_load_d16_u8 v65, v[65:66], off                      // 000000002770: ee07807c 00000041 00000041
	v_or_b32_e32 v66, 5, v73                                   // 00000000277c: 38849285
	s_wait_loadcnt 0x0                                         // 000000002780: bfc00000
	v_cndmask_b16 v65.l, 0, v65.l, vcc_lo                      // 000000002784: d65d0041 01aa8280
	s_and_b32 vcc_lo, s0, s7                                   // 00000000278c: 8b6a0700
	s_wait_alu depctr_sa_sdst(0)                               // 000000002790: bf88ff9e
	v_cndmask_b32_e32 v66, 0, v66, vcc_lo                      // 000000002794: 02848480
	v_cndmask_b32_e32 v72, 0, v74, vcc_lo                      // 000000002798: 02909480
	v_and_b16 v65.l, 0xff, v65.l                               // 00000000279c: d7620041 020282ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000027a8: bf8701a3
	v_add_co_u32 v71, s3, s24, v66                             // 0000000027ac: d7000347 02028418
	s_wait_alu depctr_va_sdst(0)                               // 0000000027b4: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s25, v72, s3                // 0000000027b8: d5207c48 000e9019
	v_or_b32_e32 v66, 6, v73                                   // 0000000027c0: 38849286
	global_load_d16_hi_u8 v65, v[71:72], off                   // 0000000027c4: ee08407c 00000041 00000047
	s_wait_loadcnt 0x0                                         // 0000000027d0: bfc00000
	v_cndmask_b16 v65.h, 0, v65.h, vcc_lo                      // 0000000027d4: d65d5041 01aa8280
	s_and_b32 vcc_lo, s0, s8                                   // 0000000027dc: 8b6a0800
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027e0: bf88ff9e
	v_cndmask_b32_e32 v66, 0, v66, vcc_lo                      // 0000000027e4: 02848480
	v_cndmask_b32_e32 v72, 0, v74, vcc_lo                      // 0000000027e8: 02909480
	v_lshlrev_b16 v65.h, 8, v65.h op_sel:[0,1,1]               // 0000000027ec: d7385041 02028288
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000027f4: bf8701a3
	v_add_co_u32 v71, s3, s24, v66                             // 0000000027f8: d7000347 02028418
	s_wait_alu depctr_va_sdst(0)                               // 000000002800: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s25, v72, s3                // 000000002804: d5207c48 000e9019
	global_load_d16_u8 v66, v[71:72], off                      // 00000000280c: ee07807c 00000042 00000047
	v_or_b32_e32 v71, 7, v73                                   // 000000002818: 388e9287
	s_wait_loadcnt 0x0                                         // 00000000281c: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, vcc_lo                      // 000000002820: d65d0042 01aa8480
	s_and_b32 vcc_lo, s0, s9                                   // 000000002828: 8b6a0900
	s_wait_alu depctr_sa_sdst(0)                               // 00000000282c: bf88ff9e
	v_dual_cndmask_b32 v71, 0, v71 :: v_dual_cndmask_b32 v72, 0, v74// 000000002830: ca528e80 47489480
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002838: bf870112
	v_and_b16 v66.l, 0xff, v66.l                               // 00000000283c: d7620042 020284ff 000000ff
	v_add_co_u32 v71, s3, s24, v71                             // 000000002848: d7000347 02028e18
	s_wait_alu depctr_va_sdst(0)                               // 000000002850: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002854: bf870003
	v_add_co_ci_u32_e64 v72, null, s25, v72, s3                // 000000002858: d5207c48 000e9019
	global_load_d16_hi_u8 v66, v[71:72], off                   // 000000002860: ee08407c 00000042 00000047
	s_wait_loadcnt 0x0                                         // 00000000286c: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, vcc_lo                      // 000000002870: d65d5042 01aa8480
	v_add_co_u32 v71, vcc_lo, s33, v38                         // 000000002878: d7006a47 02024c21
	s_wait_alu depctr_va_vcc(0)                                // 000000002880: bf88ff9d
	v_add_co_ci_u32_e64 v72, null, s61, v37, vcc_lo            // 000000002884: d5207c48 01aa4a3d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 00000000288c: bf870093
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000002890: d7385042 02028488
	v_or_b16 v66.h, v66.l, v66.h op_sel:[0,1,1]                // 000000002898: d7635042 02028542
	v_or_b16 v66.l, v65.l, v65.h op_sel:[0,1,0]                // 0000000028a0: d7631042 02028341
	v_or_b16 v65.h, v25.l, v25.h op_sel:[0,1,1]                // 0000000028a8: d7635041 02023319
	v_or_b16 v65.l, v24.l, v24.h op_sel:[0,1,0]                // 0000000028b0: d7631041 02023118
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 0000000028b8: bf8700b1
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[67:68], v[65:66], v[8:15]// 0000000028bc: cc464008 1c228343
	v_mov_b32_e32 v68, s61                                     // 0000000028c4: 7e88023d
	v_or_b32_e32 v67, s33, v20                                 // 0000000028c8: 38862821
	v_cmp_gt_u64_e64 s9, s[18:19], v[67:68]                    // 0000000028cc: d45c0009 02028612
	v_cmp_gt_i64_e64 s8, s[38:39], v[67:68]                    // 0000000028d4: d4540008 02028626
	v_cmp_gt_i64_e64 s7, s[40:41], v[67:68]                    // 0000000028dc: d4540007 02028628
	v_cmp_gt_i64_e64 s6, s[42:43], v[67:68]                    // 0000000028e4: d4540006 0202862a
	v_cmp_gt_i64_e64 s5, s[44:45], v[67:68]                    // 0000000028ec: d4540005 0202862c
	v_cmp_gt_i64_e64 s4, s[46:47], v[67:68]                    // 0000000028f4: d4540004 0202862e
	s_and_b32 vcc_lo, s2, s9                                   // 0000000028fc: 8b6a0902
	s_wait_alu depctr_sa_sdst(0)                               // 000000002900: bf88ff9e
	v_dual_cndmask_b32 v24, 0, v71 :: v_dual_cndmask_b32 v25, 0, v72// 000000002904: ca528e80 18189080
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 00000000290c: bf870121
	v_add_co_u32 v24, s3, s26, v24                             // 000000002910: d7000318 0202301a
	s_wait_alu depctr_va_sdst(0)                               // 000000002918: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s27, v25, s3                // 00000000291c: d5207c19 000e321b
	global_load_d16_u8 v24, v[24:25], off                      // 000000002924: ee07807c 00000018 00000018
	v_or_b32_e32 v25, 1, v71                                   // 000000002930: 38328e81
	s_wait_loadcnt 0x0                                         // 000000002934: bfc00000
	v_cndmask_b16 v24.l, 0, v24.l, vcc_lo                      // 000000002938: d65d0018 01aa3080
	s_and_b32 vcc_lo, s2, s8                                   // 000000002940: 8b6a0802
	s_wait_alu depctr_sa_sdst(0)                               // 000000002944: bf88ff9e
	v_dual_cndmask_b32 v25, 0, v25 :: v_dual_cndmask_b32 v66, 0, v72// 000000002948: ca523280 19429080
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002950: bf870112
	v_and_b16 v24.l, 0xff, v24.l                               // 000000002954: d7620018 020230ff 000000ff
	v_add_co_u32 v65, s3, s26, v25                             // 000000002960: d7000341 0202321a
	s_wait_alu depctr_va_sdst(0)                               // 000000002968: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 00000000296c: bf870003
	v_add_co_ci_u32_e64 v66, null, s27, v66, s3                // 000000002970: d5207c42 000e841b
	v_or_b32_e32 v25, 2, v71                                   // 000000002978: 38328e82
	global_load_d16_hi_u8 v24, v[65:66], off                   // 00000000297c: ee08407c 00000018 00000041
	s_wait_loadcnt 0x0                                         // 000000002988: bfc00000
	v_cndmask_b16 v65.l, 0, v24.h, vcc_lo                      // 00000000298c: d65d1041 01aa3080
	s_and_b32 vcc_lo, s2, s7                                   // 000000002994: 8b6a0702
	s_wait_alu depctr_sa_sdst(0)                               // 000000002998: bf88ff9e
	v_dual_cndmask_b32 v25, 0, v25 :: v_dual_cndmask_b32 v66, 0, v72// 00000000299c: ca523280 19429080
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000029a4: bf870112
	v_lshlrev_b16 v65.l, 8, v65.l                              // 0000000029a8: d7380041 02028288
	v_add_co_u32 v69, s3, s26, v25                             // 0000000029b0: d7000345 0202321a
	s_wait_alu depctr_va_sdst(0)                               // 0000000029b8: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000029bc: bf870003
	v_add_co_ci_u32_e64 v70, null, s27, v66, s3                // 0000000029c0: d5207c46 000e841b
	v_or_b32_e32 v25, 3, v71                                   // 0000000029c8: 38328e83
	v_or_b16 v24.l, v24.l, v65.l                               // 0000000029cc: d7630018 02028318
	global_load_d16_hi_u8 v24, v[69:70], off                   // 0000000029d4: ee08407c 00000018 00000045
	s_wait_loadcnt 0x0                                         // 0000000029e0: bfc00000
	v_cndmask_b16 v24.h, 0, v24.h, vcc_lo                      // 0000000029e4: d65d5018 01aa3080
	s_and_b32 vcc_lo, s2, s6                                   // 0000000029ec: 8b6a0602
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029f0: bf88ff9e
	v_dual_cndmask_b32 v25, 0, v25 :: v_dual_cndmask_b32 v66, 0, v72// 0000000029f4: ca523280 19429080
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000029fc: bf870112
	v_and_b16 v24.h, 0xff, v24.h op_sel:[0,1,1]                // 000000002a00: d7625018 020230ff 000000ff
	v_add_co_u32 v69, s3, s26, v25                             // 000000002a0c: d7000345 0202321a
	s_wait_alu depctr_va_sdst(0)                               // 000000002a14: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002a18: bf870003
	v_add_co_ci_u32_e64 v70, null, s27, v66, s3                // 000000002a1c: d5207c46 000e841b
	global_load_d16_u8 v25, v[69:70], off                      // 000000002a24: ee07807c 00000019 00000045
	s_wait_loadcnt 0x0                                         // 000000002a30: bfc00000
	v_cndmask_b16 v65.h, 0, v25.l, vcc_lo                      // 000000002a34: d65d4041 01aa3280
	v_or_b32_e32 v25, 4, v71                                   // 000000002a3c: 38328e84
	s_and_b32 vcc_lo, s2, s5                                   // 000000002a40: 8b6a0502
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a44: bf88ff9e
	v_cndmask_b32_e32 v66, 0, v72, vcc_lo                      // 000000002a48: 02849080
	v_lshlrev_b16 v65.h, 8, v65.h op_sel:[0,1,1]               // 000000002a4c: d7385041 02028288
	v_cndmask_b32_e32 v25, 0, v25, vcc_lo                      // 000000002a54: 02323280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002a58: bf870112
	v_or_b16 v24.h, v24.h, v65.h op_sel:[1,1,1]                // 000000002a5c: d7635818 02028318
	v_add_co_u32 v69, s3, s26, v25                             // 000000002a64: d7000345 0202321a
	s_wait_alu depctr_va_sdst(0)                               // 000000002a6c: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s27, v66, s3                // 000000002a70: d5207c46 000e841b
	v_or_b32_e32 v66, 5, v71                                   // 000000002a78: 38848e85
	global_load_d16_u8 v25, v[69:70], off                      // 000000002a7c: ee07807c 00000019 00000045
	s_wait_loadcnt 0x0                                         // 000000002a88: bfc00000
	v_cndmask_b16 v25.l, 0, v25.l, vcc_lo                      // 000000002a8c: d65d0019 01aa3280
	s_and_b32 vcc_lo, s2, s4                                   // 000000002a94: 8b6a0402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a98: bf88ff9e
	v_cndmask_b32_e32 v66, 0, v66, vcc_lo                      // 000000002a9c: 02848480
	v_cndmask_b32_e32 v70, 0, v72, vcc_lo                      // 000000002aa0: 028c9080
	v_and_b16 v25.l, 0xff, v25.l                               // 000000002aa4: d7620019 020232ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002ab0: bf8701a3
	v_add_co_u32 v69, s3, s26, v66                             // 000000002ab4: d7000345 0202841a
	s_wait_alu depctr_va_sdst(0)                               // 000000002abc: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s27, v70, s3                // 000000002ac0: d5207c46 000e8c1b
	v_cmp_gt_i64_e64 s3, s[48:49], v[67:68]                    // 000000002ac8: d4540003 02028630
	global_load_d16_hi_u8 v25, v[69:70], off                   // 000000002ad0: ee08407c 00000019 00000045
	v_or_b32_e32 v69, 6, v71                                   // 000000002adc: 388a8e86
	s_wait_loadcnt 0x0                                         // 000000002ae0: bfc00000
	v_cndmask_b16 v66.l, 0, v25.h, vcc_lo                      // 000000002ae4: d65d1042 01aa3280
	s_and_b32 vcc_lo, s2, s3                                   // 000000002aec: 8b6a0302
	s_wait_alu depctr_sa_sdst(0)                               // 000000002af0: bf88ff9e
	v_dual_cndmask_b32 v69, 0, v69 :: v_dual_cndmask_b32 v70, 0, v72// 000000002af4: ca528a80 45469080
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002afc: bf870112
	v_lshlrev_b16 v66.l, 8, v66.l                              // 000000002b00: d7380042 02028488
	v_add_co_u32 v69, s10, s26, v69                            // 000000002b08: d7000a45 02028a1a
	s_wait_alu depctr_va_sdst(0)                               // 000000002b10: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002b14: bf870193
	v_add_co_ci_u32_e64 v70, null, s27, v70, s10               // 000000002b18: d5207c46 002a8c1b
	v_or_b16 v25.l, v25.l, v66.l                               // 000000002b20: d7630019 02028519
	global_load_d16_hi_u8 v25, v[69:70], off                   // 000000002b28: ee08407c 00000019 00000045
	v_or_b32_e32 v69, 7, v71                                   // 000000002b34: 388a8e87
	s_wait_loadcnt 0x0                                         // 000000002b38: bfc00000
	v_cndmask_b16 v25.h, 0, v25.h, vcc_lo                      // 000000002b3c: d65d5019 01aa3280
	v_cmp_gt_i64_e32 vcc_lo, s[50:51], v[67:68]                // 000000002b44: 7ca88632
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002b48: bf870152
	v_and_b16 v25.h, 0xff, v25.h op_sel:[0,1,1]                // 000000002b4c: d7625019 020232ff 000000ff
	s_and_b32 s10, s2, vcc_lo                                  // 000000002b58: 8b0a6a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b5c: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v69, s10                         // 000000002b60: d5010043 002a8a80
	v_cndmask_b32_e64 v68, 0, v72, s10                         // 000000002b68: d5010044 002a9080
	v_add_co_u32 v67, s11, s26, v67                            // 000000002b70: d7000b43 0202861a
	s_wait_alu depctr_va_sdst(0)                               // 000000002b78: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002b7c: bf870002
	v_add_co_ci_u32_e64 v68, null, s27, v68, s11               // 000000002b80: d5207c44 002e881b
	global_load_d16_hi_u8 v66, v[67:68], off                   // 000000002b88: ee08407c 00000042 00000043
	s_wait_loadcnt 0x0                                         // 000000002b94: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, s10                         // 000000002b98: d65d5042 002a8480
	v_add_co_u32 v70, s10, s33, v41                            // 000000002ba0: d7000a46 02025221
	s_wait_alu depctr_va_sdst(0)                               // 000000002ba8: bf88f19f
	v_add_co_ci_u32_e64 v71, null, s61, v40, s10               // 000000002bac: d5207c47 002a503d
	s_delay_alu instid0(valu_dep_3)                            // 000000002bb4: bf870003
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000002bb8: d7385042 02028488
	s_and_b32 s10, s1, s9                                      // 000000002bc0: 8b0a0901
	s_and_b32 s9, s0, s9                                       // 000000002bc4: 8b090900
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bc8: bf88ff9e
	v_cndmask_b32_e64 v65, 0, v70, s10                         // 000000002bcc: d5010041 002a8c80
	v_or_b16 v25.h, v25.h, v66.h op_sel:[1,1,1]                // 000000002bd4: d7635819 02028519
	v_cndmask_b32_e64 v66, 0, v71, s10                         // 000000002bdc: d5010042 002a8e80
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 000000002be4: bf870123
	v_add_co_u32 v65, s11, s24, v65                            // 000000002be8: d7000b41 02028218
	s_wait_alu depctr_va_sdst(0)                               // 000000002bf0: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s25, v66, s11               // 000000002bf4: d5207c42 002e8419
	global_load_d16_u8 v65, v[65:66], off                      // 000000002bfc: ee07807c 00000041 00000041
	v_or_b32_e32 v66, 1, v70                                   // 000000002c08: 38848c81
	s_wait_loadcnt 0x0                                         // 000000002c0c: bfc00000
	v_cndmask_b16 v65.l, 0, v65.l, s10                         // 000000002c10: d65d0041 002a8280
	s_and_b32 s10, s1, s8                                      // 000000002c18: 8b0a0801
	s_and_b32 s8, s0, s8                                       // 000000002c1c: 8b080800
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c20: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v66, s10                         // 000000002c24: d5010042 002a8480
	v_cndmask_b32_e64 v67, 0, v71, s10                         // 000000002c2c: d5010043 002a8e80
	v_and_b16 v65.l, 0xff, v65.l                               // 000000002c34: d7620041 020282ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002c40: bf8701a3
	v_add_co_u32 v66, s11, s24, v66                            // 000000002c44: d7000b42 02028418
	s_wait_alu depctr_va_sdst(0)                               // 000000002c4c: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s25, v67, s11               // 000000002c50: d5207c43 002e8619
	global_load_d16_hi_u8 v65, v[66:67], off                   // 000000002c58: ee08407c 00000041 00000042
	v_or_b32_e32 v66, 2, v70                                   // 000000002c64: 38848c82
	s_wait_loadcnt 0x0                                         // 000000002c68: bfc00000
	v_cndmask_b16 v65.h, 0, v65.h, s10                         // 000000002c6c: d65d5041 002a8280
	s_and_b32 s10, s1, s7                                      // 000000002c74: 8b0a0701
	s_and_b32 s7, s0, s7                                       // 000000002c78: 8b070700
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c7c: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v66, s10                         // 000000002c80: d5010042 002a8480
	v_cndmask_b32_e64 v67, 0, v71, s10                         // 000000002c88: d5010043 002a8e80
	v_lshlrev_b16 v65.h, 8, v65.h op_sel:[0,1,1]               // 000000002c90: d7385041 02028288
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002c98: bf8701a3
	v_add_co_u32 v66, s11, s24, v66                            // 000000002c9c: d7000b42 02028418
	s_wait_alu depctr_va_sdst(0)                               // 000000002ca4: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s25, v67, s11               // 000000002ca8: d5207c43 002e8619
	global_load_d16_u8 v66, v[66:67], off                      // 000000002cb0: ee07807c 00000042 00000042
	v_or_b32_e32 v67, 3, v70                                   // 000000002cbc: 38868c83
	s_wait_loadcnt 0x0                                         // 000000002cc0: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, s10                         // 000000002cc4: d65d0042 002a8480
	s_and_b32 s10, s1, s6                                      // 000000002ccc: 8b0a0601
	s_and_b32 s6, s0, s6                                       // 000000002cd0: 8b060600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cd4: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v67, s10                         // 000000002cd8: d5010043 002a8680
	v_cndmask_b32_e64 v68, 0, v71, s10                         // 000000002ce0: d5010044 002a8e80
	v_and_b16 v66.l, 0xff, v66.l                               // 000000002ce8: d7620042 020284ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002cf4: bf8701a3
	v_add_co_u32 v67, s11, s24, v67                            // 000000002cf8: d7000b43 02028618
	s_wait_alu depctr_va_sdst(0)                               // 000000002d00: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s25, v68, s11               // 000000002d04: d5207c44 002e8819
	global_load_d16_hi_u8 v66, v[67:68], off                   // 000000002d0c: ee08407c 00000042 00000043
	v_or_b32_e32 v67, 4, v70                                   // 000000002d18: 38868c84
	s_wait_loadcnt 0x0                                         // 000000002d1c: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, s10                         // 000000002d20: d65d5042 002a8480
	s_and_b32 s10, s1, s5                                      // 000000002d28: 8b0a0501
	s_and_b32 s5, s0, s5                                       // 000000002d2c: 8b050500
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d30: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v67, s10                         // 000000002d34: d5010043 002a8680
	v_cndmask_b32_e64 v68, 0, v71, s10                         // 000000002d3c: d5010044 002a8e80
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000002d44: d7385042 02028488
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002d4c: bf8701a3
	v_add_co_u32 v67, s11, s24, v67                            // 000000002d50: d7000b43 02028618
	s_wait_alu depctr_va_sdst(0)                               // 000000002d58: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s25, v68, s11               // 000000002d5c: d5207c44 002e8819
	global_load_d16_u8 v67, v[67:68], off                      // 000000002d64: ee07807c 00000043 00000043
	v_or_b32_e32 v68, 5, v70                                   // 000000002d70: 38888c85
	s_wait_loadcnt 0x0                                         // 000000002d74: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, s10                         // 000000002d78: d65d0043 002a8680
	s_and_b32 s10, s1, s4                                      // 000000002d80: 8b0a0401
	s_and_b32 s4, s0, s4                                       // 000000002d84: 8b040400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d88: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v68, s10                         // 000000002d8c: d5010044 002a8880
	v_cndmask_b32_e64 v69, 0, v71, s10                         // 000000002d94: d5010045 002a8e80
	v_and_b16 v67.l, 0xff, v67.l                               // 000000002d9c: d7620043 020286ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002da8: bf8701a3
	v_add_co_u32 v68, s11, s24, v68                            // 000000002dac: d7000b44 02028818
	s_wait_alu depctr_va_sdst(0)                               // 000000002db4: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s25, v69, s11               // 000000002db8: d5207c45 002e8a19
	global_load_d16_hi_u8 v67, v[68:69], off                   // 000000002dc0: ee08407c 00000043 00000044
	v_or_b32_e32 v68, 6, v70                                   // 000000002dcc: 38888c86
	s_wait_loadcnt 0x0                                         // 000000002dd0: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, s10                         // 000000002dd4: d65d5043 002a8680
	s_and_b32 s10, s1, s3                                      // 000000002ddc: 8b0a0301
	s_and_b32 s3, s0, s3                                       // 000000002de0: 8b030300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002de4: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v68, s10                         // 000000002de8: d5010044 002a8880
	v_cndmask_b32_e64 v69, 0, v71, s10                         // 000000002df0: d5010045 002a8e80
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 000000002df8: d7385043 02028688
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002e00: bf8701a3
	v_add_co_u32 v68, s11, s24, v68                            // 000000002e04: d7000b44 02028818
	s_wait_alu depctr_va_sdst(0)                               // 000000002e0c: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s25, v69, s11               // 000000002e10: d5207c45 002e8a19
	global_load_d16_u8 v68, v[68:69], off                      // 000000002e18: ee07807c 00000044 00000044
	v_or_b32_e32 v69, 7, v70                                   // 000000002e24: 388a8c87
	s_wait_loadcnt 0x0                                         // 000000002e28: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, s10                         // 000000002e2c: d65d0044 002a8880
	s_and_b32 s10, s1, vcc_lo                                  // 000000002e34: 8b0a6a01
	s_and_b32 vcc_lo, s0, vcc_lo                               // 000000002e38: 8b6a6a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e3c: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002e40: d5010045 002a8a80
	v_cndmask_b32_e64 v70, 0, v71, s10                         // 000000002e48: d5010046 002a8e80
	v_and_b16 v68.l, 0xff, v68.l                               // 000000002e50: d7620044 020288ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002e5c: bf8701a3
	v_add_co_u32 v69, s11, s24, v69                            // 000000002e60: d7000b45 02028a18
	s_wait_alu depctr_va_sdst(0)                               // 000000002e68: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s25, v70, s11               // 000000002e6c: d5207c46 002e8c19
	global_load_d16_hi_u8 v68, v[69:70], off                   // 000000002e74: ee08407c 00000044 00000045
	v_or_b16 v69.l, v65.l, v65.h op_sel:[0,1,0]                // 000000002e80: d7631045 02028341
	v_or_b16 v69.h, v66.l, v66.h op_sel:[0,1,1]                // 000000002e88: d7635045 02028542
	v_or_b16 v70.l, v67.l, v67.h op_sel:[0,1,0]                // 000000002e90: d7631046 02028743
	s_wait_loadcnt 0x0                                         // 000000002e98: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, s10                         // 000000002e9c: d65d5044 002a8880
	v_add_co_u32 v73, s10, s33, v44                            // 000000002ea4: d7000a49 02025821
	s_wait_alu depctr_va_sdst(0)                               // 000000002eac: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s61, v42, s10               // 000000002eb0: d5207c4a 002a543d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002eb8: bf870193
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 000000002ebc: d7385044 02028888
	v_cndmask_b32_e64 v65, 0, v73, s9                          // 000000002ec4: d5010041 00269280
	s_add_nc_u64 s[60:61], s[60:61], 32                        // 000000002ecc: a9bca03c
	s_delay_alu instid0(valu_dep_3)                            // 000000002ed0: bf870003
	v_cndmask_b32_e64 v66, 0, v74, s9                          // 000000002ed4: d5010042 00269480
	v_cndmask_b32_e64 v67, 0, v74, s8                          // 000000002edc: d5010043 00229480
	v_or_b16 v70.h, v68.l, v68.h op_sel:[0,1,1]                // 000000002ee4: d7635046 02028944
	v_add_co_u32 v65, s10, s24, v65                            // 000000002eec: d7000a41 02028218
	s_wait_alu depctr_va_sdst(0)                               // 000000002ef4: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s25, v66, s10               // 000000002ef8: d5207c42 002a8419
	v_cndmask_b32_e64 v68, 0, v74, s6                          // 000000002f00: d5010044 001a9480
	v_cndmask_b32_e64 v72, 0, v74, s4                          // 000000002f08: d5010048 00129480
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[24:25], v[69:70], v[0:7]// 000000002f10: cc464000 1c028b18
	global_load_d16_u8 v65, v[65:66], off                      // 000000002f18: ee07807c 00000041 00000041
	v_or_b32_e32 v66, 1, v73                                   // 000000002f24: 38849281
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_2)// 000000002f28: bf870131
	v_cndmask_b32_e64 v66, 0, v66, s8                          // 000000002f2c: d5010042 00228480
	s_wait_loadcnt 0x0                                         // 000000002f34: bfc00000
	v_cndmask_b16 v65.l, 0, v65.l, s9                          // 000000002f38: d65d0041 00268280
	v_add_co_u32 v66, s9, s24, v66                             // 000000002f40: d7000942 02028418
	s_wait_alu depctr_va_sdst(0)                               // 000000002f48: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s25, v67, s9                // 000000002f4c: d5207c43 00268619
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000002f54: bf870143
	v_and_b16 v65.l, 0xff, v65.l                               // 000000002f58: d7620041 020282ff 000000ff
	global_load_d16_hi_u8 v65, v[66:67], off                   // 000000002f64: ee08407c 00000041 00000042
	v_or_b32_e32 v66, 2, v73                                   // 000000002f70: 38849282
	v_cndmask_b32_e64 v67, 0, v74, s7                          // 000000002f74: d5010043 001e9480
	v_cndmask_b32_e64 v66, 0, v66, s7                          // 000000002f7c: d5010042 001e8480
	s_wait_loadcnt 0x0                                         // 000000002f84: bfc00000
	v_cndmask_b16 v65.h, 0, v65.h, s8                          // 000000002f88: d65d5041 00228280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000002f90: bf8701b2
	v_add_co_u32 v66, s8, s24, v66                             // 000000002f94: d7000842 02028418
	s_wait_alu depctr_va_sdst(0)                               // 000000002f9c: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s25, v67, s8                // 000000002fa0: d5207c43 00228619
	v_lshlrev_b16 v65.h, 8, v65.h op_sel:[0,1,1]               // 000000002fa8: d7385041 02028288
	global_load_d16_u8 v66, v[66:67], off                      // 000000002fb0: ee07807c 00000042 00000042
	v_or_b32_e32 v67, 3, v73                                   // 000000002fbc: 38869283
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_2)// 000000002fc0: bf870131
	v_cndmask_b32_e64 v67, 0, v67, s6                          // 000000002fc4: d5010043 001a8680
	s_wait_loadcnt 0x0                                         // 000000002fcc: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, s7                          // 000000002fd0: d65d0042 001e8480
	v_add_co_u32 v67, s7, s24, v67                             // 000000002fd8: d7000743 02028618
	s_wait_alu depctr_va_sdst(0)                               // 000000002fe0: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s25, v68, s7                // 000000002fe4: d5207c44 001e8819
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000002fec: bf870143
	v_and_b16 v66.l, 0xff, v66.l                               // 000000002ff0: d7620042 020284ff 000000ff
	global_load_d16_hi_u8 v66, v[67:68], off                   // 000000002ffc: ee08407c 00000042 00000043
	v_or_b32_e32 v67, 4, v73                                   // 000000003008: 38869284
	v_cndmask_b32_e64 v68, 0, v74, s5                          // 00000000300c: d5010044 00169480
	v_cndmask_b32_e64 v67, 0, v67, s5                          // 000000003014: d5010043 00168680
	s_wait_loadcnt 0x0                                         // 00000000301c: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, s6                          // 000000003020: d65d5042 001a8480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000003028: bf8701b2
	v_add_co_u32 v67, s6, s24, v67                             // 00000000302c: d7000643 02028618
	s_wait_alu depctr_va_sdst(0)                               // 000000003034: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s25, v68, s6                // 000000003038: d5207c44 001a8819
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000003040: d7385042 02028488
	global_load_d16_u8 v67, v[67:68], off                      // 000000003048: ee07807c 00000043 00000043
	v_or_b32_e32 v68, 5, v73                                   // 000000003054: 38889285
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_2)// 000000003058: bf870131
	v_cndmask_b32_e64 v68, 0, v68, s4                          // 00000000305c: d5010044 00128880
	s_wait_loadcnt 0x0                                         // 000000003064: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, s5                          // 000000003068: d65d0043 00168680
	v_add_co_u32 v71, s5, s24, v68                             // 000000003070: d7000547 02028818
	s_wait_alu depctr_va_sdst(0)                               // 000000003078: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s25, v72, s5                // 00000000307c: d5207c48 00169019
	v_or_b32_e32 v68, 6, v73                                   // 000000003084: 38889286
	v_and_b16 v67.l, 0xff, v67.l                               // 000000003088: d7620043 020286ff 000000ff
	global_load_d16_hi_u8 v67, v[71:72], off                   // 000000003094: ee08407c 00000043 00000047
	v_cndmask_b32_e64 v72, 0, v74, s3                          // 0000000030a0: d5010048 000e9480
	v_cndmask_b32_e64 v68, 0, v68, s3                          // 0000000030a8: d5010044 000e8880
	s_wait_loadcnt 0x0                                         // 0000000030b0: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, s4                          // 0000000030b4: d65d5043 00128680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 0000000030bc: bf8701b2
	v_add_co_u32 v71, s4, s24, v68                             // 0000000030c0: d7000447 02028818
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s25, v72, s4                // 0000000030cc: d5207c48 00129019
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 0000000030d4: d7385043 02028688
	global_load_d16_u8 v68, v[71:72], off                      // 0000000030dc: ee07807c 00000044 00000047
	v_or_b32_e32 v71, 7, v73                                   // 0000000030e8: 388e9287
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_2)// 0000000030ec: bf870131
	v_dual_cndmask_b32 v72, 0, v74 :: v_dual_cndmask_b32 v71, 0, v71// 0000000030f0: ca529480 48468e80
	s_wait_loadcnt 0x0                                         // 0000000030f8: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, s3                          // 0000000030fc: d65d0044 000e8880
	v_add_co_u32 v71, s3, s24, v71                             // 000000003104: d7000347 02028e18
	s_wait_alu depctr_va_sdst(0)                               // 00000000310c: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s25, v72, s3                // 000000003110: d5207c48 000e9019
	s_delay_alu instid0(valu_dep_3)                            // 000000003118: bf870003
	v_and_b16 v68.l, 0xff, v68.l                               // 00000000311c: d7620044 020288ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003128: bf88ff9e
	v_cmp_lt_u64_e64 s3, s[60:61], s[56:57]                    // 00000000312c: d4590003 0200703c
	global_load_d16_hi_u8 v68, v[71:72], off                   // 000000003134: ee08407c 00000044 00000047
	s_wait_loadcnt 0x0                                         // 000000003140: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, vcc_lo                      // 000000003144: d65d5044 01aa8880
	s_and_b32 vcc_lo, exec_lo, s3                              // 00000000314c: 8b6a037e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003150: bf870091
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 000000003154: d7385044 02028888
	v_or_b16 v68.h, v68.l, v68.h op_sel:[0,1,1]                // 00000000315c: d7635044 02028944
	v_or_b16 v68.l, v67.l, v67.h op_sel:[0,1,0]                // 000000003164: d7631044 02028743
	v_or_b16 v67.h, v66.l, v66.h op_sel:[0,1,1]                // 00000000316c: d7635043 02028542
	v_or_b16 v67.l, v65.l, v65.h op_sel:[0,1,0]                // 000000003174: d7631043 02028341
	s_delay_alu instid0(valu_dep_1)                            // 00000000317c: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[24:25], v[67:68], v[8:15]// 000000003180: cc464008 1c228718
	s_wait_alu depctr_sa_sdst(0)                               // 000000003188: bf88ff9e
	s_cbranch_vccnz 64393                                      // 00000000318c: bfa4fb89 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x4b4>
	s_lshr_b64 s[4:5], s[58:59], 5                             // 000000003190: 8584853a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003194: bf88ff9e
	v_add_co_u32 v24, vcc_lo, v46, s4                          // 000000003198: d7006a18 0200092e
	s_wait_alu depctr_va_vcc(0)                                // 0000000031a0: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v47, vcc_lo             // 0000000031a4: d5207c19 01aa5e05
	v_add_co_u32 v65, vcc_lo, v49, s4                          // 0000000031ac: d7006a41 02000931
	s_wait_alu depctr_va_vcc(0)                                // 0000000031b4: bf88ff9d
	v_add_co_ci_u32_e64 v66, null, s5, v50, vcc_lo             // 0000000031b8: d5207c42 01aa6405
	v_add_co_u32 v67, vcc_lo, v52, s4                          // 0000000031c0: d7006a43 02000934
	s_wait_alu depctr_va_vcc(0)                                // 0000000031c8: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s5, v53, vcc_lo             // 0000000031cc: d5207c44 01aa6a05
	v_add_co_u32 v69, vcc_lo, v55, s4                          // 0000000031d4: d7006a45 02000937
	s_wait_alu depctr_va_vcc(0)                                // 0000000031dc: bf88ff9d
	v_add_co_ci_u32_e64 v70, null, s5, v56, vcc_lo             // 0000000031e0: d5207c46 01aa7005
	v_add_co_u32 v71, vcc_lo, v57, s4                          // 0000000031e8: d7006a47 02000939
	s_wait_alu depctr_va_vcc(0)                                // 0000000031f0: bf88ff9d
	v_add_co_ci_u32_e64 v72, null, s5, v58, vcc_lo             // 0000000031f4: d5207c48 01aa7405
	v_add_co_u32 v73, vcc_lo, v59, s4                          // 0000000031fc: d7006a49 0200093b
	s_wait_alu depctr_va_vcc(0)                                // 000000003204: bf88ff9d
	v_add_co_ci_u32_e64 v74, null, s5, v60, vcc_lo             // 000000003208: d5207c4a 01aa7805
	v_add_co_u32 v75, vcc_lo, v61, s4                          // 000000003210: d7006a4b 0200093d
	s_wait_alu depctr_va_vcc(0)                                // 000000003218: bf88ff9d
	v_add_co_ci_u32_e64 v76, null, s5, v62, vcc_lo             // 00000000321c: d5207c4c 01aa7c05
	v_add_co_u32 v77, vcc_lo, v63, s4                          // 000000003224: d7006a4d 0200093f
	s_wait_alu depctr_va_vcc(0)                                // 00000000322c: bf88ff9d
	v_add_co_ci_u32_e64 v78, null, s5, v64, vcc_lo             // 000000003230: d5207c4e 01aa8005
	s_clause 0x7                                               // 000000003238: bf850007
	global_load_b32 v24, v[24:25], off                         // 00000000323c: ee05007c 00000018 00000018
	global_load_b32 v25, v[65:66], off                         // 000000003248: ee05007c 00000019 00000041
	global_load_b32 v65, v[67:68], off                         // 000000003254: ee05007c 00000041 00000043
	global_load_b32 v66, v[69:70], off                         // 000000003260: ee05007c 00000042 00000045
	global_load_b32 v67, v[71:72], off                         // 00000000326c: ee05007c 00000043 00000047
	global_load_b32 v68, v[73:74], off                         // 000000003278: ee05007c 00000044 00000049
	global_load_b32 v69, v[75:76], off                         // 000000003284: ee05007c 00000045 0000004b
	global_load_b32 v70, v[77:78], off                         // 000000003290: ee05007c 00000046 0000004d
	s_lshr_b64 s[4:5], s[58:59], 7                             // 00000000329c: 8584873a
	s_mov_b64 s[58:59], s[56:57]                               // 0000000032a0: beba0138
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a4: bf88ff9e
	s_mul_u64 s[4:5], s[4:5], s[52:53]                         // 0000000032a8: aa843404
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032ac: bf88ff9e
	s_lshl_b64 s[4:5], s[4:5], 2                               // 0000000032b0: 84848204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032b4: bf88ff9e
	s_add_nc_u64 s[4:5], s[22:23], s[4:5]                      // 0000000032b8: a9840416
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032bc: bf88ff9e
	s_add_nc_u64 s[6:7], s[4:5], s[54:55]                      // 0000000032c0: a9863604
	s_clause 0x1                                               // 0000000032c4: bf850001
	s_load_b32 s3, s[6:7], 0x0                                 // 0000000032c8: f40000c3 f8000000
	s_load_b32 s4, s[4:5], 0x0                                 // 0000000032d0: f4000102 f8000000
	s_wait_kmcnt 0x0                                           // 0000000032d8: bfc70000
	v_mov_b32_e32 v71, s3                                      // 0000000032dc: 7e8e0203
	v_cmp_lt_u64_e64 s3, s[56:57], s[18:19]                    // 0000000032e0: d4590003 02002438
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_1)// 0000000032e8: bf8700b2
	v_cndmask_b32_e64 v72, s4, v71, s1                         // 0000000032ec: d5010048 00068e04
	s_and_b32 vcc_lo, exec_lo, s3                              // 0000000032f4: 8b6a037e
	s_wait_loadcnt 0x6                                         // 0000000032f8: bfc00006
	v_mul_f32_e32 v74, v72, v25                                // 0000000032fc: 10943348
	v_cndmask_b32_e64 v71, s4, v71, s0                         // 000000003300: d5010047 00028e04
	v_mul_f32_e32 v73, v24, v72                                // 000000003308: 10929118
	s_wait_loadcnt 0x4                                         // 00000000330c: bfc00004
	v_dual_mul_f32 v75, v72, v65 :: v_dual_mul_f32 v76, v72, v66// 000000003310: c8c68348 4b4c8548
	s_wait_loadcnt 0x2                                         // 000000003318: bfc00002
	v_dual_mul_f32 v77, v72, v67 :: v_dual_mul_f32 v78, v72, v68// 00000000331c: c8c68748 4d4e8948
	s_wait_loadcnt 0x0                                         // 000000003324: bfc00000
	v_dual_mul_f32 v79, v72, v69 :: v_dual_mul_f32 v72, v72, v70// 000000003328: c8c68b48 4f488d48
	v_dual_mul_f32 v24, v24, v71 :: v_dual_mul_f32 v25, v71, v25// 000000003330: c8c68f18 18183347
	v_dual_mul_f32 v66, v71, v66 :: v_dual_mul_f32 v65, v71, v65// 000000003338: c8c68547 42408347
	v_dual_mul_f32 v68, v71, v68 :: v_dual_mul_f32 v67, v71, v67// 000000003340: c8c68947 44428747
	v_dual_mul_f32 v70, v71, v70 :: v_dual_mul_f32 v69, v71, v69// 000000003348: c8c68d47 46448b47
	v_mul_f32_e32 v2, v2, v75                                  // 000000003350: 10049702
	v_dual_mul_f32 v0, v0, v73 :: v_dual_mul_f32 v1, v1, v74   // 000000003354: c8c69300 00009501
	v_dual_mul_f32 v3, v3, v76 :: v_dual_mul_f32 v4, v4, v77   // 00000000335c: c8c69903 03049b04
	v_dual_mul_f32 v5, v5, v78 :: v_dual_mul_f32 v6, v6, v79   // 000000003364: c8c69d05 05069f06
	v_dual_mul_f32 v7, v7, v72 :: v_dual_mul_f32 v10, v10, v65 // 00000000336c: c8c69107 070a830a
	v_dual_mul_f32 v8, v8, v24 :: v_dual_mul_f32 v9, v9, v25   // 000000003374: c8c63108 08083309
	v_dual_mul_f32 v11, v11, v66 :: v_dual_mul_f32 v12, v12, v67// 00000000337c: c8c6850b 0b0c870c
	v_dual_mul_f32 v13, v13, v68 :: v_dual_mul_f32 v14, v14, v69// 000000003384: c8c6890d 0d0e8b0e
	v_dual_mul_f32 v15, v15, v70 :: v_dual_add_f32 v54, v54, v0// 00000000338c: c8c88d0f 0f360136
	v_dual_add_f32 v51, v51, v1 :: v_dual_add_f32 v48, v48, v2 // 000000003394: c9080333 33300530
	v_dual_add_f32 v45, v45, v3 :: v_dual_add_f32 v36, v36, v6 // 00000000339c: c908072d 2d240d24
	v_dual_add_f32 v43, v43, v4 :: v_dual_add_f32 v32, v32, v10// 0000000033a4: c908092b 2b201520
	v_dual_add_f32 v39, v39, v5 :: v_dual_add_f32 v34, v34, v8 // 0000000033ac: c9080b27 27221122
	v_dual_add_f32 v35, v35, v7 :: v_dual_add_f32 v30, v30, v12// 0000000033b4: c9080f23 231e191e
	v_dual_add_f32 v33, v33, v9 :: v_dual_add_f32 v28, v28, v14// 0000000033bc: c9081321 211c1d1c
	v_add_f32_e32 v31, v31, v11                                // 0000000033c4: 063e171f
	v_add_f32_e32 v29, v29, v13                                // 0000000033c8: 063a1b1d
	v_add_f32_e32 v27, v27, v15                                // 0000000033cc: 06361f1b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033d0: bf88ff9e
	s_cbranch_vccnz 64227                                      // 0000000033d4: bfa4fae3 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x464>
	v_mul_lo_u32 v4, s15, v18                                  // 0000000033d8: d72c0004 0202240f
	v_mul_lo_u32 v5, s14, v19                                  // 0000000033e0: d72c0005 0202260e
	v_mad_co_u64_u32 v[0:1], null, s14, v18, 0                 // 0000000033e8: d6fe7c00 0202240e
	v_sub_co_u32 v2, vcc_lo, s12, v18                          // 0000000033f0: d7016a02 0202240c
	s_wait_alu depctr_va_vcc(0)                                // 0000000033f8: bf88ff9d
	v_sub_co_ci_u32_e64 v3, null, s13, v19, vcc_lo             // 0000000033fc: d5217c03 01aa260d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003404: bf870211
	v_cmp_lt_i64_e32 vcc_lo, 0, v[2:3]                         // 000000003408: 7ca20480
	v_add3_u32 v1, v1, v5, v4                                  // 00000000340c: d6550001 04120b01
	s_delay_alu instid0(valu_dep_1)                            // 000000003414: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003418: 3e000081
	s_and_b32 s2, vcc_lo, s1                                   // 00000000341c: 8b02016a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003420: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003424: be832002
	s_cbranch_execz 28                                         // 000000003428: bfa5001c <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x199c>
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 00000000342c: 3e082081
	v_add_co_u32 v7, s2, s16, v0                               // 000000003430: d7000207 02020010
	v_bfe_u32 v6, v54, 16, 1                                   // 000000003438: d6100006 02052136
	s_wait_alu depctr_va_sdst(0)                               // 000000003440: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, v1, s2                  // 000000003444: d5207c08 000a0211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000344c: bf870193
	v_add_co_u32 v4, s2, v7, v4                                // 000000003450: d7000204 02020907
	v_add3_u32 v6, v6, v54, 0x7fff                             // 000000003458: d6550006 03fe6d06 00007fff
	v_or_b32_e32 v9, 0x400000, v54                             // 000000003464: 38126cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000346c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003470: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v54, v54                               // 000000003478: d4180002 02026d36
	s_wait_alu depctr_va_sdst(0)                               // 000000003480: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003484: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003488: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003490: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000349c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000034a0: 8c7e037e
	v_cmp_lt_i64_e64 s2, 1, v[2:3]                             // 0000000034a4: d4510002 02020481
	s_and_b32 s3, s2, s1                                       // 0000000034ac: 8b030102
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034b0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000034b4: be842003
	s_cbranch_execz 35                                         // 0000000034b8: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1a48>
	v_add_co_u32 v6, s3, s16, v0                               // 0000000034bc: d7000306 02020010
	s_wait_alu depctr_va_sdst(0)                               // 0000000034c4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s3                  // 0000000034c8: d5207c07 000e0211
	s_lshl_b64 s[6:7], s[14:15], 1                             // 0000000034d0: 8486810e
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 0000000034d4: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034d8: bf88ff9e
	v_add_co_u32 v6, s3, v6, s6                                // 0000000034dc: d7000306 02000d06
	v_bfe_u32 v8, v51, 16, 1                                   // 0000000034e4: d6100008 02052133
	s_wait_alu depctr_va_sdst(0)                               // 0000000034ec: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s7, v7, s3                   // 0000000034f0: d5207c07 000e0e07
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000034f8: bf870193
	v_add_co_u32 v4, s3, v6, v4                                // 0000000034fc: d7000304 02020906
	v_add3_u32 v8, v8, v51, 0x7fff                             // 000000003504: d6550008 03fe6708 00007fff
	v_or_b32_e32 v9, 0x400000, v51                             // 000000003510: 381266ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003518: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s3                   // 00000000351c: d5207c05 000e0b07
	v_cmp_u_f32_e64 s3, v51, v51                               // 000000003524: d4180003 02026733
	s_wait_alu depctr_va_sdst(0)                               // 00000000352c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003530: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s3                           // 000000003534: d5010006 000e1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000353c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003548: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000354c: 8c7e047e
	v_cmp_lt_i64_e64 s3, 2, v[2:3]                             // 000000003550: d4510003 02020482
	s_lshl_b64 s[10:11], s[14:15], 1                           // 000000003558: 848a810e
	s_and_b32 s4, s3, s1                                       // 00000000355c: 8b040103
	s_wait_alu depctr_sa_sdst(0)                               // 000000003560: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000003564: be852004
	s_cbranch_execz 35                                         // 000000003568: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1af8>
	v_add_co_u32 v6, s4, s16, v0                               // 00000000356c: d7000406 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003574: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s4                  // 000000003578: d5207c07 00120211
	s_lshl_b64 s[6:7], s[10:11], 1                             // 000000003580: 8486810a
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003584: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003588: bf88ff9e
	v_add_co_u32 v6, s4, v6, s6                                // 00000000358c: d7000406 02000d06
	v_bfe_u32 v8, v48, 16, 1                                   // 000000003594: d6100008 02052130
	s_wait_alu depctr_va_sdst(0)                               // 00000000359c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s7, v7, s4                   // 0000000035a0: d5207c07 00120e07
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000035a8: bf870193
	v_add_co_u32 v4, s4, v6, v4                                // 0000000035ac: d7000404 02020906
	v_add3_u32 v8, v8, v48, 0x7fff                             // 0000000035b4: d6550008 03fe6108 00007fff
	v_or_b32_e32 v9, 0x400000, v48                             // 0000000035c0: 381260ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s4                   // 0000000035cc: d5207c05 00120b07
	v_cmp_u_f32_e64 s4, v48, v48                               // 0000000035d4: d4180004 02026130
	s_wait_alu depctr_va_sdst(0)                               // 0000000035dc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000035e0: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s4                           // 0000000035e4: d5010006 00121308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000035ec: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000035fc: 8c7e057e
	v_cmp_lt_i64_e64 s4, 3, v[2:3]                             // 000000003600: d4510004 02020483
	s_mul_u64 s[38:39], s[14:15], 3                            // 000000003608: aaa6830e
	s_and_b32 s5, s4, s1                                       // 00000000360c: 8b050104
	s_wait_alu depctr_sa_sdst(0)                               // 000000003610: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 000000003614: be862005
	s_cbranch_execz 35                                         // 000000003618: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1ba8>
	v_add_co_u32 v6, s5, s16, v0                               // 00000000361c: d7000506 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003624: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s5                  // 000000003628: d5207c07 00160211
	s_lshl_b64 s[8:9], s[38:39], 1                             // 000000003630: 84888126
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003634: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003638: bf88ff9e
	v_add_co_u32 v6, s5, v6, s8                                // 00000000363c: d7000506 02001106
	v_bfe_u32 v8, v45, 16, 1                                   // 000000003644: d6100008 0205212d
	s_wait_alu depctr_va_sdst(0)                               // 00000000364c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s9, v7, s5                   // 000000003650: d5207c07 00160e09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003658: bf870193
	v_add_co_u32 v4, s5, v6, v4                                // 00000000365c: d7000504 02020906
	v_add3_u32 v8, v8, v45, 0x7fff                             // 000000003664: d6550008 03fe5b08 00007fff
	v_or_b32_e32 v9, 0x400000, v45                             // 000000003670: 38125aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003678: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s5                   // 00000000367c: d5207c05 00160b07
	v_cmp_u_f32_e64 s5, v45, v45                               // 000000003684: d4180005 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 00000000368c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003690: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s5                           // 000000003694: d5010006 00161308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000369c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000036ac: 8c7e067e
	v_cmp_lt_i64_e64 s5, 4, v[2:3]                             // 0000000036b0: d4510005 02020484
	s_lshl_b64 s[40:41], s[14:15], 2                           // 0000000036b8: 84a8820e
	s_and_b32 s6, s5, s1                                       // 0000000036bc: 8b060105
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036c0: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 0000000036c4: be872006
	s_cbranch_execz 35                                         // 0000000036c8: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1c58>
	v_add_co_u32 v6, s6, s16, v0                               // 0000000036cc: d7000606 02020010
	s_wait_alu depctr_va_sdst(0)                               // 0000000036d4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s6                  // 0000000036d8: d5207c07 001a0211
	s_lshl_b64 s[8:9], s[40:41], 1                             // 0000000036e0: 84888128
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 0000000036e4: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036e8: bf88ff9e
	v_add_co_u32 v6, s6, v6, s8                                // 0000000036ec: d7000606 02001106
	v_bfe_u32 v8, v43, 16, 1                                   // 0000000036f4: d6100008 0205212b
	s_wait_alu depctr_va_sdst(0)                               // 0000000036fc: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s9, v7, s6                   // 000000003700: d5207c07 001a0e09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003708: bf870193
	v_add_co_u32 v4, s6, v6, v4                                // 00000000370c: d7000604 02020906
	v_add3_u32 v8, v8, v43, 0x7fff                             // 000000003714: d6550008 03fe5708 00007fff
	v_or_b32_e32 v9, 0x400000, v43                             // 000000003720: 381256ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003728: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s6                   // 00000000372c: d5207c05 001a0b07
	v_cmp_u_f32_e64 s6, v43, v43                               // 000000003734: d4180006 0202572b
	s_wait_alu depctr_va_sdst(0)                               // 00000000373c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003740: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s6                           // 000000003744: d5010006 001a1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000374c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003758: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 00000000375c: 8c7e077e
	v_cmp_lt_i64_e64 s6, 5, v[2:3]                             // 000000003760: d4510006 02020485
	s_mul_u64 s[42:43], s[14:15], 5                            // 000000003768: aaaa850e
	s_and_b32 s7, s6, s1                                       // 00000000376c: 8b070106
	s_wait_alu depctr_sa_sdst(0)                               // 000000003770: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 000000003774: be882007
	s_cbranch_execz 35                                         // 000000003778: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1d08>
	v_add_co_u32 v6, s7, s16, v0                               // 00000000377c: d7000706 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003784: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s7                  // 000000003788: d5207c07 001e0211
	s_lshl_b64 s[44:45], s[42:43], 1                           // 000000003790: 84ac812a
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003794: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003798: bf88ff9e
	v_add_co_u32 v6, s7, v6, s44                               // 00000000379c: d7000706 02005906
	v_bfe_u32 v8, v39, 16, 1                                   // 0000000037a4: d6100008 02052127
	s_wait_alu depctr_va_sdst(0)                               // 0000000037ac: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s45, v7, s7                  // 0000000037b0: d5207c07 001e0e2d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000037b8: bf870193
	v_add_co_u32 v4, s7, v6, v4                                // 0000000037bc: d7000704 02020906
	v_add3_u32 v8, v8, v39, 0x7fff                             // 0000000037c4: d6550008 03fe4f08 00007fff
	v_or_b32_e32 v9, 0x400000, v39                             // 0000000037d0: 38124eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000037d8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s7                   // 0000000037dc: d5207c05 001e0b07
	v_cmp_u_f32_e64 s7, v39, v39                               // 0000000037e4: d4180007 02024f27
	s_wait_alu depctr_va_sdst(0)                               // 0000000037ec: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000037f0: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s7                           // 0000000037f4: d5010006 001e1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000037fc: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003808: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 00000000380c: 8c7e087e
	v_cmp_lt_i64_e64 s7, 6, v[2:3]                             // 000000003810: d4510007 02020486
	s_mul_u64 s[44:45], s[14:15], 6                            // 000000003818: aaac860e
	s_and_b32 s8, s7, s1                                       // 00000000381c: 8b080107
	s_wait_alu depctr_sa_sdst(0)                               // 000000003820: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003824: be892008
	s_cbranch_execz 35                                         // 000000003828: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1db8>
	v_add_co_u32 v6, s8, s16, v0                               // 00000000382c: d7000806 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003834: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s8                  // 000000003838: d5207c07 00220211
	s_lshl_b64 s[46:47], s[44:45], 1                           // 000000003840: 84ae812c
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003844: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003848: bf88ff9e
	v_add_co_u32 v6, s8, v6, s46                               // 00000000384c: d7000806 02005d06
	v_bfe_u32 v8, v36, 16, 1                                   // 000000003854: d6100008 02052124
	s_wait_alu depctr_va_sdst(0)                               // 00000000385c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s47, v7, s8                  // 000000003860: d5207c07 00220e2f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003868: bf870193
	v_add_co_u32 v4, s8, v6, v4                                // 00000000386c: d7000804 02020906
	v_add3_u32 v8, v8, v36, 0x7fff                             // 000000003874: d6550008 03fe4908 00007fff
	v_or_b32_e32 v9, 0x400000, v36                             // 000000003880: 381248ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003888: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s8                   // 00000000388c: d5207c05 00220b07
	v_cmp_u_f32_e64 s8, v36, v36                               // 000000003894: d4180008 02024924
	s_wait_alu depctr_va_sdst(0)                               // 00000000389c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038a0: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s8                           // 0000000038a4: d5010006 00221308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000038ac: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000038bc: 8c7e097e
	v_cmp_lt_i64_e64 s8, 7, v[2:3]                             // 0000000038c0: d4510008 02020487
	s_mul_u64 s[46:47], s[14:15], 7                            // 0000000038c8: aaae870e
	s_and_b32 s1, s8, s1                                       // 0000000038cc: 8b010108
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038d0: bf88ff9e
	s_and_saveexec_b32 s9, s1                                  // 0000000038d4: be892001
	s_cbranch_execz 35                                         // 0000000038d8: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1e68>
	v_add_co_u32 v4, s1, s16, v0                               // 0000000038dc: d7000104 02020010
	s_wait_alu depctr_va_sdst(0)                               // 0000000038e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v1, s1                  // 0000000038e8: d5207c05 00060211
	s_lshl_b64 s[48:49], s[46:47], 1                           // 0000000038f0: 84b0812e
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 0000000038f4: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038f8: bf88ff9e
	v_add_co_u32 v4, s1, v4, s48                               // 0000000038fc: d7000104 02006104
	v_bfe_u32 v6, v35, 16, 1                                   // 000000003904: d6100006 02052123
	s_wait_alu depctr_va_sdst(0)                               // 00000000390c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s49, v5, s1                  // 000000003910: d5207c05 00060a31
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003918: bf870193
	v_add_co_u32 v2, s1, v4, v2                                // 00000000391c: d7000102 02020504
	v_add3_u32 v6, v6, v35, 0x7fff                             // 000000003924: d6550006 03fe4706 00007fff
	v_or_b32_e32 v7, 0x400000, v35                             // 000000003930: 380e46ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003938: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v5, v3, s1                   // 00000000393c: d5207c03 00060705
	v_cmp_u_f32_e64 s1, v35, v35                               // 000000003944: d4180001 02024723
	s_wait_alu depctr_va_sdst(0)                               // 00000000394c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003950: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s1                           // 000000003954: d5010004 00060f06
	global_store_d16_hi_b16 v[2:3], v4, off                    // 00000000395c: ee09407c 02000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003968: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 00000000396c: 8c7e097e
	s_and_b32 s9, vcc_lo, s0                                   // 000000003970: 8b09006a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003974: bf88ff9e
	s_and_saveexec_b32 s1, s9                                  // 000000003978: be812009
	s_cbranch_execz 25                                         // 00000000397c: bfa50019 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1ee4>
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003980: 3e042081
	v_add_co_u32 v5, vcc_lo, s16, v0                           // 000000003984: d7006a05 02020010
	v_bfe_u32 v4, v34, 16, 1                                   // 00000000398c: d6100004 02052122
	s_wait_alu depctr_va_vcc(0)                                // 000000003994: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s17, v1, vcc_lo              // 000000003998: d5207c06 01aa0211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000039a0: bf870193
	v_add_co_u32 v2, vcc_lo, v5, v2                            // 0000000039a4: d7006a02 02020505
	v_add3_u32 v4, v4, v34, 0x7fff                             // 0000000039ac: d6550004 03fe4504 00007fff
	v_or_b32_e32 v7, 0x400000, v34                             // 0000000039b8: 380e44ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000039c0: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v6, v3, vcc_lo               // 0000000039c4: d5207c03 01aa0706
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 0000000039cc: 7c304522
	s_wait_alu depctr_va_vcc(0)                                // 0000000039d0: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo                       // 0000000039d4: 02080f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 0000000039d8: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000039e8: 8c7e017e
	s_and_b32 s2, s2, s0                                       // 0000000039ec: 8b020002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039f0: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 0000000039f4: be812002
	s_cbranch_execz 31                                         // 0000000039f8: bfa5001f <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1f78>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 0000000039fc: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003a04: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003a08: d5207c05 01aa0211
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003a10: 3e042081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000003a14: bf8701c3
	v_add_co_u32 v4, vcc_lo, v4, s10                           // 000000003a18: d7006a04 02001504
	v_bfe_u32 v6, v33, 16, 1                                   // 000000003a20: d6100006 02052121
	s_wait_alu depctr_va_vcc(0)                                // 000000003a28: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s11, v5, vcc_lo              // 000000003a2c: d5207c05 01aa0a0b
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003a34: d7006a02 02020504
	s_delay_alu instid0(valu_dep_3)                            // 000000003a3c: bf870003
	v_add3_u32 v6, v6, v33, 0x7fff                             // 000000003a40: d6550006 03fe4306 00007fff
	v_or_b32_e32 v7, 0x400000, v33                             // 000000003a4c: 380e42ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003a54: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003a58: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 000000003a60: 7c304321
	s_wait_alu depctr_va_vcc(0)                                // 000000003a64: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003a68: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003a6c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a78: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003a7c: 8c7e017e
	s_and_b32 s2, s3, s0                                       // 000000003a80: 8b020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a84: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003a88: be812002
	s_cbranch_execz 32                                         // 000000003a8c: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2010>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003a90: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003a98: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003a9c: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[10:11], 1                             // 000000003aa4: 8482810a
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003aa8: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003aac: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003ab0: d7006a04 02000504
	v_bfe_u32 v6, v32, 16, 1                                   // 000000003ab8: d6100006 02052120
	s_wait_alu depctr_va_vcc(0)                                // 000000003ac0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003ac4: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003acc: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003ad0: d7006a02 02020504
	v_add3_u32 v6, v6, v32, 0x7fff                             // 000000003ad8: d6550006 03fe4106 00007fff
	v_or_b32_e32 v7, 0x400000, v32                             // 000000003ae4: 380e40ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003aec: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003af0: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 000000003af8: 7c304120
	s_wait_alu depctr_va_vcc(0)                                // 000000003afc: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003b00: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003b04: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b10: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003b14: 8c7e017e
	s_and_b32 s2, s4, s0                                       // 000000003b18: 8b020004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b1c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003b20: be812002
	s_cbranch_execz 32                                         // 000000003b24: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x20a8>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003b28: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003b30: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003b34: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[38:39], 1                             // 000000003b3c: 84828126
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003b40: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b44: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003b48: d7006a04 02000504
	v_bfe_u32 v6, v31, 16, 1                                   // 000000003b50: d6100006 0205211f
	s_wait_alu depctr_va_vcc(0)                                // 000000003b58: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003b5c: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003b64: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003b68: d7006a02 02020504
	v_add3_u32 v6, v6, v31, 0x7fff                             // 000000003b70: d6550006 03fe3f06 00007fff
	v_or_b32_e32 v7, 0x400000, v31                             // 000000003b7c: 380e3eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003b84: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003b88: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v31, v31                           // 000000003b90: 7c303f1f
	s_wait_alu depctr_va_vcc(0)                                // 000000003b94: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003b98: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003b9c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ba8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003bac: 8c7e017e
	s_and_b32 s2, s5, s0                                       // 000000003bb0: 8b020005
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bb4: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003bb8: be812002
	s_cbranch_execz 32                                         // 000000003bbc: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2140>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003bc0: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003bc8: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003bcc: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[40:41], 1                             // 000000003bd4: 84828128
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003bd8: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bdc: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003be0: d7006a04 02000504
	v_bfe_u32 v6, v30, 16, 1                                   // 000000003be8: d6100006 0205211e
	s_wait_alu depctr_va_vcc(0)                                // 000000003bf0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003bf4: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003bfc: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003c00: d7006a02 02020504
	v_add3_u32 v6, v6, v30, 0x7fff                             // 000000003c08: d6550006 03fe3d06 00007fff
	v_or_b32_e32 v7, 0x400000, v30                             // 000000003c14: 380e3cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003c1c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003c20: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 000000003c28: 7c303d1e
	s_wait_alu depctr_va_vcc(0)                                // 000000003c2c: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003c30: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003c34: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c40: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003c44: 8c7e017e
	s_and_b32 s2, s6, s0                                       // 000000003c48: 8b020006
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c4c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003c50: be812002
	s_cbranch_execz 32                                         // 000000003c54: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x21d8>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003c58: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003c60: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003c64: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[42:43], 1                             // 000000003c6c: 8482812a
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003c70: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c74: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003c78: d7006a04 02000504
	v_bfe_u32 v6, v29, 16, 1                                   // 000000003c80: d6100006 0205211d
	s_wait_alu depctr_va_vcc(0)                                // 000000003c88: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003c8c: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003c94: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003c98: d7006a02 02020504
	v_add3_u32 v6, v6, v29, 0x7fff                             // 000000003ca0: d6550006 03fe3b06 00007fff
	v_or_b32_e32 v7, 0x400000, v29                             // 000000003cac: 380e3aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003cb4: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003cb8: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 000000003cc0: 7c303b1d
	s_wait_alu depctr_va_vcc(0)                                // 000000003cc4: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003cc8: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ccc: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cd8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003cdc: 8c7e017e
	s_and_b32 s2, s7, s0                                       // 000000003ce0: 8b020007
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ce4: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003ce8: be812002
	s_cbranch_execz 32                                         // 000000003cec: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2270>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003cf0: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003cf8: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003cfc: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[44:45], 1                             // 000000003d04: 8482812c
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003d08: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d0c: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003d10: d7006a04 02000504
	v_bfe_u32 v6, v28, 16, 1                                   // 000000003d18: d6100006 0205211c
	s_wait_alu depctr_va_vcc(0)                                // 000000003d20: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003d24: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003d2c: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003d30: d7006a02 02020504
	v_add3_u32 v6, v6, v28, 0x7fff                             // 000000003d38: d6550006 03fe3906 00007fff
	v_or_b32_e32 v7, 0x400000, v28                             // 000000003d44: 380e38ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003d4c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003d50: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v28, v28                           // 000000003d58: 7c30391c
	s_wait_alu depctr_va_vcc(0)                                // 000000003d5c: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003d60: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003d64: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003d74: 8c7e017e
	s_and_b32 s1, s8, s0                                       // 000000003d78: 8b010008
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d7c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003d80: be802001
	s_cbranch_execz 32                                         // 000000003d84: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2308>
	v_add_co_u32 v2, vcc_lo, s16, v0                           // 000000003d88: d7006a02 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003d90: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s17, v1, vcc_lo              // 000000003d94: d5207c03 01aa0211
	s_lshl_b64 s[2:3], s[46:47], 1                             // 000000003d9c: 8482812e
	v_lshlrev_b64_e32 v[0:1], 1, v[16:17]                      // 000000003da0: 3e002081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003da4: bf88ff9e
	v_add_co_u32 v2, vcc_lo, v2, s2                            // 000000003da8: d7006a02 02000502
	v_bfe_u32 v4, v27, 16, 1                                   // 000000003db0: d6100004 0205211b
	s_wait_alu depctr_va_vcc(0)                                // 000000003db8: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo               // 000000003dbc: d5207c03 01aa0603
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003dc4: bf870193
	v_add_co_u32 v0, vcc_lo, v2, v0                            // 000000003dc8: d7006a00 02020102
	v_add3_u32 v4, v4, v27, 0x7fff                             // 000000003dd0: d6550004 03fe3704 00007fff
	v_or_b32_e32 v5, 0x400000, v27                             // 000000003ddc: 380a36ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003de4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo               // 000000003de8: d5207c01 01aa0303
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 000000003df0: 7c30371b
	s_wait_alu depctr_va_vcc(0)                                // 000000003df4: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000003df8: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000003dfc: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e08: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003e0c: 8c7e007e
	s_branch 63344                                             // 000000003e10: bfa0f770 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0xd4>
	s_lshr_b64 s[4:5], s[36:37], 7                             // 000000003e14: 85848724
	v_mov_b32_e32 v19, s29                                     // 000000003e18: 7e26021d
	s_lshr_b64 s[2:3], s[34:35], 7                             // 000000003e1c: 85828722
	s_add_nc_u64 s[6:7], s[4:5], -1                            // 000000003e20: a986c104
	v_or_b32_e32 v0, 1, v18                                    // 000000003e24: 38002481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e28: bf88ff9e
	v_cmp_lt_u64_e64 s1, s[2:3], s[6:7]                        // 000000003e2c: d4590001 02000c02
	v_dual_mov_b32 v1, s29 :: v_dual_mov_b32 v2, s29           // 000000003e34: ca10001d 0102001d
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[18:19]                // 000000003e3c: 7ca8240c
	v_or_b32_e32 v3, 3, v18                                    // 000000003e40: 38062483
	v_cmp_gt_i64_e64 s0, s[14:15], v[16:17]                    // 000000003e44: d4540000 0202200e
	s_and_b32 s1, s1, exec_lo                                  // 000000003e4c: 8b017e01
	v_cmp_gt_i64_e64 s1, s[12:13], v[0:1]                      // 000000003e50: d4540001 0202000c
	v_or_b32_e32 v1, 2, v18                                    // 000000003e58: 38022482
	v_cndmask_b32_e32 v24, 0, v18, vcc_lo                      // 000000003e5c: 02302480
	v_dual_mov_b32 v4, s29 :: v_dual_cndmask_b32 v25, 0, v19   // 000000003e60: ca12001d 04182680
	s_cselect_b32 s6, s2, s6                                   // 000000003e68: 98060602
	s_delay_alu instid0(valu_dep_3)                            // 000000003e6c: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[1:2]                  // 000000003e70: 7ca8020c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e74: bf88ff9e
	v_cndmask_b32_e64 v14, 0, v0, s1                           // 000000003e78: d501000e 00060080
	v_cndmask_b32_e64 v15, 0, v19, s1                          // 000000003e80: d501000f 00062680
	v_cmp_gt_i64_e64 s1, s[12:13], v[3:4]                      // 000000003e88: d4540001 0202060c
	v_or_b32_e32 v4, 7, v18                                    // 000000003e90: 38082487
	s_wait_alu depctr_va_vcc(0)                                // 000000003e94: bf88ff9d
	v_dual_mov_b32 v5, s29 :: v_dual_cndmask_b32 v12, 0, v1    // 000000003e98: ca12001d 050c0280
	v_cndmask_b32_e32 v13, 0, v19, vcc_lo                      // 000000003ea0: 021a2680
	v_or_b32_e32 v0, 4, v18                                    // 000000003ea4: 38002484
	v_mov_b32_e32 v1, s29                                      // 000000003ea8: 7e02021d
	s_delay_alu instid0(valu_dep_4)                            // 000000003eac: bf870004
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[4:5]                  // 000000003eb0: 7ca8080c
	s_wait_alu depctr_va_sdst(0)                               // 000000003eb4: bf88f19f
	v_cndmask_b32_e64 v10, 0, v3, s1                           // 000000003eb8: d501000a 00060680
	v_or_b32_e32 v2, 6, v18                                    // 000000003ec0: 38042486
	v_cndmask_b32_e64 v11, 0, v19, s1                          // 000000003ec4: d501000b 00062680
	v_mov_b32_e32 v42, 0                                       // 000000003ecc: 7e540280
	v_mul_lo_u32 v25, v25, s30                                 // 000000003ed0: d72c0019 02003d19
	s_wait_alu depctr_va_vcc(0)                                // 000000003ed8: bf88ff9d
	v_dual_cndmask_b32 v6, 0, v4 :: v_dual_cndmask_b32 v7, 0, v19// 000000003edc: ca520880 06062680
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000003ee4: 7ca8000c
	v_mov_b32_e32 v3, s29                                      // 000000003ee8: 7e06021d
	v_or_b32_e32 v4, 5, v18                                    // 000000003eec: 38082485
	v_mul_lo_u32 v35, v24, s31                                 // 000000003ef0: d72c0023 02003f18
	v_mul_lo_u32 v1, v7, s30                                   // 000000003ef8: d72c0001 02003d07
	v_mul_lo_u32 v33, v15, s30                                 // 000000003f00: d72c0021 02003d0f
	s_wait_alu depctr_va_vcc(0)                                // 000000003f08: bf88ff9d
	v_cndmask_b32_e32 v8, 0, v19, vcc_lo                       // 000000003f0c: 02102680
	v_cmp_gt_i64_e64 s1, s[12:13], v[2:3]                      // 000000003f10: d4540001 0202040c
	v_mul_lo_u32 v3, v6, s31                                   // 000000003f18: d72c0003 02003f06
	v_mad_co_u64_u32 v[6:7], null, v6, s30, 0                  // 000000003f20: d6fe7c06 02003d06
	v_cmp_gt_i64_e64 s2, s[12:13], v[4:5]                      // 000000003f28: d4540002 0202080c
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo                        // 000000003f30: 02000080
	v_mul_lo_u32 v31, v8, s30                                  // 000000003f34: d72c001f 02003d08
	s_wait_alu depctr_va_sdst(0)                               // 000000003f3c: bf88f19f
	v_cndmask_b32_e64 v2, 0, v2, s1                            // 000000003f40: d5010002 00060480
	v_cndmask_b32_e64 v5, 0, v19, s1                           // 000000003f48: d5010005 00062680
	v_cmp_gt_i64_e64 s1, s[14:15], v[22:23]                    // 000000003f50: d4540001 02022c0e
	v_cndmask_b32_e64 v9, 0, v19, s2                           // 000000003f58: d5010009 000a2680
	v_add3_u32 v7, v7, v3, v1                                  // 000000003f60: d6550007 04060707
	v_mul_lo_u32 v28, v2, s31                                  // 000000003f68: d72c001c 02003f02
	v_mul_lo_u32 v27, v5, s30                                  // 000000003f70: d72c001b 02003d05
	v_mad_co_u64_u32 v[2:3], null, v2, s30, 0                  // 000000003f78: d6fe7c02 02003d02
	v_cndmask_b32_e64 v4, 0, v4, s2                            // 000000003f80: d5010004 000a0880
	v_mul_lo_u32 v29, v9, s30                                  // 000000003f88: d72c001d 02003d09
	v_mul_lo_u32 v32, v0, s31                                  // 000000003f90: d72c0020 02003f00
	v_mad_co_u64_u32 v[8:9], null, v0, s30, 0                  // 000000003f98: d6fe7c08 02003d00
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000003fa0: 3e000c82
	v_mul_lo_u32 v6, v11, s30                                  // 000000003fa4: d72c0006 02003d0b
	v_mul_lo_u32 v7, v10, s31                                  // 000000003fac: d72c0007 02003f0a
	v_mad_co_u64_u32 v[10:11], null, v10, s30, 0               // 000000003fb4: d6fe7c0a 02003d0a
	v_add3_u32 v3, v3, v28, v27                                // 000000003fbc: d6550003 046e3903
	v_add_co_u32 v27, s2, s34, v26                             // 000000003fc4: d700021b 02023422
	s_wait_alu depctr_va_sdst(0)                               // 000000003fcc: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s35, 0, s2                  // 000000003fd0: d5207c1c 00090023
	v_mul_lo_u32 v30, v4, s31                                  // 000000003fd8: d72c001e 02003f04
	v_mad_co_u64_u32 v[4:5], null, v4, s30, 0                  // 000000003fe0: d6fe7c04 02003d04
	v_add_co_u32 v22, vcc_lo, v27, 16                          // 000000003fe8: d7006a16 0201211b
	v_add3_u32 v9, v9, v32, v31                                // 000000003ff0: d6550009 047e4109
	v_add3_u32 v11, v11, v7, v6                                // 000000003ff8: d655000b 041a0f0b
	s_wait_alu depctr_va_vcc(0)                                // 000000004000: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, 0, v28, vcc_lo              // 000000004004: d5207c17 01aa3880
	v_mul_lo_u32 v31, v13, s30                                 // 00000000400c: d72c001f 02003d0d
	v_add3_u32 v5, v5, v30, v29                                // 000000004014: d6550005 04763d05
	v_lshlrev_b64_e32 v[6:7], 2, v[8:9]                        // 00000000401c: 3e0c1082
	v_lshlrev_b64_e32 v[8:9], 2, v[10:11]                      // 000000004020: 3e101482
	v_mul_lo_u32 v29, s18, v23                                 // 000000004024: d72c001d 02022e12
	v_mul_lo_u32 v30, s19, v22                                 // 00000000402c: d72c001e 02022c13
	v_mad_co_u64_u32 v[10:11], null, s18, v22, v[20:21]        // 000000004034: d6fe7c0a 04522c12
	v_mad_co_u64_u32 v[22:23], null, v24, s30, 0               // 00000000403c: d6fe7c16 02003d18
	v_add_co_u32 v24, s2, s28, v26                             // 000000004044: d7000218 0202341c
	v_mul_lo_u32 v32, v12, s31                                 // 00000000404c: d72c0020 02003f0c
	v_mad_co_u64_u32 v[12:13], null, v12, s30, 0               // 000000004054: d6fe7c0c 02003d0c
	v_mul_lo_u32 v34, v14, s31                                 // 00000000405c: d72c0022 02003f0e
	v_mad_co_u64_u32 v[14:15], null, v14, s30, 0               // 000000004064: d6fe7c0e 02003d0e
	s_wait_alu depctr_va_sdst(0)                               // 00000000406c: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s29, 0, s2                  // 000000004070: d5207c1a 0009001d
	v_add3_u32 v11, v30, v11, v29                              // 000000004078: d655000b 0476171e
	v_add3_u32 v23, v23, v35, v25                              // 000000004080: d6550017 04664717
	v_mul_lo_u32 v29, s19, v24                                 // 000000004088: d72c001d 02023013
	s_delay_alu instid0(valu_dep_4)                            // 000000004090: bf870004
	v_mul_lo_u32 v26, s18, v26                                 // 000000004094: d72c001a 02023412
	v_mad_co_u64_u32 v[24:25], null, s18, v24, v[20:21]        // 00000000409c: d6fe7c18 04523012
	v_mul_lo_u32 v28, s18, v28                                 // 0000000040a4: d72c001c 02023812
	v_mul_lo_u32 v30, s19, v27                                 // 0000000040ac: d72c001e 02023613
	v_mad_co_u64_u32 v[20:21], null, s18, v27, v[20:21]        // 0000000040b4: d6fe7c14 04523612
	v_add3_u32 v13, v13, v32, v31                              // 0000000040bc: d655000d 047e410d
	v_add3_u32 v15, v15, v34, v33                              // 0000000040c4: d655000f 0486450f
	v_add_co_u32 v36, vcc_lo, s24, v10                         // 0000000040cc: d7006a24 02021418
	s_wait_alu depctr_va_vcc(0)                                // 0000000040d4: bf88ff9d
	v_add_co_ci_u32_e64 v37, null, s25, v11, vcc_lo            // 0000000040d8: d5207c25 01aa1619
	v_lshlrev_b64_e32 v[10:11], 2, v[12:13]                    // 0000000040e0: 3e141882
	v_lshlrev_b64_e32 v[12:13], 2, v[14:15]                    // 0000000040e4: 3e181c82
	v_lshlrev_b64_e32 v[14:15], 2, v[22:23]                    // 0000000040e8: 3e1c2c82
	v_add3_u32 v22, v29, v25, v26                              // 0000000040ec: d6550016 046a331d
	v_add3_u32 v21, v30, v21, v28                              // 0000000040f4: d6550015 04722b1e
	v_add_co_u32 v38, vcc_lo, s26, v24                         // 0000000040fc: d7006a26 0202301a
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000004104: 3e040482
	s_wait_alu depctr_va_vcc(0)                                // 000000004108: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s27, v22, vcc_lo            // 00000000410c: d5207c27 01aa2c1b
	v_add_co_u32 v40, vcc_lo, s24, v20                         // 000000004114: d7006a28 02022818
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 00000000411c: 3e080882
	s_wait_alu depctr_va_vcc(0)                                // 000000004120: bf88ff9d
	v_add_co_ci_u32_e64 v41, null, s25, v21, vcc_lo            // 000000004124: d5207c29 01aa2a19
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0             // 00000000412c: ca100080 23220080
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0             // 000000004134: ca100080 21200080
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0             // 00000000413c: ca100080 1f1e0080
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0             // 000000004144: ca100080 1d1c0080
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0             // 00000000414c: ca100080 1b1a0080
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0             // 000000004154: ca100080 19180080
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0             // 00000000415c: ca100080 17160080
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0             // 000000004164: ca100080 15140080
	s_cselect_b32 s7, s3, s7                                   // 00000000416c: 98070703
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000004170: 84848204
	s_wait_alu depctr_sa_sdst(0)                               // 000000004174: bf88ff9e
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000004178: 84828206
	s_mov_b64 s[8:9], 0                                        // 00000000417c: be880180
	s_delay_alu instid0(salu_cycle_1)                          // 000000004180: bf870009
	v_add_co_u32 v59, vcc_lo, v38, s8                          // 000000004184: d7006a3b 02001126
	s_wait_alu depctr_va_vcc(0)                                // 00000000418c: bf88ff9d
	v_add_co_ci_u32_e64 v60, null, s9, v39, vcc_lo             // 000000004190: d5207c3c 01aa4e09
	v_add_co_u32 v63, vcc_lo, v40, s8                          // 000000004198: d7006a3f 02001128
	s_wait_alu depctr_va_vcc(0)                                // 0000000041a0: bf88ff9d
	v_add_co_ci_u32_e64 v64, null, s9, v41, vcc_lo             // 0000000041a4: d5207c40 01aa5209
	v_add_co_u32 v65, vcc_lo, v36, s8                          // 0000000041ac: d7006a41 02001124
	s_wait_alu depctr_va_vcc(0)                                // 0000000041b4: bf88ff9d
	v_add_co_ci_u32_e64 v66, null, s9, v37, vcc_lo             // 0000000041b8: d5207c42 01aa4a09
	global_load_b64 v[61:62], v[59:60], off                    // 0000000041c0: ee05407c 0000003d 0000003b
	global_load_b64 v[51:52], v[63:64], off                    // 0000000041cc: ee05407c 00000033 0000003f
	s_add_nc_u64 s[6:7], s[8:9], 0x80                          // 0000000041d8: a986ff08 00000080
	global_load_b64 v[67:68], v[65:66], off                    // 0000000041e0: ee05407c 00000043 00000041
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041ec: bf88ff9e
	s_add_nc_u64 s[8:9], s[22:23], s[2:3]                      // 0000000041f0: a9880216
	s_wait_loadcnt 0x1                                         // 0000000041f4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[61:62], v[51:52], 0// 0000000041f8: cc46402b 1a02673d
	s_wait_loadcnt 0x0                                         // 000000004200: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[61:62], v[67:68], 0// 000000004204: cc464033 1a02873d
	global_load_b64 v[61:62], v[59:60], off offset:16          // 00000000420c: ee05407c 0000003d 0000103b
	s_clause 0x1                                               // 000000004218: bf850001
	global_load_b64 v[67:68], v[63:64], off offset:16          // 00000000421c: ee05407c 00000043 0000103f
	global_load_b64 v[69:70], v[65:66], off offset:16          // 000000004228: ee05407c 00000045 00001041
	s_wait_loadcnt 0x1                                         // 000000004234: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[61:62], v[67:68], v[43:50]// 000000004238: cc46402b 1cae873d
	s_wait_loadcnt 0x0                                         // 000000004240: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[61:62], v[69:70], v[51:58]// 000000004244: cc464033 1cce8b3d
	global_load_b64 v[61:62], v[59:60], off offset:32          // 00000000424c: ee05407c 0000003d 0000203b
	s_clause 0x1                                               // 000000004258: bf850001
	global_load_b64 v[67:68], v[63:64], off offset:32          // 00000000425c: ee05407c 00000043 0000203f
	global_load_b64 v[69:70], v[65:66], off offset:32          // 000000004268: ee05407c 00000045 00002041
	s_wait_loadcnt 0x1                                         // 000000004274: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[61:62], v[67:68], v[43:50]// 000000004278: cc46402b 1cae873d
	s_wait_loadcnt 0x0                                         // 000000004280: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[61:62], v[69:70], v[51:58]// 000000004284: cc464033 1cce8b3d
	global_load_b64 v[61:62], v[59:60], off offset:48          // 00000000428c: ee05407c 0000003d 0000303b
	s_clause 0x1                                               // 000000004298: bf850001
	global_load_b64 v[67:68], v[63:64], off offset:48          // 00000000429c: ee05407c 00000043 0000303f
	global_load_b64 v[69:70], v[65:66], off offset:48          // 0000000042a8: ee05407c 00000045 00003041
	s_wait_loadcnt 0x1                                         // 0000000042b4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[61:62], v[67:68], v[43:50]// 0000000042b8: cc46402b 1cae873d
	s_wait_loadcnt 0x0                                         // 0000000042c0: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[61:62], v[69:70], v[51:58]// 0000000042c4: cc464033 1cce8b3d
	global_load_b64 v[61:62], v[59:60], off offset:64          // 0000000042cc: ee05407c 0000003d 0000403b
	s_clause 0x1                                               // 0000000042d8: bf850001
	global_load_b64 v[67:68], v[63:64], off offset:64          // 0000000042dc: ee05407c 00000043 0000403f
	global_load_b64 v[69:70], v[65:66], off offset:64          // 0000000042e8: ee05407c 00000045 00004041
	s_wait_loadcnt 0x1                                         // 0000000042f4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[61:62], v[67:68], v[43:50]// 0000000042f8: cc46402b 1cae873d
	s_wait_loadcnt 0x0                                         // 000000004300: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[61:62], v[69:70], v[51:58]// 000000004304: cc464033 1cce8b3d
	global_load_b64 v[61:62], v[59:60], off offset:80          // 00000000430c: ee05407c 0000003d 0000503b
	s_clause 0x1                                               // 000000004318: bf850001
	global_load_b64 v[67:68], v[63:64], off offset:80          // 00000000431c: ee05407c 00000043 0000503f
	global_load_b64 v[69:70], v[65:66], off offset:80          // 000000004328: ee05407c 00000045 00005041
	s_wait_loadcnt 0x1                                         // 000000004334: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[61:62], v[67:68], v[43:50]// 000000004338: cc46402b 1cae873d
	s_wait_loadcnt 0x0                                         // 000000004340: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[61:62], v[69:70], v[51:58]// 000000004344: cc464033 1cce8b3d
	global_load_b64 v[61:62], v[59:60], off offset:96          // 00000000434c: ee05407c 0000003d 0000603b
	s_clause 0x1                                               // 000000004358: bf850001
	global_load_b64 v[67:68], v[63:64], off offset:96          // 00000000435c: ee05407c 00000043 0000603f
	global_load_b64 v[69:70], v[65:66], off offset:96          // 000000004368: ee05407c 00000045 00006041
	s_wait_loadcnt 0x1                                         // 000000004374: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[61:62], v[67:68], v[43:50]// 000000004378: cc46402b 1cae873d
	s_wait_loadcnt 0x0                                         // 000000004380: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[61:62], v[69:70], v[51:58]// 000000004384: cc464033 1cce8b3d
	global_load_b64 v[59:60], v[59:60], off offset:112         // 00000000438c: ee05407c 0000003b 0000703b
	s_clause 0x1                                               // 000000004398: bf850001
	global_load_b64 v[61:62], v[63:64], off offset:112         // 00000000439c: ee05407c 0000003d 0000703f
	global_load_b64 v[63:64], v[65:66], off offset:112         // 0000000043a8: ee05407c 0000003f 00007041
	s_wait_loadcnt 0x1                                         // 0000000043b4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[43:50], v[59:60], v[61:62], v[43:50]// 0000000043b8: cc46402b 1cae7b3b
	s_wait_loadcnt 0x0                                         // 0000000043c0: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[51:58], v[59:60], v[63:64], v[51:58]// 0000000043c4: cc464033 1cce7f3b
	v_add_co_u32 v59, vcc_lo, s20, v14                         // 0000000043cc: d7006a3b 02021c14
	global_load_b32 v61, v42, s[8:9]                           // 0000000043d4: ee050008 0000003d 0000002a
	s_wait_alu depctr_va_vcc(0)                                // 0000000043e0: bf88ff9d
	v_add_co_ci_u32_e64 v60, null, s21, v15, vcc_lo            // 0000000043e4: d5207c3c 01aa1e15
	s_load_b32 s8, s[22:23], 0x0                               // 0000000043ec: f400020b f8000000
	s_add_nc_u64 s[22:23], s[22:23], s[4:5]                    // 0000000043f4: a9960416
	global_load_b32 v62, v[59:60], off                         // 0000000043f8: ee05007c 0000003e 0000003b
	s_wait_loadcnt 0x1                                         // 000000004404: bfc00001
	s_wait_kmcnt 0x0                                           // 000000004408: bfc70000
	v_cndmask_b32_e64 v63, s8, v61, s0                         // 00000000440c: d501003f 00027a08
	s_wait_loadcnt 0x0                                         // 000000004414: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004418: bf870091
	v_mul_f32_e32 v59, v62, v63                                // 00000000441c: 10767f3e
	v_mul_f32_e32 v43, v43, v59                                // 000000004420: 1056772b
	v_add_co_u32 v59, vcc_lo, s20, v12                         // 000000004424: d7006a3b 02021814
	s_wait_alu depctr_va_vcc(0)                                // 00000000442c: bf88ff9d
	v_add_co_ci_u32_e64 v60, null, s21, v13, vcc_lo            // 000000004430: d5207c3c 01aa1a15
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_1)// 000000004438: bf8700c3
	v_add_f32_e32 v35, v35, v43                                // 00000000443c: 06465723
	global_load_b32 v59, v[59:60], off                         // 000000004440: ee05007c 0000003b 0000003b
	s_wait_loadcnt 0x0                                         // 00000000444c: bfc00000
	v_mul_f32_e32 v43, v63, v59                                // 000000004450: 1056773f
	v_mul_f32_e32 v43, v44, v43                                // 000000004454: 1056572c
	s_delay_alu instid0(valu_dep_1)                            // 000000004458: bf870001
	v_add_f32_e32 v34, v34, v43                                // 00000000445c: 06445722
	v_add_co_u32 v43, vcc_lo, s20, v10                         // 000000004460: d7006a2b 02021414
	s_wait_alu depctr_va_vcc(0)                                // 000000004468: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s21, v11, vcc_lo            // 00000000446c: d5207c2c 01aa1615
	global_load_b32 v60, v[43:44], off                         // 000000004474: ee05007c 0000003c 0000002b
	s_wait_loadcnt 0x0                                         // 000000004480: bfc00000
	v_mul_f32_e32 v43, v63, v60                                // 000000004484: 1056793f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004488: bf870091
	v_mul_f32_e32 v43, v45, v43                                // 00000000448c: 1056572d
	v_add_f32_e32 v33, v33, v43                                // 000000004490: 06425721
	v_add_co_u32 v43, vcc_lo, s20, v8                          // 000000004494: d7006a2b 02021014
	s_wait_alu depctr_va_vcc(0)                                // 00000000449c: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s21, v9, vcc_lo             // 0000000044a0: d5207c2c 01aa1215
	global_load_b32 v45, v[43:44], off                         // 0000000044a8: ee05007c 0000002d 0000002b
	s_wait_loadcnt 0x0                                         // 0000000044b4: bfc00000
	v_mul_f32_e32 v43, v63, v45                                // 0000000044b8: 10565b3f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000044bc: bf870091
	v_mul_f32_e32 v43, v46, v43                                // 0000000044c0: 1056572e
	v_add_f32_e32 v32, v32, v43                                // 0000000044c4: 06405720
	v_add_co_u32 v43, vcc_lo, s20, v6                          // 0000000044c8: d7006a2b 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 0000000044d0: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s21, v7, vcc_lo             // 0000000044d4: d5207c2c 01aa0e15
	global_load_b32 v46, v[43:44], off                         // 0000000044dc: ee05007c 0000002e 0000002b
	s_wait_loadcnt 0x0                                         // 0000000044e8: bfc00000
	v_mul_f32_e32 v43, v63, v46                                // 0000000044ec: 10565d3f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000044f0: bf870091
	v_mul_f32_e32 v43, v47, v43                                // 0000000044f4: 1056572f
	v_add_f32_e32 v31, v31, v43                                // 0000000044f8: 063e571f
	v_add_co_u32 v43, vcc_lo, s20, v4                          // 0000000044fc: d7006a2b 02020814
	s_wait_alu depctr_va_vcc(0)                                // 000000004504: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s21, v5, vcc_lo             // 000000004508: d5207c2c 01aa0a15
	global_load_b32 v47, v[43:44], off                         // 000000004510: ee05007c 0000002f 0000002b
	s_wait_loadcnt 0x0                                         // 00000000451c: bfc00000
	v_mul_f32_e32 v43, v63, v47                                // 000000004520: 10565f3f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004524: bf870091
	v_mul_f32_e32 v43, v48, v43                                // 000000004528: 10565730
	v_add_f32_e32 v30, v30, v43                                // 00000000452c: 063c571e
	v_add_co_u32 v43, vcc_lo, s20, v2                          // 000000004530: d7006a2b 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000004538: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s21, v3, vcc_lo             // 00000000453c: d5207c2c 01aa0615
	global_load_b32 v48, v[43:44], off                         // 000000004544: ee05007c 00000030 0000002b
	s_wait_loadcnt 0x0                                         // 000000004550: bfc00000
	v_mul_f32_e32 v43, v63, v48                                // 000000004554: 1056613f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004558: bf870091
	v_mul_f32_e32 v43, v49, v43                                // 00000000455c: 10565731
	v_add_f32_e32 v29, v29, v43                                // 000000004560: 063a571d
	v_add_co_u32 v43, vcc_lo, s20, v0                          // 000000004564: d7006a2b 02020014
	s_wait_alu depctr_va_vcc(0)                                // 00000000456c: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s21, v1, vcc_lo             // 000000004570: d5207c2c 01aa0215
	s_add_nc_u64 s[20:21], s[20:21], 4                         // 000000004578: a9948414
	global_load_b32 v43, v[43:44], off                         // 00000000457c: ee05007c 0000002b 0000002b
	s_wait_loadcnt 0x0                                         // 000000004588: bfc00000
	v_mul_f32_e32 v44, v63, v43                                // 00000000458c: 1058573f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004590: bf870091
	v_mul_f32_e32 v44, v50, v44                                // 000000004594: 10585932
	v_add_f32_e32 v28, v28, v44                                // 000000004598: 0638591c
	v_cndmask_b32_e64 v44, s8, v61, s1                         // 00000000459c: d501002c 00067a08
	v_cmp_lt_u64_e64 s8, s[6:7], s[18:19]                      // 0000000045a4: d4590008 02002406
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_2)// 0000000045ac: bf870142
	v_mul_f32_e32 v45, v44, v45                                // 0000000045b0: 105a5b2c
	v_mul_f32_e32 v43, v44, v43                                // 0000000045b4: 1056572c
	s_and_b32 vcc_lo, exec_lo, s8                              // 0000000045b8: 8b6a087e
	s_mov_b64 s[8:9], s[6:7]                                   // 0000000045bc: be880106
	v_mul_f32_e32 v45, v54, v45                                // 0000000045c0: 105a5b36
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000045c4: bf870112
	v_mul_f32_e32 v43, v58, v43                                // 0000000045c8: 1056573a
	v_dual_mul_f32 v49, v62, v44 :: v_dual_add_f32 v24, v24, v45// 0000000045cc: c8c8593e 31185b18
	v_mul_f32_e32 v45, v44, v46                                // 0000000045d4: 105a5d2c
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000045d8: bf870112
	v_dual_add_f32 v20, v20, v43 :: v_dual_mul_f32 v49, v51, v49// 0000000045dc: c9065714 14306333
	v_mul_f32_e32 v45, v55, v45                                // 0000000045e4: 105a5b37
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045e8: bf8701a2
	v_add_f32_e32 v27, v27, v49                                // 0000000045ec: 0636631b
	v_mul_f32_e32 v49, v44, v59                                // 0000000045f0: 1062772c
	v_add_f32_e32 v23, v23, v45                                // 0000000045f4: 062e5b17
	v_mul_f32_e32 v45, v44, v47                                // 0000000045f8: 105a5f2c
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000045fc: bf870113
	v_mul_f32_e32 v49, v52, v49                                // 000000004600: 10626334
	v_mul_f32_e32 v45, v56, v45                                // 000000004604: 105a5b38
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000004608: bf870112
	v_dual_add_f32 v26, v26, v49 :: v_dual_mul_f32 v49, v44, v60// 00000000460c: c906631a 1a30792c
	v_dual_add_f32 v22, v22, v45 :: v_dual_mul_f32 v45, v44, v48// 000000004614: c9065b16 162c612c
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000461c: bf870112
	v_mul_f32_e32 v49, v53, v49                                // 000000004620: 10626335
	v_mul_f32_e32 v45, v57, v45                                // 000000004624: 105a5b39
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000004628: bf870112
	v_add_f32_e32 v25, v25, v49                                // 00000000462c: 06326319
	v_add_f32_e32 v21, v21, v45                                // 000000004630: 062a5b15
	s_wait_alu depctr_sa_sdst(0)                               // 000000004634: bf88ff9e
	s_cbranch_vccnz 65233                                      // 000000004638: bfa4fed1 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2680>
	v_mul_lo_u32 v2, s15, v18                                  // 00000000463c: d72c0002 0202240f
	v_mul_lo_u32 v3, s14, v19                                  // 000000004644: d72c0003 0202260e
	v_mad_co_u64_u32 v[0:1], null, s14, v18, 0                 // 00000000464c: d6fe7c00 0202240e
	v_bfe_u32 v4, v35, 16, 1                                   // 000000004654: d6100004 02052123
	v_or_b32_e32 v5, 0x400000, v35                             // 00000000465c: 380a46ff 00400000
	v_bfe_u32 v6, v34, 16, 1                                   // 000000004664: d6100006 02052122
	v_or_b32_e32 v7, 0x400000, v34                             // 00000000466c: 380e44ff 00400000
	s_lshl_b64 s[0:1], s[14:15], 1                             // 000000004674: 8480810e
	v_add3_u32 v4, v4, v35, 0x7fff                             // 000000004678: d6550004 03fe4704 00007fff
	v_or_b32_e32 v13, 0x400000, v32                            // 000000004684: 381a40ff 00400000
	v_add3_u32 v1, v1, v3, v2                                  // 00000000468c: d6550001 040a0701
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000004694: 3e042081
	v_add3_u32 v6, v6, v34, 0x7fff                             // 000000004698: d6550006 03fe4506 00007fff
	v_bfe_u32 v15, v31, 16, 1                                  // 0000000046a4: d610000f 0205211f
	v_or_b32_e32 v16, 0x400000, v31                            // 0000000046ac: 38203eff 00400000
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000046b4: 3e000081
	v_or_b32_e32 v19, 0x400000, v29                            // 0000000046b8: 38263aff 00400000
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 0000000046c0: bf870194
	v_add3_u32 v15, v15, v31, 0x7fff                           // 0000000046c4: d655000f 03fe3f0f 00007fff
	v_add_co_u32 v8, vcc_lo, s16, v0                           // 0000000046d0: d7006a08 02020010
	s_wait_alu depctr_va_vcc(0)                                // 0000000046d8: bf88ff9d
	s_delay_alu instid0(valu_dep_4)                            // 0000000046dc: bf870004
	v_add_co_ci_u32_e64 v9, null, s17, v1, vcc_lo              // 0000000046e0: d5207c09 01aa0211
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 0000000046e8: 7c304723
	s_wait_alu depctr_va_vcc(0)                                // 0000000046ec: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 0000000046f0: 02080b04
	v_add_co_u32 v0, vcc_lo, v8, v2                            // 0000000046f4: d7006a00 02020508
	s_wait_alu depctr_va_vcc(0)                                // 0000000046fc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v9, v3, vcc_lo               // 000000004700: d5207c01 01aa0709
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 000000004708: 7c304522
	v_bfe_u32 v5, v33, 16, 1                                   // 00000000470c: d6100005 02052121
	global_store_d16_hi_b16 v[0:1], v4, off                    // 000000004714: ee09407c 02000000 00000000
	s_wait_alu depctr_va_vcc(0)                                // 000000004720: bf88ff9d
	v_cndmask_b32_e32 v10, v6, v7, vcc_lo                      // 000000004724: 02140f06
	s_wait_alu depctr_sa_sdst(0)                               // 000000004728: bf88ff9e
	v_add_co_u32 v6, vcc_lo, v8, s0                            // 00000000472c: d7006a06 02000108
	s_wait_alu depctr_va_vcc(0)                                // 000000004734: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s1, v9, vcc_lo               // 000000004738: d5207c07 01aa1201
	v_add3_u32 v8, v5, v33, 0x7fff                             // 000000004740: d6550008 03fe4305 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000474c: bf870003
	v_add_co_u32 v4, vcc_lo, v6, v2                            // 000000004750: d7006a04 02020506
	v_or_b32_e32 v9, 0x400000, v33                             // 000000004758: 381242ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004760: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v7, v3, vcc_lo               // 000000004764: d5207c05 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 00000000476c: 7c304321
	s_wait_alu depctr_va_vcc(0)                                // 000000004770: bf88ff9d
	v_cndmask_b32_e32 v11, v8, v9, vcc_lo                      // 000000004774: 02161308
	v_add_co_u32 v9, vcc_lo, v6, s0                            // 000000004778: d7006a09 02000106
	v_bfe_u32 v8, v32, 16, 1                                   // 000000004780: d6100008 02052120
	s_wait_alu depctr_va_vcc(0)                                // 000000004788: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v7, vcc_lo              // 00000000478c: d5207c0c 01aa0e01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004794: bf870193
	v_add_co_u32 v6, vcc_lo, v9, v2                            // 000000004798: d7006a06 02020509
	v_add3_u32 v8, v8, v32, 0x7fff                             // 0000000047a0: d6550008 03fe4108 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000047ac: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000047b0: bf870003
	v_add_co_ci_u32_e64 v7, null, v12, v3, vcc_lo              // 0000000047b4: d5207c07 01aa070c
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 0000000047bc: 7c304120
	s_wait_alu depctr_va_vcc(0)                                // 0000000047c0: bf88ff9d
	v_cndmask_b32_e32 v13, v8, v13, vcc_lo                     // 0000000047c4: 021a1b08
	v_add_co_u32 v14, vcc_lo, v9, s0                           // 0000000047c8: d7006a0e 02000109
	s_wait_alu depctr_va_vcc(0)                                // 0000000047d0: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo             // 0000000047d4: d5207c0c 01aa1801
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000047dc: bf870122
	v_add_co_u32 v8, vcc_lo, v14, v2                           // 0000000047e0: d7006a08 0202050e
	s_wait_alu depctr_va_vcc(0)                                // 0000000047e8: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v12, v3, vcc_lo              // 0000000047ec: d5207c09 01aa070c
	v_cmp_u_f32_e32 vcc_lo, v31, v31                           // 0000000047f4: 7c303f1f
	v_or_b32_e32 v31, 0x400000, v28                            // 0000000047f8: 383e38ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004800: bf88ff9d
	v_cndmask_b32_e32 v16, v15, v16, vcc_lo                    // 000000004804: 0220210f
	s_clause 0x2                                               // 000000004808: bf850002
	global_store_d16_hi_b16 v[4:5], v10, off                   // 00000000480c: ee09407c 05000000 00000004
	global_store_d16_hi_b16 v[6:7], v11, off                   // 000000004818: ee09407c 05800000 00000006
	global_store_d16_hi_b16 v[8:9], v13, off                   // 000000004824: ee09407c 06800000 00000008
	v_bfe_u32 v10, v30, 16, 1                                  // 000000004830: d610000a 0205211e
	v_add_co_u32 v13, vcc_lo, v14, s0                          // 000000004838: d7006a0d 0200010e
	s_wait_alu depctr_va_vcc(0)                                // 000000004840: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo             // 000000004844: d5207c0c 01aa1801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000484c: bf870193
	v_add3_u32 v14, v10, v30, 0x7fff                           // 000000004850: d655000e 03fe3d0a 00007fff
	v_add_co_u32 v10, vcc_lo, v13, v2                          // 00000000485c: d7006a0a 0202050d
	v_or_b32_e32 v15, 0x400000, v30                            // 000000004864: 381e3cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000486c: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v12, v3, vcc_lo             // 000000004870: d5207c0b 01aa070c
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 000000004878: 7c303d1e
	v_bfe_u32 v30, v28, 16, 1                                  // 00000000487c: d610001e 0205211c
	s_wait_alu depctr_va_vcc(0)                                // 000000004884: bf88ff9d
	v_cndmask_b32_e32 v17, v14, v15, vcc_lo                    // 000000004888: 02221f0e
	v_add_co_u32 v15, vcc_lo, v13, s0                          // 00000000488c: d7006a0f 0200010d
	v_bfe_u32 v14, v29, 16, 1                                  // 000000004894: d610000e 0205211d
	s_wait_alu depctr_va_vcc(0)                                // 00000000489c: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v12, vcc_lo             // 0000000048a0: d5207c12 01aa1801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000048a8: bf870193
	v_add_co_u32 v12, vcc_lo, v15, v2                          // 0000000048ac: d7006a0c 0202050f
	v_add3_u32 v14, v14, v29, 0x7fff                           // 0000000048b4: d655000e 03fe3b0e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000048c0: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000048c4: bf870003
	v_add_co_ci_u32_e64 v13, null, v18, v3, vcc_lo             // 0000000048c8: d5207c0d 01aa0712
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 0000000048d0: 7c303b1d
	v_add3_u32 v30, v30, v28, 0x7fff                           // 0000000048d4: d655001e 03fe391e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000048e0: bf88ff9d
	v_cndmask_b32_e32 v19, v14, v19, vcc_lo                    // 0000000048e4: 0226270e
	v_add_co_u32 v29, vcc_lo, v15, s0                          // 0000000048e8: d7006a1d 0200010f
	s_wait_alu depctr_va_vcc(0)                                // 0000000048f0: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v18, vcc_lo             // 0000000048f4: d5207c12 01aa2401
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000048fc: bf870122
	v_add_co_u32 v14, vcc_lo, v29, v2                          // 000000004900: d7006a0e 0202051d
	s_wait_alu depctr_va_vcc(0)                                // 000000004908: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v18, v3, vcc_lo             // 00000000490c: d5207c0f 01aa0712
	s_clause 0x2                                               // 000000004914: bf850002
	global_store_d16_hi_b16 v[10:11], v16, off                 // 000000004918: ee09407c 08000000 0000000a
	global_store_d16_hi_b16 v[12:13], v17, off                 // 000000004924: ee09407c 08800000 0000000c
	global_store_d16_hi_b16 v[14:15], v19, off                 // 000000004930: ee09407c 09800000 0000000e
	v_cmp_u_f32_e32 vcc_lo, v28, v28                           // 00000000493c: 7c30391c
	v_bfe_u32 v17, v27, 16, 1                                  // 000000004940: d6100011 0205211b
	v_or_b32_e32 v28, 0x400000, v27                            // 000000004948: 383836ff 00400000
	s_delay_alu instid0(valu_dep_2)                            // 000000004950: bf870002
	v_add3_u32 v17, v17, v27, 0x7fff                           // 000000004954: d6550011 03fe3711 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004960: bf88ff9d
	v_cndmask_b32_e32 v16, v30, v31, vcc_lo                    // 000000004964: 02203f1e
	v_add_co_u32 v19, vcc_lo, v29, s0                          // 000000004968: d7006a13 0200011d
	s_wait_alu depctr_va_vcc(0)                                // 000000004970: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v18, vcc_lo             // 000000004974: d5207c12 01aa2401
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000497c: bf870122
	v_add_co_u32 v2, vcc_lo, v19, v2                           // 000000004980: d7006a02 02020513
	s_wait_alu depctr_va_vcc(0)                                // 000000004988: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v18, v3, vcc_lo              // 00000000498c: d5207c03 01aa0712
	v_bfe_u32 v18, v26, 16, 1                                  // 000000004994: d6100012 0205211a
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 00000000499c: 7c30371b
	v_bfe_u32 v19, v25, 16, 1                                  // 0000000049a0: d6100013 02052119
	s_wait_alu depctr_va_vcc(0)                                // 0000000049a8: bf88ff9d
	v_cndmask_b32_e32 v17, v17, v28, vcc_lo                    // 0000000049ac: 02223911
	global_store_d16_hi_b16 v[2:3], v16, off                   // 0000000049b0: ee09407c 08000000 00000002
	v_add3_u32 v16, v18, v26, 0x7fff                           // 0000000049bc: d6550010 03fe3512 00007fff
	v_or_b32_e32 v18, 0x400000, v26                            // 0000000049c8: 382434ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v26, v26                           // 0000000049d0: 7c30351a
	global_store_d16_hi_b16 v[0:1], v17, off offset:32         // 0000000049d4: ee09407c 08800000 00002000
	v_add3_u32 v0, v19, v25, 0x7fff                            // 0000000049e0: d6550000 03fe3313 00007fff
	v_or_b32_e32 v1, 0x400000, v25                             // 0000000049ec: 380232ff 00400000
	v_bfe_u32 v17, v24, 16, 1                                  // 0000000049f4: d6100011 02052118
	s_wait_alu depctr_va_vcc(0)                                // 0000000049fc: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v18, vcc_lo                    // 000000004a00: 02202510
	v_cmp_u_f32_e32 vcc_lo, v25, v25                           // 000000004a04: 7c303319
	global_store_d16_hi_b16 v[4:5], v16, off offset:32         // 000000004a08: ee09407c 08000000 00002004
	s_wait_alu depctr_va_vcc(0)                                // 000000004a14: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000004a18: 02000300
	v_bfe_u32 v1, v23, 16, 1                                   // 000000004a1c: d6100001 02052117
	v_add3_u32 v4, v17, v24, 0x7fff                            // 000000004a24: d6550004 03fe3111 00007fff
	v_or_b32_e32 v5, 0x400000, v24                             // 000000004a30: 380a30ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 000000004a38: 7c303118
	global_store_d16_hi_b16 v[6:7], v0, off offset:32          // 000000004a3c: ee09407c 00000000 00002006
	v_add3_u32 v0, v1, v23, 0x7fff                             // 000000004a48: d6550000 03fe2f01 00007fff
	v_or_b32_e32 v1, 0x400000, v23                             // 000000004a54: 38022eff 00400000
	v_bfe_u32 v6, v21, 16, 1                                   // 000000004a5c: d6100006 02052115
	s_wait_alu depctr_va_vcc(0)                                // 000000004a64: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000004a68: 02080b04
	v_bfe_u32 v5, v22, 16, 1                                   // 000000004a6c: d6100005 02052116
	v_cmp_u_f32_e32 vcc_lo, v23, v23                           // 000000004a74: 7c302f17
	v_or_b32_e32 v7, 0x400000, v22                             // 000000004a78: 380e2cff 00400000
	v_add3_u32 v6, v6, v21, 0x7fff                             // 000000004a80: d6550006 03fe2b06 00007fff
	v_or_b32_e32 v16, 0x400000, v21                            // 000000004a8c: 38202aff 00400000
	v_add3_u32 v5, v5, v22, 0x7fff                             // 000000004a94: d6550005 03fe2d05 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004aa0: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000004aa4: 02000300
	v_cmp_u_f32_e32 vcc_lo, v22, v22                           // 000000004aa8: 7c302d16
	v_bfe_u32 v1, v20, 16, 1                                   // 000000004aac: d6100001 02052114
	v_or_b32_e32 v17, 0x400000, v20                            // 000000004ab4: 382228ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004abc: bf88ff9d
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo                       // 000000004ac0: 020a0f05
	v_cmp_u_f32_e32 vcc_lo, v21, v21                           // 000000004ac4: 7c302b15
	v_add3_u32 v1, v1, v20, 0x7fff                             // 000000004ac8: d6550001 03fe2901 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004ad4: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v16, vcc_lo                      // 000000004ad8: 020c2106
	v_cmp_u_f32_e32 vcc_lo, v20, v20                           // 000000004adc: 7c302914
	s_wait_alu depctr_va_vcc(0)                                // 000000004ae0: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v17, vcc_lo                      // 000000004ae4: 02022301
	s_clause 0x3                                               // 000000004ae8: bf850003
	global_store_d16_hi_b16 v[8:9], v4, off offset:32          // 000000004aec: ee09407c 02000000 00002008
	global_store_d16_hi_b16 v[10:11], v0, off offset:32        // 000000004af8: ee09407c 00000000 0000200a
	global_store_d16_hi_b16 v[12:13], v5, off offset:32        // 000000004b04: ee09407c 02800000 0000200c
	global_store_d16_hi_b16 v[14:15], v6, off offset:32        // 000000004b10: ee09407c 03000000 0000200e
	global_store_d16_hi_b16 v[2:3], v1, off offset:32          // 000000004b1c: ee09407c 00800000 00002002
	s_endpgm                                                   // 000000004b28: bfb00000
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
	s_code_end                                                 // 000000004b80: bf9f0000
	s_code_end                                                 // 000000004b84: bf9f0000
	s_code_end                                                 // 000000004b88: bf9f0000
	s_code_end                                                 // 000000004b8c: bf9f0000
	s_code_end                                                 // 000000004b90: bf9f0000
	s_code_end                                                 // 000000004b94: bf9f0000
	s_code_end                                                 // 000000004b98: bf9f0000
	s_code_end                                                 // 000000004b9c: bf9f0000
	s_code_end                                                 // 000000004ba0: bf9f0000
	s_code_end                                                 // 000000004ba4: bf9f0000
	s_code_end                                                 // 000000004ba8: bf9f0000
	s_code_end                                                 // 000000004bac: bf9f0000
	s_code_end                                                 // 000000004bb0: bf9f0000
	s_code_end                                                 // 000000004bb4: bf9f0000
	s_code_end                                                 // 000000004bb8: bf9f0000
	s_code_end                                                 // 000000004bbc: bf9f0000
	s_code_end                                                 // 000000004bc0: bf9f0000
	s_code_end                                                 // 000000004bc4: bf9f0000
	s_code_end                                                 // 000000004bc8: bf9f0000
	s_code_end                                                 // 000000004bcc: bf9f0000
	s_code_end                                                 // 000000004bd0: bf9f0000
	s_code_end                                                 // 000000004bd4: bf9f0000
	s_code_end                                                 // 000000004bd8: bf9f0000
	s_code_end                                                 // 000000004bdc: bf9f0000
	s_code_end                                                 // 000000004be0: bf9f0000
	s_code_end                                                 // 000000004be4: bf9f0000
	s_code_end                                                 // 000000004be8: bf9f0000
	s_code_end                                                 // 000000004bec: bf9f0000
	s_code_end                                                 // 000000004bf0: bf9f0000
	s_code_end                                                 // 000000004bf4: bf9f0000
	s_code_end                                                 // 000000004bf8: bf9f0000
	s_code_end                                                 // 000000004bfc: bf9f0000
	s_code_end                                                 // 000000004c00: bf9f0000
	s_code_end                                                 // 000000004c04: bf9f0000
	s_code_end                                                 // 000000004c08: bf9f0000
	s_code_end                                                 // 000000004c0c: bf9f0000
	s_code_end                                                 // 000000004c10: bf9f0000
	s_code_end                                                 // 000000004c14: bf9f0000
	s_code_end                                                 // 000000004c18: bf9f0000
	s_code_end                                                 // 000000004c1c: bf9f0000
	s_code_end                                                 // 000000004c20: bf9f0000
	s_code_end                                                 // 000000004c24: bf9f0000
	s_code_end                                                 // 000000004c28: bf9f0000
	s_code_end                                                 // 000000004c2c: bf9f0000
	s_code_end                                                 // 000000004c30: bf9f0000
	s_code_end                                                 // 000000004c34: bf9f0000
	s_code_end                                                 // 000000004c38: bf9f0000
	s_code_end                                                 // 000000004c3c: bf9f0000
	s_code_end                                                 // 000000004c40: bf9f0000
	s_code_end                                                 // 000000004c44: bf9f0000
	s_code_end                                                 // 000000004c48: bf9f0000
	s_code_end                                                 // 000000004c4c: bf9f0000
	s_code_end                                                 // 000000004c50: bf9f0000
	s_code_end                                                 // 000000004c54: bf9f0000
	s_code_end                                                 // 000000004c58: bf9f0000
	s_code_end                                                 // 000000004c5c: bf9f0000
	s_code_end                                                 // 000000004c60: bf9f0000
	s_code_end                                                 // 000000004c64: bf9f0000
	s_code_end                                                 // 000000004c68: bf9f0000
	s_code_end                                                 // 000000004c6c: bf9f0000
	s_code_end                                                 // 000000004c70: bf9f0000
	s_code_end                                                 // 000000004c74: bf9f0000
	s_code_end                                                 // 000000004c78: bf9f0000
	s_code_end                                                 // 000000004c7c: bf9f0000
	s_code_end                                                 // 000000004c80: bf9f0000
	s_code_end                                                 // 000000004c84: bf9f0000
	s_code_end                                                 // 000000004c88: bf9f0000
	s_code_end                                                 // 000000004c8c: bf9f0000
	s_code_end                                                 // 000000004c90: bf9f0000
	s_code_end                                                 // 000000004c94: bf9f0000
	s_code_end                                                 // 000000004c98: bf9f0000
	s_code_end                                                 // 000000004c9c: bf9f0000
	s_code_end                                                 // 000000004ca0: bf9f0000
	s_code_end                                                 // 000000004ca4: bf9f0000
	s_code_end                                                 // 000000004ca8: bf9f0000
	s_code_end                                                 // 000000004cac: bf9f0000
	s_code_end                                                 // 000000004cb0: bf9f0000
	s_code_end                                                 // 000000004cb4: bf9f0000
	s_code_end                                                 // 000000004cb8: bf9f0000
	s_code_end                                                 // 000000004cbc: bf9f0000
	s_code_end                                                 // 000000004cc0: bf9f0000
	s_code_end                                                 // 000000004cc4: bf9f0000
	s_code_end                                                 // 000000004cc8: bf9f0000
	s_code_end                                                 // 000000004ccc: bf9f0000
	s_code_end                                                 // 000000004cd0: bf9f0000
	s_code_end                                                 // 000000004cd4: bf9f0000
	s_code_end                                                 // 000000004cd8: bf9f0000
	s_code_end                                                 // 000000004cdc: bf9f0000
	s_code_end                                                 // 000000004ce0: bf9f0000
	s_code_end                                                 // 000000004ce4: bf9f0000
	s_code_end                                                 // 000000004ce8: bf9f0000
	s_code_end                                                 // 000000004cec: bf9f0000
	s_code_end                                                 // 000000004cf0: bf9f0000
	s_code_end                                                 // 000000004cf4: bf9f0000
	s_code_end                                                 // 000000004cf8: bf9f0000
	s_code_end                                                 // 000000004cfc: bf9f0000
