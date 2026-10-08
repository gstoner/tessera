
/tmp/tmpk7hgfd4h.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_e44055bd28b6555a>:
	s_clause 0x2                                               // 000000001b00: bf850002
	s_load_b64 s[26:27], s[0:1], 0xd8                          // 000000001b04: f4002680 f80000d8
	s_load_b128 s[12:15], s[0:1], 0xc8                         // 000000001b0c: f4004300 f80000c8
	s_load_b64 s[16:17], s[0:1], 0xa8                          // 000000001b14: f4002400 f80000a8
	v_and_b32_e32 v2, 15, v0                                   // 000000001b1c: 3604008f
	s_mov_b32 s4, ttmp7                                        // 000000001b20: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b24: 86059f73
	s_clause 0x3                                               // 000000001b28: bf850003
	s_load_b64 s[28:29], s[0:1], 0x8                           // 000000001b2c: f4002700 f8000008
	s_load_b64 s[30:31], s[0:1], 0x30                          // 000000001b34: f4002780 f8000030
	s_load_b64 s[20:21], s[0:1], 0x58                          // 000000001b3c: f4002500 f8000058
	s_load_b64 s[24:25], s[0:1], 0x80                          // 000000001b44: f4002600 f8000080
	s_lshl_b64 s[22:23], s[4:5], 4                             // 000000001b4c: 84968404
	s_mov_b32 s2, ttmp9                                        // 000000001b50: be820075
	v_or_b32_e32 v1, s22, v2                                   // 000000001b54: 38020416
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b58: 86039f75
	s_add_nc_u64 s[0:1], s[22:23], 16                          // 000000001b5c: a9809016
	s_lshl_b64 s[2:3], s[2:3], 4                               // 000000001b60: 84828402
	v_bfe_u32 v24, v0, 4, 1                                    // 000000001b64: d6100018 02050900
	s_add_nc_u64 s[4:5], s[2:3], 16                            // 000000001b6c: a9849002
	v_or_b32_e32 v11, s2, v2                                   // 000000001b70: 38160402
	v_mov_b32_e32 v12, s3                                      // 000000001b74: 7e180203
	s_wait_kmcnt 0x0                                           // 000000001b78: bfc70000
	v_mul_lo_u32 v3, s27, v1                                   // 000000001b7c: d72c0003 0202021b
	v_mad_co_u64_u32 v[13:14], null, s26, v1, 0                // 000000001b84: d6fe7c0d 0202021a
	v_cmp_gt_i64_e64 s0, s[0:1], s[12:13]                      // 000000001b8c: d4540000 02001800
	v_cmp_gt_i64_e64 s1, s[4:5], s[14:15]                      // 000000001b94: d4540001 02001c04
	s_mul_i32 s2, s26, s23                                     // 000000001b9c: 9602171a
	v_cmp_lt_i64_e64 s11, s[26:27], 32                         // 000000001ba0: d451000b 0201401a
	s_and_b32 s18, s26, 0xffffffe0                             // 000000001ba8: 8b12ff1a ffffffe0
	s_mov_b32 s19, s27                                         // 000000001bb0: be93001b
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb4: bf88ff9e
	v_add3_u32 v23, v14, s2, v3                                // 000000001bb8: d6550017 040c050e
	s_or_b32 s0, s0, s1                                        // 000000001bc0: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc4: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bc8: 8b6a007e
	s_cbranch_vccz 10                                          // 000000001bcc: bfa3000a <tessera_rocm_scaled_matmul_e44055bd28b6555a+0xf8>
	s_and_b32 s0, s11, exec_lo                                 // 000000001bd0: 8b007e0b
	s_cselect_b32 s0, 1, 0                                     // 000000001bd4: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bd8: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bdc: bf078100
	s_cbranch_scc1 8                                           // 000000001be0: bfa20008 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x104>
	v_lshl_or_b32 v14, v24, 3, s22                             // 000000001be4: d656000e 00590718
	v_mov_b32_e32 v15, s23                                     // 000000001bec: 7e1e0217
	s_mov_b32 s0, 0                                            // 000000001bf0: be800080
	s_branch 4                                                 // 000000001bf4: bfa00004 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x108>
	s_mov_b32 s0, 0                                            // 000000001bf8: be800080
	s_cbranch_execnz 1587                                      // 000000001bfc: bfa60633 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x19cc>
	s_branch 2407                                              // 000000001c00: bfa00967 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x26a0>
	s_mov_b32 s0, -1                                           // 000000001c04: be8000c1
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0             // 000000001c08: ca100080 16140080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c10: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001c14: 8b007e00
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v27, 0             // 000000001c18: ca100080 0a1a0080
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v35, 0             // 000000001c20: ca100080 14220080
	v_mov_b32_e32 v30, 0                                       // 000000001c28: 7e3c0280
	v_mov_b32_e32 v18, 0                                       // 000000001c2c: 7e240280
	s_cselect_b32 s0, 1, 0                                     // 000000001c30: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c34: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c38: bf078100
	s_cbranch_scc1 1242                                        // 000000001c3c: bfa204da <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x14a8>
	v_dual_mov_b32 v2, s23 :: v_dual_lshlrev_b32 v19, 3, v24   // 000000001c40: ca220017 02123083
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v15, s23           // 000000001c48: ca100080 120e0017
	s_lshr_b64 s[4:5], s[26:27], 5                             // 000000001c50: 8584851a
	s_delay_alu instid0(valu_dep_2)                            // 000000001c54: bf870002
	v_or_b32_e32 v14, s22, v19                                 // 000000001c58: 381c2616
	s_lshr_b32 s3, s27, 5                                      // 000000001c5c: 8503851b
	v_add_co_u32 v25, s1, v13, v19                             // 000000001c60: d7000119 0202270d
	s_wait_alu depctr_va_sdst(0)                               // 000000001c68: bf88f19f
	v_add_co_ci_u32_e64 v26, null, 0, v23, s1                  // 000000001c6c: d5207c1a 00062e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[14:15]                // 000000001c74: 7ca81c0c
	v_mov_b32_e32 v3, s23                                      // 000000001c78: 7e060217
	v_cmp_gt_i64_e64 s1, s[12:13], v[1:2]                      // 000000001c7c: d4540001 0202020c
	v_or_b32_e32 v2, 1, v14                                    // 000000001c84: 38041c81
	v_cmp_gt_i64_e64 s0, s[14:15], v[11:12]                    // 000000001c88: d4540000 0202160e
	v_mad_co_u64_u32 v[8:9], null, s14, v19, v[11:12]          // 000000001c90: d6fe7c08 042e260e
	v_cndmask_b32_e32 v0, 0, v14, vcc_lo                       // 000000001c98: 02001c80
	v_cndmask_b32_e64 v4, 0, s23, vcc_lo                       // 000000001c9c: d5010004 01a82e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001ca4: 7ca8040c
	v_mov_b32_e32 v30, 0                                       // 000000001ca8: 7e3c0280
	s_wait_alu depctr_va_sdst(0)                               // 000000001cac: bf88f19f
	v_cndmask_b32_e64 v3, 0, v11, s0                           // 000000001cb0: d5010003 00021680
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cb8: bf88ff9e
	v_mul_lo_u32 v5, s3, v0                                    // 000000001cbc: d72c0005 02020003
	v_mul_lo_u32 v6, s4, v4                                    // 000000001cc4: d72c0006 02020804
	v_mad_co_u64_u32 v[0:1], null, s4, v0, 0                   // 000000001ccc: d6fe7c00 02020004
	v_cndmask_b32_e64 v4, 0, v12, s0                           // 000000001cd4: d5010004 00021880
	s_wait_alu depctr_va_vcc(0)                                // 000000001cdc: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo                        // 000000001ce0: 02040480
	v_cndmask_b32_e64 v7, 0, s23, vcc_lo                       // 000000001ce4: d5010007 01a82e80
	v_mad_co_u64_u32 v[9:10], null, s15, v19, v[9:10]          // 000000001cec: d6fe7c09 0426260f
	s_lshl_b64 s[34:35], s[14:15], 1                           // 000000001cf4: 84a2810e
	s_mul_u64 s[36:37], s[14:15], 3                            // 000000001cf8: aaa4830e
	s_lshl_b64 s[38:39], s[14:15], 2                           // 000000001cfc: 84a6820e
	v_add3_u32 v1, v1, v6, v5                                  // 000000001d00: d6550001 04160d01
	v_or_b32_e32 v5, 2, v14                                    // 000000001d08: 380a1c82
	v_mov_b32_e32 v6, s23                                      // 000000001d0c: 7e0c0217
	v_mul_lo_u32 v7, s4, v7                                    // 000000001d10: d72c0007 02020e04
	s_mul_u64 s[40:41], s[14:15], 5                            // 000000001d18: aaa8850e
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000001d1c: 3e000082
	s_mul_u64 s[42:43], s[14:15], 6                            // 000000001d20: aaaa860e
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000001d24: 7ca80a0c
	s_mul_u64 s[44:45], s[14:15], 7                            // 000000001d28: aaac870e
	s_mov_b64 s[46:47], 0                                      // 000000001d2c: beae0180
	v_mov_b32_e32 v35, 0                                       // 000000001d30: 7e460280
	v_add_co_u32 v28, s2, s20, v0                              // 000000001d34: d700021c 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000001d3c: bf88f19f
	v_add_co_ci_u32_e64 v29, null, s21, v1, s2                 // 000000001d40: d5207c1d 000a0215
	v_lshlrev_b64_e32 v[0:1], 2, v[3:4]                        // 000000001d48: 3e000682
	v_mov_b32_e32 v3, s23                                      // 000000001d4c: 7e060217
	v_mul_lo_u32 v10, s3, v2                                   // 000000001d50: d72c000a 02020403
	v_mad_co_u64_u32 v[16:17], null, s4, v2, 0                 // 000000001d58: d6fe7c10 02020404
	v_or_b32_e32 v2, 3, v14                                    // 000000001d60: 38041c83
	s_wait_alu depctr_va_vcc(0)                                // 000000001d64: bf88ff9d
	v_cndmask_b32_e32 v4, 0, v5, vcc_lo                        // 000000001d68: 02080a80
	v_cndmask_b32_e64 v5, 0, s23, vcc_lo                       // 000000001d6c: d5010005 01a82e80
	v_add_co_u32 v31, s2, s24, v0                              // 000000001d74: d700021f 02020018
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001d7c: 7ca8040c
	s_delay_alu instid0(valu_dep_4)                            // 000000001d80: bf870004
	v_mul_lo_u32 v6, s3, v4                                    // 000000001d84: d72c0006 02020803
	v_add3_u32 v17, v17, v7, v10                               // 000000001d8c: d6550011 042a0f11
	v_mul_lo_u32 v7, s4, v5                                    // 000000001d94: d72c0007 02020a04
	v_mad_co_u64_u32 v[4:5], null, s4, v4, 0                   // 000000001d9c: d6fe7c04 02020804
	s_wait_alu depctr_va_sdst(0)                               // 000000001da4: bf88f19f
	v_add_co_ci_u32_e64 v32, null, s25, v1, s2                 // 000000001da8: d5207c20 000a0219
	v_lshlrev_b64_e32 v[0:1], 2, v[16:17]                      // 000000001db0: 3e002082
	s_wait_alu depctr_va_vcc(0)                                // 000000001db4: bf88ff9d
	v_dual_cndmask_b32 v10, 0, v2 :: v_dual_mov_b32 v27, 0     // 000000001db8: ca500480 0a1a0080
	v_cndmask_b32_e64 v16, 0, s23, vcc_lo                      // 000000001dc0: d5010010 01a82e80
	v_or_b32_e32 v2, 4, v14                                    // 000000001dc8: 38041c84
	v_add3_u32 v5, v5, v7, v6                                  // 000000001dcc: d6550005 041a0f05
	s_delay_alu instid0(valu_dep_4)                            // 000000001dd4: bf870004
	v_mul_lo_u32 v17, s3, v10                                  // 000000001dd8: d72c0011 02021403
	v_mad_co_u64_u32 v[6:7], null, s4, v10, 0                  // 000000001de0: d6fe7c06 02021404
	v_mul_lo_u32 v16, s4, v16                                  // 000000001de8: d72c0010 02022004
	v_add_co_u32 v33, s2, s20, v0                              // 000000001df0: d7000221 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000001df8: bf88f19f
	v_add_co_ci_u32_e64 v34, null, s21, v1, s2                 // 000000001dfc: d5207c22 000a0215
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001e04: 3e000882
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001e08: 7ca8040c
	v_add3_u32 v7, v7, v16, v17                                // 000000001e0c: d6550007 04462107
	v_or_b32_e32 v16, 6, v14                                   // 000000001e14: 38201c86
	v_mov_b32_e32 v17, s23                                     // 000000001e18: 7e220217
	v_add_co_u32 v36, s2, s20, v0                              // 000000001e1c: d7000224 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000001e24: bf88f19f
	v_add_co_ci_u32_e64 v37, null, s21, v1, s2                 // 000000001e28: d5207c25 000a0215
	s_delay_alu instid0(valu_dep_3)                            // 000000001e30: bf870003
	v_cmp_gt_i64_e64 s2, s[12:13], v[16:17]                    // 000000001e34: d4540002 0202200c
	s_wait_alu depctr_va_vcc(0)                                // 000000001e3c: bf88ff9d
	v_cndmask_b32_e32 v4, 0, v2, vcc_lo                        // 000000001e40: 02080480
	v_or_b32_e32 v2, 5, v14                                    // 000000001e44: 38041c85
	v_cndmask_b32_e64 v5, 0, s23, vcc_lo                       // 000000001e48: d5010005 01a82e80
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001e50: 3e000c82
	s_wait_alu depctr_va_sdst(0)                               // 000000001e54: bf88f19f
	v_cndmask_b32_e64 v16, 0, v16, s2                          // 000000001e58: d5010010 000a2080
	v_cndmask_b32_e64 v17, 0, s23, s2                          // 000000001e60: d5010011 00082e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001e68: 7ca8040c
	v_mul_lo_u32 v20, s4, v5                                   // 000000001e6c: d72c0014 02020a04
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001e74: bf870214
	v_mul_lo_u32 v21, s3, v16                                  // 000000001e78: d72c0015 02022003
	v_mul_lo_u32 v22, s4, v17                                  // 000000001e80: d72c0016 02022204
	v_mad_co_u64_u32 v[16:17], null, s4, v16, 0                // 000000001e88: d6fe7c10 02022004
	s_wait_alu depctr_va_vcc(0)                                // 000000001e90: bf88ff9d
	v_cndmask_b32_e32 v6, 0, v2, vcc_lo                        // 000000001e94: 020c0480
	v_or_b32_e32 v2, 7, v14                                    // 000000001e98: 38041c87
	v_cndmask_b32_e64 v7, 0, s23, vcc_lo                       // 000000001e9c: d5010007 01a82e80
	s_delay_alu instid0(valu_dep_2)                            // 000000001ea4: bf870002
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001ea8: 7ca8040c
	v_add3_u32 v17, v17, v22, v21                              // 000000001eac: d6550011 04562d11
	v_mov_b32_e32 v21, 0                                       // 000000001eb4: 7e2a0280
	v_mul_lo_u32 v10, s3, v4                                   // 000000001eb8: d72c000a 02020803
	v_mad_co_u64_u32 v[4:5], null, s4, v4, 0                   // 000000001ec0: d6fe7c04 02020804
	v_mov_b32_e32 v22, 0                                       // 000000001ec8: 7e2c0280
	s_wait_alu depctr_va_vcc(0)                                // 000000001ecc: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo                        // 000000001ed0: 02040480
	v_cndmask_b32_e64 v3, 0, s23, vcc_lo                       // 000000001ed4: d5010003 01a82e80
	v_add_co_u32 v38, vcc_lo, s20, v0                          // 000000001edc: d7006a26 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000001ee4: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s21, v1, vcc_lo             // 000000001ee8: d5207c27 01aa0215
	v_add3_u32 v5, v5, v20, v10                                // 000000001ef0: d6550005 042a2905
	v_mul_lo_u32 v10, s3, v6                                   // 000000001ef8: d72c000a 02020c03
	v_mul_lo_u32 v20, s4, v7                                   // 000000001f00: d72c0014 02020e04
	v_mad_co_u64_u32 v[6:7], null, s4, v6, 0                   // 000000001f08: d6fe7c06 02020c04
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_2)// 000000001f10: bf870114
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001f14: 3e000882
	v_add3_u32 v7, v7, v20, v10                                // 000000001f18: d6550007 042a2907
	v_mul_lo_u32 v10, s3, v2                                   // 000000001f20: d72c000a 02020403
	v_mul_lo_u32 v20, s4, v3                                   // 000000001f28: d72c0014 02020604
	v_mad_co_u64_u32 v[2:3], null, s4, v2, 0                   // 000000001f30: d6fe7c02 02020404
	v_add_co_u32 v40, vcc_lo, s20, v0                          // 000000001f38: d7006a28 02020014
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000001f40: 3e080c82
	s_wait_alu depctr_va_vcc(0)                                // 000000001f44: bf88ff9d
	v_add_co_ci_u32_e64 v41, null, s21, v1, vcc_lo             // 000000001f48: d5207c29 01aa0215
	v_lshlrev_b64_e32 v[0:1], 2, v[16:17]                      // 000000001f50: 3e002082
	s_lshl_b64 s[2:3], s[14:15], 4                             // 000000001f54: 8482840e
	v_add3_u32 v3, v3, v20, v10                                // 000000001f58: d6550003 042a2903
	v_add_co_u32 v42, vcc_lo, s20, v4                          // 000000001f60: d7006a2a 02020814
	s_wait_alu depctr_va_vcc(0)                                // 000000001f68: bf88ff9d
	v_add_co_ci_u32_e64 v43, null, s21, v5, vcc_lo             // 000000001f6c: d5207c2b 01aa0a15
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000001f74: bf870253
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000001f78: 3e040482
	v_add_co_u32 v44, vcc_lo, s20, v0                          // 000000001f7c: d7006a2c 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000001f84: bf88ff9d
	v_add_co_ci_u32_e64 v45, null, s21, v1, vcc_lo             // 000000001f88: d5207c2d 01aa0215
	v_mov_b32_e32 v20, 0                                       // 000000001f90: 7e280280
	v_add_co_u32 v46, vcc_lo, s20, v2                          // 000000001f94: d7006a2e 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000001f9c: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s21, v3, vcc_lo             // 000000001fa0: d5207c2f 01aa0615
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fa8: bf88ff9e
	v_add_co_u32 v16, vcc_lo, s2, v8                           // 000000001fac: d7006a10 02021002
	s_wait_alu depctr_va_vcc(0)                                // 000000001fb4: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s3, v9, vcc_lo              // 000000001fb8: d5207c11 01aa1203
	v_mov_b32_e32 v10, 0                                       // 000000001fc0: 7e140280
	v_dual_mov_b32 v5, s47 :: v_dual_mov_b32 v2, s47           // 000000001fc4: ca10002f 0502002f
	v_or_b32_e32 v4, s46, v19                                  // 000000001fcc: 3808262e
	v_add_co_u32 v48, vcc_lo, v25, s46                         // 000000001fd0: d7006a30 02005d19
	s_wait_alu depctr_va_vcc(0)                                // 000000001fd8: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s47, v26, vcc_lo            // 000000001fdc: d5207c31 01aa342f
	s_delay_alu instid0(valu_dep_3)                            // 000000001fe4: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[4:5]                  // 000000001fe8: 7ca8081a
	s_mul_i32 s33, s46, s15                                    // 000000001fec: 96210f2e
	v_mov_b32_e32 v53, s47                                     // 000000001ff0: 7e6a022f
	s_and_b32 s2, s1, vcc_lo                                   // 000000001ff4: 8b026a01
	s_and_b32 vcc_lo, s0, vcc_lo                               // 000000001ff8: 8b6a6a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ffc: bf88ff9e
	v_cndmask_b32_e64 v0, 0, v48, s2                           // 000000002000: d5010000 000a6080
	v_cndmask_b32_e64 v1, 0, v49, s2                           // 000000002008: d5010001 000a6280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002010: bf870122
	v_add_co_u32 v0, s3, s28, v0                               // 000000002014: d7000300 0202001c
	s_wait_alu depctr_va_sdst(0)                               // 00000000201c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s29, v1, s3                  // 000000002020: d5207c01 000e021d
	global_load_d16_u8 v0, v[0:1], off                         // 000000002028: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v4                                     // 000000002034: 38020881
	s_wait_loadcnt 0x0                                         // 000000002038: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s2                            // 00000000203c: d65d0000 000a0080
	v_add_co_u32 v3, s2, v48, 1                                // 000000002044: d7000203 02010330
	s_wait_alu depctr_va_sdst(0)                               // 00000000204c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v49, s2                   // 000000002050: d5207c06 000a6280
	v_cmp_gt_i64_e64 s2, s[26:27], v[1:2]                      // 000000002058: d4540002 0202021a
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002060: d7620000 020200ff 000000ff
	s_and_b32 s3, s1, s2                                       // 00000000206c: 8b030201
	s_wait_alu depctr_sa_sdst(0)                               // 000000002070: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v3, s3                            // 000000002074: d5010001 000e0680
	v_cndmask_b32_e64 v2, 0, v6, s3                            // 00000000207c: d5010002 000e0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002084: bf870122
	v_add_co_u32 v1, s4, s28, v1                               // 000000002088: d7000401 0202021c
	s_wait_alu depctr_va_sdst(0)                               // 000000002090: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s29, v2, s4                  // 000000002094: d5207c02 0012041d
	global_load_d16_hi_u8 v0, v[1:2], off                      // 00000000209c: ee08407c 00000000 00000001
	v_or_b32_e32 v1, 2, v4                                     // 0000000020a8: 38020882
	v_mov_b32_e32 v2, s47                                      // 0000000020ac: 7e04022f
	s_wait_loadcnt 0x0                                         // 0000000020b0: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, s3                            // 0000000020b4: d65d5000 000e0080
	v_add_co_u32 v3, s3, v48, 2                                // 0000000020bc: d7000303 02010530
	s_wait_alu depctr_va_sdst(0)                               // 0000000020c4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v49, s3                   // 0000000020c8: d5207c06 000e6280
	v_cmp_gt_i64_e64 s3, s[26:27], v[1:2]                      // 0000000020d0: d4540003 0202021a
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 0000000020d8: d7385000 02020088
	s_and_b32 s4, s1, s3                                       // 0000000020e0: 8b040301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020e4: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v3, s4                            // 0000000020e8: d5010001 00120680
	v_cndmask_b32_e64 v2, 0, v6, s4                            // 0000000020f0: d5010002 00120c80
	v_mov_b32_e32 v3, s47                                      // 0000000020f8: 7e06022f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000020fc: bf8701a3
	v_add_co_u32 v1, s5, s28, v1                               // 000000002100: d7000501 0202021c
	s_wait_alu depctr_va_sdst(0)                               // 000000002108: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s29, v2, s5                  // 00000000210c: d5207c02 0016041d
	global_load_d16_u8 v1, v[1:2], off                         // 000000002114: ee07807c 00000001 00000001
	v_or_b32_e32 v2, 3, v4                                     // 000000002120: 38040883
	s_wait_loadcnt 0x0                                         // 000000002124: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, s4                            // 000000002128: d65d0001 00120280
	v_add_co_u32 v6, s4, v48, 3                                // 000000002130: d7000406 02010730
	s_wait_alu depctr_va_sdst(0)                               // 000000002138: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v49, s4                   // 00000000213c: d5207c07 00126280
	v_cmp_gt_i64_e64 s4, s[26:27], v[2:3]                      // 000000002144: d4540004 0202041a
	v_and_b16 v1.l, 0xff, v1.l                                 // 00000000214c: d7620001 020202ff 000000ff
	s_and_b32 s5, s1, s4                                       // 000000002158: 8b050401
	s_wait_alu depctr_sa_sdst(0)                               // 00000000215c: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v6, s5                            // 000000002160: d5010002 00160c80
	v_cndmask_b32_e64 v3, 0, v7, s5                            // 000000002168: d5010003 00160e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002170: bf870122
	v_add_co_u32 v2, s6, s28, v2                               // 000000002174: d7000602 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 00000000217c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s6                  // 000000002180: d5207c03 001a061d
	global_load_d16_hi_u8 v1, v[2:3], off                      // 000000002188: ee08407c 00000001 00000002
	v_or_b32_e32 v2, 4, v4                                     // 000000002194: 38040884
	v_mov_b32_e32 v3, s47                                      // 000000002198: 7e06022f
	s_wait_loadcnt 0x0                                         // 00000000219c: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, s5                            // 0000000021a0: d65d5001 00160280
	v_add_co_u32 v6, s5, v48, 4                                // 0000000021a8: d7000506 02010930
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v49, s5                   // 0000000021b4: d5207c07 00166280
	v_cmp_gt_i64_e64 s5, s[26:27], v[2:3]                      // 0000000021bc: d4540005 0202041a
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 0000000021c4: d7385001 02020288
	s_and_b32 s6, s1, s5                                       // 0000000021cc: 8b060501
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021d0: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v6, s6                            // 0000000021d4: d5010002 001a0c80
	v_cndmask_b32_e64 v3, 0, v7, s6                            // 0000000021dc: d5010003 001a0e80
	v_or_b32_e32 v6, 5, v4                                     // 0000000021e4: 380c0885
	v_mov_b32_e32 v7, s47                                      // 0000000021e8: 7e0e022f
	s_delay_alu instid0(valu_dep_4)                            // 0000000021ec: bf870004
	v_add_co_u32 v2, s7, s28, v2                               // 0000000021f0: d7000702 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 0000000021f8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s7                  // 0000000021fc: d5207c03 001e061d
	global_load_d16_u8 v2, v[2:3], off                         // 000000002204: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 000000002210: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s6                            // 000000002214: d65d0002 001a0480
	v_add_co_u32 v3, s6, v48, 5                                // 00000000221c: d7000603 02010b30
	s_wait_alu depctr_va_sdst(0)                               // 000000002224: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v49, s6                  // 000000002228: d5207c32 001a6280
	v_cmp_gt_i64_e64 s6, s[26:27], v[6:7]                      // 000000002230: d4540006 02020c1a
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002238: d7620002 020204ff 000000ff
	s_and_b32 s7, s1, s6                                       // 000000002244: 8b070601
	s_wait_alu depctr_sa_sdst(0)                               // 000000002248: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s7                            // 00000000224c: d5010003 001e0680
	v_cndmask_b32_e64 v7, 0, v50, s7                           // 000000002254: d5010007 001e6480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000225c: bf870122
	v_add_co_u32 v6, s8, s28, v3                               // 000000002260: d7000806 0202061c
	s_wait_alu depctr_va_sdst(0)                               // 000000002268: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s29, v7, s8                  // 00000000226c: d5207c07 00220e1d
	global_load_d16_hi_u8 v2, v[6:7], off                      // 000000002274: ee08407c 00000002 00000006
	v_or_b32_e32 v6, 6, v4                                     // 000000002280: 380c0886
	v_mov_b32_e32 v7, s47                                      // 000000002284: 7e0e022f
	v_or_b32_e32 v4, 7, v4                                     // 000000002288: 38080887
	s_wait_loadcnt 0x0                                         // 00000000228c: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s7                            // 000000002290: d65d5002 001e0480
	v_add_co_u32 v3, s7, v48, 6                                // 000000002298: d7000703 02010d30
	s_wait_alu depctr_va_sdst(0)                               // 0000000022a0: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v49, s7                  // 0000000022a4: d5207c32 001e6280
	v_cmp_gt_i64_e64 s7, s[26:27], v[6:7]                      // 0000000022ac: d4540007 02020c1a
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 0000000022b4: d7385002 02020488
	s_and_b32 s8, s1, s7                                       // 0000000022bc: 8b080701
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022c0: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s8                            // 0000000022c4: d5010003 00220680
	v_cndmask_b32_e64 v7, 0, v50, s8                           // 0000000022cc: d5010007 00226480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000022d4: bf870122
	v_add_co_u32 v6, s9, s28, v3                               // 0000000022d8: d7000906 0202061c
	s_wait_alu depctr_va_sdst(0)                               // 0000000022e0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s29, v7, s9                  // 0000000022e4: d5207c07 00260e1d
	global_load_d16_u8 v3, v[6:7], off                         // 0000000022ec: ee07807c 00000003 00000006
	s_wait_loadcnt 0x0                                         // 0000000022f8: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s8                            // 0000000022fc: d65d0003 00220680
	v_add_co_u32 v6, s8, v48, 7                                // 000000002304: d7000806 02010f30
	s_wait_alu depctr_va_sdst(0)                               // 00000000230c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v49, s8                   // 000000002310: d5207c07 00226280
	v_cmp_gt_i64_e64 s8, s[26:27], v[4:5]                      // 000000002318: d4540008 0202081a
	v_or_b16 v48.l, v0.l, v0.h op_sel:[0,1,0]                  // 000000002320: d7631030 02020100
	v_or_b16 v48.h, v1.l, v1.h op_sel:[0,1,1]                  // 000000002328: d7635030 02020301
	v_or_b16 v49.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002330: d7631031 02020502
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002338: d7620003 020206ff 000000ff
	s_and_b32 s9, s1, s8                                       // 000000002344: 8b090801
	s_wait_alu depctr_sa_sdst(0)                               // 000000002348: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v6, s9                            // 00000000234c: d5010004 00260c80
	v_cndmask_b32_e64 v5, 0, v7, s9                            // 000000002354: d5010005 00260e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000235c: bf870122
	v_add_co_u32 v4, s10, s28, v4                              // 000000002360: d7000a04 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000002368: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s10                 // 00000000236c: d5207c05 002a0a1d
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002374: ee08407c 00000003 00000004
	v_mad_co_u64_u32 v[4:5], null, s46, s14, v[8:9]            // 000000002380: d6fe7c04 04201c2e
	s_delay_alu instid0(valu_dep_1)                            // 000000002388: bf870001
	v_cndmask_b32_e32 v0, 0, v4, vcc_lo                        // 00000000238c: 02000880
	s_wait_loadcnt 0x0                                         // 000000002390: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s9                            // 000000002394: d65d5003 00260680
	s_mul_i32 s9, s47, s14                                     // 00000000239c: 96090e2f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023a0: bf88ff9e
	s_add_co_i32 s33, s33, s9                                  // 0000000023a4: 81210921
	v_add_co_u32 v0, s9, s30, v0                               // 0000000023a8: d7000900 0202001e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023b0: bf88ff9e
	v_add_nc_u32_e32 v7, s33, v5                               // 0000000023b4: 4a0e0a21
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000023b8: d7385003 02020688
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000023c0: bf870112
	v_cndmask_b32_e32 v1, 0, v7, vcc_lo                        // 0000000023c4: 02020e80
	v_or_b16 v49.h, v3.l, v3.h op_sel:[0,1,1]                  // 0000000023c8: d7635031 02020703
	s_wait_alu depctr_va_sdst(0)                               // 0000000023d0: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000023d4: bf870002
	v_add_co_ci_u32_e64 v1, null, s31, v1, s9                  // 0000000023d8: d5207c01 0026021f
	global_load_d16_u8 v0, v[0:1], off                         // 0000000023e0: ee07807c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 0000000023ec: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, vcc_lo                        // 0000000023f0: d65d0000 01aa0080
	v_add_co_u32 v1, vcc_lo, v4, s14                           // 0000000023f8: d7006a01 02001d04
	s_wait_alu depctr_va_vcc(0)                                // 000000002400: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s15, v7, vcc_lo              // 000000002404: d5207c02 01aa0e0f
	s_and_b32 vcc_lo, s0, s2                                   // 00000000240c: 8b6a0200
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002410: d7620000 020200ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000241c: bf88ff9e
	v_dual_cndmask_b32 v1, 0, v1 :: v_dual_cndmask_b32 v2, 0, v2// 000000002420: ca520280 01020480
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002428: bf870121
	v_add_co_u32 v1, s2, s30, v1                               // 00000000242c: d7000201 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002434: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s31, v2, s2                  // 000000002438: d5207c02 000a041f
	global_load_d16_hi_u8 v0, v[1:2], off                      // 000000002440: ee08407c 00000000 00000001
	s_wait_loadcnt 0x0                                         // 00000000244c: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, vcc_lo                        // 000000002450: d65d5000 01aa0080
	v_add_co_u32 v1, vcc_lo, v4, s34                           // 000000002458: d7006a01 02004504
	s_wait_alu depctr_va_vcc(0)                                // 000000002460: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s35, v7, vcc_lo              // 000000002464: d5207c02 01aa0e23
	s_and_b32 vcc_lo, s0, s3                                   // 00000000246c: 8b6a0300
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 000000002470: d7385000 02020088
	s_wait_alu depctr_sa_sdst(0)                               // 000000002478: bf88ff9e
	v_dual_cndmask_b32 v1, 0, v1 :: v_dual_cndmask_b32 v2, 0, v2// 00000000247c: ca520280 01020480
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002484: bf870112
	v_or_b16 v50.l, v0.l, v0.h op_sel:[0,1,0]                  // 000000002488: d7631032 02020100
	v_add_co_u32 v1, s2, s30, v1                               // 000000002490: d7000201 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002498: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 00000000249c: bf870003
	v_add_co_ci_u32_e64 v2, null, s31, v2, s2                  // 0000000024a0: d5207c02 000a041f
	global_load_d16_u8 v1, v[1:2], off                         // 0000000024a8: ee07807c 00000001 00000001
	s_wait_loadcnt 0x0                                         // 0000000024b4: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, vcc_lo                        // 0000000024b8: d65d0001 01aa0280
	v_add_co_u32 v2, vcc_lo, v4, s36                           // 0000000024c0: d7006a02 02004904
	s_wait_alu depctr_va_vcc(0)                                // 0000000024c8: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s37, v7, vcc_lo              // 0000000024cc: d5207c03 01aa0e25
	s_and_b32 vcc_lo, s0, s4                                   // 0000000024d4: 8b6a0400
	v_and_b16 v1.l, 0xff, v1.l                                 // 0000000024d8: d7620001 020202ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024e4: bf88ff9e
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_cndmask_b32 v3, 0, v3// 0000000024e8: ca520480 02020680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000024f0: bf870121
	v_add_co_u32 v2, s2, s30, v2                               // 0000000024f4: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 0000000024fc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000002500: d5207c03 000a061f
	global_load_d16_hi_u8 v1, v[2:3], off                      // 000000002508: ee08407c 00000001 00000002
	s_wait_loadcnt 0x0                                         // 000000002514: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, vcc_lo                        // 000000002518: d65d5001 01aa0280
	v_add_co_u32 v2, vcc_lo, v4, s38                           // 000000002520: d7006a02 02004d04
	s_wait_alu depctr_va_vcc(0)                                // 000000002528: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s39, v7, vcc_lo              // 00000000252c: d5207c03 01aa0e27
	s_and_b32 vcc_lo, s0, s5                                   // 000000002534: 8b6a0500
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 000000002538: d7385001 02020288
	s_wait_alu depctr_sa_sdst(0)                               // 000000002540: bf88ff9e
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_cndmask_b32 v3, 0, v3// 000000002544: ca520480 02020680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000254c: bf870112
	v_or_b16 v50.h, v1.l, v1.h op_sel:[0,1,1]                  // 000000002550: d7635032 02020301
	v_add_co_u32 v2, s2, s30, v2                               // 000000002558: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000002560: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002564: bf870003
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000002568: d5207c03 000a061f
	global_load_d16_u8 v2, v[2:3], off                         // 000000002570: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 00000000257c: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, vcc_lo                        // 000000002580: d65d0002 01aa0480
	v_add_co_u32 v3, vcc_lo, v4, s40                           // 000000002588: d7006a03 02005104
	s_wait_alu depctr_va_vcc(0)                                // 000000002590: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s41, v7, vcc_lo              // 000000002594: d5207c05 01aa0e29
	s_and_b32 vcc_lo, s0, s6                                   // 00000000259c: 8b6a0600
	v_and_b16 v2.l, 0xff, v2.l                                 // 0000000025a0: d7620002 020204ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025ac: bf88ff9e
	v_dual_cndmask_b32 v3, 0, v3 :: v_dual_cndmask_b32 v6, 0, v5// 0000000025b0: ca520680 03060a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000025b8: bf870121
	v_add_co_u32 v5, s2, s30, v3                               // 0000000025bc: d7000205 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 0000000025c4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s31, v6, s2                  // 0000000025c8: d5207c06 000a0c1f
	global_load_d16_hi_u8 v2, v[5:6], off                      // 0000000025d0: ee08407c 00000002 00000005
	s_wait_loadcnt 0x0                                         // 0000000025dc: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, vcc_lo                        // 0000000025e0: d65d5002 01aa0480
	v_add_co_u32 v3, vcc_lo, v4, s42                           // 0000000025e8: d7006a03 02005504
	s_wait_alu depctr_va_vcc(0)                                // 0000000025f0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s43, v7, vcc_lo              // 0000000025f4: d5207c05 01aa0e2b
	s_and_b32 vcc_lo, s0, s7                                   // 0000000025fc: 8b6a0700
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002600: d7385002 02020488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002608: bf88ff9e
	v_dual_cndmask_b32 v3, 0, v3 :: v_dual_cndmask_b32 v6, 0, v5// 00000000260c: ca520680 03060a80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002614: bf870112
	v_or_b16 v51.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002618: d7631033 02020502
	v_add_co_u32 v5, s2, s30, v3                               // 000000002620: d7000205 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 000000002628: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 00000000262c: bf870003
	v_add_co_ci_u32_e64 v6, null, s31, v6, s2                  // 000000002630: d5207c06 000a0c1f
	global_load_d16_u8 v3, v[5:6], off                         // 000000002638: ee07807c 00000003 00000005
	s_wait_loadcnt 0x0                                         // 000000002644: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, vcc_lo                        // 000000002648: d65d0003 01aa0680
	v_add_co_u32 v4, vcc_lo, v4, s44                           // 000000002650: d7006a04 02005904
	s_wait_alu depctr_va_vcc(0)                                // 000000002658: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s45, v7, vcc_lo              // 00000000265c: d5207c05 01aa0e2d
	s_and_b32 vcc_lo, s0, s8                                   // 000000002664: 8b6a0800
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002668: d7620003 020206ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002674: bf88ff9e
	v_dual_cndmask_b32 v4, 0, v4 :: v_dual_cndmask_b32 v5, 0, v5// 000000002678: ca520880 04040a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002680: bf870121
	v_add_co_u32 v4, s2, s30, v4                               // 000000002684: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 00000000268c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 000000002690: d5207c05 000a0a1f
	s_or_b32 s2, s46, 16                                       // 000000002698: 8c02902e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000269c: bf88ff9e
	v_or_b32_e32 v52, s2, v19                                  // 0000000026a0: 38682602
	global_load_d16_hi_u8 v3, v[4:5], off                      // 0000000026a4: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 0000000026b0: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, vcc_lo                        // 0000000026b4: d65d5003 01aa0680
	v_add_co_u32 v56, vcc_lo, v25, s2                          // 0000000026bc: d7006a38 02000519
	s_wait_alu depctr_va_vcc(0)                                // 0000000026c4: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s47, v26, vcc_lo            // 0000000026c8: d5207c39 01aa342f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 0000000026d0: bf870123
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000026d4: d7385003 02020688
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[52:53]                // 0000000026dc: 7ca8681a
	v_or_b16 v51.h, v3.l, v3.h op_sel:[0,1,1]                  // 0000000026e0: d7635033 02020703
	s_and_b32 s2, s1, vcc_lo                                   // 0000000026e8: 8b026a01
	s_and_b32 vcc_lo, s0, vcc_lo                               // 0000000026ec: 8b6a6a00
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 0000000026f0: bf8701d1
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[48:49], v[50:51], 0  // 0000000026f4: cc464000 1a026530
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026fc: bf88ff9e
	v_cndmask_b32_e64 v48, 0, v56, s2                          // 000000002700: d5010030 000a7080
	v_cndmask_b32_e64 v49, 0, v57, s2                          // 000000002708: d5010031 000a7280
	v_mov_b32_e32 v50, s47                                     // 000000002710: 7e64022f
	v_add_co_u32 v48, s3, s28, v48                             // 000000002714: d7000330 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 00000000271c: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002720: bf870003
	v_add_co_ci_u32_e64 v49, null, s29, v49, s3                // 000000002724: d5207c31 000e621d
	global_load_d16_u8 v48, v[48:49], off                      // 00000000272c: ee07807c 00000030 00000030
	v_or_b32_e32 v49, 1, v52                                   // 000000002738: 38626881
	s_wait_loadcnt 0x0                                         // 00000000273c: bfc00000
	v_cndmask_b16 v48.l, 0, v48.l, s2                          // 000000002740: d65d0030 000a6080
	v_add_co_u32 v51, s2, v56, 1                               // 000000002748: d7000233 02010338
	s_wait_alu depctr_va_sdst(0)                               // 000000002750: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v57, s2                  // 000000002754: d5207c36 000a7280
	v_cmp_gt_i64_e64 s2, s[26:27], v[49:50]                    // 00000000275c: d4540002 0202621a
	v_and_b16 v48.l, 0xff, v48.l                               // 000000002764: d7620030 020260ff 000000ff
	s_and_b32 s3, s1, s2                                       // 000000002770: 8b030201
	s_wait_alu depctr_sa_sdst(0)                               // 000000002774: bf88ff9e
	v_cndmask_b32_e64 v49, 0, v51, s3                          // 000000002778: d5010031 000e6680
	v_cndmask_b32_e64 v50, 0, v54, s3                          // 000000002780: d5010032 000e6c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002788: bf870122
	v_add_co_u32 v49, s4, s28, v49                             // 00000000278c: d7000431 0202621c
	s_wait_alu depctr_va_sdst(0)                               // 000000002794: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s29, v50, s4                // 000000002798: d5207c32 0012641d
	global_load_d16_hi_u8 v48, v[49:50], off                   // 0000000027a0: ee08407c 00000030 00000031
	v_or_b32_e32 v49, 2, v52                                   // 0000000027ac: 38626882
	v_mov_b32_e32 v50, s47                                     // 0000000027b0: 7e64022f
	s_wait_loadcnt 0x0                                         // 0000000027b4: bfc00000
	v_cndmask_b16 v48.h, 0, v48.h, s3                          // 0000000027b8: d65d5030 000e6080
	v_add_co_u32 v51, s3, v56, 2                               // 0000000027c0: d7000333 02010538
	s_wait_alu depctr_va_sdst(0)                               // 0000000027c8: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v57, s3                  // 0000000027cc: d5207c36 000e7280
	v_cmp_gt_i64_e64 s3, s[26:27], v[49:50]                    // 0000000027d4: d4540003 0202621a
	v_lshlrev_b16 v48.h, 8, v48.h op_sel:[0,1,1]               // 0000000027dc: d7385030 02026088
	s_and_b32 s4, s1, s3                                       // 0000000027e4: 8b040301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027e8: bf88ff9e
	v_cndmask_b32_e64 v49, 0, v51, s4                          // 0000000027ec: d5010031 00126680
	v_cndmask_b32_e64 v50, 0, v54, s4                          // 0000000027f4: d5010032 00126c80
	v_mov_b32_e32 v51, s47                                     // 0000000027fc: 7e66022f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002800: bf8701a3
	v_add_co_u32 v49, s5, s28, v49                             // 000000002804: d7000531 0202621c
	s_wait_alu depctr_va_sdst(0)                               // 00000000280c: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s29, v50, s5                // 000000002810: d5207c32 0016641d
	global_load_d16_u8 v49, v[49:50], off                      // 000000002818: ee07807c 00000031 00000031
	v_or_b32_e32 v50, 3, v52                                   // 000000002824: 38646883
	s_wait_loadcnt 0x0                                         // 000000002828: bfc00000
	v_cndmask_b16 v49.l, 0, v49.l, s4                          // 00000000282c: d65d0031 00126280
	v_add_co_u32 v54, s4, v56, 3                               // 000000002834: d7000436 02010738
	s_wait_alu depctr_va_sdst(0)                               // 00000000283c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v57, s4                  // 000000002840: d5207c37 00127280
	v_cmp_gt_i64_e64 s4, s[26:27], v[50:51]                    // 000000002848: d4540004 0202641a
	v_and_b16 v49.l, 0xff, v49.l                               // 000000002850: d7620031 020262ff 000000ff
	s_and_b32 s5, s1, s4                                       // 00000000285c: 8b050401
	s_wait_alu depctr_sa_sdst(0)                               // 000000002860: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v54, s5                          // 000000002864: d5010032 00166c80
	v_cndmask_b32_e64 v51, 0, v55, s5                          // 00000000286c: d5010033 00166e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002874: bf870122
	v_add_co_u32 v50, s6, s28, v50                             // 000000002878: d7000632 0202641c
	s_wait_alu depctr_va_sdst(0)                               // 000000002880: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s29, v51, s6                // 000000002884: d5207c33 001a661d
	global_load_d16_hi_u8 v49, v[50:51], off                   // 00000000288c: ee08407c 00000031 00000032
	v_or_b32_e32 v50, 4, v52                                   // 000000002898: 38646884
	v_mov_b32_e32 v51, s47                                     // 00000000289c: 7e66022f
	s_wait_loadcnt 0x0                                         // 0000000028a0: bfc00000
	v_cndmask_b16 v49.h, 0, v49.h, s5                          // 0000000028a4: d65d5031 00166280
	v_add_co_u32 v54, s5, v56, 4                               // 0000000028ac: d7000536 02010938
	s_wait_alu depctr_va_sdst(0)                               // 0000000028b4: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v57, s5                  // 0000000028b8: d5207c37 00167280
	v_cmp_gt_i64_e64 s5, s[26:27], v[50:51]                    // 0000000028c0: d4540005 0202641a
	v_lshlrev_b16 v49.h, 8, v49.h op_sel:[0,1,1]               // 0000000028c8: d7385031 02026288
	s_and_b32 s6, s1, s5                                       // 0000000028d0: 8b060501
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028d4: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v54, s6                          // 0000000028d8: d5010032 001a6c80
	v_cndmask_b32_e64 v51, 0, v55, s6                          // 0000000028e0: d5010033 001a6e80
	v_or_b32_e32 v54, 5, v52                                   // 0000000028e8: 386c6885
	v_mov_b32_e32 v55, s47                                     // 0000000028ec: 7e6e022f
	s_delay_alu instid0(valu_dep_4)                            // 0000000028f0: bf870004
	v_add_co_u32 v50, s7, s28, v50                             // 0000000028f4: d7000732 0202641c
	s_wait_alu depctr_va_sdst(0)                               // 0000000028fc: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s29, v51, s7                // 000000002900: d5207c33 001e661d
	global_load_d16_u8 v50, v[50:51], off                      // 000000002908: ee07807c 00000032 00000032
	s_wait_loadcnt 0x0                                         // 000000002914: bfc00000
	v_cndmask_b16 v50.l, 0, v50.l, s6                          // 000000002918: d65d0032 001a6480
	v_add_co_u32 v51, s6, v56, 5                               // 000000002920: d7000633 02010b38
	s_wait_alu depctr_va_sdst(0)                               // 000000002928: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v57, s6                  // 00000000292c: d5207c3a 001a7280
	v_cmp_gt_i64_e64 s6, s[26:27], v[54:55]                    // 000000002934: d4540006 02026c1a
	v_and_b16 v50.l, 0xff, v50.l                               // 00000000293c: d7620032 020264ff 000000ff
	s_and_b32 s7, s1, s6                                       // 000000002948: 8b070601
	s_wait_alu depctr_sa_sdst(0)                               // 00000000294c: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v51, s7                          // 000000002950: d5010033 001e6680
	v_cndmask_b32_e64 v55, 0, v58, s7                          // 000000002958: d5010037 001e7480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002960: bf870122
	v_add_co_u32 v54, s8, s28, v51                             // 000000002964: d7000836 0202661c
	s_wait_alu depctr_va_sdst(0)                               // 00000000296c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s29, v55, s8                // 000000002970: d5207c37 00226e1d
	global_load_d16_hi_u8 v50, v[54:55], off                   // 000000002978: ee08407c 00000032 00000036
	v_or_b32_e32 v54, 6, v52                                   // 000000002984: 386c6886
	v_mov_b32_e32 v55, s47                                     // 000000002988: 7e6e022f
	v_or_b32_e32 v52, 7, v52                                   // 00000000298c: 38686887
	s_wait_loadcnt 0x0                                         // 000000002990: bfc00000
	v_cndmask_b16 v50.h, 0, v50.h, s7                          // 000000002994: d65d5032 001e6480
	v_add_co_u32 v51, s7, v56, 6                               // 00000000299c: d7000733 02010d38
	s_wait_alu depctr_va_sdst(0)                               // 0000000029a4: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v57, s7                  // 0000000029a8: d5207c3a 001e7280
	v_cmp_gt_i64_e64 s7, s[26:27], v[54:55]                    // 0000000029b0: d4540007 02026c1a
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 0000000029b8: d7385032 02026488
	s_and_b32 s8, s1, s7                                       // 0000000029c0: 8b080701
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029c4: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v51, s8                          // 0000000029c8: d5010033 00226680
	v_cndmask_b32_e64 v55, 0, v58, s8                          // 0000000029d0: d5010037 00227480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000029d8: bf870122
	v_add_co_u32 v54, s9, s28, v51                             // 0000000029dc: d7000936 0202661c
	s_wait_alu depctr_va_sdst(0)                               // 0000000029e4: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s29, v55, s9                // 0000000029e8: d5207c37 00266e1d
	global_load_d16_u8 v51, v[54:55], off                      // 0000000029f0: ee07807c 00000033 00000036
	s_wait_loadcnt 0x0                                         // 0000000029fc: bfc00000
	v_cndmask_b16 v51.l, 0, v51.l, s8                          // 000000002a00: d65d0033 00226680
	v_add_co_u32 v54, s8, v56, 7                               // 000000002a08: d7000836 02010f38
	s_wait_alu depctr_va_sdst(0)                               // 000000002a10: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v57, s8                  // 000000002a14: d5207c37 00227280
	v_cmp_gt_i64_e64 s8, s[26:27], v[52:53]                    // 000000002a1c: d4540008 0202681a
	v_and_b16 v51.l, 0xff, v51.l                               // 000000002a24: d7620033 020266ff 000000ff
	s_and_b32 s9, s1, s8                                       // 000000002a30: 8b090801
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a34: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v54, s9                          // 000000002a38: d5010034 00266c80
	v_cndmask_b32_e64 v53, 0, v55, s9                          // 000000002a40: d5010035 00266e80
	v_mad_co_u64_u32 v[54:55], null, s46, s14, v[16:17]        // 000000002a48: d6fe7c36 04401c2e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002a50: bf8701a3
	v_add_co_u32 v52, s10, s28, v52                            // 000000002a54: d7000a34 0202681c
	s_wait_alu depctr_va_sdst(0)                               // 000000002a5c: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s29, v53, s10               // 000000002a60: d5207c35 002a6a1d
	s_delay_alu instid0(valu_dep_3)                            // 000000002a68: bf870003
	v_add_nc_u32_e32 v57, s33, v55                             // 000000002a6c: 4a726e21
	global_load_d16_hi_u8 v51, v[52:53], off                   // 000000002a70: ee08407c 00000033 00000034
	v_or_b16 v52.l, v48.l, v48.h op_sel:[0,1,0]                // 000000002a7c: d7631034 02026130
	v_cndmask_b32_e32 v48, 0, v54, vcc_lo                      // 000000002a84: 02606c80
	v_or_b16 v52.h, v49.l, v49.h op_sel:[0,1,1]                // 000000002a88: d7635034 02026331
	v_cndmask_b32_e32 v49, 0, v57, vcc_lo                      // 000000002a90: 02627280
	v_or_b16 v53.l, v50.l, v50.h op_sel:[0,1,0]                // 000000002a94: d7631035 02026532
	s_wait_loadcnt 0x0                                         // 000000002a9c: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, s9                          // 000000002aa0: d65d5033 00266680
	v_add_co_u32 v48, s9, s30, v48                             // 000000002aa8: d7000930 0202601e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ab0: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s31, v49, s9                // 000000002ab4: d5207c31 0026621f
	s_delay_alu instid0(valu_dep_3)                            // 000000002abc: bf870003
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 000000002ac0: d7385033 02026688
	global_load_d16_u8 v48, v[48:49], off                      // 000000002ac8: ee07807c 00000030 00000030
	v_or_b16 v53.h, v51.l, v51.h op_sel:[0,1,1]                // 000000002ad4: d7635035 02026733
	s_wait_loadcnt 0x0                                         // 000000002adc: bfc00000
	v_cndmask_b16 v48.l, 0, v48.l, vcc_lo                      // 000000002ae0: d65d0030 01aa6080
	v_add_co_u32 v49, vcc_lo, v54, s14                         // 000000002ae8: d7006a31 02001d36
	s_wait_alu depctr_va_vcc(0)                                // 000000002af0: bf88ff9d
	v_add_co_ci_u32_e64 v50, null, s15, v57, vcc_lo            // 000000002af4: d5207c32 01aa720f
	s_and_b32 vcc_lo, s0, s2                                   // 000000002afc: 8b6a0200
	v_and_b16 v48.l, 0xff, v48.l                               // 000000002b00: d7620030 020260ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b0c: bf88ff9e
	v_dual_cndmask_b32 v49, 0, v49 :: v_dual_cndmask_b32 v50, 0, v50// 000000002b10: ca526280 31326480
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b18: bf870121
	v_add_co_u32 v49, s2, s30, v49                             // 000000002b1c: d7000231 0202621e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b24: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s31, v50, s2                // 000000002b28: d5207c32 000a641f
	global_load_d16_hi_u8 v48, v[49:50], off                   // 000000002b30: ee08407c 00000030 00000031
	s_wait_loadcnt 0x0                                         // 000000002b3c: bfc00000
	v_cndmask_b16 v48.h, 0, v48.h, vcc_lo                      // 000000002b40: d65d5030 01aa6080
	v_add_co_u32 v49, vcc_lo, v54, s34                         // 000000002b48: d7006a31 02004536
	s_wait_alu depctr_va_vcc(0)                                // 000000002b50: bf88ff9d
	v_add_co_ci_u32_e64 v50, null, s35, v57, vcc_lo            // 000000002b54: d5207c32 01aa7223
	s_and_b32 vcc_lo, s0, s3                                   // 000000002b5c: 8b6a0300
	v_lshlrev_b16 v48.h, 8, v48.h op_sel:[0,1,1]               // 000000002b60: d7385030 02026088
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b68: bf88ff9e
	v_dual_cndmask_b32 v49, 0, v49 :: v_dual_cndmask_b32 v50, 0, v50// 000000002b6c: ca526280 31326480
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b74: bf870121
	v_add_co_u32 v49, s2, s30, v49                             // 000000002b78: d7000231 0202621e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b80: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s31, v50, s2                // 000000002b84: d5207c32 000a641f
	global_load_d16_u8 v49, v[49:50], off                      // 000000002b8c: ee07807c 00000031 00000031
	s_wait_loadcnt 0x0                                         // 000000002b98: bfc00000
	v_cndmask_b16 v49.l, 0, v49.l, vcc_lo                      // 000000002b9c: d65d0031 01aa6280
	v_add_co_u32 v50, vcc_lo, v54, s36                         // 000000002ba4: d7006a32 02004936
	s_wait_alu depctr_va_vcc(0)                                // 000000002bac: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s37, v57, vcc_lo            // 000000002bb0: d5207c33 01aa7225
	s_and_b32 vcc_lo, s0, s4                                   // 000000002bb8: 8b6a0400
	v_and_b16 v49.l, 0xff, v49.l                               // 000000002bbc: d7620031 020262ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bc8: bf88ff9e
	v_dual_cndmask_b32 v50, 0, v50 :: v_dual_cndmask_b32 v51, 0, v51// 000000002bcc: ca526480 32326680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002bd4: bf870121
	v_add_co_u32 v50, s2, s30, v50                             // 000000002bd8: d7000232 0202641e
	s_wait_alu depctr_va_sdst(0)                               // 000000002be0: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s31, v51, s2                // 000000002be4: d5207c33 000a661f
	global_load_d16_hi_u8 v49, v[50:51], off                   // 000000002bec: ee08407c 00000031 00000032
	s_wait_loadcnt 0x0                                         // 000000002bf8: bfc00000
	v_cndmask_b16 v49.h, 0, v49.h, vcc_lo                      // 000000002bfc: d65d5031 01aa6280
	v_add_co_u32 v50, vcc_lo, v54, s38                         // 000000002c04: d7006a32 02004d36
	s_wait_alu depctr_va_vcc(0)                                // 000000002c0c: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s39, v57, vcc_lo            // 000000002c10: d5207c33 01aa7227
	s_and_b32 vcc_lo, s0, s5                                   // 000000002c18: 8b6a0500
	v_lshlrev_b16 v49.h, 8, v49.h op_sel:[0,1,1]               // 000000002c1c: d7385031 02026288
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c24: bf88ff9e
	v_dual_cndmask_b32 v50, 0, v50 :: v_dual_cndmask_b32 v51, 0, v51// 000000002c28: ca526480 32326680
	s_lshr_b64 s[4:5], s[46:47], 3                             // 000000002c30: 8584832e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c34: bf870121
	v_add_co_u32 v50, s2, s30, v50                             // 000000002c38: d7000232 0202641e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c40: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s31, v51, s2                // 000000002c44: d5207c33 000a661f
	global_load_d16_u8 v50, v[50:51], off                      // 000000002c4c: ee07807c 00000032 00000032
	s_wait_loadcnt 0x0                                         // 000000002c58: bfc00000
	v_cndmask_b16 v50.l, 0, v50.l, vcc_lo                      // 000000002c5c: d65d0032 01aa6480
	v_add_co_u32 v51, vcc_lo, v54, s40                         // 000000002c64: d7006a33 02005136
	s_wait_alu depctr_va_vcc(0)                                // 000000002c6c: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s41, v57, vcc_lo            // 000000002c70: d5207c37 01aa7229
	s_and_b32 vcc_lo, s0, s6                                   // 000000002c78: 8b6a0600
	v_and_b16 v50.l, 0xff, v50.l                               // 000000002c7c: d7620032 020264ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c88: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 000000002c8c: 02666680
	v_cndmask_b32_e32 v56, 0, v55, vcc_lo                      // 000000002c90: 02706e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c94: bf870122
	v_add_co_u32 v55, s2, s30, v51                             // 000000002c98: d7000237 0202661e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ca0: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s31, v56, s2                // 000000002ca4: d5207c38 000a701f
	global_load_d16_hi_u8 v50, v[55:56], off                   // 000000002cac: ee08407c 00000032 00000037
	s_wait_loadcnt 0x0                                         // 000000002cb8: bfc00000
	v_cndmask_b16 v50.h, 0, v50.h, vcc_lo                      // 000000002cbc: d65d5032 01aa6480
	v_add_co_u32 v51, vcc_lo, v54, s42                         // 000000002cc4: d7006a33 02005536
	s_wait_alu depctr_va_vcc(0)                                // 000000002ccc: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s43, v57, vcc_lo            // 000000002cd0: d5207c37 01aa722b
	s_and_b32 vcc_lo, s0, s7                                   // 000000002cd8: 8b6a0700
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 000000002cdc: d7385032 02026488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ce4: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 000000002ce8: 02666680
	v_cndmask_b32_e32 v56, 0, v55, vcc_lo                      // 000000002cec: 02706e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002cf0: bf870122
	v_add_co_u32 v55, s2, s30, v51                             // 000000002cf4: d7000237 0202661e
	s_wait_alu depctr_va_sdst(0)                               // 000000002cfc: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s31, v56, s2                // 000000002d00: d5207c38 000a701f
	global_load_d16_u8 v51, v[55:56], off                      // 000000002d08: ee07807c 00000033 00000037
	s_wait_loadcnt 0x0                                         // 000000002d14: bfc00000
	v_cndmask_b16 v51.l, 0, v51.l, vcc_lo                      // 000000002d18: d65d0033 01aa6680
	v_add_co_u32 v54, vcc_lo, v54, s44                         // 000000002d20: d7006a36 02005936
	s_wait_alu depctr_va_vcc(0)                                // 000000002d28: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s45, v57, vcc_lo            // 000000002d2c: d5207c37 01aa722d
	s_and_b32 vcc_lo, s0, s8                                   // 000000002d34: 8b6a0800
	v_and_b16 v51.l, 0xff, v51.l                               // 000000002d38: d7620033 020266ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d44: bf88ff9e
	v_dual_cndmask_b32 v54, 0, v54 :: v_dual_cndmask_b32 v55, 0, v55// 000000002d48: ca526c80 36366e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002d50: bf870121
	v_add_co_u32 v54, s2, s30, v54                             // 000000002d54: d7000236 02026c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d5c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s31, v55, s2                // 000000002d60: d5207c37 000a6e1f
	s_lshr_b64 s[2:3], s[46:47], 5                             // 000000002d68: 8582852e
	s_add_nc_u64 s[46:47], s[46:47], 32                        // 000000002d6c: a9aea02e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d70: bf88ff9e
	s_mul_u64 s[2:3], s[2:3], s[14:15]                         // 000000002d74: aa820e02
	global_load_d16_hi_u8 v51, v[54:55], off                   // 000000002d78: ee08407c 00000033 00000036
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d84: bf88ff9e
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000002d88: 84828202
	s_wait_loadcnt 0x0                                         // 000000002d8c: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, vcc_lo                      // 000000002d90: d65d5033 01aa6680
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002d98: bf870091
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 000000002d9c: d7385033 02026688
	v_or_b16 v51.h, v51.l, v51.h op_sel:[0,1,1]                // 000000002da4: d7635033 02026733
	v_or_b16 v51.l, v50.l, v50.h op_sel:[0,1,0]                // 000000002dac: d7631033 02026532
	v_or_b16 v50.l, v48.l, v48.h op_sel:[0,1,0]                // 000000002db4: d7631032 02026130
	v_add_co_u32 v48, vcc_lo, v28, s4                          // 000000002dbc: d7006a30 0200091c
	v_or_b16 v50.h, v49.l, v49.h op_sel:[0,1,1]                // 000000002dc4: d7635032 02026331
	s_wait_alu depctr_va_vcc(0)                                // 000000002dcc: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s5, v29, vcc_lo             // 000000002dd0: d5207c31 01aa3a05
	s_delay_alu instid0(valu_dep_2)                            // 000000002dd8: bf870002
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[52:53], v[50:51], v[0:7]// 000000002ddc: cc464000 1c026534
	global_load_b32 v50, v[48:49], off                         // 000000002de4: ee05007c 00000032 00000030
	s_wait_alu depctr_sa_sdst(0)                               // 000000002df0: bf88ff9e
	v_add_co_u32 v48, vcc_lo, v31, s2                          // 000000002df4: d7006a30 0200051f
	s_wait_alu depctr_va_vcc(0)                                // 000000002dfc: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s3, v32, vcc_lo             // 000000002e00: d5207c31 01aa4003
	v_cmp_lt_i64_e64 s2, s[46:47], s[18:19]                    // 000000002e08: d4510002 0200242e
	global_load_b32 v51, v[48:49], off                         // 000000002e10: ee05007c 00000033 00000030
	s_wait_loadcnt 0x0                                         // 000000002e1c: bfc00000
	v_mul_f32_e32 v48, v50, v51                                // 000000002e20: 10606732
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000002e24: bf8701c1
	v_mul_f32_e32 v0, v0, v48                                  // 000000002e28: 10006100
	v_add_co_u32 v48, vcc_lo, v33, s4                          // 000000002e2c: d7006a30 02000921
	s_wait_alu depctr_va_vcc(0)                                // 000000002e34: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s5, v34, vcc_lo             // 000000002e38: d5207c31 01aa4405
	v_add_f32_e32 v18, v18, v0                                 // 000000002e40: 06240112
	global_load_b32 v0, v[48:49], off                          // 000000002e44: ee05007c 00000000 00000030
	s_wait_loadcnt 0x0                                         // 000000002e50: bfc00000
	v_mul_f32_e32 v0, v51, v0                                  // 000000002e54: 10000133
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002e58: bf870091
	v_mul_f32_e32 v0, v1, v0                                   // 000000002e5c: 10000101
	v_add_f32_e32 v35, v35, v0                                 // 000000002e60: 06460123
	v_add_co_u32 v0, vcc_lo, v36, s4                           // 000000002e64: d7006a00 02000924
	s_wait_alu depctr_va_vcc(0)                                // 000000002e6c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s5, v37, vcc_lo              // 000000002e70: d5207c01 01aa4a05
	global_load_b32 v0, v[0:1], off                            // 000000002e78: ee05007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002e84: bfc00000
	v_mul_f32_e32 v0, v51, v0                                  // 000000002e88: 10000133
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002e8c: bf870091
	v_mul_f32_e32 v0, v2, v0                                   // 000000002e90: 10000102
	v_add_f32_e32 v30, v30, v0                                 // 000000002e94: 063c011e
	v_add_co_u32 v0, vcc_lo, v38, s4                           // 000000002e98: d7006a00 02000926
	s_wait_alu depctr_va_vcc(0)                                // 000000002ea0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s5, v39, vcc_lo              // 000000002ea4: d5207c01 01aa4e05
	global_load_b32 v0, v[0:1], off                            // 000000002eac: ee05007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002eb8: bfc00000
	v_mul_f32_e32 v0, v51, v0                                  // 000000002ebc: 10000133
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002ec0: bf870091
	v_mul_f32_e32 v0, v3, v0                                   // 000000002ec4: 10000103
	v_add_f32_e32 v27, v27, v0                                 // 000000002ec8: 0636011b
	v_add_co_u32 v0, vcc_lo, v40, s4                           // 000000002ecc: d7006a00 02000928
	s_wait_alu depctr_va_vcc(0)                                // 000000002ed4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s5, v41, vcc_lo              // 000000002ed8: d5207c01 01aa5205
	global_load_b32 v0, v[0:1], off                            // 000000002ee0: ee05007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002eec: bfc00000
	v_mul_f32_e32 v0, v51, v0                                  // 000000002ef0: 10000133
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002ef4: bf870091
	v_mul_f32_e32 v0, v4, v0                                   // 000000002ef8: 10000104
	v_add_f32_e32 v21, v21, v0                                 // 000000002efc: 062a0115
	v_add_co_u32 v0, vcc_lo, v42, s4                           // 000000002f00: d7006a00 0200092a
	s_wait_alu depctr_va_vcc(0)                                // 000000002f08: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s5, v43, vcc_lo              // 000000002f0c: d5207c01 01aa5605
	global_load_b32 v0, v[0:1], off                            // 000000002f14: ee05007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002f20: bfc00000
	v_mul_f32_e32 v0, v51, v0                                  // 000000002f24: 10000133
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002f28: bf870091
	v_mul_f32_e32 v0, v5, v0                                   // 000000002f2c: 10000105
	v_add_f32_e32 v20, v20, v0                                 // 000000002f30: 06280114
	v_add_co_u32 v0, vcc_lo, v44, s4                           // 000000002f34: d7006a00 0200092c
	s_wait_alu depctr_va_vcc(0)                                // 000000002f3c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s5, v45, vcc_lo              // 000000002f40: d5207c01 01aa5a05
	global_load_b32 v0, v[0:1], off                            // 000000002f48: ee05007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002f54: bfc00000
	v_mul_f32_e32 v0, v51, v0                                  // 000000002f58: 10000133
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002f5c: bf870091
	v_mul_f32_e32 v0, v6, v0                                   // 000000002f60: 10000106
	v_add_f32_e32 v10, v10, v0                                 // 000000002f64: 0614010a
	v_add_co_u32 v0, vcc_lo, v46, s4                           // 000000002f68: d7006a00 0200092e
	s_wait_alu depctr_va_vcc(0)                                // 000000002f70: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s5, v47, vcc_lo              // 000000002f74: d5207c01 01aa5e05
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000002f7c: 8b6a027e
	global_load_b32 v0, v[0:1], off                            // 000000002f80: ee05007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002f8c: bfc00000
	v_mul_f32_e32 v0, v51, v0                                  // 000000002f90: 10000133
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002f94: bf870091
	v_mul_f32_e32 v0, v7, v0                                   // 000000002f98: 10000107
	v_add_f32_e32 v22, v22, v0                                 // 000000002f9c: 062c0116
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fa0: bf88ff9e
	s_cbranch_vccnz 64519                                      // 000000002fa4: bfa4fc07 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x4c4>
	v_mul_lo_u32 v4, s15, v14                                  // 000000002fa8: d72c0004 02021c0f
	v_mul_lo_u32 v5, s14, v15                                  // 000000002fb0: d72c0005 02021e0e
	v_mad_co_u64_u32 v[2:3], null, s14, v14, 0                 // 000000002fb8: d6fe7c02 02021c0e
	v_sub_co_u32 v0, vcc_lo, s12, v14                          // 000000002fc0: d7016a00 02021c0c
	s_wait_alu depctr_va_vcc(0)                                // 000000002fc8: bf88ff9d
	v_sub_co_ci_u32_e64 v1, null, s13, v15, vcc_lo             // 000000002fcc: d5217c01 01aa1e0d
	v_cmp_gt_i64_e32 vcc_lo, s[14:15], v[11:12]                // 000000002fd4: 7ca8160e
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000002fd8: bf870194
	v_add3_u32 v3, v3, v5, v4                                  // 000000002fdc: d6550003 04120b03
	v_cmp_lt_i64_e64 s0, 0, v[0:1]                             // 000000002fe4: d4510000 02020080
	s_delay_alu instid0(valu_dep_2)                            // 000000002fec: bf870002
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000002ff0: 3e040481
	s_and_b32 s0, s0, vcc_lo                                   // 000000002ff4: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ff8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002ffc: be812000
	s_cbranch_execz 28                                         // 000000003000: bfa5001c <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x1574>
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 000000003004: 3e081681
	v_add_co_u32 v7, s0, s16, v2                               // 000000003008: d7000007 02020410
	v_bfe_u32 v6, v18, 16, 1                                   // 000000003010: d6100006 02052112
	s_wait_alu depctr_va_sdst(0)                               // 000000003018: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, v3, s0                  // 00000000301c: d5207c08 00020611
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003024: bf870193
	v_add_co_u32 v4, s0, v7, v4                                // 000000003028: d7000004 02020907
	v_add3_u32 v6, v6, v18, 0x7fff                             // 000000003030: d6550006 03fe2506 00007fff
	v_or_b32_e32 v9, 0x400000, v18                             // 00000000303c: 381224ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003044: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s0                   // 000000003048: d5207c05 00020b08
	v_cmp_u_f32_e64 s0, v18, v18                               // 000000003050: d4180000 02022512
	s_wait_alu depctr_va_sdst(0)                               // 000000003058: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000305c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s0                           // 000000003060: d5010006 00021306
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003068: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003074: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003078: 8c7e017e
	v_cmp_lt_i64_e64 s0, 1, v[0:1]                             // 00000000307c: d4510000 02020081
	s_and_b32 s0, s0, vcc_lo                                   // 000000003084: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003088: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000308c: be812000
	s_cbranch_execz 35                                         // 000000003090: bfa50023 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x1620>
	v_add_co_u32 v6, s0, s16, v2                               // 000000003094: d7000006 02020410
	s_wait_alu depctr_va_sdst(0)                               // 00000000309c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v3, s0                  // 0000000030a0: d5207c07 00020611
	s_lshl_b64 s[2:3], s[14:15], 1                             // 0000000030a8: 8482810e
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 0000000030ac: 3e081681
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030b0: bf88ff9e
	v_add_co_u32 v6, s0, v6, s2                                // 0000000030b4: d7000006 02000506
	v_bfe_u32 v8, v35, 16, 1                                   // 0000000030bc: d6100008 02052123
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s3, v7, s0                   // 0000000030c8: d5207c07 00020e03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000030d0: bf870193
	v_add_co_u32 v4, s0, v6, v4                                // 0000000030d4: d7000004 02020906
	v_add3_u32 v8, v8, v35, 0x7fff                             // 0000000030dc: d6550008 03fe4708 00007fff
	v_or_b32_e32 v9, 0x400000, v35                             // 0000000030e8: 381246ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s0                   // 0000000030f4: d5207c05 00020b07
	v_cmp_u_f32_e64 s0, v35, v35                               // 0000000030fc: d4180000 02024723
	s_wait_alu depctr_va_sdst(0)                               // 000000003104: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003108: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s0                           // 00000000310c: d5010006 00021308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003114: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003120: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003124: 8c7e017e
	v_cmp_lt_i64_e64 s0, 2, v[0:1]                             // 000000003128: d4510000 02020082
	s_and_b32 s0, s0, vcc_lo                                   // 000000003130: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003134: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003138: be812000
	s_cbranch_execz 34                                         // 00000000313c: bfa50022 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x16c8>
	v_add_co_u32 v8, s0, s16, v2                               // 000000003140: d7000008 02020410
	v_bfe_u32 v4, v30, 16, 1                                   // 000000003148: d6100004 0205211e
	s_wait_alu depctr_va_sdst(0)                               // 000000003150: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s17, v3, s0                  // 000000003154: d5207c09 00020611
	s_lshl_b64 s[2:3], s[14:15], 2                             // 00000000315c: 8482820e
	v_or_b32_e32 v6, 0x400000, v30                             // 000000003160: 380c3cff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003168: bf88ff9e
	v_add_co_u32 v8, s0, v8, s2                                // 00000000316c: d7000008 02000508
	v_add3_u32 v7, v4, v30, 0x7fff                             // 000000003174: d6550007 03fe3d04 00007fff
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 000000003180: 3e081681
	s_wait_alu depctr_va_sdst(0)                               // 000000003184: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s3, v9, s0                   // 000000003188: d5207c09 00021203
	v_cmp_u_f32_e64 s0, v30, v30                               // 000000003190: d4180000 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003198: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000319c: bf870001
	v_cndmask_b32_e64 v6, v7, v6, s0                           // 0000000031a0: d5010006 00020d07
	v_add_co_u32 v4, s0, v8, v4                                // 0000000031a8: d7000004 02020908
	s_wait_alu depctr_va_sdst(0)                               // 0000000031b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v9, v5, s0                   // 0000000031b4: d5207c05 00020b09
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000031bc: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000031cc: 8c7e017e
	v_cmp_lt_i64_e64 s0, 3, v[0:1]                             // 0000000031d0: d4510000 02020083
	s_and_b32 s0, s0, vcc_lo                                   // 0000000031d8: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031dc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000031e0: be812000
	s_cbranch_execz 33                                         // 0000000031e4: bfa50021 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x176c>
	v_add_co_u32 v4, s0, s16, v2                               // 0000000031e8: d7000004 02020410
	s_wait_alu depctr_va_sdst(0)                               // 0000000031f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v3, s0                  // 0000000031f4: d5207c05 00020611
	v_bfe_u32 v6, v27, 16, 1                                   // 0000000031fc: d6100006 0205211b
	v_or_b32_e32 v8, 0x400000, v27                             // 000000003204: 381036ff 00400000
	v_cmp_u_f32_e64 s0, v27, v27                               // 00000000320c: d4180000 0202371b
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003214: bf870214
	v_mad_co_u64_u32 v[4:5], null, s14, 6, v[4:5]              // 000000003218: d6fe7c04 04110c0e
	v_add3_u32 v9, v6, v27, 0x7fff                             // 000000003220: d6550009 03fe3706 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000322c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003230: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 000000003234: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s15, 6, v[5:6]              // 00000000323c: d6fe7c05 04150c0f
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 000000003244: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003248: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 00000000324c: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003254: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 000000003258: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 000000003260: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000326c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003270: 8c7e017e
	v_cmp_lt_i64_e64 s0, 4, v[0:1]                             // 000000003274: d4510000 02020084
	s_and_b32 s0, s0, vcc_lo                                   // 00000000327c: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003280: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003284: be812000
	s_cbranch_execz 34                                         // 000000003288: bfa50022 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x1814>
	v_add_co_u32 v8, s0, s16, v2                               // 00000000328c: d7000008 02020410
	v_bfe_u32 v4, v21, 16, 1                                   // 000000003294: d6100004 02052115
	s_wait_alu depctr_va_sdst(0)                               // 00000000329c: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s17, v3, s0                  // 0000000032a0: d5207c09 00020611
	s_lshl_b64 s[2:3], s[14:15], 3                             // 0000000032a8: 8482830e
	v_or_b32_e32 v6, 0x400000, v21                             // 0000000032ac: 380c2aff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032b4: bf88ff9e
	v_add_co_u32 v8, s0, v8, s2                                // 0000000032b8: d7000008 02000508
	v_add3_u32 v7, v4, v21, 0x7fff                             // 0000000032c0: d6550007 03fe2b04 00007fff
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 0000000032cc: 3e081681
	s_wait_alu depctr_va_sdst(0)                               // 0000000032d0: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s3, v9, s0                   // 0000000032d4: d5207c09 00021203
	v_cmp_u_f32_e64 s0, v21, v21                               // 0000000032dc: d4180000 02022b15
	s_wait_alu depctr_va_sdst(0)                               // 0000000032e4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000032e8: bf870001
	v_cndmask_b32_e64 v6, v7, v6, s0                           // 0000000032ec: d5010006 00020d07
	v_add_co_u32 v4, s0, v8, v4                                // 0000000032f4: d7000004 02020908
	s_wait_alu depctr_va_sdst(0)                               // 0000000032fc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v9, v5, s0                   // 000000003300: d5207c05 00020b09
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003308: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003314: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003318: 8c7e017e
	v_cmp_lt_i64_e64 s0, 5, v[0:1]                             // 00000000331c: d4510000 02020085
	s_and_b32 s0, s0, vcc_lo                                   // 000000003324: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003328: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000332c: be812000
	s_cbranch_execz 33                                         // 000000003330: bfa50021 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x18b8>
	v_add_co_u32 v4, s0, s16, v2                               // 000000003334: d7000004 02020410
	s_wait_alu depctr_va_sdst(0)                               // 00000000333c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v3, s0                  // 000000003340: d5207c05 00020611
	v_bfe_u32 v6, v20, 16, 1                                   // 000000003348: d6100006 02052114
	v_or_b32_e32 v8, 0x400000, v20                             // 000000003350: 381028ff 00400000
	v_cmp_u_f32_e64 s0, v20, v20                               // 000000003358: d4180000 02022914
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003360: bf870214
	v_mad_co_u64_u32 v[4:5], null, s14, 10, v[4:5]             // 000000003364: d6fe7c04 0411140e
	v_add3_u32 v9, v6, v20, 0x7fff                             // 00000000336c: d6550009 03fe2906 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 000000003378: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 00000000337c: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 000000003380: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s15, 10, v[5:6]             // 000000003388: d6fe7c05 0415140f
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 000000003390: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003394: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 000000003398: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000033a0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 0000000033a4: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 0000000033ac: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000033bc: 8c7e017e
	v_cmp_lt_i64_e64 s0, 6, v[0:1]                             // 0000000033c0: d4510000 02020086
	s_and_b32 s0, s0, vcc_lo                                   // 0000000033c8: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033cc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000033d0: be812000
	s_cbranch_execz 33                                         // 0000000033d4: bfa50021 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x195c>
	v_add_co_u32 v4, s0, s16, v2                               // 0000000033d8: d7000004 02020410
	s_wait_alu depctr_va_sdst(0)                               // 0000000033e0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v3, s0                  // 0000000033e4: d5207c05 00020611
	v_bfe_u32 v6, v10, 16, 1                                   // 0000000033ec: d6100006 0205210a
	v_or_b32_e32 v8, 0x400000, v10                             // 0000000033f4: 381014ff 00400000
	v_cmp_u_f32_e64 s0, v10, v10                               // 0000000033fc: d4180000 0202150a
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003404: bf870214
	v_mad_co_u64_u32 v[4:5], null, s14, 12, v[4:5]             // 000000003408: d6fe7c04 0411180e
	v_add3_u32 v9, v6, v10, 0x7fff                             // 000000003410: d6550009 03fe1506 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000341c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003420: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 000000003424: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s15, 12, v[5:6]             // 00000000342c: d6fe7c05 0415180f
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 000000003434: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003438: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 00000000343c: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003444: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 000000003448: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 000000003450: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000345c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003460: 8c7e017e
	v_cmp_lt_i64_e64 s0, 7, v[0:1]                             // 000000003464: d4510000 02020087
	s_mov_b32 s1, 0                                            // 00000000346c: be810080
	s_and_b32 s2, s0, vcc_lo                                   // 000000003470: 8b026a00
	s_mov_b32 s0, 0                                            // 000000003474: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003478: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000347c: be832002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003480: bf88ff9e
	s_xor_b32 s2, exec_lo, s3                                  // 000000003484: 8d02037e
	v_add_co_u32 v0, vcc_lo, s16, v2                           // 000000003488: d7006a00 02020410
	s_wait_alu depctr_va_vcc(0)                                // 000000003490: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v3, vcc_lo              // 000000003494: d5207c01 01aa0611
	s_mov_b32 s0, exec_lo                                      // 00000000349c: be80007e
	v_mad_co_u64_u32 v[0:1], null, s14, 14, v[0:1]             // 0000000034a0: d6fe7c00 04011c0e
	s_delay_alu instid0(valu_dep_1)                            // 0000000034a8: bf870001
	v_mad_co_u64_u32 v[1:2], null, s15, 14, v[1:2]             // 0000000034ac: d6fe7c01 04051c0f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000034b8: 8c7e027e
	s_delay_alu instid0(salu_cycle_1)                          // 0000000034bc: bf870009
	s_and_b32 vcc_lo, exec_lo, s1                              // 0000000034c0: 8b6a017e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034c4: bf88ff9e
	s_cbranch_vccz 821                                         // 0000000034c8: bfa30335 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x26a0>
	s_and_b32 s0, s11, exec_lo                                 // 0000000034cc: 8b007e0b
	s_cselect_b32 s0, 1, 0                                     // 0000000034d0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034d4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 0000000034d8: bf078100
	s_cbranch_scc1 20                                          // 0000000034dc: bfa20014 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x1a30>
	v_lshl_or_b32 v3, v24, 3, s22                              // 0000000034e0: d6560003 00590718
	v_dual_mov_b32 v4, s23 :: v_dual_mov_b32 v15, s23          // 0000000034e8: ca100017 040e0017
	v_dual_mov_b32 v10, s23 :: v_dual_mov_b32 v19, s23         // 0000000034f0: ca100017 0a120017
	v_dual_mov_b32 v6, s23 :: v_dual_mov_b32 v17, s23          // 0000000034f8: ca100017 06100017
	s_delay_alu instid0(valu_dep_4)                            // 000000003500: bf870004
	v_or_b32_e32 v9, 1, v3                                     // 000000003504: 38120681
	v_or_b32_e32 v5, 2, v3                                     // 000000003508: 380a0682
	v_or_b32_e32 v7, 3, v3                                     // 00000000350c: 380e0683
	v_dual_mov_b32 v8, s23 :: v_dual_mov_b32 v21, s23          // 000000003510: ca100017 08140017
	v_or_b32_e32 v14, 4, v3                                    // 000000003518: 381c0684
	v_or_b32_e32 v18, 5, v3                                    // 00000000351c: 38240685
	v_or_b32_e32 v16, 6, v3                                    // 000000003520: 38200686
	v_or_b32_e32 v20, 7, v3                                    // 000000003524: 38280687
	s_mov_b32 s0, 0                                            // 000000003528: be800080
	s_branch 1                                                 // 00000000352c: bfa00001 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x1a34>
	s_mov_b32 s0, -1                                           // 000000003530: be8000c1
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v27, 0             // 000000003534: ca100080 161a0080
	s_wait_alu depctr_sa_sdst(0)                               // 00000000353c: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000003540: 8b007e00
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v33, 0              // 000000003544: ca100080 02200080
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0             // 00000000354c: ca100080 1a180080
	v_mov_b32_e32 v28, 0                                       // 000000003554: 7e380280
	v_mov_b32_e32 v30, 0                                       // 000000003558: 7e3c0280
	s_cselect_b32 s0, 1, 0                                     // 00000000355c: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003560: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000003564: bf078100
	s_cbranch_scc1 532                                         // 000000003568: bfa20214 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x22bc>
	v_dual_mov_b32 v25, 0 :: v_dual_lshlrev_b32 v2, 3, v24     // 00000000356c: ca220080 19023083
	v_add_co_u32 v0, vcc_lo, s30, v11                          // 000000003574: d7006a00 0202161e
	s_wait_alu depctr_va_vcc(0)                                // 00000000357c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s31, v12, vcc_lo             // 000000003580: d5207c01 01aa181f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_2)// 000000003588: bf870153
	v_or_b32_e32 v3, s22, v2                                   // 00000000358c: 38060416
	v_mov_b32_e32 v4, s23                                      // 000000003590: 7e080217
	s_lshr_b64 s[2:3], s[26:27], 5                             // 000000003594: 8582851a
	s_lshr_b32 s1, s27, 5                                      // 000000003598: 8501851b
	v_add_co_u32 v5, s0, s28, v13                              // 00000000359c: d7000005 02021a1c
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[3:4]                  // 0000000035a4: 7ca8060c
	s_wait_alu depctr_va_sdst(0)                               // 0000000035a8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s29, v23, s0                 // 0000000035ac: d5207c06 00022e1d
	v_cmp_gt_i64_e64 s0, s[14:15], v[11:12]                    // 0000000035b4: d4540000 0202160e
	v_mov_b32_e32 v10, s23                                     // 0000000035bc: 7e140217
	v_mad_co_u64_u32 v[0:1], null, s14, v2, v[0:1]             // 0000000035c0: d6fe7c00 0402040e
	s_wait_alu depctr_va_vcc(0)                                // 0000000035c8: bf88ff9d
	v_cndmask_b32_e32 v7, 0, v3, vcc_lo                        // 0000000035cc: 020e0680
	v_cndmask_b32_e64 v9, 0, s23, vcc_lo                       // 0000000035d0: d5010009 01a82e80
	v_add_co_u32 v13, vcc_lo, v5, v2                           // 0000000035d8: d7006a0d 02020505
	s_wait_alu depctr_va_sdst(0)                               // 0000000035e0: bf88f19f
	v_cndmask_b32_e64 v15, 0, v12, s0                          // 0000000035e4: d501000f 00021880
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035ec: bf88ff9e
	v_mul_lo_u32 v16, s1, v7                                   // 0000000035f0: d72c0010 02020e01
	v_mad_co_u64_u32 v[7:8], null, s2, v7, 0                   // 0000000035f8: d6fe7c07 02020e02
	v_mul_lo_u32 v17, s2, v9                                   // 000000003600: d72c0011 02021202
	v_cndmask_b32_e64 v14, 0, v11, s0                          // 000000003608: d501000e 00021680
	v_or_b32_e32 v9, 1, v3                                     // 000000003610: 38120681
	s_wait_alu depctr_va_vcc(0)                                // 000000003614: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, 0, v6, vcc_lo               // 000000003618: d5207c17 01aa0c80
	v_mad_co_u64_u32 v[1:2], null, s15, v2, v[1:2]             // 000000003620: d6fe7c01 0406040f
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]                    // 000000003628: 3e1c1c82
	v_or_b32_e32 v5, 2, v3                                     // 00000000362c: 380a0682
	v_add3_u32 v8, v8, v17, v16                                // 000000003630: d6550008 04422308
	v_mov_b32_e32 v6, s23                                      // 000000003638: 7e0c0217
	s_mov_b64 s[4:5], 0                                        // 00000000363c: be840180
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000003640: bf870092
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]                        // 000000003644: 3e0e0e82
	v_add_co_u32 v24, s0, s20, v7                              // 000000003648: d7000018 02020e14
	s_wait_alu depctr_va_sdst(0)                               // 000000003650: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003654: bf870002
	v_add_co_ci_u32_e64 v29, null, s21, v8, s0                 // 000000003658: d5207c1d 00021015
	v_add_co_u32 v31, s0, s24, v14                             // 000000003660: d700001f 02021c18
	s_wait_alu depctr_va_sdst(0)                               // 000000003668: bf88f19f
	v_add_co_ci_u32_e64 v32, null, s25, v15, s0                // 00000000366c: d5207c20 00021e19
	v_mov_b32_e32 v15, s23                                     // 000000003674: 7e1e0217
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[9:10]                 // 000000003678: 7ca8120c
	v_or_b32_e32 v7, 3, v3                                     // 00000000367c: 380e0683
	v_mov_b32_e32 v8, s23                                      // 000000003680: 7e100217
	v_or_b32_e32 v14, 4, v3                                    // 000000003684: 381c0684
	s_wait_alu depctr_va_vcc(0)                                // 000000003688: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v9, vcc_lo                        // 00000000368c: 02041280
	v_cndmask_b32_e64 v16, 0, s23, vcc_lo                      // 000000003690: d5010010 01a82e80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003698: bf870112
	v_mul_lo_u32 v18, s1, v2                                   // 00000000369c: d72c0012 02020401
	v_mul_lo_u32 v19, s2, v16                                  // 0000000036a4: d72c0013 02022002
	v_mad_co_u64_u32 v[16:17], null, s2, v2, 0                 // 0000000036ac: d6fe7c10 02020402
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000036b4: bf870091
	v_add3_u32 v17, v17, v19, v18                              // 0000000036b8: d6550011 044a2711
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]                    // 0000000036c0: 3e202082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000036c4: bf870121
	v_add_co_u32 v34, s0, s20, v16                             // 0000000036c8: d7000022 02022014
	s_wait_alu depctr_va_sdst(0)                               // 0000000036d0: bf88f19f
	v_add_co_ci_u32_e64 v35, null, s21, v17, s0                // 0000000036d4: d5207c23 00022215
	v_mov_b32_e32 v17, s23                                     // 0000000036dc: 7e220217
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[5:6]                  // 0000000036e0: 7ca80a0c
	s_wait_alu depctr_va_vcc(0)                                // 0000000036e4: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v5, vcc_lo                        // 0000000036e8: 02040a80
	v_cndmask_b32_e64 v20, 0, s23, vcc_lo                      // 0000000036ec: d5010014 01a82e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[7:8]                  // 0000000036f4: 7ca80e0c
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000036f8: bf870193
	v_mul_lo_u32 v21, s1, v2                                   // 0000000036fc: d72c0015 02020401
	v_mul_lo_u32 v20, s2, v20                                  // 000000003704: d72c0014 02022802
	v_mad_co_u64_u32 v[18:19], null, s2, v2, 0                 // 00000000370c: d6fe7c12 02020402
	s_wait_alu depctr_va_vcc(0)                                // 000000003714: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v7, vcc_lo                        // 000000003718: 02040e80
	v_cndmask_b32_e64 v22, 0, s23, vcc_lo                      // 00000000371c: d5010016 01a82e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[14:15]                // 000000003724: 7ca81c0c
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003728: bf870193
	v_mul_lo_u32 v28, s1, v2                                   // 00000000372c: d72c001c 02020401
	v_mul_lo_u32 v22, s2, v22                                  // 000000003734: d72c0016 02022c02
	v_add3_u32 v19, v19, v20, v21                              // 00000000373c: d6550013 04562913
	v_mad_co_u64_u32 v[20:21], null, s2, v2, 0                 // 000000003744: d6fe7c14 02020402
	s_wait_alu depctr_va_vcc(0)                                // 00000000374c: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v14, vcc_lo                       // 000000003750: 02041c80
	v_cndmask_b32_e64 v16, 0, s23, vcc_lo                      // 000000003754: d5010010 01a82e80
	v_lshlrev_b64_e32 v[26:27], 2, v[18:19]                    // 00000000375c: 3e342482
	v_or_b32_e32 v18, 5, v3                                    // 000000003760: 38240685
	v_mov_b32_e32 v19, s23                                     // 000000003764: 7e260217
	v_mad_co_u64_u32 v[40:41], null, s2, v2, 0                 // 000000003768: d6fe7c28 02020402
	v_add3_u32 v21, v21, v22, v28                              // 000000003770: d6550015 04722d15
	v_mul_lo_u32 v22, s1, v2                                   // 000000003778: d72c0016 02020401
	v_mul_lo_u32 v28, s2, v16                                  // 000000003780: d72c001c 02022002
	v_or_b32_e32 v16, 6, v3                                    // 000000003788: 38200686
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[18:19]                // 00000000378c: 7ca8240c
	v_add_co_u32 v36, s0, s20, v26                             // 000000003790: d7000024 02023414
	s_wait_alu depctr_va_sdst(0)                               // 000000003798: bf88f19f
	v_add_co_ci_u32_e64 v37, null, s21, v27, s0                // 00000000379c: d5207c25 00023615
	v_lshlrev_b64_e32 v[26:27], 2, v[20:21]                    // 0000000037a4: 3e342882
	v_cmp_gt_i64_e64 s0, s[12:13], v[16:17]                    // 0000000037a8: d4540000 0202200c
	v_or_b32_e32 v20, 7, v3                                    // 0000000037b0: 38280687
	v_mov_b32_e32 v21, s23                                     // 0000000037b4: 7e2a0217
	v_add3_u32 v41, v41, v28, v22                              // 0000000037b8: d6550029 045a3929
	s_wait_alu depctr_va_vcc(0)                                // 0000000037c0: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v18, vcc_lo                       // 0000000037c4: 02042480
	v_cndmask_b32_e64 v22, 0, s23, vcc_lo                      // 0000000037c8: d5010016 01a82e80
	s_wait_alu depctr_va_sdst(0)                               // 0000000037d0: bf88f19f
	v_cndmask_b32_e64 v30, 0, v16, s0                          // 0000000037d4: d501001e 00022080
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[20:21]                // 0000000037dc: 7ca8280c
	v_cndmask_b32_e64 v33, 0, s23, s0                          // 0000000037e0: d5010021 00002e80
	v_mul_lo_u32 v28, s1, v2                                   // 0000000037e8: d72c001c 02020401
	v_mul_lo_u32 v22, s2, v22                                  // 0000000037f0: d72c0016 02022c02
	v_mad_co_u64_u32 v[42:43], null, s2, v2, 0                 // 0000000037f8: d6fe7c2a 02020402
	v_mul_lo_u32 v2, s1, v30                                   // 000000003800: d72c0002 02023c01
	v_mad_co_u64_u32 v[44:45], null, s2, v30, 0                // 000000003808: d6fe7c2c 02023c02
	s_wait_alu depctr_va_vcc(0)                                // 000000003810: bf88ff9d
	v_cndmask_b32_e32 v30, 0, v20, vcc_lo                      // 000000003814: 023c2880
	v_cndmask_b32_e64 v46, 0, s23, vcc_lo                      // 000000003818: d501002e 01a82e80
	v_mul_lo_u32 v33, s2, v33                                  // 000000003820: d72c0021 02024202
	v_add_co_u32 v38, vcc_lo, s20, v26                         // 000000003828: d7006a26 02023414
	v_add3_u32 v43, v43, v22, v28                              // 000000003830: d655002b 04722d2b
	v_mul_lo_u32 v22, s1, v30                                  // 000000003838: d72c0016 02023c01
	v_mul_lo_u32 v28, s2, v46                                  // 000000003840: d72c001c 02025c02
	v_mad_co_u64_u32 v[46:47], null, s2, v30, 0                // 000000003848: d6fe7c2e 02023c02
	s_wait_alu depctr_va_vcc(0)                                // 000000003850: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s21, v27, vcc_lo            // 000000003854: d5207c27 01aa3615
	v_lshlrev_b64_e32 v[26:27], 2, v[40:41]                    // 00000000385c: 3e345082
	v_add3_u32 v45, v45, v33, v2                               // 000000003860: d655002d 040a432d
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]                    // 000000003868: 3e545482
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v30, 0             // 00000000386c: ca100080 211e0080
	v_add3_u32 v47, v47, v28, v22                              // 000000003874: d655002f 045a392f
	v_add_co_u32 v40, vcc_lo, s20, v26                         // 00000000387c: d7006a28 02023414
	s_wait_alu depctr_va_vcc(0)                                // 000000003884: bf88ff9d
	v_add_co_ci_u32_e64 v41, null, s21, v27, vcc_lo            // 000000003888: d5207c29 01aa3615
	v_lshlrev_b64_e32 v[26:27], 2, v[44:45]                    // 000000003890: 3e345882
	v_lshlrev_b64_e32 v[46:47], 2, v[46:47]                    // 000000003894: 3e5c5c82
	v_add_co_u32 v42, vcc_lo, s20, v42                         // 000000003898: d7006a2a 02025414
	s_wait_alu depctr_va_vcc(0)                                // 0000000038a0: bf88ff9d
	v_add_co_ci_u32_e64 v43, null, s21, v43, vcc_lo            // 0000000038a4: d5207c2b 01aa5615
	s_delay_alu instid0(valu_dep_4)                            // 0000000038ac: bf870004
	v_add_co_u32 v44, vcc_lo, s20, v26                         // 0000000038b0: d7006a2c 02023414
	s_wait_alu depctr_va_vcc(0)                                // 0000000038b8: bf88ff9d
	v_add_co_ci_u32_e64 v45, null, s21, v27, vcc_lo            // 0000000038bc: d5207c2d 01aa3615
	v_add_co_u32 v46, vcc_lo, s20, v46                         // 0000000038c4: d7006a2e 02025c14
	s_wait_alu depctr_va_vcc(0)                                // 0000000038cc: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s21, v47, vcc_lo            // 0000000038d0: d5207c2f 01aa5e15
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0             // 0000000038d8: ca100080 1c1a0080
	v_mov_b32_e32 v26, 0                                       // 0000000038e0: 7e340280
	v_mov_b32_e32 v2, 0                                        // 0000000038e4: 7e040280
	v_mov_b32_e32 v22, 0                                       // 0000000038e8: 7e2c0280
	s_lshl_b64 s[2:3], s[14:15], 4                             // 0000000038ec: 8482840e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038f0: bf88ff9e
	v_add_co_u32 v48, vcc_lo, v13, s4                          // 0000000038f4: d7006a30 0200090d
	s_lshr_b64 s[6:7], s[4:5], 3                               // 0000000038fc: 85868304
	s_wait_alu depctr_va_vcc(0)                                // 000000003900: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s5, v23, vcc_lo             // 000000003904: d5207c31 01aa2e05
	s_wait_alu depctr_sa_sdst(0)                               // 00000000390c: bf88ff9e
	v_add_co_u32 v52, vcc_lo, v24, s6                          // 000000003910: d7006a34 02000d18
	s_wait_alu depctr_va_vcc(0)                                // 000000003918: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s7, v29, vcc_lo             // 00000000391c: d5207c35 01aa3a07
	v_add_co_u32 v54, vcc_lo, v34, s6                          // 000000003924: d7006a36 02000d22
	s_wait_alu depctr_va_vcc(0)                                // 00000000392c: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s7, v35, vcc_lo             // 000000003930: d5207c37 01aa4607
	v_add_co_u32 v56, vcc_lo, v36, s6                          // 000000003938: d7006a38 02000d24
	v_mad_co_u64_u32 v[50:51], null, s4, s14, v[0:1]           // 000000003940: d6fe7c32 04001c04
	s_wait_alu depctr_va_vcc(0)                                // 000000003948: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s7, v37, vcc_lo             // 00000000394c: d5207c39 01aa4a07
	v_add_co_u32 v58, vcc_lo, v38, s6                          // 000000003954: d7006a3a 02000d26
	s_wait_alu depctr_va_vcc(0)                                // 00000000395c: bf88ff9d
	v_add_co_ci_u32_e64 v59, null, s7, v39, vcc_lo             // 000000003960: d5207c3b 01aa4e07
	v_add_co_u32 v60, vcc_lo, v40, s6                          // 000000003968: d7006a3c 02000d28
	s_lshr_b64 s[0:1], s[4:5], 5                               // 000000003970: 85808504
	s_wait_alu depctr_va_vcc(0)                                // 000000003974: bf88ff9d
	v_add_co_ci_u32_e64 v61, null, s7, v41, vcc_lo             // 000000003978: d5207c3d 01aa5207
	v_add_co_u32 v62, vcc_lo, v42, s6                          // 000000003980: d7006a3e 02000d2a
	s_mul_i32 s8, s5, s14                                      // 000000003988: 96080e05
	s_mul_i32 s9, s4, s15                                      // 00000000398c: 96090f04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003990: bf88ff9e
	s_mul_u64 s[0:1], s[0:1], s[14:15]                         // 000000003994: aa800e00
	s_wait_alu depctr_va_vcc(0)                                // 000000003998: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, s7, v43, vcc_lo             // 00000000399c: d5207c3f 01aa5607
	v_add_co_u32 v64, vcc_lo, v44, s6                          // 0000000039a4: d7006a40 02000d2c
	v_add3_u32 v51, s9, s8, v51                                // 0000000039ac: d6550033 04cc1009
	s_wait_alu depctr_va_vcc(0)                                // 0000000039b4: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s7, v45, vcc_lo             // 0000000039b8: d5207c41 01aa5a07
	v_add_co_u32 v66, vcc_lo, v46, s6                          // 0000000039c0: d7006a42 02000d2e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039c8: bf88ff9e
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000039cc: 84808200
	s_wait_alu depctr_va_vcc(0)                                // 0000000039d0: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s7, v47, vcc_lo             // 0000000039d4: d5207c43 01aa5e07
	s_clause 0x1                                               // 0000000039dc: bf850001
	global_load_b64 v[68:69], v[48:49], off                    // 0000000039e0: ee05407c 00000044 00000030
	global_load_b64 v[70:71], v[48:49], off offset:16          // 0000000039ec: ee05407c 00000046 00001030
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039f8: bf88ff9e
	v_add_co_u32 v48, vcc_lo, v31, s0                          // 0000000039fc: d7006a30 0200011f
	s_clause 0x7                                               // 000000003a04: bf850007
	global_load_b32 v72, v[52:53], off                         // 000000003a08: ee05007c 00000048 00000034
	global_load_b32 v73, v[54:55], off                         // 000000003a14: ee05007c 00000049 00000036
	global_load_b32 v74, v[56:57], off                         // 000000003a20: ee05007c 0000004a 00000038
	global_load_b32 v58, v[58:59], off                         // 000000003a2c: ee05007c 0000003a 0000003a
	global_load_b32 v59, v[60:61], off                         // 000000003a38: ee05007c 0000003b 0000003c
	global_load_b32 v60, v[62:63], off                         // 000000003a44: ee05007c 0000003c 0000003e
	global_load_b32 v61, v[64:65], off                         // 000000003a50: ee05007c 0000003d 00000040
	global_load_b32 v62, v[66:67], off                         // 000000003a5c: ee05007c 0000003e 00000042
	v_add_co_u32 v54, s0, v50, s14                             // 000000003a68: d7000036 02001d32
	s_wait_alu depctr_va_sdst(0)                               // 000000003a70: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s15, v51, s0                // 000000003a74: d5207c37 0002660f
	s_wait_alu depctr_va_vcc(0)                                // 000000003a7c: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s1, v32, vcc_lo             // 000000003a80: d5207c31 01aa4001
	v_add_co_u32 v52, vcc_lo, v50, s2                          // 000000003a88: d7006a34 02000532
	v_add_co_u32 v56, s0, v54, s14                             // 000000003a90: d7000038 02001d36
	s_wait_alu depctr_va_vcc(0)                                // 000000003a98: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s3, v51, vcc_lo             // 000000003a9c: d5207c35 01aa6603
	s_wait_alu depctr_va_sdst(0)                               // 000000003aa4: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s15, v55, s0                // 000000003aa8: d5207c39 00026e0f
	s_clause 0x1                                               // 000000003ab0: bf850001
	global_load_u8 v63, v[50:51], off                          // 000000003ab4: ee04007c 0000003f 00000032
	global_load_u8 v64, v[54:55], off                          // 000000003ac0: ee04007c 00000040 00000036
	s_add_nc_u64 s[4:5], s[4:5], 32                            // 000000003acc: a984a004
	s_clause 0x1                                               // 000000003ad0: bf850001
	global_load_u8 v66, v[56:57], off                          // 000000003ad4: ee04007c 00000042 00000038
	global_load_u8 v65, v[52:53], off                          // 000000003ae0: ee04007c 00000041 00000034
	v_add_co_u32 v50, vcc_lo, v52, s14                         // 000000003aec: d7006a32 02001d34
	s_wait_alu depctr_va_vcc(0)                                // 000000003af4: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s15, v53, vcc_lo            // 000000003af8: d5207c33 01aa6a0f
	v_add_co_u32 v52, s0, v56, s14                             // 000000003b00: d7000034 02001d38
	s_delay_alu instid0(valu_dep_3)                            // 000000003b08: bf870003
	v_add_co_u32 v54, vcc_lo, v50, s14                         // 000000003b0c: d7006a36 02001d32
	s_wait_alu depctr_va_sdst(0)                               // 000000003b14: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s15, v57, s0                // 000000003b18: d5207c35 0002720f
	s_wait_alu depctr_va_vcc(0)                                // 000000003b20: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s15, v51, vcc_lo            // 000000003b24: d5207c37 01aa660f
	v_add_co_u32 v56, vcc_lo, v54, s14                         // 000000003b2c: d7006a38 02001d36
	s_clause 0x2                                               // 000000003b34: bf850002
	global_load_u8 v67, v[50:51], off                          // 000000003b38: ee04007c 00000043 00000032
	global_load_u8 v75, v[52:53], off                          // 000000003b44: ee04007c 0000004b 00000034
	global_load_u8 v76, v[54:55], off                          // 000000003b50: ee04007c 0000004c 00000036
	v_add_co_u32 v50, s0, v52, s14                             // 000000003b5c: d7000032 02001d34
	s_wait_alu depctr_va_vcc(0)                                // 000000003b64: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s15, v55, vcc_lo            // 000000003b68: d5207c39 01aa6e0f
	s_wait_alu depctr_va_sdst(0)                               // 000000003b70: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s15, v53, s0                // 000000003b74: d5207c33 00026a0f
	v_add_co_u32 v52, vcc_lo, v56, s14                         // 000000003b7c: d7006a34 02001d38
	v_add_co_u32 v54, s0, v50, s14                             // 000000003b84: d7000036 02001d32
	s_wait_alu depctr_va_vcc(0)                                // 000000003b8c: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s15, v57, vcc_lo            // 000000003b90: d5207c35 01aa720f
	s_wait_alu depctr_va_sdst(0)                               // 000000003b98: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s15, v51, s0                // 000000003b9c: d5207c37 0002660f
	s_clause 0x2                                               // 000000003ba4: bf850002
	global_load_u8 v77, v[50:51], off                          // 000000003ba8: ee04007c 0000004d 00000032
	global_load_u8 v78, v[56:57], off                          // 000000003bb4: ee04007c 0000004e 00000038
	global_load_u8 v80, v[52:53], off                          // 000000003bc0: ee04007c 00000050 00000034
	v_add_co_u32 v50, vcc_lo, v52, s14                         // 000000003bcc: d7006a32 02001d34
	v_add_co_u32 v56, s0, v54, s14                             // 000000003bd4: d7000038 02001d36
	s_wait_alu depctr_va_sdst(0)                               // 000000003bdc: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s15, v55, s0                // 000000003be0: d5207c39 00026e0f
	s_wait_alu depctr_va_vcc(0)                                // 000000003be8: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s15, v53, vcc_lo            // 000000003bec: d5207c33 01aa6a0f
	v_add_co_u32 v52, s0, v56, s14                             // 000000003bf4: d7000034 02001d38
	global_load_u8 v79, v[54:55], off                          // 000000003bfc: ee04007c 0000004f 00000036
	s_wait_alu depctr_va_sdst(0)                               // 000000003c08: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s15, v57, s0                // 000000003c0c: d5207c35 0002720f
	s_clause 0x1                                               // 000000003c14: bf850001
	global_load_u8 v57, v[56:57], off                          // 000000003c18: ee04007c 00000039 00000038
	global_load_u8 v81, v[50:51], off                          // 000000003c24: ee04007c 00000051 00000032
	v_add_co_u32 v54, vcc_lo, v50, s14                         // 000000003c30: d7006a36 02001d32
	s_wait_alu depctr_va_vcc(0)                                // 000000003c38: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s15, v51, vcc_lo            // 000000003c3c: d5207c37 01aa660f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c44: bf88ff9e
	v_cmp_lt_i64_e64 s0, s[4:5], s[18:19]                      // 000000003c48: d4510000 02002404
	v_add_co_u32 v50, vcc_lo, v54, s14                         // 000000003c50: d7006a32 02001d36
	s_wait_alu depctr_va_vcc(0)                                // 000000003c58: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s15, v55, vcc_lo            // 000000003c5c: d5207c33 01aa6e0f
	s_clause 0x1                                               // 000000003c64: bf850001
	global_load_u8 v52, v[52:53], off                          // 000000003c68: ee04007c 00000034 00000034
	global_load_u8 v53, v[54:55], off                          // 000000003c74: ee04007c 00000035 00000036
	global_load_u8 v50, v[50:51], off                          // 000000003c80: ee04007c 00000032 00000032
	global_load_b32 v48, v[48:49], off                         // 000000003c8c: ee05007c 00000030 00000030
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000003c98: 8b6a007e
	s_wait_loadcnt 0xc                                         // 000000003c9c: bfc0000c
	v_perm_b32 v49, v65, v67, 0xc0c0004                        // 000000003ca0: d6440031 03fe8741 0c0c0004
	s_wait_loadcnt 0xb                                         // 000000003cac: bfc0000b
	v_perm_b32 v51, v66, v75, 0xc0c0004                        // 000000003cb0: d6440033 03fe9742 0c0c0004
	s_wait_loadcnt 0x8                                         // 000000003cbc: bfc00008
	v_perm_b32 v54, v76, v78, 0xc0c0004                        // 000000003cc0: d6440036 03fe9d4c 0c0c0004
	s_wait_loadcnt 0x0                                         // 000000003ccc: bfc00000
	v_dual_mul_f32 v73, v48, v73 :: v_dual_mul_f32 v74, v48, v74// 000000003cd0: c8c69330 494a9530
	v_mul_f32_e32 v72, v72, v48                                // 000000003cd8: 10906148
	v_dual_mul_f32 v82, v48, v58 :: v_dual_mul_f32 v83, v48, v59// 000000003cdc: c8c67530 52527730
	v_dual_mul_f32 v60, v48, v60 :: v_dual_mul_f32 v61, v48, v61// 000000003ce4: c8c67930 3c3c7b30
	v_mul_f32_e32 v62, v48, v62                                // 000000003cec: 107c7d30
	v_perm_b32 v48, v63, v64, 0xc0c0004                        // 000000003cf0: d6440030 03fe813f 0c0c0004
	v_lshl_or_b32 v58, v54, 16, v49                            // 000000003cfc: d656003a 04c52136
	v_perm_b32 v49, v57, v52, 0xc0c0004                        // 000000003d04: d6440031 03fe6939 0c0c0004
	v_perm_b32 v59, v80, v81, 0xc0c0004                        // 000000003d10: d644003b 03fea350 0c0c0004
	v_perm_b32 v63, v53, v50, 0xc0c0004                        // 000000003d1c: d644003f 03fe6535 0c0c0004
	v_lshl_or_b32 v56, v51, 16, v48                            // 000000003d28: d6560038 04c12133
	v_perm_b32 v48, v77, v79, 0xc0c0004                        // 000000003d30: d6440030 03fe9f4d 0c0c0004
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000003d3c: bf870113
	v_lshl_or_b32 v59, v63, 16, v59                            // 000000003d40: d656003b 04ed213f
	v_lshl_or_b32 v57, v49, 16, v48                            // 000000003d48: d6560039 04c12131
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003d50: bf870091
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[68:69], v[56:57], 0// 000000003d54: cc464030 1a027144
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[70:71], v[58:59], v[48:55]// 000000003d5c: cc464030 1cc27546
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_2)// 000000003d64: bf870111
	v_dual_mul_f32 v49, v49, v73 :: v_dual_mul_f32 v48, v48, v72// 000000003d68: c8c69331 31309130
	v_mul_f32_e32 v51, v51, v82                                // 000000003d70: 1066a533
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_4)// 000000003d74: bf870213
	v_dual_mul_f32 v50, v50, v74 :: v_dual_mul_f32 v53, v53, v60// 000000003d78: c8c69532 32347935
	v_dual_mul_f32 v52, v52, v83 :: v_dual_mul_f32 v55, v55, v62// 000000003d80: c8c6a734 34367d37
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000003d88: bf870194
	v_dual_mul_f32 v54, v54, v61 :: v_dual_add_f32 v25, v25, v48// 000000003d8c: c8c87b36 36186119
	v_dual_add_f32 v33, v33, v49 :: v_dual_add_f32 v30, v30, v50// 000000003d94: c9086321 211e651e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003d9c: bf870223
	v_dual_add_f32 v28, v28, v51 :: v_dual_add_f32 v27, v27, v52// 000000003da0: c908671c 1c1a691b
	v_add_f32_e32 v26, v26, v53                                // 000000003da8: 06346b1a
	v_add_f32_e32 v2, v2, v54                                  // 000000003dac: 06046d02
	v_add_f32_e32 v22, v22, v55                                // 000000003db0: 062c6f16
	s_wait_alu depctr_sa_sdst(0)                               // 000000003db4: bf88ff9e
	s_cbranch_vccnz 65229                                      // 000000003db8: bfa4fecd <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x1df0>
	v_mul_lo_u32 v13, s15, v3                                  // 000000003dbc: d72c000d 0202060f
	v_mul_lo_u32 v4, s14, v4                                   // 000000003dc4: d72c0004 0202080e
	v_mad_co_u64_u32 v[0:1], null, s14, v3, 0                  // 000000003dcc: d6fe7c00 0202060e
	v_mul_lo_u32 v23, s15, v9                                  // 000000003dd4: d72c0017 0202120f
	v_mul_lo_u32 v24, s14, v10                                 // 000000003ddc: d72c0018 0202140e
	v_or_b32_e32 v31, 0x400000, v25                            // 000000003de4: 383e32ff 00400000
	v_bfe_u32 v29, v33, 16, 1                                  // 000000003dec: d610001d 02052121
	v_mul_lo_u32 v8, s14, v8                                   // 000000003df4: d72c0008 0202100e
	v_mul_lo_u32 v15, s14, v15                                 // 000000003dfc: d72c000f 02021e0e
	s_mov_b32 s0, -1                                           // 000000003e04: be8000c1
	v_add3_u32 v1, v1, v4, v13                                 // 000000003e08: d6550001 04360901
	v_mad_co_u64_u32 v[3:4], null, s14, v9, 0                  // 000000003e10: d6fe7c03 0202120e
	v_bfe_u32 v13, v25, 16, 1                                  // 000000003e18: d610000d 02052119
	v_lshlrev_b64_e32 v[9:10], 1, v[11:12]                     // 000000003e20: 3e121681
	v_add3_u32 v29, v29, v33, 0x7fff                           // 000000003e24: d655001d 03fe431d 00007fff
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003e30: 3e000081
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_4)// 000000003e34: bf870234
	v_add3_u32 v13, v13, v25, 0x7fff                           // 000000003e38: d655000d 03fe330d 00007fff
	v_add3_u32 v4, v4, v24, v23                                // 000000003e44: d6550004 045e3104
	v_mul_lo_u32 v24, s15, v5                                  // 000000003e4c: d72c0018 02020a0f
	v_add_co_u32 v0, vcc_lo, s16, v0                           // 000000003e54: d7006a00 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003e5c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo              // 000000003e60: d5207c01 01aa0211
	v_cmp_u_f32_e32 vcc_lo, v25, v25                           // 000000003e68: 7c303319
	v_mul_lo_u32 v25, s14, v6                                  // 000000003e6c: d72c0019 02020c0e
	v_mad_co_u64_u32 v[5:6], null, s14, v5, 0                  // 000000003e74: d6fe7c05 02020a0e
	v_lshlrev_b64_e32 v[3:4], 1, v[3:4]                        // 000000003e7c: 3e060681
	v_or_b32_e32 v23, 0x400000, v33                            // 000000003e80: 382e42ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003e88: bf88ff9d
	v_cndmask_b32_e32 v13, v13, v31, vcc_lo                    // 000000003e8c: 021a3f0d
	v_add_co_u32 v0, vcc_lo, v0, v9                            // 000000003e90: d7006a00 02021300
	s_wait_alu depctr_va_vcc(0)                                // 000000003e98: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc_lo              // 000000003e9c: d5207c01 01aa1501
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 000000003ea4: 7c304321
	v_add3_u32 v6, v6, v25, v24                                // 000000003ea8: d6550006 04623306
	v_mul_lo_u32 v25, s15, v7                                  // 000000003eb0: d72c0019 02020e0f
	global_store_d16_hi_b16 v[0:1], v13, off                   // 000000003eb8: ee09407c 06800000 00000000
	v_or_b32_e32 v24, 0x400000, v30                            // 000000003ec4: 38303cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003ecc: bf88ff9d
	v_cndmask_b32_e32 v13, v29, v23, vcc_lo                    // 000000003ed0: 021a2f1d
	v_add_co_u32 v0, vcc_lo, s16, v3                           // 000000003ed4: d7006a00 02020610
	v_bfe_u32 v3, v30, 16, 1                                   // 000000003edc: d6100003 0205211e
	s_wait_alu depctr_va_vcc(0)                                // 000000003ee4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v4, vcc_lo              // 000000003ee8: d5207c01 01aa0811
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003ef0: bf870193
	v_add_co_u32 v0, vcc_lo, v0, v9                            // 000000003ef4: d7006a00 02021300
	v_add3_u32 v23, v3, v30, 0x7fff                            // 000000003efc: d6550017 03fe3d03 00007fff
	v_lshlrev_b64_e32 v[3:4], 1, v[5:6]                        // 000000003f08: 3e060a81
	v_mad_co_u64_u32 v[5:6], null, s14, v7, 0                  // 000000003f0c: d6fe7c05 02020e0e
	s_wait_alu depctr_va_vcc(0)                                // 000000003f14: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc_lo              // 000000003f18: d5207c01 01aa1501
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 000000003f20: 7c303d1e
	global_store_d16_hi_b16 v[0:1], v13, off                   // 000000003f24: ee09407c 06800000 00000000
	v_or_b32_e32 v13, 0x400000, v28                            // 000000003f30: 381a38ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003f38: bf88ff9d
	v_cndmask_b32_e32 v7, v23, v24, vcc_lo                     // 000000003f3c: 020e3117
	v_add_co_u32 v0, vcc_lo, s16, v3                           // 000000003f40: d7006a00 02020610
	v_add3_u32 v6, v6, v8, v25                                 // 000000003f48: d6550006 04661106
	v_bfe_u32 v3, v28, 16, 1                                   // 000000003f50: d6100003 0205211c
	s_wait_alu depctr_va_vcc(0)                                // 000000003f58: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v4, vcc_lo              // 000000003f5c: d5207c01 01aa0811
	v_add_co_u32 v0, vcc_lo, v0, v9                            // 000000003f64: d7006a00 02021300
	s_delay_alu instid0(valu_dep_3)                            // 000000003f6c: bf870003
	v_add3_u32 v8, v3, v28, 0x7fff                             // 000000003f70: d6550008 03fe3903 00007fff
	v_lshlrev_b64_e32 v[3:4], 1, v[5:6]                        // 000000003f7c: 3e060a81
	v_mul_lo_u32 v23, s15, v14                                 // 000000003f80: d72c0017 02021c0f
	v_mad_co_u64_u32 v[5:6], null, s14, v14, 0                 // 000000003f88: d6fe7c05 02021c0e
	s_wait_alu depctr_va_vcc(0)                                // 000000003f90: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc_lo              // 000000003f94: d5207c01 01aa1501
	v_cmp_u_f32_e32 vcc_lo, v28, v28                           // 000000003f9c: 7c30391c
	v_mul_lo_u32 v14, s15, v18                                 // 000000003fa0: d72c000e 0202240f
	global_store_d16_hi_b16 v[0:1], v7, off                    // 000000003fa8: ee09407c 03800000 00000000
	s_wait_alu depctr_va_vcc(0)                                // 000000003fb4: bf88ff9d
	v_cndmask_b32_e32 v7, v8, v13, vcc_lo                      // 000000003fb8: 020e1b08
	v_add_co_u32 v0, vcc_lo, s16, v3                           // 000000003fbc: d7006a00 02020610
	v_add3_u32 v6, v6, v15, v23                                // 000000003fc4: d6550006 045e1f06
	v_bfe_u32 v3, v27, 16, 1                                   // 000000003fcc: d6100003 0205211b
	s_wait_alu depctr_va_vcc(0)                                // 000000003fd4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v4, vcc_lo              // 000000003fd8: d5207c01 01aa0811
	v_add_co_u32 v0, vcc_lo, v0, v9                            // 000000003fe0: d7006a00 02021300
	s_delay_alu instid0(valu_dep_3)                            // 000000003fe8: bf870003
	v_add3_u32 v8, v3, v27, 0x7fff                             // 000000003fec: d6550008 03fe3703 00007fff
	v_lshlrev_b64_e32 v[3:4], 1, v[5:6]                        // 000000003ff8: 3e060a81
	v_mul_lo_u32 v15, s14, v19                                 // 000000003ffc: d72c000f 0202260e
	v_mad_co_u64_u32 v[5:6], null, s14, v18, 0                 // 000000004004: d6fe7c05 0202240e
	s_wait_alu depctr_va_vcc(0)                                // 00000000400c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc_lo              // 000000004010: d5207c01 01aa1501
	v_or_b32_e32 v13, 0x400000, v27                            // 000000004018: 381a36ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 000000004020: 7c30371b
	v_mul_lo_u32 v18, s15, v20                                 // 000000004024: d72c0012 0202280f
	global_store_d16_hi_b16 v[0:1], v7, off                    // 00000000402c: ee09407c 03800000 00000000
	v_bfe_u32 v7, v26, 16, 1                                   // 000000004038: d6100007 0205211a
	v_add3_u32 v6, v6, v15, v14                                // 000000004040: d6550006 043a1f06
	s_wait_alu depctr_va_vcc(0)                                // 000000004048: bf88ff9d
	v_cndmask_b32_e32 v13, v8, v13, vcc_lo                     // 00000000404c: 021a1b08
	v_add_co_u32 v0, vcc_lo, s16, v3                           // 000000004050: d7006a00 02020610
	s_wait_alu depctr_va_vcc(0)                                // 000000004058: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v4, vcc_lo              // 00000000405c: d5207c01 01aa0811
	v_mul_lo_u32 v14, s15, v16                                 // 000000004064: d72c000e 0202200f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000406c: bf8701a3
	v_add_co_u32 v3, vcc_lo, v0, v9                            // 000000004070: d7006a03 02021300
	s_wait_alu depctr_va_vcc(0)                                // 000000004078: bf88ff9d
	v_add_co_ci_u32_e64 v4, null, v1, v10, vcc_lo              // 00000000407c: d5207c04 01aa1501
	v_lshlrev_b64_e32 v[0:1], 1, v[5:6]                        // 000000004084: 3e000a81
	v_mul_lo_u32 v15, s14, v17                                 // 000000004088: d72c000f 0202220e
	v_mad_co_u64_u32 v[5:6], null, s14, v16, 0                 // 000000004090: d6fe7c05 0202200e
	v_add3_u32 v7, v7, v26, 0x7fff                             // 000000004098: d6550007 03fe3507 00007fff
	v_or_b32_e32 v8, 0x400000, v26                             // 0000000040a4: 381034ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v26, v26                           // 0000000040ac: 7c30351a
	v_mul_lo_u32 v19, s14, v21                                 // 0000000040b0: d72c0013 02022a0e
	s_wait_alu depctr_va_vcc(0)                                // 0000000040b8: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000040bc: bf870003
	v_cndmask_b32_e32 v16, v7, v8, vcc_lo                      // 0000000040c0: 02201107
	v_bfe_u32 v7, v2, 16, 1                                    // 0000000040c4: d6100007 02052102
	v_add_co_u32 v8, vcc_lo, s16, v0                           // 0000000040cc: d7006a08 02020010
	v_add3_u32 v6, v6, v15, v14                                // 0000000040d4: d6550006 043a1f06
	s_wait_alu depctr_va_vcc(0)                                // 0000000040dc: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s17, v1, vcc_lo             // 0000000040e0: d5207c11 01aa0211
	v_mad_co_u64_u32 v[0:1], null, s14, v20, 0                 // 0000000040e8: d6fe7c00 0202280e
	v_add3_u32 v14, v7, v2, 0x7fff                             // 0000000040f0: d655000e 03fe0507 00007fff
	v_add_co_u32 v7, vcc_lo, v8, v9                            // 0000000040fc: d7006a07 02021308
	v_lshlrev_b64_e32 v[5:6], 1, v[5:6]                        // 000000004104: 3e0a0a81
	v_or_b32_e32 v15, 0x400000, v2                             // 000000004108: 381e04ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004110: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, v17, v10, vcc_lo             // 000000004114: d5207c08 01aa1511
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 00000000411c: 7c300502
	v_add3_u32 v1, v1, v19, v18                                // 000000004120: d6550001 044a2701
	s_clause 0x1                                               // 000000004128: bf850001
	global_store_d16_hi_b16 v[3:4], v13, off                   // 00000000412c: ee09407c 06800000 00000003
	global_store_d16_hi_b16 v[7:8], v16, off                   // 000000004138: ee09407c 08000000 00000007
	s_wait_alu depctr_va_vcc(0)                                // 000000004144: bf88ff9d
	v_cndmask_b32_e32 v2, v14, v15, vcc_lo                     // 000000004148: 02041f0e
	v_add_co_u32 v5, vcc_lo, s16, v5                           // 00000000414c: d7006a05 02020a10
	s_wait_alu depctr_va_vcc(0)                                // 000000004154: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s17, v6, vcc_lo              // 000000004158: d5207c06 01aa0c11
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000004160: 3e000081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004164: bf8701a3
	v_add_co_u32 v5, vcc_lo, v5, v9                            // 000000004168: d7006a05 02021305
	s_wait_alu depctr_va_vcc(0)                                // 000000004170: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, v6, v10, vcc_lo              // 000000004174: d5207c06 01aa1506
	s_delay_alu instid0(valu_dep_3)                            // 00000000417c: bf870003
	v_add_co_u32 v0, vcc_lo, s16, v0                           // 000000004180: d7006a00 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000004188: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo              // 00000000418c: d5207c01 01aa0211
	global_store_d16_hi_b16 v[5:6], v2, off                    // 000000004194: ee09407c 01000000 00000005
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041a0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000041a4: be812000
	s_cbranch_execnz 1                                         // 0000000041a8: bfa60001 <tessera_rocm_scaled_matmul_e44055bd28b6555a+0x26b0>
	s_endpgm                                                   // 0000000041ac: bfb00000
	v_bfe_u32 v2, v22, 16, 1                                   // 0000000041b0: d6100002 02052116
	v_or_b32_e32 v4, 0x400000, v22                             // 0000000041b8: 38082cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v22, v22                           // 0000000041c0: 7c302d16
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_2)// 0000000041c4: bf870133
	v_add3_u32 v5, v2, v22, 0x7fff                             // 0000000041c8: d6550005 03fe2d02 00007fff
	v_lshlrev_b64_e32 v[2:3], 1, v[11:12]                      // 0000000041d4: 3e041681
	s_wait_alu depctr_va_vcc(0)                                // 0000000041d8: bf88ff9d
	v_cndmask_b32_e32 v4, v5, v4, vcc_lo                       // 0000000041dc: 02080905
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000041e0: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 0000000041e4: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 0000000041ec: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 0000000041f0: d5207c01 01aa0701
	global_store_d16_hi_b16 v[0:1], v4, off                    // 0000000041f8: ee09407c 02000000 00000000
	s_endpgm                                                   // 000000004204: bfb00000
	s_code_end                                                 // 000000004208: bf9f0000
	s_code_end                                                 // 00000000420c: bf9f0000
	s_code_end                                                 // 000000004210: bf9f0000
	s_code_end                                                 // 000000004214: bf9f0000
	s_code_end                                                 // 000000004218: bf9f0000
	s_code_end                                                 // 00000000421c: bf9f0000
	s_code_end                                                 // 000000004220: bf9f0000
	s_code_end                                                 // 000000004224: bf9f0000
	s_code_end                                                 // 000000004228: bf9f0000
	s_code_end                                                 // 00000000422c: bf9f0000
	s_code_end                                                 // 000000004230: bf9f0000
	s_code_end                                                 // 000000004234: bf9f0000
	s_code_end                                                 // 000000004238: bf9f0000
	s_code_end                                                 // 00000000423c: bf9f0000
	s_code_end                                                 // 000000004240: bf9f0000
	s_code_end                                                 // 000000004244: bf9f0000
	s_code_end                                                 // 000000004248: bf9f0000
	s_code_end                                                 // 00000000424c: bf9f0000
	s_code_end                                                 // 000000004250: bf9f0000
	s_code_end                                                 // 000000004254: bf9f0000
	s_code_end                                                 // 000000004258: bf9f0000
	s_code_end                                                 // 00000000425c: bf9f0000
	s_code_end                                                 // 000000004260: bf9f0000
	s_code_end                                                 // 000000004264: bf9f0000
	s_code_end                                                 // 000000004268: bf9f0000
	s_code_end                                                 // 00000000426c: bf9f0000
	s_code_end                                                 // 000000004270: bf9f0000
	s_code_end                                                 // 000000004274: bf9f0000
	s_code_end                                                 // 000000004278: bf9f0000
	s_code_end                                                 // 00000000427c: bf9f0000
	s_code_end                                                 // 000000004280: bf9f0000
	s_code_end                                                 // 000000004284: bf9f0000
	s_code_end                                                 // 000000004288: bf9f0000
	s_code_end                                                 // 00000000428c: bf9f0000
	s_code_end                                                 // 000000004290: bf9f0000
	s_code_end                                                 // 000000004294: bf9f0000
	s_code_end                                                 // 000000004298: bf9f0000
	s_code_end                                                 // 00000000429c: bf9f0000
	s_code_end                                                 // 0000000042a0: bf9f0000
	s_code_end                                                 // 0000000042a4: bf9f0000
	s_code_end                                                 // 0000000042a8: bf9f0000
	s_code_end                                                 // 0000000042ac: bf9f0000
	s_code_end                                                 // 0000000042b0: bf9f0000
	s_code_end                                                 // 0000000042b4: bf9f0000
	s_code_end                                                 // 0000000042b8: bf9f0000
	s_code_end                                                 // 0000000042bc: bf9f0000
	s_code_end                                                 // 0000000042c0: bf9f0000
	s_code_end                                                 // 0000000042c4: bf9f0000
	s_code_end                                                 // 0000000042c8: bf9f0000
	s_code_end                                                 // 0000000042cc: bf9f0000
	s_code_end                                                 // 0000000042d0: bf9f0000
	s_code_end                                                 // 0000000042d4: bf9f0000
	s_code_end                                                 // 0000000042d8: bf9f0000
	s_code_end                                                 // 0000000042dc: bf9f0000
	s_code_end                                                 // 0000000042e0: bf9f0000
	s_code_end                                                 // 0000000042e4: bf9f0000
	s_code_end                                                 // 0000000042e8: bf9f0000
	s_code_end                                                 // 0000000042ec: bf9f0000
	s_code_end                                                 // 0000000042f0: bf9f0000
	s_code_end                                                 // 0000000042f4: bf9f0000
	s_code_end                                                 // 0000000042f8: bf9f0000
	s_code_end                                                 // 0000000042fc: bf9f0000
	s_code_end                                                 // 000000004300: bf9f0000
	s_code_end                                                 // 000000004304: bf9f0000
	s_code_end                                                 // 000000004308: bf9f0000
	s_code_end                                                 // 00000000430c: bf9f0000
	s_code_end                                                 // 000000004310: bf9f0000
	s_code_end                                                 // 000000004314: bf9f0000
	s_code_end                                                 // 000000004318: bf9f0000
	s_code_end                                                 // 00000000431c: bf9f0000
	s_code_end                                                 // 000000004320: bf9f0000
	s_code_end                                                 // 000000004324: bf9f0000
	s_code_end                                                 // 000000004328: bf9f0000
	s_code_end                                                 // 00000000432c: bf9f0000
	s_code_end                                                 // 000000004330: bf9f0000
	s_code_end                                                 // 000000004334: bf9f0000
	s_code_end                                                 // 000000004338: bf9f0000
	s_code_end                                                 // 00000000433c: bf9f0000
	s_code_end                                                 // 000000004340: bf9f0000
	s_code_end                                                 // 000000004344: bf9f0000
	s_code_end                                                 // 000000004348: bf9f0000
	s_code_end                                                 // 00000000434c: bf9f0000
	s_code_end                                                 // 000000004350: bf9f0000
	s_code_end                                                 // 000000004354: bf9f0000
	s_code_end                                                 // 000000004358: bf9f0000
	s_code_end                                                 // 00000000435c: bf9f0000
	s_code_end                                                 // 000000004360: bf9f0000
	s_code_end                                                 // 000000004364: bf9f0000
	s_code_end                                                 // 000000004368: bf9f0000
	s_code_end                                                 // 00000000436c: bf9f0000
	s_code_end                                                 // 000000004370: bf9f0000
	s_code_end                                                 // 000000004374: bf9f0000
	s_code_end                                                 // 000000004378: bf9f0000
	s_code_end                                                 // 00000000437c: bf9f0000
	s_code_end                                                 // 000000004380: bf9f0000
	s_code_end                                                 // 000000004384: bf9f0000
	s_code_end                                                 // 000000004388: bf9f0000
	s_code_end                                                 // 00000000438c: bf9f0000
	s_code_end                                                 // 000000004390: bf9f0000
	s_code_end                                                 // 000000004394: bf9f0000
	s_code_end                                                 // 000000004398: bf9f0000
	s_code_end                                                 // 00000000439c: bf9f0000
	s_code_end                                                 // 0000000043a0: bf9f0000
	s_code_end                                                 // 0000000043a4: bf9f0000
	s_code_end                                                 // 0000000043a8: bf9f0000
	s_code_end                                                 // 0000000043ac: bf9f0000
	s_code_end                                                 // 0000000043b0: bf9f0000
	s_code_end                                                 // 0000000043b4: bf9f0000
	s_code_end                                                 // 0000000043b8: bf9f0000
	s_code_end                                                 // 0000000043bc: bf9f0000
	s_code_end                                                 // 0000000043c0: bf9f0000
	s_code_end                                                 // 0000000043c4: bf9f0000
	s_code_end                                                 // 0000000043c8: bf9f0000
	s_code_end                                                 // 0000000043cc: bf9f0000
	s_code_end                                                 // 0000000043d0: bf9f0000
	s_code_end                                                 // 0000000043d4: bf9f0000
	s_code_end                                                 // 0000000043d8: bf9f0000
	s_code_end                                                 // 0000000043dc: bf9f0000
	s_code_end                                                 // 0000000043e0: bf9f0000
	s_code_end                                                 // 0000000043e4: bf9f0000
	s_code_end                                                 // 0000000043e8: bf9f0000
	s_code_end                                                 // 0000000043ec: bf9f0000
	s_code_end                                                 // 0000000043f0: bf9f0000
	s_code_end                                                 // 0000000043f4: bf9f0000
	s_code_end                                                 // 0000000043f8: bf9f0000
	s_code_end                                                 // 0000000043fc: bf9f0000
