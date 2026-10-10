
/tmp/tmpp690swpj.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703>:
	s_clause 0x6                                               // 000000001b00: bf850006
	s_load_b128 s[12:15], s[0:1], 0xc8                         // 000000001b04: f4004300 f80000c8
	s_load_b64 s[26:27], s[0:1], 0xd8                          // 000000001b0c: f4002680 f80000d8
	s_load_b64 s[16:17], s[0:1], 0xa8                          // 000000001b14: f4002400 f80000a8
	s_load_b64 s[22:23], s[0:1], 0x8                           // 000000001b1c: f4002580 f8000008
	s_load_b64 s[24:25], s[0:1], 0x30                          // 000000001b24: f4002600 f8000030
	s_load_b64 s[18:19], s[0:1], 0x58                          // 000000001b2c: f4002480 f8000058
	s_load_b64 s[28:29], s[0:1], 0x80                          // 000000001b34: f4002700 f8000080
	s_mov_b32 s2, ttmp9                                        // 000000001b3c: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b40: 86039f75
	s_mov_b32 s4, ttmp7                                        // 000000001b44: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b48: 86059f73
	s_lshl_b64 s[34:35], s[2:3], 5                             // 000000001b4c: 84a28502
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(valu_dep_1)// 000000001b50: bf8700c9
	v_dual_mov_b32 v17, s35 :: v_dual_and_b32 v34, 15, v0      // 000000001b54: ca240023 1122008f
	s_lshl_b64 s[30:31], s[4:5], 4                             // 000000001b5c: 849e8404
	s_add_nc_u64 s[2:3], s[34:35], 32                          // 000000001b60: a982a022
	s_add_nc_u64 s[0:1], s[30:31], 16                          // 000000001b64: a980901e
	v_or_b32_e32 v16, s34, v34                                 // 000000001b68: 38204422
	v_mov_b32_e32 v19, s35                                     // 000000001b6c: 7e260223
	v_bfe_u32 v35, v0, 4, 1                                    // 000000001b70: d6100023 02050900
	s_delay_alu instid0(valu_dep_3)                            // 000000001b78: bf870003
	v_or_b32_e32 v18, 16, v16                                  // 000000001b7c: 38242090
	s_wait_kmcnt 0x0                                           // 000000001b80: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[12:13]                      // 000000001b84: d4540000 02001800
	v_cmp_gt_i64_e64 s1, s[2:3], s[14:15]                      // 000000001b8c: d4540001 02001c02
	v_cmp_lt_i64_e64 s33, s[26:27], 32                         // 000000001b94: d4510021 0201401a
	s_and_b32 s20, s26, 0xffffffe0                             // 000000001b9c: 8b14ff1a ffffffe0
	s_mov_b32 s21, s27                                         // 000000001ba4: be95001b
	s_or_b32 s0, s0, s1                                        // 000000001ba8: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bac: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bb0: 8b6a007e
	s_mov_b32 s0, -1                                           // 000000001bb4: be8000c1
	s_cbranch_vccz 2405                                        // 000000001bb8: bfa30965 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2650>
	s_and_b32 s0, s33, exec_lo                                 // 000000001bbc: 8b007e21
	s_cselect_b32 s0, 1, 0                                     // 000000001bc0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bc8: bf078100
	s_cbranch_scc1 5                                           // 000000001bcc: bfa20005 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0xe4>
	v_lshl_or_b32 v22, v35, 3, s30                             // 000000001bd0: d6560016 00790723
	v_mov_b32_e32 v23, s31                                     // 000000001bd8: 7e2e021f
	s_mov_b32 s0, 0                                            // 000000001bdc: be800080
	s_branch 1                                                 // 000000001be0: bfa00001 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0xe8>
	s_mov_b32 s0, -1                                           // 000000001be4: be8000c1
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0             // 000000001be8: ca100080 2c2c0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bf0: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001bf4: 8b007e00
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0             // 000000001bf8: ca100080 2e2e0080
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v49, 0             // 000000001c00: ca100080 30300080
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v21, 0             // 000000001c08: ca100080 34140080
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0             // 000000001c10: ca100080 24240080
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0             // 000000001c18: ca100080 26260080
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0             // 000000001c20: ca100080 28280080
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0             // 000000001c28: ca100080 2a2a0080
	s_cselect_b32 s0, 1, 0                                     // 000000001c30: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c34: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c38: bf078100
	s_cbranch_scc1 1717                                        // 000000001c3c: bfa206b5 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1c14>
	v_dual_mov_b32 v21, 0 :: v_dual_lshlrev_b32 v20, 3, v35    // 000000001c40: ca220080 15144683
	v_or_b32_e32 v0, s30, v34                                  // 000000001c48: 3800441e
	v_mov_b32_e32 v23, s31                                     // 000000001c4c: 7e2e021f
	s_mul_i32 s1, s26, s31                                     // 000000001c50: 96011f1a
	s_delay_alu instid0(valu_dep_3)                            // 000000001c54: bf870003
	v_or_b32_e32 v22, s30, v20                                 // 000000001c58: 382c281e
	v_mul_lo_u32 v3, s26, v17                                  // 000000001c5c: d72c0003 0202221a
	v_mul_lo_u32 v2, s27, v0                                   // 000000001c64: d72c0002 0202001b
	v_mad_co_u64_u32 v[24:25], null, s26, v0, v[20:21]         // 000000001c6c: d6fe7c18 0452001a
	v_mul_lo_u32 v4, s27, v16                                  // 000000001c74: d72c0004 0202201b
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[22:23]                // 000000001c7c: 7ca82c0c
	v_mov_b32_e32 v1, s31                                      // 000000001c80: 7e02021f
	v_mad_co_u64_u32 v[26:27], null, s26, v16, v[20:21]        // 000000001c84: d6fe7c1a 0452201a
	s_lshr_b64 s[4:5], s[26:27], 5                             // 000000001c8c: 8584851a
	v_mul_lo_u32 v5, s26, v19                                  // 000000001c90: d72c0005 0202261a
	v_mul_lo_u32 v6, s27, v18                                  // 000000001c98: d72c0006 0202241b
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ca0: bf88ff9e
	v_add3_u32 v25, v2, v25, s1                                // 000000001ca4: d6550019 00063302
	v_cndmask_b32_e32 v2, 0, v22, vcc_lo                       // 000000001cac: 02042c80
	v_cmp_gt_i64_e64 s0, s[12:13], v[0:1]                      // 000000001cb0: d4540000 0202000c
	v_or_b32_e32 v0, 1, v22                                    // 000000001cb8: 38002c81
	v_cndmask_b32_e64 v7, 0, s31, vcc_lo                       // 000000001cbc: d5010007 01a83e80
	v_mad_co_u64_u32 v[28:29], null, s26, v18, v[20:21]        // 000000001cc4: d6fe7c1c 0452241a
	s_lshr_b32 s5, s27, 5                                      // 000000001ccc: 8505851b
	v_add3_u32 v27, v4, v27, v3                                // 000000001cd0: d655001b 040e3704
	v_mul_lo_u32 v8, s5, v2                                    // 000000001cd8: d72c0008 02020405
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001ce0: 7ca8000c
	v_mul_lo_u32 v7, s4, v7                                    // 000000001ce4: d72c0007 02020e04
	v_mad_co_u64_u32 v[1:2], null, s4, v2, 0                   // 000000001cec: d6fe7c01 02020404
	v_or_b32_e32 v3, 2, v22                                    // 000000001cf4: 38062c82
	v_mov_b32_e32 v4, s31                                      // 000000001cf8: 7e08021f
	v_add3_u32 v29, v6, v29, v5                                // 000000001cfc: d655001d 04163b06
	s_wait_alu depctr_va_vcc(0)                                // 000000001d04: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo                        // 000000001d08: 02000080
	v_cndmask_b32_e64 v5, 0, s31, vcc_lo                       // 000000001d0c: d5010005 01a83e80
	v_cmp_gt_i64_e64 s1, s[14:15], v[16:17]                    // 000000001d14: d4540001 0202200e
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[3:4]                  // 000000001d1c: 7ca8060c
	v_add3_u32 v2, v2, v7, v8                                  // 000000001d20: d6550002 04220f02
	v_mul_lo_u32 v8, s5, v0                                    // 000000001d28: d72c0008 02020005
	v_mul_lo_u32 v9, s4, v5                                    // 000000001d30: d72c0009 02020a04
	v_mad_co_u64_u32 v[4:5], null, s4, v0, 0                   // 000000001d38: d6fe7c04 02020004
	v_cmp_gt_i64_e64 s2, s[14:15], v[18:19]                    // 000000001d40: d4540002 0202240e
	v_lshlrev_b64_e32 v[0:1], 2, v[1:2]                        // 000000001d48: 3e000282
	s_wait_alu depctr_va_vcc(0)                                // 000000001d4c: bf88ff9d
	v_cndmask_b32_e32 v10, 0, v3, vcc_lo                       // 000000001d50: 02140680
	v_or_b32_e32 v2, 3, v22                                    // 000000001d54: 38042c83
	v_mov_b32_e32 v3, s31                                      // 000000001d58: 7e06021f
	v_cndmask_b32_e64 v11, 0, s31, vcc_lo                      // 000000001d5c: d501000b 01a83e80
	v_mov_b32_e32 v49, v21                                     // 000000001d64: 7e620315
	v_add3_u32 v5, v5, v9, v8                                  // 000000001d68: d6550005 04221305
	v_mul_lo_u32 v12, s5, v10                                  // 000000001d70: d72c000c 02021405
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001d78: 7ca8040c
	v_mul_lo_u32 v11, s4, v11                                  // 000000001d7c: d72c000b 02021604
	v_mad_co_u64_u32 v[8:9], null, s4, v10, 0                  // 000000001d84: d6fe7c08 02021404
	v_add_co_u32 v50, s3, s18, v0                              // 000000001d8c: d7000332 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000001d94: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s19, v1, s3                 // 000000001d98: d5207c33 000e0213
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001da0: 3e000882
	s_wait_alu depctr_va_vcc(0)                                // 000000001da4: bf88ff9d
	v_cndmask_b32_e32 v4, 0, v2, vcc_lo                        // 000000001da8: 02080480
	v_or_b32_e32 v2, 4, v22                                    // 000000001dac: 38042c84
	v_cndmask_b32_e64 v5, 0, s31, vcc_lo                       // 000000001db0: d5010005 01a83e80
	v_add3_u32 v9, v9, v11, v12                                // 000000001db8: d6550009 04321709
	v_mov_b32_e32 v47, v21                                     // 000000001dc0: 7e5e0315
	v_mul_lo_u32 v10, s5, v4                                   // 000000001dc4: d72c000a 02020805
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001dcc: 7ca8040c
	v_mul_lo_u32 v11, s4, v5                                   // 000000001dd0: d72c000b 02020a04
	v_mad_co_u64_u32 v[4:5], null, s4, v4, 0                   // 000000001dd8: d6fe7c04 02020804
	v_add_co_u32 v53, s3, s18, v0                              // 000000001de0: d7000335 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000001de8: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s19, v1, s3                 // 000000001dec: d5207c36 000e0213
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]                        // 000000001df4: 3e001082
	s_wait_alu depctr_va_vcc(0)                                // 000000001df8: bf88ff9d
	v_cndmask_b32_e32 v8, 0, v2, vcc_lo                        // 000000001dfc: 02100480
	v_or_b32_e32 v2, 5, v22                                    // 000000001e00: 38042c85
	v_cndmask_b32_e64 v9, 0, s31, vcc_lo                       // 000000001e04: d5010009 01a83e80
	v_add3_u32 v5, v5, v11, v10                                // 000000001e0c: d6550005 042a1705
	v_or_b32_e32 v10, 6, v22                                   // 000000001e14: 38142c86
	v_mov_b32_e32 v11, s31                                     // 000000001e18: 7e16021f
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001e1c: 7ca8040c
	v_add_co_u32 v55, s3, s18, v0                              // 000000001e20: d7000337 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000001e28: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s19, v1, s3                 // 000000001e2c: d5207c38 000e0213
	v_cmp_gt_i64_e64 s3, s[12:13], v[10:11]                    // 000000001e34: d4540003 0202140c
	v_mul_lo_u32 v12, s5, v8                                   // 000000001e3c: d72c000c 02021005
	v_mul_lo_u32 v13, s4, v9                                   // 000000001e44: d72c000d 02021204
	v_mad_co_u64_u32 v[8:9], null, s4, v8, 0                   // 000000001e4c: d6fe7c08 02021004
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001e54: 3e000882
	s_wait_alu depctr_va_vcc(0)                                // 000000001e58: bf88ff9d
	v_dual_cndmask_b32 v4, 0, v2 :: v_dual_mov_b32 v45, v21    // 000000001e5c: ca500480 042c0115
	v_or_b32_e32 v2, 7, v22                                    // 000000001e64: 38042c87
	v_cndmask_b32_e64 v5, 0, s31, vcc_lo                       // 000000001e68: d5010005 01a83e80
	s_wait_alu depctr_va_sdst(0)                               // 000000001e70: bf88f19f
	v_cndmask_b32_e64 v10, 0, v10, s3                          // 000000001e74: d501000a 000e1480
	v_cndmask_b32_e64 v11, 0, s31, s3                          // 000000001e7c: d501000b 000c3e80
	v_add3_u32 v9, v9, v13, v12                                // 000000001e84: d6550009 04321b09
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001e8c: 7ca8040c
	v_mul_lo_u32 v12, s5, v4                                   // 000000001e90: d72c000c 02020805
	v_mul_lo_u32 v5, s4, v5                                    // 000000001e98: d72c0005 02020a04
	v_mad_co_u64_u32 v[3:4], null, s4, v4, 0                   // 000000001ea0: d6fe7c03 02020804
	v_mul_lo_u32 v13, s5, v10                                  // 000000001ea8: d72c000d 02021405
	v_mul_lo_u32 v14, s4, v11                                  // 000000001eb0: d72c000e 02021604
	v_mad_co_u64_u32 v[10:11], null, s4, v10, 0                // 000000001eb8: d6fe7c0a 02021404
	s_wait_alu depctr_va_vcc(0)                                // 000000001ec0: bf88ff9d
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_mov_b32 v43, v21    // 000000001ec4: ca500480 022a0115
	v_cndmask_b32_e64 v15, 0, s31, vcc_lo                      // 000000001ecc: d501000f 01a83e80
	v_add_co_u32 v57, vcc_lo, s18, v0                          // 000000001ed4: d7006a39 02020012
	s_wait_alu depctr_va_vcc(0)                                // 000000001edc: bf88ff9d
	v_add_co_ci_u32_e64 v58, null, s19, v1, vcc_lo             // 000000001ee0: d5207c3a 01aa0213
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]                        // 000000001ee8: 3e001082
	v_add3_u32 v4, v4, v5, v12                                 // 000000001eec: d6550004 04320b04
	v_mul_lo_u32 v5, s5, v2                                    // 000000001ef4: d72c0005 02020405
	v_mul_lo_u32 v12, s4, v15                                  // 000000001efc: d72c000c 02021e04
	v_mad_co_u64_u32 v[8:9], null, s4, v2, 0                   // 000000001f04: d6fe7c08 02020404
	v_add3_u32 v11, v11, v14, v13                              // 000000001f0c: d655000b 04361d0b
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]                        // 000000001f14: 3e040682
	v_add_co_u32 v59, vcc_lo, s18, v0                          // 000000001f18: d7006a3b 02020012
	s_wait_alu depctr_va_vcc(0)                                // 000000001f20: bf88ff9d
	v_add_co_ci_u32_e64 v60, null, s19, v1, vcc_lo             // 000000001f24: d5207c3c 01aa0213
	v_lshlrev_b64_e32 v[0:1], 2, v[10:11]                      // 000000001f2c: 3e001482
	v_add3_u32 v9, v9, v12, v5                                 // 000000001f30: d6550009 04161909
	v_add_co_u32 v61, vcc_lo, s18, v2                          // 000000001f38: d7006a3d 02020412
	s_wait_alu depctr_va_vcc(0)                                // 000000001f40: bf88ff9d
	v_add_co_ci_u32_e64 v62, null, s19, v3, vcc_lo             // 000000001f44: d5207c3e 01aa0613
	s_delay_alu instid0(valu_dep_3)                            // 000000001f4c: bf870003
	v_lshlrev_b64_e32 v[2:3], 2, v[8:9]                        // 000000001f50: 3e041082
	v_add_co_u32 v63, vcc_lo, s18, v0                          // 000000001f54: d7006a3f 02020012
	v_cndmask_b32_e64 v7, 0, v17, s1                           // 000000001f5c: d5010007 00062280
	v_cndmask_b32_e64 v6, 0, v16, s1                           // 000000001f64: d5010006 00062080
	s_wait_alu depctr_va_vcc(0)                                // 000000001f6c: bf88ff9d
	v_add_co_ci_u32_e64 v64, null, s19, v1, vcc_lo             // 000000001f70: d5207c40 01aa0213
	v_cndmask_b32_e64 v1, 0, v19, s2                           // 000000001f78: d5010001 000a2680
	v_cndmask_b32_e64 v0, 0, v18, s2                           // 000000001f80: d5010000 000a2480
	v_add_co_u32 v65, vcc_lo, s18, v2                          // 000000001f88: d7006a41 02020412
	v_lshlrev_b64_e32 v[30:31], 2, v[6:7]                      // 000000001f90: 3e3c0c82
	s_wait_alu depctr_va_vcc(0)                                // 000000001f94: bf88ff9d
	v_add_co_ci_u32_e64 v66, null, s19, v3, vcc_lo             // 000000001f98: d5207c42 01aa0613
	v_lshlrev_b64_e32 v[32:33], 2, v[0:1]                      // 000000001fa0: 3e400082
	v_dual_mov_b32 v52, v21 :: v_dual_mov_b32 v41, v21         // 000000001fa4: ca100115 34280115
	v_dual_mov_b32 v48, v21 :: v_dual_mov_b32 v39, v21         // 000000001fac: ca100115 30260115
	v_dual_mov_b32 v46, v21 :: v_dual_mov_b32 v37, v21         // 000000001fb4: ca100115 2e240115
	v_mov_b32_e32 v44, v21                                     // 000000001fbc: 7e580315
	v_mov_b32_e32 v42, v21                                     // 000000001fc0: 7e540315
	v_mov_b32_e32 v40, v21                                     // 000000001fc4: 7e500315
	v_mov_b32_e32 v38, v21                                     // 000000001fc8: 7e4c0315
	v_mov_b32_e32 v36, v21                                     // 000000001fcc: 7e480315
	s_mov_b64 s[36:37], 0                                      // 000000001fd0: bea40180
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000001fd4: bf8701d9
	v_dual_mov_b32 v5, s37 :: v_dual_mov_b32 v2, s37           // 000000001fd8: ca100025 05020025
	v_or_b32_e32 v4, s36, v20                                  // 000000001fe0: 38082824
	v_add_co_u32 v8, vcc_lo, v24, s36                          // 000000001fe4: d7006a08 02004918
	s_wait_alu depctr_va_vcc(0)                                // 000000001fec: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s37, v25, vcc_lo             // 000000001ff0: d5207c09 01aa3225
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[4:5]                  // 000000001ff8: 7ca8081a
	s_or_b32 s38, s36, 16                                      // 000000001ffc: 8c269024
	v_mov_b32_e32 v72, s37                                     // 000000002000: 7e900225
	s_wait_alu depctr_sa_sdst(0)                               // 000000002004: bf88ff9e
	v_or_b32_e32 v71, s38, v20                                 // 000000002008: 388e2826
	s_and_b32 s3, s0, vcc_lo                                   // 00000000200c: 8b036a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002010: bf88ff9e
	v_cndmask_b32_e64 v0, 0, v8, s3                            // 000000002014: d5010000 000e1080
	v_cndmask_b32_e64 v1, 0, v9, s3                            // 00000000201c: d5010001 000e1280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002024: bf870122
	v_add_co_u32 v0, s4, s22, v0                               // 000000002028: d7000400 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000002030: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s23, v1, s4                  // 000000002034: d5207c01 00120217
	global_load_d16_u8 v0, v[0:1], off                         // 00000000203c: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v4                                     // 000000002048: 38020881
	s_wait_loadcnt 0x0                                         // 00000000204c: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s3                            // 000000002050: d65d0000 000e0080
	v_add_co_u32 v3, s3, v8, 1                                 // 000000002058: d7000303 02010308
	s_wait_alu depctr_va_sdst(0)                               // 000000002060: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v9, s3                    // 000000002064: d5207c06 000e1280
	v_cmp_gt_i64_e64 s3, s[26:27], v[1:2]                      // 00000000206c: d4540003 0202021a
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002074: d7620000 020200ff 000000ff
	s_and_b32 s4, s0, s3                                       // 000000002080: 8b040300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002084: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v3, s4                            // 000000002088: d5010001 00120680
	v_cndmask_b32_e64 v2, 0, v6, s4                            // 000000002090: d5010002 00120c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002098: bf870122
	v_add_co_u32 v1, s5, s22, v1                               // 00000000209c: d7000501 02020216
	s_wait_alu depctr_va_sdst(0)                               // 0000000020a4: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s23, v2, s5                  // 0000000020a8: d5207c02 00160417
	global_load_d16_hi_u8 v0, v[1:2], off                      // 0000000020b0: ee08407c 00000000 00000001
	v_or_b32_e32 v1, 2, v4                                     // 0000000020bc: 38020882
	v_mov_b32_e32 v2, s37                                      // 0000000020c0: 7e040225
	s_wait_loadcnt 0x0                                         // 0000000020c4: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, s4                            // 0000000020c8: d65d5000 00120080
	v_add_co_u32 v3, s4, v8, 2                                 // 0000000020d0: d7000403 02010508
	s_wait_alu depctr_va_sdst(0)                               // 0000000020d8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v9, s4                    // 0000000020dc: d5207c06 00121280
	v_cmp_gt_i64_e64 s4, s[26:27], v[1:2]                      // 0000000020e4: d4540004 0202021a
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 0000000020ec: d7385000 02020088
	s_and_b32 s5, s0, s4                                       // 0000000020f4: 8b050400
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 0000000020f8: bf8701d1
	v_or_b16 v67.l, v0.l, v0.h op_sel:[0,1,0]                  // 0000000020fc: d7631043 02020100
	s_wait_alu depctr_sa_sdst(0)                               // 000000002104: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v3, s5                            // 000000002108: d5010001 00160680
	v_cndmask_b32_e64 v2, 0, v6, s5                            // 000000002110: d5010002 00160c80
	v_mov_b32_e32 v3, s37                                      // 000000002118: 7e060225
	v_add_co_u32 v1, s6, s22, v1                               // 00000000211c: d7000601 02020216
	s_wait_alu depctr_va_sdst(0)                               // 000000002124: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002128: bf870003
	v_add_co_ci_u32_e64 v2, null, s23, v2, s6                  // 00000000212c: d5207c02 001a0417
	global_load_d16_u8 v1, v[1:2], off                         // 000000002134: ee07807c 00000001 00000001
	v_or_b32_e32 v2, 3, v4                                     // 000000002140: 38040883
	s_wait_loadcnt 0x0                                         // 000000002144: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, s5                            // 000000002148: d65d0001 00160280
	v_add_co_u32 v6, s5, v8, 3                                 // 000000002150: d7000506 02010708
	s_wait_alu depctr_va_sdst(0)                               // 000000002158: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v9, s5                    // 00000000215c: d5207c07 00161280
	v_cmp_gt_i64_e64 s5, s[26:27], v[2:3]                      // 000000002164: d4540005 0202041a
	v_and_b16 v1.l, 0xff, v1.l                                 // 00000000216c: d7620001 020202ff 000000ff
	s_and_b32 s6, s0, s5                                       // 000000002178: 8b060500
	s_wait_alu depctr_sa_sdst(0)                               // 00000000217c: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v6, s6                            // 000000002180: d5010002 001a0c80
	v_cndmask_b32_e64 v3, 0, v7, s6                            // 000000002188: d5010003 001a0e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002190: bf870122
	v_add_co_u32 v2, s7, s22, v2                               // 000000002194: d7000702 02020416
	s_wait_alu depctr_va_sdst(0)                               // 00000000219c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s7                  // 0000000021a0: d5207c03 001e0617
	global_load_d16_hi_u8 v1, v[2:3], off                      // 0000000021a8: ee08407c 00000001 00000002
	v_or_b32_e32 v2, 4, v4                                     // 0000000021b4: 38040884
	v_mov_b32_e32 v3, s37                                      // 0000000021b8: 7e060225
	s_wait_loadcnt 0x0                                         // 0000000021bc: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, s6                            // 0000000021c0: d65d5001 001a0280
	v_add_co_u32 v6, s6, v8, 4                                 // 0000000021c8: d7000606 02010908
	s_wait_alu depctr_va_sdst(0)                               // 0000000021d0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v9, s6                    // 0000000021d4: d5207c07 001a1280
	v_cmp_gt_i64_e64 s6, s[26:27], v[2:3]                      // 0000000021dc: d4540006 0202041a
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 0000000021e4: d7385001 02020288
	s_and_b32 s7, s0, s6                                       // 0000000021ec: 8b070600
	s_delay_alu instid0(valu_dep_1)                            // 0000000021f0: bf870001
	v_or_b16 v67.h, v1.l, v1.h op_sel:[0,1,1]                  // 0000000021f4: d7635043 02020301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021fc: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v6, s7                            // 000000002200: d5010002 001e0c80
	v_cndmask_b32_e64 v3, 0, v7, s7                            // 000000002208: d5010003 001e0e80
	v_or_b32_e32 v6, 5, v4                                     // 000000002210: 380c0885
	v_mov_b32_e32 v7, s37                                      // 000000002214: 7e0e0225
	s_delay_alu instid0(valu_dep_4)                            // 000000002218: bf870004
	v_add_co_u32 v2, s8, s22, v2                               // 00000000221c: d7000802 02020416
	s_wait_alu depctr_va_sdst(0)                               // 000000002224: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s8                  // 000000002228: d5207c03 00220617
	global_load_d16_u8 v2, v[2:3], off                         // 000000002230: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 00000000223c: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s7                            // 000000002240: d65d0002 001e0480
	v_add_co_u32 v3, s7, v8, 5                                 // 000000002248: d7000703 02010b08
	s_wait_alu depctr_va_sdst(0)                               // 000000002250: bf88f19f
	v_add_co_ci_u32_e64 v10, null, 0, v9, s7                   // 000000002254: d5207c0a 001e1280
	v_cmp_gt_i64_e64 s7, s[26:27], v[6:7]                      // 00000000225c: d4540007 02020c1a
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002264: d7620002 020204ff 000000ff
	s_and_b32 s8, s0, s7                                       // 000000002270: 8b080700
	s_wait_alu depctr_sa_sdst(0)                               // 000000002274: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s8                            // 000000002278: d5010003 00220680
	v_cndmask_b32_e64 v7, 0, v10, s8                           // 000000002280: d5010007 00221480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002288: bf870122
	v_add_co_u32 v6, s9, s22, v3                               // 00000000228c: d7000906 02020616
	s_wait_alu depctr_va_sdst(0)                               // 000000002294: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s23, v7, s9                  // 000000002298: d5207c07 00260e17
	global_load_d16_hi_u8 v2, v[6:7], off                      // 0000000022a0: ee08407c 00000002 00000006
	v_or_b32_e32 v6, 6, v4                                     // 0000000022ac: 380c0886
	v_mov_b32_e32 v7, s37                                      // 0000000022b0: 7e0e0225
	v_or_b32_e32 v4, 7, v4                                     // 0000000022b4: 38080887
	s_wait_loadcnt 0x0                                         // 0000000022b8: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s8                            // 0000000022bc: d65d5002 00220480
	v_add_co_u32 v3, s8, v8, 6                                 // 0000000022c4: d7000803 02010d08
	s_wait_alu depctr_va_sdst(0)                               // 0000000022cc: bf88f19f
	v_add_co_ci_u32_e64 v10, null, 0, v9, s8                   // 0000000022d0: d5207c0a 00221280
	v_cmp_gt_i64_e64 s8, s[26:27], v[6:7]                      // 0000000022d8: d4540008 02020c1a
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 0000000022e0: d7385002 02020488
	s_and_b32 s9, s0, s8                                       // 0000000022e8: 8b090800
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_2)// 0000000022ec: bf870141
	v_or_b16 v68.l, v2.l, v2.h op_sel:[0,1,0]                  // 0000000022f0: d7631044 02020502
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022f8: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s9                            // 0000000022fc: d5010003 00260680
	v_cndmask_b32_e64 v7, 0, v10, s9                           // 000000002304: d5010007 00261480
	v_add_co_u32 v6, s10, s22, v3                              // 00000000230c: d7000a06 02020616
	s_wait_alu depctr_va_sdst(0)                               // 000000002314: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002318: bf870002
	v_add_co_ci_u32_e64 v7, null, s23, v7, s10                 // 00000000231c: d5207c07 002a0e17
	global_load_d16_u8 v3, v[6:7], off                         // 000000002324: ee07807c 00000003 00000006
	s_wait_loadcnt 0x0                                         // 000000002330: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s9                            // 000000002334: d65d0003 00260680
	v_add_co_u32 v6, s9, v8, 7                                 // 00000000233c: d7000906 02010f08
	s_wait_alu depctr_va_sdst(0)                               // 000000002344: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v9, s9                    // 000000002348: d5207c07 00261280
	v_cmp_gt_i64_e64 s9, s[26:27], v[4:5]                      // 000000002350: d4540009 0202081a
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002358: d7620003 020206ff 000000ff
	s_and_b32 s10, s0, s9                                      // 000000002364: 8b0a0900
	s_wait_alu depctr_sa_sdst(0)                               // 000000002368: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v6, s10                           // 00000000236c: d5010004 002a0c80
	v_cndmask_b32_e64 v5, 0, v7, s10                           // 000000002374: d5010005 002a0e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000237c: bf870122
	v_add_co_u32 v4, s11, s22, v4                              // 000000002380: d7000b04 02020816
	s_wait_alu depctr_va_sdst(0)                               // 000000002388: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s11                 // 00000000238c: d5207c05 002e0a17
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002394: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 0000000023a0: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s10                           // 0000000023a4: d65d5003 002a0680
	v_add_co_u32 v5, s10, v26, s36                             // 0000000023ac: d7000a05 0200491a
	s_wait_alu depctr_va_sdst(0)                               // 0000000023b4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s37, v27, s10                // 0000000023b8: d5207c06 002a3625
	s_and_b32 s10, s1, vcc_lo                                  // 0000000023c0: 8b0a6a01
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000023c4: d7385003 02020688
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023cc: bf88ff9e
	v_cndmask_b32_e64 v0, 0, v5, s10                           // 0000000023d0: d5010000 002a0a80
	v_cndmask_b32_e64 v1, 0, v6, s10                           // 0000000023d8: d5010001 002a0c80
	s_and_b32 vcc_lo, s2, vcc_lo                               // 0000000023e0: 8b6a6a02
	v_or_b16 v68.h, v3.l, v3.h op_sel:[0,1,1]                  // 0000000023e4: d7635044 02020703
	s_delay_alu instid0(valu_dep_3)                            // 0000000023ec: bf870003
	v_add_co_u32 v0, s11, s24, v0                              // 0000000023f0: d7000b00 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000023f8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s25, v1, s11                 // 0000000023fc: d5207c01 002e0219
	global_load_d16_u8 v0, v[0:1], off                         // 000000002404: ee07807c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002410: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s10                           // 000000002414: d65d0000 002a0080
	v_add_co_u32 v1, s10, v5, 1                                // 00000000241c: d7000a01 02010305
	s_wait_alu depctr_va_sdst(0)                               // 000000002424: bf88f19f
	v_add_co_ci_u32_e64 v2, null, 0, v6, s10                   // 000000002428: d5207c02 002a0c80
	s_and_b32 s10, s1, s3                                      // 000000002430: 8b0a0301
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002434: d7620000 020200ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002440: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s10                           // 000000002444: d5010001 002a0280
	v_cndmask_b32_e64 v2, 0, v2, s10                           // 00000000244c: d5010002 002a0480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002454: bf870122
	v_add_co_u32 v1, s11, s24, v1                              // 000000002458: d7000b01 02020218
	s_wait_alu depctr_va_sdst(0)                               // 000000002460: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s25, v2, s11                 // 000000002464: d5207c02 002e0419
	global_load_d16_hi_u8 v0, v[1:2], off                      // 00000000246c: ee08407c 00000000 00000001
	s_wait_loadcnt 0x0                                         // 000000002478: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, s10                           // 00000000247c: d65d5000 002a0080
	v_add_co_u32 v1, s10, v5, 2                                // 000000002484: d7000a01 02010505
	s_wait_alu depctr_va_sdst(0)                               // 00000000248c: bf88f19f
	v_add_co_ci_u32_e64 v2, null, 0, v6, s10                   // 000000002490: d5207c02 002a0c80
	s_and_b32 s10, s1, s4                                      // 000000002498: 8b0a0401
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 00000000249c: d7385000 02020088
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024a4: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s10                           // 0000000024a8: d5010001 002a0280
	v_cndmask_b32_e64 v2, 0, v2, s10                           // 0000000024b0: d5010002 002a0480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000024b8: bf870122
	v_add_co_u32 v1, s11, s24, v1                              // 0000000024bc: d7000b01 02020218
	s_wait_alu depctr_va_sdst(0)                               // 0000000024c4: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s25, v2, s11                 // 0000000024c8: d5207c02 002e0419
	global_load_d16_u8 v1, v[1:2], off                         // 0000000024d0: ee07807c 00000001 00000001
	s_wait_loadcnt 0x0                                         // 0000000024dc: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, s10                           // 0000000024e0: d65d0001 002a0280
	v_add_co_u32 v2, s10, v5, 3                                // 0000000024e8: d7000a02 02010705
	s_wait_alu depctr_va_sdst(0)                               // 0000000024f0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v6, s10                   // 0000000024f4: d5207c03 002a0c80
	s_and_b32 s10, s1, s5                                      // 0000000024fc: 8b0a0501
	v_and_b16 v1.l, 0xff, v1.l                                 // 000000002500: d7620001 020202ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000250c: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v2, s10                           // 000000002510: d5010002 002a0480
	v_cndmask_b32_e64 v3, 0, v3, s10                           // 000000002518: d5010003 002a0680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002520: bf870122
	v_add_co_u32 v2, s11, s24, v2                              // 000000002524: d7000b02 02020418
	s_wait_alu depctr_va_sdst(0)                               // 00000000252c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s25, v3, s11                 // 000000002530: d5207c03 002e0619
	global_load_d16_hi_u8 v1, v[2:3], off                      // 000000002538: ee08407c 00000001 00000002
	s_wait_loadcnt 0x0                                         // 000000002544: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, s10                           // 000000002548: d65d5001 002a0280
	v_add_co_u32 v2, s10, v5, 4                                // 000000002550: d7000a02 02010905
	s_wait_alu depctr_va_sdst(0)                               // 000000002558: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v6, s10                   // 00000000255c: d5207c03 002a0c80
	s_and_b32 s10, s1, s6                                      // 000000002564: 8b0a0601
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 000000002568: d7385001 02020288
	s_wait_alu depctr_sa_sdst(0)                               // 000000002570: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v2, s10                           // 000000002574: d5010002 002a0480
	v_cndmask_b32_e64 v3, 0, v3, s10                           // 00000000257c: d5010003 002a0680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002584: bf870122
	v_add_co_u32 v2, s11, s24, v2                              // 000000002588: d7000b02 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002590: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s25, v3, s11                 // 000000002594: d5207c03 002e0619
	global_load_d16_u8 v2, v[2:3], off                         // 00000000259c: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 0000000025a8: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s10                           // 0000000025ac: d65d0002 002a0480
	v_add_co_u32 v3, s10, v5, 5                                // 0000000025b4: d7000a03 02010b05
	s_wait_alu depctr_va_sdst(0)                               // 0000000025bc: bf88f19f
	v_add_co_ci_u32_e64 v4, null, 0, v6, s10                   // 0000000025c0: d5207c04 002a0c80
	s_and_b32 s10, s1, s7                                      // 0000000025c8: 8b0a0701
	v_and_b16 v2.l, 0xff, v2.l                                 // 0000000025cc: d7620002 020204ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025d8: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s10                           // 0000000025dc: d5010003 002a0680
	v_cndmask_b32_e64 v4, 0, v4, s10                           // 0000000025e4: d5010004 002a0880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000025ec: bf870122
	v_add_co_u32 v3, s11, s24, v3                              // 0000000025f0: d7000b03 02020618
	s_wait_alu depctr_va_sdst(0)                               // 0000000025f8: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s25, v4, s11                 // 0000000025fc: d5207c04 002e0819
	global_load_d16_hi_u8 v2, v[3:4], off                      // 000000002604: ee08407c 00000002 00000003
	s_wait_loadcnt 0x0                                         // 000000002610: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s10                           // 000000002614: d65d5002 002a0480
	v_add_co_u32 v3, s10, v5, 6                                // 00000000261c: d7000a03 02010d05
	s_wait_alu depctr_va_sdst(0)                               // 000000002624: bf88f19f
	v_add_co_ci_u32_e64 v4, null, 0, v6, s10                   // 000000002628: d5207c04 002a0c80
	s_and_b32 s10, s1, s8                                      // 000000002630: 8b0a0801
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002634: d7385002 02020488
	s_wait_alu depctr_sa_sdst(0)                               // 00000000263c: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s10                           // 000000002640: d5010003 002a0680
	v_cndmask_b32_e64 v4, 0, v4, s10                           // 000000002648: d5010004 002a0880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002650: bf870122
	v_add_co_u32 v3, s11, s24, v3                              // 000000002654: d7000b03 02020618
	s_wait_alu depctr_va_sdst(0)                               // 00000000265c: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s25, v4, s11                 // 000000002660: d5207c04 002e0819
	global_load_d16_u8 v3, v[3:4], off                         // 000000002668: ee07807c 00000003 00000003
	s_wait_loadcnt 0x0                                         // 000000002674: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s10                           // 000000002678: d65d0003 002a0680
	v_add_co_u32 v4, s10, v5, 7                                // 000000002680: d7000a04 02010f05
	s_wait_alu depctr_va_sdst(0)                               // 000000002688: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v6, s10                   // 00000000268c: d5207c05 002a0c80
	s_and_b32 s10, s1, s9                                      // 000000002694: 8b0a0901
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002698: d7620003 020206ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026a4: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s10                           // 0000000026a8: d5010004 002a0880
	v_cndmask_b32_e64 v5, 0, v5, s10                           // 0000000026b0: d5010005 002a0a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000026b8: bf870122
	v_add_co_u32 v4, s11, s24, v4                              // 0000000026bc: d7000b04 02020818
	s_wait_alu depctr_va_sdst(0)                               // 0000000026c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s25, v5, s11                 // 0000000026c8: d5207c05 002e0a19
	global_load_d16_hi_u8 v3, v[4:5], off                      // 0000000026d0: ee08407c 00000003 00000004
	v_or_b16 v4.l, v0.l, v0.h op_sel:[0,1,0]                   // 0000000026dc: d7631004 02020100
	v_or_b16 v4.h, v1.l, v1.h op_sel:[0,1,1]                   // 0000000026e4: d7635004 02020301
	v_or_b16 v5.l, v2.l, v2.h op_sel:[0,1,0]                   // 0000000026ec: d7631005 02020502
	s_wait_loadcnt 0x0                                         // 0000000026f4: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s10                           // 0000000026f8: d65d5003 002a0680
	v_add_co_u32 v8, s10, v28, s36                             // 000000002700: d7000a08 0200491c
	s_wait_alu depctr_va_sdst(0)                               // 000000002708: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s37, v29, s10                // 00000000270c: d5207c09 002a3a25
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002714: bf870113
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002718: d7385003 02020688
	v_dual_cndmask_b32 v0, 0, v8 :: v_dual_cndmask_b32 v1, 0, v9// 000000002720: ca521080 00001280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002728: bf870112
	v_or_b16 v5.h, v3.l, v3.h op_sel:[0,1,1]                   // 00000000272c: d7635005 02020703
	v_add_co_u32 v0, s10, s24, v0                              // 000000002734: d7000a00 02020018
	s_wait_alu depctr_va_sdst(0)                               // 00000000273c: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002740: bf870003
	v_add_co_ci_u32_e64 v1, null, s25, v1, s10                 // 000000002744: d5207c01 002a0219
	global_load_d16_u8 v0, v[0:1], off                         // 00000000274c: ee07807c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002758: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, vcc_lo                        // 00000000275c: d65d0000 01aa0080
	v_add_co_u32 v1, vcc_lo, v8, 1                             // 000000002764: d7006a01 02010308
	s_wait_alu depctr_va_vcc(0)                                // 00000000276c: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, 0, v9, vcc_lo                // 000000002770: d5207c02 01aa1280
	s_and_b32 vcc_lo, s2, s3                                   // 000000002778: 8b6a0302
	v_and_b16 v0.l, 0xff, v0.l                                 // 00000000277c: d7620000 020200ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002788: bf88ff9e
	v_dual_cndmask_b32 v1, 0, v1 :: v_dual_cndmask_b32 v2, 0, v2// 00000000278c: ca520280 01020480
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002794: bf870121
	v_add_co_u32 v1, s3, s24, v1                               // 000000002798: d7000301 02020218
	s_wait_alu depctr_va_sdst(0)                               // 0000000027a0: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s25, v2, s3                  // 0000000027a4: d5207c02 000e0419
	global_load_d16_hi_u8 v0, v[1:2], off                      // 0000000027ac: ee08407c 00000000 00000001
	s_wait_loadcnt 0x0                                         // 0000000027b8: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, vcc_lo                        // 0000000027bc: d65d5000 01aa0080
	v_add_co_u32 v1, vcc_lo, v8, 2                             // 0000000027c4: d7006a01 02010508
	s_wait_alu depctr_va_vcc(0)                                // 0000000027cc: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, 0, v9, vcc_lo                // 0000000027d0: d5207c02 01aa1280
	s_and_b32 vcc_lo, s2, s4                                   // 0000000027d8: 8b6a0402
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 0000000027dc: d7385000 02020088
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027e4: bf88ff9e
	v_dual_cndmask_b32 v1, 0, v1 :: v_dual_cndmask_b32 v2, 0, v2// 0000000027e8: ca520280 01020480
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000027f0: bf870112
	v_or_b16 v69.l, v0.l, v0.h op_sel:[0,1,0]                  // 0000000027f4: d7631045 02020100
	v_add_co_u32 v1, s3, s24, v1                               // 0000000027fc: d7000301 02020218
	s_wait_alu depctr_va_sdst(0)                               // 000000002804: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002808: bf870003
	v_add_co_ci_u32_e64 v2, null, s25, v2, s3                  // 00000000280c: d5207c02 000e0419
	global_load_d16_u8 v1, v[1:2], off                         // 000000002814: ee07807c 00000001 00000001
	s_wait_loadcnt 0x0                                         // 000000002820: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, vcc_lo                        // 000000002824: d65d0001 01aa0280
	v_add_co_u32 v2, vcc_lo, v8, 3                             // 00000000282c: d7006a02 02010708
	s_wait_alu depctr_va_vcc(0)                                // 000000002834: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v9, vcc_lo                // 000000002838: d5207c03 01aa1280
	s_and_b32 vcc_lo, s2, s5                                   // 000000002840: 8b6a0502
	v_and_b16 v1.l, 0xff, v1.l                                 // 000000002844: d7620001 020202ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002850: bf88ff9e
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_cndmask_b32 v3, 0, v3// 000000002854: ca520480 02020680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 00000000285c: bf870121
	v_add_co_u32 v2, s3, s24, v2                               // 000000002860: d7000302 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002868: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s25, v3, s3                  // 00000000286c: d5207c03 000e0619
	global_load_d16_hi_u8 v1, v[2:3], off                      // 000000002874: ee08407c 00000001 00000002
	s_wait_loadcnt 0x0                                         // 000000002880: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, vcc_lo                        // 000000002884: d65d5001 01aa0280
	v_add_co_u32 v2, vcc_lo, v8, 4                             // 00000000288c: d7006a02 02010908
	s_wait_alu depctr_va_vcc(0)                                // 000000002894: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v9, vcc_lo                // 000000002898: d5207c03 01aa1280
	s_and_b32 vcc_lo, s2, s6                                   // 0000000028a0: 8b6a0602
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 0000000028a4: d7385001 02020288
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028ac: bf88ff9e
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_cndmask_b32 v3, 0, v3// 0000000028b0: ca520480 02020680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000028b8: bf870112
	v_or_b16 v69.h, v1.l, v1.h op_sel:[0,1,1]                  // 0000000028bc: d7635045 02020301
	v_add_co_u32 v2, s3, s24, v2                               // 0000000028c4: d7000302 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000028cc: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000028d0: bf870003
	v_add_co_ci_u32_e64 v3, null, s25, v3, s3                  // 0000000028d4: d5207c03 000e0619
	global_load_d16_u8 v2, v[2:3], off                         // 0000000028dc: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 0000000028e8: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, vcc_lo                        // 0000000028ec: d65d0002 01aa0480
	v_add_co_u32 v3, vcc_lo, v8, 5                             // 0000000028f4: d7006a03 02010b08
	s_wait_alu depctr_va_vcc(0)                                // 0000000028fc: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, 0, v9, vcc_lo                // 000000002900: d5207c06 01aa1280
	s_and_b32 vcc_lo, s2, s7                                   // 000000002908: 8b6a0702
	v_and_b16 v2.l, 0xff, v2.l                                 // 00000000290c: d7620002 020204ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002918: bf88ff9e
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo                        // 00000000291c: 02060680
	v_cndmask_b32_e32 v7, 0, v6, vcc_lo                        // 000000002920: 020e0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002924: bf870122
	v_add_co_u32 v6, s3, s24, v3                               // 000000002928: d7000306 02020618
	s_wait_alu depctr_va_sdst(0)                               // 000000002930: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s25, v7, s3                  // 000000002934: d5207c07 000e0e19
	global_load_d16_hi_u8 v2, v[6:7], off                      // 00000000293c: ee08407c 00000002 00000006
	s_wait_loadcnt 0x0                                         // 000000002948: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, vcc_lo                        // 00000000294c: d65d5002 01aa0480
	v_add_co_u32 v3, vcc_lo, v8, 6                             // 000000002954: d7006a03 02010d08
	s_wait_alu depctr_va_vcc(0)                                // 00000000295c: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, 0, v9, vcc_lo                // 000000002960: d5207c06 01aa1280
	s_and_b32 vcc_lo, s2, s8                                   // 000000002968: 8b6a0802
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 00000000296c: d7385002 02020488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002974: bf88ff9e
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo                        // 000000002978: 02060680
	v_cndmask_b32_e32 v7, 0, v6, vcc_lo                        // 00000000297c: 020e0c80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002980: bf870193
	v_or_b16 v70.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002984: d7631046 02020502
	v_add_co_u32 v6, s3, s24, v3                               // 00000000298c: d7000306 02020618
	s_wait_alu depctr_va_sdst(0)                               // 000000002994: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002998: bf870003
	v_add_co_ci_u32_e64 v7, null, s25, v7, s3                  // 00000000299c: d5207c07 000e0e19
	global_load_d16_u8 v3, v[6:7], off                         // 0000000029a4: ee07807c 00000003 00000006
	s_wait_loadcnt 0x0                                         // 0000000029b0: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, vcc_lo                        // 0000000029b4: d65d0003 01aa0680
	v_add_co_u32 v6, vcc_lo, v8, 7                             // 0000000029bc: d7006a06 02010f08
	s_wait_alu depctr_va_vcc(0)                                // 0000000029c4: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, 0, v9, vcc_lo                // 0000000029c8: d5207c07 01aa1280
	s_and_b32 vcc_lo, s2, s9                                   // 0000000029d0: 8b6a0902
	v_and_b16 v3.l, 0xff, v3.l                                 // 0000000029d4: d7620003 020206ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029e0: bf88ff9e
	v_dual_cndmask_b32 v6, 0, v6 :: v_dual_cndmask_b32 v7, 0, v7// 0000000029e4: ca520c80 06060e80
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[67:68], v[4:5], 0   // 0000000029ec: cc464008 1a020943
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000029f4: bf8701a2
	v_add_co_u32 v6, s3, s24, v6                               // 0000000029f8: d7000306 02020c18
	s_wait_alu depctr_va_sdst(0)                               // 000000002a00: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s25, v7, s3                  // 000000002a04: d5207c07 000e0e19
	global_load_d16_hi_u8 v3, v[6:7], off                      // 000000002a0c: ee08407c 00000003 00000006
	s_wait_loadcnt 0x0                                         // 000000002a18: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, vcc_lo                        // 000000002a1c: d65d5003 01aa0680
	v_add_co_u32 v75, vcc_lo, v24, s38                         // 000000002a24: d7006a4b 02004d18
	s_wait_alu depctr_va_vcc(0)                                // 000000002a2c: bf88ff9d
	v_add_co_ci_u32_e64 v76, null, s37, v25, vcc_lo            // 000000002a30: d5207c4c 01aa3225
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a38: bf870123
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002a3c: d7385003 02020688
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[71:72]                // 000000002a44: 7ca88e1a
	v_or_b16 v70.h, v3.l, v3.h op_sel:[0,1,1]                  // 000000002a48: d7635046 02020703
	s_and_b32 s3, s0, vcc_lo                                   // 000000002a50: 8b036a00
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000002a54: bf8701d1
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[67:68], v[69:70], 0  // 000000002a58: cc464000 1a028b43
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a60: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v75, s3                          // 000000002a64: d5010043 000e9680
	v_cndmask_b32_e64 v68, 0, v76, s3                          // 000000002a6c: d5010044 000e9880
	v_mov_b32_e32 v69, s37                                     // 000000002a74: 7e8a0225
	v_add_co_u32 v67, s4, s22, v67                             // 000000002a78: d7000443 02028616
	s_wait_alu depctr_va_sdst(0)                               // 000000002a80: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002a84: bf870003
	v_add_co_ci_u32_e64 v68, null, s23, v68, s4                // 000000002a88: d5207c44 00128817
	global_load_d16_u8 v67, v[67:68], off                      // 000000002a90: ee07807c 00000043 00000043
	v_or_b32_e32 v68, 1, v71                                   // 000000002a9c: 38888e81
	s_wait_loadcnt 0x0                                         // 000000002aa0: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, s3                          // 000000002aa4: d65d0043 000e8680
	v_add_co_u32 v70, s3, v75, 1                               // 000000002aac: d7000346 0201034b
	s_wait_alu depctr_va_sdst(0)                               // 000000002ab4: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v76, s3                  // 000000002ab8: d5207c49 000e9880
	v_cmp_gt_i64_e64 s3, s[26:27], v[68:69]                    // 000000002ac0: d4540003 0202881a
	v_and_b16 v67.l, 0xff, v67.l                               // 000000002ac8: d7620043 020286ff 000000ff
	s_and_b32 s4, s0, s3                                       // 000000002ad4: 8b040300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ad8: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v70, s4                          // 000000002adc: d5010044 00128c80
	v_cndmask_b32_e64 v69, 0, v73, s4                          // 000000002ae4: d5010045 00129280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002aec: bf870122
	v_add_co_u32 v68, s5, s22, v68                             // 000000002af0: d7000544 02028816
	s_wait_alu depctr_va_sdst(0)                               // 000000002af8: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s23, v69, s5                // 000000002afc: d5207c45 00168a17
	global_load_d16_hi_u8 v67, v[68:69], off                   // 000000002b04: ee08407c 00000043 00000044
	v_or_b32_e32 v68, 2, v71                                   // 000000002b10: 38888e82
	v_mov_b32_e32 v69, s37                                     // 000000002b14: 7e8a0225
	s_wait_loadcnt 0x0                                         // 000000002b18: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, s4                          // 000000002b1c: d65d5043 00128680
	v_add_co_u32 v70, s4, v75, 2                               // 000000002b24: d7000446 0201054b
	s_wait_alu depctr_va_sdst(0)                               // 000000002b2c: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v76, s4                  // 000000002b30: d5207c49 00129880
	v_cmp_gt_i64_e64 s4, s[26:27], v[68:69]                    // 000000002b38: d4540004 0202881a
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 000000002b40: d7385043 02028688
	s_and_b32 s5, s0, s4                                       // 000000002b48: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b4c: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v70, s5                          // 000000002b50: d5010044 00168c80
	v_cndmask_b32_e64 v69, 0, v73, s5                          // 000000002b58: d5010045 00169280
	v_mov_b32_e32 v70, s37                                     // 000000002b60: 7e8c0225
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002b64: bf8701a3
	v_add_co_u32 v68, s6, s22, v68                             // 000000002b68: d7000644 02028816
	s_wait_alu depctr_va_sdst(0)                               // 000000002b70: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s23, v69, s6                // 000000002b74: d5207c45 001a8a17
	global_load_d16_u8 v68, v[68:69], off                      // 000000002b7c: ee07807c 00000044 00000044
	v_or_b32_e32 v69, 3, v71                                   // 000000002b88: 388a8e83
	s_wait_loadcnt 0x0                                         // 000000002b8c: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, s5                          // 000000002b90: d65d0044 00168880
	v_add_co_u32 v73, s5, v75, 3                               // 000000002b98: d7000549 0201074b
	s_wait_alu depctr_va_sdst(0)                               // 000000002ba0: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v76, s5                  // 000000002ba4: d5207c4a 00169880
	v_cmp_gt_i64_e64 s5, s[26:27], v[69:70]                    // 000000002bac: d4540005 02028a1a
	v_and_b16 v68.l, 0xff, v68.l                               // 000000002bb4: d7620044 020288ff 000000ff
	s_and_b32 s6, s0, s5                                       // 000000002bc0: 8b060500
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bc4: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v73, s6                          // 000000002bc8: d5010045 001a9280
	v_cndmask_b32_e64 v70, 0, v74, s6                          // 000000002bd0: d5010046 001a9480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002bd8: bf870122
	v_add_co_u32 v69, s7, s22, v69                             // 000000002bdc: d7000745 02028a16
	s_wait_alu depctr_va_sdst(0)                               // 000000002be4: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s23, v70, s7                // 000000002be8: d5207c46 001e8c17
	global_load_d16_hi_u8 v68, v[69:70], off                   // 000000002bf0: ee08407c 00000044 00000045
	v_or_b32_e32 v69, 4, v71                                   // 000000002bfc: 388a8e84
	v_mov_b32_e32 v70, s37                                     // 000000002c00: 7e8c0225
	s_wait_loadcnt 0x0                                         // 000000002c04: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, s6                          // 000000002c08: d65d5044 001a8880
	v_add_co_u32 v73, s6, v75, 4                               // 000000002c10: d7000649 0201094b
	s_wait_alu depctr_va_sdst(0)                               // 000000002c18: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v76, s6                  // 000000002c1c: d5207c4a 001a9880
	v_cmp_gt_i64_e64 s6, s[26:27], v[69:70]                    // 000000002c24: d4540006 02028a1a
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 000000002c2c: d7385044 02028888
	s_and_b32 s7, s0, s6                                       // 000000002c34: 8b070600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c38: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v73, s7                          // 000000002c3c: d5010045 001e9280
	v_cndmask_b32_e64 v70, 0, v74, s7                          // 000000002c44: d5010046 001e9480
	v_or_b32_e32 v73, 5, v71                                   // 000000002c4c: 38928e85
	v_mov_b32_e32 v74, s37                                     // 000000002c50: 7e940225
	s_delay_alu instid0(valu_dep_4)                            // 000000002c54: bf870004
	v_add_co_u32 v69, s8, s22, v69                             // 000000002c58: d7000845 02028a16
	s_wait_alu depctr_va_sdst(0)                               // 000000002c60: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s23, v70, s8                // 000000002c64: d5207c46 00228c17
	global_load_d16_u8 v69, v[69:70], off                      // 000000002c6c: ee07807c 00000045 00000045
	s_wait_loadcnt 0x0                                         // 000000002c78: bfc00000
	v_cndmask_b16 v69.l, 0, v69.l, s7                          // 000000002c7c: d65d0045 001e8a80
	v_add_co_u32 v70, s7, v75, 5                               // 000000002c84: d7000746 02010b4b
	s_wait_alu depctr_va_sdst(0)                               // 000000002c8c: bf88f19f
	v_add_co_ci_u32_e64 v77, null, 0, v76, s7                  // 000000002c90: d5207c4d 001e9880
	v_cmp_gt_i64_e64 s7, s[26:27], v[73:74]                    // 000000002c98: d4540007 0202921a
	v_and_b16 v69.l, 0xff, v69.l                               // 000000002ca0: d7620045 02028aff 000000ff
	s_and_b32 s8, s0, s7                                       // 000000002cac: 8b080700
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cb0: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v70, s8                          // 000000002cb4: d5010046 00228c80
	v_cndmask_b32_e64 v74, 0, v77, s8                          // 000000002cbc: d501004a 00229a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002cc4: bf870122
	v_add_co_u32 v73, s9, s22, v70                             // 000000002cc8: d7000949 02028c16
	s_wait_alu depctr_va_sdst(0)                               // 000000002cd0: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s23, v74, s9                // 000000002cd4: d5207c4a 00269417
	global_load_d16_hi_u8 v69, v[73:74], off                   // 000000002cdc: ee08407c 00000045 00000049
	v_or_b32_e32 v73, 6, v71                                   // 000000002ce8: 38928e86
	v_mov_b32_e32 v74, s37                                     // 000000002cec: 7e940225
	v_or_b32_e32 v71, 7, v71                                   // 000000002cf0: 388e8e87
	s_wait_loadcnt 0x0                                         // 000000002cf4: bfc00000
	v_cndmask_b16 v69.h, 0, v69.h, s8                          // 000000002cf8: d65d5045 00228a80
	v_add_co_u32 v70, s8, v75, 6                               // 000000002d00: d7000846 02010d4b
	s_wait_alu depctr_va_sdst(0)                               // 000000002d08: bf88f19f
	v_add_co_ci_u32_e64 v77, null, 0, v76, s8                  // 000000002d0c: d5207c4d 00229880
	v_cmp_gt_i64_e64 s8, s[26:27], v[73:74]                    // 000000002d14: d4540008 0202921a
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 000000002d1c: d7385045 02028a88
	s_and_b32 s9, s0, s8                                       // 000000002d24: 8b090800
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d28: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v70, s9                          // 000000002d2c: d5010046 00268c80
	v_cndmask_b32_e64 v74, 0, v77, s9                          // 000000002d34: d501004a 00269a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002d3c: bf870122
	v_add_co_u32 v73, s10, s22, v70                            // 000000002d40: d7000a49 02028c16
	s_wait_alu depctr_va_sdst(0)                               // 000000002d48: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s23, v74, s10               // 000000002d4c: d5207c4a 002a9417
	global_load_d16_u8 v70, v[73:74], off                      // 000000002d54: ee07807c 00000046 00000049
	s_wait_loadcnt 0x0                                         // 000000002d60: bfc00000
	v_cndmask_b16 v70.l, 0, v70.l, s9                          // 000000002d64: d65d0046 00268c80
	v_add_co_u32 v73, s9, v75, 7                               // 000000002d6c: d7000949 02010f4b
	s_wait_alu depctr_va_sdst(0)                               // 000000002d74: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v76, s9                  // 000000002d78: d5207c4a 00269880
	v_cmp_gt_i64_e64 s9, s[26:27], v[71:72]                    // 000000002d80: d4540009 02028e1a
	v_and_b16 v70.l, 0xff, v70.l                               // 000000002d88: d7620046 02028cff 000000ff
	s_and_b32 s10, s0, s9                                      // 000000002d94: 8b0a0900
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d98: bf88ff9e
	v_cndmask_b32_e64 v71, 0, v73, s10                         // 000000002d9c: d5010047 002a9280
	v_cndmask_b32_e64 v72, 0, v74, s10                         // 000000002da4: d5010048 002a9480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002dac: bf870122
	v_add_co_u32 v71, s11, s22, v71                            // 000000002db0: d7000b47 02028e16
	s_wait_alu depctr_va_sdst(0)                               // 000000002db8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s23, v72, s11               // 000000002dbc: d5207c48 002e9017
	global_load_d16_hi_u8 v70, v[71:72], off                   // 000000002dc4: ee08407c 00000046 00000047
	v_or_b16 v71.l, v67.l, v67.h op_sel:[0,1,0]                // 000000002dd0: d7631047 02028743
	v_or_b16 v71.h, v68.l, v68.h op_sel:[0,1,1]                // 000000002dd8: d7635047 02028944
	v_or_b16 v72.l, v69.l, v69.h op_sel:[0,1,0]                // 000000002de0: d7631048 02028b45
	s_wait_loadcnt 0x0                                         // 000000002de8: bfc00000
	v_cndmask_b16 v70.h, 0, v70.h, s10                         // 000000002dec: d65d5046 002a8c80
	v_add_co_u32 v75, s10, s38, v26                            // 000000002df4: d7000a4b 02023426
	s_wait_alu depctr_va_sdst(0)                               // 000000002dfc: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s37, v27, s10               // 000000002e00: d5207c4c 002a3625
	s_and_b32 s10, s1, vcc_lo                                  // 000000002e08: 8b0a6a01
	v_lshlrev_b16 v70.h, 8, v70.h op_sel:[0,1,1]               // 000000002e0c: d7385046 02028c88
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e14: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v75, s10                         // 000000002e18: d5010043 002a9680
	v_cndmask_b32_e64 v68, 0, v76, s10                         // 000000002e20: d5010044 002a9880
	s_and_b32 vcc_lo, s2, vcc_lo                               // 000000002e28: 8b6a6a02
	v_or_b16 v72.h, v70.l, v70.h op_sel:[0,1,1]                // 000000002e2c: d7635048 02028d46
	s_delay_alu instid0(valu_dep_3)                            // 000000002e34: bf870003
	v_add_co_u32 v67, s11, s24, v67                            // 000000002e38: d7000b43 02028618
	s_wait_alu depctr_va_sdst(0)                               // 000000002e40: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s25, v68, s11               // 000000002e44: d5207c44 002e8819
	global_load_d16_u8 v67, v[67:68], off                      // 000000002e4c: ee07807c 00000043 00000043
	s_wait_loadcnt 0x0                                         // 000000002e58: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, s10                         // 000000002e5c: d65d0043 002a8680
	v_add_co_u32 v68, s10, v75, 1                              // 000000002e64: d7000a44 0201034b
	s_wait_alu depctr_va_sdst(0)                               // 000000002e6c: bf88f19f
	v_add_co_ci_u32_e64 v69, null, 0, v76, s10                 // 000000002e70: d5207c45 002a9880
	s_and_b32 s10, s1, s3                                      // 000000002e78: 8b0a0301
	v_and_b16 v67.l, 0xff, v67.l                               // 000000002e7c: d7620043 020286ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e88: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v68, s10                         // 000000002e8c: d5010044 002a8880
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002e94: d5010045 002a8a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002e9c: bf870122
	v_add_co_u32 v68, s11, s24, v68                            // 000000002ea0: d7000b44 02028818
	s_wait_alu depctr_va_sdst(0)                               // 000000002ea8: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s25, v69, s11               // 000000002eac: d5207c45 002e8a19
	global_load_d16_hi_u8 v67, v[68:69], off                   // 000000002eb4: ee08407c 00000043 00000044
	s_wait_loadcnt 0x0                                         // 000000002ec0: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, s10                         // 000000002ec4: d65d5043 002a8680
	v_add_co_u32 v68, s10, v75, 2                              // 000000002ecc: d7000a44 0201054b
	s_wait_alu depctr_va_sdst(0)                               // 000000002ed4: bf88f19f
	v_add_co_ci_u32_e64 v69, null, 0, v76, s10                 // 000000002ed8: d5207c45 002a9880
	s_and_b32 s10, s1, s4                                      // 000000002ee0: 8b0a0401
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 000000002ee4: d7385043 02028688
	s_wait_alu depctr_sa_sdst(0)                               // 000000002eec: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v68, s10                         // 000000002ef0: d5010044 002a8880
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002ef8: d5010045 002a8a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f00: bf870122
	v_add_co_u32 v68, s11, s24, v68                            // 000000002f04: d7000b44 02028818
	s_wait_alu depctr_va_sdst(0)                               // 000000002f0c: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s25, v69, s11               // 000000002f10: d5207c45 002e8a19
	global_load_d16_u8 v68, v[68:69], off                      // 000000002f18: ee07807c 00000044 00000044
	s_wait_loadcnt 0x0                                         // 000000002f24: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, s10                         // 000000002f28: d65d0044 002a8880
	v_add_co_u32 v69, s10, v75, 3                              // 000000002f30: d7000a45 0201074b
	s_wait_alu depctr_va_sdst(0)                               // 000000002f38: bf88f19f
	v_add_co_ci_u32_e64 v70, null, 0, v76, s10                 // 000000002f3c: d5207c46 002a9880
	s_and_b32 s10, s1, s5                                      // 000000002f44: 8b0a0501
	v_and_b16 v68.l, 0xff, v68.l                               // 000000002f48: d7620044 020288ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f54: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002f58: d5010045 002a8a80
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 000000002f60: d5010046 002a8c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f68: bf870122
	v_add_co_u32 v69, s11, s24, v69                            // 000000002f6c: d7000b45 02028a18
	s_wait_alu depctr_va_sdst(0)                               // 000000002f74: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s25, v70, s11               // 000000002f78: d5207c46 002e8c19
	global_load_d16_hi_u8 v68, v[69:70], off                   // 000000002f80: ee08407c 00000044 00000045
	s_wait_loadcnt 0x0                                         // 000000002f8c: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, s10                         // 000000002f90: d65d5044 002a8880
	v_add_co_u32 v69, s10, v75, 4                              // 000000002f98: d7000a45 0201094b
	s_wait_alu depctr_va_sdst(0)                               // 000000002fa0: bf88f19f
	v_add_co_ci_u32_e64 v70, null, 0, v76, s10                 // 000000002fa4: d5207c46 002a9880
	s_and_b32 s10, s1, s6                                      // 000000002fac: 8b0a0601
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 000000002fb0: d7385044 02028888
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fb8: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002fbc: d5010045 002a8a80
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 000000002fc4: d5010046 002a8c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002fcc: bf870122
	v_add_co_u32 v69, s11, s24, v69                            // 000000002fd0: d7000b45 02028a18
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd8: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s25, v70, s11               // 000000002fdc: d5207c46 002e8c19
	global_load_d16_u8 v69, v[69:70], off                      // 000000002fe4: ee07807c 00000045 00000045
	s_wait_loadcnt 0x0                                         // 000000002ff0: bfc00000
	v_cndmask_b16 v69.l, 0, v69.l, s10                         // 000000002ff4: d65d0045 002a8a80
	v_add_co_u32 v70, s10, v75, 5                              // 000000002ffc: d7000a46 02010b4b
	s_wait_alu depctr_va_sdst(0)                               // 000000003004: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v76, s10                 // 000000003008: d5207c49 002a9880
	s_and_b32 s10, s1, s7                                      // 000000003010: 8b0a0701
	v_and_b16 v69.l, 0xff, v69.l                               // 000000003014: d7620045 02028aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003020: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 000000003024: d5010046 002a8c80
	v_cndmask_b32_e64 v74, 0, v73, s10                         // 00000000302c: d501004a 002a9280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003034: bf870122
	v_add_co_u32 v73, s11, s24, v70                            // 000000003038: d7000b49 02028c18
	s_wait_alu depctr_va_sdst(0)                               // 000000003040: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s25, v74, s11               // 000000003044: d5207c4a 002e9419
	global_load_d16_hi_u8 v69, v[73:74], off                   // 00000000304c: ee08407c 00000045 00000049
	s_wait_loadcnt 0x0                                         // 000000003058: bfc00000
	v_cndmask_b16 v69.h, 0, v69.h, s10                         // 00000000305c: d65d5045 002a8a80
	v_add_co_u32 v70, s10, v75, 6                              // 000000003064: d7000a46 02010d4b
	s_wait_alu depctr_va_sdst(0)                               // 00000000306c: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v76, s10                 // 000000003070: d5207c49 002a9880
	s_and_b32 s10, s1, s8                                      // 000000003078: 8b0a0801
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 00000000307c: d7385045 02028a88
	s_wait_alu depctr_sa_sdst(0)                               // 000000003084: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 000000003088: d5010046 002a8c80
	v_cndmask_b32_e64 v74, 0, v73, s10                         // 000000003090: d501004a 002a9280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003098: bf870122
	v_add_co_u32 v73, s11, s24, v70                            // 00000000309c: d7000b49 02028c18
	s_wait_alu depctr_va_sdst(0)                               // 0000000030a4: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s25, v74, s11               // 0000000030a8: d5207c4a 002e9419
	global_load_d16_u8 v70, v[73:74], off                      // 0000000030b0: ee07807c 00000046 00000049
	s_wait_loadcnt 0x0                                         // 0000000030bc: bfc00000
	v_cndmask_b16 v70.l, 0, v70.l, s10                         // 0000000030c0: d65d0046 002a8c80
	v_add_co_u32 v73, s10, v75, 7                              // 0000000030c8: d7000a49 02010f4b
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d0: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v76, s10                 // 0000000030d4: d5207c4a 002a9880
	s_and_b32 s10, s1, s9                                      // 0000000030dc: 8b0a0901
	v_and_b16 v70.l, 0xff, v70.l                               // 0000000030e0: d7620046 02028cff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030ec: bf88ff9e
	v_cndmask_b32_e64 v73, 0, v73, s10                         // 0000000030f0: d5010049 002a9280
	v_cndmask_b32_e64 v74, 0, v74, s10                         // 0000000030f8: d501004a 002a9480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003100: bf870122
	v_add_co_u32 v73, s11, s24, v73                            // 000000003104: d7000b49 02029218
	s_wait_alu depctr_va_sdst(0)                               // 00000000310c: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s25, v74, s11               // 000000003110: d5207c4a 002e9419
	global_load_d16_hi_u8 v70, v[73:74], off                   // 000000003118: ee08407c 00000046 00000049
	v_or_b16 v73.l, v67.l, v67.h op_sel:[0,1,0]                // 000000003124: d7631049 02028743
	v_or_b16 v73.h, v68.l, v68.h op_sel:[0,1,1]                // 00000000312c: d7635049 02028944
	v_or_b16 v74.l, v69.l, v69.h op_sel:[0,1,0]                // 000000003134: d763104a 02028b45
	s_wait_loadcnt 0x0                                         // 00000000313c: bfc00000
	v_cndmask_b16 v70.h, 0, v70.h, s10                         // 000000003140: d65d5046 002a8c80
	v_add_co_u32 v77, s10, s38, v28                            // 000000003148: d7000a4d 02023826
	s_wait_alu depctr_va_sdst(0)                               // 000000003150: bf88f19f
	v_add_co_ci_u32_e64 v78, null, s37, v29, s10               // 000000003154: d5207c4e 002a3a25
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 00000000315c: bf870113
	v_lshlrev_b16 v70.h, 8, v70.h op_sel:[0,1,1]               // 000000003160: d7385046 02028c88
	v_dual_cndmask_b32 v67, 0, v77 :: v_dual_cndmask_b32 v68, 0, v78// 000000003168: ca529a80 43449c80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003170: bf870112
	v_or_b16 v74.h, v70.l, v70.h op_sel:[0,1,1]                // 000000003174: d763504a 02028d46
	v_add_co_u32 v67, s10, s24, v67                            // 00000000317c: d7000a43 02028618
	s_wait_alu depctr_va_sdst(0)                               // 000000003184: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003188: bf870193
	v_add_co_ci_u32_e64 v68, null, s25, v68, s10               // 00000000318c: d5207c44 002a8819
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[71:72], v[73:74], v[8:15]// 000000003194: cc464008 1c229347
	global_load_d16_u8 v67, v[67:68], off                      // 00000000319c: ee07807c 00000043 00000043
	s_wait_loadcnt 0x0                                         // 0000000031a8: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, vcc_lo                      // 0000000031ac: d65d0043 01aa8680
	v_add_co_u32 v68, vcc_lo, v77, 1                           // 0000000031b4: d7006a44 0201034d
	s_wait_alu depctr_va_vcc(0)                                // 0000000031bc: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, 0, v78, vcc_lo              // 0000000031c0: d5207c45 01aa9c80
	s_and_b32 vcc_lo, s2, s3                                   // 0000000031c8: 8b6a0302
	v_and_b16 v67.l, 0xff, v67.l                               // 0000000031cc: d7620043 020286ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031d8: bf88ff9e
	v_dual_cndmask_b32 v68, 0, v68 :: v_dual_cndmask_b32 v69, 0, v69// 0000000031dc: ca528880 44448a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000031e4: bf870121
	v_add_co_u32 v68, s3, s24, v68                             // 0000000031e8: d7000344 02028818
	s_wait_alu depctr_va_sdst(0)                               // 0000000031f0: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s25, v69, s3                // 0000000031f4: d5207c45 000e8a19
	global_load_d16_hi_u8 v67, v[68:69], off                   // 0000000031fc: ee08407c 00000043 00000044
	s_wait_loadcnt 0x0                                         // 000000003208: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, vcc_lo                      // 00000000320c: d65d5043 01aa8680
	v_add_co_u32 v68, vcc_lo, v77, 2                           // 000000003214: d7006a44 0201054d
	s_wait_alu depctr_va_vcc(0)                                // 00000000321c: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, 0, v78, vcc_lo              // 000000003220: d5207c45 01aa9c80
	s_and_b32 vcc_lo, s2, s4                                   // 000000003228: 8b6a0402
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 00000000322c: d7385043 02028688
	s_wait_alu depctr_sa_sdst(0)                               // 000000003234: bf88ff9e
	v_dual_cndmask_b32 v68, 0, v68 :: v_dual_cndmask_b32 v69, 0, v69// 000000003238: ca528880 44448a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003240: bf870121
	v_add_co_u32 v68, s3, s24, v68                             // 000000003244: d7000344 02028818
	s_wait_alu depctr_va_sdst(0)                               // 00000000324c: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s25, v69, s3                // 000000003250: d5207c45 000e8a19
	global_load_d16_u8 v68, v[68:69], off                      // 000000003258: ee07807c 00000044 00000044
	s_wait_loadcnt 0x0                                         // 000000003264: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, vcc_lo                      // 000000003268: d65d0044 01aa8880
	v_add_co_u32 v69, vcc_lo, v77, 3                           // 000000003270: d7006a45 0201074d
	s_wait_alu depctr_va_vcc(0)                                // 000000003278: bf88ff9d
	v_add_co_ci_u32_e64 v70, null, 0, v78, vcc_lo              // 00000000327c: d5207c46 01aa9c80
	s_and_b32 vcc_lo, s2, s5                                   // 000000003284: 8b6a0502
	s_lshr_b64 s[4:5], s[36:37], 5                             // 000000003288: 85848524
	s_wait_alu depctr_sa_sdst(0)                               // 00000000328c: bf88ff9e
	v_dual_cndmask_b32 v69, 0, v69 :: v_dual_cndmask_b32 v70, 0, v70// 000000003290: ca528a80 45468c80
	v_and_b16 v68.l, 0xff, v68.l                               // 000000003298: d7620044 020288ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000032a4: bf8701a2
	v_add_co_u32 v69, s3, s24, v69                             // 0000000032a8: d7000345 02028a18
	s_wait_alu depctr_va_sdst(0)                               // 0000000032b0: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s25, v70, s3                // 0000000032b4: d5207c46 000e8c19
	global_load_d16_hi_u8 v68, v[69:70], off                   // 0000000032bc: ee08407c 00000044 00000045
	s_wait_loadcnt 0x0                                         // 0000000032c8: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, vcc_lo                      // 0000000032cc: d65d5044 01aa8880
	v_add_co_u32 v69, vcc_lo, v77, 4                           // 0000000032d4: d7006a45 0201094d
	s_wait_alu depctr_va_vcc(0)                                // 0000000032dc: bf88ff9d
	v_add_co_ci_u32_e64 v70, null, 0, v78, vcc_lo              // 0000000032e0: d5207c46 01aa9c80
	s_and_b32 vcc_lo, s2, s6                                   // 0000000032e8: 8b6a0602
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 0000000032ec: d7385044 02028888
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032f4: bf88ff9e
	v_dual_cndmask_b32 v69, 0, v69 :: v_dual_cndmask_b32 v70, 0, v70// 0000000032f8: ca528a80 45468c80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003300: bf870121
	v_add_co_u32 v69, s3, s24, v69                             // 000000003304: d7000345 02028a18
	s_wait_alu depctr_va_sdst(0)                               // 00000000330c: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s25, v70, s3                // 000000003310: d5207c46 000e8c19
	global_load_d16_u8 v69, v[69:70], off                      // 000000003318: ee07807c 00000045 00000045
	s_wait_loadcnt 0x0                                         // 000000003324: bfc00000
	v_cndmask_b16 v69.l, 0, v69.l, vcc_lo                      // 000000003328: d65d0045 01aa8a80
	v_add_co_u32 v70, vcc_lo, v77, 5                           // 000000003330: d7006a46 02010b4d
	s_wait_alu depctr_va_vcc(0)                                // 000000003338: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, 0, v78, vcc_lo              // 00000000333c: d5207c4b 01aa9c80
	s_and_b32 vcc_lo, s2, s7                                   // 000000003344: 8b6a0702
	v_and_b16 v69.l, 0xff, v69.l                               // 000000003348: d7620045 02028aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003354: bf88ff9e
	v_cndmask_b32_e32 v70, 0, v70, vcc_lo                      // 000000003358: 028c8c80
	v_cndmask_b32_e32 v76, 0, v75, vcc_lo                      // 00000000335c: 02989680
	s_mul_u64 s[6:7], s[4:5], s[14:15]                         // 000000003360: aa860e04
	s_lshr_b64 s[4:5], s[36:37], 3                             // 000000003364: 85848324
	s_wait_alu depctr_sa_sdst(0)                               // 000000003368: bf88ff9e
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000336c: 84868206
	v_add_co_u32 v75, s3, s24, v70                             // 000000003370: d700034b 02028c18
	s_wait_alu depctr_va_sdst(0)                               // 000000003378: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s25, v76, s3                // 00000000337c: d5207c4c 000e9819
	s_wait_alu depctr_sa_sdst(0)                               // 000000003384: bf88ff9e
	s_add_nc_u64 s[6:7], s[28:29], s[6:7]                      // 000000003388: a986061c
	s_add_nc_u64 s[36:37], s[36:37], 32                        // 00000000338c: a9a4a024
	global_load_d16_hi_u8 v69, v[75:76], off                   // 000000003390: ee08407c 00000045 0000004b
	s_wait_loadcnt 0x0                                         // 00000000339c: bfc00000
	v_cndmask_b16 v69.h, 0, v69.h, vcc_lo                      // 0000000033a0: d65d5045 01aa8a80
	v_add_co_u32 v70, vcc_lo, v77, 6                           // 0000000033a8: d7006a46 02010d4d
	s_wait_alu depctr_va_vcc(0)                                // 0000000033b0: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, 0, v78, vcc_lo              // 0000000033b4: d5207c4b 01aa9c80
	s_and_b32 vcc_lo, s2, s8                                   // 0000000033bc: 8b6a0802
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 0000000033c0: d7385045 02028a88
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033c8: bf88ff9e
	v_cndmask_b32_e32 v70, 0, v70, vcc_lo                      // 0000000033cc: 028c8c80
	v_cndmask_b32_e32 v76, 0, v75, vcc_lo                      // 0000000033d0: 02989680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000033d4: bf870122
	v_add_co_u32 v75, s3, s24, v70                             // 0000000033d8: d700034b 02028c18
	s_wait_alu depctr_va_sdst(0)                               // 0000000033e0: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s25, v76, s3                // 0000000033e4: d5207c4c 000e9819
	global_load_d16_u8 v70, v[75:76], off                      // 0000000033ec: ee07807c 00000046 0000004b
	s_wait_loadcnt 0x0                                         // 0000000033f8: bfc00000
	v_cndmask_b16 v70.l, 0, v70.l, vcc_lo                      // 0000000033fc: d65d0046 01aa8c80
	v_add_co_u32 v75, vcc_lo, v77, 7                           // 000000003404: d7006a4b 02010f4d
	s_wait_alu depctr_va_vcc(0)                                // 00000000340c: bf88ff9d
	v_add_co_ci_u32_e64 v76, null, 0, v78, vcc_lo              // 000000003410: d5207c4c 01aa9c80
	s_and_b32 vcc_lo, s2, s9                                   // 000000003418: 8b6a0902
	v_and_b16 v70.l, 0xff, v70.l                               // 00000000341c: d7620046 02028cff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003428: bf88ff9e
	v_dual_cndmask_b32 v75, 0, v75 :: v_dual_cndmask_b32 v76, 0, v76// 00000000342c: ca529680 4b4c9880
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003434: bf870121
	v_add_co_u32 v75, s3, s24, v75                             // 000000003438: d700034b 02029618
	s_wait_alu depctr_va_sdst(0)                               // 000000003440: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s25, v76, s3                // 000000003444: d5207c4c 000e9819
	v_cmp_lt_i64_e64 s3, s[36:37], s[20:21]                    // 00000000344c: d4510003 02002824
	global_load_d16_hi_u8 v70, v[75:76], off                   // 000000003454: ee08407c 00000046 0000004b
	s_wait_loadcnt 0x0                                         // 000000003460: bfc00000
	v_cndmask_b16 v70.h, 0, v70.h, vcc_lo                      // 000000003464: d65d5046 01aa8c80
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000346c: bf870091
	v_lshlrev_b16 v70.h, 8, v70.h op_sel:[0,1,1]               // 000000003470: d7385046 02028c88
	v_or_b16 v70.h, v70.l, v70.h op_sel:[0,1,1]                // 000000003478: d7635046 02028d46
	v_or_b16 v70.l, v69.l, v69.h op_sel:[0,1,0]                // 000000003480: d7631046 02028b45
	v_or_b16 v69.l, v67.l, v67.h op_sel:[0,1,0]                // 000000003488: d7631045 02028743
	v_add_co_u32 v67, vcc_lo, v50, s4                          // 000000003490: d7006a43 02000932
	v_or_b16 v69.h, v68.l, v68.h op_sel:[0,1,1]                // 000000003498: d7635045 02028944
	s_wait_alu depctr_va_vcc(0)                                // 0000000034a0: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s5, v51, vcc_lo             // 0000000034a4: d5207c44 01aa6605
	s_delay_alu instid0(valu_dep_2)                            // 0000000034ac: bf870002
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[71:72], v[69:70], v[0:7]// 0000000034b0: cc464000 1c028b47
	global_load_b32 v69, v[67:68], off                         // 0000000034b8: ee05007c 00000045 00000043
	v_add_co_u32 v67, vcc_lo, s6, v30                          // 0000000034c4: d7006a43 02023c06
	s_wait_alu depctr_va_vcc(0)                                // 0000000034cc: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s7, v31, vcc_lo             // 0000000034d0: d5207c44 01aa3e07
	global_load_b32 v70, v[67:68], off                         // 0000000034d8: ee05007c 00000046 00000043
	s_wait_loadcnt 0x0                                         // 0000000034e4: bfc00000
	v_mul_f32_e32 v67, v69, v70                                // 0000000034e8: 10868d45
	s_delay_alu instid0(valu_dep_1)                            // 0000000034ec: bf870001
	v_mul_f32_e32 v8, v8, v67                                  // 0000000034f0: 10108708
	v_add_co_u32 v67, vcc_lo, v53, s4                          // 0000000034f4: d7006a43 02000935
	s_wait_alu depctr_va_vcc(0)                                // 0000000034fc: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s5, v54, vcc_lo             // 000000003500: d5207c44 01aa6c05
	global_load_b32 v67, v[67:68], off                         // 000000003508: ee05007c 00000043 00000043
	s_wait_loadcnt 0x0                                         // 000000003514: bfc00000
	v_dual_add_f32 v21, v21, v8 :: v_dual_mul_f32 v8, v70, v67 // 000000003518: c9061115 15088746
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003520: bf870091
	v_mul_f32_e32 v8, v9, v8                                   // 000000003524: 10101109
	v_add_f32_e32 v52, v52, v8                                 // 000000003528: 06681134
	v_add_co_u32 v8, vcc_lo, v55, s4                           // 00000000352c: d7006a08 02000937
	s_wait_alu depctr_va_vcc(0)                                // 000000003534: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v56, vcc_lo              // 000000003538: d5207c09 01aa7005
	global_load_b32 v68, v[8:9], off                           // 000000003540: ee05007c 00000044 00000008
	s_wait_loadcnt 0x0                                         // 00000000354c: bfc00000
	v_mul_f32_e32 v8, v70, v68                                 // 000000003550: 10108946
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003554: bf870091
	v_mul_f32_e32 v8, v10, v8                                  // 000000003558: 1010110a
	v_add_f32_e32 v49, v49, v8                                 // 00000000355c: 06621131
	v_add_co_u32 v8, vcc_lo, v57, s4                           // 000000003560: d7006a08 02000939
	s_wait_alu depctr_va_vcc(0)                                // 000000003568: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v58, vcc_lo              // 00000000356c: d5207c09 01aa7405
	global_load_b32 v10, v[8:9], off                           // 000000003574: ee05007c 0000000a 00000008
	s_wait_loadcnt 0x0                                         // 000000003580: bfc00000
	v_mul_f32_e32 v8, v70, v10                                 // 000000003584: 10101546
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003588: bf870091
	v_mul_f32_e32 v8, v11, v8                                  // 00000000358c: 1010110b
	v_add_f32_e32 v48, v48, v8                                 // 000000003590: 06601130
	v_add_co_u32 v8, vcc_lo, v59, s4                           // 000000003594: d7006a08 0200093b
	s_wait_alu depctr_va_vcc(0)                                // 00000000359c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v60, vcc_lo              // 0000000035a0: d5207c09 01aa7805
	global_load_b32 v11, v[8:9], off                           // 0000000035a8: ee05007c 0000000b 00000008
	s_wait_loadcnt 0x0                                         // 0000000035b4: bfc00000
	v_mul_f32_e32 v8, v70, v11                                 // 0000000035b8: 10101746
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000035bc: bf870091
	v_mul_f32_e32 v8, v12, v8                                  // 0000000035c0: 1010110c
	v_add_f32_e32 v47, v47, v8                                 // 0000000035c4: 065e112f
	v_add_co_u32 v8, vcc_lo, v61, s4                           // 0000000035c8: d7006a08 0200093d
	s_wait_alu depctr_va_vcc(0)                                // 0000000035d0: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v62, vcc_lo              // 0000000035d4: d5207c09 01aa7c05
	global_load_b32 v12, v[8:9], off                           // 0000000035dc: ee05007c 0000000c 00000008
	s_wait_loadcnt 0x0                                         // 0000000035e8: bfc00000
	v_mul_f32_e32 v8, v70, v12                                 // 0000000035ec: 10101946
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000035f0: bf870091
	v_mul_f32_e32 v8, v13, v8                                  // 0000000035f4: 1010110d
	v_add_f32_e32 v46, v46, v8                                 // 0000000035f8: 065c112e
	v_add_co_u32 v8, vcc_lo, v63, s4                           // 0000000035fc: d7006a08 0200093f
	s_wait_alu depctr_va_vcc(0)                                // 000000003604: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v64, vcc_lo              // 000000003608: d5207c09 01aa8005
	global_load_b32 v13, v[8:9], off                           // 000000003610: ee05007c 0000000d 00000008
	s_wait_loadcnt 0x0                                         // 00000000361c: bfc00000
	v_mul_f32_e32 v8, v70, v13                                 // 000000003620: 10101b46
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003624: bf870091
	v_mul_f32_e32 v8, v14, v8                                  // 000000003628: 1010110e
	v_add_f32_e32 v45, v45, v8                                 // 00000000362c: 065a112d
	v_add_co_u32 v8, vcc_lo, v65, s4                           // 000000003630: d7006a08 02000941
	s_wait_alu depctr_va_vcc(0)                                // 000000003638: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v66, vcc_lo              // 00000000363c: d5207c09 01aa8405
	global_load_b32 v14, v[8:9], off                           // 000000003644: ee05007c 0000000e 00000008
	s_wait_loadcnt 0x0                                         // 000000003650: bfc00000
	v_mul_f32_e32 v8, v70, v14                                 // 000000003654: 10101d46
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003658: bf870091
	v_mul_f32_e32 v8, v15, v8                                  // 00000000365c: 1010110f
	v_add_f32_e32 v44, v44, v8                                 // 000000003660: 0658112c
	v_add_co_u32 v8, vcc_lo, s6, v32                           // 000000003664: d7006a08 02024006
	s_wait_alu depctr_va_vcc(0)                                // 00000000366c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s7, v33, vcc_lo              // 000000003670: d5207c09 01aa4207
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000003678: 8b6a037e
	global_load_b32 v8, v[8:9], off                            // 00000000367c: ee05007c 00000008 00000008
	s_wait_loadcnt 0x0                                         // 000000003688: bfc00000
	v_mul_f32_e32 v9, v69, v8                                  // 00000000368c: 10121145
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003690: bf870091
	v_mul_f32_e32 v0, v0, v9                                   // 000000003694: 10001300
	v_add_f32_e32 v43, v43, v0                                 // 000000003698: 0656012b
	v_mul_f32_e32 v0, v67, v8                                  // 00000000369c: 10001143
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000036a0: bf870091
	v_mul_f32_e32 v0, v1, v0                                   // 0000000036a4: 10000101
	v_add_f32_e32 v42, v42, v0                                 // 0000000036a8: 0654012a
	v_mul_f32_e32 v0, v68, v8                                  // 0000000036ac: 10001144
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000036b0: bf870091
	v_mul_f32_e32 v0, v2, v0                                   // 0000000036b4: 10000102
	v_add_f32_e32 v41, v41, v0                                 // 0000000036b8: 06520129
	v_mul_f32_e32 v0, v10, v8                                  // 0000000036bc: 1000110a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000036c0: bf870091
	v_mul_f32_e32 v0, v3, v0                                   // 0000000036c4: 10000103
	v_add_f32_e32 v40, v40, v0                                 // 0000000036c8: 06500128
	v_mul_f32_e32 v0, v11, v8                                  // 0000000036cc: 1000110b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000036d0: bf870091
	v_mul_f32_e32 v0, v4, v0                                   // 0000000036d4: 10000104
	v_add_f32_e32 v39, v39, v0                                 // 0000000036d8: 064e0127
	v_mul_f32_e32 v0, v12, v8                                  // 0000000036dc: 1000110c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000036e0: bf870091
	v_mul_f32_e32 v0, v5, v0                                   // 0000000036e4: 10000105
	v_add_f32_e32 v38, v38, v0                                 // 0000000036e8: 064c0126
	v_mul_f32_e32 v0, v13, v8                                  // 0000000036ec: 1000110d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000036f0: bf870091
	v_mul_f32_e32 v0, v6, v0                                   // 0000000036f4: 10000106
	v_add_f32_e32 v37, v37, v0                                 // 0000000036f8: 064a0125
	v_mul_f32_e32 v0, v14, v8                                  // 0000000036fc: 1000110e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003700: bf870091
	v_mul_f32_e32 v0, v7, v0                                   // 000000003704: 10000107
	v_add_f32_e32 v36, v36, v0                                 // 000000003708: 06480124
	s_wait_alu depctr_sa_sdst(0)                               // 00000000370c: bf88ff9e
	s_cbranch_vccnz 64048                                      // 000000003710: bfa4fa30 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x4d4>
	v_mul_lo_u32 v4, s15, v22                                  // 000000003714: d72c0004 02022c0f
	v_mul_lo_u32 v5, s14, v23                                  // 00000000371c: d72c0005 02022e0e
	v_mad_co_u64_u32 v[0:1], null, s14, v22, 0                 // 000000003724: d6fe7c00 02022c0e
	v_sub_co_u32 v2, vcc_lo, s12, v22                          // 00000000372c: d7016a02 02022c0c
	s_wait_alu depctr_va_vcc(0)                                // 000000003734: bf88ff9d
	v_sub_co_ci_u32_e64 v3, null, s13, v23, vcc_lo             // 000000003738: d5217c03 01aa2e0d
	v_cmp_gt_i64_e64 s7, s[14:15], v[16:17]                    // 000000003740: d4540007 0202200e
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000003748: bf870194
	v_add3_u32 v1, v1, v5, v4                                  // 00000000374c: d6550001 04120b01
	v_cmp_lt_i64_e32 vcc_lo, 0, v[2:3]                         // 000000003754: 7ca20480
	s_delay_alu instid0(valu_dep_2)                            // 000000003758: bf870002
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 00000000375c: 3e000081
	s_and_b32 s0, vcc_lo, s7                                   // 000000003760: 8b00076a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003764: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003768: be812000
	s_cbranch_execz 28                                         // 00000000376c: bfa5001c <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1ce0>
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003770: 3e082081
	v_add_co_u32 v7, s0, s16, v0                               // 000000003774: d7000007 02020010
	v_bfe_u32 v6, v21, 16, 1                                   // 00000000377c: d6100006 02052115
	s_wait_alu depctr_va_sdst(0)                               // 000000003784: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, v1, s0                  // 000000003788: d5207c08 00020211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003790: bf870193
	v_add_co_u32 v4, s0, v7, v4                                // 000000003794: d7000004 02020907
	v_add3_u32 v6, v6, v21, 0x7fff                             // 00000000379c: d6550006 03fe2b06 00007fff
	v_or_b32_e32 v9, 0x400000, v21                             // 0000000037a8: 38122aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000037b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s0                   // 0000000037b4: d5207c05 00020b08
	v_cmp_u_f32_e64 s0, v21, v21                               // 0000000037bc: d4180000 02022b15
	s_wait_alu depctr_va_sdst(0)                               // 0000000037c4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000037c8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s0                           // 0000000037cc: d5010006 00021306
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000037d4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000037e4: 8c7e017e
	v_cmp_lt_i64_e64 s0, 1, v[2:3]                             // 0000000037e8: d4510000 02020481
	s_and_b32 s1, s0, s7                                       // 0000000037f0: 8b010700
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037f4: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000037f8: be822001
	s_cbranch_execz 35                                         // 0000000037fc: bfa50023 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1d8c>
	v_add_co_u32 v6, s1, s16, v0                               // 000000003800: d7000106 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003808: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s1                  // 00000000380c: d5207c07 00060211
	s_lshl_b64 s[4:5], s[14:15], 1                             // 000000003814: 8484810e
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003818: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000381c: bf88ff9e
	v_add_co_u32 v6, s1, v6, s4                                // 000000003820: d7000106 02000906
	v_bfe_u32 v8, v52, 16, 1                                   // 000000003828: d6100008 02052134
	s_wait_alu depctr_va_sdst(0)                               // 000000003830: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s5, v7, s1                   // 000000003834: d5207c07 00060e05
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000383c: bf870193
	v_add_co_u32 v4, s1, v6, v4                                // 000000003840: d7000104 02020906
	v_add3_u32 v8, v8, v52, 0x7fff                             // 000000003848: d6550008 03fe6908 00007fff
	v_or_b32_e32 v9, 0x400000, v52                             // 000000003854: 381268ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000385c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s1                   // 000000003860: d5207c05 00060b07
	v_cmp_u_f32_e64 s1, v52, v52                               // 000000003868: d4180001 02026934
	s_wait_alu depctr_va_sdst(0)                               // 000000003870: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003874: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s1                           // 000000003878: d5010006 00061308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003880: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000388c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003890: 8c7e027e
	v_cmp_lt_i64_e64 s1, 2, v[2:3]                             // 000000003894: d4510001 02020482
	s_lshl_b64 s[8:9], s[14:15], 1                             // 00000000389c: 8488810e
	s_and_b32 s2, s1, s7                                       // 0000000038a0: 8b020701
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038a4: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000038a8: be832002
	s_cbranch_execz 35                                         // 0000000038ac: bfa50023 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1e3c>
	v_add_co_u32 v6, s2, s16, v0                               // 0000000038b0: d7000206 02020010
	s_wait_alu depctr_va_sdst(0)                               // 0000000038b8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s2                  // 0000000038bc: d5207c07 000a0211
	s_lshl_b64 s[4:5], s[8:9], 1                               // 0000000038c4: 84848108
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 0000000038c8: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038cc: bf88ff9e
	v_add_co_u32 v6, s2, v6, s4                                // 0000000038d0: d7000206 02000906
	v_bfe_u32 v8, v49, 16, 1                                   // 0000000038d8: d6100008 02052131
	s_wait_alu depctr_va_sdst(0)                               // 0000000038e0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s5, v7, s2                   // 0000000038e4: d5207c07 000a0e05
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000038ec: bf870193
	v_add_co_u32 v4, s2, v6, v4                                // 0000000038f0: d7000204 02020906
	v_add3_u32 v8, v8, v49, 0x7fff                             // 0000000038f8: d6550008 03fe6308 00007fff
	v_or_b32_e32 v9, 0x400000, v49                             // 000000003904: 381262ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000390c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s2                   // 000000003910: d5207c05 000a0b07
	v_cmp_u_f32_e64 s2, v49, v49                               // 000000003918: d4180002 02026331
	s_wait_alu depctr_va_sdst(0)                               // 000000003920: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003924: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s2                           // 000000003928: d5010006 000a1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003930: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000393c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000003940: 8c7e037e
	v_cmp_lt_i64_e64 s2, 3, v[2:3]                             // 000000003944: d4510002 02020483
	s_mul_u64 s[10:11], s[14:15], 3                            // 00000000394c: aa8a830e
	s_and_b32 s3, s2, s7                                       // 000000003950: 8b030702
	s_wait_alu depctr_sa_sdst(0)                               // 000000003954: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000003958: be842003
	s_cbranch_execz 35                                         // 00000000395c: bfa50023 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1eec>
	v_add_co_u32 v6, s3, s16, v0                               // 000000003960: d7000306 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003968: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s3                  // 00000000396c: d5207c07 000e0211
	s_lshl_b64 s[36:37], s[10:11], 1                           // 000000003974: 84a4810a
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003978: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000397c: bf88ff9e
	v_add_co_u32 v6, s3, v6, s36                               // 000000003980: d7000306 02004906
	v_bfe_u32 v8, v48, 16, 1                                   // 000000003988: d6100008 02052130
	s_wait_alu depctr_va_sdst(0)                               // 000000003990: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s37, v7, s3                  // 000000003994: d5207c07 000e0e25
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000399c: bf870193
	v_add_co_u32 v4, s3, v6, v4                                // 0000000039a0: d7000304 02020906
	v_add3_u32 v8, v8, v48, 0x7fff                             // 0000000039a8: d6550008 03fe6108 00007fff
	v_or_b32_e32 v9, 0x400000, v48                             // 0000000039b4: 381260ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000039bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s3                   // 0000000039c0: d5207c05 000e0b07
	v_cmp_u_f32_e64 s3, v48, v48                               // 0000000039c8: d4180003 02026130
	s_wait_alu depctr_va_sdst(0)                               // 0000000039d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039d4: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s3                           // 0000000039d8: d5010006 000e1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000039e0: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000039f0: 8c7e047e
	v_cmp_lt_i64_e64 s3, 4, v[2:3]                             // 0000000039f4: d4510003 02020484
	s_lshl_b64 s[36:37], s[14:15], 2                           // 0000000039fc: 84a4820e
	s_and_b32 s4, s3, s7                                       // 000000003a00: 8b040703
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a04: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000003a08: be852004
	s_cbranch_execz 35                                         // 000000003a0c: bfa50023 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x1f9c>
	v_add_co_u32 v6, s4, s16, v0                               // 000000003a10: d7000406 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003a18: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s4                  // 000000003a1c: d5207c07 00120211
	s_lshl_b64 s[38:39], s[36:37], 1                           // 000000003a24: 84a68124
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003a28: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a2c: bf88ff9e
	v_add_co_u32 v6, s4, v6, s38                               // 000000003a30: d7000406 02004d06
	v_bfe_u32 v8, v47, 16, 1                                   // 000000003a38: d6100008 0205212f
	s_wait_alu depctr_va_sdst(0)                               // 000000003a40: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s39, v7, s4                  // 000000003a44: d5207c07 00120e27
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003a4c: bf870193
	v_add_co_u32 v4, s4, v6, v4                                // 000000003a50: d7000404 02020906
	v_add3_u32 v8, v8, v47, 0x7fff                             // 000000003a58: d6550008 03fe5f08 00007fff
	v_or_b32_e32 v9, 0x400000, v47                             // 000000003a64: 38125eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003a6c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s4                   // 000000003a70: d5207c05 00120b07
	v_cmp_u_f32_e64 s4, v47, v47                               // 000000003a78: d4180004 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003a80: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a84: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s4                           // 000000003a88: d5010006 00121308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003a90: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a9c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003aa0: 8c7e057e
	v_cmp_lt_i64_e64 s4, 5, v[2:3]                             // 000000003aa4: d4510004 02020485
	s_mul_u64 s[38:39], s[14:15], 5                            // 000000003aac: aaa6850e
	s_and_b32 s5, s4, s7                                       // 000000003ab0: 8b050704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ab4: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 000000003ab8: be862005
	s_cbranch_execz 34                                         // 000000003abc: bfa50022 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2048>
	v_add_co_u32 v6, s5, s16, v0                               // 000000003ac0: d7000506 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003ac8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s5                  // 000000003acc: d5207c07 00160211
	s_lshl_b64 s[40:41], s[38:39], 1                           // 000000003ad4: 84a88126
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003ad8: 3e082081
	v_add_co_u32 v6, s5, v6, s40                               // 000000003adc: d7000506 02005106
	v_bfe_u32 v8, v46, 16, 1                                   // 000000003ae4: d6100008 0205212e
	s_wait_alu depctr_va_sdst(0)                               // 000000003aec: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s41, v7, s5                  // 000000003af0: d5207c07 00160e29
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003af8: bf870193
	v_add_co_u32 v4, s5, v6, v4                                // 000000003afc: d7000504 02020906
	v_add3_u32 v8, v8, v46, 0x7fff                             // 000000003b04: d6550008 03fe5d08 00007fff
	v_or_b32_e32 v9, 0x400000, v46                             // 000000003b10: 38125cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003b18: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s5                   // 000000003b1c: d5207c05 00160b07
	v_cmp_u_f32_e64 s5, v46, v46                               // 000000003b24: d4180005 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003b2c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b30: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s5                           // 000000003b34: d5010006 00161308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003b3c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b48: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003b4c: 8c7e067e
	v_cmp_lt_i64_e64 s5, 6, v[2:3]                             // 000000003b50: d4510005 02020486
	s_mul_u64 s[40:41], s[14:15], 6                            // 000000003b58: aaa8860e
	s_and_b32 s6, s5, s7                                       // 000000003b5c: 8b060705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b60: bf88ff9e
	s_and_saveexec_b32 s42, s6                                 // 000000003b64: beaa2006
	s_cbranch_execz 34                                         // 000000003b68: bfa50022 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x20f4>
	v_add_co_u32 v6, s6, s16, v0                               // 000000003b6c: d7000606 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003b74: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s6                  // 000000003b78: d5207c07 001a0211
	s_lshl_b64 s[44:45], s[40:41], 1                           // 000000003b80: 84ac8128
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003b84: 3e082081
	v_add_co_u32 v6, s6, v6, s44                               // 000000003b88: d7000606 02005906
	v_bfe_u32 v8, v45, 16, 1                                   // 000000003b90: d6100008 0205212d
	s_wait_alu depctr_va_sdst(0)                               // 000000003b98: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s45, v7, s6                  // 000000003b9c: d5207c07 001a0e2d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003ba4: bf870193
	v_add_co_u32 v4, s6, v6, v4                                // 000000003ba8: d7000604 02020906
	v_add3_u32 v8, v8, v45, 0x7fff                             // 000000003bb0: d6550008 03fe5b08 00007fff
	v_or_b32_e32 v9, 0x400000, v45                             // 000000003bbc: 38125aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003bc4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s6                   // 000000003bc8: d5207c05 001a0b07
	v_cmp_u_f32_e64 s6, v45, v45                               // 000000003bd0: d4180006 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003bdc: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s6                           // 000000003be0: d5010006 001a1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003be8: ee09407c 03000000 00000004
	s_or_b32 exec_lo, exec_lo, s42                             // 000000003bf4: 8c7e2a7e
	v_cmp_lt_i64_e64 s6, 7, v[2:3]                             // 000000003bf8: d4510006 02020487
	s_mul_u64 s[42:43], s[14:15], 7                            // 000000003c00: aaaa870e
	s_and_b32 s7, s6, s7                                       // 000000003c04: 8b070706
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c08: bf88ff9e
	s_and_saveexec_b32 s44, s7                                 // 000000003c0c: beac2007
	s_cbranch_execz 34                                         // 000000003c10: bfa50022 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x219c>
	v_add_co_u32 v4, s7, s16, v0                               // 000000003c14: d7000704 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003c1c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v1, s7                  // 000000003c20: d5207c05 001e0211
	s_lshl_b64 s[46:47], s[42:43], 1                           // 000000003c28: 84ae812a
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003c2c: 3e042081
	v_add_co_u32 v4, s7, v4, s46                               // 000000003c30: d7000704 02005d04
	v_bfe_u32 v6, v44, 16, 1                                   // 000000003c38: d6100006 0205212c
	s_wait_alu depctr_va_sdst(0)                               // 000000003c40: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s47, v5, s7                  // 000000003c44: d5207c05 001e0a2f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003c4c: bf870193
	v_add_co_u32 v2, s7, v4, v2                                // 000000003c50: d7000702 02020504
	v_add3_u32 v6, v6, v44, 0x7fff                             // 000000003c58: d6550006 03fe5906 00007fff
	v_or_b32_e32 v7, 0x400000, v44                             // 000000003c64: 380e58ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003c6c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v5, v3, s7                   // 000000003c70: d5207c03 001e0705
	v_cmp_u_f32_e64 s7, v44, v44                               // 000000003c78: d4180007 0202592c
	s_wait_alu depctr_va_sdst(0)                               // 000000003c80: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c84: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s7                           // 000000003c88: d5010004 001e0f06
	global_store_d16_hi_b16 v[2:3], v4, off                    // 000000003c90: ee09407c 02000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c9c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s44                             // 000000003ca0: 8c7e2c7e
	v_cmp_gt_i64_e64 s7, s[14:15], v[18:19]                    // 000000003ca4: d4540007 0202240e
	s_and_b32 s45, vcc_lo, s7                                  // 000000003cac: 8b2d076a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cb0: bf88ff9e
	s_and_saveexec_b32 s44, s45                                // 000000003cb4: beac202d
	s_cbranch_execz 25                                         // 000000003cb8: bfa50019 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2220>
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003cbc: 3e042081
	v_add_co_u32 v5, vcc_lo, s16, v0                           // 000000003cc0: d7006a05 02020010
	v_bfe_u32 v4, v43, 16, 1                                   // 000000003cc8: d6100004 0205212b
	s_wait_alu depctr_va_vcc(0)                                // 000000003cd0: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s17, v1, vcc_lo              // 000000003cd4: d5207c06 01aa0211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003cdc: bf870193
	v_add_co_u32 v2, vcc_lo, v5, v2                            // 000000003ce0: d7006a02 02020505
	v_add3_u32 v4, v4, v43, 0x7fff                             // 000000003ce8: d6550004 03fe5704 00007fff
	v_or_b32_e32 v7, 0x400000, v43                             // 000000003cf4: 380e56ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003cfc: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v6, v3, vcc_lo               // 000000003d00: d5207c03 01aa0706
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 000000003d08: 7c30572b
	s_wait_alu depctr_va_vcc(0)                                // 000000003d0c: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo                       // 000000003d10: 02080f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003d14: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s44                             // 000000003d24: 8c7e2c7e
	s_and_b32 s44, s0, s7                                      // 000000003d28: 8b2c0700
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d2c: bf88ff9e
	s_and_saveexec_b32 s0, s44                                 // 000000003d30: be80202c
	s_cbranch_execz 31                                         // 000000003d34: bfa5001f <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x22b4>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003d38: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003d40: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003d44: d5207c05 01aa0211
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003d4c: 3e042081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000003d50: bf8701c3
	v_add_co_u32 v4, vcc_lo, v4, s8                            // 000000003d54: d7006a04 02001104
	v_bfe_u32 v6, v42, 16, 1                                   // 000000003d5c: d6100006 0205212a
	s_wait_alu depctr_va_vcc(0)                                // 000000003d64: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s9, v5, vcc_lo               // 000000003d68: d5207c05 01aa0a09
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003d70: d7006a02 02020504
	s_delay_alu instid0(valu_dep_3)                            // 000000003d78: bf870003
	v_add3_u32 v6, v6, v42, 0x7fff                             // 000000003d7c: d6550006 03fe5506 00007fff
	v_or_b32_e32 v7, 0x400000, v42                             // 000000003d88: 380e54ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003d90: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003d94: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 000000003d9c: 7c30552a
	s_wait_alu depctr_va_vcc(0)                                // 000000003da0: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003da4: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003da8: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003db4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003db8: 8c7e007e
	s_and_b32 s1, s1, s7                                       // 000000003dbc: 8b010701
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dc0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003dc4: be802001
	s_cbranch_execz 32                                         // 000000003dc8: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x234c>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003dcc: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003dd4: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003dd8: d5207c05 01aa0211
	s_lshl_b64 s[8:9], s[8:9], 1                               // 000000003de0: 84888108
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003de4: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003de8: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s8                            // 000000003dec: d7006a04 02001104
	v_bfe_u32 v6, v41, 16, 1                                   // 000000003df4: d6100006 02052129
	s_wait_alu depctr_va_vcc(0)                                // 000000003dfc: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s9, v5, vcc_lo               // 000000003e00: d5207c05 01aa0a09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003e08: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003e0c: d7006a02 02020504
	v_add3_u32 v6, v6, v41, 0x7fff                             // 000000003e14: d6550006 03fe5306 00007fff
	v_or_b32_e32 v7, 0x400000, v41                             // 000000003e20: 380e52ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003e28: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003e2c: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 000000003e34: 7c305329
	s_wait_alu depctr_va_vcc(0)                                // 000000003e38: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003e3c: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003e40: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003e50: 8c7e007e
	s_and_b32 s1, s2, s7                                       // 000000003e54: 8b010702
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e58: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003e5c: be802001
	s_cbranch_execz 32                                         // 000000003e60: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x23e4>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003e64: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003e6c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003e70: d5207c05 01aa0211
	s_lshl_b64 s[8:9], s[10:11], 1                             // 000000003e78: 8488810a
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003e7c: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e80: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s8                            // 000000003e84: d7006a04 02001104
	v_bfe_u32 v6, v40, 16, 1                                   // 000000003e8c: d6100006 02052128
	s_wait_alu depctr_va_vcc(0)                                // 000000003e94: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s9, v5, vcc_lo               // 000000003e98: d5207c05 01aa0a09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003ea0: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003ea4: d7006a02 02020504
	v_add3_u32 v6, v6, v40, 0x7fff                             // 000000003eac: d6550006 03fe5106 00007fff
	v_or_b32_e32 v7, 0x400000, v40                             // 000000003eb8: 380e50ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003ec0: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003ec4: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 000000003ecc: 7c305128
	s_wait_alu depctr_va_vcc(0)                                // 000000003ed0: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003ed4: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ed8: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ee4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003ee8: 8c7e007e
	s_and_b32 s1, s3, s7                                       // 000000003eec: 8b010703
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ef0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003ef4: be802001
	s_cbranch_execz 32                                         // 000000003ef8: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x247c>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003efc: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003f04: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003f08: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[36:37], 1                             // 000000003f10: 84828124
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003f14: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f18: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003f1c: d7006a04 02000504
	v_bfe_u32 v6, v39, 16, 1                                   // 000000003f24: d6100006 02052127
	s_wait_alu depctr_va_vcc(0)                                // 000000003f2c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003f30: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003f38: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003f3c: d7006a02 02020504
	v_add3_u32 v6, v6, v39, 0x7fff                             // 000000003f44: d6550006 03fe4f06 00007fff
	v_or_b32_e32 v7, 0x400000, v39                             // 000000003f50: 380e4eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003f58: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003f5c: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v39, v39                           // 000000003f64: 7c304f27
	s_wait_alu depctr_va_vcc(0)                                // 000000003f68: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003f6c: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003f70: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003f80: 8c7e007e
	s_and_b32 s1, s4, s7                                       // 000000003f84: 8b010704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f88: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003f8c: be802001
	s_cbranch_execz 32                                         // 000000003f90: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2514>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003f94: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003f9c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003fa0: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[38:39], 1                             // 000000003fa8: 84828126
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003fac: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fb0: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003fb4: d7006a04 02000504
	v_bfe_u32 v6, v38, 16, 1                                   // 000000003fbc: d6100006 02052126
	s_wait_alu depctr_va_vcc(0)                                // 000000003fc4: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003fc8: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003fd0: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003fd4: d7006a02 02020504
	v_add3_u32 v6, v6, v38, 0x7fff                             // 000000003fdc: d6550006 03fe4d06 00007fff
	v_or_b32_e32 v7, 0x400000, v38                             // 000000003fe8: 380e4cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003ff0: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003ff4: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 000000003ffc: 7c304d26
	s_wait_alu depctr_va_vcc(0)                                // 000000004000: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000004004: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004008: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004014: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004018: 8c7e007e
	s_and_b32 s1, s5, s7                                       // 00000000401c: 8b010705
	s_wait_alu depctr_sa_sdst(0)                               // 000000004020: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004024: be802001
	s_cbranch_execz 32                                         // 000000004028: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x25ac>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 00000000402c: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000004034: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000004038: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[40:41], 1                             // 000000004040: 84828128
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000004044: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004048: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 00000000404c: d7006a04 02000504
	v_bfe_u32 v6, v37, 16, 1                                   // 000000004054: d6100006 02052125
	s_wait_alu depctr_va_vcc(0)                                // 00000000405c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000004060: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004068: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 00000000406c: d7006a02 02020504
	v_add3_u32 v6, v6, v37, 0x7fff                             // 000000004074: d6550006 03fe4b06 00007fff
	v_or_b32_e32 v7, 0x400000, v37                             // 000000004080: 380e4aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004088: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 00000000408c: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 000000004094: 7c304b25
	s_wait_alu depctr_va_vcc(0)                                // 000000004098: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 00000000409c: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 0000000040a0: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000040b0: 8c7e007e
	s_and_b32 s1, s6, s7                                       // 0000000040b4: 8b010706
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040b8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000040bc: be802001
	s_cbranch_execz 32                                         // 0000000040c0: bfa50020 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2644>
	v_add_co_u32 v2, vcc_lo, s16, v0                           // 0000000040c4: d7006a02 02020010
	s_wait_alu depctr_va_vcc(0)                                // 0000000040cc: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s17, v1, vcc_lo              // 0000000040d0: d5207c03 01aa0211
	s_lshl_b64 s[2:3], s[42:43], 1                             // 0000000040d8: 8482812a
	v_lshlrev_b64_e32 v[0:1], 1, v[16:17]                      // 0000000040dc: 3e002081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040e0: bf88ff9e
	v_add_co_u32 v2, vcc_lo, v2, s2                            // 0000000040e4: d7006a02 02000502
	v_bfe_u32 v4, v36, 16, 1                                   // 0000000040ec: d6100004 02052124
	s_wait_alu depctr_va_vcc(0)                                // 0000000040f4: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo               // 0000000040f8: d5207c03 01aa0603
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004100: bf870193
	v_add_co_u32 v0, vcc_lo, v2, v0                            // 000000004104: d7006a00 02020102
	v_add3_u32 v4, v4, v36, 0x7fff                             // 00000000410c: d6550004 03fe4904 00007fff
	v_or_b32_e32 v5, 0x400000, v36                             // 000000004118: 380a48ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004120: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo               // 000000004124: d5207c01 01aa0303
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 00000000412c: 7c304924
	s_wait_alu depctr_va_vcc(0)                                // 000000004130: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000004134: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004138: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004144: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004148: 8c7e007e
	s_mov_b32 s0, 0                                            // 00000000414c: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004150: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000004154: 8b6a007e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004158: bf88ff9e
	s_cbranch_vccz 24                                          // 00000000415c: bfa30018 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x26c0>
	s_and_b32 s0, s33, exec_lo                                 // 000000004160: 8b007e21
	s_cselect_b32 s0, 1, 0                                     // 000000004164: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004168: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 00000000416c: bf078100
	s_cbranch_scc1 22                                          // 000000004170: bfa20016 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x26cc>
	v_lshl_or_b32 v12, v35, 3, s30                             // 000000004174: d656000c 00790723
	v_mov_b32_e32 v13, s31                                     // 00000000417c: 7e1a021f
	v_mov_b32_e32 v15, s31                                     // 000000004180: 7e1e021f
	v_mov_b32_e32 v11, s31                                     // 000000004184: 7e16021f
	v_mov_b32_e32 v9, s31                                      // 000000004188: 7e12021f
	v_or_b32_e32 v14, 1, v12                                   // 00000000418c: 381c1881
	v_or_b32_e32 v10, 2, v12                                   // 000000004190: 38141882
	v_or_b32_e32 v8, 3, v12                                    // 000000004194: 38101883
	v_or_b32_e32 v6, 4, v12                                    // 000000004198: 380c1884
	v_mov_b32_e32 v7, s31                                      // 00000000419c: 7e0e021f
	v_or_b32_e32 v4, 5, v12                                    // 0000000041a0: 38081885
	v_mov_b32_e32 v5, s31                                      // 0000000041a4: 7e0a021f
	v_or_b32_e32 v2, 6, v12                                    // 0000000041a8: 38041886
	v_mov_b32_e32 v3, s31                                      // 0000000041ac: 7e06021f
	v_or_b32_e32 v0, 7, v12                                    // 0000000041b0: 38001887
	v_mov_b32_e32 v1, s31                                      // 0000000041b4: 7e02021f
	s_mov_b32 s0, 0                                            // 0000000041b8: be800080
	s_branch 4                                                 // 0000000041bc: bfa00004 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x26d0>
	s_nop 0                                                    // 0000000041c0: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000041c4: bfb60003
	s_endpgm                                                   // 0000000041c8: bfb00000
	s_mov_b32 s0, -1                                           // 0000000041cc: be8000c1
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v55, 0             // 0000000041d0: ca100080 34360080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041d8: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 0000000041dc: 8b007e00
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v57, 0             // 0000000041e0: ca100080 36380080
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v59, 0             // 0000000041e8: ca100080 383a0080
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v21, 0             // 0000000041f0: ca100080 3a140080
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v47, 0             // 0000000041f8: ca100080 142e0080
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v49, 0             // 000000004200: ca100080 2e300080
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v51, 0             // 000000004208: ca100080 30320080
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v53, 0             // 000000004210: ca100080 32340080
	s_cselect_b32 s0, 1, 0                                     // 000000004218: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000421c: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000004220: bf078100
	s_cbranch_scc1 429                                         // 000000004224: bfa201ad <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2ddc>
	v_dual_mov_b32 v21, 0 :: v_dual_lshlrev_b32 v20, 3, v35    // 000000004228: ca220080 15144683
	v_mov_b32_e32 v15, s31                                     // 000000004230: 7e1e021f
	v_mov_b32_e32 v11, s31                                     // 000000004234: 7e16021f
	v_mov_b32_e32 v7, s31                                      // 000000004238: 7e0e021f
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_4)// 00000000423c: bf870244
	v_or_b32_e32 v12, s30, v20                                 // 000000004240: 3818281e
	v_mov_b32_e32 v58, v21                                     // 000000004244: 7e740315
	v_dual_mov_b32 v56, v21 :: v_dual_mov_b32 v5, s31          // 000000004248: ca100115 3804001f
	v_mov_b32_e32 v1, s31                                      // 000000004250: 7e02021f
	v_or_b32_e32 v14, 1, v12                                   // 000000004254: 381c1881
	v_or_b32_e32 v10, 2, v12                                   // 000000004258: 38141882
	v_or_b32_e32 v6, 4, v12                                    // 00000000425c: 380c1884
	v_or_b32_e32 v4, 5, v12                                    // 000000004260: 38081885
	v_or_b32_e32 v0, 7, v12                                    // 000000004264: 38001887
	v_cmp_gt_i64_e64 s0, s[12:13], v[14:15]                    // 000000004268: d4540000 02021c0c
	v_cmp_gt_i64_e64 s1, s[12:13], v[10:11]                    // 000000004270: d4540001 0202140c
	v_cmp_gt_i64_e32 vcc_lo, s[14:15], v[16:17]                // 000000004278: 7ca8200e
	v_mov_b32_e32 v13, s31                                     // 00000000427c: 7e1a021f
	s_lshr_b64 s[2:3], s[26:27], 5                             // 000000004280: 8582851a
	v_mov_b32_e32 v9, s31                                      // 000000004284: 7e12021f
	v_cndmask_b32_e64 v35, 0, v14, s0                          // 000000004288: d5010023 00021c80
	v_cndmask_b32_e64 v36, 0, s31, s0                          // 000000004290: d5010024 00003e80
	v_cmp_gt_i64_e64 s0, s[12:13], v[6:7]                      // 000000004298: d4540000 02020c0c
	v_cndmask_b32_e64 v37, 0, v10, s1                          // 0000000042a0: d5010025 00061480
	v_cndmask_b32_e64 v38, 0, s31, s1                          // 0000000042a8: d5010026 00043e80
	v_cmp_gt_i64_e64 s1, s[12:13], v[4:5]                      // 0000000042b0: d4540001 0202080c
	s_wait_alu depctr_va_vcc(0)                                // 0000000042b8: bf88ff9d
	v_dual_cndmask_b32 v23, 0, v17 :: v_dual_cndmask_b32 v22, 0, v16// 0000000042bc: ca522280 17162080
	s_wait_alu depctr_va_sdst(0)                               // 0000000042c4: bf88f19f
	v_cndmask_b32_e64 v41, 0, v6, s0                           // 0000000042c8: d5010029 00020c80
	v_cndmask_b32_e64 v42, 0, s31, s0                          // 0000000042d0: d501002a 00003e80
	v_add_co_u32 v28, s0, s34, v34                             // 0000000042d8: d700001c 02024422
	s_wait_alu depctr_va_sdst(0)                               // 0000000042e0: bf88f19f
	v_add_co_ci_u32_e64 v29, null, s35, 0, s0                  // 0000000042e4: d5207c1d 00010023
	v_cmp_gt_i64_e64 s0, s[12:13], v[0:1]                      // 0000000042ec: d4540000 0202000c
	v_cndmask_b32_e64 v43, 0, v4, s1                           // 0000000042f4: d501002b 00060880
	v_cndmask_b32_e64 v44, 0, s31, s1                          // 0000000042fc: d501002c 00043e80
	s_wait_alu depctr_sa_sdst(0)                               // 000000004304: bf88ff9e
	v_mul_lo_u32 v51, v42, s2                                  // 000000004308: d72c0033 0200052a
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[12:13]                // 000000004310: 7ca8180c
	v_or_b32_e32 v8, 3, v12                                    // 000000004314: 38101883
	s_wait_alu depctr_va_sdst(0)                               // 000000004318: bf88f19f
	v_cndmask_b32_e64 v47, 0, v0, s0                           // 00000000431c: d501002f 00020080
	v_cndmask_b32_e64 v48, 0, s31, s0                          // 000000004324: d5010030 00003e80
	s_lshr_b32 s0, s27, 5                                      // 00000000432c: 8500851b
	v_mul_lo_u32 v53, v44, s2                                  // 000000004330: d72c0035 0200052c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004338: bf88ff9e
	v_mul_lo_u32 v54, v43, s0                                  // 00000000433c: d72c0036 0200012b
	v_mad_co_u64_u32 v[42:43], null, v43, s2, 0                // 000000004344: d6fe7c2a 0200052b
	s_wait_alu depctr_va_vcc(0)                                // 00000000434c: bf88ff9d
	v_cndmask_b32_e32 v32, 0, v12, vcc_lo                      // 000000004350: 02401880
	v_cndmask_b32_e64 v33, 0, s31, vcc_lo                      // 000000004354: d5010021 01a83e80
	v_or_b32_e32 v2, 6, v12                                    // 00000000435c: 38041886
	v_mov_b32_e32 v3, s31                                      // 000000004360: 7e06021f
	v_cmp_gt_i64_e64 s1, s[14:15], v[18:19]                    // 000000004364: d4540001 0202240e
	v_lshlrev_b64_e32 v[22:23], 2, v[22:23]                    // 00000000436c: 3e2c2c82
	v_mul_lo_u32 v48, v48, s2                                  // 000000004370: d72c0030 02000530
	v_add3_u32 v43, v43, v54, v53                              // 000000004378: d655002b 04d66d2b
	v_mov_b32_e32 v54, v21                                     // 000000004380: 7e6c0315
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[8:9]                  // 000000004384: 7ca8100c
	v_mul_lo_u32 v49, v47, s0                                  // 000000004388: d72c0031 0200012f
	v_mul_lo_u32 v36, v36, s2                                  // 000000004390: d72c0024 02000524
	v_mul_lo_u32 v50, v35, s0                                  // 000000004398: d72c0032 02000123
	v_mul_lo_u32 v52, v41, s0                                  // 0000000043a0: d72c0034 02000129
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]                    // 0000000043a8: 3e545482
	s_wait_alu depctr_va_vcc(0)                                // 0000000043ac: bf88ff9d
	v_cndmask_b32_e32 v39, 0, v8, vcc_lo                       // 0000000043b0: 024e1080
	v_cndmask_b32_e64 v40, 0, s31, vcc_lo                      // 0000000043b4: d5010028 01a83e80
	v_add_co_u32 v26, vcc_lo, v28, 16                          // 0000000043bc: d7006a1a 0201211c
	s_wait_alu depctr_va_vcc(0)                                // 0000000043c4: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, 0, v29, vcc_lo              // 0000000043c8: d5207c1b 01aa3a80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 0000000043d0: 7ca8040c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000043d4: bf870223
	v_mad_co_u64_u32 v[24:25], null, s26, v26, v[20:21]        // 0000000043d8: d6fe7c18 0452341a
	v_mul_lo_u32 v31, s27, v26                                 // 0000000043e0: d72c001f 0202341b
	v_mul_lo_u32 v30, s26, v27                                 // 0000000043e8: d72c001e 0202361a
	s_wait_alu depctr_va_sdst(0)                               // 0000000043f0: bf88f19f
	v_cndmask_b32_e64 v27, 0, v19, s1                          // 0000000043f4: d501001b 00062680
	v_cndmask_b32_e64 v26, 0, v18, s1                          // 0000000043fc: d501001a 00062480
	s_wait_alu depctr_va_vcc(0)                                // 000000004404: bf88ff9d
	v_cndmask_b32_e32 v45, 0, v2, vcc_lo                       // 000000004408: 025a0480
	v_cndmask_b32_e64 v46, 0, s31, vcc_lo                      // 00000000440c: d501002e 01a83e80
	v_add_co_u32 v18, vcc_lo, s28, v22                         // 000000004414: d7006a12 02022c1c
	s_wait_alu depctr_va_vcc(0)                                // 00000000441c: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s29, v23, vcc_lo            // 000000004420: d5207c13 01aa2e1d
	v_add3_u32 v25, v31, v25, v30                              // 000000004428: d6550019 047a331f
	v_add_co_u32 v22, vcc_lo, s24, v24                         // 000000004430: d7006a16 02023018
	v_mul_lo_u32 v30, s26, v29                                 // 000000004438: d72c001e 02023a1a
	v_mul_lo_u32 v31, s27, v28                                 // 000000004440: d72c001f 0202381b
	s_wait_alu depctr_va_vcc(0)                                // 000000004448: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s25, v25, vcc_lo            // 00000000444c: d5207c17 01aa3219
	v_lshlrev_b64_e32 v[24:25], 2, v[26:27]                    // 000000004454: 3e303482
	v_mad_co_u64_u32 v[26:27], null, s26, v28, v[20:21]        // 000000004458: d6fe7c1a 0452381a
	v_mad_co_u64_u32 v[28:29], null, v47, s2, 0                // 000000004460: d6fe7c1c 0200052f
	v_add_co_u32 v34, s1, s30, v34                             // 000000004468: d7000122 0202441e
	s_wait_alu depctr_va_sdst(0)                               // 000000004470: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s31, 0, s1                  // 000000004474: d5207c2f 0005001f
	v_mul_lo_u32 v46, v46, s2                                  // 00000000447c: d72c002e 0200052e
	v_mul_lo_u32 v55, v45, s0                                  // 000000004484: d72c0037 0200012d
	v_add3_u32 v27, v31, v27, v30                              // 00000000448c: d655001b 047a371f
	v_add3_u32 v29, v29, v49, v48                              // 000000004494: d655001d 04c2631d
	v_mad_co_u64_u32 v[30:31], null, s26, v34, v[20:21]        // 00000000449c: d6fe7c1e 0452441a
	v_mul_lo_u32 v20, s26, v47                                 // 0000000044a4: d72c0014 02025e1a
	v_mul_lo_u32 v47, s27, v34                                 // 0000000044ac: d72c002f 0202441b
	v_mul_lo_u32 v48, v33, s2                                  // 0000000044b4: d72c0030 02000521
	v_mul_lo_u32 v49, v32, s0                                  // 0000000044bc: d72c0031 02000120
	v_mad_co_u64_u32 v[32:33], null, v32, s2, 0                // 0000000044c4: d6fe7c20 02000520
	v_mad_co_u64_u32 v[34:35], null, v35, s2, 0                // 0000000044cc: d6fe7c22 02000523
	v_mad_co_u64_u32 v[44:45], null, v45, s2, 0                // 0000000044d4: d6fe7c2c 0200052d
	v_add_co_u32 v24, vcc_lo, s28, v24                         // 0000000044dc: d7006a18 0202301c
	v_add3_u32 v20, v47, v31, v20                              // 0000000044e4: d6550014 04523f2f
	v_mul_lo_u32 v47, v38, s2                                  // 0000000044ec: d72c002f 02000526
	s_wait_alu depctr_va_vcc(0)                                // 0000000044f4: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s29, v25, vcc_lo            // 0000000044f8: d5207c19 01aa321d
	v_add3_u32 v33, v33, v49, v48                              // 000000004500: d6550021 04c26321
	v_add3_u32 v35, v35, v50, v36                              // 000000004508: d6550023 04926523
	v_mul_lo_u32 v48, v37, s0                                  // 000000004510: d72c0030 02000125
	v_mad_co_u64_u32 v[36:37], null, v37, s2, 0                // 000000004518: d6fe7c24 02000525
	v_mul_lo_u32 v49, v40, s2                                  // 000000004520: d72c0031 02000528
	v_mad_co_u64_u32 v[40:41], null, v41, s2, 0                // 000000004528: d6fe7c28 02000529
	v_add3_u32 v45, v45, v55, v46                              // 000000004530: d655002d 04ba6f2d
	v_add_co_u32 v26, vcc_lo, s24, v26                         // 000000004538: d7006a1a 02023418
	s_wait_alu depctr_va_vcc(0)                                // 000000004540: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, s25, v27, vcc_lo            // 000000004544: d5207c1b 01aa3619
	v_add3_u32 v37, v37, v48, v47                              // 00000000454c: d6550025 04be6125
	v_mov_b32_e32 v48, v21                                     // 000000004554: 7e600315
	v_mul_lo_u32 v50, v39, s0                                  // 000000004558: d72c0032 02000127
	v_mad_co_u64_u32 v[38:39], null, v39, s2, 0                // 000000004560: d6fe7c26 02000527
	v_add3_u32 v41, v41, v52, v51                              // 000000004568: d6550029 04ce6929
	v_add_co_u32 v30, vcc_lo, s22, v30                         // 000000004570: d7006a1e 02023c16
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]                    // 000000004578: 3e383882
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 00000000457c: 3e404082
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]                    // 000000004580: 3e444482
	v_lshlrev_b64_e32 v[36:37], 2, v[36:37]                    // 000000004584: 3e484882
	v_add3_u32 v39, v39, v50, v49                              // 000000004588: d6550027 04c66527
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000004590: 3e505082
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]                    // 000000004594: 3e585882
	s_wait_alu depctr_va_vcc(0)                                // 000000004598: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, s23, v20, vcc_lo            // 00000000459c: d5207c1f 01aa2817
	v_lshlrev_b64_e32 v[38:39], 2, v[38:39]                    // 0000000045a4: 3e4c4c82
	v_mov_b32_e32 v59, v21                                     // 0000000045a8: 7e760315
	v_mov_b32_e32 v57, v21                                     // 0000000045ac: 7e720315
	v_dual_mov_b32 v55, v21 :: v_dual_mov_b32 v52, v21         // 0000000045b0: ca100115 37340115
	v_mov_b32_e32 v53, v21                                     // 0000000045b8: 7e6a0315
	v_dual_mov_b32 v51, v21 :: v_dual_mov_b32 v50, v21         // 0000000045bc: ca100115 33320115
	v_mov_b32_e32 v49, v21                                     // 0000000045c4: 7e620315
	v_dual_mov_b32 v47, v21 :: v_dual_mov_b32 v46, v21         // 0000000045c8: ca100115 2f2e0115
	v_mov_b32_e32 v20, v21                                     // 0000000045d0: 7e280315
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000045d4: 8480820e
	s_mov_b64 s[2:3], 0                                        // 0000000045d8: be820180
	v_add_co_u32 v60, vcc_lo, s18, v32                         // 0000000045dc: d7006a3c 02024012
	s_wait_alu depctr_va_vcc(0)                                // 0000000045e4: bf88ff9d
	v_add_co_ci_u32_e64 v61, null, s19, v33, vcc_lo            // 0000000045e8: d5207c3d 01aa4213
	v_add_co_u32 v62, vcc_lo, s18, v34                         // 0000000045f0: d7006a3e 02024412
	s_wait_alu depctr_va_vcc(0)                                // 0000000045f8: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, s19, v35, vcc_lo            // 0000000045fc: d5207c3f 01aa4613
	v_add_co_u32 v64, vcc_lo, s18, v36                         // 000000004604: d7006a40 02024812
	s_wait_alu depctr_va_vcc(0)                                // 00000000460c: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s19, v37, vcc_lo            // 000000004610: d5207c41 01aa4a13
	v_add_co_u32 v66, vcc_lo, s18, v38                         // 000000004618: d7006a42 02024c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004620: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s19, v39, vcc_lo            // 000000004624: d5207c43 01aa4e13
	v_add_co_u32 v70, vcc_lo, s18, v40                         // 00000000462c: d7006a46 02025012
	s_wait_alu depctr_va_vcc(0)                                // 000000004634: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, s19, v41, vcc_lo            // 000000004638: d5207c47 01aa5213
	v_add_co_u32 v72, vcc_lo, s18, v42                         // 000000004640: d7006a48 02025412
	s_clause 0x1                                               // 000000004648: bf850001
	global_load_b64 v[76:77], v[30:31], off                    // 00000000464c: ee05407c 0000004c 0000001e
	global_load_b64 v[78:79], v[30:31], off offset:16          // 000000004658: ee05407c 0000004e 0000101e
	s_clause 0x1                                               // 000000004664: bf850001
	global_load_b64 v[68:69], v[26:27], off                    // 000000004668: ee05407c 00000044 0000001a
	global_load_b64 v[80:81], v[26:27], off offset:16          // 000000004674: ee05407c 00000050 0000101a
	s_clause 0x1                                               // 000000004680: bf850001
	global_load_b64 v[82:83], v[22:23], off                    // 000000004684: ee05407c 00000052 00000016
	global_load_b64 v[84:85], v[22:23], off offset:16          // 000000004690: ee05407c 00000054 00001016
	s_wait_alu depctr_va_vcc(0)                                // 00000000469c: bf88ff9d
	v_add_co_ci_u32_e64 v73, null, s19, v43, vcc_lo            // 0000000046a0: d5207c49 01aa5613
	v_add_co_u32 v74, vcc_lo, s18, v44                         // 0000000046a8: d7006a4a 02025812
	s_wait_alu depctr_va_vcc(0)                                // 0000000046b0: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, s19, v45, vcc_lo            // 0000000046b4: d5207c4b 01aa5a13
	v_add_co_u32 v86, vcc_lo, s18, v28                         // 0000000046bc: d7006a56 02023812
	global_load_b32 v88, v[18:19], off                         // 0000000046c4: ee05007c 00000058 00000012
	s_wait_alu depctr_va_vcc(0)                                // 0000000046d0: bf88ff9d
	v_add_co_ci_u32_e64 v87, null, s19, v29, vcc_lo            // 0000000046d4: d5207c57 01aa3a13
	global_load_b32 v89, v[24:25], off                         // 0000000046dc: ee05007c 00000059 00000018
	s_clause 0x7                                               // 0000000046e8: bf850007
	global_load_b32 v90, v[60:61], off                         // 0000000046ec: ee05007c 0000005a 0000003c
	global_load_b32 v91, v[62:63], off                         // 0000000046f8: ee05007c 0000005b 0000003e
	global_load_b32 v92, v[64:65], off                         // 000000004704: ee05007c 0000005c 00000040
	global_load_b32 v93, v[66:67], off                         // 000000004710: ee05007c 0000005d 00000042
	global_load_b32 v94, v[70:71], off                         // 00000000471c: ee05007c 0000005e 00000046
	global_load_b32 v95, v[72:73], off                         // 000000004728: ee05007c 0000005f 00000048
	global_load_b32 v96, v[74:75], off                         // 000000004734: ee05007c 00000060 0000004a
	global_load_b32 v86, v[86:87], off                         // 000000004740: ee05007c 00000056 00000056
	s_wait_alu depctr_sa_sdst(0)                               // 00000000474c: bf88ff9e
	v_add_co_u32 v18, vcc_lo, v18, s0                          // 000000004750: d7006a12 02000112
	s_wait_alu depctr_va_vcc(0)                                // 000000004758: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s1, v19, vcc_lo             // 00000000475c: d5207c13 01aa2601
	v_add_co_u32 v22, vcc_lo, v22, 32                          // 000000004764: d7006a16 02014116
	s_wait_alu depctr_va_vcc(0)                                // 00000000476c: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, 0, v23, vcc_lo              // 000000004770: d5207c17 01aa2e80
	v_add_co_u32 v24, vcc_lo, v24, s0                          // 000000004778: d7006a18 02000118
	s_add_nc_u64 s[2:3], s[2:3], 32                            // 000000004780: a982a002
	s_wait_alu depctr_va_vcc(0)                                // 000000004784: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s1, v25, vcc_lo             // 000000004788: d5207c19 01aa3201
	v_add_co_u32 v26, vcc_lo, v26, 32                          // 000000004790: d7006a1a 0201411a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004798: bf88ff9e
	v_cmp_lt_i64_e64 s4, s[2:3], s[20:21]                      // 00000000479c: d4510004 02002802
	s_wait_alu depctr_va_vcc(0)                                // 0000000047a4: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, 0, v27, vcc_lo              // 0000000047a8: d5207c1b 01aa3680
	v_add_co_u32 v30, vcc_lo, v30, 32                          // 0000000047b0: d7006a1e 0201411e
	s_wait_alu depctr_va_vcc(0)                                // 0000000047b8: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, 0, v31, vcc_lo              // 0000000047bc: d5207c1f 01aa3e80
	s_and_b32 vcc_lo, exec_lo, s4                              // 0000000047c4: 8b6a047e
	s_add_nc_u64 s[18:19], s[18:19], 4                         // 0000000047c8: a9928412
	s_wait_loadcnt 0xd                                         // 0000000047cc: bfc0000d
	v_wmma_f32_16x16x16_fp8_fp8 v[60:67], v[76:77], v[68:69], 0// 0000000047d0: cc46403c 1a02894c
	s_wait_loadcnt 0xb                                         // 0000000047d8: bfc0000b
	v_wmma_f32_16x16x16_fp8_fp8 v[68:75], v[76:77], v[82:83], 0// 0000000047dc: cc464044 1a02a54c
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000047e4: bf870122
	v_wmma_f32_16x16x16_fp8_fp8 v[60:67], v[78:79], v[80:81], v[60:67]// 0000000047e8: cc46403c 1cf2a14e
	s_wait_loadcnt 0xa                                         // 0000000047f0: bfc0000a
	v_wmma_f32_16x16x16_fp8_fp8 v[68:75], v[78:79], v[84:85], v[68:75]// 0000000047f4: cc464044 1d12a94e
	s_wait_loadcnt 0x6                                         // 0000000047fc: bfc00006
	v_dual_mul_f32 v84, v90, v89 :: v_dual_mul_f32 v85, v91, v89// 000000004800: c8c6b35a 5454b35b
	s_wait_loadcnt 0x5                                         // 000000004808: bfc00005
	v_dual_mul_f32 v87, v92, v89 :: v_dual_mul_f32 v76, v90, v88// 00000000480c: c8c6b35c 574cb15a
	v_dual_mul_f32 v77, v88, v91 :: v_dual_mul_f32 v78, v88, v92// 000000004814: c8c6b758 4d4eb958
	s_wait_loadcnt 0x3                                         // 00000000481c: bfc00003
	v_dual_mul_f32 v79, v88, v93 :: v_dual_mul_f32 v80, v88, v94// 000000004820: c8c6bb58 4f50bd58
	s_wait_loadcnt 0x1                                         // 000000004828: bfc00001
	v_dual_mul_f32 v81, v88, v95 :: v_dual_mul_f32 v82, v88, v96// 00000000482c: c8c6bf58 5152c158
	s_wait_loadcnt 0x0                                         // 000000004834: bfc00000
	v_dual_mul_f32 v83, v88, v86 :: v_dual_mul_f32 v88, v93, v89// 000000004838: c8c6ad58 5358b35d
	v_dual_mul_f32 v90, v94, v89 :: v_dual_mul_f32 v91, v95, v89// 000000004840: c8c6b35e 5a5ab35f
	v_dual_mul_f32 v92, v96, v89 :: v_dual_mul_f32 v63, v63, v79// 000000004848: c8c6b360 5c3e9f3f
	s_delay_alu instid0(valu_dep_3)                            // 000000004850: bf870003
	v_dual_mul_f32 v86, v86, v89 :: v_dual_mul_f32 v67, v67, v83// 000000004854: c8c6b356 5642a743
	v_dual_mul_f32 v60, v60, v76 :: v_dual_mul_f32 v61, v61, v77// 00000000485c: c8c6993c 3c3c9b3d
	v_dual_mul_f32 v62, v62, v78 :: v_dual_mul_f32 v65, v65, v81// 000000004864: c8c69d3e 3e40a341
	v_dual_mul_f32 v64, v64, v80 :: v_dual_mul_f32 v69, v69, v85// 00000000486c: c8c6a140 4044ab45
	v_dual_mul_f32 v66, v66, v82 :: v_dual_mul_f32 v71, v71, v88// 000000004874: c8c6a542 4246b147
	v_dual_mul_f32 v68, v68, v84 :: v_dual_mul_f32 v73, v73, v91// 00000000487c: c8c6a944 4448b749
	v_dual_mul_f32 v70, v70, v87 :: v_dual_mul_f32 v75, v75, v86// 000000004884: c8c6af46 464aad4b
	v_dual_mul_f32 v72, v72, v90 :: v_dual_add_f32 v21, v21, v60// 00000000488c: c8c8b548 48147915
	v_dual_mul_f32 v74, v74, v92 :: v_dual_add_f32 v59, v59, v61// 000000004894: c8c8b94a 4a3a7b3b
	v_dual_add_f32 v58, v58, v62 :: v_dual_add_f32 v57, v57, v63// 00000000489c: c9087d3a 3a387f39
	v_dual_add_f32 v56, v56, v64 :: v_dual_add_f32 v55, v55, v65// 0000000048a4: c9088138 38368337
	v_dual_add_f32 v54, v54, v66 :: v_dual_add_f32 v53, v53, v68// 0000000048ac: c9088536 36348935
	v_dual_add_f32 v52, v52, v67 :: v_dual_add_f32 v51, v51, v69// 0000000048b4: c9088734 34328b33
	v_dual_add_f32 v50, v50, v70 :: v_dual_add_f32 v49, v49, v71// 0000000048bc: c9088d32 32308f31
	v_dual_add_f32 v48, v48, v72 :: v_dual_add_f32 v47, v47, v73// 0000000048c4: c9089130 302e932f
	v_add_f32_e32 v46, v46, v74                                // 0000000048cc: 065c952e
	v_add_f32_e32 v20, v20, v75                                // 0000000048d0: 06289714
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048d4: bf88ff9e
	s_cbranch_vccnz 65344                                      // 0000000048d8: bfa4ff40 <tessera_rocm_scaled_matmul_f02c4a0cd9bed703+0x2adc>
	v_mul_lo_u32 v18, s15, v12                                 // 0000000048dc: d72c0012 0202180f
	v_mul_lo_u32 v19, s14, v13                                 // 0000000048e4: d72c0013 02021a0e
	v_mad_co_u64_u32 v[12:13], null, s14, v12, 0               // 0000000048ec: d6fe7c0c 0202180e
	v_mul_lo_u32 v22, s15, v14                                 // 0000000048f4: d72c0016 02021c0f
	v_mul_lo_u32 v23, s14, v15                                 // 0000000048fc: d72c0017 02021e0e
	v_mad_co_u64_u32 v[14:15], null, s14, v14, 0               // 000000004904: d6fe7c0e 02021c0e
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 00000000490c: 3e202081
	v_mul_lo_u32 v25, s15, v10                                 // 000000004910: d72c0019 0202140f
	v_bfe_u32 v24, v59, 16, 1                                  // 000000004918: d6100018 0205213b
	v_add3_u32 v13, v13, v19, v18                              // 000000004920: d655000d 044a270d
	v_bfe_u32 v18, v21, 16, 1                                  // 000000004928: d6100012 02052115
	v_or_b32_e32 v19, 0x400000, v21                            // 000000004930: 38262aff 00400000
	v_add3_u32 v15, v15, v23, v22                              // 000000004938: d655000f 045a2f0f
	v_or_b32_e32 v22, 0x400000, v59                            // 000000004940: 382c76ff 00400000
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000004948: 3e181881
	v_add3_u32 v18, v18, v21, 0x7fff                           // 00000000494c: d6550012 03fe2b12 00007fff
	v_mul_lo_u32 v23, s14, v9                                  // 000000004958: d72c0017 0202120e
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000004960: 3e1c1c81
	s_delay_alu instid0(valu_dep_4)                            // 000000004964: bf870004
	v_add_co_u32 v12, vcc_lo, s16, v12                         // 000000004968: d7006a0c 02021810
	s_wait_alu depctr_va_vcc(0)                                // 000000004970: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc_lo            // 000000004974: d5207c0d 01aa1a11
	v_cmp_u_f32_e32 vcc_lo, v21, v21                           // 00000000497c: 7c302b15
	v_add3_u32 v21, v24, v59, 0x7fff                           // 000000004980: d6550015 03fe7718 00007fff
	v_or_b32_e32 v24, 0x400000, v57                            // 00000000498c: 383072ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004994: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo                    // 000000004998: 02242712
	v_mul_lo_u32 v19, s14, v11                                 // 00000000499c: d72c0013 0202160e
	v_mad_co_u64_u32 v[10:11], null, s14, v10, 0               // 0000000049a4: d6fe7c0a 0202140e
	v_add_co_u32 v12, vcc_lo, v12, v16                         // 0000000049ac: d7006a0c 0202210c
	s_wait_alu depctr_va_vcc(0)                                // 0000000049b4: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, v13, v17, vcc_lo            // 0000000049b8: d5207c0d 01aa230d
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 0000000049c0: 7c30773b
	s_delay_alu instid0(valu_dep_4)                            // 0000000049c4: bf870004
	v_add3_u32 v11, v11, v19, v25                              // 0000000049c8: d655000b 0466270b
	global_store_d16_hi_b16 v[12:13], v18, off                 // 0000000049d0: ee09407c 09000000 0000000c
	s_wait_alu depctr_va_vcc(0)                                // 0000000049dc: bf88ff9d
	v_cndmask_b32_e32 v18, v21, v22, vcc_lo                    // 0000000049e0: 02242d15
	v_add_co_u32 v14, vcc_lo, s16, v14                         // 0000000049e4: d7006a0e 02021c10
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 0000000049ec: 3e141481
	s_wait_alu depctr_va_vcc(0)                                // 0000000049f0: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, s17, v15, vcc_lo            // 0000000049f4: d5207c0f 01aa1e11
	v_bfe_u32 v19, v58, 16, 1                                  // 0000000049fc: d6100013 0205213a
	v_mul_lo_u32 v22, s15, v8                                  // 000000004a04: d72c0016 0202100f
	v_mad_co_u64_u32 v[8:9], null, s14, v8, 0                  // 000000004a0c: d6fe7c08 0202100e
	v_add_co_u32 v14, vcc_lo, v14, v16                         // 000000004a14: d7006a0e 0202210e
	s_wait_alu depctr_va_vcc(0)                                // 000000004a1c: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v15, v17, vcc_lo            // 000000004a20: d5207c0f 01aa230f
	v_add_co_u32 v10, vcc_lo, s16, v10                         // 000000004a28: d7006a0a 02021410
	v_add3_u32 v19, v19, v58, 0x7fff                           // 000000004a30: d6550013 03fe7513 00007fff
	v_or_b32_e32 v21, 0x400000, v58                            // 000000004a3c: 382a74ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004a44: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc_lo            // 000000004a48: d5207c0b 01aa1611
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 000000004a50: 7c30753a
	v_add3_u32 v9, v9, v23, v22                                // 000000004a54: d6550009 045a2f09
	v_mul_lo_u32 v22, s15, v6                                  // 000000004a5c: d72c0016 02020c0f
	v_mul_lo_u32 v23, s14, v7                                  // 000000004a64: d72c0017 02020e0e
	v_mad_co_u64_u32 v[6:7], null, s14, v6, 0                  // 000000004a6c: d6fe7c06 02020c0e
	s_wait_alu depctr_va_vcc(0)                                // 000000004a74: bf88ff9d
	v_cndmask_b32_e32 v19, v19, v21, vcc_lo                    // 000000004a78: 02262b13
	v_bfe_u32 v21, v57, 16, 1                                  // 000000004a7c: d6100015 02052139
	v_add_co_u32 v10, vcc_lo, v10, v16                         // 000000004a84: d7006a0a 0202210a
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000004a8c: 3e101081
	s_wait_alu depctr_va_vcc(0)                                // 000000004a90: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v11, v17, vcc_lo            // 000000004a94: d5207c0b 01aa230b
	v_add3_u32 v21, v21, v57, 0x7fff                           // 000000004a9c: d6550015 03fe7315 00007fff
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000004aa8: 7c307339
	v_add3_u32 v7, v7, v23, v22                                // 000000004aac: d6550007 045a2f07
	s_clause 0x1                                               // 000000004ab4: bf850001
	global_store_d16_hi_b16 v[14:15], v18, off                 // 000000004ab8: ee09407c 09000000 0000000e
	global_store_d16_hi_b16 v[10:11], v19, off                 // 000000004ac4: ee09407c 09800000 0000000a
	v_bfe_u32 v19, v56, 16, 1                                  // 000000004ad0: d6100013 02052138
	s_wait_alu depctr_va_vcc(0)                                // 000000004ad8: bf88ff9d
	v_cndmask_b32_e32 v18, v21, v24, vcc_lo                    // 000000004adc: 02243115
	v_add_co_u32 v8, vcc_lo, s16, v8                           // 000000004ae0: d7006a08 02021010
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 000000004ae8: 3e0c0c81
	s_wait_alu depctr_va_vcc(0)                                // 000000004aec: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc_lo              // 000000004af0: d5207c09 01aa1211
	s_delay_alu instid0(valu_dep_3)                            // 000000004af8: bf870003
	v_add_co_u32 v8, vcc_lo, v8, v16                           // 000000004afc: d7006a08 02022108
	v_mul_lo_u32 v22, s15, v4                                  // 000000004b04: d72c0016 0202080f
	v_mul_lo_u32 v23, s14, v5                                  // 000000004b0c: d72c0017 02020a0e
	v_mad_co_u64_u32 v[4:5], null, s14, v4, 0                  // 000000004b14: d6fe7c04 0202080e
	s_wait_alu depctr_va_vcc(0)                                // 000000004b1c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo              // 000000004b20: d5207c09 01aa2309
	v_add_co_u32 v6, vcc_lo, s16, v6                           // 000000004b28: d7006a06 02020c10
	v_add3_u32 v19, v19, v56, 0x7fff                           // 000000004b30: d6550013 03fe7113 00007fff
	v_or_b32_e32 v21, 0x400000, v56                            // 000000004b3c: 382a70ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004b44: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc_lo              // 000000004b48: d5207c07 01aa0e11
	v_cmp_u_f32_e32 vcc_lo, v56, v56                           // 000000004b50: 7c307138
	v_add3_u32 v5, v5, v23, v22                                // 000000004b54: d6550005 045a2f05
	v_mul_lo_u32 v22, s15, v2                                  // 000000004b5c: d72c0016 0202040f
	v_mul_lo_u32 v23, s14, v3                                  // 000000004b64: d72c0017 0202060e
	v_mad_co_u64_u32 v[2:3], null, s14, v2, 0                  // 000000004b6c: d6fe7c02 0202040e
	s_wait_alu depctr_va_vcc(0)                                // 000000004b74: bf88ff9d
	v_cndmask_b32_e32 v19, v19, v21, vcc_lo                    // 000000004b78: 02262b13
	v_bfe_u32 v21, v55, 16, 1                                  // 000000004b7c: d6100015 02052137
	v_add_co_u32 v6, vcc_lo, v6, v16                           // 000000004b84: d7006a06 02022106
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004b8c: 3e080881
	s_wait_alu depctr_va_vcc(0)                                // 000000004b90: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v7, v17, vcc_lo              // 000000004b94: d5207c07 01aa2307
	v_add3_u32 v21, v21, v55, 0x7fff                           // 000000004b9c: d6550015 03fe6f15 00007fff
	v_or_b32_e32 v24, 0x400000, v55                            // 000000004ba8: 38306eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 000000004bb0: 7c306f37
	s_clause 0x1                                               // 000000004bb4: bf850001
	global_store_d16_hi_b16 v[8:9], v18, off                   // 000000004bb8: ee09407c 09000000 00000008
	global_store_d16_hi_b16 v[6:7], v19, off                   // 000000004bc4: ee09407c 09800000 00000006
	v_add3_u32 v3, v3, v23, v22                                // 000000004bd0: d6550003 045a2f03
	v_bfe_u32 v19, v54, 16, 1                                  // 000000004bd8: d6100013 02052136
	v_mul_lo_u32 v22, s15, v0                                  // 000000004be0: d72c0016 0202000f
	s_wait_alu depctr_va_vcc(0)                                // 000000004be8: bf88ff9d
	v_cndmask_b32_e32 v18, v21, v24, vcc_lo                    // 000000004bec: 02243115
	v_add_co_u32 v4, vcc_lo, s16, v4                           // 000000004bf0: d7006a04 02020810
	s_wait_alu depctr_va_vcc(0)                                // 000000004bf8: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo              // 000000004bfc: d5207c05 01aa0a11
	v_mul_lo_u32 v23, s14, v1                                  // 000000004c04: d72c0017 0202020e
	v_mad_co_u64_u32 v[0:1], null, s14, v0, 0                  // 000000004c0c: d6fe7c00 0202000e
	v_add_co_u32 v4, vcc_lo, v4, v16                           // 000000004c14: d7006a04 02022104
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004c1c: 3e040481
	v_add3_u32 v19, v19, v54, 0x7fff                           // 000000004c20: d6550013 03fe6d13 00007fff
	v_or_b32_e32 v21, 0x400000, v54                            // 000000004c2c: 382a6cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c34: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v5, v17, vcc_lo              // 000000004c38: d5207c05 01aa2305
	v_cmp_u_f32_e32 vcc_lo, v54, v54                           // 000000004c40: 7c306d36
	v_add3_u32 v1, v1, v23, v22                                // 000000004c44: d6550001 045a2f01
	v_or_b32_e32 v22, 0x400000, v52                            // 000000004c4c: 382c68ff 00400000
	global_store_d16_hi_b16 v[4:5], v18, off                   // 000000004c54: ee09407c 09000000 00000004
	s_wait_alu depctr_va_vcc(0)                                // 000000004c60: bf88ff9d
	v_cndmask_b32_e32 v19, v19, v21, vcc_lo                    // 000000004c64: 02262b13
	v_add_co_u32 v2, vcc_lo, s16, v2                           // 000000004c68: d7006a02 02020410
	s_wait_alu depctr_va_vcc(0)                                // 000000004c70: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s17, v3, vcc_lo              // 000000004c74: d5207c03 01aa0611
	v_bfe_u32 v21, v52, 16, 1                                  // 000000004c7c: d6100015 02052134
	s_delay_alu instid0(valu_dep_3)                            // 000000004c84: bf870003
	v_add_co_u32 v2, vcc_lo, v2, v16                           // 000000004c88: d7006a02 02022102
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000004c90: 3e000081
	s_wait_alu depctr_va_vcc(0)                                // 000000004c94: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v3, v17, vcc_lo              // 000000004c98: d5207c03 01aa2303
	v_add3_u32 v21, v21, v52, 0x7fff                           // 000000004ca0: d6550015 03fe6915 00007fff
	v_cmp_u_f32_e32 vcc_lo, v52, v52                           // 000000004cac: 7c306934
	global_store_d16_hi_b16 v[2:3], v19, off                   // 000000004cb0: ee09407c 09800000 00000002
	v_bfe_u32 v19, v53, 16, 1                                  // 000000004cbc: d6100013 02052135
	s_wait_alu depctr_va_vcc(0)                                // 000000004cc4: bf88ff9d
	v_cndmask_b32_e32 v18, v21, v22, vcc_lo                    // 000000004cc8: 02242d15
	v_add_co_u32 v0, vcc_lo, s16, v0                           // 000000004ccc: d7006a00 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000004cd4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo              // 000000004cd8: d5207c01 01aa0211
	v_add3_u32 v19, v19, v53, 0x7fff                           // 000000004ce0: d6550013 03fe6b13 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004cec: bf870003
	v_add_co_u32 v0, vcc_lo, v0, v16                           // 000000004cf0: d7006a00 02022100
	v_or_b32_e32 v21, 0x400000, v53                            // 000000004cf8: 382a6aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004d00: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v17, vcc_lo              // 000000004d04: d5207c01 01aa2301
	v_bfe_u32 v16, v51, 16, 1                                  // 000000004d0c: d6100010 02052133
	v_cmp_u_f32_e32 vcc_lo, v53, v53                           // 000000004d14: 7c306b35
	global_store_d16_hi_b16 v[0:1], v18, off                   // 000000004d18: ee09407c 09000000 00000000
	v_or_b32_e32 v18, 0x400000, v51                            // 000000004d24: 382466ff 00400000
	v_add3_u32 v16, v16, v51, 0x7fff                           // 000000004d2c: d6550010 03fe6710 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004d38: bf88ff9d
	v_cndmask_b32_e32 v17, v19, v21, vcc_lo                    // 000000004d3c: 02222b13
	v_bfe_u32 v19, v50, 16, 1                                  // 000000004d40: d6100013 02052132
	v_cmp_u_f32_e32 vcc_lo, v51, v51                           // 000000004d48: 7c306733
	global_store_d16_hi_b16 v[12:13], v17, off offset:32       // 000000004d4c: ee09407c 08800000 0000200c
	v_add3_u32 v12, v19, v50, 0x7fff                           // 000000004d58: d655000c 03fe6513 00007fff
	v_or_b32_e32 v13, 0x400000, v50                            // 000000004d64: 381a64ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004d6c: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v18, vcc_lo                    // 000000004d70: 02202510
	v_bfe_u32 v17, v49, 16, 1                                  // 000000004d74: d6100011 02052131
	v_cmp_u_f32_e32 vcc_lo, v50, v50                           // 000000004d7c: 7c306532
	global_store_d16_hi_b16 v[14:15], v16, off offset:32       // 000000004d80: ee09407c 08000000 0000200e
	v_add3_u32 v14, v17, v49, 0x7fff                           // 000000004d8c: d655000e 03fe6311 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004d98: bf88ff9d
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo                    // 000000004d9c: 02181b0c
	v_bfe_u32 v13, v48, 16, 1                                  // 000000004da0: d610000d 02052130
	v_or_b32_e32 v15, 0x400000, v49                            // 000000004da8: 381e62ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v49, v49                           // 000000004db0: 7c306331
	v_or_b32_e32 v16, 0x400000, v46                            // 000000004db4: 38205cff 00400000
	global_store_d16_hi_b16 v[10:11], v12, off offset:32       // 000000004dbc: ee09407c 06000000 0000200a
	v_add3_u32 v10, v13, v48, 0x7fff                           // 000000004dc8: d655000a 03fe610d 00007fff
	v_or_b32_e32 v11, 0x400000, v48                            // 000000004dd4: 381660ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004ddc: bf88ff9d
	v_cndmask_b32_e32 v12, v14, v15, vcc_lo                    // 000000004de0: 02181f0e
	v_bfe_u32 v13, v47, 16, 1                                  // 000000004de4: d610000d 0205212f
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000004dec: 7c306130
	v_bfe_u32 v14, v46, 16, 1                                  // 000000004df0: d610000e 0205212e
	v_or_b32_e32 v15, 0x400000, v47                            // 000000004df8: 381e5eff 00400000
	v_or_b32_e32 v17, 0x400000, v20                            // 000000004e00: 382228ff 00400000
	v_add3_u32 v13, v13, v47, 0x7fff                           // 000000004e08: d655000d 03fe5f0d 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004e14: bf88ff9d
	v_cndmask_b32_e32 v10, v10, v11, vcc_lo                    // 000000004e18: 0214170a
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000004e1c: 7c305f2f
	v_bfe_u32 v11, v20, 16, 1                                  // 000000004e20: d610000b 02052114
	v_add3_u32 v14, v14, v46, 0x7fff                           // 000000004e28: d655000e 03fe5d0e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004e34: bf88ff9d
	v_cndmask_b32_e32 v13, v13, v15, vcc_lo                    // 000000004e38: 021a1f0d
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 000000004e3c: 7c305d2e
	v_add3_u32 v11, v11, v20, 0x7fff                           // 000000004e40: d655000b 03fe290b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004e4c: bf88ff9d
	v_cndmask_b32_e32 v14, v14, v16, vcc_lo                    // 000000004e50: 021c210e
	v_cmp_u_f32_e32 vcc_lo, v20, v20                           // 000000004e54: 7c302914
	s_wait_alu depctr_va_vcc(0)                                // 000000004e58: bf88ff9d
	v_cndmask_b32_e32 v11, v11, v17, vcc_lo                    // 000000004e5c: 0216230b
	s_clause 0x4                                               // 000000004e60: bf850004
	global_store_d16_hi_b16 v[8:9], v12, off offset:32         // 000000004e64: ee09407c 06000000 00002008
	global_store_d16_hi_b16 v[6:7], v10, off offset:32         // 000000004e70: ee09407c 05000000 00002006
	global_store_d16_hi_b16 v[4:5], v13, off offset:32         // 000000004e7c: ee09407c 06800000 00002004
	global_store_d16_hi_b16 v[2:3], v14, off offset:32         // 000000004e88: ee09407c 07000000 00002002
	global_store_d16_hi_b16 v[0:1], v11, off offset:32         // 000000004e94: ee09407c 05800000 00002000
	s_nop 0                                                    // 000000004ea0: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000004ea4: bfb60003
	s_endpgm                                                   // 000000004ea8: bfb00000
	s_code_end                                                 // 000000004eac: bf9f0000
	s_code_end                                                 // 000000004eb0: bf9f0000
	s_code_end                                                 // 000000004eb4: bf9f0000
	s_code_end                                                 // 000000004eb8: bf9f0000
	s_code_end                                                 // 000000004ebc: bf9f0000
	s_code_end                                                 // 000000004ec0: bf9f0000
	s_code_end                                                 // 000000004ec4: bf9f0000
	s_code_end                                                 // 000000004ec8: bf9f0000
	s_code_end                                                 // 000000004ecc: bf9f0000
	s_code_end                                                 // 000000004ed0: bf9f0000
	s_code_end                                                 // 000000004ed4: bf9f0000
	s_code_end                                                 // 000000004ed8: bf9f0000
	s_code_end                                                 // 000000004edc: bf9f0000
	s_code_end                                                 // 000000004ee0: bf9f0000
	s_code_end                                                 // 000000004ee4: bf9f0000
	s_code_end                                                 // 000000004ee8: bf9f0000
	s_code_end                                                 // 000000004eec: bf9f0000
	s_code_end                                                 // 000000004ef0: bf9f0000
	s_code_end                                                 // 000000004ef4: bf9f0000
	s_code_end                                                 // 000000004ef8: bf9f0000
	s_code_end                                                 // 000000004efc: bf9f0000
	s_code_end                                                 // 000000004f00: bf9f0000
	s_code_end                                                 // 000000004f04: bf9f0000
	s_code_end                                                 // 000000004f08: bf9f0000
	s_code_end                                                 // 000000004f0c: bf9f0000
	s_code_end                                                 // 000000004f10: bf9f0000
	s_code_end                                                 // 000000004f14: bf9f0000
	s_code_end                                                 // 000000004f18: bf9f0000
	s_code_end                                                 // 000000004f1c: bf9f0000
	s_code_end                                                 // 000000004f20: bf9f0000
	s_code_end                                                 // 000000004f24: bf9f0000
	s_code_end                                                 // 000000004f28: bf9f0000
	s_code_end                                                 // 000000004f2c: bf9f0000
	s_code_end                                                 // 000000004f30: bf9f0000
	s_code_end                                                 // 000000004f34: bf9f0000
	s_code_end                                                 // 000000004f38: bf9f0000
	s_code_end                                                 // 000000004f3c: bf9f0000
	s_code_end                                                 // 000000004f40: bf9f0000
	s_code_end                                                 // 000000004f44: bf9f0000
	s_code_end                                                 // 000000004f48: bf9f0000
	s_code_end                                                 // 000000004f4c: bf9f0000
	s_code_end                                                 // 000000004f50: bf9f0000
	s_code_end                                                 // 000000004f54: bf9f0000
	s_code_end                                                 // 000000004f58: bf9f0000
	s_code_end                                                 // 000000004f5c: bf9f0000
	s_code_end                                                 // 000000004f60: bf9f0000
	s_code_end                                                 // 000000004f64: bf9f0000
	s_code_end                                                 // 000000004f68: bf9f0000
	s_code_end                                                 // 000000004f6c: bf9f0000
	s_code_end                                                 // 000000004f70: bf9f0000
	s_code_end                                                 // 000000004f74: bf9f0000
	s_code_end                                                 // 000000004f78: bf9f0000
	s_code_end                                                 // 000000004f7c: bf9f0000
	s_code_end                                                 // 000000004f80: bf9f0000
	s_code_end                                                 // 000000004f84: bf9f0000
	s_code_end                                                 // 000000004f88: bf9f0000
	s_code_end                                                 // 000000004f8c: bf9f0000
	s_code_end                                                 // 000000004f90: bf9f0000
	s_code_end                                                 // 000000004f94: bf9f0000
	s_code_end                                                 // 000000004f98: bf9f0000
	s_code_end                                                 // 000000004f9c: bf9f0000
	s_code_end                                                 // 000000004fa0: bf9f0000
	s_code_end                                                 // 000000004fa4: bf9f0000
	s_code_end                                                 // 000000004fa8: bf9f0000
	s_code_end                                                 // 000000004fac: bf9f0000
	s_code_end                                                 // 000000004fb0: bf9f0000
	s_code_end                                                 // 000000004fb4: bf9f0000
	s_code_end                                                 // 000000004fb8: bf9f0000
	s_code_end                                                 // 000000004fbc: bf9f0000
	s_code_end                                                 // 000000004fc0: bf9f0000
	s_code_end                                                 // 000000004fc4: bf9f0000
	s_code_end                                                 // 000000004fc8: bf9f0000
	s_code_end                                                 // 000000004fcc: bf9f0000
	s_code_end                                                 // 000000004fd0: bf9f0000
	s_code_end                                                 // 000000004fd4: bf9f0000
	s_code_end                                                 // 000000004fd8: bf9f0000
	s_code_end                                                 // 000000004fdc: bf9f0000
	s_code_end                                                 // 000000004fe0: bf9f0000
	s_code_end                                                 // 000000004fe4: bf9f0000
	s_code_end                                                 // 000000004fe8: bf9f0000
	s_code_end                                                 // 000000004fec: bf9f0000
	s_code_end                                                 // 000000004ff0: bf9f0000
	s_code_end                                                 // 000000004ff4: bf9f0000
	s_code_end                                                 // 000000004ff8: bf9f0000
	s_code_end                                                 // 000000004ffc: bf9f0000
	s_code_end                                                 // 000000005000: bf9f0000
	s_code_end                                                 // 000000005004: bf9f0000
	s_code_end                                                 // 000000005008: bf9f0000
	s_code_end                                                 // 00000000500c: bf9f0000
	s_code_end                                                 // 000000005010: bf9f0000
	s_code_end                                                 // 000000005014: bf9f0000
	s_code_end                                                 // 000000005018: bf9f0000
	s_code_end                                                 // 00000000501c: bf9f0000
	s_code_end                                                 // 000000005020: bf9f0000
	s_code_end                                                 // 000000005024: bf9f0000
	s_code_end                                                 // 000000005028: bf9f0000
	s_code_end                                                 // 00000000502c: bf9f0000
	s_code_end                                                 // 000000005030: bf9f0000
	s_code_end                                                 // 000000005034: bf9f0000
	s_code_end                                                 // 000000005038: bf9f0000
	s_code_end                                                 // 00000000503c: bf9f0000
	s_code_end                                                 // 000000005040: bf9f0000
	s_code_end                                                 // 000000005044: bf9f0000
	s_code_end                                                 // 000000005048: bf9f0000
	s_code_end                                                 // 00000000504c: bf9f0000
	s_code_end                                                 // 000000005050: bf9f0000
	s_code_end                                                 // 000000005054: bf9f0000
	s_code_end                                                 // 000000005058: bf9f0000
	s_code_end                                                 // 00000000505c: bf9f0000
	s_code_end                                                 // 000000005060: bf9f0000
	s_code_end                                                 // 000000005064: bf9f0000
	s_code_end                                                 // 000000005068: bf9f0000
	s_code_end                                                 // 00000000506c: bf9f0000
	s_code_end                                                 // 000000005070: bf9f0000
	s_code_end                                                 // 000000005074: bf9f0000
	s_code_end                                                 // 000000005078: bf9f0000
	s_code_end                                                 // 00000000507c: bf9f0000
