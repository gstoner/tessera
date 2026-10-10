
/tmp/tmpntzbhder.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434>:
	s_load_b128 s[24:27], s[0:1], 0xc8                         // 000000001b00: f4004600 f80000c8
	v_lshrrev_b32_e32 v6, 1, v0                                // 000000001b08: 320c0081
	s_mov_b32 s6, ttmp7                                        // 000000001b0c: be860073
	s_ashr_i32 s7, ttmp7, 31                                   // 000000001b10: 86079f73
	s_clause 0x4                                               // 000000001b14: bf850004
	s_load_b64 s[2:3], s[0:1], 0xd8                            // 000000001b18: f4002080 f80000d8
	s_load_b64 s[10:11], s[0:1], 0x8                           // 000000001b20: f4002280 f8000008
	s_load_b64 s[12:13], s[0:1], 0x30                          // 000000001b28: f4002300 f8000030
	s_load_b64 s[8:9], s[0:1], 0x58                            // 000000001b30: f4002200 f8000058
	s_load_b64 s[28:29], s[0:1], 0x80                          // 000000001b38: f4002700 f8000080
	s_lshl_b64 s[6:7], s[6:7], 7                               // 000000001b40: 84868706
	s_mov_b32 s4, ttmp9                                        // 000000001b44: be840075
	v_dual_mov_b32 v2, s7 :: v_dual_lshlrev_b32 v5, 1, v0      // 000000001b48: ca220007 02040081
	v_or_b32_e32 v1, s6, v6                                    // 000000001b50: 38020c06
	s_ashr_i32 s5, ttmp9, 31                                   // 000000001b54: 86059f75
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v3, s7             // 000000001b58: ca100080 0e020007
	s_lshl_b64 s[22:23], s[4:5], 7                             // 000000001b60: 84968704
	v_dual_mov_b32 v4, s7 :: v_dual_and_b32 v23, 64, v5        // 000000001b64: ca240007 04160ac0
	v_or_b32_e32 v5, s22, v6                                   // 000000001b6c: 380a0c16
	v_mul_u32_u24_e32 v7, 48, v6                               // 000000001b70: 160e0cb0
	v_dual_mov_b32 v105, 0 :: v_dual_and_b32 v26, 8, v6        // 000000001b74: ca240080 691a0c88
	v_mov_b32_e32 v64, 0                                       // 000000001b7c: 7e800280
	s_wait_kmcnt 0x0                                           // 000000001b80: bfc70000
	s_add_nc_u64 s[4:5], s[24:25], -1                          // 000000001b84: a984c118
	s_lshr_b64 s[30:31], s[2:3], 5                             // 000000001b88: 859e8502
	v_cmp_gt_u64_e32 vcc_lo, s[4:5], v[1:2]                    // 000000001b8c: 7cb80204
	v_and_b32_e32 v22, 15, v0                                  // 000000001b90: 362c008f
	v_lshlrev_b32_e32 v0, 4, v0                                // 000000001b94: 30000084
	v_dual_mov_b32 v99, 0 :: v_dual_and_b32 v2, 0x60, v6       // 000000001b98: ca240080 63020cff 00000060
	v_dual_mov_b32 v54, 0 :: v_dual_cndmask_b32 v1, s4, v1     // 000000001ba4: ca120080 36000204
	v_cndmask_b32_e32 v4, s5, v4, vcc_lo                       // 000000001bac: 02080805
	s_delay_alu instid0(valu_dep_4)                            // 000000001bb0: bf870004
	v_and_b32_e32 v8, 16, v0                                   // 000000001bb4: 36100090
	v_or_b32_e32 v6, v23, v22                                  // 000000001bb8: 380c2d17
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v52, 0             // 000000001bbc: ca100080 5f340080
	v_mul_lo_u32 v10, v1, s3                                   // 000000001bc4: d72c000a 02000701
	v_mad_co_u64_u32 v[0:1], null, v1, s2, s[10:11]            // 000000001bcc: d6fe7c00 00280501
	v_mul_lo_u32 v11, v4, s2                                   // 000000001bd4: d72c000b 02000504
	v_add_nc_u32_e32 v15, v7, v8                               // 000000001bdc: 4a1e1107
	v_mul_lo_u32 v7, s3, v5                                    // 000000001be0: d72c0007 02020a03
	v_mad_co_u64_u32 v[4:5], null, s2, v5, s[12:13]            // 000000001be8: d6fe7c04 00320a02
	s_mul_i32 s2, s2, s23                                      // 000000001bf0: 96021702
	v_or_b32_e32 v27, 16, v6                                   // 000000001bf4: 38360c90
	v_or_b32_e32 v28, 32, v6                                   // 000000001bf8: 38380ca0
	v_add_co_u32 v16, vcc_lo, v0, v8                           // 000000001bfc: d7006a10 02021100
	v_add3_u32 v1, v11, v1, v10                                // 000000001c04: d6550001 042a030b
	s_lshr_b32 s10, s3, 5                                      // 000000001c0c: 850a8503
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c10: bf88ff9e
	v_add3_u32 v0, v7, v5, s2                                  // 000000001c14: d6550000 000a0b07
	v_mov_b32_e32 v5, s7                                       // 000000001c1c: 7e0a0207
	v_or_b32_e32 v9, 16, v2                                    // 000000001c20: 38120490
	v_or_b32_e32 v24, s6, v2                                   // 000000001c24: 38300406
	v_or_b32_e32 v2, v2, v22                                   // 000000001c28: 38042d02
	s_wait_alu depctr_va_vcc(0)                                // 000000001c2c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, 0, v1, vcc_lo               // 000000001c30: d5207c11 01aa0280
	v_or_b32_e32 v34, s6, v9                                   // 000000001c38: 38441206
	v_or_b32_e32 v9, v9, v22                                   // 000000001c3c: 38122d09
	v_mul_u32_u24_e32 v1, 48, v2                               // 000000001c40: 160204b0
	v_or_b32_e32 v30, 48, v6                                   // 000000001c44: 383c0cb0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v50, 0             // 000000001c48: ca100080 47320080
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001c50: bf870214
	v_mul_u32_u24_e32 v2, 48, v9                               // 000000001c54: 160412b0
	v_or_b32_e32 v20, v1, v26                                  // 000000001c58: 38283501
	v_mul_u32_u24_e32 v1, 48, v27                              // 000000001c5c: 160236b0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v48, 0             // 000000001c60: ca100080 45300080
	s_delay_alu instid0(valu_dep_4) | instskip(skip_1) | instid1(valu_dep_4)// 000000001c68: bf870224
	v_or_b32_e32 v21, v2, v26                                  // 000000001c6c: 382a3502
	v_or_b32_e32 v2, v24, v26                                  // 000000001c70: 38043518
	v_or_b32_e32 v36, v1, v26                                  // 000000001c74: 38483501
	v_mov_b32_e32 v1, s23                                      // 000000001c78: 7e020217
	v_add_co_u32 v18, vcc_lo, v4, v8                           // 000000001c7c: d7006a12 02021104
	s_wait_alu depctr_va_vcc(0)                                // 000000001c84: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, 0, v0, vcc_lo               // 000000001c88: d5207c13 01aa0080
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[2:3]                  // 000000001c90: 7ca80418
	v_mul_u32_u24_e32 v0, 48, v6                               // 000000001c94: 16000cb0
	v_mul_u32_u24_e32 v4, 48, v30                              // 000000001c98: 16083cb0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v67, 0             // 000000001c9c: ca100080 44420080
	v_mov_b32_e32 v66, 0                                       // 000000001ca4: 7e840280
	s_delay_alu instid0(valu_dep_4)                            // 000000001ca8: bf870004
	v_or_b32_e32 v35, v0, v26                                  // 000000001cac: 38463500
	v_mul_u32_u24_e32 v0, 48, v28                              // 000000001cb0: 160038b0
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb4: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v3 :: v_dual_cndmask_b32 v8, 0, v2// 000000001cb8: ca520680 07080480
	v_mov_b32_e32 v65, 0                                       // 000000001cc0: 7e820280
	v_add_nc_u32_e32 v119, 0x1800, v35                         // 000000001cc4: 4aee46ff 00001800
	v_or_b32_e32 v37, v0, v26                                  // 000000001ccc: 384a3500
	v_or_b32_e32 v0, s22, v6                                   // 000000001cd0: 38000c16
	v_mul_lo_u32 v12, s10, v8                                  // 000000001cd4: d72c000c 0202100a
	v_mul_lo_u32 v13, s30, v7                                  // 000000001cdc: d72c000d 02020e1e
	v_mad_co_u64_u32 v[6:7], null, s30, v8, 0                  // 000000001ce4: d6fe7c06 0202101e
	v_add_nc_u32_e32 v121, 0x1800, v37                         // 000000001cec: 4af24aff 00001800
	v_cmp_gt_i64_e64 s4, s[26:27], v[0:1]                      // 000000001cf4: d4540004 0202001a
	v_mov_b32_e32 v55, 0                                       // 000000001cfc: 7e6e0280
	v_mov_b32_e32 v53, 0                                       // 000000001d00: 7e6a0280
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v82, 0             // 000000001d04: ca100080 33520080
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v76, 0             // 000000001d0c: ca100080 314c0080
	v_add3_u32 v7, v7, v13, v12                                // 000000001d14: d6550007 04321b07
	v_mov_b32_e32 v13, s7                                      // 000000001d1c: 7e1a0207
	v_or_b32_e32 v29, 1, v26                                   // 000000001d20: 383a3481
	v_or_b32_e32 v38, v4, v26                                  // 000000001d24: 384c3504
	v_or_b32_e32 v31, 2, v26                                   // 000000001d28: 383e3482
	v_or_b32_e32 v33, 3, v26                                   // 000000001d2c: 38423483
	v_or_b32_e32 v42, 7, v26                                   // 000000001d30: 38543487
	v_or_b32_e32 v4, v29, v24                                  // 000000001d34: 3808311d
	s_wait_alu depctr_va_sdst(0)                               // 000000001d38: bf88f19f
	v_cndmask_b32_e64 v9, 0, v1, s4                            // 000000001d3c: d5010009 00120280
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v72, 0             // 000000001d44: ca100080 25480080
	v_or_b32_e32 v12, v33, v24                                 // 000000001d4c: 38183121
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[4:5]                  // 000000001d50: 7ca80818
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v62, 0             // 000000001d54: ca100080 233e0080
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v58, 0             // 000000001d5c: ca100080 553a0080
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v56, 0             // 000000001d64: ca100080 51380080
	s_wait_alu depctr_va_vcc(0)                                // 000000001d6c: bf88ff9d
	v_cndmask_b32_e32 v10, 0, v4, vcc_lo                       // 000000001d70: 02140880
	v_or_b32_e32 v4, v31, v24                                  // 000000001d74: 3808311f
	v_dual_cndmask_b32 v8, 0, v5 :: v_dual_mov_b32 v75, 0      // 000000001d78: ca500a80 084a0080
	v_mov_b32_e32 v73, 0                                       // 000000001d80: 7e920280
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001d84: bf870214
	v_mul_lo_u32 v25, s10, v10                                 // 000000001d88: d72c0019 0202140a
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[4:5]                  // 000000001d90: 7ca80818
	v_mad_co_u64_u32 v[10:11], null, s30, v10, 0               // 000000001d94: d6fe7c0a 0202141e
	v_mov_b32_e32 v63, 0                                       // 000000001d9c: 7e7e0280
	v_mov_b32_e32 v61, 0                                       // 000000001da0: 7e7a0280
	v_mov_b32_e32 v59, 0                                       // 000000001da4: 7e760280
	v_mov_b32_e32 v57, 0                                       // 000000001da8: 7e720280
	s_wait_alu depctr_va_vcc(0)                                // 000000001dac: bf88ff9d
	v_dual_cndmask_b32 v39, 0, v5 :: v_dual_cndmask_b32 v40, 0, v4// 000000001db0: ca520a80 27280880
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[12:13]                // 000000001db8: 7ca81818
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000001dbc: 3e080c82
	v_mov_b32_e32 v47, 0                                       // 000000001dc0: 7e5e0280
	s_delay_alu instid0(valu_dep_4)                            // 000000001dc4: bf870004
	v_mul_lo_u32 v39, s30, v39                                 // 000000001dc8: d72c0027 02024e1e
	v_mad_co_u64_u32 v[6:7], null, s30, v40, 0                 // 000000001dd0: d6fe7c06 0202501e
	s_mov_b64 s[34:35], 0                                      // 000000001dd8: bea20180
	s_wait_alu depctr_va_vcc(0)                                // 000000001ddc: bf88ff9d
	v_cndmask_b32_e32 v41, 0, v12, vcc_lo                      // 000000001de0: 02521880
	v_mul_lo_u32 v32, s30, v8                                  // 000000001de4: d72c0020 0202101e
	v_cndmask_b32_e64 v8, 0, v0, s4                            // 000000001dec: d5010008 00120080
	v_add_nc_u32_e32 v122, 0x1800, v38                         // 000000001df4: 4af44cff 00001800
	v_mov_b32_e32 v38, 0                                       // 000000001dfc: 7e4c0280
	v_add_nc_u32_e32 v120, 0x1800, v36                         // 000000001e00: 4af048ff 00001800
	v_mov_b32_e32 v36, 0                                       // 000000001e08: 7e480280
	v_mov_b32_e32 v106, 0                                      // 000000001e0c: 7ed40280
	v_mov_b32_e32 v100, 0                                      // 000000001e10: 7ec80280
	v_add3_u32 v11, v11, v32, v25                              // 000000001e14: d655000b 0466410b
	v_or_b32_e32 v32, 4, v26                                   // 000000001e1c: 38403484
	v_mul_lo_u32 v25, s10, v40                                 // 000000001e20: d72c0019 0202500a
	v_cndmask_b32_e32 v40, 0, v13, vcc_lo                      // 000000001e28: 02501a80
	v_add_co_u32 v77, vcc_lo, s8, v4                           // 000000001e2c: d7006a4d 02020808
	s_delay_alu instid0(valu_dep_4)                            // 000000001e34: bf870004
	v_or_b32_e32 v12, v32, v24                                 // 000000001e38: 38183120
	s_wait_alu depctr_va_vcc(0)                                // 000000001e3c: bf88ff9d
	v_add_co_ci_u32_e64 v78, null, s9, v5, vcc_lo              // 000000001e40: d5207c4e 01aa0a09
	v_lshlrev_b64_e32 v[4:5], 2, v[10:11]                      // 000000001e48: 3e081482
	v_add3_u32 v7, v7, v39, v25                                // 000000001e4c: d6550007 04664f07
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[12:13]                // 000000001e54: 7ca81818
	v_mul_lo_u32 v25, s10, v41                                 // 000000001e58: d72c0019 0202520a
	v_mad_co_u64_u32 v[10:11], null, s30, v41, 0               // 000000001e60: d6fe7c0a 0202521e
	v_or_b32_e32 v41, 6, v26                                   // 000000001e68: 38523486
	v_mov_b32_e32 v98, 0                                       // 000000001e6c: 7ec40280
	s_wait_alu depctr_va_vcc(0)                                // 000000001e70: bf88ff9d
	v_dual_mov_b32 v86, 0 :: v_dual_cndmask_b32 v13, 0, v13    // 000000001e74: ca120080 560c1a80
	v_cndmask_b32_e32 v12, 0, v12, vcc_lo                      // 000000001e7c: 02181880
	v_add_co_u32 v79, vcc_lo, s8, v4                           // 000000001e80: d7006a4f 02020808
	s_wait_alu depctr_va_vcc(0)                                // 000000001e88: bf88ff9d
	v_add_co_ci_u32_e64 v80, null, s9, v5, vcc_lo              // 000000001e8c: d5207c50 01aa0a09
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000001e94: 3e080c82
	v_mad_co_u64_u32 v[6:7], null, s30, v12, 0                 // 000000001e98: d6fe7c06 0202181e
	v_mov_b32_e32 v70, 0                                       // 000000001ea0: 7e8c0280
	v_mov_b32_e32 v74, 0                                       // 000000001ea4: 7e940280
	v_mov_b32_e32 v60, 0                                       // 000000001ea8: 7e780280
	v_add_co_u32 v83, vcc_lo, s8, v4                           // 000000001eac: d7006a53 02020808
	s_wait_alu depctr_va_vcc(0)                                // 000000001eb4: bf88ff9d
	v_add_co_ci_u32_e64 v84, null, s9, v5, vcc_lo              // 000000001eb8: d5207c54 01aa0a09
	v_mov_b32_e32 v5, s7                                       // 000000001ec0: 7e0a0207
	v_mul_lo_u32 v39, s30, v40                                 // 000000001ec4: d72c0027 0202501e
	v_or_b32_e32 v40, 5, v26                                   // 000000001ecc: 38503485
	v_or_b32_e32 v4, v41, v24                                  // 000000001ed0: 38083129
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000001ed4: bf870211
	v_cmp_gt_i64_e64 s2, s[24:25], v[4:5]                      // 000000001ed8: d4540002 02020818
	v_add3_u32 v11, v11, v39, v25                              // 000000001ee0: d655000b 04664f0b
	v_mul_lo_u32 v25, s10, v12                                 // 000000001ee8: d72c0019 0202180a
	v_mul_lo_u32 v39, s30, v13                                 // 000000001ef0: d72c0027 02021a1e
	v_or_b32_e32 v12, v40, v24                                 // 000000001ef8: 38183128
	v_mov_b32_e32 v13, s7                                      // 000000001efc: 7e1a0207
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]                    // 000000001f00: 3e141482
	s_wait_alu depctr_va_sdst(0)                               // 000000001f04: bf88f19f
	v_cndmask_b32_e64 v43, 0, v4, s2                           // 000000001f08: d501002b 000a0880
	s_delay_alu instid0(valu_dep_3)                            // 000000001f10: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[12:13]                // 000000001f14: 7ca81818
	v_add3_u32 v7, v7, v39, v25                                // 000000001f18: d6550007 04664f07
	s_wait_alu depctr_va_vcc(0)                                // 000000001f20: bf88ff9d
	v_cndmask_b32_e32 v39, 0, v12, vcc_lo                      // 000000001f24: 024e1880
	v_or_b32_e32 v12, v42, v24                                 // 000000001f28: 3818312a
	v_cndmask_b32_e32 v25, 0, v13, vcc_lo                      // 000000001f2c: 02321a80
	v_cndmask_b32_e64 v24, 0, v5, s2                           // 000000001f30: d5010018 000a0a80
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]                        // 000000001f38: 3e0c0c82
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001f3c: bf870214
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[12:13]                // 000000001f40: 7ca81818
	v_mul_lo_u32 v45, s30, v25                                 // 000000001f44: d72c002d 0202321e
	s_delay_alu instid0(valu_dep_4)                            // 000000001f4c: bf870004
	v_mul_lo_u32 v46, s30, v24                                 // 000000001f50: d72c002e 0202301e
	v_mad_co_u64_u32 v[24:25], null, s30, v43, 0               // 000000001f58: d6fe7c18 0202561e
	s_wait_alu depctr_va_vcc(0)                                // 000000001f60: bf88ff9d
	v_dual_cndmask_b32 v13, 0, v13 :: v_dual_cndmask_b32 v12, 0, v12// 000000001f64: ca521a80 0d0c1880
	v_mul_lo_u32 v44, s10, v39                                 // 000000001f6c: d72c002c 02024e0a
	v_mad_co_u64_u32 v[4:5], null, s30, v39, 0                 // 000000001f74: d6fe7c04 02024e1e
	v_mul_lo_u32 v39, s10, v43                                 // 000000001f7c: d72c0027 0202560a
	v_add_co_u32 v87, vcc_lo, s8, v10                          // 000000001f84: d7006a57 02021408
	s_wait_alu depctr_va_vcc(0)                                // 000000001f8c: bf88ff9d
	v_add_co_ci_u32_e64 v88, null, s9, v11, vcc_lo             // 000000001f90: d5207c58 01aa1609
	v_mul_lo_u32 v43, s10, v12                                 // 000000001f98: d72c002b 0202180a
	v_mul_lo_u32 v13, s30, v13                                 // 000000001fa0: d72c000d 02021a1e
	v_mad_co_u64_u32 v[10:11], null, s30, v12, 0               // 000000001fa8: d6fe7c0a 0202181e
	v_add3_u32 v5, v5, v45, v44                                // 000000001fb0: d6550005 04b25b05
	v_add3_u32 v25, v25, v46, v39                              // 000000001fb8: d6550019 049e5d19
	v_add_co_u32 v89, vcc_lo, s8, v6                           // 000000001fc0: d7006a59 02020c08
	s_wait_alu depctr_va_vcc(0)                                // 000000001fc8: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s9, v7, vcc_lo              // 000000001fcc: d5207c5a 01aa0e09
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000001fd4: 3e080882
	v_add3_u32 v11, v11, v13, v43                              // 000000001fd8: d655000b 04ae1b0b
	v_lshlrev_b64_e32 v[6:7], 2, v[24:25]                      // 000000001fe0: 3e0c3082
	v_or_b32_e32 v12, s22, v27                                 // 000000001fe4: 38183616
	v_dual_mov_b32 v27, s23 :: v_dual_mov_b32 v46, 0           // 000000001fe8: ca100017 1b2e0080
	s_delay_alu instid0(valu_dep_4)                            // 000000001ff0: bf870004
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]                    // 000000001ff4: 3e141482
	v_add_co_u32 v91, vcc_lo, s8, v4                           // 000000001ff8: d7006a5b 02020808
	s_wait_alu depctr_va_vcc(0)                                // 000000002000: bf88ff9d
	v_add_co_ci_u32_e64 v92, null, s9, v5, vcc_lo              // 000000002004: d5207c5c 01aa0a09
	v_mov_b32_e32 v5, s7                                       // 00000000200c: 7e0a0207
	v_or_b32_e32 v4, v34, v26                                  // 000000002010: 38083522
	v_add_co_u32 v93, vcc_lo, s8, v6                           // 000000002014: d7006a5d 02020c08
	s_wait_alu depctr_va_vcc(0)                                // 00000000201c: bf88ff9d
	v_add_co_ci_u32_e64 v94, null, s9, v7, vcc_lo              // 000000002020: d5207c5e 01aa0e09
	v_add_co_u32 v96, vcc_lo, s8, v10                          // 000000002028: d7006a60 02021408
	s_wait_alu depctr_va_vcc(0)                                // 000000002030: bf88ff9d
	v_add_co_ci_u32_e64 v97, null, s9, v11, vcc_lo             // 000000002034: d5207c61 01aa1609
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[4:5]                  // 00000000203c: 7ca80818
	v_mov_b32_e32 v13, s23                                     // 000000002040: 7e1a0217
	v_mov_b32_e32 v7, s23                                      // 000000002044: 7e0e0217
	v_or_b32_e32 v6, s22, v28                                  // 000000002048: 380c3816
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0             // 00000000204c: ca100080 2d2c0080
	s_wait_alu depctr_va_vcc(0)                                // 000000002054: bf88ff9d
	v_cndmask_b32_e32 v24, 0, v5, vcc_lo                       // 000000002058: 02300a80
	v_cmp_gt_i64_e64 s3, s[26:27], v[12:13]                    // 00000000205c: d4540003 0202181a
	v_cmp_gt_i64_e64 s2, s[26:27], v[6:7]                      // 000000002064: d4540002 02020c1a
	v_cndmask_b32_e32 v26, 0, v4, vcc_lo                       // 00000000206c: 02340880
	s_delay_alu instid0(valu_dep_4) | instskip(skip_1) | instid1(valu_dep_4)// 000000002070: bf870224
	v_mul_lo_u32 v39, s30, v24                                 // 000000002074: d72c0027 0202301e
	s_wait_alu depctr_va_sdst(0)                               // 00000000207c: bf88f19f
	v_cndmask_b32_e64 v11, 0, v13, s3                          // 000000002080: d501000b 000e1a80
	v_cndmask_b32_e64 v10, 0, v12, s3                          // 000000002088: d501000a 000e1880
	v_mov_b32_e32 v13, s7                                      // 000000002090: 7e1a0207
	v_or_b32_e32 v12, v34, v29                                 // 000000002094: 38183b22
	v_cndmask_b32_e64 v25, 0, v7, s2                           // 000000002098: d5010019 000a0e80
	v_mul_lo_u32 v7, s10, v26                                  // 0000000020a0: d72c0007 0202340a
	v_mad_co_u64_u32 v[28:29], null, s30, v26, 0               // 0000000020a8: d6fe7c1c 0202341e
	v_or_b32_e32 v26, s22, v30                                 // 0000000020b0: 38343c16
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[12:13]                // 0000000020b4: 7ca81818
	v_cndmask_b32_e64 v24, 0, v6, s2                           // 0000000020b8: d5010018 000a0c80
	s_wait_alu depctr_va_vcc(0)                                // 0000000020c0: bf88ff9d
	v_cndmask_b32_e32 v43, 0, v12, vcc_lo                      // 0000000020c4: 02561880
	v_or_b32_e32 v12, v34, v31                                 // 0000000020c8: 38183f22
	v_add3_u32 v29, v29, v39, v7                               // 0000000020cc: d655001d 041e4f1d
	v_cndmask_b32_e32 v30, 0, v13, vcc_lo                      // 0000000020d4: 023c1a80
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[26:27]                // 0000000020d8: 7ca8341a
	v_mul_lo_u32 v31, s10, v43                                 // 0000000020dc: d72c001f 0202560a
	v_cmp_gt_i64_e64 s5, s[24:25], v[12:13]                    // 0000000020e4: d4540005 02021818
	v_mad_co_u64_u32 v[6:7], null, s30, v43, 0                 // 0000000020ec: d6fe7c06 0202561e
	v_mul_lo_u32 v30, s30, v30                                 // 0000000020f4: d72c001e 02023c1e
	s_wait_alu depctr_va_vcc(0)                                // 0000000020fc: bf88ff9d
	v_dual_cndmask_b32 v27, 0, v27 :: v_dual_cndmask_b32 v26, 0, v26// 000000002100: ca523680 1b1a3480
	s_wait_alu depctr_va_sdst(0)                               // 000000002108: bf88f19f
	v_cndmask_b32_e64 v39, 0, v13, s5                          // 00000000210c: d5010027 00161a80
	v_cndmask_b32_e64 v43, 0, v12, s5                          // 000000002114: d501002b 00161880
	v_lshlrev_b64_e32 v[12:13], 2, v[28:29]                    // 00000000211c: 3e183882
	v_mov_b32_e32 v29, s7                                      // 000000002120: 7e3a0207
	v_or_b32_e32 v28, v34, v33                                 // 000000002124: 38384322
	v_add3_u32 v7, v7, v30, v31                                // 000000002128: d6550007 047e3d07
	v_mul_lo_u32 v33, s10, v43                                 // 000000002130: d72c0021 0202560a
	v_mul_lo_u32 v39, s30, v39                                 // 000000002138: d72c0027 02024e1e
	v_add_co_u32 v101, s6, s8, v12                             // 000000002140: d7000665 02021808
	v_cmp_gt_i64_e64 s5, s[24:25], v[28:29]                    // 000000002148: d4540005 02023818
	s_wait_alu depctr_va_sdst(0)                               // 000000002150: bf88f19f
	v_add_co_ci_u32_e64 v102, null, s9, v13, s6                // 000000002154: d5207c66 001a1a09
	v_mov_b32_e32 v13, s7                                      // 00000000215c: 7e1a0207
	v_or_b32_e32 v12, v34, v32                                 // 000000002160: 38184122
	v_mad_co_u64_u32 v[30:31], null, s30, v43, 0               // 000000002164: d6fe7c1e 0202561e
	v_cndmask_b32_e64 v29, 0, v29, s5                          // 00000000216c: d501001d 00163a80
	v_cndmask_b32_e64 v28, 0, v28, s5                          // 000000002174: d501001c 00163880
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]                        // 00000000217c: 3e0c0c82
	v_cmp_gt_i64_e64 s5, s[24:25], v[12:13]                    // 000000002180: d4540005 02021818
	s_delay_alu instid0(valu_dep_3)                            // 000000002188: bf870003
	v_mul_lo_u32 v32, s10, v28                                 // 00000000218c: d72c0020 0202380a
	v_add3_u32 v31, v31, v39, v33                              // 000000002194: d655001f 04864f1f
	v_mul_lo_u32 v33, s30, v29                                 // 00000000219c: d72c0021 02023a1e
	v_mad_co_u64_u32 v[28:29], null, s30, v28, 0               // 0000000021a4: d6fe7c1c 0202381e
	s_wait_alu depctr_va_sdst(0)                               // 0000000021ac: bf88f19f
	v_cndmask_b32_e64 v43, 0, v12, s5                          // 0000000021b0: d501002b 00161880
	v_or_b32_e32 v12, v34, v40                                 // 0000000021b8: 38185122
	v_add_co_u32 v103, s6, s8, v6                              // 0000000021bc: d7000667 02020c08
	s_wait_alu depctr_va_sdst(0)                               // 0000000021c4: bf88f19f
	v_add_co_ci_u32_e64 v104, null, s9, v7, s6                 // 0000000021c8: d5207c68 001a0e09
	v_lshlrev_b64_e32 v[6:7], 2, v[30:31]                      // 0000000021d0: 3e0c3c82
	v_cndmask_b32_e64 v39, 0, v13, s5                          // 0000000021d4: d5010027 00161a80
	v_cmp_gt_i64_e64 s5, s[24:25], v[12:13]                    // 0000000021dc: d4540005 02021818
	v_add3_u32 v29, v29, v33, v32                              // 0000000021e4: d655001d 0482431d
	v_mov_b32_e32 v33, s7                                      // 0000000021ec: 7e420207
	v_or_b32_e32 v32, v34, v41                                 // 0000000021f0: 38405322
	v_add_co_u32 v107, s6, s8, v6                              // 0000000021f4: d700066b 02020c08
	s_wait_alu depctr_va_sdst(0)                               // 0000000021fc: bf88f19f
	v_add_co_ci_u32_e64 v108, null, s9, v7, s6                 // 000000002200: d5207c6c 001a0e09
	v_lshlrev_b64_e32 v[6:7], 2, v[28:29]                      // 000000002208: 3e0c3882
	v_cndmask_b32_e64 v29, 0, v12, s5                          // 00000000220c: d501001d 00161880
	v_or_b32_e32 v12, v34, v42                                 // 000000002214: 38185522
	v_mul_lo_u32 v40, s10, v43                                 // 000000002218: d72c0028 0202560a
	v_mul_lo_u32 v39, s30, v39                                 // 000000002220: d72c0027 02024e1e
	v_mad_co_u64_u32 v[30:31], null, s30, v43, 0               // 000000002228: d6fe7c1e 0202561e
	v_cmp_gt_i64_e64 s6, s[24:25], v[32:33]                    // 000000002230: d4540006 02024018
	v_cndmask_b32_e64 v28, 0, v13, s5                          // 000000002238: d501001c 00161a80
	v_cmp_gt_i64_e64 s5, s[24:25], v[12:13]                    // 000000002240: d4540005 02021818
	v_mul_lo_u32 v34, s10, v29                                 // 000000002248: d72c0022 02023a0a
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0             // 000000002250: ca100080 2b2a0080
	s_wait_alu depctr_va_sdst(0)                               // 000000002258: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s6                          // 00000000225c: d5010021 001a4280
	v_cndmask_b32_e64 v32, 0, v32, s6                          // 000000002264: d5010020 001a4080
	v_add3_u32 v31, v31, v39, v40                              // 00000000226c: d655001f 04a24f1f
	v_cndmask_b32_e64 v13, 0, v13, s5                          // 000000002274: d501000d 00161a80
	v_cndmask_b32_e64 v12, 0, v12, s5                          // 00000000227c: d501000c 00161880
	v_mul_lo_u32 v39, s30, v28                                 // 000000002284: d72c0027 0202381e
	v_mad_co_u64_u32 v[28:29], null, s30, v29, 0               // 00000000228c: d6fe7c1c 02023a1e
	v_mul_lo_u32 v40, s10, v32                                 // 000000002294: d72c0028 0202400a
	v_mul_lo_u32 v41, s30, v33                                 // 00000000229c: d72c0029 0202421e
	v_mad_co_u64_u32 v[32:33], null, s30, v32, 0               // 0000000022a4: d6fe7c20 0202401e
	v_add_co_u32 v109, s5, s8, v6                              // 0000000022ac: d700056d 02020c08
	s_wait_alu depctr_va_sdst(0)                               // 0000000022b4: bf88f19f
	v_add_co_ci_u32_e64 v110, null, s9, v7, s5                 // 0000000022b8: d5207c6e 00160e09
	v_lshlrev_b64_e32 v[6:7], 2, v[30:31]                      // 0000000022c0: 3e0c3c82
	v_mul_lo_u32 v30, s10, v12                                 // 0000000022c4: d72c001e 0202180a
	v_mul_lo_u32 v31, s30, v13                                 // 0000000022cc: d72c001f 02021a1e
	v_mad_co_u64_u32 v[12:13], null, s30, v12, 0               // 0000000022d4: d6fe7c0c 0202181e
	v_add3_u32 v29, v29, v39, v34                              // 0000000022dc: d655001d 048a4f1d
	v_add3_u32 v33, v33, v41, v40                              // 0000000022e4: d6550021 04a25321
	v_add_co_u32 v111, s5, s8, v6                              // 0000000022ec: d700056f 02020c08
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f4: bf88f19f
	v_add_co_ci_u32_e64 v112, null, s9, v7, s5                 // 0000000022f8: d5207c70 00160e09
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]                    // 000000002300: 3e383882
	v_add3_u32 v13, v13, v31, v30                              // 000000002304: d655000d 047a3f0d
	v_lshlrev_b64_e32 v[6:7], 2, v[32:33]                      // 00000000230c: 3e0c4082
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v34, 0             // 000000002310: ca100080 27220080
	v_mov_b32_e32 v33, 0                                       // 000000002318: 7e420280
	s_delay_alu instid0(valu_dep_4)                            // 00000000231c: bf870004
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]                    // 000000002320: 3e181882
	v_add_co_u32 v113, s5, s8, v28                             // 000000002324: d7000571 02023808
	s_wait_alu depctr_va_sdst(0)                               // 00000000232c: bf88f19f
	v_add_co_ci_u32_e64 v114, null, s9, v29, s5                // 000000002330: d5207c72 00163a09
	v_add_co_u32 v115, s5, s8, v6                              // 000000002338: d7000573 02020c08
	s_wait_alu depctr_va_sdst(0)                               // 000000002340: bf88f19f
	v_add_co_ci_u32_e64 v116, null, s9, v7, s5                 // 000000002344: d5207c74 00160e09
	v_add_co_u32 v117, s5, s8, v12                             // 00000000234c: d7000575 02021808
	s_wait_alu depctr_va_sdst(0)                               // 000000002354: bf88f19f
	v_add_co_ci_u32_e64 v118, null, s9, v13, s5                // 000000002358: d5207c76 00161a09
	v_lshlrev_b64_e32 v[6:7], 2, v[8:9]                        // 000000002360: 3e0c1082
	v_lshlrev_b64_e32 v[8:9], 2, v[10:11]                      // 000000002364: 3e101482
	v_lshlrev_b64_e32 v[10:11], 2, v[24:25]                    // 000000002368: 3e143082
	v_lshlrev_b64_e32 v[12:13], 2, v[26:27]                    // 00000000236c: 3e183482
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v41, 0             // 000000002370: ca100080 20280080
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v31, 0             // 000000002378: ca100080 281e0080
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0             // 000000002380: ca100080 1e1c0080
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0             // 000000002388: ca100080 1c1a0080
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0             // 000000002390: ca100080 1a180080
	v_mov_b32_e32 v24, 0                                       // 000000002398: 7e300280
	s_lshl_b64 s[6:7], s[34:35], 5                             // 00000000239c: 84868522
	s_lshl_b64 s[20:21], s[34:35], 2                           // 0000000023a0: 84948222
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023a4: bf88ff9e
	v_add_co_u32 v123, s5, v16, s6                             // 0000000023a8: d700057b 02000d10
	v_add_co_u32 v127, s6, v18, s6                             // 0000000023b0: d700067f 02000d12
	s_wait_alu depctr_va_sdst(0)                               // 0000000023b8: bf88f19f
	v_add_co_ci_u32_e64 v124, null, s7, v17, s5                // 0000000023bc: d5207c7c 00162207
	v_add_co_ci_u32_e64 v128, null, s7, v19, s6                // 0000000023c4: d5207c80 001a2607
	s_mul_u64 s[6:7], s[34:35], s[26:27]                       // 0000000023cc: aa861a22
	global_load_b128 v[123:126], v[123:124], off               // 0000000023d0: ee05c07c 0000007b 0000007b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023dc: bf88ff9e
	s_lshl_b64 s[36:37], s[6:7], 2                             // 0000000023e0: 84a48206
	global_load_b128 v[127:130], v[127:128], off               // 0000000023e4: ee05c07c 0000007f 0000007f
	v_add_co_u32 v131, s5, v77, s20                            // 0000000023f0: d7000583 0200294d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023f8: bf88ff9e
	s_add_nc_u64 s[36:37], s[28:29], s[36:37]                  // 0000000023fc: a9a4241c
	v_add_co_u32 v133, s6, v79, s20                            // 000000002400: d7000685 0200294f
	v_add_co_u32 v135, s7, v83, s20                            // 000000002408: d7000787 02002953
	v_add_co_ci_u32_e64 v132, null, s21, v78, s5               // 000000002410: d5207c84 00169c15
	s_wait_alu depctr_sa_sdst(0)                               // 000000002418: bf88ff9e
	v_add_co_u32 v163, s5, s36, v6                             // 00000000241c: d70005a3 02020c24
	s_wait_alu depctr_va_sdst(0)                               // 000000002424: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s21, v80, s6               // 000000002428: d5207c86 001aa015
	v_add_co_ci_u32_e64 v136, null, s21, v84, s7               // 000000002430: d5207c88 001ea815
	v_add_co_ci_u32_e64 v164, null, s37, v7, s5                // 000000002438: d5207ca4 00160e25
	v_add_co_u32 v137, s8, v87, s20                            // 000000002440: d7000889 02002957
	v_add_co_u32 v139, s9, v89, s20                            // 000000002448: d700098b 02002959
	v_add_co_u32 v141, s10, v91, s20                           // 000000002450: d7000a8d 0200295b
	v_add_co_u32 v143, s11, v93, s20                           // 000000002458: d7000b8f 0200295d
	v_add_co_u32 v145, s12, v96, s20                           // 000000002460: d7000c91 02002960
	v_add_co_u32 v165, s6, s36, v8                             // 000000002468: d70006a5 02021024
	s_wait_alu depctr_va_sdst(0)                               // 000000002470: bf88f19f
	v_add_co_ci_u32_e64 v138, null, s21, v88, s8               // 000000002474: d5207c8a 0022b015
	v_add_co_ci_u32_e64 v140, null, s21, v90, s9               // 00000000247c: d5207c8c 0026b415
	v_add_co_ci_u32_e64 v142, null, s21, v92, s10              // 000000002484: d5207c8e 002ab815
	v_add_co_ci_u32_e64 v144, null, s21, v94, s11              // 00000000248c: d5207c90 002ebc15
	v_add_co_ci_u32_e64 v146, null, s21, v97, s12              // 000000002494: d5207c92 0032c215
	v_add_co_ci_u32_e64 v166, null, s37, v9, s6                // 00000000249c: d5207ca6 001a1225
	s_barrier_signal -1                                        // 0000000024a4: be804ec1
	s_barrier_wait 0xffff                                      // 0000000024a8: bf94ffff
	v_add_co_u32 v167, s7, s36, v10                            // 0000000024ac: d70007a7 02021424
	v_add_co_u32 v169, s8, s36, v12                            // 0000000024b4: d70008a9 02021824
	s_wait_alu depctr_va_sdst(0)                               // 0000000024bc: bf88f19f
	v_add_co_ci_u32_e64 v168, null, s37, v11, s7               // 0000000024c0: d5207ca8 001e1625
	v_add_co_ci_u32_e64 v170, null, s37, v13, s8               // 0000000024c8: d5207caa 00221a25
	v_add_co_u32 v147, s13, v101, s20                          // 0000000024d0: d7000d93 02002965
	v_add_co_u32 v149, s14, v103, s20                          // 0000000024d8: d7000e95 02002967
	v_add_co_u32 v151, s15, v107, s20                          // 0000000024e0: d7000f97 0200296b
	v_add_co_u32 v153, s16, v109, s20                          // 0000000024e8: d7001099 0200296d
	v_add_co_u32 v155, s17, v111, s20                          // 0000000024f0: d700119b 0200296f
	v_add_co_u32 v157, s18, v113, s20                          // 0000000024f8: d700129d 02002971
	v_add_co_u32 v159, s19, v115, s20                          // 000000002500: d700139f 02002973
	v_add_co_u32 v161, s20, v117, s20                          // 000000002508: d70014a1 02002975
	s_wait_alu depctr_va_sdst(0)                               // 000000002510: bf88f19f
	v_add_co_ci_u32_e64 v148, null, s21, v102, s13             // 000000002514: d5207c94 0036cc15
	v_add_co_ci_u32_e64 v150, null, s21, v104, s14             // 00000000251c: d5207c96 003ad015
	v_add_co_ci_u32_e64 v152, null, s21, v108, s15             // 000000002524: d5207c98 003ed815
	v_add_co_ci_u32_e64 v154, null, s21, v110, s16             // 00000000252c: d5207c9a 0042dc15
	v_add_co_ci_u32_e64 v156, null, s21, v112, s17             // 000000002534: d5207c9c 0046e015
	v_add_co_ci_u32_e64 v158, null, s21, v114, s18             // 00000000253c: d5207c9e 004ae415
	v_add_co_ci_u32_e64 v160, null, s21, v116, s19             // 000000002544: d5207ca0 004ee815
	v_add_co_ci_u32_e64 v162, null, s21, v118, s20             // 00000000254c: d5207ca2 0052ec15
	s_add_nc_u64 s[34:35], s[34:35], 1                         // 000000002554: a9a28122
	s_delay_alu instid0(salu_cycle_1)                          // 000000002558: bf870009
	s_cmp_lg_u64 s[34:35], s[30:31]                            // 00000000255c: bf111e22
	s_wait_loadcnt 0x1                                         // 000000002560: bfc00001
	ds_store_b128 v15, v[123:126]                              // 000000002564: db7c0000 00007b0f
	s_wait_loadcnt 0x0                                         // 00000000256c: bfc00000
	ds_store_b128 v15, v[127:130] offset:6144                  // 000000002570: db7c1800 00007f0f
	s_wait_dscnt 0x0                                           // 000000002578: bfc60000
	s_barrier_signal -1                                        // 00000000257c: be804ec1
	s_barrier_wait 0xffff                                      // 000000002580: bf94ffff
	global_load_b32 v197, v[131:132], off                      // 000000002584: ee05007c 000000c5 00000083
	global_load_b32 v198, v[163:164], off                      // 000000002590: ee05007c 000000c6 000000a3
	s_clause 0x6                                               // 00000000259c: bf850006
	global_load_b32 v199, v[133:134], off                      // 0000000025a0: ee05007c 000000c7 00000085
	global_load_b32 v200, v[135:136], off                      // 0000000025ac: ee05007c 000000c8 00000087
	global_load_b32 v201, v[137:138], off                      // 0000000025b8: ee05007c 000000c9 00000089
	global_load_b32 v202, v[139:140], off                      // 0000000025c4: ee05007c 000000ca 0000008b
	global_load_b32 v203, v[141:142], off                      // 0000000025d0: ee05007c 000000cb 0000008d
	global_load_b32 v204, v[143:144], off                      // 0000000025dc: ee05007c 000000cc 0000008f
	global_load_b32 v205, v[145:146], off                      // 0000000025e8: ee05007c 000000cd 00000091
	s_clause 0x2                                               // 0000000025f4: bf850002
	global_load_b32 v206, v[165:166], off                      // 0000000025f8: ee05007c 000000ce 000000a5
	global_load_b32 v207, v[167:168], off                      // 000000002604: ee05007c 000000cf 000000a7
	global_load_b32 v208, v[169:170], off                      // 000000002610: ee05007c 000000d0 000000a9
	s_clause 0x7                                               // 00000000261c: bf850007
	global_load_b32 v209, v[147:148], off                      // 000000002620: ee05007c 000000d1 00000093
	global_load_b32 v210, v[149:150], off                      // 00000000262c: ee05007c 000000d2 00000095
	global_load_b32 v211, v[151:152], off                      // 000000002638: ee05007c 000000d3 00000097
	global_load_b32 v212, v[153:154], off                      // 000000002644: ee05007c 000000d4 00000099
	global_load_b32 v213, v[155:156], off                      // 000000002650: ee05007c 000000d5 0000009b
	global_load_b32 v214, v[157:158], off                      // 00000000265c: ee05007c 000000d6 0000009d
	global_load_b32 v215, v[159:160], off                      // 000000002668: ee05007c 000000d7 0000009f
	global_load_b32 v216, v[161:162], off                      // 000000002674: ee05007c 000000d8 000000a1
	ds_load_2addr_b64 v[169:172], v20 offset1:2                // 000000002680: d9dc0200 a9000014
	ds_load_2addr_b64 v[177:180], v119 offset1:2               // 000000002688: d9dc0200 b1000077
	ds_load_2addr_b64 v[181:184], v120 offset1:2               // 000000002690: d9dc0200 b5000078
	ds_load_2addr_b64 v[185:188], v121 offset1:2               // 000000002698: d9dc0200 b9000079
	ds_load_2addr_b64 v[189:192], v122 offset1:2               // 0000000026a0: d9dc0200 bd00007a
	ds_load_2addr_b64 v[193:196], v21 offset1:2                // 0000000026a8: d9dc0200 c1000015
	s_wait_dscnt 0x4                                           // 0000000026b0: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[123:130], v[169:170], v[177:178], 0// 0000000026b4: cc46407b 1a0363a9
	s_wait_dscnt 0x3                                           // 0000000026bc: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[131:138], v[169:170], v[181:182], 0// 0000000026c0: cc464083 1a036ba9
	s_wait_dscnt 0x2                                           // 0000000026c8: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[139:146], v[169:170], v[185:186], 0// 0000000026cc: cc46408b 1a0373a9
	s_wait_dscnt 0x1                                           // 0000000026d4: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[147:154], v[169:170], v[189:190], 0// 0000000026d8: cc464093 1a037ba9
	s_wait_dscnt 0x0                                           // 0000000026e0: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[155:162], v[193:194], v[177:178], 0// 0000000026e4: cc46409b 1a0363c1
	v_wmma_f32_16x16x16_fp8_fp8 v[163:170], v[193:194], v[181:182], 0// 0000000026ec: cc4640a3 1a036bc1
	v_wmma_f32_16x16x16_fp8_fp8 v[123:130], v[171:172], v[179:180], v[123:130]// 0000000026f4: cc46407b 1def67ab
	v_wmma_f32_16x16x16_fp8_fp8 v[131:138], v[171:172], v[183:184], v[131:138]// 0000000026fc: cc464083 1e0f6fab
	v_wmma_f32_16x16x16_fp8_fp8 v[139:146], v[171:172], v[187:188], v[139:146]// 000000002704: cc46408b 1e2f77ab
	v_wmma_f32_16x16x16_fp8_fp8 v[147:154], v[171:172], v[191:192], v[147:154]// 00000000270c: cc464093 1e4f7fab
	v_wmma_f32_16x16x16_fp8_fp8 v[171:178], v[193:194], v[185:186], 0// 000000002714: cc4640ab 1a0373c1
	v_wmma_f32_16x16x16_fp8_fp8 v[155:162], v[195:196], v[179:180], v[155:162]// 00000000271c: cc46409b 1e6f67c3
	v_wmma_f32_16x16x16_fp8_fp8 v[163:170], v[195:196], v[183:184], v[163:170]// 000000002724: cc4640a3 1e8f6fc3
	v_wmma_f32_16x16x16_fp8_fp8 v[179:186], v[193:194], v[189:190], 0// 00000000272c: cc4640b3 1a037bc1
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_2)// 000000002734: bf870114
	v_wmma_f32_16x16x16_fp8_fp8 v[171:178], v[195:196], v[187:188], v[171:178]// 000000002738: cc4640ab 1eaf77c3
	v_wmma_f32_16x16x16_fp8_fp8 v[179:186], v[195:196], v[191:192], v[179:186]// 000000002740: cc4640b3 1ecf7fc3
	s_wait_loadcnt 0x11                                        // 000000002748: bfc00011
	v_dual_mul_f32 v187, v197, v198 :: v_dual_mul_f32 v188, v198, v199// 00000000274c: c8c78dc5 bbbd8fc6
	s_wait_loadcnt 0xf                                         // 000000002754: bfc0000f
	v_dual_mul_f32 v189, v198, v200 :: v_dual_mul_f32 v190, v198, v201// 000000002758: c8c791c6 bdbf93c6
	s_wait_loadcnt 0xd                                         // 000000002760: bfc0000d
	v_dual_mul_f32 v191, v198, v202 :: v_dual_mul_f32 v192, v198, v203// 000000002764: c8c795c6 bfc197c6
	s_wait_loadcnt 0xb                                         // 00000000276c: bfc0000b
	v_dual_mul_f32 v193, v198, v204 :: v_dual_mul_f32 v194, v198, v205// 000000002770: c8c799c6 c1c39bc6
	s_wait_loadcnt 0xa                                         // 000000002778: bfc0000a
	v_dual_mul_f32 v195, v197, v206 :: v_dual_mul_f32 v196, v199, v206// 00000000277c: c8c79dc5 c3c59dc7
	v_dual_mul_f32 v123, v123, v187 :: v_dual_mul_f32 v124, v124, v188// 000000002784: c8c7777b 7b7d797c
	v_mul_f32_e32 v125, v125, v189                             // 00000000278c: 10fb7b7d
	v_dual_mul_f32 v187, v203, v206 :: v_dual_mul_f32 v188, v204, v206// 000000002790: c8c79dcb bbbd9dcc
	v_mul_f32_e32 v189, v205, v206                             // 000000002798: 117b9dcd
	v_dual_mul_f32 v217, v200, v206 :: v_dual_mul_f32 v218, v201, v206// 00000000279c: c8c79dc8 d9db9dc9
	v_mul_f32_e32 v219, v202, v206                             // 0000000027a4: 11b79dca
	v_dual_mul_f32 v126, v126, v190 :: v_dual_mul_f32 v127, v127, v191// 0000000027a8: c8c77d7e 7e7f7f7f
	v_dual_mul_f32 v128, v128, v192 :: v_dual_mul_f32 v129, v129, v193// 0000000027b0: c8c78180 80818381
	v_mul_f32_e32 v130, v130, v194                             // 0000000027b8: 11058582
	s_wait_loadcnt 0x9                                         // 0000000027bc: bfc00009
	v_dual_mul_f32 v190, v197, v207 :: v_dual_mul_f32 v191, v199, v207// 0000000027c0: c8c79fc5 bebf9fc7
	v_dual_mul_f32 v192, v200, v207 :: v_dual_mul_f32 v193, v201, v207// 0000000027c8: c8c79fc8 c0c19fc9
	v_mul_f32_e32 v194, v202, v207                             // 0000000027d0: 11859fca
	v_dual_mul_f32 v131, v131, v195 :: v_dual_mul_f32 v132, v132, v196// 0000000027d4: c8c78783 83858984
	v_dual_mul_f32 v136, v136, v187 :: v_dual_mul_f32 v137, v137, v188// 0000000027dc: c8c77788 88897989
	v_dual_mul_f32 v138, v138, v189 :: v_dual_mul_f32 v187, v203, v207// 0000000027e4: c8c77b8a 8abb9fcb
	v_dual_mul_f32 v188, v204, v207 :: v_dual_mul_f32 v189, v205, v207// 0000000027ec: c8c79fcc bcbd9fcd
	s_wait_loadcnt 0x8                                         // 0000000027f4: bfc00008
	v_dual_mul_f32 v195, v197, v208 :: v_dual_mul_f32 v196, v199, v208// 0000000027f8: c8c7a1c5 c3c5a1c7
	v_mul_f32_e32 v197, v200, v208                             // 000000002800: 118ba1c8
	v_dual_mul_f32 v199, v201, v208 :: v_dual_mul_f32 v200, v202, v208// 000000002804: c8c7a1c9 c7c9a1ca
	v_dual_mul_f32 v201, v203, v208 :: v_dual_mul_f32 v202, v204, v208// 00000000280c: c8c7a1cb c9cba1cc
	v_mul_f32_e32 v203, v205, v208                             // 000000002814: 1197a1cd
	v_dual_mul_f32 v133, v133, v217 :: v_dual_mul_f32 v134, v134, v218// 000000002818: c8c7b385 8587b586
	s_wait_loadcnt 0x7                                         // 000000002820: bfc00007
	v_dual_mul_f32 v135, v135, v219 :: v_dual_mul_f32 v204, v198, v209// 000000002824: c8c7b787 87cda3c6
	s_wait_loadcnt 0x6                                         // 00000000282c: bfc00006
	v_mul_f32_e32 v205, v198, v210                             // 000000002830: 119ba5c6
	s_wait_loadcnt 0x4                                         // 000000002834: bfc00004
	v_dual_mul_f32 v217, v198, v211 :: v_dual_mul_f32 v218, v198, v212// 000000002838: c8c7a7c6 d9dba9c6
	s_wait_loadcnt 0x3                                         // 000000002840: bfc00003
	v_mul_f32_e32 v219, v198, v213                             // 000000002844: 11b7abc6
	v_dual_mul_f32 v139, v139, v190 :: v_dual_mul_f32 v140, v140, v191// 000000002848: c8c77d8b 8b8d7f8c
	v_dual_mul_f32 v141, v141, v192 :: v_dual_mul_f32 v142, v142, v193// 000000002850: c8c7818d 8d8f838e
	v_dual_mul_f32 v143, v143, v194 :: v_dual_mul_f32 v144, v144, v187// 000000002858: c8c7858f 8f917790
	v_dual_mul_f32 v145, v145, v188 :: v_dual_mul_f32 v146, v146, v189// 000000002860: c8c77991 91937b92
	s_wait_loadcnt 0x1                                         // 000000002868: bfc00001
	v_dual_mul_f32 v187, v198, v214 :: v_dual_mul_f32 v188, v198, v215// 00000000286c: c8c7adc6 bbbdafc6
	s_wait_loadcnt 0x0                                         // 000000002874: bfc00000
	v_mul_f32_e32 v189, v198, v216                             // 000000002878: 117bb1c6
	v_dual_mul_f32 v190, v206, v209 :: v_dual_mul_f32 v191, v206, v210// 00000000287c: c8c7a3ce bebfa5ce
	v_dual_mul_f32 v192, v206, v211 :: v_dual_mul_f32 v193, v206, v212// 000000002884: c8c7a7ce c0c1a9ce
	v_mul_f32_e32 v194, v206, v213                             // 00000000288c: 1185abce
	v_dual_mul_f32 v198, v206, v214 :: v_dual_mul_f32 v147, v147, v195// 000000002890: c8c7adce c6938793
	v_dual_mul_f32 v148, v148, v196 :: v_dual_mul_f32 v149, v149, v197// 000000002898: c8c78994 94958b95
	v_dual_mul_f32 v150, v150, v199 :: v_dual_mul_f32 v151, v151, v200// 0000000028a0: c8c78f96 96979197
	v_dual_mul_f32 v152, v152, v201 :: v_dual_mul_f32 v153, v153, v202// 0000000028a8: c8c79398 98999599
	v_mul_f32_e32 v154, v154, v203                             // 0000000028b0: 1135979a
	v_dual_mul_f32 v195, v206, v215 :: v_dual_mul_f32 v196, v206, v216// 0000000028b4: c8c7afce c3c5b1ce
	v_mul_f32_e32 v197, v207, v209                             // 0000000028bc: 118ba3cf
	v_dual_mul_f32 v199, v207, v210 :: v_dual_mul_f32 v200, v207, v211// 0000000028c0: c8c7a5cf c7c9a7cf
	v_dual_mul_f32 v201, v207, v212 :: v_dual_mul_f32 v202, v207, v213// 0000000028c8: c8c7a9cf c9cbabcf
	v_dual_mul_f32 v203, v207, v214 :: v_dual_mul_f32 v206, v207, v215// 0000000028d0: c8c7adcf cbcfafcf
	v_mul_f32_e32 v207, v207, v216                             // 0000000028d8: 119fb1cf
	v_dual_mul_f32 v209, v208, v209 :: v_dual_mul_f32 v210, v208, v210// 0000000028dc: c8c7a3d0 d1d3a5d0
	v_dual_mul_f32 v211, v208, v211 :: v_dual_mul_f32 v212, v208, v212// 0000000028e4: c8c7a7d0 d3d5a9d0
	v_dual_mul_f32 v213, v208, v213 :: v_dual_mul_f32 v214, v208, v214// 0000000028ec: c8c7abd0 d5d7add0
	v_dual_mul_f32 v215, v208, v215 :: v_dual_mul_f32 v208, v208, v216// 0000000028f4: c8c7afd0 d7d1b1d0
	v_dual_mul_f32 v155, v155, v204 :: v_dual_mul_f32 v156, v156, v205// 0000000028fc: c8c7999b 9b9d9b9c
	v_dual_mul_f32 v157, v157, v217 :: v_dual_mul_f32 v158, v158, v218// 000000002904: c8c7b39d 9d9fb59e
	v_mul_f32_e32 v159, v159, v219                             // 00000000290c: 113fb79f
	v_dual_mul_f32 v160, v160, v187 :: v_dual_mul_f32 v161, v161, v188// 000000002910: c8c777a0 a0a179a1
	v_dual_mul_f32 v162, v162, v189 :: v_dual_mul_f32 v163, v163, v190// 000000002918: c8c77ba2 a2a37da3
	v_dual_mul_f32 v164, v164, v191 :: v_dual_mul_f32 v165, v165, v192// 000000002920: c8c77fa4 a4a581a5
	v_dual_mul_f32 v166, v166, v193 :: v_dual_mul_f32 v167, v167, v194// 000000002928: c8c783a6 a6a785a7
	v_dual_mul_f32 v168, v168, v198 :: v_dual_mul_f32 v169, v169, v195// 000000002930: c8c78da8 a8a987a9
	v_dual_mul_f32 v170, v170, v196 :: v_dual_mul_f32 v171, v171, v197// 000000002938: c8c789aa aaab8bab
	v_dual_mul_f32 v172, v172, v199 :: v_dual_mul_f32 v173, v173, v200// 000000002940: c8c78fac acad91ad
	v_dual_mul_f32 v174, v174, v201 :: v_dual_mul_f32 v175, v175, v202// 000000002948: c8c793ae aeaf95af
	v_dual_mul_f32 v176, v176, v203 :: v_dual_mul_f32 v177, v177, v206// 000000002950: c8c797b0 b0b19db1
	v_dual_mul_f32 v178, v178, v207 :: v_dual_mul_f32 v179, v179, v209// 000000002958: c8c79fb2 b2b3a3b3
	v_dual_mul_f32 v180, v180, v210 :: v_dual_mul_f32 v181, v181, v211// 000000002960: c8c7a5b4 b4b5a7b5
	v_dual_mul_f32 v182, v182, v212 :: v_dual_mul_f32 v183, v183, v213// 000000002968: c8c7a9b6 b6b7abb7
	v_dual_mul_f32 v184, v184, v214 :: v_dual_mul_f32 v185, v185, v215// 000000002970: c8c7adb8 b8b9afb9
	v_mul_f32_e32 v186, v186, v208                             // 000000002978: 1175a1ba
	v_add_f32_e32 v14, v14, v123                               // 00000000297c: 061cf70e
	v_dual_add_f32 v106, v106, v124 :: v_dual_add_f32 v105, v105, v125// 000000002980: c908f96a 6a68fb69
	v_dual_add_f32 v100, v100, v126 :: v_dual_add_f32 v99, v99, v127// 000000002988: c908fd64 6462ff63
	v_dual_add_f32 v98, v98, v128 :: v_dual_add_f32 v95, v95, v129// 000000002990: c9090162 625f035f
	v_dual_add_f32 v86, v86, v130 :: v_dual_add_f32 v71, v71, v131// 000000002998: c9090556 56470747
	v_dual_add_f32 v70, v70, v132 :: v_dual_add_f32 v69, v69, v133// 0000000029a0: c9090946 46450b45
	v_dual_add_f32 v68, v68, v134 :: v_dual_add_f32 v67, v67, v135// 0000000029a8: c9090d44 44430f43
	v_dual_add_f32 v66, v66, v136 :: v_dual_add_f32 v65, v65, v137// 0000000029b0: c9091142 42411341
	v_dual_add_f32 v64, v64, v138 :: v_dual_add_f32 v55, v55, v139// 0000000029b8: c9091540 40371737
	v_dual_add_f32 v54, v54, v140 :: v_dual_add_f32 v53, v53, v141// 0000000029c0: c9091936 36351b35
	v_dual_add_f32 v52, v52, v142 :: v_dual_add_f32 v51, v51, v143// 0000000029c8: c9091d34 34331f33
	v_dual_add_f32 v50, v50, v144 :: v_dual_add_f32 v49, v49, v145// 0000000029d0: c9092132 32312331
	v_dual_add_f32 v48, v48, v146 :: v_dual_add_f32 v39, v39, v147// 0000000029d8: c9092530 30272727
	v_dual_add_f32 v38, v38, v148 :: v_dual_add_f32 v37, v37, v149// 0000000029e0: c9092926 26252b25
	v_dual_add_f32 v36, v36, v150 :: v_dual_add_f32 v35, v35, v151// 0000000029e8: c9092d24 24232f23
	v_dual_add_f32 v34, v34, v152 :: v_dual_add_f32 v33, v33, v153// 0000000029f0: c9093122 22213321
	v_dual_add_f32 v32, v32, v154 :: v_dual_add_f32 v85, v85, v155// 0000000029f8: c9093520 20553755
	v_dual_add_f32 v82, v82, v156 :: v_dual_add_f32 v81, v81, v157// 000000002a00: c9093952 52513b51
	v_dual_add_f32 v76, v76, v158 :: v_dual_add_f32 v75, v75, v159// 000000002a08: c9093d4c 4c4b3f4b
	v_dual_add_f32 v74, v74, v160 :: v_dual_add_f32 v73, v73, v161// 000000002a10: c909414a 4a494349
	v_dual_add_f32 v72, v72, v162 :: v_dual_add_f32 v63, v63, v163// 000000002a18: c9094548 483f473f
	v_dual_add_f32 v62, v62, v164 :: v_dual_add_f32 v61, v61, v165// 000000002a20: c909493e 3e3d4b3d
	v_dual_add_f32 v60, v60, v166 :: v_dual_add_f32 v59, v59, v167// 000000002a28: c9094d3c 3c3b4f3b
	v_dual_add_f32 v58, v58, v168 :: v_dual_add_f32 v57, v57, v169// 000000002a30: c909513a 3a395339
	v_dual_add_f32 v56, v56, v170 :: v_dual_add_f32 v47, v47, v171// 000000002a38: c9095538 382f572f
	v_dual_add_f32 v46, v46, v172 :: v_dual_add_f32 v45, v45, v173// 000000002a40: c909592e 2e2d5b2d
	v_dual_add_f32 v44, v44, v174 :: v_dual_add_f32 v43, v43, v175// 000000002a48: c9095d2c 2c2b5f2b
	v_dual_add_f32 v42, v42, v176 :: v_dual_add_f32 v41, v41, v177// 000000002a50: c909612a 2a296329
	v_dual_add_f32 v40, v40, v178 :: v_dual_add_f32 v31, v31, v179// 000000002a58: c9096528 281f671f
	v_dual_add_f32 v30, v30, v180 :: v_dual_add_f32 v29, v29, v181// 000000002a60: c909691e 1e1d6b1d
	v_dual_add_f32 v28, v28, v182 :: v_dual_add_f32 v27, v27, v183// 000000002a68: c9096d1c 1c1b6f1b
	v_dual_add_f32 v26, v26, v184 :: v_dual_add_f32 v25, v25, v185// 000000002a70: c909711a 1a197319
	v_add_f32_e32 v24, v24, v186                               // 000000002a78: 06317518
	s_cbranch_scc1 65095                                       // 000000002a7c: bfa2fe47 <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x89c>
	s_load_b64 s[20:21], s[0:1], 0xa8                          // 000000002a80: f4002500 f80000a8
	v_mul_lo_u32 v8, s27, v2                                   // 000000002a88: d72c0008 0202041b
	v_mul_lo_u32 v9, s26, v3                                   // 000000002a90: d72c0009 0202061a
	v_mad_co_u64_u32 v[6:7], null, s26, v2, 0                  // 000000002a98: d6fe7c06 0202041a
	v_sub_co_u32 v20, s0, s24, v2                              // 000000002aa0: d7010014 02020418
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002aa8: bf870191
	v_sub_co_ci_u32_e64 v21, null, s25, v3, s0                 // 000000002aac: d5217c15 00020619
	v_add3_u32 v7, v7, v9, v8                                  // 000000002ab4: d6550007 04221307
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002abc: bf870112
	v_cmp_lt_i64_e64 s17, 0, v[20:21]                          // 000000002ac0: d4510011 02022880
	v_lshlrev_b64_e32 v[2:3], 1, v[6:7]                        // 000000002ac8: 3e040c81
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]                        // 000000002acc: 3e0c0081
	s_and_b32 s0, s17, s4                                      // 000000002ad0: 8b000411
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ad4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002ad8: be812000
	s_cbranch_execz 28                                         // 000000002adc: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1050>
	v_bfe_u32 v8, v14, 16, 1                                   // 000000002ae0: d6100008 0205210e
	s_wait_kmcnt 0x0                                           // 000000002ae8: bfc70000
	v_add_co_u32 v9, s0, s20, v2                               // 000000002aec: d7000009 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002af4: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s21, v3, s0                 // 000000002af8: d5207c0a 00020615
	v_add3_u32 v11, v8, v14, 0x7fff                            // 000000002b00: d655000b 03fe1d08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002b0c: bf870003
	v_add_co_u32 v8, s0, v9, v6                                // 000000002b10: d7000008 02020d09
	v_or_b32_e32 v12, 0x400000, v14                            // 000000002b18: 38181cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002b20: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v7, s0                  // 000000002b24: d5207c09 00020f0a
	v_cmp_u_f32_e64 s0, v14, v14                               // 000000002b2c: d4180000 02021d0e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b34: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002b38: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s0                        // 000000002b3c: d501000a 0002190b
	global_store_d16_hi_b16 v[8:9], v10, off                   // 000000002b44: ee09407c 05000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002b54: 8c7e017e
	v_add_co_u32 v8, s0, s26, v0                               // 000000002b58: d7000008 0202001a
	s_wait_alu depctr_va_sdst(0)                               // 000000002b60: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s27, v1, s0                  // 000000002b64: d5207c09 0002021b
	v_cmp_lt_i64_e64 s18, 1, v[20:21]                          // 000000002b6c: d4510012 02022881
	s_delay_alu instid0(valu_dep_2)                            // 000000002b74: bf870002
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000002b78: 3e101081
	s_and_b32 s0, s18, s4                                      // 000000002b7c: 8b000412
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b80: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002b84: be812000
	s_cbranch_execz 28                                         // 000000002b88: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x10fc>
	v_bfe_u32 v10, v106, 16, 1                                 // 000000002b8c: d610000a 0205216a
	s_wait_kmcnt 0x0                                           // 000000002b94: bfc70000
	v_add_co_u32 v11, s0, s20, v2                              // 000000002b98: d700000b 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002ba0: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s21, v3, s0                 // 000000002ba4: d5207c0c 00020615
	v_add3_u32 v13, v10, v106, 0x7fff                          // 000000002bac: d655000d 03fed50a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002bb8: bf870003
	v_add_co_u32 v10, s0, v11, v8                              // 000000002bbc: d700000a 0202110b
	v_or_b32_e32 v14, 0x400000, v106                           // 000000002bc4: 381cd4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002bcc: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s0                 // 000000002bd0: d5207c0b 0002130c
	v_cmp_u_f32_e64 s0, v106, v106                             // 000000002bd8: d4180000 0202d56a
	s_wait_alu depctr_va_sdst(0)                               // 000000002be0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002be4: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 000000002be8: d501000c 00021d0d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 000000002bf0: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bfc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002c00: 8c7e017e
	s_lshl_b64 s[40:41], s[26:27], 1                           // 000000002c04: 84a8811a
	v_cmp_lt_i64_e64 s16, 2, v[20:21]                          // 000000002c08: d4510010 02022882
	v_add_co_u32 v10, s0, s40, v0                              // 000000002c10: d700000a 02020028
	s_wait_alu depctr_va_sdst(0)                               // 000000002c18: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s41, v1, s0                 // 000000002c1c: d5207c0b 00020229
	s_and_b32 s0, s16, s4                                      // 000000002c24: 8b000410
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 000000002c28: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c2c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002c30: be812000
	s_cbranch_execz 28                                         // 000000002c34: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x11a8>
	v_bfe_u32 v12, v105, 16, 1                                 // 000000002c38: d610000c 02052169
	s_wait_kmcnt 0x0                                           // 000000002c40: bfc70000
	v_add_co_u32 v13, s0, s20, v2                              // 000000002c44: d700000d 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002c4c: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s21, v3, s0                 // 000000002c50: d5207c0e 00020615
	v_add3_u32 v15, v12, v105, 0x7fff                          // 000000002c58: d655000f 03fed30c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c64: bf870003
	v_add_co_u32 v12, s0, v13, v10                             // 000000002c68: d700000c 0202150d
	v_or_b32_e32 v16, 0x400000, v105                           // 000000002c70: 3820d2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002c78: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v11, s0                // 000000002c7c: d5207c0d 0002170e
	v_cmp_u_f32_e64 s0, v105, v105                             // 000000002c84: d4180000 0202d369
	s_wait_alu depctr_va_sdst(0)                               // 000000002c8c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002c90: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 000000002c94: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 000000002c9c: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ca8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002cac: 8c7e017e
	s_mul_u64 s[38:39], s[26:27], 3                            // 000000002cb0: aaa6831a
	v_cmp_lt_i64_e64 s15, 3, v[20:21]                          // 000000002cb4: d451000f 02022883
	v_add_co_u32 v12, s0, s38, v0                              // 000000002cbc: d700000c 02020026
	s_wait_alu depctr_va_sdst(0)                               // 000000002cc4: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s39, v1, s0                 // 000000002cc8: d5207c0d 00020227
	s_and_b32 s0, s15, s4                                      // 000000002cd0: 8b00040f
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000002cd4: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cd8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002cdc: be812000
	s_cbranch_execz 28                                         // 000000002ce0: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1254>
	v_bfe_u32 v14, v100, 16, 1                                 // 000000002ce4: d610000e 02052164
	s_wait_kmcnt 0x0                                           // 000000002cec: bfc70000
	v_add_co_u32 v15, s0, s20, v2                              // 000000002cf0: d700000f 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002cf8: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s21, v3, s0                 // 000000002cfc: d5207c10 00020615
	v_add3_u32 v17, v14, v100, 0x7fff                          // 000000002d04: d6550011 03fec90e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d10: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 000000002d14: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v100                           // 000000002d1c: 3824c8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d24: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 000000002d28: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v100, v100                             // 000000002d30: d4180000 0202c964
	s_wait_alu depctr_va_sdst(0)                               // 000000002d38: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d3c: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 000000002d40: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 000000002d48: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d54: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002d58: 8c7e017e
	s_lshl_b64 s[36:37], s[26:27], 2                           // 000000002d5c: 84a4821a
	v_cmp_lt_i64_e64 s14, 4, v[20:21]                          // 000000002d60: d451000e 02022884
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d68: bf88ff9e
	v_add_co_u32 v14, s0, s36, v0                              // 000000002d6c: d700000e 02020024
	s_wait_alu depctr_va_sdst(0)                               // 000000002d74: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s37, v1, s0                 // 000000002d78: d5207c0f 00020225
	s_and_b32 s0, s14, s4                                      // 000000002d80: 8b00040e
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000002d84: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d88: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002d8c: be812000
	s_cbranch_execz 28                                         // 000000002d90: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1304>
	v_bfe_u32 v16, v99, 16, 1                                  // 000000002d94: d6100010 02052163
	s_wait_kmcnt 0x0                                           // 000000002d9c: bfc70000
	v_add_co_u32 v17, s0, s20, v2                              // 000000002da0: d7000011 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002da8: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s21, v3, s0                 // 000000002dac: d5207c12 00020615
	v_add3_u32 v19, v16, v99, 0x7fff                           // 000000002db4: d6550013 03fec710 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002dc0: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 000000002dc4: d7000010 02021d11
	v_or_b32_e32 v77, 0x400000, v99                            // 000000002dcc: 389ac6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002dd4: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 000000002dd8: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v99, v99                               // 000000002de0: d4180000 0202c763
	s_wait_alu depctr_va_sdst(0)                               // 000000002de8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002dec: bf870001
	v_cndmask_b32_e64 v18, v19, v77, s0                        // 000000002df0: d5010012 00029b13
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000002df8: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002e08: 8c7e017e
	s_mul_u64 s[34:35], s[26:27], 5                            // 000000002e0c: aaa2851a
	v_cmp_lt_i64_e64 s13, 5, v[20:21]                          // 000000002e10: d451000d 02022885
	v_add_co_u32 v16, s0, s34, v0                              // 000000002e18: d7000010 02020022
	s_wait_alu depctr_va_sdst(0)                               // 000000002e20: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s35, v1, s0                 // 000000002e24: d5207c11 00020223
	s_and_b32 s0, s13, s4                                      // 000000002e2c: 8b00040d
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000002e30: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e34: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002e38: be812000
	s_cbranch_execz 28                                         // 000000002e3c: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x13b0>
	v_bfe_u32 v18, v98, 16, 1                                  // 000000002e40: d6100012 02052162
	s_wait_kmcnt 0x0                                           // 000000002e48: bfc70000
	v_add_co_u32 v19, s0, s20, v2                              // 000000002e4c: d7000013 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002e54: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s21, v3, s0                 // 000000002e58: d5207c4d 00020615
	v_add3_u32 v78, v18, v98, 0x7fff                           // 000000002e60: d655004e 03fec512 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e6c: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 000000002e70: d7000012 02022113
	v_or_b32_e32 v79, 0x400000, v98                            // 000000002e78: 389ec4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e80: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v77, v17, s0                // 000000002e84: d5207c13 0002234d
	v_cmp_u_f32_e64 s0, v98, v98                               // 000000002e8c: d4180000 0202c562
	s_wait_alu depctr_va_sdst(0)                               // 000000002e94: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e98: bf870001
	v_cndmask_b32_e64 v77, v78, v79, s0                        // 000000002e9c: d501004d 00029f4e
	global_store_d16_hi_b16 v[18:19], v77, off                 // 000000002ea4: ee09407c 26800000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 000000002eb0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002eb4: 8c7e017e
	s_mul_u64 s[30:31], s[26:27], 6                            // 000000002eb8: aa9e861a
	v_cmp_lt_i64_e64 s11, 6, v[20:21]                          // 000000002ebc: d451000b 02022886
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ec4: bf88ff9e
	v_add_co_u32 v18, s0, s30, v0                              // 000000002ec8: d7000012 0202001e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ed0: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s31, v1, s0                 // 000000002ed4: d5207c13 0002021f
	s_and_b32 s0, s11, s4                                      // 000000002edc: 8b00040b
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 000000002ee0: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ee4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002ee8: be812000
	s_cbranch_execz 28                                         // 000000002eec: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1460>
	v_bfe_u32 v77, v95, 16, 1                                  // 000000002ef0: d610004d 0205215f
	s_wait_kmcnt 0x0                                           // 000000002ef8: bfc70000
	v_add_co_u32 v78, s0, s20, v2                              // 000000002efc: d700004e 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002f04: bf88f19f
	v_add_co_ci_u32_e64 v79, null, s21, v3, s0                 // 000000002f08: d5207c4f 00020615
	v_add3_u32 v80, v77, v95, 0x7fff                           // 000000002f10: d6550050 03febf4d 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f1c: bf870003
	v_add_co_u32 v77, s0, v78, v18                             // 000000002f20: d700004d 0202254e
	v_or_b32_e32 v83, 0x400000, v95                            // 000000002f28: 38a6beff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f30: bf88f19f
	v_add_co_ci_u32_e64 v78, null, v79, v19, s0                // 000000002f34: d5207c4e 0002274f
	v_cmp_u_f32_e64 s0, v95, v95                               // 000000002f3c: d4180000 0202bf5f
	s_wait_alu depctr_va_sdst(0)                               // 000000002f44: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f48: bf870001
	v_cndmask_b32_e64 v79, v80, v83, s0                        // 000000002f4c: d501004f 0002a750
	global_store_d16_hi_b16 v[77:78], v79, off                 // 000000002f54: ee09407c 27800000 0000004d
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f60: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002f64: 8c7e017e
	s_mul_u64 s[28:29], s[26:27], 7                            // 000000002f68: aa9c871a
	v_cmp_lt_i64_e64 s10, 7, v[20:21]                          // 000000002f6c: d451000a 02022887
	v_add_co_u32 v0, s0, s28, v0                               // 000000002f74: d7000000 0202001c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f7c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s29, v1, s0                  // 000000002f80: d5207c01 0002021d
	s_and_b32 s0, s10, s4                                      // 000000002f88: 8b00040a
	v_lshlrev_b64_e32 v[20:21], 1, v[0:1]                      // 000000002f8c: 3e280081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f90: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f94: be812000
	s_cbranch_execz 28                                         // 000000002f98: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x150c>
	v_bfe_u32 v0, v86, 16, 1                                   // 000000002f9c: d6100000 02052156
	s_wait_kmcnt 0x0                                           // 000000002fa4: bfc70000
	v_add_co_u32 v1, s0, s20, v2                               // 000000002fa8: d7000001 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002fb0: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s21, v3, s0                 // 000000002fb4: d5207c4d 00020615
	v_add3_u32 v78, v0, v86, 0x7fff                            // 000000002fbc: d655004e 03fead00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002fc8: bf870003
	v_add_co_u32 v0, s0, v1, v20                               // 000000002fcc: d7000000 02022901
	v_or_b32_e32 v79, 0x400000, v86                            // 000000002fd4: 389eacff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002fdc: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v77, v21, s0                 // 000000002fe0: d5207c01 00022b4d
	v_cmp_u_f32_e64 s0, v86, v86                               // 000000002fe8: d4180000 0202ad56
	s_wait_alu depctr_va_sdst(0)                               // 000000002ff0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ff4: bf870001
	v_cndmask_b32_e64 v77, v78, v79, s0                        // 000000002ff8: d501004d 00029f4e
	global_store_d16_hi_b16 v[0:1], v77, off                   // 000000003000: ee09407c 26800000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000300c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003010: 8c7e017e
	v_mul_lo_u32 v77, s27, v4                                  // 000000003014: d72c004d 0202081b
	v_mul_lo_u32 v78, s26, v5                                  // 00000000301c: d72c004e 02020a1a
	v_mad_co_u64_u32 v[0:1], null, s26, v4, 0                  // 000000003024: d6fe7c00 0202081a
	v_sub_co_u32 v4, s0, s24, v4                               // 00000000302c: d7010004 02020818
	s_wait_alu depctr_va_sdst(0)                               // 000000003034: bf88f19f
	v_sub_co_ci_u32_e64 v5, null, s25, v5, s0                  // 000000003038: d5217c05 00020a19
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003040: bf870211
	v_cmp_lt_i64_e64 s12, 0, v[4:5]                            // 000000003044: d451000c 02020880
	v_add3_u32 v1, v1, v78, v77                                // 00000000304c: d6550001 05369d01
	s_delay_alu instid0(valu_dep_1)                            // 000000003054: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003058: 3e000081
	s_and_b32 s0, s12, s4                                      // 00000000305c: 8b00040c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003060: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003064: be812000
	s_cbranch_execz 28                                         // 000000003068: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x15dc>
	s_wait_kmcnt 0x0                                           // 00000000306c: bfc70000
	v_add_co_u32 v78, s0, s20, v0                              // 000000003070: d700004e 02020014
	v_bfe_u32 v77, v85, 16, 1                                  // 000000003078: d610004d 02052155
	s_wait_alu depctr_va_sdst(0)                               // 000000003080: bf88f19f
	v_add_co_ci_u32_e64 v79, null, s21, v1, s0                 // 000000003084: d5207c4f 00020215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000308c: bf870193
	v_add_co_u32 v6, s0, v78, v6                               // 000000003090: d7000006 02020d4e
	v_add3_u32 v77, v77, v85, 0x7fff                           // 000000003098: d655004d 03feab4d 00007fff
	v_or_b32_e32 v80, 0x400000, v85                            // 0000000030a4: 38a0aaff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030ac: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v79, v7, s0                  // 0000000030b0: d5207c07 00020f4f
	v_cmp_u_f32_e64 s0, v85, v85                               // 0000000030b8: d4180000 0202ab55
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000030c4: bf870001
	v_cndmask_b32_e64 v77, v77, v80, s0                        // 0000000030c8: d501004d 0002a14d
	global_store_d16_hi_b16 v[6:7], v77, off                   // 0000000030d0: ee09407c 26800000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000030e0: 8c7e017e
	v_cmp_lt_i64_e64 s9, 1, v[4:5]                             // 0000000030e4: d4510009 02020881
	s_and_b32 s0, s9, s4                                       // 0000000030ec: 8b000409
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030f0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000030f4: be812000
	s_cbranch_execz 28                                         // 0000000030f8: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x166c>
	v_bfe_u32 v6, v82, 16, 1                                   // 0000000030fc: d6100006 02052152
	s_wait_kmcnt 0x0                                           // 000000003104: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 000000003108: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003110: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s21, v1, s0                 // 000000003114: d5207c4d 00020215
	v_add3_u32 v78, v6, v82, 0x7fff                            // 00000000311c: d655004e 03fea506 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003128: bf870003
	v_add_co_u32 v6, s0, v7, v8                                // 00000000312c: d7000006 02021107
	v_or_b32_e32 v79, 0x400000, v82                            // 000000003134: 389ea4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000313c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v77, v9, s0                  // 000000003140: d5207c07 0002134d
	v_cmp_u_f32_e64 s0, v82, v82                               // 000000003148: d4180000 0202a552
	s_wait_alu depctr_va_sdst(0)                               // 000000003150: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003154: bf870001
	v_cndmask_b32_e64 v8, v78, v79, s0                         // 000000003158: d5010008 00029f4e
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003160: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000316c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003170: 8c7e017e
	v_cmp_lt_i64_e64 s8, 2, v[4:5]                             // 000000003174: d4510008 02020882
	s_and_b32 s0, s8, s4                                       // 00000000317c: 8b000408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003180: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003184: be812000
	s_cbranch_execz 28                                         // 000000003188: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x16fc>
	v_bfe_u32 v6, v81, 16, 1                                   // 00000000318c: d6100006 02052151
	s_wait_kmcnt 0x0                                           // 000000003194: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 000000003198: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000031a0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 0000000031a4: d5207c08 00020215
	v_add3_u32 v9, v6, v81, 0x7fff                             // 0000000031ac: d6550009 03fea306 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000031b8: bf870003
	v_add_co_u32 v6, s0, v7, v10                               // 0000000031bc: d7000006 02021507
	v_or_b32_e32 v77, 0x400000, v81                            // 0000000031c4: 389aa2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000031cc: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v11, s0                  // 0000000031d0: d5207c07 00021708
	v_cmp_u_f32_e64 s0, v81, v81                               // 0000000031d8: d4180000 0202a351
	s_wait_alu depctr_va_sdst(0)                               // 0000000031e0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000031e4: bf870001
	v_cndmask_b32_e64 v8, v9, v77, s0                          // 0000000031e8: d5010008 00029b09
	global_store_d16_hi_b16 v[6:7], v8, off                    // 0000000031f0: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003200: 8c7e017e
	v_cmp_lt_i64_e64 s7, 3, v[4:5]                             // 000000003204: d4510007 02020883
	s_and_b32 s0, s7, s4                                       // 00000000320c: 8b000407
	s_wait_alu depctr_sa_sdst(0)                               // 000000003210: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003214: be812000
	s_cbranch_execz 28                                         // 000000003218: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x178c>
	v_bfe_u32 v6, v76, 16, 1                                   // 00000000321c: d6100006 0205214c
	s_wait_kmcnt 0x0                                           // 000000003224: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 000000003228: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003230: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 000000003234: d5207c08 00020215
	v_add3_u32 v9, v6, v76, 0x7fff                             // 00000000323c: d6550009 03fe9906 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003248: bf870003
	v_add_co_u32 v6, s0, v7, v12                               // 00000000324c: d7000006 02021907
	v_or_b32_e32 v10, 0x400000, v76                            // 000000003254: 381498ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000325c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v13, s0                  // 000000003260: d5207c07 00021b08
	v_cmp_u_f32_e64 s0, v76, v76                               // 000000003268: d4180000 0202994c
	s_wait_alu depctr_va_sdst(0)                               // 000000003270: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003274: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003278: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003280: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000328c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003290: 8c7e017e
	v_cmp_lt_i64_e64 s6, 4, v[4:5]                             // 000000003294: d4510006 02020884
	s_and_b32 s0, s6, s4                                       // 00000000329c: 8b000406
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032a4: be812000
	s_cbranch_execz 28                                         // 0000000032a8: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x181c>
	v_bfe_u32 v6, v75, 16, 1                                   // 0000000032ac: d6100006 0205214b
	s_wait_kmcnt 0x0                                           // 0000000032b4: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 0000000032b8: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000032c0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 0000000032c4: d5207c08 00020215
	v_add3_u32 v9, v6, v75, 0x7fff                             // 0000000032cc: d6550009 03fe9706 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000032d8: bf870003
	v_add_co_u32 v6, s0, v7, v14                               // 0000000032dc: d7000006 02021d07
	v_or_b32_e32 v10, 0x400000, v75                            // 0000000032e4: 381496ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000032ec: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v15, s0                  // 0000000032f0: d5207c07 00021f08
	v_cmp_u_f32_e64 s0, v75, v75                               // 0000000032f8: d4180000 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 000000003300: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003304: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003308: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003310: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000331c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003320: 8c7e017e
	v_cmp_lt_i64_e64 s5, 5, v[4:5]                             // 000000003324: d4510005 02020885
	s_and_b32 s0, s5, s4                                       // 00000000332c: 8b000405
	s_wait_alu depctr_sa_sdst(0)                               // 000000003330: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003334: be812000
	s_cbranch_execz 28                                         // 000000003338: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x18ac>
	v_bfe_u32 v6, v74, 16, 1                                   // 00000000333c: d6100006 0205214a
	s_wait_kmcnt 0x0                                           // 000000003344: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 000000003348: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003350: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 000000003354: d5207c08 00020215
	v_add3_u32 v9, v6, v74, 0x7fff                             // 00000000335c: d6550009 03fe9506 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003368: bf870003
	v_add_co_u32 v6, s0, v7, v16                               // 00000000336c: d7000006 02022107
	v_or_b32_e32 v10, 0x400000, v74                            // 000000003374: 381494ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000337c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v17, s0                  // 000000003380: d5207c07 00022308
	v_cmp_u_f32_e64 s0, v74, v74                               // 000000003388: d4180000 0202954a
	s_wait_alu depctr_va_sdst(0)                               // 000000003390: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003394: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003398: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 0000000033a0: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000033b0: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[4:5]                             // 0000000033b4: d4510001 02020886
	s_and_b32 s0, s1, s4                                       // 0000000033bc: 8b000401
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033c0: bf88ff9e
	s_and_saveexec_b32 s19, s0                                 // 0000000033c4: be932000
	s_cbranch_execz 28                                         // 0000000033c8: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x193c>
	v_bfe_u32 v6, v73, 16, 1                                   // 0000000033cc: d6100006 02052149
	s_wait_kmcnt 0x0                                           // 0000000033d4: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 0000000033d8: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000033e0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 0000000033e4: d5207c08 00020215
	v_add3_u32 v9, v6, v73, 0x7fff                             // 0000000033ec: d6550009 03fe9306 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000033f8: bf870003
	v_add_co_u32 v6, s0, v7, v18                               // 0000000033fc: d7000006 02022507
	v_or_b32_e32 v10, 0x400000, v73                            // 000000003404: 381492ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000340c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v19, s0                  // 000000003410: d5207c07 00022708
	v_cmp_u_f32_e64 s0, v73, v73                               // 000000003418: d4180000 02029349
	s_wait_alu depctr_va_sdst(0)                               // 000000003420: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003424: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003428: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003430: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000343c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003440: 8c7e137e
	v_cmp_lt_i64_e64 s0, 7, v[4:5]                             // 000000003444: d4510000 02020887
	s_and_b32 s4, s0, s4                                       // 00000000344c: 8b040400
	s_wait_alu depctr_sa_sdst(0)                               // 000000003450: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003454: be932004
	s_cbranch_execz 28                                         // 000000003458: bfa5001c <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x19cc>
	v_bfe_u32 v4, v72, 16, 1                                   // 00000000345c: d6100004 02052148
	s_wait_kmcnt 0x0                                           // 000000003464: bfc70000
	v_add_co_u32 v5, s4, s20, v0                               // 000000003468: d7000405 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003470: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s4                  // 000000003474: d5207c06 00120215
	v_add3_u32 v7, v4, v72, 0x7fff                             // 00000000347c: d6550007 03fe9104 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003488: bf870003
	v_add_co_u32 v4, s4, v5, v20                               // 00000000348c: d7000404 02022905
	v_or_b32_e32 v8, 0x400000, v72                             // 000000003494: 381090ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000349c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v21, s4                  // 0000000034a0: d5207c05 00122b06
	v_cmp_u_f32_e64 s4, v72, v72                               // 0000000034a8: d4180004 02029148
	s_wait_alu depctr_va_sdst(0)                               // 0000000034b0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000034b4: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s4                           // 0000000034b8: d5010006 00121107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000034c0: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000034d0: 8c7e137e
	s_and_b32 s4, s17, s3                                      // 0000000034d4: 8b040311
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034d8: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000034dc: be932004
	s_cbranch_execz 40                                         // 0000000034e0: bfa50028 <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1a84>
	v_add_co_u32 v4, s4, v23, s22                              // 0000000034e4: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000034ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 0000000034f0: d5207c05 00102e80
	v_bfe_u32 v6, v71, 16, 1                                   // 0000000034f8: d6100006 02052147
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003500: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000003504: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000350c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003510: d5207c05 00120a80
	s_wait_kmcnt 0x0                                           // 000000003518: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 00000000351c: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003524: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003528: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003530: 3e080881
	v_add3_u32 v6, v6, v71, 0x7fff                             // 000000003534: d6550006 03fe8f06 00007fff
	v_or_b32_e32 v9, 0x400000, v71                             // 000000003540: 38128eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003548: bf870223
	v_add_co_u32 v4, s4, v7, v4                                // 00000000354c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003554: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003558: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v71, v71                               // 000000003560: d4180004 02028f47
	s_wait_alu depctr_va_sdst(0)                               // 000000003568: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000356c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003570: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003578: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003584: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003588: 8c7e137e
	s_and_b32 s4, s18, s3                                      // 00000000358c: 8b040312
	s_wait_alu depctr_sa_sdst(0)                               // 000000003590: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003594: be932004
	s_cbranch_execz 46                                         // 000000003598: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1b54>
	v_add_co_u32 v4, s4, v23, s22                              // 00000000359c: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000035a4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 0000000035a8: d5207c05 00102e80
	v_bfe_u32 v6, v70, 16, 1                                   // 0000000035b0: d6100006 02052146
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035b8: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 0000000035bc: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000035c8: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v70                             // 0000000035d0: 38128cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035d8: bf8701a3
	v_add_co_u32 v4, s4, s26, v4                               // 0000000035dc: d7000404 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 0000000035e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s4                  // 0000000035e8: d5207c05 00120a1b
	s_wait_kmcnt 0x0                                           // 0000000035f0: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 0000000035f4: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000035fc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003600: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003608: 3e080881
	v_add3_u32 v6, v6, v70, 0x7fff                             // 00000000360c: d6550006 03fe8d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003618: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 00000000361c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003624: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003628: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v70, v70                               // 000000003630: d4180004 02028d46
	s_wait_alu depctr_va_sdst(0)                               // 000000003638: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000363c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003640: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003648: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003654: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003658: 8c7e137e
	s_and_b32 s4, s16, s3                                      // 00000000365c: 8b040310
	s_wait_alu depctr_sa_sdst(0)                               // 000000003660: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003664: be932004
	s_cbranch_execz 46                                         // 000000003668: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1c24>
	v_add_co_u32 v4, s4, v23, s22                              // 00000000366c: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003674: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003678: d5207c05 00102e80
	v_bfe_u32 v6, v69, 16, 1                                   // 000000003680: d6100006 02052145
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003688: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 00000000368c: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003694: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003698: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v69                             // 0000000036a0: 38128aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000036a8: bf8701a3
	v_add_co_u32 v4, s4, s40, v4                               // 0000000036ac: d7000404 02020828
	s_wait_alu depctr_va_sdst(0)                               // 0000000036b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s4                  // 0000000036b8: d5207c05 00120a29
	s_wait_kmcnt 0x0                                           // 0000000036c0: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 0000000036c4: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000036cc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 0000000036d0: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000036d8: 3e080881
	v_add3_u32 v6, v6, v69, 0x7fff                             // 0000000036dc: d6550006 03fe8b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000036e8: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 0000000036ec: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000036f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 0000000036f8: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v69, v69                               // 000000003700: d4180004 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 000000003708: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000370c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003710: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003718: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003724: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003728: 8c7e137e
	s_and_b32 s4, s15, s3                                      // 00000000372c: 8b04030f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003730: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003734: be932004
	s_cbranch_execz 46                                         // 000000003738: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1cf4>
	v_add_co_u32 v4, s4, v23, s22                              // 00000000373c: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003744: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003748: d5207c05 00102e80
	v_bfe_u32 v6, v68, 16, 1                                   // 000000003750: d6100006 02052144
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003758: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 00000000375c: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003764: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003768: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v68                             // 000000003770: 381288ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003778: bf8701a3
	v_add_co_u32 v4, s4, s38, v4                               // 00000000377c: d7000404 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000003784: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s4                  // 000000003788: d5207c05 00120a27
	s_wait_kmcnt 0x0                                           // 000000003790: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003794: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000379c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 0000000037a0: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000037a8: 3e080881
	v_add3_u32 v6, v6, v68, 0x7fff                             // 0000000037ac: d6550006 03fe8906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000037b8: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 0000000037bc: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000037c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 0000000037c8: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v68, v68                               // 0000000037d0: d4180004 02028944
	s_wait_alu depctr_va_sdst(0)                               // 0000000037d8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000037dc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 0000000037e0: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000037e8: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037f4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000037f8: 8c7e137e
	s_and_b32 s4, s14, s3                                      // 0000000037fc: 8b04030e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003800: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003804: be932004
	s_cbranch_execz 46                                         // 000000003808: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1dc4>
	v_add_co_u32 v4, s4, v23, s22                              // 00000000380c: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003814: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003818: d5207c05 00102e80
	v_bfe_u32 v6, v67, 16, 1                                   // 000000003820: d6100006 02052143
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003828: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 00000000382c: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003834: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003838: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v67                             // 000000003840: 381286ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003848: bf8701a3
	v_add_co_u32 v4, s4, s36, v4                               // 00000000384c: d7000404 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000003854: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s4                  // 000000003858: d5207c05 00120a25
	s_wait_kmcnt 0x0                                           // 000000003860: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003864: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000386c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003870: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003878: 3e080881
	v_add3_u32 v6, v6, v67, 0x7fff                             // 00000000387c: d6550006 03fe8706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003888: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 00000000388c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003894: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003898: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v67, v67                               // 0000000038a0: d4180004 02028743
	s_wait_alu depctr_va_sdst(0)                               // 0000000038a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038ac: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 0000000038b0: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000038b8: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000038c8: 8c7e137e
	s_and_b32 s4, s13, s3                                      // 0000000038cc: 8b04030d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038d0: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000038d4: be932004
	s_cbranch_execz 46                                         // 0000000038d8: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1e94>
	v_add_co_u32 v4, s4, v23, s22                              // 0000000038dc: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000038e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 0000000038e8: d5207c05 00102e80
	v_bfe_u32 v6, v66, 16, 1                                   // 0000000038f0: d6100006 02052142
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000038f8: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 0000000038fc: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003904: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003908: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v66                             // 000000003910: 381284ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003918: bf8701a3
	v_add_co_u32 v4, s4, s34, v4                               // 00000000391c: d7000404 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000003924: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s4                  // 000000003928: d5207c05 00120a23
	s_wait_kmcnt 0x0                                           // 000000003930: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003934: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000393c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003940: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003948: 3e080881
	v_add3_u32 v6, v6, v66, 0x7fff                             // 00000000394c: d6550006 03fe8506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003958: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 00000000395c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003964: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003968: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v66, v66                               // 000000003970: d4180004 02028542
	s_wait_alu depctr_va_sdst(0)                               // 000000003978: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000397c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003980: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003988: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003994: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003998: 8c7e137e
	s_and_b32 s4, s11, s3                                      // 00000000399c: 8b04030b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039a0: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000039a4: be932004
	s_cbranch_execz 46                                         // 0000000039a8: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x1f64>
	v_add_co_u32 v4, s4, v23, s22                              // 0000000039ac: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000039b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 0000000039b8: d5207c05 00102e80
	v_bfe_u32 v6, v65, 16, 1                                   // 0000000039c0: d6100006 02052141
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039c8: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 0000000039cc: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000039d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000039d8: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v65                             // 0000000039e0: 381282ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039e8: bf8701a3
	v_add_co_u32 v4, s4, s30, v4                               // 0000000039ec: d7000404 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000039f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s4                  // 0000000039f8: d5207c05 00120a1f
	s_wait_kmcnt 0x0                                           // 000000003a00: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003a04: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003a0c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003a10: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003a18: 3e080881
	v_add3_u32 v6, v6, v65, 0x7fff                             // 000000003a1c: d6550006 03fe8306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a28: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003a2c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003a34: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003a38: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v65, v65                               // 000000003a40: d4180004 02028341
	s_wait_alu depctr_va_sdst(0)                               // 000000003a48: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a4c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003a50: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003a58: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a64: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003a68: 8c7e137e
	s_and_b32 s4, s10, s3                                      // 000000003a6c: 8b04030a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a70: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003a74: be932004
	s_cbranch_execz 46                                         // 000000003a78: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2034>
	v_add_co_u32 v4, s4, v23, s22                              // 000000003a7c: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003a84: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003a88: d5207c05 00102e80
	v_bfe_u32 v6, v64, 16, 1                                   // 000000003a90: d6100006 02052140
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a98: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000003a9c: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003aa4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003aa8: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v64                             // 000000003ab0: 381280ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ab8: bf8701a3
	v_add_co_u32 v4, s4, s28, v4                               // 000000003abc: d7000404 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000003ac4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s4                  // 000000003ac8: d5207c05 00120a1d
	s_wait_kmcnt 0x0                                           // 000000003ad0: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003ad4: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003adc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003ae0: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003ae8: 3e080881
	v_add3_u32 v6, v6, v64, 0x7fff                             // 000000003aec: d6550006 03fe8106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003af8: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003afc: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003b04: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003b08: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v64, v64                               // 000000003b10: d4180004 02028140
	s_wait_alu depctr_va_sdst(0)                               // 000000003b18: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b1c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003b20: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003b28: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b34: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003b38: 8c7e137e
	s_and_b32 s4, s12, s3                                      // 000000003b3c: 8b04030c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b40: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003b44: be932004
	s_cbranch_execz 40                                         // 000000003b48: bfa50028 <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x20ec>
	v_add_co_u32 v4, s4, v23, s22                              // 000000003b4c: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003b54: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003b58: d5207c05 00102e80
	v_bfe_u32 v6, v63, 16, 1                                   // 000000003b60: d6100006 0205213f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b68: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000003b6c: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003b74: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003b78: d5207c05 00120a80
	s_wait_kmcnt 0x0                                           // 000000003b80: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000003b84: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003b8c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000003b90: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003b98: 3e080881
	v_add3_u32 v6, v6, v63, 0x7fff                             // 000000003b9c: d6550006 03fe7f06 00007fff
	v_or_b32_e32 v9, 0x400000, v63                             // 000000003ba8: 38127eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003bb0: bf870223
	v_add_co_u32 v4, s4, v7, v4                                // 000000003bb4: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003bbc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003bc0: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v63, v63                               // 000000003bc8: d4180004 02027f3f
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003bd4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003bd8: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003be0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003bf0: 8c7e137e
	s_and_b32 s4, s9, s3                                       // 000000003bf4: 8b040309
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bf8: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003bfc: be932004
	s_cbranch_execz 46                                         // 000000003c00: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x21bc>
	v_add_co_u32 v4, s4, v23, s22                              // 000000003c04: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003c0c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003c10: d5207c05 00102e80
	v_bfe_u32 v6, v62, 16, 1                                   // 000000003c18: d6100006 0205213e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c20: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000003c24: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003c2c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003c30: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v62                             // 000000003c38: 38127cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c40: bf8701a3
	v_add_co_u32 v4, s4, s26, v4                               // 000000003c44: d7000404 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000003c4c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s4                  // 000000003c50: d5207c05 00120a1b
	s_wait_kmcnt 0x0                                           // 000000003c58: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000003c5c: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003c64: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000003c68: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003c70: 3e080881
	v_add3_u32 v6, v6, v62, 0x7fff                             // 000000003c74: d6550006 03fe7d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c80: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003c84: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003c8c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003c90: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v62, v62                               // 000000003c98: d4180004 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000003ca0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ca4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003ca8: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003cb0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cbc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003cc0: 8c7e137e
	s_and_b32 s4, s8, s3                                       // 000000003cc4: 8b040308
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cc8: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003ccc: be932004
	s_cbranch_execz 46                                         // 000000003cd0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x228c>
	v_add_co_u32 v4, s4, v23, s22                              // 000000003cd4: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003cdc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003ce0: d5207c05 00102e80
	v_bfe_u32 v6, v61, 16, 1                                   // 000000003ce8: d6100006 0205213d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cf0: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000003cf4: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003cfc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003d00: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v61                             // 000000003d08: 38127aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d10: bf8701a3
	v_add_co_u32 v4, s4, s40, v4                               // 000000003d14: d7000404 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000003d1c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s4                  // 000000003d20: d5207c05 00120a29
	s_wait_kmcnt 0x0                                           // 000000003d28: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000003d2c: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003d34: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000003d38: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003d40: 3e080881
	v_add3_u32 v6, v6, v61, 0x7fff                             // 000000003d44: d6550006 03fe7b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d50: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003d54: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003d5c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003d60: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v61, v61                               // 000000003d68: d4180004 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000003d70: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003d74: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003d78: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003d80: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d8c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003d90: 8c7e137e
	s_and_b32 s4, s7, s3                                       // 000000003d94: 8b040307
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d98: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003d9c: be932004
	s_cbranch_execz 46                                         // 000000003da0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x235c>
	v_add_co_u32 v4, s4, v23, s22                              // 000000003da4: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003dac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003db0: d5207c05 00102e80
	v_bfe_u32 v6, v60, 16, 1                                   // 000000003db8: d6100006 0205213c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003dc0: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000003dc4: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003dcc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003dd0: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v60                             // 000000003dd8: 381278ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003de0: bf8701a3
	v_add_co_u32 v4, s4, s38, v4                               // 000000003de4: d7000404 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000003dec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s4                  // 000000003df0: d5207c05 00120a27
	s_wait_kmcnt 0x0                                           // 000000003df8: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000003dfc: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003e04: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000003e08: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003e10: 3e080881
	v_add3_u32 v6, v6, v60, 0x7fff                             // 000000003e14: d6550006 03fe7906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e20: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003e24: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003e2c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003e30: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v60, v60                               // 000000003e38: d4180004 0202793c
	s_wait_alu depctr_va_sdst(0)                               // 000000003e40: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003e44: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003e48: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003e50: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e5c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003e60: 8c7e137e
	s_and_b32 s4, s6, s3                                       // 000000003e64: 8b040306
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e68: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003e6c: be932004
	s_cbranch_execz 46                                         // 000000003e70: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x242c>
	v_add_co_u32 v4, s4, v23, s22                              // 000000003e74: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003e7c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003e80: d5207c05 00102e80
	v_bfe_u32 v6, v59, 16, 1                                   // 000000003e88: d6100006 0205213b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e90: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000003e94: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003e9c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003ea0: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v59                             // 000000003ea8: 381276ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003eb0: bf8701a3
	v_add_co_u32 v4, s4, s36, v4                               // 000000003eb4: d7000404 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000003ebc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s4                  // 000000003ec0: d5207c05 00120a25
	s_wait_kmcnt 0x0                                           // 000000003ec8: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000003ecc: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003ed4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000003ed8: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003ee0: 3e080881
	v_add3_u32 v6, v6, v59, 0x7fff                             // 000000003ee4: d6550006 03fe7706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ef0: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003ef4: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003efc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003f00: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v59, v59                               // 000000003f08: d4180004 0202773b
	s_wait_alu depctr_va_sdst(0)                               // 000000003f10: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003f14: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003f18: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003f20: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f2c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003f30: 8c7e137e
	s_and_b32 s4, s5, s3                                       // 000000003f34: 8b040305
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f38: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003f3c: be932004
	s_cbranch_execz 46                                         // 000000003f40: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x24fc>
	v_add_co_u32 v4, s4, v23, s22                              // 000000003f44: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000003f4c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003f50: d5207c05 00102e80
	v_bfe_u32 v6, v58, 16, 1                                   // 000000003f58: d6100006 0205213a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f60: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000003f64: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003f6c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003f70: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v58                             // 000000003f78: 381274ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f80: bf8701a3
	v_add_co_u32 v4, s4, s34, v4                               // 000000003f84: d7000404 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000003f8c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s4                  // 000000003f90: d5207c05 00120a23
	s_wait_kmcnt 0x0                                           // 000000003f98: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000003f9c: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003fa4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000003fa8: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003fb0: 3e080881
	v_add3_u32 v6, v6, v58, 0x7fff                             // 000000003fb4: d6550006 03fe7506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fc0: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003fc4: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003fcc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003fd0: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v58, v58                               // 000000003fd8: d4180004 0202753a
	s_wait_alu depctr_va_sdst(0)                               // 000000003fe0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003fe4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003fe8: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003ff0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ffc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000004000: 8c7e137e
	s_and_b32 s4, s1, s3                                       // 000000004004: 8b040301
	s_wait_alu depctr_sa_sdst(0)                               // 000000004008: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 00000000400c: be932004
	s_cbranch_execz 46                                         // 000000004010: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x25cc>
	v_add_co_u32 v4, s4, v23, s22                              // 000000004014: d7000404 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 00000000401c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000004020: d5207c05 00102e80
	v_bfe_u32 v6, v57, 16, 1                                   // 000000004028: d6100006 02052139
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004030: bf8701a3
	v_add_co_u32 v4, s4, v4, v22                               // 000000004034: d7000404 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000403c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000004040: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v57                             // 000000004048: 381272ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004050: bf8701a3
	v_add_co_u32 v4, s4, s30, v4                               // 000000004054: d7000404 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 00000000405c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s4                  // 000000004060: d5207c05 00120a1f
	s_wait_kmcnt 0x0                                           // 000000004068: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 00000000406c: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004074: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004078: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004080: 3e080881
	v_add3_u32 v6, v6, v57, 0x7fff                             // 000000004084: d6550006 03fe7306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004090: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004094: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000409c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 0000000040a0: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v57, v57                               // 0000000040a8: d4180004 02027339
	s_wait_alu depctr_va_sdst(0)                               // 0000000040b0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000040b4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 0000000040b8: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000040c0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000040d0: 8c7e137e
	s_and_b32 s3, s0, s3                                       // 0000000040d4: 8b030300
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040d8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000040dc: be842003
	s_cbranch_execz 46                                         // 0000000040e0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x269c>
	v_add_co_u32 v4, s3, v23, s22                              // 0000000040e4: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000040ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000040f0: d5207c05 000c2e80
	v_bfe_u32 v6, v56, 16, 1                                   // 0000000040f8: d6100006 02052138
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004100: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 000000004104: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000410c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004110: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v56                             // 000000004118: 381270ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004120: bf8701a3
	v_add_co_u32 v4, s3, s28, v4                               // 000000004124: d7000304 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 00000000412c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s3                  // 000000004130: d5207c05 000e0a1d
	s_wait_kmcnt 0x0                                           // 000000004138: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 00000000413c: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004144: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004148: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004150: 3e080881
	v_add3_u32 v6, v6, v56, 0x7fff                             // 000000004154: d6550006 03fe7106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004160: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004164: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000416c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004170: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v56, v56                               // 000000004178: d4180003 02027138
	s_wait_alu depctr_va_sdst(0)                               // 000000004180: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004184: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004188: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000004190: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000419c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000041a0: 8c7e047e
	s_and_b32 s3, s17, s2                                      // 0000000041a4: 8b030211
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041a8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000041ac: be842003
	s_cbranch_execz 40                                         // 0000000041b0: bfa50028 <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2754>
	v_add_co_u32 v4, s3, v23, s22                              // 0000000041b4: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000041bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000041c0: d5207c05 000c2e80
	v_bfe_u32 v6, v55, 16, 1                                   // 0000000041c8: d6100006 02052137
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000041d0: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 0000000041d4: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000041dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000041e0: d5207c05 000e0a80
	s_wait_kmcnt 0x0                                           // 0000000041e8: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 0000000041ec: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000041f4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000041f8: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004200: 3e080881
	v_add3_u32 v6, v6, v55, 0x7fff                             // 000000004204: d6550006 03fe6f06 00007fff
	v_or_b32_e32 v9, 0x400000, v55                             // 000000004210: 38126eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000004218: bf870223
	v_add_co_u32 v4, s3, v7, v4                                // 00000000421c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004224: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004228: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v55, v55                               // 000000004230: d4180003 02026f37
	s_wait_alu depctr_va_sdst(0)                               // 000000004238: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000423c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004240: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004248: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004254: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004258: 8c7e047e
	s_and_b32 s3, s18, s2                                      // 00000000425c: 8b030212
	s_wait_alu depctr_sa_sdst(0)                               // 000000004260: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004264: be842003
	s_cbranch_execz 46                                         // 000000004268: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2824>
	v_add_co_u32 v4, s3, v23, s22                              // 00000000426c: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004274: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004278: d5207c05 000c2e80
	v_bfe_u32 v6, v54, 16, 1                                   // 000000004280: d6100006 02052136
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004288: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 00000000428c: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004294: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004298: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v54                             // 0000000042a0: 38126cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000042a8: bf8701a3
	v_add_co_u32 v4, s3, s26, v4                               // 0000000042ac: d7000304 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 0000000042b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s3                  // 0000000042b8: d5207c05 000e0a1b
	s_wait_kmcnt 0x0                                           // 0000000042c0: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 0000000042c4: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000042cc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000042d0: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000042d8: 3e080881
	v_add3_u32 v6, v6, v54, 0x7fff                             // 0000000042dc: d6550006 03fe6d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000042e8: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 0000000042ec: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000042f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000042f8: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v54, v54                               // 000000004300: d4180003 02026d36
	s_wait_alu depctr_va_sdst(0)                               // 000000004308: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000430c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004310: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004318: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004324: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004328: 8c7e047e
	s_and_b32 s3, s16, s2                                      // 00000000432c: 8b030210
	s_wait_alu depctr_sa_sdst(0)                               // 000000004330: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004334: be842003
	s_cbranch_execz 46                                         // 000000004338: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x28f4>
	v_add_co_u32 v4, s3, v23, s22                              // 00000000433c: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004344: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004348: d5207c05 000c2e80
	v_bfe_u32 v6, v53, 16, 1                                   // 000000004350: d6100006 02052135
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004358: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 00000000435c: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004364: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004368: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v53                             // 000000004370: 38126aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004378: bf8701a3
	v_add_co_u32 v4, s3, s40, v4                               // 00000000437c: d7000304 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000004384: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s3                  // 000000004388: d5207c05 000e0a29
	s_wait_kmcnt 0x0                                           // 000000004390: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004394: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000439c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000043a0: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000043a8: 3e080881
	v_add3_u32 v6, v6, v53, 0x7fff                             // 0000000043ac: d6550006 03fe6b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000043b8: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 0000000043bc: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000043c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000043c8: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v53, v53                               // 0000000043d0: d4180003 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 0000000043d8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000043dc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 0000000043e0: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000043e8: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043f4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000043f8: 8c7e047e
	s_and_b32 s3, s15, s2                                      // 0000000043fc: 8b03020f
	s_wait_alu depctr_sa_sdst(0)                               // 000000004400: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004404: be842003
	s_cbranch_execz 46                                         // 000000004408: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x29c4>
	v_add_co_u32 v4, s3, v23, s22                              // 00000000440c: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004414: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004418: d5207c05 000c2e80
	v_bfe_u32 v6, v52, 16, 1                                   // 000000004420: d6100006 02052134
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004428: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 00000000442c: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004434: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004438: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v52                             // 000000004440: 381268ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004448: bf8701a3
	v_add_co_u32 v4, s3, s38, v4                               // 00000000444c: d7000304 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000004454: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s3                  // 000000004458: d5207c05 000e0a27
	s_wait_kmcnt 0x0                                           // 000000004460: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004464: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000446c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004470: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004478: 3e080881
	v_add3_u32 v6, v6, v52, 0x7fff                             // 00000000447c: d6550006 03fe6906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004488: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 00000000448c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004494: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004498: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v52, v52                               // 0000000044a0: d4180003 02026934
	s_wait_alu depctr_va_sdst(0)                               // 0000000044a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000044ac: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 0000000044b0: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000044b8: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000044c8: 8c7e047e
	s_and_b32 s3, s14, s2                                      // 0000000044cc: 8b03020e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044d0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000044d4: be842003
	s_cbranch_execz 46                                         // 0000000044d8: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2a94>
	v_add_co_u32 v4, s3, v23, s22                              // 0000000044dc: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000044e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000044e8: d5207c05 000c2e80
	v_bfe_u32 v6, v51, 16, 1                                   // 0000000044f0: d6100006 02052133
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000044f8: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 0000000044fc: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004504: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004508: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v51                             // 000000004510: 381266ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004518: bf8701a3
	v_add_co_u32 v4, s3, s36, v4                               // 00000000451c: d7000304 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000004524: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s3                  // 000000004528: d5207c05 000e0a25
	s_wait_kmcnt 0x0                                           // 000000004530: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004534: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000453c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004540: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004548: 3e080881
	v_add3_u32 v6, v6, v51, 0x7fff                             // 00000000454c: d6550006 03fe6706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004558: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 00000000455c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004564: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004568: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v51, v51                               // 000000004570: d4180003 02026733
	s_wait_alu depctr_va_sdst(0)                               // 000000004578: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000457c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004580: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004588: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004594: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004598: 8c7e047e
	s_and_b32 s3, s13, s2                                      // 00000000459c: 8b03020d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045a0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000045a4: be842003
	s_cbranch_execz 46                                         // 0000000045a8: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2b64>
	v_add_co_u32 v4, s3, v23, s22                              // 0000000045ac: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000045b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000045b8: d5207c05 000c2e80
	v_bfe_u32 v6, v50, 16, 1                                   // 0000000045c0: d6100006 02052132
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045c8: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 0000000045cc: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000045d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000045d8: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v50                             // 0000000045e0: 381264ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045e8: bf8701a3
	v_add_co_u32 v4, s3, s34, v4                               // 0000000045ec: d7000304 02020822
	s_wait_alu depctr_va_sdst(0)                               // 0000000045f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s3                  // 0000000045f8: d5207c05 000e0a23
	s_wait_kmcnt 0x0                                           // 000000004600: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004604: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000460c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004610: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004618: 3e080881
	v_add3_u32 v6, v6, v50, 0x7fff                             // 00000000461c: d6550006 03fe6506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004628: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 00000000462c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004634: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004638: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v50, v50                               // 000000004640: d4180003 02026532
	s_wait_alu depctr_va_sdst(0)                               // 000000004648: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000464c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004650: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004658: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004664: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004668: 8c7e047e
	s_and_b32 s3, s11, s2                                      // 00000000466c: 8b03020b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004670: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004674: be842003
	s_cbranch_execz 46                                         // 000000004678: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2c34>
	v_add_co_u32 v4, s3, v23, s22                              // 00000000467c: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004684: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004688: d5207c05 000c2e80
	v_bfe_u32 v6, v49, 16, 1                                   // 000000004690: d6100006 02052131
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004698: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 00000000469c: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000046a4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000046a8: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v49                             // 0000000046b0: 381262ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000046b8: bf8701a3
	v_add_co_u32 v4, s3, s30, v4                               // 0000000046bc: d7000304 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000046c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s3                  // 0000000046c8: d5207c05 000e0a1f
	s_wait_kmcnt 0x0                                           // 0000000046d0: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 0000000046d4: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000046dc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000046e0: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000046e8: 3e080881
	v_add3_u32 v6, v6, v49, 0x7fff                             // 0000000046ec: d6550006 03fe6306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000046f8: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 0000000046fc: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004704: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004708: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v49, v49                               // 000000004710: d4180003 02026331
	s_wait_alu depctr_va_sdst(0)                               // 000000004718: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000471c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004720: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004728: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004734: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004738: 8c7e047e
	s_and_b32 s3, s10, s2                                      // 00000000473c: 8b03020a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004740: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004744: be842003
	s_cbranch_execz 46                                         // 000000004748: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2d04>
	v_add_co_u32 v4, s3, v23, s22                              // 00000000474c: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004754: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004758: d5207c05 000c2e80
	v_bfe_u32 v6, v48, 16, 1                                   // 000000004760: d6100006 02052130
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004768: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 00000000476c: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004774: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004778: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v48                             // 000000004780: 381260ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004788: bf8701a3
	v_add_co_u32 v4, s3, s28, v4                               // 00000000478c: d7000304 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000004794: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s3                  // 000000004798: d5207c05 000e0a1d
	s_wait_kmcnt 0x0                                           // 0000000047a0: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 0000000047a4: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000047ac: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000047b0: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000047b8: 3e080881
	v_add3_u32 v6, v6, v48, 0x7fff                             // 0000000047bc: d6550006 03fe6106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000047c8: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 0000000047cc: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000047d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000047d8: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v48, v48                               // 0000000047e0: d4180003 02026130
	s_wait_alu depctr_va_sdst(0)                               // 0000000047e8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000047ec: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 0000000047f0: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000047f8: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004804: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004808: 8c7e047e
	s_and_b32 s3, s12, s2                                      // 00000000480c: 8b03020c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004810: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004814: be842003
	s_cbranch_execz 40                                         // 000000004818: bfa50028 <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2dbc>
	v_add_co_u32 v4, s3, v23, s22                              // 00000000481c: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004824: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004828: d5207c05 000c2e80
	v_bfe_u32 v6, v47, 16, 1                                   // 000000004830: d6100006 0205212f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004838: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 00000000483c: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004844: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004848: d5207c05 000e0a80
	s_wait_kmcnt 0x0                                           // 000000004850: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004854: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 00000000485c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004860: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004868: 3e080881
	v_add3_u32 v6, v6, v47, 0x7fff                             // 00000000486c: d6550006 03fe5f06 00007fff
	v_or_b32_e32 v9, 0x400000, v47                             // 000000004878: 38125eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000004880: bf870223
	v_add_co_u32 v4, s3, v7, v4                                // 000000004884: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000488c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004890: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v47, v47                               // 000000004898: d4180003 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000048a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000048a4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 0000000048a8: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000048b0: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000048c0: 8c7e047e
	s_and_b32 s3, s9, s2                                       // 0000000048c4: 8b030209
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048c8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000048cc: be842003
	s_cbranch_execz 46                                         // 0000000048d0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2e8c>
	v_add_co_u32 v4, s3, v23, s22                              // 0000000048d4: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000048dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000048e0: d5207c05 000c2e80
	v_bfe_u32 v6, v46, 16, 1                                   // 0000000048e8: d6100006 0205212e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000048f0: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 0000000048f4: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000048fc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004900: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v46                             // 000000004908: 38125cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004910: bf8701a3
	v_add_co_u32 v4, s3, s26, v4                               // 000000004914: d7000304 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 00000000491c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s3                  // 000000004920: d5207c05 000e0a1b
	s_wait_kmcnt 0x0                                           // 000000004928: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 00000000492c: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004934: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004938: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004940: 3e080881
	v_add3_u32 v6, v6, v46, 0x7fff                             // 000000004944: d6550006 03fe5d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004950: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004954: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000495c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004960: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v46, v46                               // 000000004968: d4180003 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004970: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004974: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004978: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004980: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000498c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004990: 8c7e047e
	s_and_b32 s3, s8, s2                                       // 000000004994: 8b030208
	s_wait_alu depctr_sa_sdst(0)                               // 000000004998: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 00000000499c: be842003
	s_cbranch_execz 46                                         // 0000000049a0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x2f5c>
	v_add_co_u32 v4, s3, v23, s22                              // 0000000049a4: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000049ac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000049b0: d5207c05 000c2e80
	v_bfe_u32 v6, v45, 16, 1                                   // 0000000049b8: d6100006 0205212d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000049c0: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 0000000049c4: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000049cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000049d0: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v45                             // 0000000049d8: 38125aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000049e0: bf8701a3
	v_add_co_u32 v4, s3, s40, v4                               // 0000000049e4: d7000304 02020828
	s_wait_alu depctr_va_sdst(0)                               // 0000000049ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s3                  // 0000000049f0: d5207c05 000e0a29
	s_wait_kmcnt 0x0                                           // 0000000049f8: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 0000000049fc: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004a04: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004a08: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004a10: 3e080881
	v_add3_u32 v6, v6, v45, 0x7fff                             // 000000004a14: d6550006 03fe5b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004a20: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004a24: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004a2c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004a30: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v45, v45                               // 000000004a38: d4180003 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 000000004a40: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a44: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004a48: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004a50: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a5c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a60: 8c7e047e
	s_and_b32 s3, s7, s2                                       // 000000004a64: 8b030207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a68: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004a6c: be842003
	s_cbranch_execz 46                                         // 000000004a70: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x302c>
	v_add_co_u32 v4, s3, v23, s22                              // 000000004a74: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004a7c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004a80: d5207c05 000c2e80
	v_bfe_u32 v6, v44, 16, 1                                   // 000000004a88: d6100006 0205212c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004a90: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 000000004a94: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004a9c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004aa0: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v44                             // 000000004aa8: 381258ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ab0: bf8701a3
	v_add_co_u32 v4, s3, s38, v4                               // 000000004ab4: d7000304 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000004abc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s3                  // 000000004ac0: d5207c05 000e0a27
	s_wait_kmcnt 0x0                                           // 000000004ac8: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004acc: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004ad4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004ad8: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004ae0: 3e080881
	v_add3_u32 v6, v6, v44, 0x7fff                             // 000000004ae4: d6550006 03fe5906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004af0: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004af4: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004afc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004b00: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v44, v44                               // 000000004b08: d4180003 0202592c
	s_wait_alu depctr_va_sdst(0)                               // 000000004b10: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004b14: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004b18: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004b20: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b2c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004b30: 8c7e047e
	s_and_b32 s3, s6, s2                                       // 000000004b34: 8b030206
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b38: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004b3c: be842003
	s_cbranch_execz 46                                         // 000000004b40: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x30fc>
	v_add_co_u32 v4, s3, v23, s22                              // 000000004b44: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004b4c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004b50: d5207c05 000c2e80
	v_bfe_u32 v6, v43, 16, 1                                   // 000000004b58: d6100006 0205212b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004b60: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 000000004b64: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004b6c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004b70: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v43                             // 000000004b78: 381256ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004b80: bf8701a3
	v_add_co_u32 v4, s3, s36, v4                               // 000000004b84: d7000304 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000004b8c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s3                  // 000000004b90: d5207c05 000e0a25
	s_wait_kmcnt 0x0                                           // 000000004b98: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004b9c: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004ba4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004ba8: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004bb0: 3e080881
	v_add3_u32 v6, v6, v43, 0x7fff                             // 000000004bb4: d6550006 03fe5706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004bc0: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004bc4: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004bcc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004bd0: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v43, v43                               // 000000004bd8: d4180003 0202572b
	s_wait_alu depctr_va_sdst(0)                               // 000000004be0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004be4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004be8: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004bf0: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bfc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004c00: 8c7e047e
	s_and_b32 s3, s5, s2                                       // 000000004c04: 8b030205
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c08: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004c0c: be842003
	s_cbranch_execz 46                                         // 000000004c10: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x31cc>
	v_add_co_u32 v4, s3, v23, s22                              // 000000004c14: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004c1c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004c20: d5207c05 000c2e80
	v_bfe_u32 v6, v42, 16, 1                                   // 000000004c28: d6100006 0205212a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004c30: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 000000004c34: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004c3c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004c40: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v42                             // 000000004c48: 381254ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004c50: bf8701a3
	v_add_co_u32 v4, s3, s34, v4                               // 000000004c54: d7000304 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000004c5c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s3                  // 000000004c60: d5207c05 000e0a23
	s_wait_kmcnt 0x0                                           // 000000004c68: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004c6c: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004c74: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004c78: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004c80: 3e080881
	v_add3_u32 v6, v6, v42, 0x7fff                             // 000000004c84: d6550006 03fe5506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004c90: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004c94: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004c9c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004ca0: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v42, v42                               // 000000004ca8: d4180003 0202552a
	s_wait_alu depctr_va_sdst(0)                               // 000000004cb0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004cb4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004cb8: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004cc0: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ccc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004cd0: 8c7e047e
	s_and_b32 s3, s1, s2                                       // 000000004cd4: 8b030201
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cd8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004cdc: be842003
	s_cbranch_execz 46                                         // 000000004ce0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x329c>
	v_add_co_u32 v4, s3, v23, s22                              // 000000004ce4: d7000304 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004cec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004cf0: d5207c05 000c2e80
	v_bfe_u32 v6, v41, 16, 1                                   // 000000004cf8: d6100006 02052129
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004d00: bf8701a3
	v_add_co_u32 v4, s3, v4, v22                               // 000000004d04: d7000304 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004d0c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004d10: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v41                             // 000000004d18: 381252ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004d20: bf8701a3
	v_add_co_u32 v4, s3, s30, v4                               // 000000004d24: d7000304 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000004d2c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s3                  // 000000004d30: d5207c05 000e0a1f
	s_wait_kmcnt 0x0                                           // 000000004d38: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004d3c: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004d44: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004d48: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004d50: 3e080881
	v_add3_u32 v6, v6, v41, 0x7fff                             // 000000004d54: d6550006 03fe5306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004d60: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004d64: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004d6c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004d70: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v41, v41                               // 000000004d78: d4180003 02025329
	s_wait_alu depctr_va_sdst(0)                               // 000000004d80: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004d84: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004d88: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004d90: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d9c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004da0: 8c7e047e
	s_and_b32 s2, s0, s2                                       // 000000004da4: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 000000004da8: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000004dac: be832002
	s_cbranch_execz 46                                         // 000000004db0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x336c>
	v_add_co_u32 v4, s2, v23, s22                              // 000000004db4: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004dbc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000004dc0: d5207c05 00082e80
	v_bfe_u32 v6, v40, 16, 1                                   // 000000004dc8: d6100006 02052128
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004dd0: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000004dd4: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004ddc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000004de0: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v40                             // 000000004de8: 381250ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004df0: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 000000004df4: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000004dfc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 000000004e00: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 000000004e08: bfc70000
	v_add_co_u32 v7, s2, s20, v0                               // 000000004e0c: d7000207 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004e14: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s2                  // 000000004e18: d5207c08 000a0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004e20: 3e080881
	v_add3_u32 v6, v6, v40, 0x7fff                             // 000000004e24: d6550006 03fe5106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004e30: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000004e34: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004e3c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000004e40: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v40, v40                               // 000000004e48: d4180002 02025128
	s_wait_alu depctr_va_sdst(0)                               // 000000004e50: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004e54: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000004e58: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004e60: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004e70: 8c7e037e
	s_and_b32 s2, s17, vcc_lo                                  // 000000004e74: 8b026a11
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e78: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000004e7c: be832002
	s_cbranch_execz 40                                         // 000000004e80: bfa50028 <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3424>
	v_add_co_u32 v4, s2, v23, s22                              // 000000004e84: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004e8c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000004e90: d5207c05 00082e80
	v_bfe_u32 v6, v39, 16, 1                                   // 000000004e98: d6100006 02052127
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ea0: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000004ea4: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004eac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000004eb0: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 000000004eb8: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000004ebc: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004ec4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000004ec8: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004ed0: 3e080881
	v_add3_u32 v6, v6, v39, 0x7fff                             // 000000004ed4: d6550006 03fe4f06 00007fff
	v_or_b32_e32 v9, 0x400000, v39                             // 000000004ee0: 38124eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000004ee8: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 000000004eec: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004ef4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000004ef8: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v39, v39                               // 000000004f00: d4180002 02024f27
	s_wait_alu depctr_va_sdst(0)                               // 000000004f08: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004f0c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000004f10: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000004f18: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f24: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004f28: 8c7e037e
	s_and_b32 s2, s18, vcc_lo                                  // 000000004f2c: 8b026a12
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f30: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000004f34: be832002
	s_cbranch_execz 46                                         // 000000004f38: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x34f4>
	v_add_co_u32 v4, s2, v23, s22                              // 000000004f3c: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000004f44: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000004f48: d5207c05 00082e80
	v_bfe_u32 v6, v38, 16, 1                                   // 000000004f50: d6100006 02052126
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004f58: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000004f5c: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004f64: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000004f68: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v38                             // 000000004f70: 38124cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004f78: bf8701a3
	v_add_co_u32 v4, s2, s26, v4                               // 000000004f7c: d7000204 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000004f84: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s2                  // 000000004f88: d5207c05 000a0a1b
	s_wait_kmcnt 0x0                                           // 000000004f90: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000004f94: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004f9c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000004fa0: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004fa8: 3e080881
	v_add3_u32 v6, v6, v38, 0x7fff                             // 000000004fac: d6550006 03fe4d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004fb8: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000004fbc: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004fc4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000004fc8: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v38, v38                               // 000000004fd0: d4180002 02024d26
	s_wait_alu depctr_va_sdst(0)                               // 000000004fd8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004fdc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000004fe0: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000004fe8: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ff4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004ff8: 8c7e037e
	s_and_b32 s2, s16, vcc_lo                                  // 000000004ffc: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 000000005000: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005004: be832002
	s_cbranch_execz 46                                         // 000000005008: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x35c4>
	v_add_co_u32 v4, s2, v23, s22                              // 00000000500c: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000005014: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000005018: d5207c05 00082e80
	v_bfe_u32 v6, v37, 16, 1                                   // 000000005020: d6100006 02052125
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005028: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 00000000502c: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005034: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005038: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v37                             // 000000005040: 38124aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005048: bf8701a3
	v_add_co_u32 v4, s2, s40, v4                               // 00000000504c: d7000204 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000005054: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s2                  // 000000005058: d5207c05 000a0a29
	s_wait_kmcnt 0x0                                           // 000000005060: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005064: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000506c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005070: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005078: 3e080881
	v_add3_u32 v6, v6, v37, 0x7fff                             // 00000000507c: d6550006 03fe4b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005088: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 00000000508c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005094: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005098: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v37, v37                               // 0000000050a0: d4180002 02024b25
	s_wait_alu depctr_va_sdst(0)                               // 0000000050a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000050ac: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000050b0: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 0000000050b8: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000050c8: 8c7e037e
	s_and_b32 s2, s15, vcc_lo                                  // 0000000050cc: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050d0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000050d4: be832002
	s_cbranch_execz 46                                         // 0000000050d8: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3694>
	v_add_co_u32 v4, s2, v23, s22                              // 0000000050dc: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000050e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000050e8: d5207c05 00082e80
	v_bfe_u32 v6, v36, 16, 1                                   // 0000000050f0: d6100006 02052124
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000050f8: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 0000000050fc: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005104: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005108: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v36                             // 000000005110: 381248ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005118: bf8701a3
	v_add_co_u32 v4, s2, s38, v4                               // 00000000511c: d7000204 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000005124: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s2                  // 000000005128: d5207c05 000a0a27
	s_wait_kmcnt 0x0                                           // 000000005130: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005134: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000513c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005140: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005148: 3e080881
	v_add3_u32 v6, v6, v36, 0x7fff                             // 00000000514c: d6550006 03fe4906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005158: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 00000000515c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005164: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005168: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v36, v36                               // 000000005170: d4180002 02024924
	s_wait_alu depctr_va_sdst(0)                               // 000000005178: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000517c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000005180: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005188: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005194: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005198: 8c7e037e
	s_and_b32 s2, s14, vcc_lo                                  // 00000000519c: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051a0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000051a4: be832002
	s_cbranch_execz 46                                         // 0000000051a8: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3764>
	v_add_co_u32 v4, s2, v23, s22                              // 0000000051ac: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000051b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000051b8: d5207c05 00082e80
	v_bfe_u32 v6, v35, 16, 1                                   // 0000000051c0: d6100006 02052123
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000051c8: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 0000000051cc: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000051d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000051d8: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v35                             // 0000000051e0: 381246ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000051e8: bf8701a3
	v_add_co_u32 v4, s2, s36, v4                               // 0000000051ec: d7000204 02020824
	s_wait_alu depctr_va_sdst(0)                               // 0000000051f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s2                  // 0000000051f8: d5207c05 000a0a25
	s_wait_kmcnt 0x0                                           // 000000005200: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005204: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000520c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005210: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005218: 3e080881
	v_add3_u32 v6, v6, v35, 0x7fff                             // 00000000521c: d6550006 03fe4706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005228: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 00000000522c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005234: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005238: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v35, v35                               // 000000005240: d4180002 02024723
	s_wait_alu depctr_va_sdst(0)                               // 000000005248: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000524c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000005250: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005258: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005264: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005268: 8c7e037e
	s_and_b32 s2, s13, vcc_lo                                  // 00000000526c: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000005270: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005274: be832002
	s_cbranch_execz 46                                         // 000000005278: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3834>
	v_add_co_u32 v4, s2, v23, s22                              // 00000000527c: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000005284: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000005288: d5207c05 00082e80
	v_bfe_u32 v6, v34, 16, 1                                   // 000000005290: d6100006 02052122
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005298: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 00000000529c: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000052a4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000052a8: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v34                             // 0000000052b0: 381244ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000052b8: bf8701a3
	v_add_co_u32 v4, s2, s34, v4                               // 0000000052bc: d7000204 02020822
	s_wait_alu depctr_va_sdst(0)                               // 0000000052c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s2                  // 0000000052c8: d5207c05 000a0a23
	s_wait_kmcnt 0x0                                           // 0000000052d0: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 0000000052d4: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000052dc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 0000000052e0: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000052e8: 3e080881
	v_add3_u32 v6, v6, v34, 0x7fff                             // 0000000052ec: d6550006 03fe4506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000052f8: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000052fc: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005304: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005308: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v34, v34                               // 000000005310: d4180002 02024522
	s_wait_alu depctr_va_sdst(0)                               // 000000005318: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000531c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000005320: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005328: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005334: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005338: 8c7e037e
	s_and_b32 s2, s11, vcc_lo                                  // 00000000533c: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005340: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005344: be832002
	s_cbranch_execz 46                                         // 000000005348: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3904>
	v_add_co_u32 v4, s2, v23, s22                              // 00000000534c: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000005354: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000005358: d5207c05 00082e80
	v_bfe_u32 v6, v33, 16, 1                                   // 000000005360: d6100006 02052121
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005368: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 00000000536c: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005374: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005378: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v33                             // 000000005380: 381242ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005388: bf8701a3
	v_add_co_u32 v4, s2, s30, v4                               // 00000000538c: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000005394: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 000000005398: d5207c05 000a0a1f
	s_wait_kmcnt 0x0                                           // 0000000053a0: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 0000000053a4: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000053ac: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 0000000053b0: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000053b8: 3e080881
	v_add3_u32 v6, v6, v33, 0x7fff                             // 0000000053bc: d6550006 03fe4306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000053c8: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000053cc: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000053d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000053d8: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v33, v33                               // 0000000053e0: d4180002 02024321
	s_wait_alu depctr_va_sdst(0)                               // 0000000053e8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000053ec: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000053f0: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 0000000053f8: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005404: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005408: 8c7e037e
	s_and_b32 s2, s10, vcc_lo                                  // 00000000540c: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005410: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005414: be832002
	s_cbranch_execz 46                                         // 000000005418: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x39d4>
	v_add_co_u32 v4, s2, v23, s22                              // 00000000541c: d7000204 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000005424: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000005428: d5207c05 00082e80
	v_bfe_u32 v6, v32, 16, 1                                   // 000000005430: d6100006 02052120
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005438: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 00000000543c: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005444: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005448: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v32                             // 000000005450: 380e40ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005458: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 00000000545c: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000005464: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 000000005468: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 000000005470: bfc70000
	v_add_co_u32 v2, s2, s20, v2                               // 000000005474: d7000202 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000547c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s21, v3, s2                  // 000000005480: d5207c03 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005488: 3e080881
	v_add3_u32 v6, v6, v32, 0x7fff                             // 00000000548c: d6550006 03fe4106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005498: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 00000000549c: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 0000000054a4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 0000000054a8: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v32, v32                               // 0000000054b0: d4180002 02024120
	s_wait_alu depctr_va_sdst(0)                               // 0000000054b8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000054bc: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 0000000054c0: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 0000000054c8: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000054d8: 8c7e037e
	s_and_b32 s2, s12, vcc_lo                                  // 0000000054dc: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054e0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000054e4: be832002
	s_cbranch_execz 40                                         // 0000000054e8: bfa50028 <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3a8c>
	v_add_co_u32 v2, s2, v23, s22                              // 0000000054ec: d7000202 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000054f4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 0000000054f8: d5207c03 00082e80
	v_bfe_u32 v4, v31, 16, 1                                   // 000000005500: d6100004 0205211f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005508: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 00000000550c: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000005514: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005518: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 000000005520: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005524: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 00000000552c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005530: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005538: 3e040481
	v_add3_u32 v4, v4, v31, 0x7fff                             // 00000000553c: d6550004 03fe3f04 00007fff
	v_or_b32_e32 v7, 0x400000, v31                             // 000000005548: 380e3eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000005550: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 000000005554: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 00000000555c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005560: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v31, v31                               // 000000005568: d4180002 02023f1f
	s_wait_alu depctr_va_sdst(0)                               // 000000005570: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005574: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005578: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005580: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000558c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005590: 8c7e037e
	s_and_b32 s2, s9, vcc_lo                                   // 000000005594: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000005598: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000559c: be832002
	s_cbranch_execz 46                                         // 0000000055a0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3b5c>
	v_add_co_u32 v2, s2, v23, s22                              // 0000000055a4: d7000202 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000055ac: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 0000000055b0: d5207c03 00082e80
	v_bfe_u32 v4, v30, 16, 1                                   // 0000000055b8: d6100004 0205211e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000055c0: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 0000000055c4: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000055cc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000055d0: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v30                             // 0000000055d8: 380e3cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000055e0: bf8701a3
	v_add_co_u32 v2, s2, s26, v2                               // 0000000055e4: d7000202 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 0000000055ec: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s27, v3, s2                  // 0000000055f0: d5207c03 000a061b
	s_wait_kmcnt 0x0                                           // 0000000055f8: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 0000000055fc: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005604: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005608: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005610: 3e040481
	v_add3_u32 v4, v4, v30, 0x7fff                             // 000000005614: d6550004 03fe3d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005620: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005624: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 00000000562c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005630: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v30, v30                               // 000000005638: d4180002 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000005640: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005644: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005648: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005650: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000565c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005660: 8c7e037e
	s_and_b32 s2, s8, vcc_lo                                   // 000000005664: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 000000005668: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000566c: be832002
	s_cbranch_execz 46                                         // 000000005670: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3c2c>
	v_add_co_u32 v2, s2, v23, s22                              // 000000005674: d7000202 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 00000000567c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005680: d5207c03 00082e80
	v_bfe_u32 v4, v29, 16, 1                                   // 000000005688: d6100004 0205211d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005690: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000005694: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 00000000569c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000056a0: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v29                             // 0000000056a8: 380e3aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000056b0: bf8701a3
	v_add_co_u32 v2, s2, s40, v2                               // 0000000056b4: d7000202 02020428
	s_wait_alu depctr_va_sdst(0)                               // 0000000056bc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s41, v3, s2                  // 0000000056c0: d5207c03 000a0629
	s_wait_kmcnt 0x0                                           // 0000000056c8: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 0000000056cc: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000056d4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 0000000056d8: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000056e0: 3e040481
	v_add3_u32 v4, v4, v29, 0x7fff                             // 0000000056e4: d6550004 03fe3b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000056f0: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 0000000056f4: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 0000000056fc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005700: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v29, v29                               // 000000005708: d4180002 02023b1d
	s_wait_alu depctr_va_sdst(0)                               // 000000005710: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005714: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005718: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005720: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000572c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005730: 8c7e037e
	s_and_b32 s2, s7, vcc_lo                                   // 000000005734: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000005738: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000573c: be832002
	s_cbranch_execz 46                                         // 000000005740: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3cfc>
	v_add_co_u32 v2, s2, v23, s22                              // 000000005744: d7000202 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 00000000574c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005750: d5207c03 00082e80
	v_bfe_u32 v4, v28, 16, 1                                   // 000000005758: d6100004 0205211c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005760: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000005764: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 00000000576c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005770: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v28                             // 000000005778: 380e38ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005780: bf8701a3
	v_add_co_u32 v2, s2, s38, v2                               // 000000005784: d7000202 02020426
	s_wait_alu depctr_va_sdst(0)                               // 00000000578c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s39, v3, s2                  // 000000005790: d5207c03 000a0627
	s_wait_kmcnt 0x0                                           // 000000005798: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 00000000579c: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000057a4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 0000000057a8: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000057b0: 3e040481
	v_add3_u32 v4, v4, v28, 0x7fff                             // 0000000057b4: d6550004 03fe3904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000057c0: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 0000000057c4: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 0000000057cc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 0000000057d0: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v28, v28                               // 0000000057d8: d4180002 0202391c
	s_wait_alu depctr_va_sdst(0)                               // 0000000057e0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000057e4: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 0000000057e8: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 0000000057f0: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005800: 8c7e037e
	s_and_b32 s2, s6, vcc_lo                                   // 000000005804: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000005808: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000580c: be832002
	s_cbranch_execz 46                                         // 000000005810: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3dcc>
	v_add_co_u32 v2, s2, v23, s22                              // 000000005814: d7000202 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 00000000581c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005820: d5207c03 00082e80
	v_bfe_u32 v4, v27, 16, 1                                   // 000000005828: d6100004 0205211b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005830: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000005834: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 00000000583c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005840: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v27                             // 000000005848: 380e36ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005850: bf8701a3
	v_add_co_u32 v2, s2, s36, v2                               // 000000005854: d7000202 02020424
	s_wait_alu depctr_va_sdst(0)                               // 00000000585c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, v3, s2                  // 000000005860: d5207c03 000a0625
	s_wait_kmcnt 0x0                                           // 000000005868: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 00000000586c: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005874: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005878: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005880: 3e040481
	v_add3_u32 v4, v4, v27, 0x7fff                             // 000000005884: d6550004 03fe3704 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005890: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005894: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 00000000589c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 0000000058a0: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v27, v27                               // 0000000058a8: d4180002 0202371b
	s_wait_alu depctr_va_sdst(0)                               // 0000000058b0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000058b4: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 0000000058b8: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 0000000058c0: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000058d0: 8c7e037e
	s_and_b32 s2, s5, vcc_lo                                   // 0000000058d4: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058d8: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000058dc: be832002
	s_cbranch_execz 46                                         // 0000000058e0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3e9c>
	v_add_co_u32 v2, s2, v23, s22                              // 0000000058e4: d7000202 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000058ec: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 0000000058f0: d5207c03 00082e80
	v_bfe_u32 v4, v26, 16, 1                                   // 0000000058f8: d6100004 0205211a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005900: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000005904: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 00000000590c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005910: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v26                             // 000000005918: 380e34ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005920: bf8701a3
	v_add_co_u32 v2, s2, s34, v2                               // 000000005924: d7000202 02020422
	s_wait_alu depctr_va_sdst(0)                               // 00000000592c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s2                  // 000000005930: d5207c03 000a0623
	s_wait_kmcnt 0x0                                           // 000000005938: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 00000000593c: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005944: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005948: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005950: 3e040481
	v_add3_u32 v4, v4, v26, 0x7fff                             // 000000005954: d6550004 03fe3504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005960: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005964: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 00000000596c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005970: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v26, v26                               // 000000005978: d4180002 0202351a
	s_wait_alu depctr_va_sdst(0)                               // 000000005980: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005984: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005988: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005990: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000599c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000059a0: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 0000000059a4: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059a8: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000059ac: be822001
	s_cbranch_execz 46                                         // 0000000059b0: bfa5002e <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x3f6c>
	v_add_co_u32 v2, s1, v23, s22                              // 0000000059b4: d7000102 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 0000000059bc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s1                   // 0000000059c0: d5207c03 00042e80
	v_bfe_u32 v4, v25, 16, 1                                   // 0000000059c8: d6100004 02052119
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000059d0: bf8701a3
	v_add_co_u32 v2, s1, v2, v22                               // 0000000059d4: d7000102 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000059dc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 0000000059e0: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v25                             // 0000000059e8: 380e32ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000059f0: bf8701a3
	v_add_co_u32 v2, s1, s30, v2                               // 0000000059f4: d7000102 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 0000000059fc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s1                  // 000000005a00: d5207c03 0006061f
	s_wait_kmcnt 0x0                                           // 000000005a08: bfc70000
	v_add_co_u32 v5, s1, s20, v0                               // 000000005a0c: d7000105 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005a14: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s1                  // 000000005a18: d5207c06 00060215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005a20: 3e040481
	v_add3_u32 v4, v4, v25, 0x7fff                             // 000000005a24: d6550004 03fe3304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005a30: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 000000005a34: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005a3c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 000000005a40: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v25, v25                               // 000000005a48: d4180001 02023319
	s_wait_alu depctr_va_sdst(0)                               // 000000005a50: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005a54: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 000000005a58: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005a60: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000005a70: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 000000005a74: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a78: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005a7c: be812000
	s_cbranch_execz 43                                         // 000000005a80: bfa5002b <tessera_rocm_scaled_matmul_lds_7ddd1c0ffae49434+0x4030>
	v_add_co_u32 v2, s0, v23, s22                              // 000000005a84: d7000002 02002d17
	s_wait_alu depctr_va_sdst(0)                               // 000000005a8c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s0                   // 000000005a90: d5207c03 00002e80
	v_bfe_u32 v4, v24, 16, 1                                   // 000000005a98: d6100004 02052118
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005aa0: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v22                           // 000000005aa4: d7006a02 02022d02
	s_wait_alu depctr_va_vcc(0)                                // 000000005aac: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000005ab0: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v24                             // 000000005ab8: 380a30ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005ac0: bf8701a3
	v_add_co_u32 v2, vcc_lo, s28, v2                           // 000000005ac4: d7006a02 0202041c
	s_wait_alu depctr_va_vcc(0)                                // 000000005acc: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s29, v3, vcc_lo              // 000000005ad0: d5207c03 01aa061d
	s_wait_kmcnt 0x0                                           // 000000005ad8: bfc70000
	v_add_co_u32 v0, vcc_lo, s20, v0                           // 000000005adc: d7006a00 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000005ae4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo              // 000000005ae8: d5207c01 01aa0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005af0: 3e040481
	v_add3_u32 v4, v4, v24, 0x7fff                             // 000000005af4: d6550004 03fe3104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005b00: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 000000005b04: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 000000005b0c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 000000005b10: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 000000005b18: 7c303118
	s_wait_alu depctr_va_vcc(0)                                // 000000005b1c: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000005b20: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:96          // 000000005b24: ee09407c 01000000 00006000
	s_nop 0                                                    // 000000005b30: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000005b34: bfb60003
	s_endpgm                                                   // 000000005b38: bfb00000
	s_code_end                                                 // 000000005b3c: bf9f0000
	s_code_end                                                 // 000000005b40: bf9f0000
	s_code_end                                                 // 000000005b44: bf9f0000
	s_code_end                                                 // 000000005b48: bf9f0000
	s_code_end                                                 // 000000005b4c: bf9f0000
	s_code_end                                                 // 000000005b50: bf9f0000
	s_code_end                                                 // 000000005b54: bf9f0000
	s_code_end                                                 // 000000005b58: bf9f0000
	s_code_end                                                 // 000000005b5c: bf9f0000
	s_code_end                                                 // 000000005b60: bf9f0000
	s_code_end                                                 // 000000005b64: bf9f0000
	s_code_end                                                 // 000000005b68: bf9f0000
	s_code_end                                                 // 000000005b6c: bf9f0000
	s_code_end                                                 // 000000005b70: bf9f0000
	s_code_end                                                 // 000000005b74: bf9f0000
	s_code_end                                                 // 000000005b78: bf9f0000
	s_code_end                                                 // 000000005b7c: bf9f0000
	s_code_end                                                 // 000000005b80: bf9f0000
	s_code_end                                                 // 000000005b84: bf9f0000
	s_code_end                                                 // 000000005b88: bf9f0000
	s_code_end                                                 // 000000005b8c: bf9f0000
	s_code_end                                                 // 000000005b90: bf9f0000
	s_code_end                                                 // 000000005b94: bf9f0000
	s_code_end                                                 // 000000005b98: bf9f0000
	s_code_end                                                 // 000000005b9c: bf9f0000
	s_code_end                                                 // 000000005ba0: bf9f0000
	s_code_end                                                 // 000000005ba4: bf9f0000
	s_code_end                                                 // 000000005ba8: bf9f0000
	s_code_end                                                 // 000000005bac: bf9f0000
	s_code_end                                                 // 000000005bb0: bf9f0000
	s_code_end                                                 // 000000005bb4: bf9f0000
	s_code_end                                                 // 000000005bb8: bf9f0000
	s_code_end                                                 // 000000005bbc: bf9f0000
	s_code_end                                                 // 000000005bc0: bf9f0000
	s_code_end                                                 // 000000005bc4: bf9f0000
	s_code_end                                                 // 000000005bc8: bf9f0000
	s_code_end                                                 // 000000005bcc: bf9f0000
	s_code_end                                                 // 000000005bd0: bf9f0000
	s_code_end                                                 // 000000005bd4: bf9f0000
	s_code_end                                                 // 000000005bd8: bf9f0000
	s_code_end                                                 // 000000005bdc: bf9f0000
	s_code_end                                                 // 000000005be0: bf9f0000
	s_code_end                                                 // 000000005be4: bf9f0000
	s_code_end                                                 // 000000005be8: bf9f0000
	s_code_end                                                 // 000000005bec: bf9f0000
	s_code_end                                                 // 000000005bf0: bf9f0000
	s_code_end                                                 // 000000005bf4: bf9f0000
	s_code_end                                                 // 000000005bf8: bf9f0000
	s_code_end                                                 // 000000005bfc: bf9f0000
	s_code_end                                                 // 000000005c00: bf9f0000
	s_code_end                                                 // 000000005c04: bf9f0000
	s_code_end                                                 // 000000005c08: bf9f0000
	s_code_end                                                 // 000000005c0c: bf9f0000
	s_code_end                                                 // 000000005c10: bf9f0000
	s_code_end                                                 // 000000005c14: bf9f0000
	s_code_end                                                 // 000000005c18: bf9f0000
	s_code_end                                                 // 000000005c1c: bf9f0000
	s_code_end                                                 // 000000005c20: bf9f0000
	s_code_end                                                 // 000000005c24: bf9f0000
	s_code_end                                                 // 000000005c28: bf9f0000
	s_code_end                                                 // 000000005c2c: bf9f0000
	s_code_end                                                 // 000000005c30: bf9f0000
	s_code_end                                                 // 000000005c34: bf9f0000
	s_code_end                                                 // 000000005c38: bf9f0000
	s_code_end                                                 // 000000005c3c: bf9f0000
	s_code_end                                                 // 000000005c40: bf9f0000
	s_code_end                                                 // 000000005c44: bf9f0000
	s_code_end                                                 // 000000005c48: bf9f0000
	s_code_end                                                 // 000000005c4c: bf9f0000
	s_code_end                                                 // 000000005c50: bf9f0000
	s_code_end                                                 // 000000005c54: bf9f0000
	s_code_end                                                 // 000000005c58: bf9f0000
	s_code_end                                                 // 000000005c5c: bf9f0000
	s_code_end                                                 // 000000005c60: bf9f0000
	s_code_end                                                 // 000000005c64: bf9f0000
	s_code_end                                                 // 000000005c68: bf9f0000
	s_code_end                                                 // 000000005c6c: bf9f0000
	s_code_end                                                 // 000000005c70: bf9f0000
	s_code_end                                                 // 000000005c74: bf9f0000
	s_code_end                                                 // 000000005c78: bf9f0000
	s_code_end                                                 // 000000005c7c: bf9f0000
	s_code_end                                                 // 000000005c80: bf9f0000
	s_code_end                                                 // 000000005c84: bf9f0000
	s_code_end                                                 // 000000005c88: bf9f0000
	s_code_end                                                 // 000000005c8c: bf9f0000
	s_code_end                                                 // 000000005c90: bf9f0000
	s_code_end                                                 // 000000005c94: bf9f0000
	s_code_end                                                 // 000000005c98: bf9f0000
	s_code_end                                                 // 000000005c9c: bf9f0000
	s_code_end                                                 // 000000005ca0: bf9f0000
	s_code_end                                                 // 000000005ca4: bf9f0000
	s_code_end                                                 // 000000005ca8: bf9f0000
	s_code_end                                                 // 000000005cac: bf9f0000
	s_code_end                                                 // 000000005cb0: bf9f0000
	s_code_end                                                 // 000000005cb4: bf9f0000
	s_code_end                                                 // 000000005cb8: bf9f0000
	s_code_end                                                 // 000000005cbc: bf9f0000
	s_code_end                                                 // 000000005cc0: bf9f0000
	s_code_end                                                 // 000000005cc4: bf9f0000
	s_code_end                                                 // 000000005cc8: bf9f0000
	s_code_end                                                 // 000000005ccc: bf9f0000
	s_code_end                                                 // 000000005cd0: bf9f0000
	s_code_end                                                 // 000000005cd4: bf9f0000
	s_code_end                                                 // 000000005cd8: bf9f0000
	s_code_end                                                 // 000000005cdc: bf9f0000
	s_code_end                                                 // 000000005ce0: bf9f0000
	s_code_end                                                 // 000000005ce4: bf9f0000
	s_code_end                                                 // 000000005ce8: bf9f0000
	s_code_end                                                 // 000000005cec: bf9f0000
	s_code_end                                                 // 000000005cf0: bf9f0000
	s_code_end                                                 // 000000005cf4: bf9f0000
	s_code_end                                                 // 000000005cf8: bf9f0000
	s_code_end                                                 // 000000005cfc: bf9f0000
